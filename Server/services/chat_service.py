# services/chat_service.py
import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from bson import ObjectId
from pydantic import BaseModel
import re

import google.generativeai as genai
from pinecone import Pinecone
import anthropic

from core.config import settings
from core.db import get_db

logger = logging.getLogger(__name__)

# Initialize clients
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
pc_index = pc.Index(settings.PINECONE_INDEX_NAME)
genai.configure(api_key=settings.GEMINI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

# Available models
AVAILABLE_MODELS = {
    "gemini-1.5-flash": "gemini",
    "gemini-1.5-pro": "gemini", 
    "claude-opus-4-1-20250805": "claude",
    "claude-opus-4-20250514": "claude",
    "claude-sonnet-4-20250514": "claude",
    "claude-3-7-sonnet-20250219": "claude",
    "claude-3-5-sonnet-20241022": "claude",
    "claude-3-5-haiku-20241022": "claude",
    "claude-3-haiku-20240307": "claude"
}
DEFAULT_MODEL = "gemini-1.5-flash"

# ---------- Request/Response Models ----------
class QueryRequest(BaseModel):
    kb_id: str
    kb_name: str
    query: str
    model: str = DEFAULT_MODEL
    max_results: int = 10
    temperature: float = 0.7

class SourceDocument(BaseModel):
    text: str
    source_file: str
    chunk_index: int
    relevance_score: float

class ChatMessage(BaseModel):
    message_id: str
    kb_id: str
    created_at: datetime
    message: str
    by: str  # "user" or "ai"
    model_used: Optional[str] = None
    sources: Optional[List[SourceDocument]] = None

class QueryResponse(BaseModel):
    message_id: str
    kb_id: str
    created_at: datetime
    message: str
    by: str
    model_used: str
    sources: List[SourceDocument]
    processing_time_ms: int

# ---------- Chat Service Class ----------
class ChatService:
    def __init__(self):
        self.db = get_db()
    
    def clean_text(self, text: str) -> str:
        """Clean and format text for better display."""
        if not text:
            return ""
        
        # Remove excessive whitespace and newlines
        cleaned = re.sub(r'\n\s*\n', '\n\n', text)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for the user's query."""
        try:
            logger.info(f"Generating embedding for query: {query[:50]}...")
            
            resp = genai.embed_content(
                model=settings.GEMINI_EMBED_MODEL,
                content=query,
                task_type="retrieval_query"
            )
            
            embedding = resp["embedding"]
            logger.info(f"Generated query embedding with dimension: {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise RuntimeError(f"Failed to generate query embedding: {e}")
    
    async def search_similar_documents(self, kb_id: str, query_embedding: List[float], max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for similar documents in Pinecone vector database."""
        try:
            logger.info(f"Searching for similar documents in KB {kb_id}")
            
            # Query Pinecone with filters
            search_results = pc_index.query(
                vector=query_embedding,
                top_k=max_results,
                filter={"kb_id": {"$eq": kb_id}},
                include_metadata=True
            )
            
            # Filter results by relevance threshold
            relevance_threshold = 0.1  # Adjust as needed
            documents = []
            
            for match in search_results.matches:
                if match.score >= relevance_threshold:
                    doc_data = {
                        "text": self.clean_text(match.metadata.get("text", "")),
                        "source_file": match.metadata.get("source_file", "unknown"),
                        "chunk_index": match.metadata.get("chunk_index", 0),
                        "relevance_score": float(match.score),
                        "metadata": match.metadata
                    }
                    documents.append(doc_data)
            
            logger.info(f"Found {len(documents)} relevant documents above threshold")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to search similar documents: {e}")
            raise RuntimeError(f"Failed to search similar documents: {e}")
    
    def format_context_for_llm(self, documents: List[Dict[str, Any]], max_context_length: int = 4000) -> str:
        """Format retrieved documents as context for the LLM."""
        if not documents:
            return "No relevant documents found in the knowledge base."
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents, 1):
            doc_text = doc["text"]
            source_file = doc["source_file"]
            relevance = doc["relevance_score"]
            
            doc_section = f"""
Document {i} (Source: {source_file}, Relevance: {relevance:.3f}):
{doc_text}
---
"""
            
            if current_length + len(doc_section) > max_context_length:
                if i == 1:
                    truncated = doc_text[:max_context_length - 200] + "...(truncated)"
                    doc_section = f"""
Document 1 (Source: {source_file}, Relevance: {relevance:.3f}):
{truncated}
---
"""
                    context_parts.append(doc_section)
                break
            
            context_parts.append(doc_section)
            current_length += len(doc_section)
        
        return "\n".join(context_parts)
    
    async def generate_gemini_response(self, query: str, context: str, kb_name: str, model: str, temperature: float, chat_history: List[ChatMessage] = None) -> str:
        """Generate response using Gemini models."""
        try:
            # Build conversation context
            conversation_context = ""
            if chat_history:
                recent_history = chat_history[-6:]  # Last 3 exchanges
                for msg in recent_history:
                    role = "Human" if msg.by == "user" else "Assistant"
                    conversation_context += f"{role}: {msg.message}\n\n"
            
            system_prompt = f"""You are an AI assistant helping users query the "{kb_name}" knowledge base.

INSTRUCTIONS:
1. Answer the user's question based on the provided context documents
2. Be accurate, helpful, and concise
3. If the context doesn't contain relevant information, say so clearly
4. Cite specific sources when making claims
5. If you're unsure about something, express that uncertainty
6. Maintain conversation continuity with previous messages when relevant

CONTEXT DOCUMENTS:
{context}

{conversation_context}Human: {query}

Please provide a comprehensive answer based on the context documents above."""
            
            model_instance = genai.GenerativeModel(
                model_name=model,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=2000,
                    top_p=0.9,
                    top_k=40
                )
            )
            
            response = model_instance.generate_content(system_prompt)
            return response.text.strip() if response.text else "I couldn't generate a proper response. Please try again."
            
        except Exception as e:
            logger.error(f"Failed to generate Gemini response: {e}")
            return f"Error generating response with {model}: {str(e)}"
    
    async def generate_claude_response(self, query: str, context: str, kb_name: str, model: str, temperature: float, chat_history: List[ChatMessage] = None) -> str:
        """Generate response using Claude models."""
        try:
            # Build conversation context
            messages = []
            if chat_history:
                recent_history = chat_history[-6:]  # Last 3 exchanges
                for msg in recent_history:
                    role = "user" if msg.by == "user" else "assistant"
                    messages.append({"role": role, "content": msg.message})
            
            # Add current query
            messages.append({"role": "user", "content": query})
            
            system_prompt = f"""You are an AI assistant helping users query the "{kb_name}" knowledge base.

INSTRUCTIONS:
1. Answer the user's question based on the provided context documents
2. Be accurate, helpful, and concise
3. If the context doesn't contain relevant information, say so clearly
4. Cite specific sources when making claims
5. If you're unsure about something, express that uncertainty

CONTEXT DOCUMENTS:
{context}

Please provide a comprehensive answer based on the context documents above."""
            
            response = anthropic_client.messages.create(
                model=model,
                max_tokens=2000,
                temperature=temperature,
                system=system_prompt,
                messages=messages
            )
            
            return response.content[0].text.strip() if response.content else "I couldn't generate a proper response. Please try again."
            
        except Exception as e:
            logger.error(f"Failed to generate Claude response: {e}")
            return f"Error generating response with {model}: {str(e)}"
    
    async def generate_response(self, query: str, context: str, kb_name: str, model: str, temperature: float, chat_history: List[ChatMessage] = None) -> str:
        """Generate AI response using the specified model."""
        try:
            if model not in AVAILABLE_MODELS:
                logger.warning(f"Unknown model {model}, using default {DEFAULT_MODEL}")
                model = DEFAULT_MODEL
            
            model_type = AVAILABLE_MODELS[model]
            
            if model_type == "gemini":
                return await self.generate_gemini_response(query, context, kb_name, model, temperature, chat_history)
            elif model_type == "claude":
                return await self.generate_claude_response(query, context, kb_name, model, temperature, chat_history)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return f"I encountered an error while generating the response: {str(e)}"
    
    async def get_chat_history(self, kb_id: str, limit: int = 20) -> List[ChatMessage]:
        """Get recent chat history for a knowledge base."""
        try:
            cursor = self.db.chat_messages.find(
                {"kb_id": kb_id}
            ).sort("created_at", -1).limit(limit)
            
            messages = []
            async for msg in cursor:
                sources = None
                if msg.get("sources"):
                    sources = [
                        SourceDocument(**source) for source in msg["sources"]
                    ]
                
                messages.append(ChatMessage(
                    message_id=str(msg["_id"]),
                    kb_id=msg["kb_id"],
                    created_at=msg["created_at"],
                    message=msg["message"],
                    by=msg["by"],
                    model_used=msg.get("model_used"),
                    sources=sources
                ))
            
            return list(reversed(messages))  # Return in chronological order
            
        except Exception as e:
            logger.error(f"Failed to get chat history: {e}")
            return []
    
    async def save_message(self, kb_id: str, message: str, by: str, model_used: Optional[str] = None, sources: Optional[List[Dict]] = None) -> str:
        """Save a chat message to the database."""
        try:
            message_doc = {
                "kb_id": kb_id,
                "created_at": datetime.utcnow(),
                "message": message,
                "by": by,
                "model_used": model_used,
                "sources": sources or []
            }
            
            result = await self.db.chat_messages.insert_one(message_doc)
            message_id = str(result.inserted_id)
            
            logger.info(f"Saved {by} message for KB {kb_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
            raise RuntimeError(f"Failed to save message: {e}")
    
    async def query_and_respond(self, request: QueryRequest) -> QueryResponse:
        """Main method to handle query, search, and response generation."""
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Processing query for KB {request.kb_id}: {request.query[:100]}...")
            
            # Get chat history for context
            chat_history = await self.get_chat_history(request.kb_id)
            
            # Save user message
            user_message_id = await self.save_message(request.kb_id, request.query, "user")
            
            # Generate query embedding
            query_embedding = await self.generate_query_embedding(request.query)
            
            # Search for similar documents
            similar_docs = await self.search_similar_documents(
                request.kb_id, 
                query_embedding, 
                request.max_results
            )
            
            # Format context for LLM
            context = self.format_context_for_llm(similar_docs)
            
            # Generate AI response
            ai_response = await self.generate_response(
                request.query,
                context,
                request.kb_name,
                request.model,
                request.temperature,
                chat_history
            )
            
            # Prepare source documents for response
            sources = [
                SourceDocument(
                    text=doc["text"],
                    source_file=doc["source_file"],
                    chunk_index=doc["chunk_index"],
                    relevance_score=doc["relevance_score"]
                )
                for doc in similar_docs
            ]
            
            # Save AI response with sources
            ai_message_id = await self.save_message(
                request.kb_id, 
                ai_response,
                "ai",
                model_used=request.model,
                sources=[source.dict() for source in sources]
            )
            
            # Calculate processing time
            end_time = datetime.utcnow()
            processing_time = int((end_time - start_time).total_seconds() * 1000)
            
            logger.info(f"Query processed successfully in {processing_time}ms")
            
            return QueryResponse(
                message_id=ai_message_id,
                kb_id=request.kb_id,
                created_at=end_time,
                message=ai_response,
                by="ai",
                model_used=request.model,
                sources=sources,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            
            # Save error message
            error_message = f"I apologize, but I encountered an error while processing your query: {str(e)}"
            try:
                await self.save_message(request.kb_id, error_message, "ai", model_used=request.model)
            except:
                pass
            
            raise RuntimeError(f"Failed to process query: {e}")

    async def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available models grouped by provider."""
        return {
            "gemini": [model for model, provider in AVAILABLE_MODELS.items() if provider == "gemini"],
            "claude": [model for model, provider in AVAILABLE_MODELS.items() if provider == "claude"]
        }