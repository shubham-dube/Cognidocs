# services/chat_service.py
import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from bson import ObjectId
from pydantic import BaseModel

import google.generativeai as genai
from pinecone import Pinecone

from core.config import settings
from core.db import get_db

logger = logging.getLogger(__name__)

# Initialize clients
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
pc_index = pc.Index(settings.PINECONE_INDEX_NAME)
genai.configure(api_key=settings.GEMINI_API_KEY)

# ---------- Request/Response Models ----------
class QueryRequest(BaseModel):
    kb_id: str
    kb_name: str
    query: str
    chat_id: Optional[str] = None
    chat_title: Optional[str] = None
    model: str = "gemini-1.5-flash"
    max_results: int = 5
    temperature: float = 0.7

class SourceDocument(BaseModel):
    text: str
    source_file: str
    chunk_index: int
    relevance_score: float

class QueryResponse(BaseModel):
    chat_id: str
    message_id: str
    query: str
    response: str
    model_used: str
    sources: List[SourceDocument]
    timestamp: datetime
    processing_time_ms: int

# ---------- Chat Service Class ----------
class ChatService:
    def __init__(self):
        self.db = get_db()
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for the user's query."""
        try:
            logger.info(f"Generating embedding for query: {query[:50]}...")
            
            resp = genai.embed_content(
                model=settings.GEMINI_EMBED_MODEL,
                content=query,
                task_type="retrieval_query"  # Different task type for queries
            )
            
            embedding = resp["embedding"]
            logger.info(f"Generated query embedding with dimension: {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise RuntimeError(f"Failed to generate query embedding: {e}")
    
    async def search_similar_documents(self, kb_id: str, query_embedding: List[float], max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents in Pinecone vector database."""
        try:
            logger.info(f"Searching for similar documents in KB {kb_id}")
            
            # Query Pinecone with filters
            search_results = pc_index.query(
                vector=query_embedding,
                top_k=max_results,
                filter={"kb_id": {"$eq": kb_id}},  # Only search within this KB
                include_metadata=True
            )
            
            documents = []
            for match in search_results.matches:
                doc_data = {
                    "text": match.metadata.get("text", ""),
                    "source_file": match.metadata.get("source_file", "unknown"),
                    "chunk_index": match.metadata.get("chunk_index", 0),
                    "relevance_score": float(match.score),
                    "metadata": match.metadata
                }
                documents.append(doc_data)
            
            logger.info(f"Found {len(documents)} similar documents")
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
            doc_text = doc["text"].strip()
            source_file = doc["source_file"]
            relevance = doc["relevance_score"]
            
            # Format document with metadata
            doc_section = f"""
Document {i} (Source: {source_file}, Relevance: {relevance:.3f}):
{doc_text}
---
"""
            
            # Check if adding this document would exceed length limit
            if current_length + len(doc_section) > max_context_length:
                if i == 1:  # If even first document is too long, truncate it
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
    
    async def generate_response(self, query: str, context: str, kb_name: str, model: str, temperature: float, chat_history: List[Dict] = None) -> str:
        """Generate AI response using Gemini with the provided context."""
        try:
            logger.info(f"Generating response using model: {model}")
            
            # Build conversation history for context
            conversation_context = ""
            if chat_history:
                recent_history = chat_history[-6:]  # Last 3 exchanges
                for msg in recent_history:
                    role = "Human" if msg["role"] == "user" else "Assistant"
                    conversation_context += f"{role}: {msg['content']}\n\n"
            
            # Create prompt with system instructions
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
            
            # Initialize the model
            model_instance = genai.GenerativeModel(
                model_name=model,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=2000,
                    top_p=0.9,
                    top_k=40
                )
            )
            
            # Generate response
            response = model_instance.generate_content(system_prompt)
            answer = response.text
            
            if not answer or answer.strip() == "":
                answer = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
            logger.info(f"Generated response of length: {len(answer)}")
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate AI response: {e}")
            return f"I encountered an error while generating the response. Please try again later. Error: {str(e)}"
    
    async def get_or_create_chat(self, kb_id: str, kb_name: str, chat_id: Optional[str] = None, chat_title: Optional[str] = None) -> str:
        """Get existing chat or create a new one."""
        try:
            if chat_id:
                # Verify existing chat
                existing_chat = await self.db.chats.find_one({
                    "_id": ObjectId(chat_id),
                    "kb_id": kb_id
                })
                if existing_chat:
                    logger.info(f"Using existing chat: {chat_id}")
                    return chat_id
                else:
                    logger.warning(f"Chat {chat_id} not found, creating new chat")
            
            # Create new chat
            now = datetime.utcnow()
            chat_doc = {
                "kb_id": kb_id,
                "kb_name": kb_name,
                "title": chat_title,
                "created_at": now,
                "last_updated": now,
                "messages": []
            }
            
            result = await self.db.chats.insert_one(chat_doc)
            new_chat_id = str(result.inserted_id)
            logger.info(f"Created new chat: {new_chat_id}")
            return new_chat_id
            
        except Exception as e:
            logger.error(f"Failed to get or create chat: {e}")
            raise RuntimeError(f"Failed to get or create chat: {e}")
    
    async def get_chat_history(self, chat_id: str) -> List[Dict]:
        """Get chat message history."""
        try:
            chat = await self.db.chats.find_one({"_id": ObjectId(chat_id)})
            if chat:
                return chat.get("messages", [])
            return []
        except Exception as e:
            logger.error(f"Failed to get chat history: {e}")
            return []
    
    async def save_message_to_chat(self, chat_id: str, role: str, content: str, model_used: Optional[str] = None, sources: Optional[List[Dict]] = None):
        """Save a message to the chat history."""
        try:
            message = {
                "message_id": str(uuid.uuid4()),
                "role": role,  # 'user' or 'assistant'
                "content": content,
                "timestamp": datetime.utcnow(),
                "model_used": model_used,
                "sources": sources or []
            }
            
            await self.db.chats.update_one(
                {"_id": ObjectId(chat_id)},
                {
                    "$push": {"messages": message},
                    "$set": {"last_updated": datetime.utcnow()}
                }
            )
            
            logger.info(f"Saved {role} message to chat {chat_id}")
            return message["message_id"]
            
        except Exception as e:
            logger.error(f"Failed to save message to chat: {e}")
            raise RuntimeError(f"Failed to save message to chat: {e}")
    
    async def auto_generate_chat_title(self, query: str, response: str) -> str:
        """Auto-generate a concise title for the chat based on the first query."""
        try:
            # Simple heuristic title generation
            query_words = query.strip().split()
            if len(query_words) <= 6:
                return query.strip()[:50]
            
            # Extract key terms (simple approach)
            title = " ".join(query_words[:6]) + "..."
            return title[:50]
            
        except Exception as e:
            logger.warning(f"Failed to auto-generate title: {e}")
            return "New Chat"
    
    async def query_and_respond(self, request: QueryRequest) -> QueryResponse:
        """Main method to handle query, search, and response generation."""
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Processing query for KB {request.kb_id}: {request.query[:100]}...")
            
            # Get or create chat
            chat_id = await self.get_or_create_chat(
                request.kb_id, 
                request.kb_name, 
                request.chat_id, 
                request.chat_title
            )
            
            # Get chat history for context
            chat_history = await self.get_chat_history(chat_id)
            
            # Save user message
            user_message_id = await self.save_message_to_chat(chat_id, "user", request.query)
            
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
                    text=doc["text"][:500] + "..." if len(doc["text"]) > 500 else doc["text"],
                    source_file=doc["source_file"],
                    chunk_index=doc["chunk_index"],
                    relevance_score=doc["relevance_score"]
                )
                for doc in similar_docs
            ]
            
            # Save AI response with sources
            await self.save_message_to_chat(
                chat_id, 
                "assistant", 
                ai_response,
                model_used=request.model,
                sources=[source.dict() for source in sources]
            )
            
            # Auto-generate title for new chats
            if not request.chat_id and not request.chat_title and len(chat_history) == 0:
                auto_title = await self.auto_generate_chat_title(request.query, ai_response)
                await self.db.chats.update_one(
                    {"_id": ObjectId(chat_id)},
                    {"$set": {"title": auto_title}}
                )
            
            # Calculate processing time
            end_time = datetime.utcnow()
            processing_time = int((end_time - start_time).total_seconds() * 1000)
            
            logger.info(f"Query processed successfully in {processing_time}ms")
            
            return QueryResponse(
                chat_id=chat_id,
                message_id=str(uuid.uuid4()),
                query=request.query,
                response=ai_response,
                model_used=request.model,
                sources=sources,
                timestamp=end_time,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            
            # Try to save error message to chat if chat exists
            if 'chat_id' in locals():
                try:
                    await self.save_message_to_chat(
                        chat_id, 
                        "assistant", 
                        f"I apologize, but I encountered an error while processing your query: {str(e)}"
                    )
                except:
                    pass
            
            raise RuntimeError(f"Failed to process query: {e}")