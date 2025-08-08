# api/chats.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Set
from bson import ObjectId
from datetime import datetime
import logging

from core.db import get_db
from core.config import settings
from services.chat_service import ChatService, QueryRequest, QueryResponse, ChatMessage, SourceDocument

router = APIRouter()
logger = logging.getLogger(__name__)

# ---------- Pydantic Schemas ----------
class ChatMessageResponse(BaseModel):
    message_id: str
    created_at: datetime
    message: str
    by: str  # "user" or "ai"
    model_used: Optional[str] = None
    sources: Optional[List[SourceDocument]] = None
    unique_files_referenced: Optional[List[str]] = None  # Unique source files for this message

class KBChatResponse(BaseModel):
    kb_id: str
    messages: List[ChatMessageResponse]
    total_messages: int
    unique_files_referenced: Set[str]  # All unique files referenced in this KB's chats
    models_used: Set[str]  # All models used in this KB's chats

class KBWithRecentChatResponse(BaseModel):
    kb_id: str
    kb_name: str
    kb_type: str
    kb_description: Optional[str] = None
    created_at: datetime
    last_updated: datetime
    status: str
    processing_info: Optional[Dict[str, Any]] = None
    total_messages: int
    recent_chat: Optional[ChatMessageResponse] = None
    unique_files_referenced: Set[str]
    models_used: Set[str]

class QueryChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User's question")
    model: Optional[str] = Field("gemini-1.5-flash", description="Model to use for response")
    max_results: Optional[int] = Field(10, ge=1, le=20, description="Max number of source documents")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Response creativity level")

class ModelsResponse(BaseModel):
    available_models: Dict[str, List[str]]
    default_model: str

# ---------- Helper Functions ----------
def extract_unique_files_from_sources(sources: Optional[List[SourceDocument]]) -> List[str]:
    """Extract unique source files from sources."""
    if not sources:
        return []
    return list(set(source.source_file for source in sources))

# ---------- Routes ----------

@router.get("/models", response_model=ModelsResponse)
async def get_available_models():
    """Get list of available models for chat."""
    try:
        chat_service = ChatService()
        available_models = await chat_service.get_available_models()
        
        return ModelsResponse(
            available_models=available_models,
            default_model="gemini-1.5-flash"
        )
        
    except Exception as e:
        logger.error(f"Error fetching available models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch available models"
        )

@router.get("/kbs-overview", response_model=List[KBWithRecentChatResponse])
async def get_all_kbs_with_recent_chat():
    """
    Get all knowledge bases with their most recent chat message and metadata.
    Includes processing status and other relevant information.
    """
    try:
        db = get_db()
        
        # Get all KBs with their status
        kbs_cursor = db.kbs.find({}).sort("created_at", -1)
        results = []
        
        async for kb in kbs_cursor:
            kb_id = str(kb["_id"])
            
            # Get total message count for this KB
            total_messages = await db.chat_messages.count_documents({"kb_id": kb_id})
            
            # Get most recent chat message
            recent_message = None
            recent_msg_doc = await db.chat_messages.find_one(
                {"kb_id": kb_id},
                sort=[("created_at", -1)]
            )
            
            # Collect unique files and models used
            unique_files = set()
            models_used = set()
            
            if recent_msg_doc:
                sources = None
                unique_files_list = []
                
                if recent_msg_doc.get("sources"):
                    sources = [SourceDocument(**source) for source in recent_msg_doc["sources"]]
                    unique_files_list = extract_unique_files_from_sources(sources)
                    unique_files.update(unique_files_list)
                
                if recent_msg_doc.get("model_used"):
                    models_used.add(recent_msg_doc["model_used"])
                
                recent_message = ChatMessageResponse(
                    message_id=str(recent_msg_doc["_id"]),
                    created_at=recent_msg_doc["created_at"],
                    message=recent_msg_doc["message"],
                    by=recent_msg_doc["by"],
                    model_used=recent_msg_doc.get("model_used"),
                    sources=sources,
                    unique_files_referenced=unique_files_list
                )
            
            # Get all unique files and models for this KB (for complete metadata)
            all_messages_cursor = db.chat_messages.find(
                {"kb_id": kb_id, "sources": {"$exists": True, "$ne": []}},
                {"sources": 1, "model_used": 1}
            )
            
            async for msg in all_messages_cursor:
                if msg.get("sources"):
                    for source in msg["sources"]:
                        if source.get("source_file"):
                            unique_files.add(source["source_file"])
                if msg.get("model_used"):
                    models_used.add(msg["model_used"])
            
            # Get processing info if available
            processing_info = None
            if kb.get("processing_stats"):
                processing_info = {
                    "total_files": kb["processing_stats"].get("total_files", 0),
                    "processed_files": kb["processing_stats"].get("processed_files", 0),
                    "failed_files": kb["processing_stats"].get("failed_files", 0),
                    "total_chunks": kb["processing_stats"].get("total_chunks", 0)
                }
            
            results.append(KBWithRecentChatResponse(
                kb_id=kb_id,
                kb_name=kb["name"],
                kb_type=kb.get("kb_type", "generic"),
                kb_description=kb.get("description"),
                created_at=kb["created_at"],
                last_updated=kb.get("last_updated", kb["created_at"]),
                status=kb.get("status", "unknown"),
                processing_info=processing_info,
                total_messages=total_messages,
                recent_chat=recent_message,
                unique_files_referenced=unique_files,
                models_used=models_used
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Error fetching KBs overview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch knowledge bases overview"
        )

@router.get("/{kb_id}/chats", response_model=KBChatResponse)
async def get_all_chats_for_kb(
    kb_id: str,
    skip: int = Query(0, ge=0, description="Number of messages to skip"),
    limit: int = Query(50, ge=1, le=200, description="Number of messages to return")
):
    """
    Get all chat messages for a specific knowledge base with complete details.
    No separate chat detail endpoint needed - everything is returned here.
    """
    try:
        db = get_db()
        
        # Verify KB exists
        kb = await db.kbs.find_one({"_id": ObjectId(kb_id)})
        if not kb:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Knowledge Base not found"
            )
        
        # Get total message count
        total_messages = await db.chat_messages.count_documents({"kb_id": kb_id})
        
        # Get messages with pagination
        messages_cursor = db.chat_messages.find(
            {"kb_id": kb_id}
        ).sort("created_at", -1).skip(skip).limit(limit)
        
        messages = []
        unique_files = set()
        models_used = set()
        
        async for msg_doc in messages_cursor:
            sources = None
            unique_files_list = []
            
            if msg_doc.get("sources"):
                sources = [SourceDocument(**source) for source in msg_doc["sources"]]
                unique_files_list = extract_unique_files_from_sources(sources)
                unique_files.update(unique_files_list)
            
            if msg_doc.get("model_used"):
                models_used.add(msg_doc["model_used"])
            
            messages.append(ChatMessageResponse(
                message_id=str(msg_doc["_id"]),
                created_at=msg_doc["created_at"],
                message=msg_doc["message"],
                by=msg_doc["by"],
                model_used=msg_doc.get("model_used"),
                sources=sources,
                unique_files_referenced=unique_files_list
            ))
        
        # Reverse to get chronological order (oldest first)
        messages.reverse()
        
        return KBChatResponse(
            kb_id=kb_id,
            messages=messages,
            total_messages=total_messages,
            unique_files_referenced=unique_files,
            models_used=models_used
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching chats for KB {kb_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch chats for knowledge base"
        )

@router.post("/{kb_id}/query", response_model=QueryResponse)
async def query_knowledge_base(kb_id: str, request: QueryChatRequest):
    """
    Query a knowledge base and get an AI response with sources.
    Uses the specified model for generating the response.
    """
    try:
        db = get_db()
        
        # Verify KB exists and is ready
        kb = await db.kbs.find_one({"_id": ObjectId(kb_id)})
        if not kb:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Knowledge Base not found"
            )
        
        # Check if KB has completed ingestion
        kb_status = kb.get("status", "unknown")
        if kb_status not in ["completed", "partially_completed"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Knowledge Base is not ready for queries. Status: {kb_status}"
            )
        
        # Use chat service to handle the query
        chat_service = ChatService()
        
        query_request = QueryRequest(
            kb_id=kb_id,
            kb_name=kb["name"],
            query=request.query,
            model=request.model,
            max_results=request.max_results,
            temperature=request.temperature
        )
        
        response = await chat_service.query_and_respond(query_request)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying KB {kb_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )

@router.delete("/{kb_id}/chats")
async def delete_all_chats_for_kb(kb_id: str):
    """
    Delete all chat messages for a specific knowledge base.
    """
    try:
        db = get_db()
        
        # Verify KB exists
        kb = await db.kbs.find_one({"_id": ObjectId(kb_id)})
        if not kb:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Knowledge Base not found"
            )
        
        # Delete all chat messages for this KB
        result = await db.chat_messages.delete_many({"kb_id": kb_id})
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": f"Deleted {result.deleted_count} chat messages for KB {kb_id}",
                "deleted_count": result.deleted_count
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chats for KB {kb_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete chats"
        )

@router.delete("/{kb_id}/messages/{message_id}")
async def delete_specific_message(kb_id: str, message_id: str):
    """
    Delete a specific chat message.
    """
    try:
        db = get_db()
        
        # Delete the specific message
        result = await db.chat_messages.delete_one({
            "_id": ObjectId(message_id),
            "kb_id": kb_id
        })
        
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Message not found"
            )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "Message deleted successfully", "message_id": message_id}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting message {message_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete message"
        )