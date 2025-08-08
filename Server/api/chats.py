# api/chats.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from bson import ObjectId
from datetime import datetime
import uuid
import logging

from core.db import get_db
from core.config import settings
from services.chat_service import ChatService, QueryRequest, QueryResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# ---------- Utilities ----------
def oid_to_str(o: ObjectId) -> str:
    return str(o)

# ---------- Pydantic Schemas ----------
class MessageResponse(BaseModel):
    message_id: str
    role: str = Field(..., description="'user' or 'assistant'")
    content: str
    timestamp: datetime
    model_used: Optional[str] = Field(None, description="Model used for assistant responses")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Source documents for assistant responses")

class ChatResponse(BaseModel):
    chat_id: str
    kb_id: str
    kb_name: str
    title: Optional[str] = None
    created_at: datetime
    last_updated: datetime
    message_count: int
    last_message_preview: Optional[str] = None

class ChatDetailResponse(BaseModel):
    chat_id: str
    kb_id: str
    kb_name: str
    title: Optional[str] = None
    created_at: datetime
    last_updated: datetime
    messages: List[MessageResponse]

class KBWithChatsResponse(BaseModel):
    kb_id: str
    kb_name: str
    kb_type: str
    kb_description: Optional[str] = None
    created_at: datetime
    chat_count: int
    recent_chats: List[ChatResponse]

class QueryChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User's question")
    chat_id: Optional[str] = Field(None, description="Existing chat ID, if continuing conversation")
    chat_title: Optional[str] = Field(None, description="Optional title for new chat")
    model: Optional[str] = Field("gemini-1.5-flash", description="Model to use for response")
    max_results: Optional[int] = Field(5, ge=1, le=20, description="Max number of source documents")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Response creativity level")

# ---------- Routes ----------

@router.get("/kbs-with-chats", response_model=List[KBWithChatsResponse])
async def get_all_kbs_with_chats(
    limit: int = Query(10, ge=1, le=50, description="Number of recent chats per KB"),
    include_empty: bool = Query(True, description="Include KBs with no chats")
):
    """
    Get all knowledge bases with their recent chat data.
    Useful for dashboard/overview pages.
    """
    try:
        db = get_db()
        
        # Get all KBs
        kbs_cursor = db.kbs.find({}).sort("created_at", -1)
        results = []
        
        async for kb in kbs_cursor:
            kb_id = str(kb["_id"])
            
            # Get chat count and recent chats for this KB
            chat_count = await db.chats.count_documents({"kb_id": kb_id})
            
            # Skip KBs with no chats if requested
            if not include_empty and chat_count == 0:
                continue
            
            recent_chats = []
            if chat_count > 0:
                chats_cursor = db.chats.find(
                    {"kb_id": kb_id}
                ).sort("last_updated", -1).limit(limit)
                
                async for chat in chats_cursor:
                    # Get message count and last message preview
                    message_count = len(chat.get("messages", []))
                    last_message_preview = None
                    
                    if chat.get("messages"):
                        last_msg = chat["messages"][-1]
                        content = last_msg.get("content", "")
                        last_message_preview = content[:100] + "..." if len(content) > 100 else content
                    
                    recent_chats.append(ChatResponse(
                        chat_id=str(chat["_id"]),
                        kb_id=kb_id,
                        kb_name=kb["name"],
                        title=chat.get("title"),
                        created_at=chat["created_at"],
                        last_updated=chat["last_updated"],
                        message_count=message_count,
                        last_message_preview=last_message_preview
                    ))
            
            results.append(KBWithChatsResponse(
                kb_id=kb_id,
                kb_name=kb["name"],
                kb_type=kb.get("kb_type", "generic"),
                kb_description=kb.get("description"),
                created_at=kb["created_at"],
                chat_count=chat_count,
                recent_chats=recent_chats
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Error fetching KBs with chats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch knowledge bases with chats"
        )

@router.get("/{kb_id}/chats", response_model=List[ChatResponse])
async def get_chats_for_kb(
    kb_id: str,
    skip: int = Query(0, ge=0, description="Number of chats to skip"),
    limit: int = Query(20, ge=1, le=100, description="Number of chats to return")
):
    """
    Get all chats for a specific knowledge base with pagination.
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
        
        # Get chats for this KB
        chats_cursor = db.chats.find(
            {"kb_id": kb_id}
        ).sort("last_updated", -1).skip(skip).limit(limit)
        
        results = []
        async for chat in chats_cursor:
            # Calculate message count and preview
            messages = chat.get("messages", [])
            message_count = len(messages)
            last_message_preview = None
            
            if messages:
                last_msg = messages[-1]
                content = last_msg.get("content", "")
                last_message_preview = content[:100] + "..." if len(content) > 100 else content
            
            results.append(ChatResponse(
                chat_id=str(chat["_id"]),
                kb_id=kb_id,
                kb_name=kb["name"],
                title=chat.get("title"),
                created_at=chat["created_at"],
                last_updated=chat["last_updated"],
                message_count=message_count,
                last_message_preview=last_message_preview
            ))
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching chats for KB {kb_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch chats for knowledge base"
        )

@router.get("/{kb_id}/chats/{chat_id}", response_model=ChatDetailResponse)
async def get_chat_detail(kb_id: str, chat_id: str):
    """
    Get detailed chat information with all messages for a specific chat.
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
        
        # Get chat
        chat = await db.chats.find_one({
            "_id": ObjectId(chat_id),
            "kb_id": kb_id
        })
        if not chat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found"
            )
        
        # Convert messages to response format
        messages = []
        for msg in chat.get("messages", []):
            messages.append(MessageResponse(
                message_id=msg.get("message_id", str(uuid.uuid4())),
                role=msg["role"],
                content=msg["content"],
                timestamp=msg["timestamp"],
                model_used=msg.get("model_used"),
                sources=msg.get("sources")
            ))
        
        return ChatDetailResponse(
            chat_id=str(chat["_id"]),
            kb_id=kb_id,
            kb_name=kb["name"],
            title=chat.get("title"),
            created_at=chat["created_at"],
            last_updated=chat["last_updated"],
            messages=messages
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching chat detail {chat_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch chat details"
        )

@router.post("/{kb_id}/query", response_model=QueryResponse)
async def query_knowledge_base(kb_id: str, request: QueryChatRequest):
    """
    Query a knowledge base and get an AI response with sources.
    Can create a new chat or continue an existing one.
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
            chat_id=request.chat_id,
            chat_title=request.chat_title,
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

@router.delete("/{kb_id}/chats/{chat_id}")
async def delete_chat(kb_id: str, chat_id: str):
    """
    Delete a specific chat and all its messages.
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
        
        # Delete the chat
        result = await db.chats.delete_one({
            "_id": ObjectId(chat_id),
            "kb_id": kb_id
        })
        
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found"
            )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "Chat deleted successfully", "chat_id": chat_id}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chat {chat_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete chat"
        )

@router.patch("/{kb_id}/chats/{chat_id}/title")
async def update_chat_title(kb_id: str, chat_id: str, title: str = Query(..., min_length=1, max_length=200)):
    """
    Update the title of a specific chat.
    """
    try:
        db = get_db()
        
        # Update chat title
        result = await db.chats.update_one(
            {
                "_id": ObjectId(chat_id),
                "kb_id": kb_id
            },
            {
                "$set": {
                    "title": title.strip(),
                    "last_updated": datetime.utcnow()
                }
            }
        )
        
        if result.matched_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found"
            )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "Chat title updated successfully", "title": title.strip()}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating chat title {chat_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update chat title"
        )