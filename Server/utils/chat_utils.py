# utils/chat_utils.py
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from bson import ObjectId
import hashlib

from core.db import get_db

logger = logging.getLogger(__name__)

# ---------- Text Processing Utilities ----------

def clean_query_text(query: str) -> str:
    """Clean and normalize user query text."""
    if not query:
        return ""
    
    # Remove extra whitespace
    query = re.sub(r'\s+', ' ', query.strip())
    
    # Remove potentially harmful characters
    query = re.sub(r'[<>{}]', '', query)
    
    # Limit length
    if len(query) > 2000:
        query = query[:2000] + "..."
        logger.warning("Query truncated due to length limit")
    
    return query

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract relevant keywords from text."""
    if not text:
        return []
    
    # Simple keyword extraction (you can replace with more sophisticated NLP)
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
        'could', 'can', 'may', 'might', 'must', 'shall', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
        'us', 'them', 'what', 'when', 'where', 'why', 'how', 'who'
    }
    
    # Extract words (alphanumeric, 3+ chars, not stop words)
    words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', text.lower())
    keywords = [word for word in words if word not in stop_words]
    
    # Count frequency and return top keywords
    word_freq = {}
    for word in keywords:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_keywords[:max_keywords]]

def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to specified length with proper word boundaries."""
    if not text or len(text) <= max_length:
        return text
    
    # Find last space before max_length
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If space is reasonably close to limit
        truncated = truncated[:last_space]
    
    return truncated + suffix

def generate_text_hash(text: str) -> str:
    """Generate a hash for text content (useful for deduplication)."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# ---------- Chat Management Utilities ----------

async def get_user_chat_history(user_id: str, kb_id: Optional[str] = None, 
                               limit: int = 20) -> List[Dict[str, Any]]:
    """Get user's chat history with optional KB filtering."""
    try:
        db = get_db()
        
        # Build filter
        filter_dict = {"user_id": user_id} if user_id else {}
        if kb_id:
            filter_dict["kb_id"] = kb_id
        
        # Get chats
        cursor = db.chats.find(filter_dict).sort("last_updated", -1).limit(limit)
        chats = []
        
        async for chat in cursor:
            chat_info = {
                "chat_id": str(chat["_id"]),
                "kb_id": chat["kb_id"],
                "kb_name": chat.get("kb_name", "Unknown KB"),
                "title": chat.get("title"),
                "created_at": chat["created_at"],
                "last_updated": chat["last_updated"],
                "message_count": len(chat.get("messages", []))
            }
            chats.append(chat_info)
        
        return chats
        
    except Exception as e:
        logger.error(f"Failed to get user chat history: {e}")
        return []

async def cleanup_old_chats(days_old: int = 90, dry_run: bool = True) -> Dict[str, int]:
    """Clean up old chats to free up storage."""
    try:
        db = get_db()
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        # Find old chats
        old_chats_cursor = db.chats.find({
            "last_updated": {"$lt": cutoff_date}
        })
        
        chat_count = 0
        message_count = 0
        
        async for chat in old_chats_cursor:
            chat_count += 1
            message_count += len(chat.get("messages", []))
            
            if not dry_run:
                await db.chats.delete_one({"_id": chat["_id"]})
        
        result = {
            "chats_found": chat_count,
            "messages_found": message_count,
            "chats_deleted": chat_count if not dry_run else 0,
            "dry_run": dry_run
        }
        
        logger.info(f"Cleanup old chats result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to cleanup old chats: {e}")
        return {"error": str(e)}

def validate_chat_message(content: str, role: str) -> Tuple[bool, Optional[str]]:
    """Validate chat message content and role."""
    if not content or not content.strip():
        return False, "Message content cannot be empty"
    
    if role not in ["user", "assistant"]:
        return False, "Role must be either 'user' or 'assistant'"
    
    if len(content) > 10000:
        return False, "Message content too long (max 10,000 characters)"
    
    # Check for potentially harmful content
    harmful_patterns = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
        r'<iframe[^>]*>.*?</iframe>'
    ]
    
    for pattern in harmful_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return False, "Message contains potentially harmful content"
    
    return True, None

# ---------- Response Generation Utilities ----------

def format_sources_for_display(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format source documents for user display."""
    formatted_sources = []
    
    for i, source in enumerate(sources, 1):
        formatted = {
            "index": i,
            "source_file": source.get("source_file", "Unknown"),
            "snippet": truncate_text(source.get("text", ""), 200),
            "relevance_score": round(source.get("relevance_score", 0.0), 3),
            "chunk_index": source.get("chunk_index", 0)
        }
        formatted_sources.append(formatted)
    
    return formatted_sources

def calculate_response_quality_score(response: str, sources: List[Dict], query: str) -> float:
    """Calculate a quality score for the generated response."""
    try:
        score = 0.0
        
        # Length score (not too short, not too long)
        length = len(response)
        if 50 <= length <= 1000:
            score += 0.3
        elif length > 20:
            score += 0.1
        
        # Source utilization score
        if sources:
            score += min(0.3, len(sources) * 0.1)
        
        # Query relevance (simple keyword overlap)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words.intersection(response_words))
        relevance = overlap / max(len(query_words), 1)
        score += relevance * 0.4
        
        return min(1.0, score)
        
    except Exception as e:
        logger.warning(f"Failed to calculate response quality score: {e}")
        return 0.5

# ---------- Analytics Utilities ----------

async def get_chat_analytics(kb_id: Optional[str] = None, 
                           days_back: int = 30) -> Dict[str, Any]:
    """Get analytics for chats and queries."""
    try:
        db = get_db()
        
        # Date filter
        start_date = datetime.utcnow() - timedelta(days=days_back)
        
        # Build filter
        date_filter = {"created_at": {"$gte": start_date}}
        if kb_id:
            date_filter["kb_id"] = kb_id
        
        # Get basic stats
        total_chats = await db.chats.count_documents(date_filter)
        
        # Get chats with messages for detailed analysis
        cursor = db.chats.find(date_filter, {"messages": 1, "kb_id": 1, "created_at": 1})
        
        total_messages = 0
        user_messages = 0
        assistant_messages = 0
        kb_usage = {}
        daily_activity = {}
        
        async for chat in cursor:
            messages = chat.get("messages", [])
            total_messages += len(messages)
            
            kb_id_current = chat.get("kb_id", "unknown")
            kb_usage[kb_id_current] = kb_usage.get(kb_id_current, 0) + 1
            
            # Daily activity
            date_key = chat["created_at"].strftime("%Y-%m-%d")
            daily_activity[date_key] = daily_activity.get(date_key, 0) + 1
            
            for msg in messages:
                if msg.get("role") == "user":
                    user_messages += 1
                elif msg.get("role") == "assistant":
                    assistant_messages += 1
        
        # Calculate averages
        avg_messages_per_chat = total_messages / max(total_chats, 1)
        
        analytics = {
            "period_days": days_back,
            "total_chats": total_chats,
            "total_messages": total_messages,
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "avg_messages_per_chat": round(avg_messages_per_chat, 2),
            "kb_usage": kb_usage,
            "daily_activity": daily_activity,
            "generated_at": datetime.utcnow()
        }
        
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to get chat analytics: {e}")
        return {"error": str(e)}

async def track_query_performance(kb_id: str, query: str, response_time_ms: int, 
                                sources_count: int, user_feedback: Optional[str] = None):
    """Track query performance metrics."""
    try:
        db = get_db()
        
        performance_doc = {
            "kb_id": kb_id,
            "query_hash": generate_text_hash(query),
            "query_length": len(query),
            "response_time_ms": response_time_ms,
            "sources_count": sources_count,
            "user_feedback": user_feedback,
            "timestamp": datetime.utcnow()
        }
        
        await db.query_performance.insert_one(performance_doc)
        logger.debug(f"Tracked query performance for KB {kb_id}")
        
    except Exception as e:
        logger.warning(f"Failed to track query performance: {e}")

# ---------- Content Moderation Utilities ----------

def moderate_content(text: str) -> Tuple[bool, List[str]]:
    """Basic content moderation for chat messages."""
    issues = []
    
    # Check for excessive repetition
    words = text.split()
    if len(words) > 10:
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        max_count = max(word_counts.values())
        if max_count > len(words) * 0.3:
            issues.append("excessive_repetition")
    
    # Check for extremely long words (potential spam)
    long_words = [word for word in words if len(word) > 50]
    if long_words:
        issues.append("suspicious_long_words")
    
    # Check for excessive special characters
    special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', text)) / max(len(text), 1)
    if special_char_ratio > 0.3:
        issues.append("excessive_special_characters")
    
    # Check for potential injection attempts
    sql_patterns = ['union select', 'drop table', 'insert into', 'delete from']
    if any(pattern in text.lower() for pattern in sql_patterns):
        issues.append("potential_sql_injection")
    
    is_safe = len(issues) == 0
    return is_safe, issues

# ---------- Export Utilities ----------

async def export_chat_history(kb_id: str, format_type: str = "json") -> Dict[str, Any]:
    """Export chat history for a KB in various formats."""
    try:
        db = get_db()
        
        # Get KB info
        kb = await db.kbs.find_one({"_id": ObjectId(kb_id)})
        if not kb:
            raise ValueError("Knowledge base not found")
        
        # Get all chats for this KB
        cursor = db.chats.find({"kb_id": kb_id}).sort("created_at", 1)
        
        export_data = {
            "kb_info": {
                "kb_id": kb_id,
                "kb_name": kb.get("name", "Unknown"),
                "kb_type": kb.get("kb_type", "generic"),
                "description": kb.get("description", "")
            },
            "export_metadata": {
                "generated_at": datetime.utcnow(),
                "format": format_type,
                "total_chats": 0,
                "total_messages": 0
            },
            "chats": []
        }
        
        total_chats = 0
        total_messages = 0
        
        async for chat in cursor:
            chat_data = {
                "chat_id": str(chat["_id"]),
                "title": chat.get("title"),
                "created_at": chat["created_at"],
                "last_updated": chat["last_updated"],
                "messages": []
            }
            
            for msg in chat.get("messages", []):
                message_data = {
                    "message_id": msg.get("message_id"),
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg["timestamp"],
                    "model_used": msg.get("model_used"),
                    "sources": msg.get("sources", [])
                }
                chat_data["messages"].append(message_data)
                total_messages += 1
            
            export_data["chats"].append(chat_data)
            total_chats += 1
        
        export_data["export_metadata"]["total_chats"] = total_chats
        export_data["export_metadata"]["total_messages"] = total_messages
        
        return export_data
        
    except Exception as e:
        logger.error(f"Failed to export chat history: {e}")
        raise RuntimeError(f"Export failed: {e}")

# ---------- Health Check Utilities ----------

async def health_check_chat_system() -> Dict[str, Any]:
    """Perform health check on chat system components."""
    try:
        db = get_db()
        
        # Test database connectivity
        db_healthy = True
        try:
            await db.chats.count_documents({}, limit=1)
        except Exception as e:
            db_healthy = False
            logger.error(f"Database health check failed: {e}")
        
        # Test recent activity
        recent_chats = await db.chats.count_documents({
            "created_at": {"$gte": datetime.utcnow() - timedelta(hours=24)}
        })
        
        # Test embedding service (basic)
        embedding_healthy = True
        try:
            import google.generativeai as genai
            from core.config import settings
            genai.configure(api_key=settings.GEMINI_API_KEY)
            # Simple test
            test_response = genai.embed_content(
                model=settings.GEMINI_EMBED_MODEL,
                content="test"
            )
            if not test_response.get("embedding"):
                embedding_healthy = False
        except Exception as e:
            embedding_healthy = False
            logger.error(f"Embedding service health check failed: {e}")
        
        health_status = {
            "status": "healthy" if (db_healthy and embedding_healthy) else "degraded",
            "components": {
                "database": "healthy" if db_healthy else "unhealthy",
                "embedding_service": "healthy" if embedding_healthy else "unhealthy"
            },
            "metrics": {
                "recent_chats_24h": recent_chats
            },
            "timestamp": datetime.utcnow()
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow()
        }