# api/vector_search.py
from datetime import datetime
from fastapi import APIRouter, HTTPException, status, Query, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import logging

from services.vector_service import (
    VectorSearchService, 
    VectorSearchRequest, 
    VectorSearchResponse,
    SimilaritySearchRequest,
    SearchResult
)
from core.db import get_db
from bson import ObjectId

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize service
vector_service = VectorSearchService()

# ---------- Dependency ----------
async def get_vector_service() -> VectorSearchService:
    return vector_service

# ---------- Routes ----------

@router.post("/search", response_model=VectorSearchResponse)
async def semantic_search(
    request: VectorSearchRequest,
    service: VectorSearchService = Depends(get_vector_service)
):
    """
    Perform semantic search across vector database.
    Can search within a specific KB or across all KBs.
    """
    try:
        logger.info(f"Semantic search request: {request.query[:50]}...")
        
        # Validate KB if specified
        if request.kb_id:
            db = get_db()
            kb = await db.kbs.find_one({"_id": ObjectId(request.kb_id)})
            if not kb:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Knowledge Base not found"
                )
        
        response = await service.semantic_search(request)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {str(e)}"
        )

@router.post("/{kb_id}/search", response_model=VectorSearchResponse)
async def search_within_kb(
    kb_id: str,
    query: str,
    top_k: int = Query(10, ge=1, le=50, description="Number of results to return"),
    score_threshold: float = Query(0.0, ge=0.0, le=1.0, description="Minimum similarity score"),
    service: VectorSearchService = Depends(get_vector_service)
):
    """
    Search for similar content within a specific knowledge base.
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
        
        request = VectorSearchRequest(
            query=query,
            kb_id=kb_id,
            top_k=top_k,
            score_threshold=score_threshold
        )
        
        response = await service.semantic_search(request)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"KB search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"KB search failed: {str(e)}"
        )

@router.post("/{kb_id}/find-similar", response_model=List[SearchResult])
async def find_similar_content(
    kb_id: str,
    text: str,
    top_k: int = Query(5, ge=1, le=20, description="Number of similar chunks to return"),
    min_score: float = Query(0.7, ge=0.0, le=1.0, description="Minimum similarity score"),
    exclude_files: Optional[List[str]] = Query(None, description="Source files to exclude"),
    service: VectorSearchService = Depends(get_vector_service)
):
    """
    Find similar text chunks to the provided text within a KB.
    Useful for finding related content or detecting duplicates.
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
        
        request = SimilaritySearchRequest(
            text=text,
            kb_id=kb_id,
            exclude_source_files=exclude_files,
            top_k=top_k,
            min_score=min_score
        )
        
        results = await service.find_similar_chunks(request)
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Find similar content failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Find similar content failed: {str(e)}"
        )

@router.get("/{kb_id}/stats")
async def get_kb_vector_stats(
    kb_id: str,
    service: VectorSearchService = Depends(get_vector_service)
):
    """
    Get statistics about vectors stored for a knowledge base.
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
        
        stats = await service.get_kb_statistics(kb_id)
        return JSONResponse(content=stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get KB stats failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get KB statistics: {str(e)}"
        )

@router.post("/{kb_id}/hybrid-search", response_model=List[SearchResult])
async def hybrid_search(
    kb_id: str,
    query: str,
    top_k: int = Query(10, ge=1, le=50, description="Number of results to return"),
    rerank: bool = Query(True, description="Whether to rerank results"),
    service: VectorSearchService = Depends(get_vector_service)
):
    """
    Advanced hybrid search combining semantic and keyword matching.
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
        
        results = await service.hybrid_search(
            query=query,
            kb_id=kb_id,
            top_k=top_k,
            rerank=rerank
        )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid search failed: {str(e)}"
        )

@router.delete("/{kb_id}/vectors")
async def delete_kb_vectors(
    kb_id: str,
    service: VectorSearchService = Depends(get_vector_service)
):
    """
    Delete all vectors for a knowledge base.
    WARNING: This action is irreversible!
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
        
        result = await service.delete_kb_vectors(kb_id)
        
        # Update KB status to indicate vectors were deleted
        await db.kbs.update_one(
            {"_id": ObjectId(kb_id)},
            {
                "$set": {
                    "status": "vectors_deleted",
                    "last_updated": datetime.utcnow()
                }
            }
        )
        
        return JSONResponse(
            content={
                "message": "All vectors for the knowledge base have been deleted",
                "details": result
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete KB vectors failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete KB vectors: {str(e)}"
        )

@router.post("/batch-similarity")
async def batch_similarity_search(
    texts: List[str],
    kb_id: str,
    top_k_per_text: int = Query(5, ge=1, le=10, description="Results per text"),
    service: VectorSearchService = Depends(get_vector_service)
):
    """
    Find similar content for multiple texts in batch.
    Useful for content analysis and bulk similarity checks.
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
        
        # Limit batch size to prevent abuse
        if len(texts) > 20:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch size cannot exceed 20 texts"
            )
        
        results = await service.batch_similarity_search(
            texts=texts,
            kb_id=kb_id,
            top_k_per_text=top_k_per_text
        )
        
        return JSONResponse(content=results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch similarity search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch similarity search failed: {str(e)}"
        )

@router.get("/health")
async def vector_service_health():
    """
    Check the health of vector search service and Pinecone connection.
    """
    try:
        # Test basic connectivity
        from services.vector_service import pc_index
        
        # Perform a minimal test query
        test_vector = [0.1] * 768
        test_results = pc_index.query(
            vector=test_vector,
            top_k=1,
            include_metadata=False
        )
        
        return JSONResponse(content={
            "status": "healthy",
            "service": "vector_search",
            "pinecone_connected": True,
            "embedding_model": "text-embedding-004",
            "embedding_dimension": 768,
            "test_query_successful": True
        })
        
    except Exception as e:
        logger.error(f"Vector service health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "service": "vector_search",
                "error": str(e),
                "pinecone_connected": False
            }
        )