# services/vector_service.py
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

import google.generativeai as genai
from pinecone import Pinecone
from pydantic import BaseModel
from bson import ObjectId

from core.config import settings
from core.db import get_db

logger = logging.getLogger(__name__)

# Initialize clients
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
pc_index = pc.Index(settings.PINECONE_INDEX_NAME)
genai.configure(api_key=settings.GEMINI_API_KEY)

# ---------- Models ----------
class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    source_file: str
    chunk_index: int
    kb_id: str
    metadata: Dict[str, Any]

class VectorSearchRequest(BaseModel):
    query: str
    kb_id: Optional[str] = None
    top_k: int = 10
    score_threshold: float = 0.0
    include_metadata: bool = True

class VectorSearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time_ms: int
    embedding_dimension: int

class SimilaritySearchRequest(BaseModel):
    text: str
    kb_id: Optional[str] = None
    exclude_source_files: Optional[List[str]] = None
    top_k: int = 5
    min_score: float = 0.7

# ---------- Vector Service Class ----------
class VectorSearchService:
    def __init__(self):
        self.db = None 
        self.embed_model = settings.GEMINI_EMBED_MODEL

    def _ensure_db(self):
        if self.db is None:
            self.db = get_db()  # connect once
        return self.db
    
    async def generate_embedding(self, text: str, task_type: str = "retrieval_query") -> List[float]:
        """Generate embedding for text using Gemini."""
        try:
            logger.debug(f"Generating embedding for text: {text[:100]}...")
            
            resp = genai.embed_content(
                model=self.embed_model,
                content=text,
                task_type=task_type
            )
            
            embedding = resp["embedding"]
            logger.debug(f"Generated embedding with dimension: {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")
    
    async def semantic_search(self, request: VectorSearchRequest) -> VectorSearchResponse:
        """
        Perform semantic search in the vector database.
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Performing semantic search: {request.query[:100]}...")
            
            # Generate query embedding
            query_embedding = await self.generate_embedding(request.query, "retrieval_query")
            
            # Build filter for Pinecone query
            filter_dict = {}
            if request.kb_id:
                filter_dict["kb_id"] = {"$eq": request.kb_id}
            
            # Query Pinecone
            search_results = pc_index.query(
                vector=query_embedding,
                top_k=request.top_k,
                filter=filter_dict if filter_dict else None,
                include_metadata=request.include_metadata
            )
            
            # Process results
            results = []
            for match in search_results.matches:
                if match.score >= request.score_threshold:
                    result = SearchResult(
                        id=match.id,
                        score=float(match.score),
                        text=match.metadata.get("text", ""),
                        source_file=match.metadata.get("source_file", "unknown"),
                        chunk_index=match.metadata.get("chunk_index", 0),
                        kb_id=match.metadata.get("kb_id", ""),
                        metadata=match.metadata or {}
                    )
                    results.append(result)
            
            # Calculate processing time
            end_time = datetime.utcnow()
            processing_time = int((end_time - start_time).total_seconds() * 1000)
            
            logger.info(f"Found {len(results)} results in {processing_time}ms")
            
            return VectorSearchResponse(
                query=request.query,
                results=results,
                total_results=len(results),
                processing_time_ms=processing_time,
                embedding_dimension=len(query_embedding)
            )
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise RuntimeError(f"Semantic search failed: {e}")
    
    async def find_similar_chunks(self, request: SimilaritySearchRequest) -> List[SearchResult]:
        """
        Find similar text chunks for a given text (useful for duplicate detection, related content).
        """
        try:
            logger.info(f"Finding similar chunks for text: {request.text[:100]}...")
            
            # Generate embedding for the input text
            text_embedding = await self.generate_embedding(request.text, "retrieval_document")
            
            # Build filter
            filter_dict = {}
            if request.kb_id:
                filter_dict["kb_id"] = {"$eq": request.kb_id}
            
            if request.exclude_source_files:
                filter_dict["source_file"] = {"$nin": request.exclude_source_files}
            
            # Search for similar chunks
            search_results = pc_index.query(
                vector=text_embedding,
                top_k=request.top_k,
                filter=filter_dict if filter_dict else None,
                include_metadata=True
            )
            
            # Filter by minimum score and convert to results
            similar_chunks = []
            for match in search_results.matches:
                if match.score >= request.min_score:
                    result = SearchResult(
                        id=match.id,
                        score=float(match.score),
                        text=match.metadata.get("text", ""),
                        source_file=match.metadata.get("source_file", "unknown"),
                        chunk_index=match.metadata.get("chunk_index", 0),
                        kb_id=match.metadata.get("kb_id", ""),
                        metadata=match.metadata or {}
                    )
                    similar_chunks.append(result)
            
            logger.info(f"Found {len(similar_chunks)} similar chunks")
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Finding similar chunks failed: {e}")
            raise RuntimeError(f"Finding similar chunks failed: {e}")
    
    async def get_kb_statistics(self, kb_id: str) -> Dict[str, Any]:
        """
        Get statistics about vectors stored for a specific knowledge base.
        """
        self.db = self._ensure_db()
        try:
            logger.info(f"Getting statistics for KB: {kb_id}")
            
            # Query a small sample to get stats (Pinecone doesn't have direct count)
            sample_results = pc_index.query(
                vector=[0.0] * 768,  # Dummy vector
                top_k=1,
                filter={"kb_id": {"$eq": kb_id}},
                include_metadata=True
            )
            
            if not sample_results.matches:
                return {
                    "kb_id": kb_id,
                    "total_vectors": 0,
                    "source_files": [],
                    "total_chunks": 0,
                    "embedding_dimension": 768
                }
            
            # Get more comprehensive stats by querying database
            kb_doc = await self.db.kbs.find_one({"_id": ObjectId(kb_id)})
            if not kb_doc:
                raise ValueError(f"Knowledge base {kb_id} not found")
            
            # Estimate vector count by doing multiple queries (approximate)
            # This is a limitation of Pinecone - no direct count method
            vector_count_estimate = await self._estimate_vector_count(kb_id)
            
            # Get source file information from processing info
            processing_info = kb_doc.get("processing_info", {})
            completed_files = processing_info.get("completed_files", [])
            
            stats = {
                "kb_id": kb_id,
                "kb_name": kb_doc.get("name", ""),
                "total_vectors_estimate": vector_count_estimate,
                "source_files": completed_files,
                "total_source_files": len(completed_files),
                "embedding_dimension": 768,
                "status": kb_doc.get("status", "unknown"),
                # "created_at": kb_doc.get("created_at"),
                # "last_updated": kb_doc.get("last_updated")
            }
            
            logger.info(f"Retrieved statistics for KB {kb_id}")
            return stats
            
        except Exception as e:
            logger.error(f"Getting KB statistics failed: {e}")
            raise RuntimeError(f"Getting KB statistics failed: {e}")
    
    async def _estimate_vector_count(self, kb_id: str) -> int:
        """
        Estimate vector count using sampling method (Pinecone limitation).
        """
        try:
            # Use multiple random queries to estimate count
            sample_queries = 5
            total_found = 0
            
            for _ in range(sample_queries):
                # Use random vector for sampling
                import random
                random_vector = [random.uniform(-1, 1) for _ in range(768)]
                
                results = pc_index.query(
                    vector=random_vector,
                    top_k=100,  # Max per query
                    filter={"kb_id": {"$eq": kb_id}},
                    include_metadata=False
                )
                total_found += len(results.matches)
            
            # Rough estimate (not very accurate but gives an idea)
            estimated_count = int(total_found / sample_queries * 50)  # Rough multiplier
            return max(estimated_count, total_found)
            
        except Exception:
            logger.warning("Could not estimate vector count, returning 0")
            return 0
    
    async def delete_kb_vectors(self, kb_id: str) -> Dict[str, Any]:
        """
        Delete all vectors for a specific knowledge base.
        WARNING: This is irreversible!
        """
        try:
            logger.info(f"Deleting all vectors for KB: {kb_id}")
            
            # Get stats before deletion
            stats_before = await self.get_kb_statistics(kb_id)
            
            # Delete vectors by filter
            delete_response = pc_index.delete(filter={"kb_id": {"$eq": kb_id}})
            
            logger.info(f"Deleted vectors for KB {kb_id}")
            
            return {
                "kb_id": kb_id,
                "deleted": True,
                "vectors_before_deletion": stats_before.get("total_vectors_estimate", 0),
                "delete_response": delete_response
            }
            
        except Exception as e:
            logger.error(f"Deleting KB vectors failed: {e}")
            raise RuntimeError(f"Deleting KB vectors failed: {e}")
    
    async def hybrid_search(self, query: str, kb_id: str, top_k: int = 10, 
                           rerank: bool = True) -> List[SearchResult]:
        """
        Advanced hybrid search combining semantic similarity with potential keyword matching.
        """
        try:
            logger.info(f"Performing hybrid search: {query[:100]}...")
            
            # Primary semantic search
            semantic_request = VectorSearchRequest(
                query=query,
                kb_id=kb_id,
                top_k=top_k * 2,  # Get more results for reranking
                score_threshold=0.3
            )
            
            semantic_results = await self.semantic_search(semantic_request)
            
            if not rerank or len(semantic_results.results) <= top_k:
                return semantic_results.results[:top_k]
            
            # Simple reranking based on text similarity and score
            reranked = await self._rerank_results(query, semantic_results.results)
            
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise RuntimeError(f"Hybrid search failed: {e}")
    
    async def _rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """
        Simple reranking based on keyword overlap and semantic score.
        """
        try:
            query_terms = set(query.lower().split())
            
            def calculate_combined_score(result: SearchResult) -> float:
                # Semantic score (0-1)
                semantic_score = result.score
                
                # Keyword overlap score
                text_terms = set(result.text.lower().split())
                overlap = len(query_terms.intersection(text_terms))
                keyword_score = overlap / max(len(query_terms), 1)
                
                # Combined score (weighted)
                combined = 0.7 * semantic_score + 0.3 * keyword_score
                return combined
            
            # Sort by combined score
            reranked = sorted(results, key=calculate_combined_score, reverse=True)
            
            logger.debug(f"Reranked {len(results)} results")
            return reranked
            
        except Exception as e:
            logger.warning(f"Reranking failed, returning original order: {e}")
            return results
    
    async def batch_similarity_search(self, texts: List[str], kb_id: str, 
                                    top_k_per_text: int = 5) -> Dict[str, List[SearchResult]]:
        """
        Perform similarity search for multiple texts efficiently.
        """
        try:
            logger.info(f"Performing batch similarity search for {len(texts)} texts")
            
            results = {}
            
            for i, text in enumerate(texts):
                request = SimilaritySearchRequest(
                    text=text,
                    kb_id=kb_id,
                    top_k=top_k_per_text,
                    min_score=0.5
                )
                
                similar_chunks = await self.find_similar_chunks(request)
                results[f"text_{i}"] = similar_chunks
                
                logger.debug(f"Processed text {i+1}/{len(texts)}")
            
            logger.info(f"Batch similarity search completed for {len(texts)} texts")
            return results
            
        except Exception as e:
            logger.error(f"Batch similarity search failed: {e}")
            raise RuntimeError(f"Batch similarity search failed: {e}")

# Global service instance
vector_service = VectorSearchService()