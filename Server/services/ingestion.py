# app/services/ingestion.py
import os
import uuid
import traceback
import asyncio
import logging
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import threading

from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
import fitz  # PyMuPDF
import docx
import chardet
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId
from core.config import settings
from core.db import get_db

# ----------------- Logging Setup -----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ingestion.log'),
        logging.StreamHandler()
    ]
)

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

logger = logging.getLogger(__name__)

# ----------------- Init Clients -----------------
try:
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)

    # FIXED: Corrected typo in index name variable
    # Check what embedding model you're using and its dimensions
    if settings.PINECONE_INDEX_NAME not in pc.list_indexes().names():
        # CRITICAL FIX: Match dimension with your embedding model
        # Gemini's text-embedding-004 produces 768 dimensions
        # If you need 1024, use a different model or recreate the index
        pc.create_index(
            name=settings.PINECONE_INDEX_NAME,  # Fixed typo from PINECONE_INDEX_NAM
            dimension=768,  # CHANGED: Match your embedding model's dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1" 
            )
        )
    pc_index = pc.Index(settings.PINECONE_INDEX_NAME)
    logger.info("Pinecone client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")
    raise

try:
    genai.configure(api_key=settings.GEMINI_API_KEY)
    logger.info("Gemini AI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini AI: {e}")
    raise

# ----------------- Status Enums -----------------
class ProcessingStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    PARTIALLY_COMPLETED = "partially_completed"

# ----------------- Utilities -----------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for better embedding context.
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for chunking")
        return []
    
    chunks = []
    text = text.strip()
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        start += chunk_size - overlap
    
    logger.info(f"Text chunked into {len(chunks)} chunks")
    return chunks

# ----------------- File Parsers -----------------
def parse_pdf(path: str) -> str:
    """Parse PDF file and extract text."""
    try:
        logger.info(f"Parsing PDF: {path}")
        text = ""
        with fitz.open(path) as pdf:
            total_pages = len(pdf)
            logger.info(f"PDF has {total_pages} pages")
            
            for page_num, page in enumerate(pdf, 1):
                page_text = page.get_text()
                text += page_text + "\n"
                logger.debug(f"Extracted text from page {page_num}/{total_pages}")
        
        logger.info(f"Successfully parsed PDF: {path}, extracted {len(text)} characters")
        return text.strip()
    except Exception as e:
        logger.error(f"Error parsing PDF {path}: {e}")
        raise RuntimeError(f"Error parsing PDF: {e}")

def parse_docx(path: str) -> str:
    """Parse DOCX file and extract text."""
    try:
        logger.info(f"Parsing DOCX: {path}")
        doc = docx.Document(path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        logger.info(f"Successfully parsed DOCX: {path}, extracted {len(text)} characters")
        return text
    except Exception as e:
        logger.error(f"Error parsing DOCX {path}: {e}")
        raise RuntimeError(f"Error parsing DOCX: {e}")

def parse_txt(path: str) -> str:
    """Parse text file with encoding detection."""
    try:
        logger.info(f"Parsing TXT: {path}")
        with open(path, "rb") as f:
            raw = f.read()
        
        encoding = chardet.detect(raw).get("encoding") or "utf-8"
        logger.info(f"Detected encoding: {encoding} for file: {path}")
        
        text = raw.decode(encoding, errors="ignore")
        logger.info(f"Successfully parsed TXT: {path}, extracted {len(text)} characters")
        return text
    except Exception as e:
        logger.error(f"Error parsing TXT {path}: {e}")
        raise RuntimeError(f"Error parsing TXT: {e}")

def extract_text(path: str) -> str:
    """Extract text from file based on extension."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    ext = os.path.splitext(path)[1].lower()
    logger.info(f"Extracting text from {path} (type: {ext})")
    
    try:
        if ext == ".pdf":
            return parse_pdf(path)
        elif ext in [".docx", ".doc"]:
            return parse_docx(path)
        elif ext in [".txt", ".md"]:
            return parse_txt(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    except Exception as e:
        logger.error(f"Failed to extract text from {path}: {e}")
        raise

# ----------------- Embedding -----------------
async def embed_texts(chunks: List[str]) -> List[List[float]]:
    """
    Get embeddings for text chunks using Gemini's embed_content API with proper error handling.
    """
    if not chunks:
        logger.warning("No chunks provided for embedding")
        return []
    
    logger.info(f"Generating embeddings for {len(chunks)} chunks")
    embeddings = []
    failed_chunks = 0
    
    # Process in smaller batches to avoid rate limits
    batch_size = 5  # Reduced batch size further
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(chunks))
        batch_chunks = chunks[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_num + 1}/{total_batches} (chunks {start_idx}-{end_idx-1})")
        
        for i, chunk in enumerate(batch_chunks):
            chunk_idx = start_idx + i
            
            try:
                if not chunk.strip():
                    logger.warning(f"Skipping empty chunk at index {chunk_idx}")
                    continue
                
                # Add timeout and retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        logger.debug(f"Generating embedding for chunk {chunk_idx + 1}/{len(chunks)} (attempt {attempt + 1})")
                        
                        # FIXED: Use proper synchronous call with await
                        resp = await asyncio.to_thread(
                            genai.embed_content,
                            model=settings.GEMINI_EMBED_MODEL,
                            content=chunk,
                            task_type="retrieval_document"  # Added task type for better embeddings
                        )
                        
                        embedding = resp["embedding"]
                        
                        # Log embedding dimension for debugging
                        logger.debug(f"Embedding dimension: {len(embedding)}")
                        
                        embeddings.append(embedding)
                        logger.debug(f"Successfully generated embedding for chunk {chunk_idx + 1}")
                        break  # Success, exit retry loop
                        
                    except Exception as api_error:
                        logger.warning(f"API error on chunk {chunk_idx} attempt {attempt + 1}: {api_error}")
                        if attempt == max_retries - 1:
                            raise
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
                # Rate limiting delay between chunks
                await asyncio.sleep(0.5)  # Increased delay
                
            except Exception as e:
                failed_chunks += 1
                logger.error(f"Failed to generate embedding for chunk {chunk_idx} after {max_retries} attempts: {e}")
                
                # Fail fast if too many errors
                if failed_chunks > len(chunks) * 0.2:  # If more than 20% fail
                    raise RuntimeError(f"Too many embedding failures: {failed_chunks}/{chunk_idx + 1}")
                continue
        
        # Longer delay between batches to avoid rate limits
        if batch_num < total_batches - 1:
            logger.info(f"Completed batch {batch_num + 1}, waiting before next batch...")
            await asyncio.sleep(2)  # Reduced wait time
    
    success_rate = (len(embeddings) / len(chunks)) * 100
    logger.info(f"Generated {len(embeddings)}/{len(chunks)} embeddings ({success_rate:.1f}% success rate)")
    
    if len(embeddings) == 0:
        raise RuntimeError("No embeddings were generated successfully")
    
    return embeddings

# ----------------- Pinecone Store -----------------
async def store_embeddings(
    kb_id: str,
    chunks: List[str],
    embeddings: List[List[float]],
    metadata: Dict
):
    """Store embeddings in Pinecone with metadata."""
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings count mismatch")
    
    logger.info(f"Storing {len(embeddings)} embeddings for KB: {kb_id}")
    
    # Log embedding dimensions for debugging
    if embeddings:
        logger.info(f"Embedding dimension: {len(embeddings[0])}")
    
    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vector_id = str(uuid.uuid4())
        vector_metadata = {
            "kb_id": kb_id,
            "chunk_index": i,
            "text": chunk[:1000],  # Limit text size in metadata
            **metadata
        }
        vectors.append((vector_id, emb, vector_metadata))
    
    try:
        # Upsert in smaller batches to avoid size limits
        batch_size = 50  # Reduced batch size
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            
            # FIXED: Proper upsert format
            upsert_data = [{"id": vid, "values": emb, "metadata": meta} for vid, emb, meta in batch]
            pc_index.upsert(vectors=upsert_data)
            
            logger.info(f"Upserted batch {i//batch_size + 1}, vectors {i+1}-{min(i+batch_size, len(vectors))}")
            await asyncio.sleep(1)  # Rate limiting
        
        logger.info(f"Successfully stored all {len(vectors)} vectors for KB: {kb_id}")
    except Exception as e:
        logger.error(f"Pinecone upsert failed for KB {kb_id}: {e}")
        raise RuntimeError(f"Pinecone upsert failed: {e}")

# ----------------- Status Updates -----------------
async def update_kb_status(
    db: AsyncIOMotorDatabase,
    kb_id: str,
    status: str,
    processed_files: int = 0,
    total_files: int = 0,
    current_file: Optional[str] = None,
    error_message: Optional[str] = None,
    completed_files: Optional[List[str]] = None,
    failed_files: Optional[List[str]] = None
):
    """Update knowledge base processing status."""
    try:
        percentage = 0
        if total_files > 0:
            percentage = (processed_files / total_files) * 100
        
        update_doc = {
            "status": status,
            "last_updated": datetime.utcnow(),
            "processing_info": {
                "processed_files": processed_files,
                "total_files": total_files,
                "percentage": round(percentage, 2),
                "current_file": current_file,
                "completed_files": completed_files or [],
                "failed_files": failed_files or []
            }
        }
        
        if error_message:
            update_doc["last_error"] = error_message
        else:
            update_doc["last_error"] = None
        
        await db.kbs.update_one(
            {"_id": ObjectId(kb_id)},
            {"$set": update_doc}
        )
        
        logger.info(f"Updated KB {kb_id} status: {status} ({percentage:.1f}%)")
    except Exception as e:
        logger.error(f"Failed to update KB status: {e}")

# ----------------- Main Ingestion Task -----------------
async def process_file_for_kb(
    kb_id: str,
    file_path: str,
    db: AsyncIOMotorDatabase
) -> bool:
    """Process a single file for the knowledge base."""
    filename = os.path.basename(file_path)
    logger.info(f"Processing file {filename} for KB {kb_id}")
    
    try:
        # Extract text
        text = extract_text(file_path)
        if not text or not text.strip():
            logger.warning(f"No text extracted from {filename}")
            return False
        
        # Chunk text
        chunks = chunk_text(text)
        if not chunks:
            logger.warning(f"No chunks created from {filename}")
            return False
        
        # Generate embeddings
        embeddings = await embed_texts(chunks)
        if not embeddings:
            logger.error(f"No embeddings generated for {filename}")
            return False
        
        # Store in Pinecone
        await store_embeddings(
            kb_id,
            chunks,
            embeddings,
            metadata={
                "source_file": filename,
                "ingested_at": datetime.utcnow().isoformat(),
                "file_path": file_path
            }
        )
        
        logger.info(f"Successfully processed file {filename} for KB {kb_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process file {filename}: {e}")
        return False

async def start_ingestion_task(kb_id: str, file_paths: List[str], metadata: Dict = None):
    """
    Orchestrates ingestion pipeline for uploaded files with comprehensive status tracking.
    """
    logger.info(f"Starting ingestion task for KB {kb_id} with {len(file_paths)} files")
    
    db: AsyncIOMotorDatabase = get_db()
    
    # Initialize tracking variables
    total_files = len(file_paths)
    processed_files = 0
    completed_files = []
    failed_files = []
    
    try:
        # Set initial status
        await update_kb_status(
            db, kb_id, ProcessingStatus.PROCESSING,
            processed_files=0, total_files=total_files,
            current_file="Initializing..."
        )
        
        # Process each file
        for i, file_path in enumerate(file_paths):
            filename = os.path.basename(file_path)
            
            try:
                # Update current processing status
                await update_kb_status(
                    db, kb_id, ProcessingStatus.PROCESSING,
                    processed_files=processed_files, total_files=total_files,
                    current_file=filename,
                    completed_files=completed_files,
                    failed_files=failed_files
                )
                
                # Process the file
                success = await process_file_for_kb(kb_id, file_path, db)
                
                if success:
                    completed_files.append(filename)
                    logger.info(f"Successfully processed {filename}")
                else:
                    failed_files.append(filename)
                    logger.error(f"Failed to process {filename}")
                
                processed_files += 1
                
                # Cleanup processed file
                try:
                    os.remove(file_path)
                    logger.info(f"Cleaned up temporary file: {file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup file {file_path}: {cleanup_error}")
                
            except Exception as file_error:
                failed_files.append(filename)
                processed_files += 1
                logger.error(f"Error processing file {filename}: {file_error}")
                continue
        
        # Determine final status
        if not failed_files:
            final_status = ProcessingStatus.COMPLETED
            logger.info(f"All files processed successfully for KB {kb_id}")
        elif not completed_files:
            final_status = ProcessingStatus.ERROR
            logger.error(f"All files failed for KB {kb_id}")
        else:
            final_status = ProcessingStatus.PARTIALLY_COMPLETED
            logger.warning(f"Partially completed for KB {kb_id}: {len(completed_files)} succeeded, {len(failed_files)} failed")
        
        # Final status update
        await update_kb_status(
            db, kb_id, final_status,
            processed_files=processed_files, total_files=total_files,
            current_file=None,
            completed_files=completed_files,
            failed_files=failed_files
        )
        
        logger.info(f"Ingestion task completed for KB {kb_id}")
        
    except Exception as e:
        error_message = f"Ingestion pipeline failed: {str(e)}"
        logger.error(f"Critical error in ingestion task for KB {kb_id}: {e}")
        logger.error(traceback.format_exc())
        
        # Update with error status
        await update_kb_status(
            db, kb_id, ProcessingStatus.ERROR,
            processed_files=processed_files, total_files=total_files,
            error_message=error_message,
            completed_files=completed_files,
            failed_files=failed_files
        )
        
    finally:
        # Cleanup any remaining temporary files
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Cleaned up remaining file: {file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup remaining file {file_path}: {cleanup_error}")

# ----------------- FIXED: Event Loop Management -----------------
def run_ingestion_task(kb_id: str, file_paths: List[str], metadata: Dict = None):
    """
    FIXED: Proper synchronous wrapper for the async ingestion task.
    Uses threading to avoid event loop conflicts.
    """
    logger.info(f"Starting sync wrapper for ingestion task: KB {kb_id}")
    
    def run_in_thread():
        """Run the async task in a separate thread with its own event loop."""
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                loop.run_until_complete(start_ingestion_task(kb_id, file_paths, metadata))
                logger.info(f"Ingestion task completed successfully for KB {kb_id}")
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Error in ingestion thread for KB {kb_id}: {e}")
            logger.error(traceback.format_exc())
            
            # Update status to error using a new event loop
            try:
                error_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(error_loop)
                try:
                    error_loop.run_until_complete(update_error_status(kb_id, str(e)))
                finally:
                    error_loop.close()
            except Exception as status_error:
                logger.error(f"Failed to update error status: {status_error}")
    
    # Run in a separate thread to avoid event loop conflicts
    thread = threading.Thread(target=run_in_thread)
    thread.daemon = True  # Allow main process to exit even if thread is running
    thread.start()
    
    logger.info(f"Ingestion task started in background thread for KB {kb_id}")

async def update_error_status(kb_id: str, error_message: str):
    """Helper function to update KB status to error."""
    try:
        db = get_db()
        await update_kb_status(
            db, kb_id, ProcessingStatus.ERROR,
            error_message=f"Ingestion failed: {error_message}"
        )
        logger.info(f"Updated KB {kb_id} status to ERROR")
    except Exception as e:
        logger.error(f"Failed to update error status for KB {kb_id}: {e}")