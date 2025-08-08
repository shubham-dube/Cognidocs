# Include routers
# app.include_router(chat.router, prefix="api/v1/chat", tags=["Chat APIs"])

# main.py
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import time
import uuid

from core.config import settings
from core.db import connect_to_mongo, close_mongo_connection, check_database_health, get_db_stats
from api.knowledge_bases import router as kb_router

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{settings.LOG_DIR}/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Application lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # Startup
    logger.info("Starting Knowledge Base API application...")
    
    try:
        # Connect to MongoDB
        await connect_to_mongo()
        logger.info("Database connection established")
        
        # You can add other startup tasks here
        # - Initialize Pinecone connection
        # - Warm up ML models
        # - Setup background task queues
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Knowledge Base API application...")
    
    try:
        await close_mongo_connection()
        logger.info("Database connection closed")
        
        # Add other cleanup tasks here
        
        logger.info("Application shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Create FastAPI application
app = FastAPI(
    title="Knowledge Base API",
    description="Advanced Knowledge Base Management System with Vector Search",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware for security
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure this properly for production
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing and correlation ID."""
    correlation_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Add correlation ID to request state
    request.state.correlation_id = correlation_id
    
    logger.info(
        f"Request started - {correlation_id} - {request.method} {request.url.path} - "
        f"Client: {request.client.host if request.client else 'unknown'}"
    )
    
    try:
        response = await call_next(request)
        
        process_time = time.time() - start_time
        logger.info(
            f"Request completed - {correlation_id} - Status: {response.status_code} - "
            f"Duration: {process_time:.3f}s"
        )
        
        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Request failed - {correlation_id} - Error: {str(e)} - "
            f"Duration: {process_time:.3f}s"
        )
        raise

# Global exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    logger.error(f"Validation error - {correlation_id}: {exc.errors()}")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "The request contains invalid data",
            "details": exc.errors(),
            "correlation_id": correlation_id
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "message": exc.detail,
            "correlation_id": correlation_id
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    logger.error(f"Unexpected error - {correlation_id}: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "correlation_id": correlation_id
        }
    )

# Health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0"
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check including database connectivity."""
    health_data = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0",
        "components": {}
    }
    
    # Check database health
    try:
        db_healthy, db_message = await check_database_health()
        health_data["components"]["database"] = {
            "status": "healthy" if db_healthy else "unhealthy",
            "message": db_message
        }
    except Exception as e:
        health_data["components"]["database"] = {
            "status": "unhealthy",
            "message": str(e)
        }
    
    # Check if any component is unhealthy
    if any(comp["status"] == "unhealthy" for comp in health_data["components"].values()):
        health_data["status"] = "unhealthy"
    
    return health_data

@app.get("/stats")
async def get_system_stats():
    """Get system statistics for monitoring."""
    try:
        stats = {
            "timestamp": time.time(),
            "database": await get_db_stats(),
            "system": {
                "upload_dir": settings.UPLOAD_DIR,
                "max_file_size": settings.MAX_FILE_SIZE,
                "max_files_per_request": settings.MAX_FILES_PER_REQUEST,
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system statistics")

# Include routers
app.include_router(
    kb_router,
    prefix=f"/api/v1/knowledge-bases",
    tags=["Knowledge Bases"]
)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Knowledge Base Management API",
        "version": "2.0.0",
        "description": "Advanced Knowledge Base Management System with Vector Search",
        "docs": "/docs",
        "health": "/health",
        "stats": "/stats"
    }

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )
