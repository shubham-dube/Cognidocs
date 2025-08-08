# core/config.py
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

# Load .env file
load_dotenv()

class Settings(BaseSettings):
    # Project Configuration
    PROJECT_NAME: str = Field(
        default="CogniDocs",
        description="Project name"
    )
    API_V1_STR: str = Field(
        default="/api/v1",
        description="API version string"
    )
    
    # Database Configuration
    MONGO_URI: str = Field(
        default=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        description="MongoDB connection string"
    )
    MONGO_DB: str = Field(
        default=os.getenv("MONGO_DB", "cognidocs"),
        description="MongoDB database name"
    )
    
    # Pinecone Configuration
    PINECONE_API_KEY: str = Field(
        default=os.getenv("PINECONE_API_KEY"),
        description="Pinecone API key for vector storage"
    )
    PINECONE_ENVIRONMENT: str = Field(
        default=os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp"),
        description="Pinecone environment"
    )
    PINECONE_INDEX_NAME: str = Field(
        default=os.getenv("PINECONE_INDEX_NAME", "cognidocs"),
        description="Pinecone index name"
    )
    
    # AI Configuration
    GEMINI_API_KEY: str = Field(
        default=os.getenv("GEMINI_API_KEY"),
        description="Google Gemini API key for embeddings"
    )
    ANTHROPIC_API_KEY: str = Field(
        default=os.getenv("ANTHROPIC_API_KEY"),
        description="Anthropic Claude API key for chat"
    )
    GEMINI_EMBED_MODEL: str = Field(
        default="models/text-embedding-004",
        description="Gemini embedding model"
    )
    
    # File Storage Configuration
    UPLOAD_DIR: str = Field(
        default="uploads",
        description="Directory for temporary file uploads"
    )
    MAX_FILE_SIZE: int = Field(
        default=50 * 1024 * 1024,  # 50MB
        description="Maximum file size in bytes"
    )
    MAX_FILES_PER_REQUEST: int = Field(
        default=10,
        description="Maximum files per upload request"
    )
    
    # Processing Configuration
    CHUNK_SIZE: int = Field(
        default=500,
        description="Text chunk size for embeddings"
    )
    CHUNK_OVERLAP: int = Field(
        default=50,
        description="Text chunk overlap"
    )
    EMBEDDING_BATCH_SIZE: int = Field(
        default=100,
        description="Batch size for Pinecone upserts"
    )
    
    # Logging Configuration
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )
    LOG_DIR: str = Field(
        default="logs",
        description="Directory for log files"
    )
    
    # API Configuration
    API_V1_PREFIX: str = Field(
        default="/api/v1",
        description="API version prefix"
    )
    CORS_ORIGINS: list = Field(
        default=["*"],
        description="CORS allowed origins"
    )
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(
        default=60,
        description="API rate limit per minute"
    )
    
    # Background Task Configuration
    MAX_CONCURRENT_INGESTIONS: int = Field(
        default=3,
        description="Maximum concurrent ingestion tasks"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings()

# Ensure required directories exist
def ensure_directories():
    """Create required directories if they don't exist."""
    import os
    directories = [
        settings.UPLOAD_DIR,
        settings.LOG_DIR,
        os.path.join(settings.UPLOAD_DIR, "temp"),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory exists: {directory}")

# Initialize directories on import
ensure_directories()