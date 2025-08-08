import os
from dotenv import load_dotenv
from pydantic import BaseSettings

# Load .env file
load_dotenv()

class Settings(BaseSettings):
    # MongoDB
    MONGO_URI: str = os.getenv("MONGO_URI")
    MONGO_DB: str = os.getenv("MONGO_DB", "cognidocs")

    # Pinecone
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "docmind-index")

    # Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")

    # Claude
    CLAUDE_API_KEY: str = os.getenv("CLAUDE_API_KEY")

    # Project settings
    PROJECT_NAME: str = "CogniDocs"
    API_V1_STR: str = "/api/v1"

settings = Settings()