from fastapi import FastAPI
from core.config import settings
from fastapi.middleware.cors import CORSMiddleware

# Import routers (will create them later)
from api import chat

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0"
)

# CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="api/v1/chat", tags=["Chat APIs"])

@app.get("/")
def read_root():
    return {"message": f"Welcome to {settings.PROJECT_NAME} API"}