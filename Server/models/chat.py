from pydantic import BaseModel
from typing import List, Optional

class ChatQuery(BaseModel):
    query: str

class ChatResponse(BaseModel):
    chat_id: str
    answer: str
    sources: Optional[List[str]] = []
