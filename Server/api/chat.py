from fastapi import APIRouter, HTTPException
from models.chat import ChatQuery, ChatResponse
from services.chat_service import save_chat, get_chats_for_policy
from services.policy_service import get_all_policies
from core.rag_pipeline import run_rag_pipeline

router = APIRouter()

@router.post("/policies/{policy_id}/query", response_model=ChatResponse)
async def query_policy(policy_id: str, query: ChatQuery):
    result = await run_rag_pipeline(policy_id, query.query)
    chat_id = await save_chat(policy_id, query.query, result["answer"], result["sources"])
    return ChatResponse(chat_id=chat_id, answer=result["answer"], sources=result["sources"])


@router.get("/policies")
async def get_policies():
    return await get_all_policies()


@router.get("/policies/{policy_id}/chats")
async def get_policy_chats(policy_id: str):
    return await get_chats_for_policy(policy_id)
