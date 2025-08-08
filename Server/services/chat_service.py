from datetime import datetime
from bson import ObjectId
from services.mongo import get_collection

async def save_chat(policy_id: str, query: str, answer: str, sources: list):
    chats = get_collection("chats")
    doc = {
        "policy_id": policy_id,
        "query": query,
        "answer": answer,
        "sources": sources,
        "timestamp": datetime.utcnow()
    }
    result = await chats.insert_one(doc)
    return str(result.inserted_id)

async def get_chats_for_policy(policy_id: str):
    chats = get_collection("chats")
    return await chats.find({"policy_id": policy_id}).sort("timestamp", -1)
