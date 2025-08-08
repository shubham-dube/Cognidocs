# app/services/query.py
import traceback
from typing import List, Dict

import pinecone
import google.generativeai as genai
from core.config import settings
from core.db import get_db
from datetime import datetime

# ------------ Init Clients ------------
pinecone.init(
    api_key=settings.PINECONE_API_KEY,
    environment=settings.PINECONE_ENVIRONMENT
)
pc_index = pinecone.Index(settings.PINECONE_INDEX_NAME)

genai.configure(api_key=settings.GEMINI_API_KEY)


# ------------ Retrieve Chunks from Pinecone ------------
async def retrieve_context(kb_id: str, query: str, top_k: int = 5) -> List[Dict]:
    """
    Get top_k relevant chunks for a query from the KB using Pinecone.
    """
    try:
        # Embed query using the same model as ingestion
        q_embedding = genai.embed_content(
            model=settings.GEMINI_EMBED_MODEL,
            content=query
        )["embedding"]

        # Pinecone metadata filter
        results = pc_index.query(
            vector=q_embedding,
            top_k=top_k,
            include_metadata=True,
            filter={"kb_id": {"$eq": kb_id}}
        )

        return [
            {
                "text": match["metadata"]["text"],
                "source_file": match["metadata"].get("source_file", ""),
                "score": match["score"]
            }
            for match in results.get("matches", [])
        ]

    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Error retrieving context: {e}")


# ------------ Build Prompt for LLM ------------
def build_prompt(user_query: str, context_chunks: List[Dict]) -> str:
    context_text = "\n\n".join(
        [f"Source: {c['source_file']}\n{c['text']}" for c in context_chunks]
    )

    return f"""
You are a knowledgeable assistant. Answer the question based ONLY on the following context:

{context_text}

User Question: {user_query}

If the answer is not found in the context, say "I don't have enough information from the provided documents."
"""


# ------------ Ask LLM ------------
async def ask_llm(prompt: str) -> str:
    """
    Uses Gemini reasoning/chat model to answer based on prompt.
    """
    try:
        resp = genai.chat(
            model=settings.GEMINI_REASONING_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        # Ensure correct content access
        return resp.candidates[0].content[0].text if resp.candidates else ""
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Error calling LLM: {e}")


# ------------ Main Query API ------------
async def query_kb(kb_id: str, user_query: str) -> Dict:
    """
    Full retrieval → reasoning pipeline.
    """
    try:
        # Step 1: Get context
        context_chunks = await retrieve_context(kb_id, user_query)

        # Step 2: Build prompt
        prompt = build_prompt(user_query, context_chunks)

        # Step 3: Ask LLM
        answer = await ask_llm(prompt)

        # Step 4: Save chat history
        db = get_db()
        await db.chats.insert_one({
            "kb_id": kb_id,
            "query": user_query,
            "answer": answer,
            "context_used": context_chunks,
            "created_at": datetime.utcnow()
        })

        return {
            "answer": answer,
            "context": context_chunks
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
