from core.vectorstore import search_similar_chunks
from services.policy_service import get_policy_by_id
from core.llm import ask_llm  # You can implement this later

async def run_rag_pipeline(policy_id: str, query: str):
    # Step 1: Get relevant chunks from vector DB
    context_chunks = await search_similar_chunks(policy_id, query)
    
    context_text = "\n".join([chunk["text"] for chunk in context_chunks])
    
    # Step 2: Construct prompt and ask LLM
    prompt = f"Use the following context to answer:\n\n{context_text}\n\nQuestion: {query}"
    answer = await ask_llm(prompt)
    
    sources = [chunk["source"] for chunk in context_chunks]
    
    return {"answer": answer, "sources": sources}
