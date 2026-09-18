import os
import sys

from dotenv import load_dotenv
from groq import Groq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "w3")))
from vector_store import VectorStore

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

RAG_SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question based ONLY on the following context.
If the context does not contain enough information, say: "I cannot find enough information in the provided documents."

Context:
{context}

Question: {question}"""

def build_rag_promp(question: str, chunks: list) -> list[dict]:
    context_text = "\n\n".join(
        f"[{index + 1}] (Source: {chunk.metadata.get('title', 'doc')}) {chunk.content}"
        for index, chunk in enumerate(chunks)
    )

    return [
        {
            "role": "system",
            "content": RAG_SYSTEM_PROMPT.format(
                context=context_text,
                question=question,
            ),
        },
        {"role": "user", "content": question},
    ]

def rag_pipeline(query: str, store: VectorStore, top_k: int = 3) -> tuple[str, list]:
    results = store.search(query, top_k=top_k)
    if not results:
        return "I cannot find enough information in the provided documents.", []

    chunks = [chunk for chunk, _score in results]
    messages = build_rag_promp(query, chunks)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
    )

    answer = response.choices[0].message.content or ""
    return answer, results