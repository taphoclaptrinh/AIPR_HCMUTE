import argparse
import os
import sys
from pathlib import Path
from rank_bm25 import BM25Okapi
from rich import print

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import CrossEncoder

sys.path.append(r"../w3")

from vector_store import Document, VectorStore  # type: ignore[reportMissingImports]
from ex2_1_hybrid_search import build_bm25_index, hybrid_search


load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

RAG_SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question based ONLY on the following context.
If the context does not contain enough information, say: "I cannot find enough information in the provided documents."

Context:
{context}

Question: {question}"""


def build_rag_prompt(question: str, chunks: list) -> list[dict]:
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


def search_chunks(query: str, store: VectorStore, top_k: int, search_mode: str) -> list[tuple]:
    """Retrieve chunks with either cosine similarity or hybrid search."""
    if search_mode == "cosine":
        return store.search(query, top_k=top_k)

    chunks = list(store.chunks.values())
    bm25_index = build_bm25_index(chunks)
    embeddings = [chunk.embedding for chunk in chunks]
    return [
        (chunks[chunk_index], score)
        for chunk_index, score in hybrid_search(
            query,
            chunks,
            bm25_index,
            embeddings,
            top_k=top_k,
        )
    ]


def rag_pipeline(
    query: str,
    store: VectorStore,
    top_k: int = 3,
    search_mode: str = "cosine",
) -> tuple[str, list]:
    """Retrieve context, ask the LLM, and return the answer and scored sources."""
    if search_mode not in {"cosine", "hybrid"}:
        raise ValueError("search_mode must be 'cosine' or 'hybrid'")

    results = search_chunks(query, store, top_k, search_mode)
    if not results:
        return "I cannot find enough information in the provided documents.", []

    results = rerank(query, results, top_k=top_k)

    messages = build_rag_prompt(query, [chunk for chunk, _score in results])
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=0.0,
    )
    return response.choices[0].message.content, results

def rerank(
    query: str,
    candidates: list[tuple],
    top_k: int = 3,
) -> list[tuple]:
    """Rerank retrieved chunks with a cross-encoder relevance model."""
    if not candidates:
        return []

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(query, chunk.content) for chunk, _score in candidates]
    rerank_scores = reranker.predict(pairs)

    reranked = [
        (chunk, float(score))
        for (chunk, _old_score), score in zip(candidates, rerank_scores)
    ]
    return sorted(reranked, key=lambda item: item[1], reverse=True)[:top_k]

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG pipeline.")
    parser.add_argument(
        "--search-mode",
        choices=("cosine", "hybrid"),
        default="cosine",
        help="Retrieval strategy to use (default: cosine).",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--question",
        default="For approved business travel, what is the meal reimbursement limit and the deadline for submitting receipts?",
    )
    args = parser.parse_args()

    store = VectorStore()
    policy_path = "data/company_policy.txt"
    with open(policy_path, encoding="utf-8") as policy_file:
        store.add(
            Document(
                content=policy_file.read(),
                metadata={"title": "Company Policy", "source": str(policy_path)},
            )
        )

    answer, sources = rag_pipeline(
        args.question,
        store,
        top_k=args.top_k,
        search_mode=args.search_mode,
    )
    print(f"Answer ({args.search_mode}): {answer}")
    print(f"Sources ({len(sources)}):")
    for chunk, score in sources:
        print(f"  [{score:.3f}] {chunk.content[:300]}...")

if __name__ == "__main__":
    main()