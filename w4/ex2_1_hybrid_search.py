import os
import sys
from rank_bm25 import BM25Okapi
import numpy as np

# Đảm bảo import được vector_store dù chạy script từ thư mục nào
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "w3")))

from vector_store import VectorStore, Document, DocumentChunk

def normalise_scores(scores: list[float]) -> list[float]:
    """Normalise scores to [0, 1] using min-max scaling."""
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)  # All equal → all 1.0
    return [(s - min_s) / (max_s - min_s) for s in scores]

# ===== Vector Embedding & Similarity Helpers =====
_vector_store = VectorStore()

def _embed_texts(text_or_texts: str | list[str]) -> list[float] | list[list[float]]:
    """Helper: embed a single string or list of strings using Jina AI API via VectorStore."""
    if isinstance(text_or_texts, str):
        return _vector_store._embed_texts([text_or_texts])[0]
    return _vector_store._embed_texts(text_or_texts)

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two embedding vectors."""
    return VectorStore._cosine_similarity(a, b)

# ===== Step 1: Build a BM25 index over the chunks =====
def build_bm25_index(chunks: list) -> BM25Okapi:
    """Build a BM25 index from chunk content (tokenized by whitespace)."""
    tokenized = [chunk.content.lower().split() for chunk in chunks]
    return BM25Okapi(tokenized)

# ===== Step 2: Hybrid search =====
def hybrid_search(
    query:  str,
    chunks: list,
    bm25_index: BM25Okapi,
    embeddings: list[list[float]],
    top_k:  int = 3,
    w_bm25: float = 0.3,
    w_vector: float = 0.7
) -> list[tuple]:
    """
    Hybrid search: combine BM25 (keyword) and vector (semantic) scores.
    Returns: list of (chunk_index, hybrid_score) tuples sorted descending.
    """
    # BM25 scores (raw, unbounded)
    tokenized_query = query.lower().split()
    print(f"Tokenized query: {tokenized_query}")
    bm25_scores = bm25_index.get_scores(tokenized_query)
    print(f"BM25 scores: {bm25_scores}")

    # Vector scores (cosine similarity, bounded)
    query_vec = _embed_texts(query)
    vector_scores = [
        cosine_similarity(query_vec, e) if e is not None else 0.0 
        for e in embeddings
    ]

    # Normalise BM25 to [0, 1]
    bm25_norm = normalise_scores(list(bm25_scores))

    # Fuse
    hybrid = [
        w_bm25 * b + w_vector * v
        for b, v in zip(bm25_norm, vector_scores)
    ]

    # Sort by hybrid score descending, return top_k
    ranked = sorted(
        enumerate(hybrid), key=lambda x: x[1], reverse=True
    )[:top_k]

    return ranked  # list of (chunk_index, score)

# ===== Test =====
if __name__ == "__main__":
    # Đảm bảo console Windows in tiếng Việt/UTF-8 không bị lỗi
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Build a small test store
    store = VectorStore()

    # Tìm đường dẫn file company_policy.txt bất kể script được gọi từ đâu
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "company_policy.txt"),
        os.path.join(os.path.dirname(__file__), "..", "data", "company_policy.txt"),
        os.path.join("data", "company_policy.txt"),
        "company_policy.txt",
    ]
    policy_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if not policy_path:
        raise FileNotFoundError("Không tìm thấy file company_policy.txt")

    print(f"Đang tải tài liệu từ: {policy_path}")
    with open(policy_path, "r", encoding="utf-8") as f:
        content = f.read()
        doc = Document(content=content, metadata={"title": "Company Policy", "source": policy_path})
        store.add(doc)

    test_questions = [
        "What is the remote work policy?",
        "How many vacation days do employees get?",
        "What can I claim for travel?",
    ]

    chunks = list(store.chunks.values())
    bm25_index = build_bm25_index(chunks)
    embeddings = [chunk.embedding for chunk in chunks]

    for question in test_questions:
        results = hybrid_search(
            question,
            chunks,
            bm25_index,
            embeddings,
            top_k=3,
        )

        print(f"\n=== Query: {question} ===")
        for chunk_index, score in results:
            chunk = chunks[chunk_index]
            print(f"  [{score:.4f}] {chunk.chunk_id}")
            print(f"    {chunk.content[:300].replace(chr(10), ' ')}...")
