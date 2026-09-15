import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from typing import TypeAlias

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Type alias — makes the nested type readable
Candidate: TypeAlias = tuple[str, list[float]]  # (text, embedding vector)
Result: TypeAlias = tuple[str, float]           # (text, similarity score)

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts."""
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [data.embedding for data in response.data]

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    if np.linalg.norm(a_arr) == 0 or np.linalg.norm(b_arr) == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

def find_most_similar(
    query: str,
    candidates: list[Candidate],
    top_k: int = 3
) -> list[Result]:
    """
    Find the top_k most similar items to a query.
    """
    query_vec = embed_texts([query])[0]

    results = []
    for text, vec in candidates:
        score = cosine_similarity(query_vec, vec)
        results.append((text, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

# === Test with sample data ===
sentences = [
    "RAG is a technique for combining retrieval with generation",
    "Python is a high-level programming language",
    "The capital of France is Paris",
    "Embeddings map text to numerical vectors",
    "Cosine similarity measures the angle between vectors",
    "I love hiking and exploring nature trails",
]

print("Building embedding index...")
candidates = [(s, vec) for s, vec in zip(sentences, embed_texts(sentences))]

# Test queries
test_queries = [
    "What is RAG?",
    "Tell me about vector similarity",
    "What is the capital of France?",
]

for query in test_queries:
    print(f"\n=== Query: '{query}' ===")
    results = find_most_similar(query, candidates, top_k=2)
    for text, score in results:
        print(f"  [{score:.4f}] {text}")
