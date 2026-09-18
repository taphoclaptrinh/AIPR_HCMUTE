from rank_bm25 import BM25Okapi
from rich import print

corpus = [
    "The eiffel Tower is located in Paris, France",
    "Python is a popular programming language for data science",
    "BBM25 is a ranking function used in information retrieval",
    "RAG combines retrieval with language model generation",
    "The Greatest Wall of China is a historic fortification",
    "Vector databases store embeddings for similarity search",
    "Cross-encoders rerank documents by scoring query-doc pairs jointly",
    "Paris is the capital of France and a major European city."
]

query = "Historical tall structure"

tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

scores = bm25.get_scores(query.lower().split())
top_k_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:5]

print("=== BM25 only no reranking ===")
for i in top_k_idx:
    print(f"scores[i]:.3f | {corpus[i]}")

reranker = CrossEncoder("cross-encoder/mms-marco-MiniLm-L-6-v2")
K = 6
top_k_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:K]
candidates = [corpus[i] for i in top_k_idx]

pairs = [(query, doc) for doc in candidates]
rerank_scores = reranker.predict(pairs)
reranked = sorted(
    zip(candidates, rerank_scores),
    key=lambda pair: pair[1],
    reverse =True
)
print("\n=== BM25 + Cross-Encoder Rerank ===")
for doc, s in reranked:
    print(f"{s:.3f} | {doc}")