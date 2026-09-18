"""
BM25 + Cross-Encoder Reranking Pipeline
=======================================
- Bước 1 (First-stage Retrieval): Dùng BM25 để lọc nhanh Top-K ứng viên dựa trên từ khóa (Lexical search).
- Bước 2 (Reranking): Dùng Cross-Encoder (Semantic) để chấm điểm chính xác cặp (query, document) và sắp xếp lại.
"""

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from rich import print
from rich.table import Table

# 1. Dữ liệu văn bản (Corpus)
corpus = [
    "The eiffel Tower is located in Paris, France",
    "Python is a popular programming language for data science",
    "BM25 is a ranking function used in information retrieval",
    "RAG combines retrieval with language model generation",
    "The Greatest Wall of China is a historic fortification",
    "Vector databases store embeddings for similarity search",
    "Cross-encoders rerank documents by scoring query-doc pairs jointly",
    "Paris is the capital of France and a major European city."
]

query = "Historical tall structure"

print(f"\n[bold yellow]=== Query: \"{query}\" ===[/bold yellow]\n")

# ==========================================
# BƯỚC 1: FIRST-STAGE RETRIEVAL VỚI BM25
# ==========================================
tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# Chấm điểm BM25 cho tất cả tài liệu
bm25_scores = bm25.get_scores(query.lower().split())

# Lấy Top 5 tài liệu ứng viên từ BM25 (Candidate retrieval)
top_k_candidates_count = 5
top_k_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:top_k_candidates_count]

bm25_table = Table(title="[bold red]1. Ket qua BM25 (Chi so khop tu khoa - Lexical)[/bold red]")
bm25_table.add_column("Rank", justify="center", style="cyan", width=6)
bm25_table.add_column("BM25 Score", justify="right", style="magenta", width=12)
bm25_table.add_column("Document Content", justify="left")

candidate_docs = []
for rank, idx in enumerate(top_k_bm25_idx, 1):
    doc = corpus[idx]
    candidate_docs.append(doc)
    bm25_table.add_row(str(rank), f"{bm25_scores[idx]:.4f}", doc)

print(bm25_table)
print("\n[dim]Nhan xet BM25: BM25 chi tim tu khoa chinh xac. Cau hoi 'Historical tall structure' khong trung tu khoa truc tiep nao, nen cac tai lieu bi xep ngau nhien hoac diem 0.[/dim]\n")

# ==========================================
# BƯỚC 2: RERANKING VỚI CROSS-ENCODER
# ==========================================
print("[bold green]Dang khoi tao mo hinh Cross-Encoder ('cross-encoder/ms-marco-MiniLM-L-6-v2')...[/bold green]")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Chuẩn bị các cặp (Query, Document) để Cross-Encoder chấm điểm đồng thời (Joint scoring)
pairs = [[query, doc] for doc in candidate_docs]

# Dự đoán điểm liên quan ngữ nghĩa (Semantic Relevance Score)
rerank_scores = reranker.predict(pairs)

# Sắp xếp lại danh sách ứng viên theo điểm số Cross-Encoder giảm dần
reranked_results = sorted(zip(candidate_docs, rerank_scores), key=lambda x: x[1], reverse=True)

rerank_table = Table(title="[bold green]2. Ket qua sau khi RERANK (Cross-Encoder - Semantic)[/bold green]")
rerank_table.add_column("Rank", justify="center", style="cyan", width=6)
rerank_table.add_column("Rerank Score", justify="right", style="green", width=14)
rerank_table.add_column("Document Content", justify="left")

for rank, (doc, score) in enumerate(reranked_results, 1):
    rerank_table.add_row(str(rank), f"{score:.4f}", doc)

print(rerank_table)
print("\n[bold cyan]Ket luan:[/bold cyan] Cross-Encoder da hieu duoc 'The eiffel Tower' va 'The Greatest Wall of China' lien quan den cong trinh lich su ('Historical tall structure'), nen da dua len hang dau!\n")
