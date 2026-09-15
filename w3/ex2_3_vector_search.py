from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===== Reuse Document, DocumentChunk from Exercise 1.2 =====

@dataclass
class Document:
    content: str
    doc_id: str = ""
    title: str = ""
    source: str = ""
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class DocumentChunk:
    content: str
    chunk_id: str = ""
    doc_id: str = ""
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None

# ===== DocumentStore (from Session 1) =====

@dataclass
class DocumentStore:
    documents: dict[str, Document] = field(default_factory=dict)
    chunks: dict[str, DocumentChunk] = field(default_factory=dict)

    def add(self, document: Document) -> str:
        if not document.doc_id:
            document.doc_id = f"doc_{len(self.documents) + 1:04d}"
        self.documents[document.doc_id] = document
        chunks = chunk_document(document)
        for chunk in chunks:
            self.add_chunk(chunk)
        return document.doc_id
    
    def get(self, doc_id: str) -> Optional[Document]:
        return self.documents.get(doc_id)

    def delete(self, doc_id: str) -> Optional[Document]:
        if doc_id in self.documents:
            del self.documents[doc_id]
        chunk_ids = [cid for cid, c in self.chunks.items() if c.doc_id == doc_id]
        for cid in chunk_ids:
            del self.chunks[cid]
            return True
        return False

    def list_all(self) -> list[Document]:
        return list(self.documents.values())
    
    def add_chunk(self, chunk: DocumentChunk) -> str:
        if not chunk.chunk_id:
            chunk.chunk_id = f"chunk_{len(self.chunks) + 1:04d}"
        self.chunks[chunk.chunk_id] = chunk
        return chunk.chunk_id
    
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        return self.chunks[chunk_id]
    
    def get_chunks_by_doc(self, doc_id: str) -> list[DocumentChunk]:
        return [c for c in self.chunks.values() if c.doc_id == doc_id]
    
    def count(self) -> dict:
        return {"documents": len(self.documents), "chunks": len(self.chunks)}
    
    
def chunk_document(doc: Document, chunk_size: int = 150, overlap: int = 30) -> list[DocumentChunk]:
    chunks = []
    text = doc.content
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = DocumentChunk(
            content=text[start:end],
            chunk_id=f"{doc.doc_id}_chunk_{len(chunks)}",
            doc_id=doc.doc_id,
            chunk_index=len(chunks),
            metadata={**doc.metadata, "chunk_size": chunk_size}
        )
        chunks.append(chunk)
        start += chunk_size - overlap
        if start >= len(text):
            break
    return chunks

# ===== Embedding helpers =====

def embed_text(text: str) -> list[float]:
    """Embed a single text."""
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    if np.linalg.norm(a_arr) == 0 or np.linalg.norm(b_arr) == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

# ===== VectorStore (your task) =====

@dataclass
class VectorStore(DocumentStore):
    """A DocumentStore with embedding-based semantic search."""

    def add_chunk(self, chunk: DocumentChunk) -> str:
        """
        Add a chunk and auto-generate its embedding.
        If chunk.embedding is already set, skip embedding.
        """
        # Your code here


    def search(self, query: str, top_k: int = 3) -> list[tuple[DocumentChunk, float]]:
        """
        Search for the most relevant chunks to a query.

        Returns list of (chunk, similarity_score) sorted by score descending.
        """
        # Your code here


    def search_with_threshold(
        self, query: str, top_k: int = 3, min_score: float = 0.5
    ) -> list[tuple[DocumentChunk, float]]:
        """Search with a minimum similarity threshold."""
        # Your code here



# ===== Main =====
def main():
    store = VectorStore()

    # Add 3 documents (from Session 1)
    doc1 = Document(
        content="Python is a high-level programming language. It was created by Guido van Rossum in 1991. Python emphasizes code readability. It is widely used in AI and data science.",
        title="Introduction to Python",
        source="course_materials"
    )
    doc2 = Document(
        content="RAG stands for Retrieval-Augmented Generation. It combines a retrieval system with a generative model. The retriever finds relevant documents. The generator produces an answer based on those documents.",
        title="What is RAG?",
        source="course_materials"
    )
    doc3 = Document(
        content="Embeddings are numerical representations of text. Similar texts have similar embeddings. Cosine similarity measures how similar two embeddings are. Values range from -1 to 1.",
        title="Understanding Embeddings",
        source="course_materials"
    )

    for doc in [doc1, doc2, doc3]:
        store.add(doc)
        print(f"Added: {doc.title}")

    print(f"\nStore: {store.count()}")

    # Test search
    test_queries = [
        "What is RAG?",
        "Tell me about Python",
        "How do embeddings work?",
        "What is cosine similarity?",
    ]

    for query in test_queries:
        print(f"\n=== Query: '{query}' ===")
        results = store.search(query, top_k=2)
        for chunk, score in results:
            print(f"  [{score:.4f}] {chunk.content[:80]}...")

    # Test threshold
    print("\n=== Search with threshold (min_score=0.7) ===")
    results = store.search_with_threshold("What is RAG?", top_k=5, min_score=0.7)
    print(f"Results with score >= 0.7: {len(results)}")
    for chunk, score in results:
        print(f"  [{score:.4f}] {chunk.content[:60]}...")

if __name__ == "__main__":
    main()