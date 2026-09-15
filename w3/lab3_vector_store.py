from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import json
import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"


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


@dataclass
class DocumentStore:
    documents: dict[str, Document] = field(default_factory=dict)
    chunks: dict[str, DocumentChunk] = field(default_factory=dict)

    def add(self, document: Document) -> str:
        if not document.doc_id:
            document.doc_id = f"doc_{len(self.documents) + 1:04d}"
        self.documents[document.doc_id] = document
        chunks = self._chunk_document(document)
        for chunk in chunks:
            self.add_chunk(chunk)
        return document.doc_id

    def get(self, doc_id: str) -> Optional[Document]:
        return self.documents.get(doc_id)

    def delete(self, doc_id: str) -> bool:
        if doc_id not in self.documents:
            return False
        del self.documents[doc_id]
        chunk_ids = [cid for cid, c in self.chunks.items() if c.doc_id == doc_id]
        for cid in chunk_ids:
            del self.chunks[cid]
        return True

    def list_all(self) -> list[Document]:
        return list(self.documents.values())

    def add_chunk(self, chunk: DocumentChunk) -> str:
        if not chunk.chunk_id:
            chunk.chunk_id = f"chunk_{len(self.chunks) + 1:05d}"
        self.chunks[chunk.chunk_id] = chunk
        return chunk.chunk_id

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        return self.chunks.get(chunk_id)

    def get_chunks_by_doc(self, doc_id: str) -> list[DocumentChunk]:
        return [c for c in self.chunks.values() if c.doc_id == doc_id]

    def count(self) -> dict:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
        }

    def _chunk_document(self, doc: Document, chunk_size: int = 150, overlap: int = 30) -> list[DocumentChunk]:
        """Split a Document into overlapping word-based chunks."""
        words = doc.content.split()
        if not words:
            return []

        step = max(chunk_size - overlap, 1)
        chunks = []
        index = 0
        for start in range(0, len(words), step):
            chunk_words = words[start:start + chunk_size]
            if not chunk_words:
                continue
            chunk_text = " ".join(chunk_words)
            chunks.append(
                DocumentChunk(
                    content=chunk_text,
                    doc_id=doc.doc_id,
                    chunk_index=index,
                    metadata={"source": doc.source, "title": doc.title},
                )
            )
            index += 1
            if start + chunk_size >= len(words):
                break
        return chunks


# ===== VectorStore =====

@dataclass
class VectorStore(DocumentStore):
    """A DocumentStore with embedding-based semantic search."""

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Helper: call OpenAI embeddings API for a batch of texts."""
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        # response.data is ordered the same as input
        return [item.embedding for item in response.data]

    def add_chunk(self, chunk: DocumentChunk) -> str:
        """
        Add a chunk and auto-generate its embedding.
        If chunk.embedding is already set, skip embedding.
        """
        if chunk.embedding is None:
            embedding = self._embed_texts([chunk.content])[0]
            chunk.embedding = embedding

        return super().add_chunk(chunk)

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if denom == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / denom)

    def search(self, query: str, top_k: int = 3, min_score: float = 0.0) -> list[tuple[DocumentChunk, float]]:
        """
        Search for the most relevant chunks to a query.

        Args:
            query: The search query
            top_k: Number of results to return
            min_score: Minimum similarity score threshold

        Returns:
            List of (chunk, similarity_score) sorted by score descending
        """
        if not self.chunks:
            return []

        query_embedding = self._embed_texts([query])[0]

        scored = []
        for chunk in self.chunks.values():
            if chunk.embedding is None:
                continue
            score = self._cosine_similarity(query_embedding, chunk.embedding)
            if score >= min_score:
                scored.append((chunk, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def batch_search(self, queries: list[str], top_k: int = 3) -> list[list[tuple[DocumentChunk, float]]]:
        """
        Search for multiple queries at once.

        More efficient than calling search() in a loop because
        it can embed all queries in a single API call.

        Returns:
            List of results per query
        """
        if not queries:
            return []

        query_embeddings = self._embed_texts(queries)

        # Pre-extract chunk embeddings once to avoid recomputation per query
        valid_chunks = [c for c in self.chunks.values() if c.embedding is not None]

        all_results = []
        for q_emb in query_embeddings:
            scored = [
                (chunk, self._cosine_similarity(q_emb, chunk.embedding))
                for chunk in valid_chunks
            ]
            scored.sort(key=lambda pair: pair[1], reverse=True)
            all_results.append(scored[:top_k])

        return all_results

    def save(self, path: str) -> None:
        """Save the store to a JSON file (excluding embeddings to save space)."""
        data = {
            "documents": {
                doc_id: {
                    **asdict(doc),
                    "created_at": doc.created_at.isoformat(),
                }
                for doc_id, doc in self.documents.items()
            },
            "chunks": {
                chunk_id: {
                    "content": chunk.content,
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata,
                    # embedding intentionally excluded
                }
                for chunk_id, chunk in self.chunks.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """Load the store from a JSON file.

        Note: since embeddings are not persisted, they are
        regenerated automatically on load via add_chunk().
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.documents = {}
        self.chunks = {}

        for doc_id, doc_data in data.get("documents", {}).items():
            created_at_str = doc_data.pop("created_at", None)
            doc = Document(**doc_data)
            if created_at_str:
                doc.created_at = datetime.fromisoformat(created_at_str)
            self.documents[doc_id] = doc

        for chunk_id, chunk_data in data.get("chunks", {}).items():
            chunk = DocumentChunk(**chunk_data)
            # add_chunk will regenerate the embedding since it's None
            self.add_chunk(chunk)

def main():
    store = VectorStore()

    # Load and process the sample document
    doc = Document(
        content=open("data/sample_doc.txt", "r").read(),
        doc_id="DOC001",
        title="Introduction to Python Programming",
        source="data/sample_doc.txt"
    )
    store.add(doc)
    print(f"Loaded: {store.count()}")

    # Test single search
    print("\n=== Single Search ===")
    results = store.search("What is Python used for?", top_k=3)
    for chunk, score in results:
        print(f"  [{score:.4f}] {chunk.content[:70]}...")

    # Test batch search
    print("\n=== Batch Search ===")
    queries = ["What is Python?", "Tell me about data types", "How do I start?"]
    batch_results = store.batch_search(queries, top_k=2)
    for q, results in zip(queries, batch_results):
        print(f"\nQuery: '{q}'")
        for chunk, score in results:
            print(f"  [{score:.4f}] {chunk.content[:60]}...")

    # Test save/load
    store.save("data/vector_store.json")
    print(f"\nSaved to data/vector_store.json")

    store2 = VectorStore()
    store2.load("data/vector_store.json")
    print(f"Loaded: {store2.count()}")
    print(f"Search after load: {len(store2.search('Python', top_k=2))} results")

if __name__ == "__main__":
    main()