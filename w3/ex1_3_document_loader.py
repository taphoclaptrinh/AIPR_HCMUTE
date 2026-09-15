from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import os


# ===== Document, DocumentChunk, DocumentStore (from Exercise 1.2) =====

@dataclass
class Document:
    content: str
    doc_id: str
    title: str = ""
    source: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class DocumentStore:
    """Simple in-memory store for documents and their chunks."""

    def __init__(self):
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, DocumentChunk] = {}

    def add_document(self, document: Document) -> None:
        self.documents[document.doc_id] = document

    def add_chunk(self, chunk: DocumentChunk) -> None:
        self.chunks[chunk.chunk_id] = chunk

    def get_chunks_for_doc(self, doc_id: str) -> list[DocumentChunk]:
        return [c for c in self.chunks.values() if c.doc_id == doc_id]

    def count(self) -> int:
        return len(self.chunks)


# ===== Chunking strategies =====

def chunk_fixed(text: str, chunk_size: int = 200, overlap: int = 30) -> list[str]:
    """Fixed-size chunking with character overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
        if start >= len(text):
            break
    return chunks


def chunk_recursive(text: str, chunk_size: int = 200) -> list[str]:
    """
    Recursive chunking — split on paragraph boundaries first,
    then fall back to sentence boundaries.

    This preserves natural document structure without requiring regex.
    """
    # Level 1: Split on double newlines (paragraphs)
    paragraphs = [p for p in text.split('\n\n') if p.strip()]

    chunks = []
    buffer = ""

    for para in paragraphs:
        # If adding this paragraph exceeds chunk_size, finish the current chunk
        if len(buffer) + len(para) > chunk_size and buffer:
            chunks.append(buffer.strip())
            buffer = ""

        # If a single paragraph is larger than chunk_size, split it by sentences
        if len(para) > chunk_size:
            sentences = [s.strip() + "." for s in para.split(".") if s.strip()]
            for sentence in sentences:
                if len(buffer) + len(sentence) > chunk_size and buffer:
                    chunks.append(buffer.strip())
                    buffer = sentence
                else:
                    buffer += " " + sentence if buffer else sentence
        else:
            buffer += "\n\n" + para if buffer else para

    if buffer:
        chunks.append(buffer.strip())

    return chunks


# ===== Document Loader =====

def load_document(path: str, doc_id: str = "", title: str = "") -> Document:
    """Read a text file and return a Document."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return Document(
        content=content,
        doc_id=doc_id or f"doc_{abs(hash(path)) % 10000:04d}",
        title=title or path,
        source=path
    )


def process_document(
    path: str,
    store: DocumentStore,
    strategy: str = "recursive",
    chunk_size: int = 200,
    overlap: int = 30
) -> dict:
    """
    Read a file, chunk it, store in DocumentStore, and return statistics.

    Args:
        path: Path to the text file
        store: DocumentStore to add to
        strategy: "fixed" or "recursive"
        chunk_size: Chunk size in characters
        overlap: Overlap between chunks (fixed strategy only)

    Returns:
        Statistics dict with keys: doc_id, num_chunks, avg_length, min_length, max_length, strategy
    """
    # 1. Load the document
    document = load_document(path)
    store.add_document(document)

    # 2. Choose the chunking strategy
    if strategy == "fixed":
        raw_chunks = chunk_fixed(document.content, chunk_size=chunk_size, overlap=overlap)
    elif strategy == "recursive":
        raw_chunks = chunk_recursive(document.content, chunk_size=chunk_size)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'fixed' or 'recursive'.")

    # 3. Create DocumentChunk instances
    # 4. Add to store
    lengths = []
    for i, chunk_text in enumerate(raw_chunks):
        chunk = DocumentChunk(
            chunk_id=f"{document.doc_id}_{strategy}_{i:04d}",
            doc_id=document.doc_id,
            content=chunk_text,
            chunk_index=i,
            metadata={"strategy": strategy, "chunk_size": chunk_size, "overlap": overlap},
        )
        store.add_chunk(chunk)
        lengths.append(len(chunk_text))

    # 5. Return statistics
    num_chunks = len(lengths)
    stats = {
        "doc_id": document.doc_id,
        "num_chunks": num_chunks,
        "avg_length": (sum(lengths) / num_chunks) if num_chunks else 0.0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "strategy": strategy,
    }
    return stats


def print_stats(stats: dict):
    """Print chunking statistics."""
    print(f"\nDocument: {stats['doc_id']}")
    print(f"Strategy: {stats['strategy']}")
    print(f"Chunks: {stats['num_chunks']}")
    print(f"Avg length: {stats['avg_length']:.1f} chars")
    print(f"Min length: {stats['min_length']} chars")
    print(f"Max length: {stats['max_length']} chars")


# ===== Main =====
def main():
    store = DocumentStore()

    sample_path = "data/sample_doc.txt"

    # Create a sample file automatically if it doesn't exist,
    # so the script can be run standalone.
    if not os.path.exists(sample_path):
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        sample_text = (
            "Retrieval-Augmented Generation (RAG) combines a retriever with a "
            "generator to produce answers grounded in external documents. "
            "Instead of relying solely on parameters learned during training, "
            "the model looks up relevant passages at inference time.\n\n"
            "Chunking is the process of splitting long documents into smaller "
            "pieces so they can be embedded and searched efficiently. The size "
            "and overlap of chunks affect retrieval quality significantly. "
            "Too large, and irrelevant text dilutes the embedding. Too small, "
            "and context is lost.\n\n"
            "Fixed-size chunking is simple and fast, but it can cut sentences "
            "or ideas in half. Recursive chunking tries to respect natural "
            "boundaries like paragraphs and sentences before falling back to "
            "hard splits, producing more coherent chunks.\n\n"
            "Once chunks are created, they are typically embedded using a "
            "vector model and stored in a vector database. At query time, the "
            "most similar chunks are retrieved and passed to the language "
            "model as context for generating a final answer."
        )
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(sample_text)

    # Process the sample document with both strategies
    print("=== Recursive Chunking ===")
    stats1 = process_document(
        sample_path,
        store,
        strategy="recursive",
        chunk_size=200
    )
    print_stats(stats1)

    print("\n=== Fixed-Size Chunking ===")
    stats2 = process_document(
        sample_path,
        store,
        strategy="fixed",
        chunk_size=200,
        overlap=30
    )
    print_stats(stats2)

    print(f"\nTotal in store: {store.count()}")


if __name__ == "__main__":
    main()