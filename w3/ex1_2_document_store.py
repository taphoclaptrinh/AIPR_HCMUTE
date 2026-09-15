from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from ex1_1document_model import chunk_document

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

    def add(self, document: Document)->str:
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

# === Test it ===
store = DocumentStore()

# Add 3 sample documents
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

print(store.count())
print(f"\nDocuments ({len(store.list_all())}):")
for d in store.list_all():
    chunks = store.get_chunks_by_doc(d.doc_id)
    print(f"  - {d.title} ({d.doc_id}): {len(chunks)} chunks")

print(f"\nAll chunks ({len(store.chunks)}):")
for cid, chunk in store.chunks.items():
    print(f"  - [{cid}] {chunk.content[:60]}...")