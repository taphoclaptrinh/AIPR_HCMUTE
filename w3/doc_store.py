from test_embedding import Document, DocumentChunk
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass

class DoucmentStore:
    """
    
    An in-memory store for documents and their chunks.

    this is the foundation. Later, we'll replace the internal dict
    With a vector database but the interface stays the same.

    """

    documents: dict[str, Document] = field(default_factory=dict)
    chunks: dict[str, DocumentChunk] = field(default_factory=dict)

    def add(self, doc: Document) -> str:
        """
        add a document to the store
        Args:
            doc(Document):the document to be added
        Returns:
        str: doc_id of the added document
        """
        if not doc.doc_id:
            doc.doc_id = f"DOC{len(self.documents):04d}"
        self.documents[doc.doc_id] = doc
        return doc.doc_id

    def delete(self, doc_id: str) -> bool:
        """
        Delete a doc and it chunks. Return true if deleted
        """

        if doc_id in self.documents:
            self.documents.pop(doc_id)
            self.chunks = {k: v for k,v in self.chunks.items() if v.doc_id != doc_id}
            return True
        return False

    def list_all(self) -> list[Document]:
        return list(self.documents.values)

    def add_chunks(self, chunk: DocumentChunk) ->str:

        """
        add a chunk and return it chunk_id
        """

        if not chunk.chunk_id:
            chunk.chunk_id = f"DOC{len(self.chunks):04d}"
        self.chunks[chunk.chunk_id] = chunk
        return chunk.chunk_id

    def get_chunks(self, chunk_id: str) -> Optional[DocumentChunk]:
        return self.chunks.get(chunk_id)

    def get_chunk_by_doc(self, doc_id: str) -> list[DocumentChunk]:
        """
        get all chunks
        """
        return [chunk for chunk in self.chunks.values() if chunk.doc_id == doc_id]
    
