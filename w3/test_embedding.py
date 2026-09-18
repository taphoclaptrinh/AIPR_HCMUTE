from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

@dataclass
class Document:
    content : str
    doc_id : str = ""
    title : str = ""
    source: str = ""
    metadata: dict = field(default_factory=dict)
    created_at: Optional[datetime] = field(default_factory=datetime.now)

@dataclass
class DocumentChunk:
    content : str
    chunk_id : str = ""
    chunk_index : int = 0
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None

def chunk_document(doc: Document, chunk_size: int = 512, overlap: int = 64) -> list[DocumentChunk]:
    
    """
    Args:
        doc (Document): the document to be chunked
        chunk_size (int): the size of each chunk
        overlap (int): The number of overlapping characters between chunks
    returns:
        list[DocumentChunk]: A list of DocumentChunk instances
    """

    chunks = []
    start = 0
    text_len = len(doc.content)
    index = 0

    if not doc.content:
        return chunks

    step = chunk_size - overlap

    while(start < text_len):
        end = min(start + chunk_size, text_len)
        chunk_text = doc.content[start:end]

        chunk_obj = DocumentChunk(
            content=chunk_text,
            chunk_id=f"{doc.doc_id}_chunk_{index}",
            chunk_index=index,
            metadata={
                "source": doc.source,
                "start_char": start,
                "end_char": min(end, text_len)
            }
        )
        chunks.append(chunk_obj)
        index += 1
        start += step

        if step <= 0:
            break

    return chunks

