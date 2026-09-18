from dataclasses import dataclass, field, asdict
from datetime import datetime
import json

@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: str
    embedding: list[float] | None

def load_chunks(path: str) -> list[DocumentChunk]:
    with open(path, "r") as f:
        data = json.load(f)
    for r in data:
        chunk = DocumentChunk(**r)
    return [DocumentChunk(**r) for r in data]

def find_missing_embeddings(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    null_embedding = []
    for chunk in chunks:
        if chunk.embedding is None:
            null_embedding.append(chunk)

    dict_list = [asdict(chunk) for chunk in null_embedding]
    print(json.dumps(dict_list, indent=2, default = str))
    return null_embedding

def add_char_count(chunks: list[DocumentChunk]) ->list[DocumentChunk]:
    pass

def save_chunks(chunks: list[DocumentChunk], path: str) -> None:
    pass

class DataClassEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '_dataclass_fields_'):
            return asdict(obj)
        return super().default(obj)

chunks = load_chunks("w1\\data\\document_chunks.json")

missing_chunks = find_missing_embeddings(chunks)