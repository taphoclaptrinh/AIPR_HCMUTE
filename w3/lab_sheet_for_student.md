# Week 3 — Lab Sheet (Student)

## RAG Foundations: Documents & Embeddings

> **Course:** AI Programming with Python
> **Week:** 3 of 8
> **Student Name:** ____________________
> **Date:** ____________________

---

## Learning Objectives (Week 3)

By the end of this week, you will be able to:

- **A3.1** Design a `Document` and `DocumentChunk` @dataclass with metadata.
- **A3.2** Implement a `DocumentStore` with add, get, delete, and list operations.
- **A3.3** Implement a chunking strategy (fixed-size, recursive, semantic).
- **A3.4** Generate embeddings using an embedding API and compute cosine similarity between vectors.

---

## Setup

### Install Dependencies

```bash
pip install openai python-dotenv numpy
```

### Create Data Directory

```bash
mkdir -p data
```

### Create Sample Data File

Create `data/sample_doc.txt` with the following content:

```text
# Introduction to Python Programming

Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991. It emphasizes code readability with its notable use of significant indentation.

## Key Features

Python is dynamically-typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented, and functional programming. Python is often described as a "batteries included" language due to its comprehensive standard library.

## Python in AI and Data Science

Python has become one of the most popular programming languages for artificial intelligence and data science. Libraries like NumPy, pandas, scikit-learn, and TensorFlow make Python the go-to choice for machine learning projects.

## Getting Started with Python

To start programming in Python, you need to install the Python interpreter from python.org. You can write Python code in any text editor, but using an IDE like VS Code or PyCharm is recommended for larger projects.

## Python Data Types

Python has several built-in data types: integers, floats, strings, lists, tuples, dictionaries, and sets. Lists are mutable sequences, tuples are immutable, and dictionaries store key-value pairs. Understanding these types is essential for effective Python programming.
```

---

> **Cá nhân hóa:** Các lab trong tuần này được gán tham số riêng cho bạn (chunk_size, overlap, top_k, min_score, bộ 3 documents) — giảng viên sẽ cung cấp. Dùng đúng tham số được gán (số chunk kỳ vọng của bạn phụ thuộc chunk_size/overlap của bạn).

## Session 1: In-Lab Exercises

---

### Exercise 1.1 — Code Along: Document Model & Chunking (Guided)

**File:** `ex1_1_document_model.py`

> **Note:** The `Document` and `DocumentChunk` @dataclass definitions are from the lecture (Concept 1). Copy them into your file, then focus on the new code: the `chunk_document()` function below.

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# (Document and DocumentChunk @dataclass — copy from Concept 1 of the lecture)

# === Step 1: Create a sample document ===

doc = Document(
    content="""Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991. It emphasizes code readability with its notable use of significant indentation. Python is dynamically-typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented, and functional programming. Python is often described as a "batteries included" language due to its comprehensive standard library. It has become one of the most popular programming languages, particularly in data science, artificial intelligence, and web development.""",
    doc_id="DOC001",
    title="Introduction to Python",
    source="course_materials"
)

print(f"Document: {doc.title} ({doc.doc_id})")
print(f"Content length: {len(doc.content)} chars\n")

# === Step 3: Implement chunking ===

def chunk_document(doc: Document, chunk_size: int = 150, overlap: int = 30) -> list[DocumentChunk]:
    """
    Split a Document into DocumentChunks using fixed-size windowing.
    """
    chunks = []
    text = doc.content
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]

        chunk = DocumentChunk(
            content=chunk_text,
            chunk_id=f"{doc.doc_id}_chunk_{len(chunks)}",
            doc_id=doc.doc_id,
            chunk_index=len(chunks),
            metadata={**doc.metadata, "chunk_size": chunk_size, "overlap": overlap}
        )
        chunks.append(chunk)

        start += chunk_size - overlap
        if start >= len(text):
            break

    return chunks

# === Step 4: Test it ===

chunks = chunk_document(doc, chunk_size=150, overlap=30)
print(f"Created {len(chunks)} chunks:\n")

for chunk in chunks:
    print(f"[{chunk.chunk_id}] (index {chunk.chunk_index})")
    print(f"  Text: {chunk.content[:80]}...")
    print(f"  Length: {len(chunk.content)} chars")
    print()
```

**Checkpoint:**
```
Document: Introduction to Python (DOC001)
Content length: 498 chars

Created 4 chunks:
...
```

---

### Exercise 1.2 — Code Along: DocumentStore (Guided)

**File:** `ex1_2_document_store.py`

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# Reuse the Document and DocumentChunk from Exercise 1.1
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
    """An in-memory store for Documents and their Chunks."""
    documents: dict[str, Document] = field(default_factory=dict)
    chunks: dict[str, DocumentChunk] = field(default_factory=dict)

    def add(self, document: Document) -> str:
        """Add a document and auto-chunk it."""
        if not document.doc_id:
            document.doc_id = f"doc_{len(self.documents) + 1:04d}"
        self.documents[document.doc_id] = document

        # Auto-chunk the document
        chunks = chunk_document(document)
        for chunk in chunks:
            self.add_chunk(chunk)

        return document.doc_id

    def get(self, doc_id: str) -> Optional[Document]:
        return self.documents.get(doc_id)

    def get(self, doc_id: str) -> Optional[Document]:
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
        return self.chunks.get(chunk_id)

    def get_chunks_by_doc(self, doc_id: str) -> list[DocumentChunk]:
        return [c for c in self.chunks.values() if c.doc_id == doc_id]

    def count(self) -> dict:
        return {"documents": len(self.documents), "chunks": len(self.chunks)}

def chunk_document(doc: Document, chunk_size: int = 150, overlap: int = 30) -> list[DocumentChunk]:
    """Split a Document into DocumentChunks."""
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
```

**Checkpoint:** You should see 3 documents and 12 chunks (4 per document).

---

### Exercise 1.3 — Independent Lab: Document Loader

**File:** `ex1_3_document_loader.py`

**⏱ Time:** 40 minutes. Work independently.

**Your task:** Write a function that reads a text file, chunks it with a configurable strategy, stores chunks in a `DocumentStore`, and prints statistics.

**Starter code:**

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# ===== Reuse Document, DocumentChunk, DocumentStore from Exercise 1.2 =====
# (Copy the classes here or import from ex1_2_document_store.py)

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
    # Your code here:
    # 1. Load the document
    # 2. Choose the chunking strategy
    # 3. Create DocumentChunk instances
    # 4. Add to store
    # 5. Return statistics
    pass

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

    # Process the sample document with both strategies
    print("=== Recursive Chunking ===")
    stats1 = process_document(
        "data/sample_doc.txt",
        store,
        strategy="recursive",
        chunk_size=200
    )
    print_stats(stats1)

    print("\n=== Fixed-Size Chunking ===")
    stats2 = process_document(
        "data/sample_doc.txt",
        store,
        strategy="fixed",
        chunk_size=200,
        overlap=30
    )
    print_stats(stats2)

    print(f"\nTotal in store: {store.count()}")

if __name__ == "__main__":
    main()
```

**Expected output (approximate):**
```
=== Recursive Chunking ===
Document: doc_XXXX
Strategy: recursive
Chunks: 3
Avg length: 156.0 chars
Min length: 98 chars
Max length: 198 chars

=== Fixed-Size Chunking ===
Document: doc_XXXX
Strategy: fixed
Chunks: 4
Avg length: 200.0 chars
Min length: 200 chars
Max length: 200 chars

Total in store: {'documents': 2, 'chunks': 7}
```

**Extension (if you finish early):**
- Add a `chunk_semantic()` function that uses an LLM to identify topic boundaries
- Handle edge cases: empty file, single word, file with only whitespace
- Add a `--strategy` command-line argument to choose the strategy

---

### Exercise 1.4 — Peer Review

**⏱ Time:** 15 minutes

**Pair up with a partner.** Review Exercise 1.3 using this checklist:

- [ ] Does the chunking preserve paragraph boundaries? (recursive strategy)
- [ ] Are edge cases handled? (empty doc, single word, single character)
- [ ] Are the statistics useful? (num chunks, avg length, min/max)
- [ ] Is the DocumentStore correctly populated?
- [ ] Is the code readable?

---

### Knowledge Check 1

**MCQ:** Complete this during class.

> Why do we use overlapping chunks when splitting a document?
>
> (a) To increase the number of chunks (more chunks = more retrieval)
> (b) To avoid cutting a sentence or thought in half at the boundary
> (c) To reduce storage space
>
> **Your answer:** ____

---

### Knowledge Check 2

**MCQ:**

> A chunk of 200 chars returns relevant results but misses nuance. A chunk of 2000 chars includes all the context but also a lot of noise. What is the trade-off?
>
> (a) Precision vs. recall
> (b) Speed vs. accuracy
> (c) Both precision/recall AND speed/accuracy
>
> **Your answer:** ____

---

### Reflection (Session 1)

Write 3–5 sentences:

> What's the hardest part about chunking? What questions do you have about the trade-offs?

```
________________________________________________________________________

________________________________________________________________________

________________________________________________________________________

________________________________________________________________________

________________________________________________________________________
```

---

## Session 2: In-Lab Exercises

---

### Exercise 2.1 — Code Along: Embedding API (Guided)

**File:** `ex2_1_embeddings.py`

> **Remember the two API structures** (keep this in mind for the whole session):
>
> ```
> CHAT API (Week 2):                     EMBEDDING API (Week 3):
> --------------------                   ------------------------
> client.chat.completions.create(...)    client.embeddings.create(...)
> response.choices[0]                    response.data[0]
>   .message.content   → str             .embedding   → list[float]
>   .finish_reason     → str             (no finish_reason)
> response.usage.prompt_tokens           response.usage.prompt_tokens
> response.model                         response.model
> ```

```python
import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_text(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Embed a single text and return the vector."""
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding

def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Embed multiple texts and return vectors."""
    response = client.embeddings.create(model=model, input=texts)
    return [data.embedding for data in response.data]

# === Step 1: Embed one sentence ===
sentence = "RAG stands for Retrieval-Augmented Generation"
vec = embed_text(sentence)
print(f"Sentence: {sentence}")
print(f"Dimension: {len(vec)}")
print(f"First 5 values: {vec[:5]}")
print(f"Vector norm: {np.linalg.norm(vec):.4f}")
print(f"Min: {min(vec):.4f}, Max: {max(vec):.4f}")
print()

# === Step 2: Embed three sentences about different topics ===
sentences = [
    "RAG combines retrieval with text generation",
    "Python is a popular programming language",
    "I enjoy hiking in the mountains"
]

embeddings = embed_texts(sentences)

for s, v in zip(sentences, embeddings):
    print(f"  '{s[:40]}...' → dim {len(v)}, norm {np.linalg.norm(v):.4f}")
```

**Checkpoint:**
```
Sentence: RAG stands for Retrieval-Augmented Generation
Dimension: 1536
First 5 values: [0.023, -0.045, 0.112, -0.067, 0.089]
Vector norm: 1.0000

  'RAG combines retrieval with text generation...' → dim 1536, norm 1.0000
  'Python is a popular programming language...' → dim 1536, norm 1.0000
  'I enjoy hiking in the mountains...' → dim 1536, norm 1.0000
```

---

### Exercise 2.2 — Code Along: Cosine Similarity (Guided)

**File:** `ex2_2_similarity_search.py`

**Integrated annotation — formula → code mapping:**

```
cosine_similarity(a, b) = (a · b) / (||a|| × ||b||)
                           │      │    │      │
                           │      │    │      └ np.linalg.norm(b_arr)
                           │      │    └ np.linalg.norm(a_arr)
                           │      └ np.dot(a_arr, b_arr)
                           └ return float(dot_product / (norm_a * norm_b))
```

```python
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
```

**Expected output:**
```
=== Query: 'What is RAG?' ===
  [0.89] RAG is a technique for combining retrieval with generation
  [0.45] Embeddings map text to numerical vectors

=== Query: 'Tell me about vector similarity' ===
  [0.82] Cosine similarity measures the angle between vectors
  [0.65] Embeddings map text to numerical vectors

=== Query: 'What is the capital of France?' ===
  [0.91] The capital of France is Paris
  [0.25] RAG is a technique for combining retrieval with generation
```

---

### Exercise 2.3 — Independent Lab: Vector Search Engine

**File:** `ex2_3_vector_search.py`

**⏱ Time:** 40 minutes. Work independently.

**Recall sprint (2 min, pair activity):** Before you start coding, activate what you already know. **In 30 seconds, explain to your partner:**

> 1. What does `embed_text()` return? *(a list of 1536 floats)*
> 2. What does `cosine_similarity()` return? *(a float between -1 and 1)*
> 3. What does `DocumentStore.add()` do? *(stores a document and creates chunks)*

**Your task:** Build a `VectorStore` that extends `DocumentStore` with embedding-based search.

**Starter code:**

```python
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
        # implemented from previous in-class exercise

    def delete(self, doc_id: str) -> bool:
        # implemented from previous in-class exercise

    def list_all(self) -> list[Document]:
         # implemented from previous in-class exercise       

    def add_chunk(self, chunk: DocumentChunk) -> str:
        # implemented from previous in-class exercise
        

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        # implemented from previous in-class exercise        

    def get_chunks_by_doc(self, doc_id: str) -> list[DocumentChunk]:
        # implemented from previous in-class exercise

    def count(self) -> dict:
        # implemented from previous in-class exercise
    
    
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
```

**Expected output:**
```
Added: Introduction to Python
Added: What is RAG?
Added: Understanding Embeddings

Store: {'documents': 3, 'chunks': 12}

=== Query: 'What is RAG?' ===
  [0.XX] RAG stands for Retrieval-Augmented Generation...
  [0.XX] The retriever finds relevant documents...

=== Query: 'Tell me about Python' ===
  [0.XX] Python is a high-level programming language...
  [0.XX] Python emphasizes code readability...
```

**Hints:**
- In `add_chunk()`, check if `chunk.embedding is None`, then call `embed_text(chunk.content)`
- In `search()`, embed the query, then loop through all chunks and compute cosine similarity
- Sort results by score descending and return top_k
- `search_with_threshold()` can call `search()` with `top_k=len(self.chunks)` then filter

**Extension (if you finish early):**
- Add a `batch_search(queries: list[str])` method that embeds multiple queries at once
- Add a `save(path)` / `load(path)` method that serialises/deserialises the store to JSON
- Add a `--query` command-line argument for interactive search

---

### Exercise 2.4 — Peer Review

**⏱ Time:** 15 minutes

**Pair up with a partner.** Review Exercise 2.3 using this checklist:

- [ ] Does the search return semantically relevant results? (not just keyword matches)
- [ ] Are scores reasonable? (0.7–1.0 for relevant, < 0.5 for irrelevant)
- [ ] Is cosine similarity correctly implemented?
- [ ] Are edge cases handled? (empty store, query with no matches)
- [ ] Does `search_with_threshold` work correctly?

---

### Knowledge Check 1

**MCQ:** Complete this during class.

> What does a 1536-dimensional embedding from `text-embedding-3-small` represent?
>
> (a) 1536 keywords extracted from the text
> (b) 1536 numerical coordinates in a semantic vector space
> (c) 1536 tokens from the text
>
> **Your answer:** ____

---

### Knowledge Check 2

**MCQ:**

> A query returns chunks with similarity scores of 0.92, 0.88, and 0.45. Should you include the 0.45 result in the context sent to the LLM?
>
> (a) Yes — always include all results up to top_k
> (b) No — set a threshold > 0.7 and exclude low-quality results
> (c) It depends on the application — sometimes more context helps, sometimes it adds noise
>
> **Your answer:** ____

---

### Reflection (Session 2)

Write 3–5 sentences:

> What surprised you most about how embeddings work? Did any search results surprise you?

```
________________________________________________________________________

________________________________________________________________________

________________________________________________________________________

________________________________________________________________________

________________________________________________________________________
```

---

## Lab 3 — Homework (Due before Session 7)

### Overview

Build a complete `VectorStore` class that extends `DocumentStore` with embedding-based semantic search, batch search, and save/load persistence.

### Requirements

**File:** `lab3_vector_store.py`

**Part 1 — VectorStore class:**

```python
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
        # implemented from previous in-class exercise

    def delete(self, doc_id: str) -> bool:
        # implemented from previous in-class exercise

    def list_all(self) -> list[Document]:
         # implemented from previous in-class exercise       

    def add_chunk(self, chunk: DocumentChunk) -> str:
        # implemented from previous in-class exercise
        

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        # implemented from previous in-class exercise        

    def get_chunks_by_doc(self, doc_id: str) -> list[DocumentChunk]:
        # implemented from previous in-class exercise

    def count(self) -> dict:
        # implemented from previous in-class exercise

    def _chunk_document(self, doc: Document, chunk_size: int = 150, overlap: int = 30) -> list[DocumentChunk]:
        """Split a Document into chunks."""
        # implemented from previous in-class exercise

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
        # Your code here


    def batch_search(self, queries: list[str], top_k: int = 3) -> list[list[tuple[DocumentChunk, float]]]:
        """
        Search for multiple queries at once.

        More efficient than calling search() in a loop because
        it can embed all queries in a single API call.

        Returns:
            List of results per query
        """
        # Your code here


    def save(self, path: str) -> None:
        """Save the store to a JSON file (excluding embeddings to save space)."""
        # Your code here


    def load(self, path: str) -> None:
        """Load the store from a JSON file."""
        # Your code here

```

### Requirements Checklist

Your `VectorStore` class must:

- [ ] **`add_chunk()`** — auto-generates embedding if `chunk.embedding is None`
- [ ] **`search()`** — embeds query, computes cosine similarity against all chunks, returns top_k with scores
- [ ] **`search()` respects `min_score`** — only returns results with score >= min_score
- [ ] **`batch_search()`** — embeds multiple queries in one API call, returns results for each
- [ ] **`save()`** — saves documents and chunks to JSON (exclude embeddings by default, or include optionally)
- [ ] **`load()`** — restores documents and chunks from a JSON file
- [ ] **Edge cases** — empty store, query with no matches, store with unembedded chunks
- [ ] **Reasonable scores** — relevant results should score > 0.7, irrelevant < 0.5

### Main Function

```python
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
```

### Hints

- **`add_chunk()`**: `if chunk.embedding is None: chunk.embedding = embed_text(chunk.content)` then call `super().add_chunk(chunk)`
- **`search()`**: embed query → loop through `self.chunks.values()` → compute similarity → filter by `min_score` → sort → return top_k
- **`batch_search()`**: embed all queries in one API call → for each query vector, compute similarities against all chunks
- **`save()`**: use `asdict()` from dataclasses to convert to dicts, then `json.dump()`
- **`load()`**: read JSON, iterate through dicts, reconstruct Document and DocumentChunk instances

### Submission Checklist

Before submitting, verify:

- [ ] `add_chunk()` auto-generates embeddings
- [ ] `search()` returns semantically relevant results
- [ ] `search()` respects `min_score` parameter
- [ ] `batch_search()` handles multiple queries efficiently
- [ ] `save()` / `load()` round-trips correctly
- [ ] The script runs without errors
- [ ] All edge cases handled (empty store, no matches)

### How to Submit

- Save your file as `lab3_vector_store.py`
- Also save the output file `data/vector_store.json`
- Submit both files to your instructor by the due date

---

## Week 3 Quick Reference Card

### Document Model

```python
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
```

### Chunking

| Strategy | Method | Best For |
|---|---|---|
| Fixed-size | `text[start:start+size]` | Quick prototyping |
| Recursive | Split on `\n\n`, then `.`, buffer accumulation | Most production RAG |
| Semantic | LLM identifies topic boundaries (bonus) | High-stakes QA |

### Embedding API

```python
# Embedding — NOT chat.completions!
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="text to embed"
)
vector = response.data[0].embedding  # list of 1536 floats
```

**Don't confuse the two API response structures:**

| | CHAT API (Week 2) | EMBEDDING API (Week 3) |
|---|---|---|
| Method | `client.chat.completions.create()` | `client.embeddings.create()` |
| Response access | `response.choices[0].message.content` | `response.data[0].embedding` |
| Content type | string | `list[float]` |
| `finish_reason` | present | none |

### Cosine Similarity

```python
def cosine_similarity(a, b):
    a_arr, b_arr = np.array(a), np.array(b)
    if np.linalg.norm(a_arr) == 0 or np.linalg.norm(b_arr) == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))
```

### Vector Store Pattern

```python
store = VectorStore()
store.add(document)  # auto-chunks + embeds
results = store.search(query, top_k=3, min_score=0.5)
# results = [(chunk, score), ...]
```

---

## Common Errors & Solutions

| Error | Likely Cause | Fix |
|---|---|---|
| `ValueError: mutable default ... is not allowed` | `tags: list = []` in @dataclass | Use `field(default_factory=list)` |
| `TypeError: Object of type datetime is not JSON serialisable` | `json.dump()` on Document | Use custom encoder or `asdict()` |
| `ModuleNotFoundError: No module named 'numpy'` | numpy not installed | `pip install numpy` |
| `Embedding dimension mismatch` | Using different models | Use same model for all embeddings |
| `Cosine similarity > 1.0` | Floating point error | Clip to `[-1, 1]`: `max(-1, min(1, score))` |
| `Search returns 0 results` | `min_score` threshold too high | Lower threshold or check embeddings exist |
| `API key error for embeddings` | Using wrong API key | Embedding API uses same key as chat |
| `response.data[0].embedding` → KeyError | Confusing with chat response | Embedding response has `data[0].embedding`, not `choices[0]` |

*End of Week 3 Lab Sheet*