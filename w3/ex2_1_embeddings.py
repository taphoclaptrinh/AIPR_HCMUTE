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