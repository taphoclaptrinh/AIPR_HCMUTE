from jina_embeddings import jina_embeddings
import os
from numpy import np
from rich import print
from dotenv import load_dotenv

load_dotenv()

def embed_text(text:str, model="jina-embeddings-v3", api_key=None) -> list[float]:
    if api_key is None:
        api_key = os.getenv("JINA_API_KEY")
    resp = jina_embeddings(input=text, model=model, api_key=api_key)
    return resp["data"][0]["embedding"]

def cosine_similarity(a:list[float], b:list[float]) ->float:
    """compute cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)

    
    return np.dot(a, b) / (np.linalg.norm(a) + np.linalg.norm(b))

sentences = [
    "RAQ combines retrieval with text generation",
    "Python is a popular programming language",
    "I enjoy hiking in the mountains",
    "C++ is second Python in my heart",
    "I love python programming",
]

sin_0_1 = cosine_similarity(embed_text(sentences[0]), embed_text(sentences[1]))
sin_1_2 = cosine_similarity(embed_text(sentences[1]), embed_text(sentences[2]))
sin_0_2 = cosine_similarity(embed_text(sentences[0]), embed_text(sentences[2]))
sin_1_4 = cosine_similarity(embed_text(sentences[1]), embed_text(sentences[4]))
sin_4_5 = cosine_similarity(embed_text(sentences[4]), embed_text(sentences[5]))

print(f"Rag vs Python {sin_0_1: .4f}")
print(f"Rag vs Hiking {sin_0_1: .4f}")
print(f"Hiking vs Python {sin_0_1: .4f}")