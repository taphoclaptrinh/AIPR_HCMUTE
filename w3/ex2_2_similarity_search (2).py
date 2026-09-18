from jina_embeddings import jina_embeddings
import os
from numpy import np
from rich import print
from dotenv import load_dotenv

load_dotenv()

def batch_embed_text(input:list[str], model="jina-embeddings-v3", api_key=None) -> list[list[float]]:
    if api_key is None:
        api_key = os.getenv("JINA_API_KEY")
    resp = jina_embeddings(input=input, model=model, api_key=api_key)
    return [item["embedding"] for item in resp["data"]]

def cosine_similarity(a:list[float], b:list[float]) ->float:
    """compute cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (np.linalg.norm(a) + np.linalg.norm(b))

def find_most_simila(query: str, candidates : list[str], top_k: int=3) ->list[tuple[str, float]]:
    query_vec = batch_embed_text([query], model="jina-embeddings-v3", api_key=None)
    candidates = batch_embed_text(candidates, model="jina-embeddings-v3", api_key=None)
    result = []
    for u,v in candidates:
        score = cosine_similarity(query_vec, v)
        result.append(u, score)

    result.sort(key=lambda x: x[1], reverse = True)
    return result[:top_k]

if __name__ == "__main__":

    sentences = [
        "RAQ combines retrieval with text generation",
        "Python is a popular programming language",
        "I enjoy hiking in the mountains",
        "C++ is second Python in my heart",
        "I love python programming",
    ]

    text = ["Thang Tin toc toc"]

    print(find_most_simila(text, sentences, top_k=3))

