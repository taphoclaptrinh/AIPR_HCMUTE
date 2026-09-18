from jina_embeddings import jina_embeddings
import os
from rich import print
from dotenv import load_dotenv

load_dotenv()

def embed_text(text:str, model="jina-embeddings-v3", api_key=None) -> list[float]:
    if api_key is None:
        api_key = os.getenv("JINA_API_KEY")
    resp = jina_embeddings(input=text, model=model, api_key=api_key)
    return resp["data"][0]["embedding"]
def batch_embed_text(input:list[str], model="jina-embeddings-v3", api_key=None) -> list[list[float]]:
    if api_key is None:
        api_key = os.getenv("JINA_API_KEY")
    resp = jina_embeddings(input=input, model=model, api_key=api_key)
    return [item["embedding"] for item in resp["data"]]

# sentence = "RAQ stands for Retrieval-Augented Generation"
# vec = embed_text(sentence)
# print(f"Sentence: {sentence}")
# print(f"Dimension: {len(vec)}")
# print(f"First 5 values: {vec[:5]}")

sentences = [
    "RAQ combines retrieval with text generation",
    "Python is a popular programming language",
    "I enjoy hiking in the mountains",
]

# for sentence in sentences:
#     print(f"Sentence:{sentence}")
#     print(f"Dimension: {len(embed_text(sentence))}")
#     print(f"First 5 values: {embed_text(sentence)[:5]}")

embeddings = batch_embed_text(sentences)
print(type(embeddings))
for s,v in zip(sentences, embeddings):
    print(f"    '{s[:40]}...' -> {v[:5]}")