from jina_embeddings import jina_embeddings
import os
from rich import print
from dotenv import load_dotenv
load_dotenv()

SENTENCES = [
    "I love programming in Python",
]

resp = jina_embeddings(input=SENTENCES, api_key=environ["JINA_API_KEY"])

print(resp.data[0].embedding[:10])
print(resp["data"][0]["embeding"][:10])
print(resp.model)
