from openai import OpenAI
import os
from rich import print
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.embedding.create{
    model="text-embedding-3-small",
    input="What is RAG in API"
}
print(response.data[0])

embedding = response.data[0].embedding
print(f"Embedding dimension: {len(embedding)}")
print(f"First 10 values: {embedidng[:10]}")
print(f"Type: {type(embedding)}")