from rank_bm25 import BM25Okapi
from rich import print

documents = [
    "The pink dolphin swims in the river",
    "Pink flowers grow in the summer garden",
    "A dolphin is a smart marine animal"
]

tokenized = [doc.lower().split() for doc in documents]
print(f"Tokenized documents: {tokenized}")

bm25 = BM25Okapi(tokenized)

query = "pink dolphin".lower().split()
scores = bm25.get_scores(query)

for doc, score in zip(documents, scores):
    print(f"Score: {score:.3f} | Document: {doc}")