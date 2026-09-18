import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
import httpx
from dotenv import load_dotenv
from groq import Groq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "w3")))
from vector_store import VectorStore, Document

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
    http_client=httpx.Client(verify=False),
)

RAG_SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question based ONLY on the following context.
If the context does not contain enough information, say: \"I cannot find enough information in the provided documents.\"

Context:
{context}

Question: {question}"""

def build_rag_prompt(question: str, chunks: list) -> list[dict]:
    """Create the full message list for the LLM.
    The system prompt is formatted with the concatenated context.
    Each chunk is displayed with its source title for reference.
    """
    context_text = "\n\n".join(
        f"[{idx + 1}] (Source: {chunk.metadata.get('title', 'doc')}) {chunk.content}" for idx, chunk in enumerate(chunks)
    )
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT.format(context=context_text, question=question)},
        {"role": "user", "content": question},
    ]

def rag_pipeline(query: str, store: VectorStore, top_k: int = 3) -> tuple[str, list]:
    """Run a retrieval‑augmented generation step.
    Returns the answer string and the list of (chunk, score) results.
    """
    results = store.search(query, top_k=top_k)
    if not results:
        return "I cannot find enough information in the provided documents.", []
    chunks = [chunk for chunk, _ in results]
    messages = build_rag_prompt(query, chunks)
    response = client.chat.completions.create(model="openai/gpt-oss-120b", messages=messages)
    answer = response.choices[0].message.content or ""
    return answer, results

def load_policy_into_store(store: VectorStore, path: str) -> Document:
    """Read the policy file and add it to the VectorStore.
    Metadata includes source, title, and type.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    doc = Document(
        content=content,
        doc_id="policy_001",
        title="Company Policy",
        source=path,
        metadata={"type": "policy", "title": "Company Policy", "source": path},
    )
    store.add(doc)
    return doc

if __name__ == "__main__":
    # Initialise store and load the policy document
    store = VectorStore()
    policy_path = os.path.join(os.path.dirname(__file__), "company_policy.txt")
    load_policy_into_store(store, policy_path)

    # Define the three questions
    questions = [
        "How many vacation days are provided in the first year?",
        "Can I work from home and claim home office expenses?",
        "What is the stock option vesting schedule?",
    ]

    for i, q in enumerate(questions, start=1):
        answer, results = rag_pipeline(q, store)
        rendered_prompt = build_rag_prompt(q, [chunk for chunk, _ in results])
        print(f"\n=== Q{i}: {q} ===")
        print("--- Rendered Prompt (system message) ---")
        print(rendered_prompt[0]["content"])
        print("--- Answer ---")
        print(answer)
        print("--- Sources (score) ---")
        for chunk, score in results:
            src = chunk.metadata.get("title", "doc")
            print(f"- [{score:.4f}] {src}: {chunk.content[:120].replace('\n', ' ')}...")

    print("\n=== Observation Table ===")
    print("| Question | Type | Answer Starts With | LLM Said \"I cant find\" | Sources Relevant? |")
    print("|---|---|---|---|---|")
    answers = []
    for q, (ans, res) in zip(questions, [rag_pipeline(q, store) for q in questions]):
        ans_start = ans.split(" ")[0] if ans else ""
        cant_find = "I cannot find" in ans
        sources_rel = "Yes" if res else "No"
        if "vacation" in q.lower():
            qtype = "Direct"
        elif "home office" in q.lower():
            qtype = "Partial"
        else:
            qtype = "None"
        print(f"| {q} | {qtype} | {ans_start} | {cant_find} | {sources_rel} |")