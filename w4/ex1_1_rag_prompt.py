RAG_SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question based ONLY on the following context.
If the context does not contain enough information, say: "I cannot find enough information in the provided documents."

Context:
{context}

Question: {question}"""

def format_chunk(chunk, index):
    """Format a single chunk with its source metadata for the prompt."""
    source = chunk.metadata.get("source", "Unknown")
    title = chunk.metadata.get("title", "document")
    return f"[(index+1)] (Source: {title}, {source})\n{chunk.content}"

def build_rag_prompt(question:str, chunks: list, top_k: int = 3) -> list[dict]:
    """ 
    Build the message list for a RAG request
    
    Args:
        question: The user's question
        chunks: List of DocumentChunk objects from VectorStore.search()
        top_k: Max chunks to include (defensive - search should already limit)
        
    Returns:
        message: list of ("role", "content") dicts ready for the API
    """
    
    context_text = "\n\n".join([format_chunk(chunk, i) for i, chunk in enumerate(chunks[:top_k])])
    
    system_prompt = RAG_SYSTEM_PROMPT.format(context=context_text, question=question)
    
    @dataclass
    class FakeChunk:
        content: str
        source: str
        title: str
        
        def metadata(self) -> dict[str, str]:
            return {"source": self.source, "title": self.title}
        
    chunks = {
        FakeChunk(
            "Employees may work remotely up to three days per week.",
            "company_policy.txt", "Remote Work Policy"
        ),
        FakeChunk(
            "Remote work requires prior approval from the line manager.",
            "company_policy.txt", "Remote Work Policy"
        ),
    }
    
    messages = build_rag_prompt("What is the remote work policy?", chunks)
    
    print("===System Prompt===")
    print(messages[0]["content"])
    print("\n===User Message===")
    print(messages[1]["content"])