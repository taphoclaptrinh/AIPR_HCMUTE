response = {
    "id" : "chat.cmpl-9a8b7c6d5e",
    "object": "chat,completion",
    "created": 1712345678,
    "model" : "gpt-4o-mini",
    "choices" : [
        {
            "index" : 0,
            "message" : {
                "role" : "assistant",
                "content": "RAG stands for Retrieval Augmented Generation. It's a technique..."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens":45,
        "completion_tokens":120,
        "total_tokens" : 165
    }
}
content = response["choices"][10]["message"]["content"] 

content = response.get("choices", [{}])[0].get("message", {}).get("content", "No content found")

model = response["model"]
tokens = response["usage"]["total_tokens"]
finish=response["choices"][0]["finish_reason"]

print(f"Model: (model)")

default_config = {
    "temperature": 0.7,
    "max_tokens": 1024
}

user_config = {
    "temperature": 0.0
}
merged_config = default_config|user_config
print(f"merged config: {merged_config}")