import os
from dotenv import load_dotenv
from groq import Groq

# Nạp các biến môi trường từ file .env
load_dotenv()

# Khởi tạo client Groq (lúc này client sẽ tự lấy GROQ_API_KEY vừa được load)
client = Groq()

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",  # Hoặc tên model bạn chọn
    messages=[
        {
            "role": "user",
            "content": "Xin chào! Giới thiệu ngắn gọn về bạn."
        }
    ],
    temperature=1,
    max_completion_tokens=2048,
    top_p=1,
    reasoning_effort="medium",
    stop=None
)

def estimate_cost(usage):
    default_token = 1_000_000
    total_cost = 0.15 * (usage.prompt_tokens / default_token) + 0.6 * (usage.completion_tokens / default_token)
    return total_cost

print(response.choices[0].message.content)
print("===Metadata===")
print(f"model={response.model}")
print(f"prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")

usage = response.usage
print(estimate_cost(usage))