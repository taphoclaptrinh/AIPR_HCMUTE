"""
Streaming Demo — Week 2, Session 2
So sánh non-streaming vs streaming khi gọi LLM API (OpenAI-compatible).
"""

import os
import time
from groq import Groq
from dotenv import load_dotenv

# 1. Nạp biến môi trường từ file .env
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise SystemExit(
        "Không tìm thấy OPENAI_API_KEY.\n"
        "→ Tạo file .env cùng thư mục, thêm dòng: OPENAI_API_KEY=sk-..."
    )

client = Groq()

MODEL = "openai/gpt-oss-120b"
PROMPT = "Write a 200-word story about a robot learning to paint."


def run_non_streaming():
    print("\n" + "=" * 60)
    print("NON-STREAMING — đợi phản hồi đầy đủ rồi mới in ra")
    print("=" * 60)

    start = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
    )
    elapsed = time.time() - start

    print(response.choices[0].message.content)
    print(f"\n[Hoàn thành sau {elapsed:.2f}s — chỉ in được SAU KHI có toàn bộ kết quả]")


def run_streaming():
    print("\n" + "=" * 60)
    print("STREAMING — token xuất hiện dần dần")
    print("=" * 60)

    start = time.time()
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        stream=True,
    )

    full_response = ""
    finish_reason = None

    try:
        for chunk in stream:
            # Một số chunk có thể không mang choices (ví dụ chunk metadata đầu/cuối)
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # delta.content là None ở chunk cuối cùng -> phải kiểm tra trước khi cộng dồn
            if delta.content:
                full_response += delta.content
                print(delta.content, end="", flush=True)

            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

    except KeyboardInterrupt:
        # Cho phép người dùng Ctrl+C dừng giữa chừng mà không bị crash xấu xí
        print("\n\n[Đã dừng bởi người dùng — Ctrl+C]")
        return

    elapsed = time.time() - start
    print(f"\n\n[Finished: {finish_reason} — mất {elapsed:.2f}s, nhưng chữ xuất hiện NGAY LẬP TỨC]")
    print(f"[Tổng độ dài phản hồi: {len(full_response)} ký tự]")


if __name__ == "__main__":
    run_non_streaming()
    run_streaming()

    print("\n" + "=" * 60)
    print("So sánh: cả hai cách đều tốn thời gian gần bằng nhau,")
    print("nhưng streaming cho CẢM GIÁC nhanh hơn vì người dùng thấy")
    print("chữ xuất hiện ngay, không phải chờ trong im lặng.")
    print("=" * 60)
