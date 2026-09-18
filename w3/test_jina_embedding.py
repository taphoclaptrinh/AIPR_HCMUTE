import os
from pathlib import Path
from dotenv import load_dotenv
from jina_embeddings import jina_embeddings # Import hàm từ file chứa code của bạn

# Load biến môi trường từ file .env
load_dotenv()

# Văn bản bạn muốn AI xử lý
van_ban = ["Tôi thích ăn phở", "Thời tiết hôm nay rất đẹp"]

# Gọi hàm với các thông số tùy chỉnh
ket_qua = jina_embeddings(
    input=van_ban,
    api_key=os.getenv("JINA_API_KEY"),
    task="retrieval.query", # Tùy chỉnh task
    dimensions=512          # Tùy chỉnh số chiều
)

# In kết quả
for item in ket_qua.data:
    print(f"Vector của câu {item.index} có độ dài {len(item.embedding)}")