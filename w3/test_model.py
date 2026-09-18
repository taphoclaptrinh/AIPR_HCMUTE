from test_embedding import Document, DocumentChunk, chunk_document

# ==========================================
# KHỐI CODE CHẠY THỬ (TEST RUN)
# ==========================================
if __name__ == "__main__":
    # 1. Tạo thử một tài liệu mẫu
    sample_text = (
        "Hà Nội là thủ đô của nước Cộng hòa Xã hội Chủ nghĩa Việt Nam. "
        "Đây là thành phố lớn nhất Việt Nam về diện tích và là thành phố đứng thứ hai về dân số. "
        "Nằm ở trung tâm đồng bằng sông Hồng, Hà Nội là trung tâm chính trị, văn hóa và kinh tế quan trọng."
    )
    
    my_doc = Document(
        content=sample_text,
        doc_id="DOC_001",
        title="Giới thiệu Hà Nội",
        source="wikipedia"
    )

    print("=== DỮ LIỆU GỐC ===")
    print(f"ID: {my_doc.doc_id}")
    print(f"Thời gian tạo: {my_doc.created_at}")
    print(f"Độ dài tài liệu: {len(my_doc.content)} ký tự\n")

    # 2. Thực hiện chia chunk (ví dụ chunk_size = 80, overlap = 20)
    result_chunks = chunk_document(my_doc, chunk_size=80, overlap=20)

    # 3. In kết quả các chunk ra màn hình
    print(f"=== KẾT QUẢ PHÂN CHUNK (Tổng số: {len(result_chunks)} chunks) ===")
    for chunk in result_chunks:
        print(f"[{chunk.chunk_id}] (Index {chunk.chunk_index}):")
        print(f"  - Nội dung : \"{chunk.content}\"")
        print(f"  - Vị trí    : {chunk.metadata['start_char']} -> {chunk.metadata['end_char']}")
        print("-" * 50)
