# Groq Chatbot

Chatbot dùng Groq API, gồm bản CLI và giao diện Streamlit.

## Cấu trúc project

```text
Groq-chatbot/
|-- app.py                  # Giao diện Streamlit
|-- chatbot.py              # Chatbot chạy trong terminal
|-- src/
|   |-- config.py           # Đọc cấu hình từ .env
|   |-- prompts.py          # System prompt mặc định
|   |-- groq_client.py      # Tạo Groq client
|   |-- chat_service.py     # Logic tạo câu trả lời và streaming
|   |-- memory.py           # Lưu và đọc lịch sử chat bằng SQLite
|   `-- rag.py              # Đọc tài liệu, chunk và retrieve context
|-- data/
|   `-- chats.db            # Database tự tạo khi app chạy
|-- .env                    # API key thật, không commit
|-- .env.example            # File mẫu cấu hình
|-- requirements.txt
`-- run-ui.cmd
```

## Cài thư viện

Chạy từ thư mục `AI Agent`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\Groq-chatbot\requirements.txt
```

## Cấu hình API key

Mở file `.env` trong thư mục `Groq-chatbot` và điền API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Bạn có thể tùy chỉnh cấu hình mặc định:

```env
GROQ_MODEL=llama-3.1-8b-instant
GROQ_TEMPERATURE=0.7
GROQ_MAX_TOKENS=800
```

## Chạy bản CLI

```powershell
cd .\Groq-chatbot
..\.venv\Scripts\python.exe .\chatbot.py
```

Gõ `thoát`, `exit` hoặc `quit` để dừng.

## Chạy giao diện chat

Chạy nhanh bằng file:

```powershell
.\run-ui.cmd
```

Hoặc chạy trực tiếp:

```powershell
cd .\Groq-chatbot
..\.venv\Scripts\python.exe -m streamlit run .\app.py
```

Mở trình duyệt tại:

```text
http://localhost:8501
```

## Cài đặt trong giao diện

Trong sidebar, mở mục `Cài đặt` để chỉnh:

- `Model`: tên model Groq muốn dùng
- `Temperature`: độ ổn định/sáng tạo của câu trả lời
- `Max tokens`: giới hạn độ dài câu trả lời
- `System prompt`: chỉ dẫn nền cho trợ lý AI

Các cài đặt này áp dụng trong phiên giao diện hiện tại. File `.env` vẫn là nơi lưu cấu hình mặc định.

## Hỏi đáp với tài liệu riêng

Trong sidebar, mở mục `Tài liệu RAG`:

1. Upload file `.txt`, `.md` hoặc `.pdf`.
2. Bấm `Nạp tài liệu`.
3. Bật `Dùng tài liệu khi trả lời`.
4. Đặt câu hỏi trong ô chat như bình thường.

Khi có tài liệu phù hợp, chatbot sẽ lấy các đoạn liên quan đưa vào prompt và yêu cầu model trả lời kèm nguồn dạng `[1]`, `[2]`.

Phiên bản RAG hiện tại là bản cơ bản:

- Hỗ trợ TXT, Markdown, PDF
- Chia tài liệu thành chunks
- Retrieve bằng scoring từ khóa có IDF
- Chưa dùng embedding/vector database
- Tài liệu đã nạp được lưu trong session hiện tại, chưa lưu vĩnh viễn

## Tính năng hiện có

- Chat với Groq API
- Giao diện đen trắng bằng Streamlit
- Streaming response trong giao diện chat
- Tùy chỉnh model, temperature, max tokens và system prompt
- Upload tài liệu và hỏi đáp bằng RAG cơ bản
- Tạo chat mới
- Xem lại chat cũ sau khi tắt/mở app
- Tự đặt tiêu đề chat theo câu hỏi đầu tiên
- Xóa chat hiện tại
- Lưu lịch sử chat vào SQLite tại `data/chats.db`

Bạn không cần cài thêm thư viện cho SQLite vì Python đã có sẵn `sqlite3`.
