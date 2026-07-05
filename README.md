# Groq Assistant — AI Agent

Hệ thống trợ lý AI Agent hoàn chỉnh dùng Groq API, kiến trúc client/server:

- **Backend FastAPI** (`api.py`): REST + SSE streaming, đăng nhập đa người dùng,
  rate limiting, theo dõi token, dữ liệu cô lập theo từng người.
- **Giao diện web** (`web/`): SPA thuần HTML/CSS/JS, streaming thời gian thực,
  hiển thị tiến trình agent dùng công cụ.
- **Giao diện Streamlit** (`app.py`): bản chạy nhanh một người dùng.
- **CLI** (`chatbot.py`): chat trong terminal.

Ở chế độ Agent, AI tự quyết định dùng công cụ: máy tính, tìm kiếm web, tra cứu
tài liệu đã nạp (RAG ngữ nghĩa) và ghi nhớ thông tin về bạn (trí nhớ dài hạn).

## Kiến trúc

```text
Trình duyệt (web/)  ──HTTP/SSE──>  api.py (FastAPI)
Streamlit (app.py)  ──────────┐         │
CLI (chatbot.py)    ──────────┤         │
                              v         v
                        src/ (lõi dùng chung)
        agent.py · tools.py · chat_service.py · context.py
        vector_store.py · user_memory.py · memory.py · auth.py · usage.py
                              │
                              v
                 SQLite (data/chats.db, data/knowledge.db)
```

## Cấu trúc project

```text
Groq-chatbot/
|-- api.py                  # Backend API: REST + SSE, auth, rate limit
|-- app.py                  # Giao diện Streamlit (chạy local 1 người dùng)
|-- chatbot.py              # Chatbot chạy trong terminal
|-- web/                    # Giao diện web (đi kèm backend API)
|   |-- index.html
|   |-- style.css
|   `-- app.js
|-- src/
|   |-- config.py           # Đọc cấu hình từ .env
|   |-- prompts.py          # System prompt + chỉ dẫn agent
|   |-- groq_client.py      # Tạo Groq client
|   |-- chat_types.py       # Kiểu dữ liệu dùng chung (ChatMessage)
|   |-- context.py          # Ước lượng token, cắt lịch sử theo ngân sách
|   |-- chat_service.py     # Logic tạo câu trả lời, streaming, retry
|   |-- agent.py            # Vòng lặp agent: model tự gọi công cụ nhiều bước
|   |-- tools.py            # Khung Tool + máy tính, web search, giờ, ghi nhớ...
|   |-- embeddings.py       # Embedding đa ngôn ngữ (fastembed, hỗ trợ tiếng Việt)
|   |-- vector_store.py     # Kho tri thức bền vững: chunks + embedding trong SQLite
|   |-- user_memory.py      # Trí nhớ dài hạn về người dùng, xuyên các cuộc chat
|   |-- memory.py           # Lưu và đọc lịch sử chat bằng SQLite
|   |-- auth.py             # Tài khoản + phiên đăng nhập (PBKDF2, token băm)
|   |-- usage.py            # Theo dõi mức sử dụng token
|   `-- rag.py              # Đọc tài liệu, chunk và retrieve context
|-- tests/                  # Bộ test pytest (130 test)
|-- data/
|   |-- chats.db            # Lịch sử chat + tài khoản + usage (tự tạo)
|   `-- knowledge.db        # Kho tri thức + trí nhớ dài hạn (tự tạo)
|-- Dockerfile              # Đóng gói backend + web
|-- docker-compose.yml      # Chạy production một lệnh
|-- .env                    # API key thật, không commit
|-- .env.example            # File mẫu cấu hình
|-- requirements.txt
|-- run-api.cmd             # Chạy backend API + giao diện web
`-- run-ui.cmd              # Chạy giao diện Streamlit
```

## Cài thư viện

Chạy từ thư mục `Groq-chatbot` (venv nằm ngay trong project):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
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
GROQ_CONTEXT_TOKENS=6000
GROQ_MAX_RETRIES=3
```

- `GROQ_CONTEXT_TOKENS`: ngân sách token cho lịch sử hội thoại. Khi chat quá dài, các message cũ nhất sẽ được cắt bớt trước khi gửi lên API (lịch sử đầy đủ vẫn lưu trong SQLite).
- `GROQ_MAX_RETRIES`: số lần tự thử lại khi gặp lỗi tạm thời (rate limit, mất mạng, lỗi server Groq) với exponential backoff.
- `GROQ_AGENT_MAX_STEPS`: số bước tối đa agent được lặp trong một lượt (mỗi bước có thể gọi công cụ; bước cuối buộc trả lời).
- `EMBEDDING_MODEL`: model embedding cho tìm kiếm ngữ nghĩa. Mặc định là bản đa ngôn ngữ hiểu tiếng Việt tốt.

Gợi ý: model mặc định `llama-3.1-8b-instant` nhanh và rẻ nhưng gọi công cụ ở mức khá.
Nếu muốn agent thông minh hơn, đặt `GROQ_MODEL=llama-3.3-70b-versatile`.

## Chạy backend API + giao diện web (khuyến nghị)

Đây là chế độ đầy đủ nhất: đăng nhập đa người dùng, SSE streaming, rate limiting.

Chạy nhanh bằng file:

```powershell
.\run-api.cmd
```

Hoặc chạy trực tiếp:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Mở trình duyệt tại `http://localhost:8000`, đăng ký tài khoản rồi chat.
Tài liệu API tự sinh (Swagger UI) tại `http://localhost:8000/api/docs`.

Các endpoint chính:

| Endpoint | Tác dụng |
|---|---|
| `POST /api/auth/register`, `/login`, `/logout` | Tài khoản + phiên đăng nhập |
| `POST /api/chat` | Chat (trả về SSE stream: text, tool_call, tool_result, done) |
| `GET/POST/PATCH/DELETE /api/conversations` | Quản lý hội thoại |
| `GET/POST/DELETE /api/documents` | Upload/tra cứu tài liệu RAG |
| `GET/DELETE /api/memories` | Xem/xóa trí nhớ dài hạn |
| `GET /api/usage` | Thống kê token 30 ngày |
| `GET /api/health` | Health check |

## Chạy bằng Docker

```powershell
docker compose up --build
```

- Đọc API key từ `.env`, mở cổng `8000`.
- Database mount ra `./data` — dữ liệu sống ngoài container.
- Model embedding cache trong volume `fastembed-cache` — chỉ tải một lần.
- Có sẵn healthcheck cho orchestrator.

## Chạy bản CLI

```powershell
.\.venv\Scripts\python.exe .\chatbot.py
```

Gõ `thoát`, `exit` hoặc `quit` để dừng.

## Chạy giao diện Streamlit (bản local một người dùng)

Chạy nhanh bằng file:

```powershell
.\run-ui.cmd
```

Hoặc chạy trực tiếp:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
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

## Chế độ Agent

Bật/tắt trong sidebar, mục `Cài đặt` > `Chế độ Agent (dùng công cụ)`. Khi bật,
AI chạy theo vòng lặp agent: suy luận, tự quyết định gọi công cụ, đọc kết quả
rồi suy luận tiếp cho đến khi chốt câu trả lời. Tiến trình dùng công cụ hiển thị
ngay trong khung chat.

Các công cụ hiện có:

| Công cụ | Tác dụng |
|---|---|
| `calculator` | Tính toán chính xác (an toàn, không chạy code tùy ý) |
| `get_current_time` | Xem ngày giờ hiện tại |
| `web_search` | Tìm kiếm Internet qua DuckDuckGo, không cần API key |
| `search_documents` | Tra cứu tài liệu bạn đã nạp (RAG-as-a-tool) |
| `remember` | Ghi một sự thật về bạn vào trí nhớ dài hạn |

## Hỏi đáp với tài liệu riêng (RAG)

Trong sidebar, mở mục `Tài liệu RAG`:

1. Upload file `.txt`, `.md` hoặc `.pdf`.
2. Bấm `Nạp tài liệu` (lần đầu sẽ tải model embedding ~470MB, chỉ một lần).
3. Bật `Dùng tài liệu khi trả lời`.
4. Đặt câu hỏi trong ô chat như bình thường.

RAG phiên bản hiện tại:

- Hỗ trợ TXT, Markdown, PDF
- Tìm kiếm ngữ nghĩa bằng embedding đa ngôn ngữ (hiểu tiếng Việt), cosine similarity
- Tự fallback về tìm kiếm từ khóa có IDF nếu embedding không khả dụng
- Tài liệu lưu bền vững trong `data/knowledge.db` — không mất khi tắt app
- Nạp lại cùng tên file sẽ thay thế bản cũ
- Chế độ Agent: AI tự tra tài liệu qua công cụ `search_documents` khi thấy cần
- Chế độ thường: các đoạn liên quan được chèn thẳng vào prompt, có ghi nguồn

## Trí nhớ dài hạn

Giống tính năng Memory của ChatGPT/Claude: khi bạn chia sẻ thông tin đáng nhớ
(tên, nghề nghiệp, sở thích, dự án...) hoặc nói "hãy nhớ rằng...", agent tự lưu
vào trí nhớ. Các cuộc chat sau, AI tự nhớ lại những điều liên quan.

- Xem/xóa từng mục trong sidebar, mục `Trí nhớ dài hạn`
- Lưu trong `data/knowledge.db`, xuyên suốt mọi cuộc chat
- Tắt được bằng checkbox `Dùng trí nhớ dài hạn`

## Chạy test

```powershell
cd .\Groq-chatbot
.\.venv\Scripts\python.exe -m pytest
```

Bộ test (130 test) bao phủ: backend API và cô lập đa người dùng (`api.py`, `tests/test_api.py`, `tests/test_multiuser.py`), xác thực (`src/auth.py`), theo dõi token (`src/usage.py`), vòng lặp agent và tool calling (`src/agent.py`, `src/tools.py`), kho tri thức và trí nhớ dài hạn (`src/vector_store.py`, `src/user_memory.py`), cắt cửa sổ ngữ cảnh (`src/context.py`), retry và xử lý lỗi API (`src/chat_service.py`), lịch sử chat SQLite (`src/memory.py`) và RAG (`src/rag.py`). Test dùng client giả nên không cần API key, không tốn token, không gọi mạng.

## Bảo mật

- Mật khẩu băm PBKDF2-HMAC-SHA256, 600.000 vòng, salt riêng từng user — không lưu mật khẩu gốc
- Token phiên 256-bit ngẫu nhiên, chỉ lưu bản băm SHA-256 trong DB, hết hạn sau 30 ngày, thu hồi được
- So sánh mật khẩu bằng `hmac.compare_digest` chống timing attack
- Dữ liệu (hội thoại, tài liệu, trí nhớ, usage) cô lập tuyệt đối theo user — kiểm tra quyền sở hữu ở mọi endpoint
- Rate limiting: 20 lượt chat/phút/user, 10 lượt đăng nhập/phút/IP, 6 lượt upload/phút/user
- Giới hạn upload 10MB/file, chỉ nhận `.txt`, `.md`, `.pdf`
- Máy tính của agent duyệt AST, không `eval()` — không thể chạy code tùy ý
- Prompt chống injection: kết quả công cụ/tài liệu là dữ liệu, không phải mệnh lệnh
- SQL tham số hóa toàn bộ, chống SQL injection

## Tính năng hiện có

- **Kiến trúc client/server**: backend FastAPI (REST + SSE) + giao diện web riêng, Streamlit và CLI dùng chung lõi `src/`
- **Đăng nhập đa người dùng**: mỗi người có lịch sử chat, tài liệu, trí nhớ, thống kê token riêng
- **Chế độ Agent**: AI tự quyết định gọi công cụ nhiều bước (máy tính, web search, tra tài liệu, ghi nhớ), hiển thị tiến trình từng bước
- **Trí nhớ dài hạn** xuyên các cuộc chat, tự ghi nhớ và nhớ lại (giống Memory của ChatGPT/Claude)
- **RAG ngữ nghĩa**: embedding đa ngôn ngữ + vector store bền vững trong SQLite, fallback từ khóa
- **Theo dõi token**: đọc usage thật từ Groq trả về, thống kê 30 ngày theo user
- **Docker**: đóng gói + docker-compose, healthcheck, volume cho data và model cache
- Quản lý cửa sổ ngữ cảnh: tự cắt bớt message cũ khi hội thoại quá dài
- Tự thử lại (retry + exponential backoff, tôn trọng header Retry-After) khi Groq gặp lỗi tạm thời
- Thông báo lỗi tiếng Việt dễ hiểu cho từng loại lỗi API
- Logging từng request kèm request-id và thời gian xử lý
- Tùy chỉnh model, temperature, max tokens và system prompt
- Tạo chat mới, xem lại chat cũ, tự đặt tiêu đề, xóa chat

Bạn không cần cài thêm thư viện cho SQLite vì Python đã có sẵn `sqlite3`.
