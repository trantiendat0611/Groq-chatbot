"""Backend API cho Groq Assistant — kiến trúc client/server chuyên nghiệp.

- REST + SSE streaming (giao diện web, mobile hay CLI đều dùng chung API này)
- Xác thực bằng Bearer token, dữ liệu cô lập theo từng người dùng
- Rate limiting chống lạm dụng, logging từng request, theo dõi token

Chạy:  uvicorn api:app --host 0.0.0.0 --port 8000
"""

import json
import logging
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import auth, memory, usage, user_memory, vector_store
from src.agent import AgentService
from src.chat_service import ChatService, ChatServiceError, GenerationSettings
from src.config import PROJECT_ROOT
from src.context import estimate_tokens
from src.embeddings import get_default_embedder, get_ready_embedder
from src.prompts import AGENT_PROMPT_SUFFIX, SYSTEM_PROMPT
from src.rag import build_chunks, format_rag_context, is_supported_file
from src.tools import (
    Tool,
    make_calculator_tool,
    make_document_search_tool,
    make_remember_tool,
    make_time_tool,
    make_web_search_tool,
)


logger = logging.getLogger("groq_assistant.api")

WEB_DIR = PROJECT_ROOT / "web"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB mỗi file

RATE_LIMIT_CHAT_PER_MINUTE = 20
RATE_LIMIT_AUTH_PER_MINUTE = 10
RATE_LIMIT_UPLOAD_PER_MINUTE = 6

TOOL_LABELS = {
    "calculator": "Máy tính",
    "get_current_time": "Xem giờ hiện tại",
    "web_search": "Tìm kiếm web",
    "search_documents": "Tra cứu tài liệu",
    "remember": "Ghi nhớ",
}


# ---------------------------------------------------------------------------
# Rate limiting: sliding window trong bộ nhớ, theo user hoặc IP.
# ---------------------------------------------------------------------------

_rate_buckets: dict[str, deque] = defaultdict(deque)


def check_rate_limit(key: str, limit: int, window_seconds: float = 60.0) -> None:
    now = time.monotonic()
    bucket = _rate_buckets[key]
    while bucket and bucket[0] <= now - window_seconds:
        bucket.popleft()
    if len(bucket) >= limit:
        raise HTTPException(
            status_code=429,
            detail="Bạn thao tác quá nhanh. Hãy đợi một lát rồi thử lại.",
        )
    bucket.append(now)


# ---------------------------------------------------------------------------
# App + middleware
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    memory.init_database()
    auth.init_auth_tables()
    usage.init_usage_table()
    vector_store.init_store()
    user_memory.init_memory_store()
    removed = auth.cleanup_expired_sessions()
    if removed:
        logger.info("Đã dọn %d phiên đăng nhập hết hạn.", removed)
    logger.info("Groq Assistant API sẵn sàng.")
    yield


app = FastAPI(
    title="Groq Assistant API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = uuid.uuid4().hex[:8]
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if request.url.path.startswith("/api/"):
        logger.info(
            "[%s] %s %s -> %d (%.0f ms)",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
    response.headers["X-Request-Id"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_chat_service(request: Request) -> ChatService:
    service = getattr(request.app.state, "chat_service", None)
    if service is None:
        service = ChatService()
        request.app.state.chat_service = service
    return service


def get_current_user(authorization: str = Header(default="")) -> auth.User:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Thiếu token đăng nhập.")
    user = auth.get_user_by_token(authorization.removeprefix("Bearer ").strip())
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Phiên đăng nhập không hợp lệ hoặc đã hết hạn.",
        )
    return user


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class CredentialsRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    conversation_id: int | None = None
    agent_mode: bool = True
    use_documents: bool = True
    use_memory: bool = True
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=1.5)
    max_tokens: int | None = Field(default=None, ge=64, le=4096)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def make_title(message: str) -> str:
    title = " ".join(message.strip().split())
    if len(title) > 36:
        return f"{title[:33]}..."
    return title or "Chat mới"


def resolve_settings(chat_service: ChatService, body: ChatRequest) -> GenerationSettings:
    defaults = chat_service.default_settings()
    return GenerationSettings(
        model=(body.model or defaults.model).strip() or defaults.model,
        temperature=body.temperature if body.temperature is not None else defaults.temperature,
        max_tokens=body.max_tokens if body.max_tokens is not None else defaults.max_tokens,
    )


def build_user_tools(
    user_id: int,
    include_documents: bool,
    include_memory: bool,
    top_k: int = 4,
) -> list[Tool]:
    tools = [
        make_calculator_tool(),
        make_time_tool(),
        make_web_search_tool(),
    ]

    if include_documents and vector_store.count_chunks(user_id=user_id) > 0:
        def document_search(query: str, k: int) -> str:
            results, _method = vector_store.search(
                query, top_k=k, embedder=get_default_embedder(), user_id=user_id
            )
            if not results:
                return "Không tìm thấy đoạn nào liên quan trong tài liệu đã nạp."
            return format_rag_context(results)

        tools.append(make_document_search_tool(document_search, top_k=top_k))

    if include_memory:
        def save_fact(fact: str) -> bool:
            embedder = get_ready_embedder()
            embedding = embedder.embed_query(fact) if embedder else None
            return (
                user_memory.add_memory(fact, embedding=embedding, user_id=user_id)
                is not None
            )

        tools.append(make_remember_tool(save_fact))

    return tools


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------


@app.post("/api/auth/register")
def register(body: CredentialsRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(f"auth:{client_ip}", RATE_LIMIT_AUTH_PER_MINUTE)

    try:
        user = auth.create_user(body.username, body.password)
    except auth.AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    token = auth.create_session(user.id)
    logger.info("Đăng ký tài khoản mới: %s", user.username)
    return {"token": token, "username": user.username}


@app.post("/api/auth/login")
def login(body: CredentialsRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(f"auth:{client_ip}", RATE_LIMIT_AUTH_PER_MINUTE)

    user = auth.verify_user(body.username, body.password)
    if user is None:
        raise HTTPException(
            status_code=401, detail="Sai tên đăng nhập hoặc mật khẩu."
        )

    token = auth.create_session(user.id)
    return {"token": token, "username": user.username}


@app.post("/api/auth/logout")
def logout(authorization: str = Header(default="")):
    if authorization.startswith("Bearer "):
        auth.revoke_session(authorization.removeprefix("Bearer ").strip())
    return {"ok": True}


@app.get("/api/auth/me")
def me(user: auth.User = Depends(get_current_user)):
    return {"username": user.username, "created_at": user.created_at}


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


@app.get("/api/conversations")
def get_conversations(user: auth.User = Depends(get_current_user)):
    return [
        {
            "id": conversation.id,
            "title": conversation.title,
            "updated_at": conversation.updated_at,
        }
        for conversation in memory.list_chats(user_id=user.id)
    ]


@app.post("/api/conversations", status_code=201)
def create_conversation(user: auth.User = Depends(get_current_user)):
    conversation_id = memory.create_chat(user_id=user.id)
    return {"id": conversation_id, "title": "Chat mới"}


def _require_owned_conversation(conversation_id: int, user: auth.User) -> None:
    if not memory.chat_exists(conversation_id, user_id=user.id):
        raise HTTPException(status_code=404, detail="Không tìm thấy cuộc trò chuyện.")


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: int, user: auth.User = Depends(get_current_user)):
    _require_owned_conversation(conversation_id, user)
    return {
        "id": conversation_id,
        "title": memory.get_chat_title(conversation_id, user_id=user.id),
        "messages": memory.get_chat_messages(conversation_id),
    }


@app.patch("/api/conversations/{conversation_id}")
def rename_conversation(
    conversation_id: int,
    body: RenameRequest,
    user: auth.User = Depends(get_current_user),
):
    _require_owned_conversation(conversation_id, user)
    memory.rename_chat(conversation_id, body.title.strip(), user_id=user.id)
    return {"ok": True}


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: int, user: auth.User = Depends(get_current_user)):
    _require_owned_conversation(conversation_id, user)
    memory.delete_chat(conversation_id, user_id=user.id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Chat (SSE streaming)
# ---------------------------------------------------------------------------


@app.post("/api/chat")
def chat(
    body: ChatRequest,
    user: auth.User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
):
    check_rate_limit(f"chat:{user.id}", RATE_LIMIT_CHAT_PER_MINUTE)

    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Tin nhắn trống.")

    if body.conversation_id is None:
        conversation_id = memory.create_chat(user_id=user.id)
    else:
        conversation_id = body.conversation_id
        _require_owned_conversation(conversation_id, user)

    history = memory.get_chat_messages(conversation_id)
    messages_for_llm = [*history, {"role": "user", "content": message}]
    settings = resolve_settings(chat_service, body)

    system_prompt = SYSTEM_PROMPT.strip()

    if body.use_memory:
        embedder = get_ready_embedder()
        recall_embedding = embedder.embed_query(message) if embedder else None
        recalled = user_memory.search_memories(
            query_embedding=recall_embedding, top_k=5, user_id=user.id
        )
        memories_block = user_memory.format_memories_block(recalled)
        if memories_block:
            system_prompt = f"{system_prompt}\n\n{memories_block}"

    tools: list[Tool] = []
    if body.agent_mode:
        system_prompt = f"{system_prompt}\n\n{AGENT_PROMPT_SUFFIX.strip()}"
        tools = build_user_tools(
            user.id,
            include_documents=body.use_documents,
            include_memory=body.use_memory,
        )

    agent_service = AgentService(chat_service, tools=tools)

    def event_stream():
        answer_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        got_real_usage = False

        try:
            for event in agent_service.run_stream(messages_for_llm, system_prompt, settings):
                if event.type == "text":
                    answer_parts.append(event.content)
                    yield sse_event("text", {"content": event.content})
                elif event.type == "tool_call":
                    yield sse_event(
                        "tool_call",
                        {
                            "tool": event.tool_name,
                            "label": TOOL_LABELS.get(event.tool_name, event.tool_name),
                            "arguments": event.arguments,
                        },
                    )
                elif event.type == "tool_result":
                    yield sse_event(
                        "tool_result",
                        {
                            "tool": event.tool_name,
                            "label": TOOL_LABELS.get(event.tool_name, event.tool_name),
                            "content": event.content[:1500],
                        },
                    )
                elif event.type == "usage":
                    got_real_usage = True
                    prompt_tokens += event.prompt_tokens
                    completion_tokens += event.completion_tokens
        except ChatServiceError as exc:
            yield sse_event("error", {"message": str(exc)})
            return
        except Exception:
            logger.exception("Lỗi không mong đợi trong lượt chat.")
            yield sse_event(
                "error", {"message": "Có lỗi không mong đợi. Hãy thử lại."}
            )
            return

        answer = "".join(answer_parts).strip()
        if not answer:
            answer = "(Không nhận được nội dung trả lời. Hãy thử lại.)"

        title = memory.get_chat_title(conversation_id, user_id=user.id)
        if title == "Chat mới":
            title = make_title(message)
            memory.rename_chat(conversation_id, title, user_id=user.id)

        memory.add_chat_message(conversation_id, "user", message)
        memory.add_chat_message(conversation_id, "assistant", answer)

        if not got_real_usage:
            prompt_tokens = sum(
                estimate_tokens(item.get("content", "")) for item in messages_for_llm
            ) + estimate_tokens(system_prompt)
            completion_tokens = estimate_tokens(answer)
        usage.record_usage(
            model=settings.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            user_id=user.id,
            estimated=not got_real_usage,
        )

        yield sse_event(
            "done",
            {
                "conversation_id": conversation_id,
                "title": title,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Documents (RAG)
# ---------------------------------------------------------------------------


@app.get("/api/documents")
def get_documents(user: auth.User = Depends(get_current_user)):
    return [
        {"source": source, "chunks": chunk_count}
        for source, chunk_count in vector_store.list_sources(user_id=user.id)
    ]


@app.post("/api/documents", status_code=201)
async def upload_documents(
    files: list[UploadFile] = File(...),
    user: auth.User = Depends(get_current_user),
):
    check_rate_limit(f"upload:{user.id}", RATE_LIMIT_UPLOAD_PER_MINUTE)

    payload: list[tuple[str, bytes]] = []
    for file in files:
        filename = file.filename or "khong_ten"
        if not is_supported_file(filename):
            raise HTTPException(
                status_code=400,
                detail=f"File '{filename}' không được hỗ trợ (chỉ nhận .txt, .md, .pdf).",
            )
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File '{filename}' vượt quá giới hạn 10MB.",
            )
        payload.append((filename, data))

    chunks = build_chunks(payload)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Không trích xuất được nội dung từ tài liệu đã chọn.",
        )

    embedder = get_default_embedder()
    embeddings_list = (
        embedder.embed_texts([chunk.text for chunk in chunks]) if embedder else None
    )

    # Nạp lại cùng tên file thì thay thế bản cũ.
    for filename, _data in payload:
        vector_store.delete_source(filename, user_id=user.id)
    vector_store.add_chunks(chunks, embeddings_list, user_id=user.id)

    return {
        "files": len(payload),
        "chunks": len(chunks),
        "semantic": embedder is not None,
    }


@app.delete("/api/documents")
def delete_documents(
    source: str | None = None,
    user: auth.User = Depends(get_current_user),
):
    if source:
        vector_store.delete_source(source, user_id=user.id)
    else:
        vector_store.clear_store(user_id=user.id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Memories
# ---------------------------------------------------------------------------


@app.get("/api/memories")
def get_memories(user: auth.User = Depends(get_current_user)):
    return [
        {"id": item.id, "content": item.content, "created_at": item.created_at}
        for item in user_memory.list_memories(user_id=user.id)
    ]


@app.delete("/api/memories/{memory_id}")
def delete_memory(memory_id: int, user: auth.User = Depends(get_current_user)):
    user_memory.delete_memory(memory_id, user_id=user.id)
    return {"ok": True}


@app.delete("/api/memories")
def clear_memories(user: auth.User = Depends(get_current_user)):
    user_memory.clear_memories(user_id=user.id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Usage + health
# ---------------------------------------------------------------------------


@app.get("/api/usage")
def get_usage(days: int = 30, user: auth.User = Depends(get_current_user)):
    summary = usage.usage_summary(user_id=user.id, days=max(1, min(days, 365)))
    return {
        "days": max(1, min(days, 365)),
        "requests": summary.total_requests,
        "prompt_tokens": summary.prompt_tokens,
        "completion_tokens": summary.completion_tokens,
        "total_tokens": summary.total_tokens,
    }


@app.get("/api/health")
def health(request: Request):
    model = None
    try:
        model = get_chat_service(request).config.model
    except Exception:
        pass  # thiếu API key vẫn báo sống, nhưng model = null
    return {"status": "ok", "model": model}


# Frontend tĩnh — mount cuối cùng để /api/* luôn được ưu tiên.
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
