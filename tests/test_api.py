"""Test tích hợp backend API: auth, hội thoại, chat SSE, tài liệu, trí nhớ."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api as api_module
from fakes import (
    FakeCompletions,
    make_config,
    make_fake_client,
    text_chunk,
    tool_call_chunk,
    usage_chunk,
)
from src import auth, memory, usage, user_memory, vector_store
from src.chat_service import ChatService


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "DB_PATH", tmp_path / "chats.db")
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "chats.db")
    monkeypatch.setattr(usage, "DB_PATH", tmp_path / "chats.db")
    monkeypatch.setattr(vector_store, "DB_PATH", tmp_path / "knowledge.db")
    monkeypatch.setattr(user_memory, "DB_PATH", tmp_path / "knowledge.db")
    monkeypatch.setattr(auth, "PBKDF2_ITERATIONS", 1000)
    # Không bao giờ tải model embedding thật trong test.
    monkeypatch.setattr(api_module, "get_default_embedder", lambda: None)
    monkeypatch.setattr(api_module, "get_ready_embedder", lambda: None)
    api_module._rate_buckets.clear()
    api_module.app.state.chat_service = None
    yield


@pytest.fixture
def client():
    with TestClient(api_module.app) as test_client:
        yield test_client


def install_fake_chat_service(streams) -> ChatService:
    service = ChatService(client=make_fake_client(streams), config=make_config())
    service._sleep = lambda seconds: None
    api_module.app.state.chat_service = service
    return service


def register(client, username="dat_uit", password="matkhau123") -> dict:
    response = client.post(
        "/api/auth/register", json={"username": username, "password": password}
    )
    assert response.status_code == 200, response.text
    return response.json()


def auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        event_type, data_text = None, ""
        for line in block.split("\n"):
            if line.startswith("event: "):
                event_type = line.removeprefix("event: ").strip()
            elif line.startswith("data: "):
                data_text += line.removeprefix("data: ")
        if event_type and data_text:
            events.append((event_type, json.loads(data_text)))
    return events


# --- Auth ---


def test_register_login_me_flow(client):
    data = register(client)
    assert data["username"] == "dat_uit"
    assert data["token"]

    login = client.post(
        "/api/auth/login", json={"username": "dat_uit", "password": "matkhau123"}
    )
    assert login.status_code == 200

    me = client.get("/api/auth/me", headers=auth_header(login.json()["token"]))
    assert me.status_code == 200
    assert me.json()["username"] == "dat_uit"


def test_login_wrong_password(client):
    register(client)
    response = client.post(
        "/api/auth/login", json={"username": "dat_uit", "password": "sai_roi_nha"}
    )
    assert response.status_code == 401


def test_register_invalid_username(client):
    response = client.post(
        "/api/auth/register", json={"username": "a", "password": "matkhau123"}
    )
    assert response.status_code == 400


def test_protected_endpoint_requires_token(client):
    assert client.get("/api/conversations").status_code == 401
    assert client.get(
        "/api/conversations", headers=auth_header("token-gia")
    ).status_code == 401


def test_logout_revokes_token(client):
    token = register(client)["token"]
    client.post("/api/auth/logout", headers=auth_header(token))
    assert client.get("/api/auth/me", headers=auth_header(token)).status_code == 401


# --- Conversations ---


def test_conversation_crud(client):
    token = register(client)["token"]
    headers = auth_header(token)

    created = client.post("/api/conversations", headers=headers)
    assert created.status_code == 201
    conversation_id = created.json()["id"]

    listing = client.get("/api/conversations", headers=headers).json()
    assert [c["id"] for c in listing] == [conversation_id]

    renamed = client.patch(
        f"/api/conversations/{conversation_id}",
        json={"title": "Tên mới"},
        headers=headers,
    )
    assert renamed.status_code == 200

    detail = client.get(f"/api/conversations/{conversation_id}", headers=headers).json()
    assert detail["title"] == "Tên mới"
    assert detail["messages"] == []

    deleted = client.delete(f"/api/conversations/{conversation_id}", headers=headers)
    assert deleted.status_code == 200
    assert client.get("/api/conversations", headers=headers).json() == []


def test_users_cannot_see_each_others_conversations(client):
    token_a = register(client, "user_a")["token"]
    token_b = register(client, "user_b")["token"]

    conversation_id = client.post(
        "/api/conversations", headers=auth_header(token_a)
    ).json()["id"]

    response = client.get(
        f"/api/conversations/{conversation_id}", headers=auth_header(token_b)
    )
    assert response.status_code == 404

    # User B cũng không xóa/đổi tên được.
    assert client.delete(
        f"/api/conversations/{conversation_id}", headers=auth_header(token_b)
    ).status_code == 404


# --- Chat SSE ---


def test_chat_streams_text_and_persists(client):
    install_fake_chat_service([[text_chunk("Xin "), text_chunk("chào!"), usage_chunk(120, 8)]])
    token = register(client)["token"]
    headers = auth_header(token)

    response = client.post(
        "/api/chat",
        json={"message": "chào bạn", "agent_mode": False},
        headers=headers,
    )

    assert response.status_code == 200
    events = parse_sse(response.text)
    types = [event_type for event_type, _ in events]
    # "start" phải đến trước mọi thứ để client biết conversation_id ngay lập tức.
    assert types[0] == "start"
    assert "text" in types
    assert types[-1] == "done"
    assert events[0][1]["conversation_id"] == events[-1][1]["conversation_id"]

    done = events[-1][1]
    conversation_id = done["conversation_id"]
    assert done["title"] == "chào bạn"
    assert done["prompt_tokens"] == 120
    assert done["completion_tokens"] == 8

    detail = client.get(f"/api/conversations/{conversation_id}", headers=headers).json()
    assert detail["messages"] == [
        {"role": "user", "content": "chào bạn"},
        {"role": "assistant", "content": "Xin chào!"},
    ]

    summary = client.get("/api/usage", headers=headers).json()
    assert summary["requests"] == 1
    assert summary["total_tokens"] == 128


def test_chat_with_tool_call_emits_tool_events(client):
    install_fake_chat_service(
        [
            [tool_call_chunk("call_1", "calculator", '{"expression": "6*7"}')],
            [text_chunk("Kết quả là 42.")],
        ]
    )
    token = register(client)["token"]

    response = client.post(
        "/api/chat",
        json={"message": "tính 6*7"},
        headers=auth_header(token),
    )

    events = parse_sse(response.text)
    types = [event_type for event_type, _ in events]
    assert "tool_call" in types
    assert "tool_result" in types

    tool_call = next(data for event_type, data in events if event_type == "tool_call")
    assert tool_call["tool"] == "calculator"
    assert tool_call["label"] == "Máy tính"

    tool_result = next(data for event_type, data in events if event_type == "tool_result")
    assert tool_result["content"] == "42"


def test_question_saved_before_model_runs(client):
    """Câu hỏi phải nằm trong DB ngay từ đầu, trước khi model trả lời."""
    seen: dict[str, list] = {}

    def spy_stream(**kwargs):
        # Lúc model bắt đầu chạy, câu hỏi đã phải được lưu.
        seen["messages"] = memory.get_chat_messages(seen["conversation_id"])
        return iter([text_chunk("ok")])

    token = register(client)["token"]
    headers = auth_header(token)
    conversation_id = client.post("/api/conversations", headers=headers).json()["id"]
    seen["conversation_id"] = conversation_id

    service = ChatService(client=make_fake_client([]), config=make_config())
    service._sleep = lambda seconds: None
    service._create_completion = lambda kwargs: spy_stream(**kwargs)
    api_module.app.state.chat_service = service

    client.post(
        "/api/chat",
        json={"message": "xin chào", "conversation_id": conversation_id, "agent_mode": False},
        headers=headers,
    )

    assert seen["messages"] == [{"role": "user", "content": "xin chào"}]


def test_partial_answer_saved_when_client_disconnects(client):
    """Bấm 'Dừng' giữa chừng: câu hỏi và phần đã trả lời không được mất.

    Không dựa vào `finally` (Starlette bỏ rơi sync generator khi client ngắt),
    mà dựa vào việc ghi dần xuống DB trong lúc stream.
    """
    many_chunks = [text_chunk(f"tu{index} ") for index in range(200)]
    install_fake_chat_service([many_chunks])
    token = register(client)["token"]
    headers = auth_header(token)
    conversation_id = client.post("/api/conversations", headers=headers).json()["id"]

    with client.stream(
        "POST",
        "/api/chat",
        json={"message": "xin chào", "conversation_id": conversation_id, "agent_mode": False},
        headers=headers,
    ) as response:
        assert response.status_code == 200
        lines = response.iter_lines()
        for _ in range(6):  # đọc vài sự kiện rồi bỏ đi giữa chừng
            next(lines)

    detail = client.get(f"/api/conversations/{conversation_id}", headers=headers).json()

    assert detail["messages"][0] == {"role": "user", "content": "xin chào"}
    # Phần đã nhận được giữ lại (dù chưa đủ 200 từ).
    assert detail["messages"][1]["role"] == "assistant"
    assert detail["messages"][1]["content"].startswith("tu0")
    assert detail["title"] == "xin chào"  # tiêu đề cũng đã đặt


def test_title_set_even_if_generation_fails(client):
    """Model lỗi ngay: câu hỏi vẫn được lưu và chat vẫn có tiêu đề."""
    import httpx
    from groq import NotFoundError

    request = httpx.Request("POST", "https://api.groq.com/v1/chat/completions")
    error = NotFoundError(
        "khong co model", response=httpx.Response(404, request=request), body=None
    )

    completions = FakeCompletions([])
    completions.create = lambda **kwargs: (_ for _ in ()).throw(error)
    service = ChatService(
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        config=make_config(),
    )
    service._sleep = lambda seconds: None
    api_module.app.state.chat_service = service

    token = register(client)["token"]
    headers = auth_header(token)

    response = client.post(
        "/api/chat", json={"message": "câu hỏi hỏng", "agent_mode": False}, headers=headers
    )
    events = parse_sse(response.text)
    conversation_id = events[0][1]["conversation_id"]
    assert any(event_type == "error" for event_type, _ in events)

    detail = client.get(f"/api/conversations/{conversation_id}", headers=headers).json()
    assert detail["messages"] == [{"role": "user", "content": "câu hỏi hỏng"}]
    assert detail["title"] == "câu hỏi hỏng"


def test_answer_persisted_exactly_once(client):
    """persist_answer chạy ở cả nhánh thường lẫn finally -> không được lưu 2 lần."""
    install_fake_chat_service([[text_chunk("Một lần thôi")]])
    token = register(client)["token"]
    headers = auth_header(token)

    response = client.post(
        "/api/chat", json={"message": "hi", "agent_mode": False}, headers=headers
    )
    conversation_id = parse_sse(response.text)[-1][1]["conversation_id"]

    detail = client.get(f"/api/conversations/{conversation_id}", headers=headers).json()
    assert len(detail["messages"]) == 2  # đúng 1 user + 1 assistant

    summary = client.get("/api/usage", headers=headers).json()
    assert summary["requests"] == 1  # usage cũng chỉ ghi một lần


def test_chat_rejects_foreign_conversation(client):
    install_fake_chat_service([[text_chunk("ok")]])
    token_a = register(client, "user_a")["token"]
    token_b = register(client, "user_b")["token"]

    conversation_id = client.post(
        "/api/conversations", headers=auth_header(token_a)
    ).json()["id"]

    response = client.post(
        "/api/chat",
        json={"message": "xin chào", "conversation_id": conversation_id},
        headers=auth_header(token_b),
    )
    assert response.status_code == 404


def test_chat_rate_limit(client):
    install_fake_chat_service([[text_chunk("ok")] for _ in range(30)])
    token = register(client)["token"]
    headers = auth_header(token)

    last_status = None
    for _ in range(api_module.RATE_LIMIT_CHAT_PER_MINUTE + 1):
        last_status = client.post(
            "/api/chat", json={"message": "hi", "agent_mode": False}, headers=headers
        ).status_code

    assert last_status == 429


# --- Documents ---


def upload_file(client, headers, conversation_id=None, filename="notes.txt", content=None):
    content = content or "Groq siêu nhanh ".encode("utf-8") * 100
    data = {}
    if conversation_id is not None:
        data["conversation_id"] = str(conversation_id)
    return client.post(
        "/api/documents",
        files=[("files", (filename, content, "text/plain"))],
        data=data,
        headers=headers,
    )


def test_upload_list_delete_documents_in_conversation(client):
    token = register(client)["token"]
    headers = auth_header(token)
    conversation_id = client.post("/api/conversations", headers=headers).json()["id"]

    upload = upload_file(client, headers, conversation_id)
    assert upload.status_code == 201, upload.text
    assert upload.json()["chunks"] >= 1
    assert upload.json()["conversation_id"] == conversation_id

    listing = client.get(
        f"/api/documents?conversation_id={conversation_id}", headers=headers
    ).json()
    assert listing[0]["source"] == "notes.txt"

    client.delete(
        f"/api/documents?conversation_id={conversation_id}&source=notes.txt",
        headers=headers,
    )
    assert client.get(
        f"/api/documents?conversation_id={conversation_id}", headers=headers
    ).json() == []


def test_upload_without_conversation_creates_one(client):
    token = register(client)["token"]
    headers = auth_header(token)

    upload = upload_file(client, headers)
    assert upload.status_code == 201, upload.text
    conversation_id = upload.json()["conversation_id"]
    assert conversation_id is not None

    # Đoạn chat mới đã được tạo và chứa tài liệu.
    conversations = client.get("/api/conversations", headers=headers).json()
    assert [c["id"] for c in conversations] == [conversation_id]
    listing = client.get(
        f"/api/documents?conversation_id={conversation_id}", headers=headers
    ).json()
    assert listing[0]["source"] == "notes.txt"


def test_documents_isolated_between_conversations(client):
    token = register(client)["token"]
    headers = auth_header(token)
    conversation_a = client.post("/api/conversations", headers=headers).json()["id"]
    conversation_b = client.post("/api/conversations", headers=headers).json()["id"]

    upload_file(client, headers, conversation_a, filename="rieng_cua_a.txt")

    docs_a = client.get(
        f"/api/documents?conversation_id={conversation_a}", headers=headers
    ).json()
    docs_b = client.get(
        f"/api/documents?conversation_id={conversation_b}", headers=headers
    ).json()

    assert [d["source"] for d in docs_a] == ["rieng_cua_a.txt"]
    assert docs_b == []


def test_delete_conversation_removes_its_documents(client):
    token = register(client)["token"]
    headers = auth_header(token)
    conversation_id = client.post("/api/conversations", headers=headers).json()["id"]
    upload_file(client, headers, conversation_id)

    client.delete(f"/api/conversations/{conversation_id}", headers=headers)

    user = auth.verify_user("dat_uit", "matkhau123")
    assert vector_store.count_chunks(user_id=user.id, conversation_id=conversation_id) == 0


def test_documents_of_foreign_conversation_return_404(client):
    token_a = register(client, "user_a")["token"]
    token_b = register(client, "user_b")["token"]
    conversation_id = client.post(
        "/api/conversations", headers=auth_header(token_a)
    ).json()["id"]

    assert client.get(
        f"/api/documents?conversation_id={conversation_id}",
        headers=auth_header(token_b),
    ).status_code == 404
    assert upload_file(
        client, auth_header(token_b), conversation_id
    ).status_code == 404


def test_upload_rejects_unsupported_extension(client):
    token = register(client)["token"]
    response = client.post(
        "/api/documents",
        files=[("files", ("virus.exe", b"MZ...", "application/octet-stream"))],
        headers=auth_header(token),
    )
    assert response.status_code == 400


def test_upload_rejects_oversized_file(client):
    token = register(client)["token"]
    huge = b"x" * (api_module.MAX_UPLOAD_BYTES + 1024)

    response = upload_file(client, auth_header(token), filename="to.txt", content=huge)

    assert response.status_code == 413


def test_upload_rejects_too_many_files(client):
    token = register(client)["token"]
    files = [
        ("files", (f"f{index}.txt", b"noi dung", "text/plain"))
        for index in range(api_module.MAX_UPLOAD_FILES + 1)
    ]
    response = client.post("/api/documents", files=files, headers=auth_header(token))
    assert response.status_code == 400


# --- Memories ---


def test_memories_endpoints(client):
    token = register(client)["token"]
    headers = auth_header(token)

    user = auth.verify_user("dat_uit", "matkhau123")
    user_memory.add_memory("thích Python", user_id=user.id)

    listing = client.get("/api/memories", headers=headers).json()
    assert listing[0]["content"] == "thích Python"

    client.delete(f"/api/memories/{listing[0]['id']}", headers=headers)
    assert client.get("/api/memories", headers=headers).json() == []


# --- Health ---


def test_health_endpoint(client):
    install_fake_chat_service([])
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
