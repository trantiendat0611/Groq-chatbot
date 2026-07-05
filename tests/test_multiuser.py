"""Kiểm tra cô lập dữ liệu giữa các user (và chế độ local user_id=None)."""

import pytest

from src import memory, user_memory, vector_store
from src.rag import DocumentChunk


@pytest.fixture(autouse=True)
def temp_databases(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "chats.db")
    monkeypatch.setattr(vector_store, "DB_PATH", tmp_path / "knowledge.db")
    monkeypatch.setattr(user_memory, "DB_PATH", tmp_path / "knowledge.db")
    memory.init_database()
    vector_store.init_store()
    user_memory.init_memory_store()


def test_conversations_isolated_between_users():
    chat_a = memory.create_chat("Chat của A", user_id=1)
    chat_b = memory.create_chat("Chat của B", user_id=2)

    titles_a = [c.title for c in memory.list_chats(user_id=1)]
    titles_b = [c.title for c in memory.list_chats(user_id=2)]

    assert titles_a == ["Chat của A"]
    assert titles_b == ["Chat của B"]
    assert memory.chat_exists(chat_a, user_id=1)
    assert not memory.chat_exists(chat_a, user_id=2)
    assert not memory.chat_exists(chat_b, user_id=1)


def test_local_mode_sees_only_null_user_chats():
    memory.create_chat("Chat local")
    memory.create_chat("Chat của user 1", user_id=1)

    local_titles = [c.title for c in memory.list_chats()]
    assert local_titles == ["Chat local"]


def test_delete_chat_cannot_cross_users():
    chat_a = memory.create_chat("Chat của A", user_id=1)

    memory.delete_chat(chat_a, user_id=2)  # user khác cố xóa
    assert memory.chat_exists(chat_a, user_id=1)

    memory.delete_chat(chat_a, user_id=1)  # chính chủ xóa được
    assert not memory.chat_exists(chat_a, user_id=1)


def test_rename_chat_cannot_cross_users():
    chat_a = memory.create_chat("Tên gốc", user_id=1)
    memory.rename_chat(chat_a, "Bị đổi", user_id=2)
    assert memory.get_chat_title(chat_a, user_id=1) == "Tên gốc"


def make_chunk(text, source="doc.txt"):
    return DocumentChunk(source=source, text=text, chunk_index=1, page=None)


def test_documents_isolated_between_users():
    vector_store.add_chunks([make_chunk("tài liệu của A")], user_id=1)
    vector_store.add_chunks([make_chunk("tài liệu của B")], user_id=2)

    assert vector_store.count_chunks(user_id=1) == 1
    assert vector_store.count_chunks(user_id=2) == 1
    assert vector_store.count_chunks(user_id=None) == 0

    results, _ = vector_store.search("tài liệu", top_k=5, user_id=1)
    assert all("của A" in r.chunk.text for r in results)


def test_clear_store_only_affects_one_user():
    vector_store.add_chunks([make_chunk("của A")], user_id=1)
    vector_store.add_chunks([make_chunk("của B")], user_id=2)

    vector_store.clear_store(user_id=1)

    assert vector_store.count_chunks(user_id=1) == 0
    assert vector_store.count_chunks(user_id=2) == 1


def test_memories_isolated_between_users():
    user_memory.add_memory("A thích mèo", user_id=1)
    user_memory.add_memory("B thích chó", user_id=2)

    contents_a = [m.content for m in user_memory.list_memories(user_id=1)]
    contents_b = [m.content for m in user_memory.list_memories(user_id=2)]

    assert contents_a == ["A thích mèo"]
    assert contents_b == ["B thích chó"]


def test_delete_memory_cannot_cross_users():
    memory_id = user_memory.add_memory("của A", user_id=1)

    user_memory.delete_memory(memory_id, user_id=2)
    assert len(user_memory.list_memories(user_id=1)) == 1


def test_same_memory_content_allowed_for_different_users():
    assert user_memory.add_memory("thích Python", user_id=1) is not None
    assert user_memory.add_memory("thích Python", user_id=2) is not None
    # Nhưng trùng trong cùng user thì bị chặn.
    assert user_memory.add_memory("thích Python", user_id=1) is None
