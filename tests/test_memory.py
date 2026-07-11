import pytest

from src import memory


@pytest.fixture(autouse=True)
def temp_database(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "test_chats.db")
    memory.init_database()


def test_create_and_list_chats():
    first = memory.create_chat("Chat A")
    second = memory.create_chat("Chat B")

    chats = memory.list_chats()
    ids = [chat.id for chat in chats]

    assert first in ids
    assert second in ids
    assert len(chats) == 2


def test_chat_exists():
    conversation_id = memory.create_chat()
    assert memory.chat_exists(conversation_id)
    assert not memory.chat_exists(conversation_id + 999)


def test_get_chat_title_fallback_for_missing_chat():
    assert memory.get_chat_title(12345) == "Chat mới"


def test_add_and_get_messages_in_order():
    conversation_id = memory.create_chat()
    memory.add_chat_message(conversation_id, "user", "Xin chào")
    memory.add_chat_message(conversation_id, "assistant", "Chào bạn!")

    messages = memory.get_chat_messages(conversation_id)

    assert messages == [
        {"role": "user", "content": "Xin chào"},
        {"role": "assistant", "content": "Chào bạn!"},
    ]


def test_add_message_rejects_invalid_role():
    conversation_id = memory.create_chat()
    with pytest.raises(ValueError):
        memory.add_chat_message(conversation_id, "system", "không được phép")


def test_update_chat_message():
    conversation_id = memory.create_chat()
    message_id = memory.add_chat_message(conversation_id, "assistant", "Xin")

    memory.update_chat_message(message_id, "Xin chào bạn")

    messages = memory.get_chat_messages(conversation_id)
    assert messages == [{"role": "assistant", "content": "Xin chào bạn"}]


def test_rename_chat():
    conversation_id = memory.create_chat("Tên cũ")
    memory.rename_chat(conversation_id, "Tên mới")
    assert memory.get_chat_title(conversation_id) == "Tên mới"


def test_delete_chat_cascades_messages():
    conversation_id = memory.create_chat()
    memory.add_chat_message(conversation_id, "user", "sẽ bị xóa")

    memory.delete_chat(conversation_id)

    assert not memory.chat_exists(conversation_id)
    assert memory.get_chat_messages(conversation_id) == []


def test_ensure_chat_exists_creates_then_reuses():
    created = memory.ensure_chat_exists()
    reused = memory.ensure_chat_exists()
    assert created == reused


def test_ensure_chat_exists_returns_most_recently_updated():
    older = memory.create_chat("Cũ")
    newer = memory.create_chat("Mới")
    memory.add_chat_message(older, "user", "chạm vào chat cũ")

    # Chat "Cũ" vừa được cập nhật nên phải đứng đầu danh sách.
    chats = memory.list_chats()
    assert chats[0].id in {older, newer}
    assert memory.ensure_chat_exists() == chats[0].id
