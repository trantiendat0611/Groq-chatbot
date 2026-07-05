import pytest

from src import user_memory


@pytest.fixture(autouse=True)
def temp_memory_db(tmp_path, monkeypatch):
    monkeypatch.setattr(user_memory, "DB_PATH", tmp_path / "test_memory.db")
    user_memory.init_memory_store()


def test_add_and_list_memories():
    user_memory.add_memory("Người dùng tên Đạt, học UIT")
    user_memory.add_memory("Thích lập trình Python")

    memories = user_memory.list_memories()

    assert len(memories) == 2
    # Mới nhất đứng đầu.
    assert memories[0].content == "Thích lập trình Python"


def test_add_memory_normalizes_whitespace():
    memory_id = user_memory.add_memory("  nhiều    khoảng   trắng  ")
    assert memory_id is not None
    assert user_memory.list_memories()[0].content == "nhiều khoảng trắng"


def test_add_memory_rejects_empty():
    assert user_memory.add_memory("   ") is None
    assert user_memory.list_memories() == []


def test_add_memory_deduplicates():
    first = user_memory.add_memory("sự thật lặp lại")
    second = user_memory.add_memory("sự thật lặp lại")

    assert first is not None
    assert second is None
    assert len(user_memory.list_memories()) == 1


def test_add_memory_truncates_very_long_content():
    user_memory.add_memory("x" * 2000)
    assert len(user_memory.list_memories()[0].content) <= user_memory.MAX_MEMORY_LENGTH


def test_delete_memory():
    memory_id = user_memory.add_memory("sẽ bị xóa")
    user_memory.delete_memory(memory_id)
    assert user_memory.list_memories() == []


def test_clear_memories():
    user_memory.add_memory("một")
    user_memory.add_memory("hai")
    user_memory.clear_memories()
    assert user_memory.list_memories() == []


def test_search_without_embedding_returns_most_recent():
    for index in range(8):
        user_memory.add_memory(f"sự thật số {index}")

    results = user_memory.search_memories(query_embedding=None, top_k=3)

    assert len(results) == 3
    assert results[0].content == "sự thật số 7"


def test_search_with_embedding_ranks_by_similarity():
    user_memory.add_memory("thích mèo", embedding=[1.0, 0.0])
    user_memory.add_memory("học python", embedding=[0.0, 1.0])

    results = user_memory.search_memories(query_embedding=[0.9, 0.1], top_k=1)

    assert results[0].content == "thích mèo"


def test_search_includes_memories_without_embedding_as_backfill():
    user_memory.add_memory("có embedding", embedding=[1.0, 0.0])
    user_memory.add_memory("không có embedding")

    results = user_memory.search_memories(query_embedding=[1.0, 0.0], top_k=5)

    contents = {memory.content for memory in results}
    assert contents == {"có embedding", "không có embedding"}


def test_format_memories_block():
    user_memory.add_memory("tên là Đạt")
    block = user_memory.format_memories_block(user_memory.list_memories())
    assert "- tên là Đạt" in block
    assert "người dùng" in block


def test_format_memories_block_empty():
    assert user_memory.format_memories_block([]) == ""
