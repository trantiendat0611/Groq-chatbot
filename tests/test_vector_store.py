import pytest

from src import vector_store
from src.rag import DocumentChunk


@pytest.fixture(autouse=True)
def temp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "DB_PATH", tmp_path / "test_knowledge.db")
    vector_store.init_store()


def make_chunk(text: str, source: str = "doc.txt", index: int = 1) -> DocumentChunk:
    return DocumentChunk(source=source, text=text, chunk_index=index, page=None)


class FakeEmbedder:
    """Embedder giả: vector 3 chiều xác định trước theo nội dung."""

    vectors = {
        "mèo là động vật nuôi trong nhà": [1.0, 0.0, 0.0],
        "chó là bạn trung thành của con người": [0.9, 0.1, 0.0],
        "lập trình python cho người mới": [0.0, 0.0, 1.0],
    }

    def embed_texts(self, texts):
        return [self.vectors[text] for text in texts]

    def embed_query(self, query):
        if "mèo" in query:
            return [1.0, 0.0, 0.0]
        return [0.0, 0.0, 1.0]


def seeded_chunks():
    texts = list(FakeEmbedder.vectors.keys())
    return [make_chunk(text, index=i + 1) for i, text in enumerate(texts)]


def test_add_and_count_chunks():
    chunks = seeded_chunks()
    added = vector_store.add_chunks(chunks, FakeEmbedder().embed_texts([c.text for c in chunks]))
    assert added == 3
    assert vector_store.count_chunks() == 3


def test_add_chunks_embedding_count_mismatch_raises():
    with pytest.raises(ValueError):
        vector_store.add_chunks([make_chunk("a")], [[1.0], [2.0]])


def test_semantic_search_ranks_by_cosine():
    chunks = seeded_chunks()
    vector_store.add_chunks(chunks, FakeEmbedder().embed_texts([c.text for c in chunks]))

    results = vector_store.semantic_search([1.0, 0.0, 0.0], top_k=2)

    assert results[0].chunk.text == "mèo là động vật nuôi trong nhà"
    assert results[0].score > results[1].score


def test_search_prefers_semantic_with_embedder():
    chunks = seeded_chunks()
    vector_store.add_chunks(chunks, FakeEmbedder().embed_texts([c.text for c in chunks]))

    results, method = vector_store.search("con mèo của tôi", top_k=2, embedder=FakeEmbedder())

    assert method == "semantic"
    assert "mèo" in results[0].chunk.text


def test_search_falls_back_to_keyword_without_embedder():
    chunks = seeded_chunks()
    vector_store.add_chunks(chunks)  # không có embedding

    results, method = vector_store.search("lập trình python", top_k=2, embedder=None)

    assert method == "keyword"
    assert "python" in results[0].chunk.text


def test_search_returns_none_method_when_empty():
    results, method = vector_store.search("bất kỳ", top_k=2, embedder=None)
    assert results == []
    assert method == "none"


def test_list_sources_and_delete_source():
    vector_store.add_chunks([make_chunk("a", source="one.txt"), make_chunk("b", source="two.txt")])

    sources = vector_store.list_sources()
    assert ("one.txt", 1) in sources
    assert ("two.txt", 1) in sources

    vector_store.delete_source("one.txt")
    assert vector_store.list_sources() == [("two.txt", 1)]


def test_clear_store():
    vector_store.add_chunks([make_chunk("a")])
    vector_store.clear_store()
    assert vector_store.count_chunks() == 0


def test_load_all_chunks_roundtrip():
    original = make_chunk("nội dung gốc", source="s.txt", index=7)
    vector_store.add_chunks([original])

    loaded = vector_store.load_all_chunks()

    assert loaded == [original]
