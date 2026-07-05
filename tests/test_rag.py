from src.rag import (
    build_chunks,
    build_rag_system_prompt,
    chunk_text,
    decode_text,
    format_rag_context,
    format_source_label,
    is_supported_file,
    retrieve_chunks,
    retrieve_chunks_with_fallback,
    tokenize,
)


def test_is_supported_file():
    assert is_supported_file("notes.txt")
    assert is_supported_file("README.md")
    assert is_supported_file("paper.PDF")
    assert not is_supported_file("image.png")


def test_decode_text_utf8_vietnamese():
    text = "Trí tuệ nhân tạo"
    assert decode_text(text.encode("utf-8")) == text


def test_tokenize_removes_stopwords_and_short_tokens():
    tokens = tokenize("Đây là một hệ thống AI a b")
    assert "là" not in tokens
    assert "một" not in tokens
    assert "ai" in tokens
    assert "hệ" in tokens


def test_chunk_text_splits_long_text_with_overlap():
    words = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text("doc.txt", words, page=None, start_index=0, chunk_words=200, overlap_words=50)

    assert len(chunks) > 1
    # Chunk sau phải lặp lại phần cuối chunk trước (overlap).
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-50:] == second_words[:50]
    # chunk_index tăng dần từ start_index + 1.
    assert [chunk.chunk_index for chunk in chunks] == list(range(1, len(chunks) + 1))


def test_chunk_text_empty_returns_nothing():
    assert chunk_text("doc.txt", "   ", page=None, start_index=0) == []


def test_build_chunks_skips_unsupported_files():
    files = [
        ("notes.txt", "Groq là nền tảng suy luận LLM tốc độ cao".encode("utf-8")),
        ("photo.png", b"\x89PNG..."),
    ]
    chunks = build_chunks(files)
    assert chunks
    assert all(chunk.source == "notes.txt" for chunk in chunks)


def test_retrieve_chunks_ranks_relevant_chunk_first():
    files = [
        (
            "kb.txt",
            (
                "Python là ngôn ngữ lập trình phổ biến cho khoa học dữ liệu. " * 30
                + "\n\n"
                + "Hà Nội là thủ đô của Việt Nam với nhiều di tích lịch sử. " * 30
            ).encode("utf-8"),
        )
    ]
    chunks = build_chunks(files)
    results = retrieve_chunks("thủ đô Việt Nam nằm ở đâu?", chunks, top_k=2)

    assert results
    assert "Hà Nội" in results[0].chunk.text


def test_retrieve_chunks_empty_query_or_chunks():
    assert retrieve_chunks("", [], top_k=3) == []


def test_retrieve_with_fallback_uses_preview_when_no_match():
    files = [("kb.txt", ("nội dung về ẩm thực miền Trung " * 50).encode("utf-8"))]
    chunks = build_chunks(files)

    results, used_fallback = retrieve_chunks_with_fallback("xyzabc123", chunks, top_k=2)

    assert used_fallback
    assert len(results) <= 2
    assert results[0].chunk == chunks[0]


def test_retrieve_with_fallback_no_fallback_when_matched():
    files = [("kb.txt", ("tài liệu nói về máy học và mạng nơ-ron " * 50).encode("utf-8"))]
    chunks = build_chunks(files)

    results, used_fallback = retrieve_chunks_with_fallback("mạng nơ-ron", chunks, top_k=2)

    assert not used_fallback
    assert results


def test_format_source_label_with_and_without_page():
    files = [("kb.txt", b"some plain text content here")]
    chunk = build_chunks(files)[0]
    assert format_source_label(chunk) == "kb.txt, chunk 1"


def test_build_rag_system_prompt_embeds_context_and_base():
    prompt = build_rag_system_prompt("Bạn là trợ lý.", "[1] Source: a.txt\nnội dung")
    assert "Bạn là trợ lý." in prompt
    assert "nội dung" in prompt
    assert "[1]" in prompt


def test_build_rag_system_prompt_without_context_returns_base():
    base = "Bạn là trợ lý."
    assert build_rag_system_prompt(base, "   ") == base


def test_format_rag_context_numbers_sections():
    files = [("kb.txt", ("đoạn một " * 300).encode("utf-8"))]
    chunks = build_chunks(files)
    results, _ = retrieve_chunks_with_fallback("đoạn", chunks, top_k=2)

    context = format_rag_context(results)
    assert context.startswith("[1] Source:")
