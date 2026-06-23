import math
import re
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
DEFAULT_CHUNK_WORDS = 220
DEFAULT_OVERLAP_WORDS = 45

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
    "bị",
    "bằng",
    "các",
    "cái",
    "cần",
    "cho",
    "có",
    "của",
    "đã",
    "để",
    "được",
    "hay",
    "khi",
    "không",
    "là",
    "một",
    "này",
    "những",
    "theo",
    "thì",
    "trong",
    "và",
    "với",
}


@dataclass(frozen=True)
class DocumentChunk:
    source: str
    text: str
    chunk_index: int
    page: int | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    score: float


def is_supported_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def decode_text(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1258", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def extract_text_pages(filename: str, data: bytes) -> list[tuple[int | None, str]]:
    extension = Path(filename).suffix.lower()

    if extension in {".txt", ".md"}:
        return [(None, decode_text(data))]

    if extension == ".pdf":
        reader = PdfReader(BytesIO(data))
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((index, text))
        return pages

    raise ValueError(f"Định dạng file chưa được hỗ trợ: {extension}")


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def split_words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def chunk_text(
    source: str,
    text: str,
    page: int | None,
    start_index: int,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[DocumentChunk]:
    words = split_words(text)
    if not words:
        return []

    chunks = []
    step = max(chunk_words - overlap_words, 1)

    for offset in range(0, len(words), step):
        chunk = " ".join(words[offset : offset + chunk_words]).strip()
        if not chunk:
            continue

        chunks.append(
            DocumentChunk(
                source=source,
                page=page,
                text=chunk,
                chunk_index=start_index + len(chunks) + 1,
            )
        )

        if offset + chunk_words >= len(words):
            break

    return chunks


def build_chunks(files: list[tuple[str, bytes]]) -> list[DocumentChunk]:
    all_chunks = []

    for filename, data in files:
        if not is_supported_file(filename):
            continue

        pages = extract_text_pages(filename, data)

        for page, text in pages:
            chunks = chunk_text(
                source=filename,
                text=text,
                page=page,
                start_index=len(all_chunks),
            )
            all_chunks.extend(chunks)

    return all_chunks


def retrieve_chunks(
    query: str,
    chunks: list[DocumentChunk],
    top_k: int = 4,
) -> list[RetrievedChunk]:
    query_terms = Counter(tokenize(query))
    if not query_terms or not chunks:
        return []

    chunk_terms = [Counter(tokenize(chunk.text)) for chunk in chunks]
    document_frequency = Counter()

    for terms in chunk_terms:
        document_frequency.update(terms.keys())

    scored = []
    total_chunks = len(chunks)

    for chunk, terms in zip(chunks, chunk_terms):
        if not terms:
            continue

        score = 0.0
        for term, query_count in query_terms.items():
            if term not in terms:
                continue

            idf = math.log((total_chunks + 1) / (document_frequency[term] + 1)) + 1
            score += query_count * terms[term] * idf

        if score <= 0:
            continue

        normalized_score = score / math.sqrt(sum(value * value for value in terms.values()))
        scored.append(RetrievedChunk(chunk=chunk, score=normalized_score))

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_k]


def preview_chunks(
    chunks: list[DocumentChunk],
    top_k: int = 4,
) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk=chunk, score=0.0)
        for chunk in chunks[:top_k]
    ]


def retrieve_chunks_with_fallback(
    query: str,
    chunks: list[DocumentChunk],
    top_k: int = 4,
) -> tuple[list[RetrievedChunk], bool]:
    retrieved_chunks = retrieve_chunks(query, chunks, top_k=top_k)
    if retrieved_chunks:
        return retrieved_chunks, False

    return preview_chunks(chunks, top_k=top_k), True


def format_source_label(chunk: DocumentChunk) -> str:
    page_label = f", page {chunk.page}" if chunk.page is not None else ""
    return f"{chunk.source}{page_label}, chunk {chunk.chunk_index}"


def format_rag_context(retrieved_chunks: list[RetrievedChunk]) -> str:
    sections = []
    for index, result in enumerate(retrieved_chunks, start=1):
        sections.append(
            f"[{index}] Source: {format_source_label(result.chunk)}\n"
            f"{result.chunk.text}"
        )
    return "\n\n".join(sections)


def build_rag_system_prompt(base_prompt: str, context: str) -> str:
    if not context.strip():
        return base_prompt

    return f"""
{base_prompt.strip()}

Bạn đang được cung cấp thêm các đoạn tài liệu liên quan bên dưới.
Hãy ưu tiên trả lời dựa trên tài liệu này.
Nếu tài liệu không đủ thông tin để trả lời, hãy nói rõ là tài liệu chưa cung cấp đủ thông tin.
Khi dùng thông tin từ tài liệu, hãy nhắc nguồn ngắn gọn theo dạng [1], [2].

Tài liệu liên quan:
{context}
""".strip()
