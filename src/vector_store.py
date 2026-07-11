"""Kho tri thức bền vững: lưu chunks tài liệu + embedding vào SQLite.

Giải quyết hai hạn chế của bản RAG cũ:
- Tài liệu không còn mất khi tắt app (lưu vào data/knowledge.db).
- Tìm kiếm ngữ nghĩa bằng embedding (cosine similarity); nếu không có
  embedding thì fallback về tìm kiếm từ khóa có IDF như trước.

Phạm vi tài liệu:
- Mỗi tài liệu gắn với MỘT đoạn chat (conversation_id) — giống cách
  Claude/ChatGPT đính kèm file theo từng cuộc hội thoại. Agent chỉ đọc
  được tài liệu đã nạp trong chính đoạn chat đang mở.
- user_id=None là chế độ local (Streamlit); backend API truyền user_id
  thật để cô lập dữ liệu của từng người.
"""

import sqlite3
from pathlib import Path

import numpy as np

from src import db
from src.config import PROJECT_ROOT
from src.rag import DocumentChunk, RetrievedChunk, retrieve_chunks


DB_PATH = PROJECT_ROOT / "data" / "knowledge.db"


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    return db.connect(db_path or DB_PATH)


def _ensure_column(connection: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def init_store() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS doc_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                page INTEGER,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB,
                user_id INTEGER,
                conversation_id INTEGER,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _ensure_column(connection, "doc_chunks", "user_id", "user_id INTEGER")
        _ensure_column(connection, "doc_chunks", "conversation_id", "conversation_id INTEGER")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_doc_chunks_source ON doc_chunks (source)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_doc_chunks_user ON doc_chunks (user_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_doc_chunks_conversation "
            "ON doc_chunks (conversation_id)"
        )


def _embedding_to_blob(embedding: list[float]) -> bytes:
    return np.asarray(embedding, dtype=np.float32).tobytes()


def _blob_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def add_chunks(
    chunks: list[DocumentChunk],
    embeddings: list[list[float]] | None = None,
    user_id: int | None = None,
    conversation_id: int | None = None,
) -> int:
    """Lưu chunks (kèm embedding nếu có) vào một đoạn chat. Trả về số chunk đã lưu."""
    if embeddings is not None and len(embeddings) != len(chunks):
        raise ValueError("Số embedding phải bằng số chunk.")

    init_store()
    with _connect() as connection:
        for index, chunk in enumerate(chunks):
            blob = _embedding_to_blob(embeddings[index]) if embeddings else None
            connection.execute(
                """
                INSERT INTO doc_chunks
                    (source, page, chunk_index, text, embedding, user_id, conversation_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.source,
                    chunk.page,
                    chunk.chunk_index,
                    chunk.text,
                    blob,
                    user_id,
                    conversation_id,
                ),
            )
    return len(chunks)


def _row_to_chunk(row: sqlite3.Row) -> DocumentChunk:
    return DocumentChunk(
        source=row["source"],
        page=row["page"],
        chunk_index=row["chunk_index"],
        text=row["text"],
    )


def load_all_chunks(
    user_id: int | None = None,
    conversation_id: int | None = None,
) -> list[DocumentChunk]:
    init_store()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT source, page, chunk_index, text
            FROM doc_chunks
            WHERE user_id IS ? AND conversation_id IS ?
            ORDER BY id
            """,
            (user_id, conversation_id),
        ).fetchall()
    return [_row_to_chunk(row) for row in rows]


def count_chunks(
    user_id: int | None = None,
    conversation_id: int | None = None,
) -> int:
    init_store()
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS total FROM doc_chunks
            WHERE user_id IS ? AND conversation_id IS ?
            """,
            (user_id, conversation_id),
        ).fetchone()
    return int(row["total"])


def list_sources(
    user_id: int | None = None,
    conversation_id: int | None = None,
) -> list[tuple[str, int]]:
    """Danh sách (tên file, số chunk) đã nạp trong một đoạn chat."""
    init_store()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT source, COUNT(*) AS total
            FROM doc_chunks
            WHERE user_id IS ? AND conversation_id IS ?
            GROUP BY source
            ORDER BY source
            """,
            (user_id, conversation_id),
        ).fetchall()
    return [(row["source"], int(row["total"])) for row in rows]


def delete_source(
    source: str,
    user_id: int | None = None,
    conversation_id: int | None = None,
) -> None:
    init_store()
    with _connect() as connection:
        connection.execute(
            """
            DELETE FROM doc_chunks
            WHERE source = ? AND user_id IS ? AND conversation_id IS ?
            """,
            (source, user_id, conversation_id),
        )


def clear_store(
    user_id: int | None = None,
    conversation_id: int | None = None,
) -> None:
    """Xóa toàn bộ tài liệu của một đoạn chat."""
    init_store()
    with _connect() as connection:
        connection.execute(
            "DELETE FROM doc_chunks WHERE user_id IS ? AND conversation_id IS ?",
            (user_id, conversation_id),
        )


def semantic_search(
    query_embedding: list[float],
    top_k: int = 4,
    user_id: int | None = None,
    conversation_id: int | None = None,
) -> list[RetrievedChunk]:
    """Tìm chunks gần nhất theo cosine similarity trên embedding đã lưu."""
    init_store()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT source, page, chunk_index, text, embedding
            FROM doc_chunks
            WHERE embedding IS NOT NULL AND user_id IS ? AND conversation_id IS ?
            """,
            (user_id, conversation_id),
        ).fetchall()

    if not rows:
        return []

    query = np.asarray(query_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return []

    scored: list[RetrievedChunk] = []
    for row in rows:
        vector = _blob_to_embedding(row["embedding"])
        norm = np.linalg.norm(vector)
        if norm == 0:
            continue
        score = float(np.dot(query, vector) / (query_norm * norm))
        scored.append(RetrievedChunk(chunk=_row_to_chunk(row), score=score))

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_k]


def search(
    query: str,
    top_k: int = 4,
    embedder=None,
    user_id: int | None = None,
    conversation_id: int | None = None,
) -> tuple[list[RetrievedChunk], str]:
    """Tìm kiếm lai trong tài liệu của một đoạn chat: ưu tiên ngữ nghĩa, fallback từ khóa.

    Trả về (kết quả, phương thức) với phương thức là "semantic" | "keyword" | "none".
    """
    if embedder is not None:
        try:
            results = semantic_search(
                embedder.embed_query(query),
                top_k=top_k,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if results:
                return results, "semantic"
        except Exception:
            pass  # embedding lỗi thì âm thầm chuyển sang từ khóa

    chunks = load_all_chunks(user_id=user_id, conversation_id=conversation_id)
    results = retrieve_chunks(query, chunks, top_k=top_k)
    if results:
        return results, "keyword"

    return [], "none"
