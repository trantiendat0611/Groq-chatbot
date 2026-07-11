"""Trí nhớ dài hạn về người dùng, xuyên suốt mọi cuộc chat.

Cơ chế giống tính năng Memory của ChatGPT/Claude: agent tự quyết định
lưu một sự thật đáng nhớ (qua tool `remember`), các lượt chat sau
tự động nhớ lại những điều liên quan và đưa vào system prompt.

Đa người dùng: user_id=None là chế độ local; backend API truyền user_id
thật để trí nhớ của mỗi người tách biệt hoàn toàn.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import db
from src.config import PROJECT_ROOT


DB_PATH = PROJECT_ROOT / "data" / "knowledge.db"

MAX_MEMORY_LENGTH = 500


@dataclass(frozen=True)
class UserMemory:
    id: int
    content: str
    created_at: str


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    return db.connect(db_path or DB_PATH)


def _ensure_column(connection: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def init_memory_store() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS user_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding BLOB,
                user_id INTEGER,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _ensure_column(connection, "user_memories", "user_id", "user_id INTEGER")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_memories_user ON user_memories (user_id)"
        )


def add_memory(
    content: str,
    embedding: list[float] | None = None,
    user_id: int | None = None,
) -> int | None:
    """Lưu một điều đáng nhớ. Trả về id, hoặc None nếu trùng/không hợp lệ."""
    content = " ".join(content.strip().split())
    if not content:
        return None
    content = content[:MAX_MEMORY_LENGTH]

    init_memory_store()
    with _connect() as connection:
        existing = connection.execute(
            "SELECT id FROM user_memories WHERE content = ? AND user_id IS ?",
            (content, user_id),
        ).fetchone()
        if existing is not None:
            return None

        blob = (
            np.asarray(embedding, dtype=np.float32).tobytes()
            if embedding is not None
            else None
        )
        cursor = connection.execute(
            "INSERT INTO user_memories (content, embedding, user_id) VALUES (?, ?, ?)",
            (content, blob, user_id),
        )
        return int(cursor.lastrowid)


def list_memories(user_id: int | None = None) -> list[UserMemory]:
    init_memory_store()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, content, created_at
            FROM user_memories
            WHERE user_id IS ?
            ORDER BY id DESC
            """,
            (user_id,),
        ).fetchall()
    return [
        UserMemory(id=row["id"], content=row["content"], created_at=row["created_at"])
        for row in rows
    ]


def delete_memory(memory_id: int, user_id: int | None = None) -> None:
    init_memory_store()
    with _connect() as connection:
        connection.execute(
            "DELETE FROM user_memories WHERE id = ? AND user_id IS ?",
            (memory_id, user_id),
        )


def clear_memories(user_id: int | None = None) -> None:
    init_memory_store()
    with _connect() as connection:
        connection.execute("DELETE FROM user_memories WHERE user_id IS ?", (user_id,))


def search_memories(
    query_embedding: list[float] | None = None,
    top_k: int = 5,
    user_id: int | None = None,
) -> list[UserMemory]:
    """Nhớ lại các điều liên quan nhất; không có embedding thì lấy mới nhất."""
    init_memory_store()
    with _connect() as connection:
        rows = connection.execute(
            "SELECT id, content, embedding, created_at FROM user_memories WHERE user_id IS ?",
            (user_id,),
        ).fetchall()

    if not rows:
        return []

    def to_memory(row: sqlite3.Row) -> UserMemory:
        return UserMemory(id=row["id"], content=row["content"], created_at=row["created_at"])

    if query_embedding is None:
        recent = sorted(rows, key=lambda row: row["id"], reverse=True)[:top_k]
        return [to_memory(row) for row in recent]

    query = np.asarray(query_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query)

    scored: list[tuple[float, sqlite3.Row]] = []
    without_embedding: list[sqlite3.Row] = []

    for row in rows:
        if row["embedding"] is None:
            without_embedding.append(row)
            continue
        vector = np.frombuffer(row["embedding"], dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0 or query_norm == 0:
            continue
        score = float(np.dot(query, vector) / (query_norm * norm))
        scored.append((score, row))

    scored.sort(key=lambda item: item[0], reverse=True)
    results = [to_memory(row) for _, row in scored[:top_k]]

    # Memory chưa có embedding (lưu khi embedder tắt) vẫn được nhớ tới, xếp sau.
    for row in without_embedding:
        if len(results) >= top_k:
            break
        results.append(to_memory(row))

    return results


def format_memories_block(memories: list[UserMemory]) -> str:
    if not memories:
        return ""
    lines = "\n".join(f"- {memory.content}" for memory in memories)
    return (
        "Những điều đã biết về người dùng từ các cuộc trò chuyện trước "
        "(dùng khi liên quan, không cần nhắc lại trừ khi hữu ích):\n" + lines
    )
