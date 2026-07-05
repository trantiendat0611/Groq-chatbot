import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src.config import PROJECT_ROOT


DB_PATH = PROJECT_ROOT / "data" / "chats.db"


@dataclass(frozen=True)
class Conversation:
    id: int
    title: str
    created_at: str
    updated_at: str


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    # Đọc DB_PATH lúc gọi (không phải lúc định nghĩa) để test thay được đường dẫn.
    db_path = db_path or DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _ensure_column(connection: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    """Migration nhẹ: thêm cột nếu database cũ chưa có."""
    columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def init_database() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                user_id INTEGER,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations (id)
                    ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
            ON messages (conversation_id, id)
            """
        )
        # Database tạo từ phiên bản cũ chưa có cột user_id.
        _ensure_column(connection, "conversations", "user_id", "user_id INTEGER")
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversations_user_id
            ON conversations (user_id, updated_at)
            """
        )


def create_chat(title: str = "Chat mới", user_id: int | None = None) -> int:
    with _connect() as connection:
        cursor = connection.execute(
            "INSERT INTO conversations (title, user_id) VALUES (?, ?)",
            (title, user_id),
        )
        return int(cursor.lastrowid)


def list_chats(user_id: int | None = None) -> list[Conversation]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            WHERE user_id IS ?
            ORDER BY updated_at DESC, id DESC
            """,
            (user_id,),
        ).fetchall()

    return [
        Conversation(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        for row in rows
    ]


def ensure_chat_exists(user_id: int | None = None) -> int:
    init_database()
    chats = list_chats(user_id)
    if chats:
        return chats[0].id
    return create_chat(user_id=user_id)


def chat_exists(conversation_id: int, user_id: int | None = None) -> bool:
    with _connect() as connection:
        row = connection.execute(
            "SELECT 1 FROM conversations WHERE id = ? AND user_id IS ?",
            (conversation_id, user_id),
        ).fetchone()
    return row is not None


def get_chat_title(conversation_id: int, user_id: int | None = None) -> str:
    with _connect() as connection:
        row = connection.execute(
            "SELECT title FROM conversations WHERE id = ? AND user_id IS ?",
            (conversation_id, user_id),
        ).fetchone()

    if row is None:
        return "Chat mới"
    return row["title"]


def get_chat_messages(conversation_id: int) -> list[dict[str, str]]:
    """Lấy message theo conversation. Quyền sở hữu phải được kiểm tra trước
    bằng chat_exists(conversation_id, user_id)."""
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()

    return [
        {
            "role": row["role"],
            "content": row["content"],
        }
        for row in rows
    ]


def add_chat_message(conversation_id: int, role: str, content: str) -> int:
    if role not in {"user", "assistant"}:
        raise ValueError("Message role must be 'user' or 'assistant'.")

    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO messages (conversation_id, role, content)
            VALUES (?, ?, ?)
            """,
            (conversation_id, role, content),
        )
        connection.execute(
            """
            UPDATE conversations
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (conversation_id,),
        )
        return int(cursor.lastrowid)


def rename_chat(conversation_id: int, title: str, user_id: int | None = None) -> None:
    with _connect() as connection:
        connection.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id IS ?
            """,
            (title, conversation_id, user_id),
        )


def delete_chat(conversation_id: int, user_id: int | None = None) -> None:
    with _connect() as connection:
        connection.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id IS ?",
            (conversation_id, user_id),
        )
