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


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_database() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
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


def create_chat(title: str = "Chat mới") -> int:
    with _connect() as connection:
        cursor = connection.execute(
            "INSERT INTO conversations (title) VALUES (?)",
            (title,),
        )
        return int(cursor.lastrowid)


def list_chats() -> list[Conversation]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC, id DESC
            """
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


def ensure_chat_exists() -> int:
    init_database()
    chats = list_chats()
    if chats:
        return chats[0].id
    return create_chat()


def chat_exists(conversation_id: int) -> bool:
    with _connect() as connection:
        row = connection.execute(
            "SELECT 1 FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
    return row is not None


def get_chat_title(conversation_id: int) -> str:
    with _connect() as connection:
        row = connection.execute(
            "SELECT title FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()

    if row is None:
        return "Chat mới"
    return row["title"]


def get_chat_messages(conversation_id: int) -> list[dict[str, str]]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """
            ,
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


def rename_chat(conversation_id: int, title: str) -> None:
    with _connect() as connection:
        connection.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (title, conversation_id),
        )


def delete_chat(conversation_id: int) -> None:
    with _connect() as connection:
        connection.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )
