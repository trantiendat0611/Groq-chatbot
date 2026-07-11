"""Theo dõi mức sử dụng token: mỗi lượt gọi API ghi một dòng vào SQLite.

Số liệu lấy từ trường usage do Groq trả về trong stream (chính xác);
nếu không có thì backend ước lượng bằng công cụ đếm token nội bộ.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src import db
from src.config import PROJECT_ROOT


DB_PATH = PROJECT_ROOT / "data" / "chats.db"


@dataclass(frozen=True)
class UsageSummary:
    total_requests: int
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    return db.connect(db_path or DB_PATH)


def init_usage_table() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                estimated INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_usage_log_user ON usage_log (user_id, created_at)"
        )


def record_usage(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    user_id: int | None = None,
    estimated: bool = False,
) -> None:
    init_usage_table()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO usage_log (user_id, model, prompt_tokens, completion_tokens, estimated)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, model, int(prompt_tokens), int(completion_tokens), int(estimated)),
        )


def usage_summary(user_id: int | None = None, days: int = 30) -> UsageSummary:
    init_usage_table()
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total_requests,
                COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) AS completion_tokens
            FROM usage_log
            WHERE user_id IS ?
              AND created_at >= datetime('now', ?)
            """,
            (user_id, f"-{int(days)} days"),
        ).fetchone()

    return UsageSummary(
        total_requests=int(row["total_requests"]),
        prompt_tokens=int(row["prompt_tokens"]),
        completion_tokens=int(row["completion_tokens"]),
    )
