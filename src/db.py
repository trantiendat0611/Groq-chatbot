"""Tiện ích kết nối SQLite dùng chung cho mọi module.

Gom cấu hình vào một chỗ để mọi kết nối đều bật WAL: chế độ mặc định
(`delete`) khóa toàn bộ file khi ghi, nên nhiều người dùng chat cùng lúc
dễ gặp "database is locked". WAL cho phép nhiều luồng đọc song song với
một luồng ghi.
"""

import sqlite3
from pathlib import Path


BUSY_TIMEOUT_MS = 5000


def connect(db_path: Path, foreign_keys: bool = False) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path, timeout=BUSY_TIMEOUT_MS / 1000)
    connection.row_factory = sqlite3.Row

    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    # NORMAL an toàn khi đi kèm WAL và nhanh hơn nhiều so với FULL.
    connection.execute("PRAGMA synchronous = NORMAL")

    if foreign_keys:
        connection.execute("PRAGMA foreign_keys = ON")

    return connection
