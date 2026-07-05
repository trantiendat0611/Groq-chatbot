"""Xác thực người dùng: tài khoản + phiên đăng nhập.

Nguyên tắc bảo mật:
- Mật khẩu băm bằng PBKDF2-HMAC-SHA256 với salt riêng từng user
  (600.000 vòng theo khuyến nghị OWASP) — không bao giờ lưu mật khẩu gốc.
- Token phiên là chuỗi ngẫu nhiên 256-bit, chỉ lưu BẢN BĂM SHA-256 trong DB:
  lộ database cũng không lấy được token đang dùng.
- So sánh bằng hmac.compare_digest chống timing attack.
"""

import hashlib
import hmac
import re
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config import PROJECT_ROOT


DB_PATH = PROJECT_ROOT / "data" / "chats.db"

PBKDF2_ITERATIONS = 600_000
SESSION_TTL_DAYS = 30

USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.]{3,32}$")
MIN_PASSWORD_LENGTH = 8


class AuthError(ValueError):
    """Lỗi xác thực có thông điệp an toàn để hiển thị cho người dùng."""


@dataclass(frozen=True)
class User:
    id: int
    username: str
    created_at: str


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    db_path = db_path or DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_auth_tables() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_token_hash ON sessions (token_hash)"
        )


def _hash_password(password: str, salt_hex: str) -> str:
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        PBKDF2_ITERATIONS,
    )
    return derived.hex()


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_timestamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def validate_credentials(username: str, password: str) -> None:
    if not USERNAME_PATTERN.match(username or ""):
        raise AuthError(
            "Tên đăng nhập phải dài 3-32 ký tự, chỉ gồm chữ, số, dấu chấm hoặc gạch dưới."
        )
    if len(password or "") < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Mật khẩu phải có ít nhất {MIN_PASSWORD_LENGTH} ký tự.")


def create_user(username: str, password: str) -> User:
    username = (username or "").strip()
    validate_credentials(username, password)

    salt_hex = secrets.token_hex(16)
    password_hash = _hash_password(password, salt_hex)

    init_auth_tables()
    try:
        with _connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO users (username, password_hash, password_salt)
                VALUES (?, ?, ?)
                """,
                (username, password_hash, salt_hex),
            )
            row = connection.execute(
                "SELECT id, username, created_at FROM users WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
    except sqlite3.IntegrityError as exc:
        raise AuthError("Tên đăng nhập này đã được sử dụng.") from exc

    return User(id=row["id"], username=row["username"], created_at=row["created_at"])


def verify_user(username: str, password: str) -> User | None:
    init_auth_tables()
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT id, username, password_hash, password_salt, created_at
            FROM users WHERE username = ?
            """,
            ((username or "").strip(),),
        ).fetchone()

    if row is None:
        # Vẫn băm một lần để thời gian phản hồi không tiết lộ username tồn tại hay không.
        _hash_password(password or "", secrets.token_hex(16))
        return None

    expected = row["password_hash"]
    actual = _hash_password(password or "", row["password_salt"])
    if not hmac.compare_digest(expected, actual):
        return None

    return User(id=row["id"], username=row["username"], created_at=row["created_at"])


def create_session(user_id: int, ttl_days: int = SESSION_TTL_DAYS) -> str:
    """Tạo phiên mới, trả về token gốc (chỉ xuất hiện đúng một lần ở đây)."""
    token = secrets.token_urlsafe(32)
    expires_at = _format_timestamp(_utc_now() + timedelta(days=ttl_days))

    init_auth_tables()
    with _connect() as connection:
        connection.execute(
            "INSERT INTO sessions (user_id, token_hash, expires_at) VALUES (?, ?, ?)",
            (user_id, _hash_token(token), expires_at),
        )
    return token


def get_user_by_token(token: str) -> User | None:
    if not token:
        return None

    init_auth_tables()
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT users.id, users.username, users.created_at, sessions.expires_at
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token_hash = ?
            """,
            (_hash_token(token),),
        ).fetchone()

    if row is None:
        return None

    if row["expires_at"] < _format_timestamp(_utc_now()):
        revoke_session(token)
        return None

    return User(id=row["id"], username=row["username"], created_at=row["created_at"])


def revoke_session(token: str) -> None:
    init_auth_tables()
    with _connect() as connection:
        connection.execute(
            "DELETE FROM sessions WHERE token_hash = ?",
            (_hash_token(token),),
        )


def cleanup_expired_sessions() -> int:
    """Xóa các phiên hết hạn; trả về số phiên đã dọn."""
    init_auth_tables()
    with _connect() as connection:
        cursor = connection.execute(
            "DELETE FROM sessions WHERE expires_at < ?",
            (_format_timestamp(_utc_now()),),
        )
        return cursor.rowcount
