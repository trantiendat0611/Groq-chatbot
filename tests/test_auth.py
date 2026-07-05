import pytest

from src import auth


@pytest.fixture(autouse=True)
def temp_auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "DB_PATH", tmp_path / "test_auth.db")
    # Giảm số vòng PBKDF2 trong test cho nhanh (production giữ 600k).
    monkeypatch.setattr(auth, "PBKDF2_ITERATIONS", 1000)
    auth.init_auth_tables()


def test_create_user_and_verify():
    user = auth.create_user("dat_uit", "matkhau123")
    assert user.username == "dat_uit"

    verified = auth.verify_user("dat_uit", "matkhau123")
    assert verified is not None
    assert verified.id == user.id


def test_verify_wrong_password():
    auth.create_user("dat_uit", "matkhau123")
    assert auth.verify_user("dat_uit", "sai_mat_khau") is None


def test_verify_unknown_user():
    assert auth.verify_user("khong_ton_tai", "matkhau123") is None


def test_duplicate_username_rejected():
    auth.create_user("dat_uit", "matkhau123")
    with pytest.raises(auth.AuthError, match="đã được sử dụng"):
        auth.create_user("dat_uit", "matkhaukhac1")


def test_duplicate_username_case_insensitive():
    auth.create_user("DatUIT", "matkhau123")
    with pytest.raises(auth.AuthError):
        auth.create_user("datuit", "matkhau456")


def test_username_validation():
    with pytest.raises(auth.AuthError):
        auth.create_user("ab", "matkhau123")  # quá ngắn
    with pytest.raises(auth.AuthError):
        auth.create_user("có dấu cách", "matkhau123")  # ký tự không hợp lệ


def test_password_min_length():
    with pytest.raises(auth.AuthError, match="ít nhất"):
        auth.create_user("dat_uit", "ngan")


def test_password_is_not_stored_in_plaintext(tmp_path):
    auth.create_user("dat_uit", "matkhau123")
    with auth._connect() as connection:
        row = connection.execute("SELECT password_hash, password_salt FROM users").fetchone()
    assert "matkhau123" not in row["password_hash"]
    assert len(row["password_salt"]) == 32  # 16 bytes hex


def test_session_roundtrip():
    user = auth.create_user("dat_uit", "matkhau123")
    token = auth.create_session(user.id)

    resolved = auth.get_user_by_token(token)
    assert resolved is not None
    assert resolved.id == user.id


def test_token_is_stored_hashed():
    user = auth.create_user("dat_uit", "matkhau123")
    token = auth.create_session(user.id)

    with auth._connect() as connection:
        row = connection.execute("SELECT token_hash FROM sessions").fetchone()
    assert row["token_hash"] != token


def test_invalid_token_rejected():
    assert auth.get_user_by_token("token-gia-mao") is None
    assert auth.get_user_by_token("") is None


def test_revoke_session():
    user = auth.create_user("dat_uit", "matkhau123")
    token = auth.create_session(user.id)

    auth.revoke_session(token)
    assert auth.get_user_by_token(token) is None


def test_expired_session_rejected():
    user = auth.create_user("dat_uit", "matkhau123")
    token = auth.create_session(user.id, ttl_days=-1)  # hết hạn từ hôm qua
    assert auth.get_user_by_token(token) is None


def test_cleanup_expired_sessions():
    user = auth.create_user("dat_uit", "matkhau123")
    auth.create_session(user.id, ttl_days=-1)
    auth.create_session(user.id, ttl_days=30)

    removed = auth.cleanup_expired_sessions()

    assert removed == 1
