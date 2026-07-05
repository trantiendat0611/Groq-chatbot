import pytest

from src import usage


@pytest.fixture(autouse=True)
def temp_usage_db(tmp_path, monkeypatch):
    monkeypatch.setattr(usage, "DB_PATH", tmp_path / "test_usage.db")
    usage.init_usage_table()


def test_record_and_summarize():
    usage.record_usage("model-a", prompt_tokens=100, completion_tokens=50, user_id=1)
    usage.record_usage("model-a", prompt_tokens=200, completion_tokens=80, user_id=1)

    summary = usage.usage_summary(user_id=1)

    assert summary.total_requests == 2
    assert summary.prompt_tokens == 300
    assert summary.completion_tokens == 130
    assert summary.total_tokens == 430


def test_summary_isolated_per_user():
    usage.record_usage("model-a", 100, 50, user_id=1)
    usage.record_usage("model-a", 999, 999, user_id=2)

    assert usage.usage_summary(user_id=1).total_tokens == 150
    assert usage.usage_summary(user_id=2).total_tokens == 1998


def test_summary_empty():
    summary = usage.usage_summary(user_id=42)
    assert summary.total_requests == 0
    assert summary.total_tokens == 0


def test_local_mode_uses_null_user():
    usage.record_usage("model-a", 10, 5, user_id=None)
    assert usage.usage_summary(user_id=None).total_requests == 1
    assert usage.usage_summary(user_id=1).total_requests == 0
