from src.context import (
    estimate_message_tokens,
    estimate_tokens,
    trim_messages_to_budget,
)


def make_message(role: str, length: int) -> dict[str, str]:
    return {"role": role, "content": "x" * length}


def test_estimate_tokens_empty_text():
    assert estimate_tokens("") == 0


def test_estimate_tokens_short_text_is_at_least_one():
    assert estimate_tokens("ab") == 1


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("x" * 300) == 100


def test_estimate_message_tokens_includes_overhead():
    message = make_message("user", 30)
    assert estimate_message_tokens(message) > estimate_tokens(message["content"])


def test_trim_empty_history():
    assert trim_messages_to_budget([], max_history_tokens=100) == []


def test_trim_keeps_everything_under_budget():
    messages = [
        make_message("user", 30),
        make_message("assistant", 30),
        make_message("user", 30),
    ]
    kept = trim_messages_to_budget(messages, max_history_tokens=1000)
    assert kept == messages


def test_trim_drops_oldest_first():
    messages = [
        make_message("user", 300),
        make_message("assistant", 300),
        make_message("user", 300),
    ]
    # Mỗi message ~104 token; ngân sách 250 chỉ đủ cho 2 message.
    kept = trim_messages_to_budget(messages, max_history_tokens=250)
    assert kept == messages[1:] or kept == messages[2:]
    assert kept[-1] == messages[-1]


def test_trim_always_keeps_last_message_even_if_over_budget():
    messages = [make_message("user", 10_000)]
    kept = trim_messages_to_budget(messages, max_history_tokens=10)
    assert kept == messages


def test_trim_does_not_start_with_assistant_message():
    messages = [
        make_message("user", 300),
        make_message("assistant", 300),
        make_message("user", 300),
        make_message("assistant", 300),
        make_message("user", 300),
    ]
    # Ngân sách vừa đủ giữ 4 message cuối -> message đầu còn lại là assistant,
    # phải bị loại để hội thoại không mở đầu bằng câu trả lời mồ côi.
    kept = trim_messages_to_budget(messages, max_history_tokens=450)
    assert kept[0]["role"] == "user"
    assert kept[-1] == messages[-1]
