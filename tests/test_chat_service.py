from types import SimpleNamespace

import httpx
import pytest
from groq import NotFoundError, RateLimitError

from src.chat_service import (
    ChatService,
    ChatServiceError,
    GenerationSettings,
    build_groq_messages,
)
from src.config import AppConfig


def make_config(**overrides) -> AppConfig:
    defaults = {
        "groq_api_key": "test-key",
        "model": "test-model",
        "temperature": 0.5,
        "max_tokens": 100,
        "context_tokens": 6000,
        "max_retries": 2,
    }
    defaults.update(overrides)
    return AppConfig(**defaults)


def make_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def make_http_error(error_class, status_code: int, headers: dict | None = None):
    request = httpx.Request("POST", "https://api.groq.com/v1/chat/completions")
    response = httpx.Response(status_code, request=request, headers=headers or {})
    return error_class("lỗi giả lập", response=response, body=None)


class FakeCompletions:
    """Client giả: trả lần lượt các 'kịch bản' — exception thì raise, còn lại thì return."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_service(outcomes, config: AppConfig | None = None) -> tuple[ChatService, FakeCompletions]:
    completions = FakeCompletions(outcomes)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    service = ChatService(client=client, config=config or make_config())
    service._sleep = lambda seconds: None  # không đợi thật trong test
    return service, completions


def test_build_groq_messages_puts_system_first_and_strips():
    messages = build_groq_messages(
        [{"role": "user", "content": "hi"}],
        system_prompt="  Bạn là trợ lý.  ",
    )
    assert messages[0] == {"role": "system", "content": "Bạn là trợ lý."}
    assert messages[1] == {"role": "user", "content": "hi"}


def test_generate_reply_returns_content():
    service, completions = make_service([make_response("Xin chào!")])

    answer = service.generate_reply([{"role": "user", "content": "chào"}])

    assert answer == "Xin chào!"
    assert len(completions.calls) == 1
    sent = completions.calls[0]["messages"]
    assert sent[0]["role"] == "system"
    assert sent[-1] == {"role": "user", "content": "chào"}


def test_generate_reply_trims_long_history():
    config = make_config(context_tokens=150)
    long_history = [
        {"role": "user", "content": "câu hỏi cũ " * 40},
        {"role": "assistant", "content": "trả lời cũ " * 40},
        {"role": "user", "content": "câu hỏi mới nhất"},
    ]
    service, completions = make_service([make_response("ok")], config=config)

    service.generate_reply(long_history)

    sent = completions.calls[0]["messages"]
    # system + tối đa vài message mới nhất; message cũ nhất phải bị cắt.
    assert sent[-1]["content"] == "câu hỏi mới nhất"
    assert all("câu hỏi cũ" not in m["content"] for m in sent)


def test_generate_reply_retries_on_rate_limit_then_succeeds():
    outcomes = [
        make_http_error(RateLimitError, 429),
        make_http_error(RateLimitError, 429),
        make_response("thành công sau retry"),
    ]
    service, completions = make_service(outcomes)

    answer = service.generate_reply([{"role": "user", "content": "hi"}])

    assert answer == "thành công sau retry"
    assert len(completions.calls) == 3


def test_generate_reply_gives_up_after_max_retries():
    outcomes = [make_http_error(RateLimitError, 429)] * 3
    service, completions = make_service(outcomes, config=make_config(max_retries=2))

    with pytest.raises(ChatServiceError, match="rate limit"):
        service.generate_reply([{"role": "user", "content": "hi"}])

    assert len(completions.calls) == 3


def test_generate_reply_does_not_retry_non_retryable_error():
    outcomes = [make_http_error(NotFoundError, 404)]
    service, completions = make_service(outcomes)

    with pytest.raises(ChatServiceError, match="Model không tồn tại"):
        service.generate_reply([{"role": "user", "content": "hi"}])

    assert len(completions.calls) == 1


def test_retry_honors_retry_after_header():
    delays = []
    outcomes = [
        make_http_error(RateLimitError, 429, headers={"retry-after": "7"}),
        make_response("ok"),
    ]
    service, _ = make_service(outcomes)
    service._sleep = delays.append

    service.generate_reply([{"role": "user", "content": "hi"}])

    assert delays == [7.0]


def test_stream_reply_yields_chunks():
    def make_chunk(text):
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])

    stream = iter([make_chunk("Xin "), make_chunk(None), make_chunk("chào!")])
    service, _ = make_service([stream])

    chunks = list(service.stream_reply([{"role": "user", "content": "hi"}]))

    assert chunks == ["Xin ", "chào!"]


def test_custom_settings_are_passed_through():
    service, completions = make_service([make_response("ok")])
    settings = GenerationSettings(model="custom-model", temperature=0.1, max_tokens=42)

    service.generate_reply([{"role": "user", "content": "hi"}], settings=settings)

    call = completions.calls[0]
    assert call["model"] == "custom-model"
    assert call["temperature"] == 0.1
    assert call["max_tokens"] == 42
