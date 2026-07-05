"""Đối tượng giả dùng chung cho test: client Groq giả, chunk stream giả."""

from types import SimpleNamespace

from src.config import AppConfig


def make_config(**overrides) -> AppConfig:
    defaults = {
        "groq_api_key": "test-key",
        "model": "test-model",
        "temperature": 0.5,
        "max_tokens": 100,
        "context_tokens": 6000,
        "max_retries": 1,
        "agent_max_steps": 4,
    }
    defaults.update(overrides)
    return AppConfig(**defaults)


def text_chunk(text: str):
    delta = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=None)])


def usage_chunk(prompt_tokens: int, completion_tokens: int):
    """Chunk cuối stream mang thông tin usage như Groq trả về (x_groq.usage)."""
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return SimpleNamespace(
        choices=[],
        x_groq=SimpleNamespace(usage=usage),
    )


def tool_call_chunk(call_id: str, name: str, arguments: str, index: int = 0):
    call = SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    delta = SimpleNamespace(content=None, tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=None)])


class FakeCompletions:
    """Trả lần lượt các stream đã chuẩn bị sẵn cho từng lượt gọi API."""

    def __init__(self, streams):
        self.streams = list(streams)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.streams.pop(0))


def make_fake_client(streams) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(streams)))
