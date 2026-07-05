import json
from types import SimpleNamespace

from src.agent import AgentEvent, AgentService
from src.chat_service import ChatService
from src.config import AppConfig
from src.tools import make_calculator_tool


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


def tool_call_chunk(call_id: str, name: str, arguments: str, index: int = 0):
    call = SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    delta = SimpleNamespace(content=None, tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=None)])


def partial_args_chunk(arguments: str, index: int = 0):
    """Mảnh arguments rơi rải rác giữa stream (không có id/name)."""
    call = SimpleNamespace(
        index=index,
        id=None,
        function=SimpleNamespace(name=None, arguments=arguments),
    )
    delta = SimpleNamespace(content=None, tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=None)])


class FakeCompletions:
    def __init__(self, streams):
        self.streams = list(streams)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.streams.pop(0))


def make_agent(streams, tools=None, config=None):
    completions = FakeCompletions(streams)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    chat_service = ChatService(client=client, config=config or make_config())
    chat_service._sleep = lambda seconds: None
    agent = AgentService(chat_service, tools=tools or [make_calculator_tool()])
    return agent, completions


def collect_events(agent, prompt="tính 6*7"):
    return list(agent.run_stream([{"role": "user", "content": prompt}]))


def test_direct_answer_without_tools():
    streams = [[text_chunk("Xin "), text_chunk("chào!")]]
    agent, completions = make_agent(streams)

    events = collect_events(agent, "chào bạn")

    assert [e.content for e in events if e.type == "text"] == ["Xin ", "chào!"]
    assert len(completions.calls) == 1
    # Lượt đầu phải kèm danh sách tools cho model lựa chọn.
    assert "tools" in completions.calls[0]


def test_tool_call_then_final_answer():
    streams = [
        [tool_call_chunk("call_1", "calculator", '{"expression": "6*7"}')],
        [text_chunk("Kết quả là 42.")],
    ]
    agent, completions = make_agent(streams)

    events = collect_events(agent)

    types = [e.type for e in events]
    assert types == ["tool_call", "tool_result", "text"]
    assert events[0].tool_name == "calculator"
    assert events[1].content == "42"
    assert events[2].content == "Kết quả là 42."

    # Lượt thứ hai phải chứa message role=tool mang kết quả.
    second_call_messages = completions.calls[1]["messages"]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "42"
    assert tool_messages[0]["tool_call_id"] == "call_1"


def test_streamed_tool_arguments_are_accumulated():
    # arguments bị cắt thành nhiều mảnh — phải được ghép lại thành JSON hoàn chỉnh.
    streams = [
        [
            tool_call_chunk("call_1", "calculator", '{"expres'),
            partial_args_chunk('sion": "2+2"}'),
        ],
        [text_chunk("Bằng 4.")],
    ]
    agent, _ = make_agent(streams)

    events = collect_events(agent)

    tool_call = next(e for e in events if e.type == "tool_call")
    assert json.loads(tool_call.arguments) == {"expression": "2+2"}
    tool_result = next(e for e in events if e.type == "tool_result")
    assert tool_result.content == "4"


def test_multiple_tool_calls_in_one_round():
    streams = [
        [
            tool_call_chunk("call_1", "calculator", '{"expression": "1+1"}', index=0),
            tool_call_chunk("call_2", "calculator", '{"expression": "2+2"}', index=1),
        ],
        [text_chunk("Xong.")],
    ]
    agent, completions = make_agent(streams)

    events = collect_events(agent)

    results = [e.content for e in events if e.type == "tool_result"]
    assert results == ["2", "4"]
    tool_messages = [
        m for m in completions.calls[1]["messages"] if m.get("role") == "tool"
    ]
    assert [m["tool_call_id"] for m in tool_messages] == ["call_1", "call_2"]


def test_unknown_tool_error_is_fed_back_to_model():
    streams = [
        [tool_call_chunk("call_1", "tool_ma", "{}")],
        [text_chunk("Xin lỗi, không có công cụ đó.")],
    ]
    agent, completions = make_agent(streams)

    events = collect_events(agent)

    tool_result = next(e for e in events if e.type == "tool_result")
    assert "không có công cụ" in tool_result.content
    # Vòng lặp vẫn tiếp tục và model chốt được câu trả lời.
    assert events[-1].type == "text"


def test_last_step_forces_answer_without_tools():
    config = make_config(agent_max_steps=2)
    streams = [
        [tool_call_chunk("call_1", "calculator", '{"expression": "1+1"}')],
        [text_chunk("Chốt: 2.")],
    ]
    agent, completions = make_agent(streams, config=config)

    collect_events(agent)

    # Bước cuối cùng không được phép đưa tools (buộc model trả lời).
    assert "tools" in completions.calls[0]
    assert "tools" not in completions.calls[1]


def test_agent_event_dataclass_defaults():
    event = AgentEvent(type="text", content="hi")
    assert event.tool_name == ""
    assert event.arguments == ""


def make_api_error():
    import httpx
    from groq import APIError

    request = httpx.Request("POST", "https://api.groq.com/v1/chat/completions")
    return APIError("Failed to call a function.", request, body=None)


def stream_then_raise(chunks, exc):
    yield from chunks
    raise exc


def test_failed_tool_generation_retries_without_tools():
    # Vòng 1: model sinh tool call hỏng -> Groq ném APIError giữa stream.
    # Agent phải tự phục hồi: chạy lại một vòng KHÔNG kèm tools.
    streams = [
        stream_then_raise([], make_api_error()),
        [text_chunk("Trả lời thường sau khi phục hồi.")],
    ]
    agent, completions = make_agent(streams)

    events = collect_events(agent)

    assert [e.type for e in events] == ["text"]
    assert events[0].content == "Trả lời thường sau khi phục hồi."
    assert "tools" in completions.calls[0]
    assert "tools" not in completions.calls[1]  # vòng phục hồi không đưa tool


def test_failed_generation_after_text_emitted_raises_friendly_error():
    import pytest

    from src.chat_service import ChatServiceError

    # Đã phát text ra UI rồi mới lỗi -> không thể retry (sẽ lặp nội dung),
    # phải báo lỗi dễ hiểu.
    streams = [stream_then_raise([text_chunk("Đang trả lời...")], make_api_error())]
    agent, _ = make_agent(streams)

    with pytest.raises(ChatServiceError, match="không tạo được lời gọi công cụ"):
        collect_events(agent)


def test_repeated_identical_tool_call_uses_cache():
    # Model gọi calculator với CÙNG tham số ở 2 vòng liền -> vòng 2 không
    # chạy lại tool mà trả kết quả cache kèm nhắc nhở.
    streams = [
        [tool_call_chunk("call_1", "calculator", '{"expression": "6*7"}')],
        [tool_call_chunk("call_2", "calculator", '{"expression": "6*7"}')],
        [text_chunk("Kết quả là 42.")],
    ]
    agent, _ = make_agent(streams)

    events = collect_events(agent)
    results = [e.content for e in events if e.type == "tool_result"]

    assert results[0] == "42"
    assert "Nhắc lại" in results[1]
    assert "42" in results[1]
    assert events[-1].content == "Kết quả là 42."


def test_failed_generation_twice_raises():
    import pytest

    from src.chat_service import ChatServiceError

    # Cả vòng phục hồi cũng lỗi -> dừng với thông điệp rõ ràng.
    streams = [
        stream_then_raise([], make_api_error()),
        stream_then_raise([], make_api_error()),
    ]
    agent, _ = make_agent(streams)

    with pytest.raises(ChatServiceError):
        collect_events(agent)
