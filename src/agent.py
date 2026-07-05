"""Vòng lặp agent: model tự quyết định gọi công cụ, nhận kết quả, suy luận tiếp.

Luồng một lượt trả lời:
    1. Gửi hội thoại + danh sách tool lên Groq (streaming).
    2. Model trả text -> phát sự kiện "text" cho UI hiển thị dần.
       Model yêu cầu gọi tool -> thực thi, phát sự kiện "tool_call"/"tool_result",
       nối kết quả vào hội thoại rồi quay lại bước 1.
    3. Lặp tối đa max_steps; bước cuối cùng không đưa tool để buộc model
       chốt câu trả lời, tránh lặp vô hạn.
"""

from dataclasses import dataclass, field

from groq import APIError

from src.chat_service import (
    RETRYABLE_EXCEPTIONS,
    ChatService,
    ChatServiceError,
    GenerationSettings,
    _friendly_error_message,
)
from src.chat_types import ChatMessage
from src.prompts import SYSTEM_PROMPT
from src.tools import Tool, execute_tool, tool_schemas


@dataclass(frozen=True)
class AgentEvent:
    """Sự kiện phát ra trong lúc agent làm việc, để UI hiển thị tiến trình."""

    type: str  # "text" | "tool_call" | "tool_result" | "usage"
    content: str = ""
    tool_name: str = ""
    arguments: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _extract_usage(chunk) -> tuple[int, int] | None:
    """Đọc usage do Groq đính kèm cuối stream (nếu có), an toàn với mọi phiên bản SDK."""
    for container in (getattr(chunk, "x_groq", None), chunk):
        usage = getattr(container, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            if prompt_tokens is not None or completion_tokens is not None:
                return int(prompt_tokens or 0), int(completion_tokens or 0)
    return None


@dataclass
class _ToolCallAccumulator:
    """Gom các mảnh tool call rơi rải rác trong stream thành lời gọi hoàn chỉnh."""

    id: str = ""
    name: str = ""
    arguments: str = ""

    def merge(self, delta_call) -> None:
        if getattr(delta_call, "id", None):
            self.id = delta_call.id
        function = getattr(delta_call, "function", None)
        if function is not None:
            if getattr(function, "name", None):
                self.name = function.name
            if getattr(function, "arguments", None):
                self.arguments += function.arguments


class AgentService:
    def __init__(
        self,
        chat_service: ChatService,
        tools: list[Tool],
        max_steps: int | None = None,
    ) -> None:
        self.chat_service = chat_service
        self.tools = tools
        self.max_steps = max_steps or chat_service.config.agent_max_steps

    def run_stream(
        self,
        conversation_messages: list[ChatMessage],
        system_prompt: str = SYSTEM_PROMPT,
        settings: GenerationSettings | None = None,
    ):
        """Chạy agent, yield AgentEvent theo thời gian thực."""
        settings = self.chat_service._resolve_settings(settings)
        messages = self.chat_service._prepare_messages(conversation_messages, system_prompt)

        # Bật khi model sinh tool call hỏng: vòng kế tiếp chạy không tool
        # để buộc trả lời bằng văn bản thay vì lỗi cả lượt chat.
        force_plain_answer = False

        # Model nhỏ hay gọi lại cùng một tool với cùng tham số nhiều vòng liền.
        # Cache kết quả trong lượt này để trả ngay kèm nhắc nhở, đỡ tốn token.
        executed_calls: dict[tuple[str, str], str] = {}

        for step in range(self.max_steps):
            # Bước cuối không đưa tool: buộc model trả lời dứt điểm.
            offer_tools = (
                bool(self.tools) and step < self.max_steps - 1 and not force_plain_answer
            )

            request_kwargs = {
                "model": settings.model,
                "messages": messages,
                "temperature": settings.temperature,
                "max_tokens": settings.max_tokens,
                "stream": True,
            }
            if offer_tools:
                request_kwargs["tools"] = tool_schemas(self.tools)
                request_kwargs["tool_choice"] = "auto"

            stream = self.chat_service._create_completion(request_kwargs)

            text_parts: list[str] = []
            pending_calls: dict[int, _ToolCallAccumulator] = {}
            round_usage: tuple[int, int] | None = None

            try:
                for chunk in stream:
                    usage = _extract_usage(chunk)
                    if usage is not None:
                        round_usage = usage

                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    if delta is not None and delta.content:
                        text_parts.append(delta.content)
                        yield AgentEvent(type="text", content=delta.content)

                    for delta_call in (delta.tool_calls or []) if delta is not None else []:
                        index = getattr(delta_call, "index", 0) or 0
                        pending_calls.setdefault(index, _ToolCallAccumulator()).merge(delta_call)
            except RETRYABLE_EXCEPTIONS as exc:
                raise ChatServiceError(
                    f"Kết nối bị ngắt giữa chừng khi đang nhận câu trả lời: {_friendly_error_message(exc)}"
                ) from exc
            except APIError as exc:
                # Model sinh tool call không hợp lệ (hay gặp ở model nhỏ).
                # Chưa phát chữ nào ra ngoài -> an toàn để thử lại một vòng
                # không kèm tool, buộc model trả lời bằng văn bản.
                if not text_parts and not force_plain_answer:
                    force_plain_answer = True
                    continue
                raise ChatServiceError(
                    f"Model không tạo được lời gọi công cụ hợp lệ: {exc}"
                ) from exc

            if round_usage is not None:
                yield AgentEvent(
                    type="usage",
                    prompt_tokens=round_usage[0],
                    completion_tokens=round_usage[1],
                )

            if not pending_calls:
                return  # Model đã trả lời xong, không cần tool.

            # Ghi lại "ý định gọi tool" của model vào hội thoại.
            messages.append(
                {
                    "role": "assistant",
                    "content": "".join(text_parts),
                    "tool_calls": [
                        {
                            "id": call.id or f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments or "{}",
                            },
                        }
                        for index, call in sorted(pending_calls.items())
                    ],
                }
            )

            # Thực thi từng tool và nối kết quả vào hội thoại.
            for index, call in sorted(pending_calls.items()):
                yield AgentEvent(
                    type="tool_call",
                    tool_name=call.name,
                    arguments=call.arguments or "{}",
                )

                call_key = (call.name, call.arguments or "{}")
                if call_key in executed_calls:
                    result = (
                        "(Nhắc lại) Bạn đã gọi công cụ này với cùng tham số trong lượt này. "
                        f"Kết quả vẫn là:\n{executed_calls[call_key]}\n"
                        "Hãy dùng kết quả trên để trả lời ngay, đừng gọi lại công cụ nữa."
                    )
                else:
                    result = execute_tool(self.tools, call.name, call.arguments or "{}")
                    executed_calls[call_key] = result

                yield AgentEvent(type="tool_result", tool_name=call.name, content=result)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id or f"call_{index}",
                        "content": result,
                    }
                )
