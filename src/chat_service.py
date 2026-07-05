import random
import time
from dataclasses import dataclass

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    Groq,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)

from src.chat_types import ChatMessage
from src.config import AppConfig, load_config
from src.context import trim_messages_to_budget
from src.groq_client import build_client
from src.prompts import SYSTEM_PROMPT


# Lỗi tạm thời: đáng thử lại vì lần sau có thể thành công.
RETRYABLE_EXCEPTIONS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)

RETRY_BASE_DELAY_SECONDS = 1.0
RETRY_MAX_DELAY_SECONDS = 20.0


class ChatServiceError(RuntimeError):
    """Lỗi gọi Groq API đã được diễn giải thành thông điệp dễ hiểu."""


def _friendly_error_message(exc: Exception) -> str:
    if isinstance(exc, AuthenticationError):
        return "API key không hợp lệ hoặc đã hết hạn. Kiểm tra GROQ_API_KEY trong file .env."
    if isinstance(exc, RateLimitError):
        return "Đã chạm giới hạn tần suất của Groq (rate limit). Hãy đợi một lát rồi thử lại."
    if isinstance(exc, NotFoundError):
        return "Model không tồn tại hoặc tài khoản của bạn không truy cập được model này."
    if isinstance(exc, BadRequestError):
        return f"Yêu cầu không hợp lệ (thường do model/tham số sai): {exc}"
    if isinstance(exc, APITimeoutError):
        return "Groq API phản hồi quá chậm (timeout). Đã thử lại nhưng không thành công."
    if isinstance(exc, APIConnectionError):
        return "Không kết nối được tới Groq API. Kiểm tra mạng rồi thử lại."
    if isinstance(exc, InternalServerError):
        return "Groq đang gặp sự cố phía máy chủ. Đã thử lại nhưng không thành công."
    if isinstance(exc, APIStatusError):
        return f"Groq API trả về lỗi {exc.status_code}: {exc}"
    return f"Lỗi không xác định khi gọi Groq API: {exc}"


def _retry_after_seconds(exc: Exception) -> float | None:
    """Đọc header Retry-After nếu server cho biết cần đợi bao lâu."""
    response = getattr(exc, "response", None)
    if response is None:
        return None

    value = response.headers.get("retry-after")
    if value is None:
        return None

    try:
        return min(float(value), RETRY_MAX_DELAY_SECONDS)
    except ValueError:
        return None


@dataclass(frozen=True)
class GenerationSettings:
    model: str
    temperature: float
    max_tokens: int


def build_groq_messages(
    conversation_messages: list[ChatMessage],
    system_prompt: str = SYSTEM_PROMPT,
) -> list[ChatMessage]:
    return [
        {
            "role": "system",
            "content": system_prompt.strip(),
        },
        *conversation_messages,
    ]


class ChatService:
    def __init__(
        self,
        client: Groq | None = None,
        config: AppConfig | None = None,
    ) -> None:
        self.config = config or load_config()
        self.client = client or build_client(self.config)
        # Tách ra để test có thể thay bằng hàm giả, không phải đợi thật.
        self._sleep = time.sleep

    def default_settings(self) -> GenerationSettings:
        return GenerationSettings(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

    def _resolve_settings(
        self,
        settings: GenerationSettings | None = None,
    ) -> GenerationSettings:
        return settings or self.default_settings()

    def _prepare_messages(
        self,
        conversation_messages: list[ChatMessage],
        system_prompt: str,
    ) -> list[ChatMessage]:
        trimmed = trim_messages_to_budget(
            conversation_messages,
            max_history_tokens=self.config.context_tokens,
        )
        return build_groq_messages(trimmed, system_prompt)

    def _backoff_delay(self, attempt: int, exc: Exception) -> float:
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            return retry_after

        delay = RETRY_BASE_DELAY_SECONDS * (2**attempt)
        jitter = random.uniform(0, delay / 2)
        return min(delay + jitter, RETRY_MAX_DELAY_SECONDS)

    def _create_completion(self, request_kwargs: dict):
        """Gọi API kèm retry với exponential backoff cho lỗi tạm thời."""
        last_exc: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return self.client.chat.completions.create(**request_kwargs)
            except RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                if attempt < self.config.max_retries:
                    self._sleep(self._backoff_delay(attempt, exc))
            except APIStatusError as exc:
                # Lỗi 4xx còn lại (sai model, sai tham số...): thử lại vô ích.
                raise ChatServiceError(_friendly_error_message(exc)) from exc

        raise ChatServiceError(_friendly_error_message(last_exc)) from last_exc

    def generate_reply(
        self,
        conversation_messages: list[ChatMessage],
        system_prompt: str = SYSTEM_PROMPT,
        settings: GenerationSettings | None = None,
    ) -> str:
        settings = self._resolve_settings(settings)
        response = self._create_completion(
            {
                "model": settings.model,
                "messages": self._prepare_messages(conversation_messages, system_prompt),
                "temperature": settings.temperature,
                "max_tokens": settings.max_tokens,
            }
        )

        return response.choices[0].message.content

    def stream_reply(
        self,
        conversation_messages: list[ChatMessage],
        system_prompt: str = SYSTEM_PROMPT,
        settings: GenerationSettings | None = None,
    ):
        settings = self._resolve_settings(settings)
        stream = self._create_completion(
            {
                "model": settings.model,
                "messages": self._prepare_messages(conversation_messages, system_prompt),
                "temperature": settings.temperature,
                "max_tokens": settings.max_tokens,
                "stream": True,
            }
        )

        # Stream đã bắt đầu thì không retry nữa: retry sẽ làm lặp nội dung.
        try:
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except RETRYABLE_EXCEPTIONS as exc:
            raise ChatServiceError(
                f"Kết nối bị ngắt giữa chừng khi đang nhận câu trả lời: {_friendly_error_message(exc)}"
            ) from exc
