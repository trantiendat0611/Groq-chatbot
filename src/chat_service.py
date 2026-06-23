from dataclasses import dataclass

from groq import Groq

from src.config import AppConfig, load_config
from src.groq_client import build_client
from src.prompts import SYSTEM_PROMPT


ChatMessage = dict[str, str]


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

    def generate_reply(
        self,
        conversation_messages: list[ChatMessage],
        system_prompt: str = SYSTEM_PROMPT,
        settings: GenerationSettings | None = None,
    ) -> str:
        settings = self._resolve_settings(settings)
        response = self.client.chat.completions.create(
            model=settings.model,
            messages=build_groq_messages(conversation_messages, system_prompt),
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

        return response.choices[0].message.content

    def stream_reply(
        self,
        conversation_messages: list[ChatMessage],
        system_prompt: str = SYSTEM_PROMPT,
        settings: GenerationSettings | None = None,
    ):
        settings = self._resolve_settings(settings)
        stream = self.client.chat.completions.create(
            model=settings.model,
            messages=build_groq_messages(conversation_messages, system_prompt),
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
