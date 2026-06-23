import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 800


@dataclass(frozen=True)
class AppConfig:
    groq_api_key: str
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default

    try:
        return float(value)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default

    try:
        return int(value)
    except ValueError:
        return default


def load_config() -> AppConfig:
    load_dotenv(ENV_PATH)

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Chưa tìm thấy GROQ_API_KEY. Hãy mở file .env và điền API key của bạn."
        )

    return AppConfig(
        groq_api_key=api_key,
        model=os.getenv("GROQ_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        temperature=_get_float("GROQ_TEMPERATURE", DEFAULT_TEMPERATURE),
        max_tokens=_get_int("GROQ_MAX_TOKENS", DEFAULT_MAX_TOKENS),
    )
