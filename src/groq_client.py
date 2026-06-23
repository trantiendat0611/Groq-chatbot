from groq import Groq

from src.config import AppConfig, load_config


def build_client(config: AppConfig | None = None) -> Groq:
    config = config or load_config()
    return Groq(api_key=config.groq_api_key)
