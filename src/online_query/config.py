# Loads online question-answering configuration from .env files.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values, load_dotenv


@dataclass(frozen=True)
class ChatConfig:
    """OpenAI-compatible chat completion settings."""

    provider: str
    api_key: str
    base_url: str
    model: str
    timeout: float
    max_retries: int


@dataclass(frozen=True)
class RetrievalConfig:
    """V0 dense retrieval and context settings."""

    top_k: int
    max_context_chars: int


@dataclass(frozen=True)
class OnlineConfig:
    """Complete online QA configuration."""

    chat: ChatConfig
    retrieval: RetrievalConfig


def load_online_config(env_file: Path = Path(".env")) -> OnlineConfig:
    """Load online QA settings from a dotenv file."""

    env_file = Path(env_file)
    load_dotenv(env_file, override=False)
    values = dotenv_values(env_file)
    return OnlineConfig(
        chat=ChatConfig(
            provider=_get_str(values, "CHAT_PROVIDER", "openai-compatible"),
            api_key=_get_str(values, "CHAT_API_KEY", ""),
            base_url=_get_str_with_fallback(values, "CHAT_BASE_URL", ""),
            model=_get_str_with_fallback(values, "CHAT_MODEL", "qwen-plus"),
            timeout=_get_float(values, "CHAT_TIMEOUT_SECONDS", 60),
            max_retries=_get_int(values, "CHAT_MAX_RETRIES", 3),
        ),
        retrieval=RetrievalConfig(
            top_k=_get_int(values, "CHAT_RETRIEVAL_TOP_K", 5),
            max_context_chars=_get_int(values, "CHAT_MAX_CONTEXT_CHARS", 6000),
        ),
    )


def _get_str(values: dict, key: str, default: str) -> str:
    """Read a string value from parsed dotenv values."""

    value = values.get(key)
    if value is None or value == "":
        return default
    return str(value)


def _get_str_with_fallback(values: dict, key: str, default: str) -> str:
    """Read a string value, falling back to an older compatible key."""

    value = _get_str(values, key, "")
    return value


def _get_int(values: dict, key: str, default: int) -> int:
    """Read and convert an integer value from parsed dotenv values."""

    value = values.get(key)
    if value is None or value == "":
        return default
    return int(str(value).strip())


def _get_float(values: dict, key: str, default: float) -> float:
    """Read and convert a float value from parsed dotenv values."""

    value = values.get(key)
    if value is None or value == "":
        return default
    return float(str(value).strip())
