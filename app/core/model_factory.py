from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain.chat_models import init_chat_model

from app.core.settings import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PROVIDER,
    DEFAULT_TEMPERATURE,
    OLLAMA_BASE_URL,
)


@lru_cache(maxsize=32)
def get_chat_model(
    model_name: str = DEFAULT_MODEL_NAME,
    provider: str = DEFAULT_MODEL_PROVIDER,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Any:
    """Build and cache chat models by parameter tuple."""
    return init_chat_model(
        model_name,
        model_provider=provider,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def clear_model_cache() -> None:
    """Clear cached model instances, useful after runtime config updates."""
    get_chat_model.cache_clear()