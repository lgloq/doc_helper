from __future__ import annotations

import time
from typing import Any

from openai import APIConnectionError, APITimeoutError, OpenAI

from app.core.config import Settings

OPENAI_COMPATIBLE_PROVIDER_NAMES = {"openai", "openai_compatible"}


def uses_openai_compatible_provider(provider_name: str) -> bool:
    return provider_name.strip().lower() in OPENAI_COMPATIBLE_PROVIDER_NAMES


def has_openai_compatible_credentials(settings: Settings) -> bool:
    return bool(settings.effective_llm_api_key)


def create_openai_compatible_client(settings: Settings) -> OpenAI:
    client_kwargs: dict[str, Any] = {
        "api_key": settings.effective_llm_api_key or "",
        "max_retries": 0,
    }
    if settings.effective_llm_base_url:
        client_kwargs["base_url"] = settings.effective_llm_base_url
    return OpenAI(**client_kwargs)


def request_chat_completion(
    client: OpenAI,
    /,
    *,
    max_attempts: int = 4,
    retry_delay_seconds: float = 1.2,
    **kwargs: Any,
) -> Any:
    kwargs.setdefault("timeout", 90.0)
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except (APIConnectionError, APITimeoutError) as error:
            last_error = error
            if attempt >= max_attempts:
                raise
            time.sleep(retry_delay_seconds * attempt)
    if last_error is not None:
        raise last_error
    raise RuntimeError("request_chat_completion exited without response or error")
