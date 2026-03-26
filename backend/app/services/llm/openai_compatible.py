from __future__ import annotations

from openai import OpenAI

from app.core.config import Settings

OPENAI_COMPATIBLE_PROVIDER_NAMES = {"openai", "openai_compatible"}


def uses_openai_compatible_provider(provider_name: str) -> bool:
    return provider_name.strip().lower() in OPENAI_COMPATIBLE_PROVIDER_NAMES


def has_openai_compatible_credentials(settings: Settings) -> bool:
    return bool(settings.effective_llm_api_key)


def create_openai_compatible_client(settings: Settings) -> OpenAI:
    client_kwargs: dict[str, str] = {"api_key": settings.effective_llm_api_key or ""}
    if settings.effective_llm_base_url:
        client_kwargs["base_url"] = settings.effective_llm_base_url
    return OpenAI(**client_kwargs)
