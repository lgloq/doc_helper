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
    langfuse_generation_name: str | None = None,
    langfuse_trace_id: str | None = None,
    langfuse_metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    from app.services.observability.langfuse_adapter import LangfuseAdapter

    kwargs.setdefault("timeout", 90.0)
    model_name = kwargs.get("model")
    messages = kwargs.get("messages")

    generation = None
    if langfuse_generation_name:
        _, generation = LangfuseAdapter.create_generation(
            trace_id=langfuse_trace_id,
            name=langfuse_generation_name,
            model=model_name,
            messages=messages,
            metadata=langfuse_metadata,
        )

    started = time.perf_counter()
    last_error: Exception | None = None
    error_text: str | None = None
    response = None
    try:
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.chat.completions.create(**kwargs)
                break
            except (APIConnectionError, APITimeoutError) as error:
                last_error = error
                if attempt >= max_attempts:
                    raise
                time.sleep(retry_delay_seconds * attempt)
        if response is None and last_error is not None:
            raise last_error
        if response is None:
            raise RuntimeError("request_chat_completion exited without response or error")
    except Exception as exc:
        error_text = str(exc)
        raise
    finally:
        if generation is not None:
            latency_ms = int((time.perf_counter() - started) * 1000)
            usage = getattr(response, "usage", None) if response else None
            output_content = None
            if response and response.choices:
                output_content = response.choices[0].message.content
            LangfuseAdapter.end_generation(
                generation,
                output_content=output_content,
                latency_ms=latency_ms,
                prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
                model=model_name,
                error_text=error_text,
            )
    return response
