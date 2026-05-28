from __future__ import annotations

import logging
import threading
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger("app.observability.langfuse")

_client_lock = threading.Lock()
_langfuse_client: Any | None = None


def get_langfuse_client() -> Any | None:
    """延迟初始化全局 Langfuse 客户端，线程安全。"""
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client
    settings = get_settings()
    if not settings.enable_langfuse or not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None
    with _client_lock:
        if _langfuse_client is not None:
            return _langfuse_client
        try:
            from langfuse import Langfuse

            client_kwargs: dict[str, Any] = {
                "public_key": settings.langfuse_public_key,
                "secret_key": settings.langfuse_secret_key,
            }
            if settings.langfuse_host:
                client_kwargs["host"] = settings.langfuse_host
            _langfuse_client = Langfuse(**client_kwargs)
            logger.info("Langfuse 客户端已初始化 host=%s", settings.langfuse_host or "cloud.langfuse.com")
        except Exception:
            logger.exception("Langfuse 客户端初始化失败")
            _langfuse_client = None
    return _langfuse_client


def shutdown_langfuse() -> None:
    """关闭 Langfuse 客户端，刷新未发送事件。"""
    global _langfuse_client
    if _langfuse_client is not None:
        try:
            _langfuse_client.flush()
        except Exception:
            logger.exception("Langfuse flush 失败")
        _langfuse_client = None


class LangfuseAdapter:
    """将 LLM 调用 trace 发送到 Langfuse 的适配器。"""

    def emit_trace(self, payload: dict[str, Any]) -> None:
        """基于 ObservabilityService 的 payload 创建 Langfuse trace。"""
        client = get_langfuse_client()
        if client is None:
            return
        try:
            trace_id = payload.get("trace_id")
            trace = client.trace(
                id=trace_id,
                name=payload.get("trace_type", "chat_qa"),
                user_id=payload.get("user_id"),
                session_id=payload.get("session_id"),
                metadata={
                    "model_name": payload.get("model_name"),
                    "confidence": (payload.get("trace_metadata") or {}).get("confidence"),
                    "insufficient_evidence": (payload.get("trace_metadata") or {}).get("insufficient_evidence"),
                    "error_text": payload.get("error_text"),
                },
                input={"query_text": payload.get("query_text")},
                output={"answer_basis": (payload.get("trace_metadata") or {}).get("confidence")},
            )
            # 在该 trace 下记录 generation span（LLM 调用详情）
            if payload.get("model_name") or payload.get("prompt_tokens"):
                trace.generation(
                    name="chat_completion",
                    model=payload.get("model_name"),
                    usage={
                        "input": payload.get("prompt_tokens") or 0,
                        "output": payload.get("completion_tokens") or 0,
                    },
                    latency_ms=payload.get("latency_ms"),
                    metadata={"source": "observability_service"},
                )
            client.flush()
        except Exception:
            logger.exception("Langfuse emit_trace 失败")

    @staticmethod
    def create_generation(
        *,
        trace_id: str | None = None,
        name: str = "llm_call",
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any | None, Any | None]:
        """创建 Langfuse generation span，返回 (trace, generation)。"""
        client = get_langfuse_client()
        if client is None:
            return None, None
        try:
            trace = client.trace(
                id=trace_id,
                name=name,
                metadata=metadata or {},
            )
            generation = trace.generation(
                name=name,
                model=model,
                input=messages,
            )
            return trace, generation
        except Exception:
            logger.exception("Langfuse create_generation 失败")
            return None, None

    @staticmethod
    def end_generation(
        generation: Any,
        *,
        output_content: str | None = None,
        latency_ms: int | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        model: str | None = None,
        error_text: str | None = None,
    ) -> None:
        """结束 Langfuse generation span。"""
        if generation is None:
            return
        try:
            usage: dict[str, int] = {}
            if prompt_tokens is not None:
                usage["input"] = prompt_tokens
            if completion_tokens is not None:
                usage["output"] = completion_tokens

            end_kwargs: dict[str, Any] = {}
            if output_content is not None:
                end_kwargs["output"] = output_content
            if usage:
                end_kwargs["usage"] = usage
            if model:
                end_kwargs["model"] = model
            if error_text:
                end_kwargs["level"] = "ERROR"
                end_kwargs["status_message"] = error_text

            generation.end(**end_kwargs)
        except Exception:
            logger.exception("Langfuse end_generation 失败")
