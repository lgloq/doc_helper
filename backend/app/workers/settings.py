from __future__ import annotations

from arq.connections import RedisSettings

from app.core.config import get_settings

ARQ_QUEUE_DEFAULT = "arq:queue"
ARQ_QUEUE_CHAT = "arq:queue:chat"
ARQ_QUEUE_DIFF = "arq:queue:diff"
ARQ_QUEUE_EVAL = "arq:queue:eval"
ARQ_QUEUE_INGEST = "arq:queue:ingest"


def get_arq_redis_settings() -> RedisSettings:
    """从全局配置派生 ARQ 所需的 Redis 连接参数。"""
    from urllib.parse import urlparse

    url = urlparse(get_settings().redis_url)
    return RedisSettings(
        host=url.hostname or "localhost",
        port=url.port or 6379,
        database=int((url.path or "/0").lstrip("/") or 0),
        password=url.password,
    )
