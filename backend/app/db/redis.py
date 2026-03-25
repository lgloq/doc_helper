from __future__ import annotations

from functools import lru_cache

from redis import Redis
from redis.exceptions import RedisError

from app.core.config import get_settings


@lru_cache
def get_redis_client() -> Redis:
    settings = get_settings()
    return Redis.from_url(settings.redis_url, decode_responses=True)


def check_redis_connection() -> bool:
    try:
        return bool(get_redis_client().ping())
    except RedisError:
        return False


def close_redis_client() -> None:
    try:
        get_redis_client().close()
    finally:
        get_redis_client.cache_clear()
