from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.redis import close_redis_client
from app.services.auth.bootstrap import seed_mock_data
from app.services.eval.bootstrap import seed_demo_eval_cases
from app.services.observability.langfuse_adapter import get_langfuse_client, shutdown_langfuse

logger = logging.getLogger(__name__)


async def _start_arq_worker(app: FastAPI) -> None:
    """在 API 进程内运行 ARQ Worker，用于不启动独立 worker 的本地开发场景。"""
    try:
        from arq.worker import create_worker

        from app.workers.tasks import ChatWorkerSettings, DiffWorkerSettings, EvalWorkerSettings, IngestWorkerSettings

        worker_settings = [ChatWorkerSettings, DiffWorkerSettings, EvalWorkerSettings, IngestWorkerSettings]
        app.state.arq_workers = [create_worker(settings) for settings in worker_settings]
        app.state.arq_worker_tasks = [asyncio.create_task(worker.async_run()) for worker in app.state.arq_workers]
        logger.info("ARQ Workers 已在 API 进程内启动: %s", ", ".join(settings.queue_name for settings in worker_settings))
    except Exception:
        logger.exception("启动 ARQ Workers 失败")


async def _stop_arq_worker(app: FastAPI) -> None:
    workers = getattr(app.state, "arq_workers", [])
    for worker in workers:
        await worker.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    try:
        seed_mock_data()
    except Exception:
        logger.exception("Failed to seed mock auth data during startup.")
    try:
        seed_demo_eval_cases()
    except Exception:
        logger.exception("Failed to seed demo eval cases during startup.")

    # 初始化 Langfuse 可观测性
    try:
        get_langfuse_client()
    except Exception:
        logger.exception("Failed to initialize Langfuse client.")

    # 默认通过独立 worker 进程消费 ARQ 队列；只在显式启用时嵌入 API 进程。
    _is_sqlite = settings.database_url.startswith("sqlite")
    should_start_embedded_worker = settings.enable_embedded_worker and not _is_sqlite
    if should_start_embedded_worker:
        await _start_arq_worker(app)

    yield

    if should_start_embedded_worker:
        await _stop_arq_worker(app)
    shutdown_langfuse()
    close_redis_client()


def create_application() -> FastAPI:
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version=__version__,
        debug=settings.debug,
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(api_router, prefix=settings.api_v1_prefix)
    return application


app = create_application()
