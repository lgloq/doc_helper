from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.redis import close_redis_client
from app.services.auth.bootstrap import seed_mock_data
from app.services.eval.bootstrap import seed_demo_eval_cases

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    try:
        seed_mock_data()
    except Exception:
        logger.exception("Failed to seed mock auth data during startup.")
    try:
        seed_demo_eval_cases()
    except Exception:
        logger.exception("Failed to seed demo eval cases during startup.")
    yield
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
