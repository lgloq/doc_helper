from __future__ import annotations

import logging
from uuid import UUID

from arq import create_pool
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.user import User
from app.schemas.eval import EvalRunDetailRead, EvalRunRequest
from app.services.eval.service import EvalService
from app.workers.settings import get_arq_redis_settings

logger = logging.getLogger(__name__)


class AsyncEvalService:
    def __init__(self, session: Session):
        self.session = session
        self.eval_service = EvalService(session)

    async def enqueue_eval(self, actor: User, payload: EvalRunRequest) -> EvalRunDetailRead:
        run_detail, should_enqueue = self.eval_service.create_queued_run(actor, payload)
        if not should_enqueue:
            return run_detail

        redis = None
        try:
            redis = await create_pool(get_arq_redis_settings())
            job = await redis.enqueue_job("run_eval_run", str(run_detail.id))
            if job is None:
                raise RuntimeError("ARQ did not return a job id.")
            logger.info("已提交异步评测任务: job_id=%s run_id=%s", job.job_id, run_detail.id)
            return self.eval_service.attach_job_to_run(UUID(str(run_detail.id)), job.job_id)
        except Exception as exc:
            logger.exception("提交异步评测任务失败: run_id=%s", run_detail.id)
            self.eval_service.mark_run_enqueue_failed(UUID(str(run_detail.id)), "异步评测任务提交失败，请稍后重试。")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="异步评测任务提交失败，请稍后重试。",
            ) from exc
        finally:
            if redis is not None:
                await redis.aclose()