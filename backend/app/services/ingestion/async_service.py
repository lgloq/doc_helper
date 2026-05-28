"""异步文档入库服务。

封装 ARQ 任务提交，提供从 API 层到 Worker 的桥接。
"""

from __future__ import annotations

import logging
from uuid import UUID

from arq import create_pool
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.enums import IngestStatus
from app.models.user import User
from app.repositories.document_repository import DocumentRepository
from app.services.permissions.service import PermissionFilterBuilder
from app.workers.settings import get_arq_redis_settings

logger = logging.getLogger(__name__)

TASK_STATUS_KEY_PREFIX = "ingest_task:"
TASK_STATUS_TTL = 3600  # 任务状态保留 1 小时


class AsyncIngestionService:
    def __init__(self, session: Session):
        self.session = session
        self.document_repository = DocumentRepository(session)
        self.permission_builder = PermissionFilterBuilder()

    async def enqueue_ingest(
        self,
        actor: User,
        document_id: UUID,
        version_id: UUID | None = None,
    ) -> dict:
        """将文档入库任务提交到 ARQ 队列，立即返回任务信息。"""
        document = self._get_manageable_document(actor, document_id)
        version = self._resolve_version(document, version_id)

        # 检查是否已有进行中的任务
        if version.ingest_status == IngestStatus.PROCESSING:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="该版本正在进行入库处理，请勿重复提交。",
            )

        previous_ingest_status = version.ingest_status
        previous_ingest_error = version.ingest_error

        # 更新状态为 PROCESSING
        version.ingest_status = IngestStatus.PROCESSING
        version.ingest_error = None
        self.session.commit()

        # 提交 ARQ 异步任务
        redis = None
        try:
            redis = await create_pool(get_arq_redis_settings())
            job = await redis.enqueue_job(
                "run_ingest_document",
                str(document.id),
                str(version.id),
            )
            if job is None:
                raise RuntimeError("ARQ 未返回任务 ID。")

            logger.info("已提交异步入库任务: job_id=%s document=%s version=%s", job.job_id, document.id, version.id)
        except Exception as exc:
            logger.exception("提交异步入库任务失败: document=%s version=%s", document.id, version.id)
            self.session.rollback()
            version.ingest_status = previous_ingest_status
            version.ingest_error = previous_ingest_error
            self.session.commit()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="异步入库任务提交失败，请稍后重试。",
            ) from exc
        else:
            try:
                # 在 Redis 中记录任务关联信息，便于状态查询；失败不影响已提交的 ARQ 任务。
                await redis.set(
                    f"{TASK_STATUS_KEY_PREFIX}{version.id}",
                    job.job_id,
                    ex=TASK_STATUS_TTL,
                )
            except Exception:
                logger.exception("记录异步入库任务状态失败: job_id=%s version=%s", job.job_id, version.id)
        finally:
            if redis is not None:
                await redis.aclose()

        return {
            "document_id": document.id,
            "document_version_id": version.id,
            "job_id": job.job_id,
            "ingest_status": IngestStatus.PROCESSING,
            "message": "文档已提交异步处理，正在后台执行解析和入库。",
        }

    async def get_task_status(self, version_id: UUID) -> dict | None:
        """查询异步任务状态。"""
        redis = await create_pool(get_arq_redis_settings())
        try:
            job_id = await redis.get(f"{TASK_STATUS_KEY_PREFIX}{version_id}")
            if job_id is None:
                return None

            # 从 ARQ 获取任务结果
            job_result = await redis._redis.hgetall(f"arq:result:{job_id}")
            if not job_result:
                return {"job_id": job_id, "status": "processing"}

            return {"job_id": job_id, "status": "completed", "result": job_result}
        finally:
            await redis.aclose()

    def _get_manageable_document(self, actor: User, document_id: UUID):
        visibility_query = self.permission_builder.build_accessible_document_ids_query(actor, require_manage=True)
        document = self.document_repository.get_visible_by_id(document_id, visibility_query)
        if document is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
        return document

    def _resolve_version(self, document, version_id: UUID | None):
        version = None
        if version_id is not None:
            version = self.document_repository.get_version(document.id, version_id)
        elif document.current_version_id is not None:
            version = self.document_repository.get_version(document.id, document.current_version_id)
        if version is None:
            version = self.document_repository.get_latest_version(document.id)
        if version is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document version not found.")
        return version
