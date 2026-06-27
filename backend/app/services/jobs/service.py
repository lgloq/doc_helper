from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import UUID

from arq import create_pool
from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.enums import IngestStatus, RoleName
from app.models.operation_job import OperationJob
from app.models.user import User
from app.repositories.document_repository import DocumentRepository
from app.repositories.operation_job_repository import OperationJobRepository
from app.repositories.user_repository import UserRepository
from app.schemas.chat import ChatMessageCreate
from app.schemas.document import DocumentDiffRequest, DocumentIngestRequest
from app.schemas.eval import EvalRunRequest
from app.schemas.operation_job import OperationJobRead
from app.services.chat.service import ChatService
from app.services.diff.service import DocumentDiffService
from app.services.eval.service import EvalService
from app.services.ingestion.service import DocumentIngestionService
from app.services.permissions.service import PermissionFilterBuilder
from app.workers.settings import (
    ARQ_QUEUE_CHAT,
    ARQ_QUEUE_DIFF,
    ARQ_QUEUE_EVAL,
    ARQ_QUEUE_INGEST,
    get_arq_redis_settings,
)

logger = logging.getLogger(__name__)

JOB_TYPE_CHAT_MESSAGE = "chat_message"
JOB_TYPE_DOCUMENT_DIFF_SUMMARY = "document_diff_summary"
JOB_TYPE_DOCUMENT_INGEST = "document_ingest"
JOB_TYPE_EVAL_RUN = "eval_run"

JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"


class OperationJobService:
    def __init__(self, session: Session):
        self.session = session
        self.jobs = OperationJobRepository(session)
        self.users = UserRepository(session)
        self.documents = DocumentRepository(session)
        self.permission_builder = PermissionFilterBuilder()

    async def enqueue_chat_message(self, actor: User, session_id: UUID, payload: ChatMessageCreate) -> OperationJobRead:
        client_request_id = self._normalize_client_request_id(payload.client_request_id)
        request_payload = {
            "session_id": str(session_id),
            "content": payload.content,
            "top_k": payload.top_k,
            "client_request_id": client_request_id,
        }
        job = self._find_existing_job(JOB_TYPE_CHAT_MESSAGE, actor.id, client_request_id)
        if job is not None:
            return self._serialize_existing_job_or_raise(
                job,
                resource_type="chat_session",
                resource_id=str(session_id),
                expected_payload=request_payload,
            )

        job, created = self._create_job(
            actor,
            JOB_TYPE_CHAT_MESSAGE,
            client_request_id=client_request_id,
            resource_type="chat_session",
            resource_id=str(session_id),
            request_payload=request_payload,
        )
        if not created:
            return self._serialize_existing_job_or_raise(
                job,
                resource_type="chat_session",
                resource_id=str(session_id),
                expected_payload=request_payload,
            )
        return await self._enqueue(job)

    async def enqueue_document_diff_summary(
        self,
        actor: User,
        document_id: UUID,
        payload: DocumentDiffRequest,
    ) -> OperationJobRead:
        client_request_id = self._normalize_client_request_id(getattr(payload, "client_request_id", None))
        request_payload = {
            "document_id": str(document_id),
            "from_version_id": str(payload.from_version_id),
            "to_version_id": str(payload.to_version_id),
            "force_refresh": payload.force_refresh,
            "client_request_id": client_request_id,
        }
        job = self._find_existing_job(JOB_TYPE_DOCUMENT_DIFF_SUMMARY, actor.id, client_request_id)
        if job is not None:
            return self._serialize_existing_job_or_raise(
                job,
                resource_type="document",
                resource_id=str(document_id),
                expected_payload=request_payload,
            )

        job, created = self._create_job(
            actor,
            JOB_TYPE_DOCUMENT_DIFF_SUMMARY,
            client_request_id=client_request_id,
            resource_type="document",
            resource_id=str(document_id),
            request_payload=request_payload,
        )
        if not created:
            return self._serialize_existing_job_or_raise(
                job,
                resource_type="document",
                resource_id=str(document_id),
                expected_payload=request_payload,
            )
        return await self._enqueue(job)

    async def enqueue_document_ingest(
        self,
        actor: User,
        document_id: UUID,
        payload: DocumentIngestRequest | None = None,
    ) -> OperationJobRead:
        document = self._get_manageable_document(actor, document_id)
        version = self._resolve_version(document, payload.version_id if payload else None)
        client_request_id = self._normalize_client_request_id(getattr(payload, "client_request_id", None))
        request_payload = {
            "document_id": str(document.id),
            "document_version_id": str(version.id),
            "client_request_id": client_request_id,
        }
        job = self._find_existing_job(JOB_TYPE_DOCUMENT_INGEST, actor.id, client_request_id)
        if job is not None:
            return self._serialize_existing_job_or_raise(
                job,
                resource_type="document_version",
                resource_id=str(version.id),
                expected_payload=request_payload,
            )
        if version.ingest_status == IngestStatus.PROCESSING:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="该版本正在进行入库处理，请勿重复提交。",
            )

        previous_ingest_status = version.ingest_status
        previous_ingest_error = version.ingest_error
        version.ingest_status = IngestStatus.PROCESSING
        version.ingest_error = None
        self.session.commit()

        job, created = self._create_job(
            actor,
            JOB_TYPE_DOCUMENT_INGEST,
            client_request_id=client_request_id,
            resource_type="document_version",
            resource_id=str(version.id),
            request_payload={
                **request_payload,
                "previous_ingest_status": previous_ingest_status.value,
                "previous_ingest_error": previous_ingest_error,
            },
        )
        if not created:
            try:
                return self._serialize_existing_job_or_raise(
                    job,
                    resource_type="document_version",
                    resource_id=str(version.id),
                    expected_payload=request_payload,
                )
            except Exception:
                version.ingest_status = previous_ingest_status
                version.ingest_error = previous_ingest_error
                self.session.commit()
                raise
        try:
            return await self._enqueue(job)
        except Exception:
            version.ingest_status = previous_ingest_status
            version.ingest_error = previous_ingest_error
            self.session.commit()
            raise

    async def enqueue_eval_run(self, actor: User, payload: EvalRunRequest) -> OperationJobRead:
        client_request_id = self._normalize_client_request_id(payload.client_request_id)
        request_payload = {
            "dataset_name": payload.dataset_name,
            "case_ids": [str(item) for item in payload.case_ids],
            "top_k": payload.top_k,
            "seed_demo_cases": payload.seed_demo_cases,
            "client_request_id": client_request_id,
        }
        job = self._find_existing_job(JOB_TYPE_EVAL_RUN, actor.id, client_request_id)
        if job is not None:
            return self._serialize_existing_job_or_raise(
                job,
                resource_type="eval_run",
                resource_id=job.resource_id,
                expected_payload=request_payload,
            )

        eval_service = EvalService(self.session)
        run_detail, should_enqueue = eval_service.create_queued_run(actor, payload)
        if not should_enqueue:
            job, created = self._create_job(
                actor,
                JOB_TYPE_EVAL_RUN,
                client_request_id=client_request_id,
                resource_type="eval_run",
                resource_id=str(run_detail.id),
                request_payload={
                    **request_payload,
                    "run_id": str(run_detail.id),
                },
            )
            if not created:
                return self._serialize_existing_job_or_raise(
                    job,
                    resource_type="eval_run",
                    resource_id=job.resource_id,
                    expected_payload=request_payload,
                )
            job.status = self._map_eval_status(run_detail.status)
            job.result_payload = run_detail.model_dump(mode="json")
            job.error_text = run_detail.error_text
            if job.status in {JOB_STATUS_COMPLETED, JOB_STATUS_FAILED}:
                job.finished_at = datetime.now(UTC)
            self.session.commit()
            self.session.refresh(job)
            return self._serialize(job)

        job, created = self._create_job(
            actor,
            JOB_TYPE_EVAL_RUN,
            client_request_id=client_request_id,
            resource_type="eval_run",
            resource_id=str(run_detail.id),
            request_payload={
                **request_payload,
                "run_id": str(run_detail.id),
            },
        )
        if not created:
            eval_service.discard_unstarted_queued_run(run_detail.id)
            return self._serialize_existing_job_or_raise(
                job,
                resource_type="eval_run",
                resource_id=job.resource_id,
                expected_payload=request_payload,
            )
        return await self._enqueue(job)

    def get_job(self, actor: User, job_id: UUID) -> OperationJobRead:
        job = self._require_job(actor, job_id)
        return self._serialize(job)

    async def execute_job(self, job_id: UUID) -> OperationJobRead:
        job = self.jobs.get_by_id(job_id)
        if job is None:
            raise RuntimeError(f"Operation job {job_id} was not found.")
        if job.status in {JOB_STATUS_COMPLETED, JOB_STATUS_FAILED}:
            return self._serialize(job)

        job.status = JOB_STATUS_RUNNING
        job.running_at = job.running_at or datetime.now(UTC)
        self.session.commit()

        try:
            result = await self._dispatch(job)
            result_payload = self._serialize_result(result)
            job.result_payload = result_payload
            job.status = self._derive_completion_status(job.job_type, result_payload)
            job.error_text = self._derive_result_error_text(job.job_type, result_payload)
        except Exception as exc:
            logger.exception("Operation job failed: job_id=%s job_type=%s", job.id, job.job_type)
            job.status = JOB_STATUS_FAILED
            job.error_text = self._exc_message(exc)
            raise
        finally:
            if job.status in {JOB_STATUS_COMPLETED, JOB_STATUS_FAILED}:
                job.finished_at = datetime.now(UTC)
            self.session.commit()
        return self._serialize(job)

    def mark_enqueue_failed(self, job_id: UUID, error_text: str) -> OperationJobRead:
        job = self.jobs.get_by_id(job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Operation job not found.")
        job.status = JOB_STATUS_FAILED
        job.finished_at = datetime.now(UTC)
        job.error_text = error_text[:1000]
        self.session.commit()
        self.session.refresh(job)
        return self._serialize(job)

    async def _enqueue(self, job: OperationJob) -> OperationJobRead:
        redis = None
        try:
            redis = await create_pool(get_arq_redis_settings())
            arq_job = await redis.enqueue_job(
                "run_operation_job",
                str(job.id),
                _queue_name=self._queue_name_for_job_type(job.job_type),
            )
            if arq_job is None:
                raise RuntimeError("ARQ did not return a job id.")
            job.arq_job_id = arq_job.job_id
            self.session.commit()
            self.session.refresh(job)
            return self._serialize(job)
        except Exception as exc:
            logger.exception("Failed to enqueue operation job: job_type=%s job_id=%s", job.job_type, job.id)
            if job.job_type == JOB_TYPE_EVAL_RUN and job.resource_id:
                try:
                    EvalService(self.session).mark_run_enqueue_failed(
                        UUID(str(job.resource_id)),
                        "异步评测任务提交失败，请稍后重试。",
                    )
                except Exception:
                    logger.exception("Failed to reconcile eval run after enqueue failure: job_id=%s", job.id)
            self.mark_enqueue_failed(job.id, "后台任务提交失败，请稍后重试。")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="后台任务提交失败，请稍后重试。",
            ) from exc
        finally:
            if redis is not None:
                await redis.aclose()

    async def _dispatch(self, job: OperationJob):
        if job.job_type == JOB_TYPE_CHAT_MESSAGE:
            return self._run_chat_message(job)
        if job.job_type == JOB_TYPE_DOCUMENT_DIFF_SUMMARY:
            return self._run_document_diff_summary(job)
        if job.job_type == JOB_TYPE_DOCUMENT_INGEST:
            return self._run_document_ingest(job)
        if job.job_type == JOB_TYPE_EVAL_RUN:
            return self._run_eval_run(job)
        raise RuntimeError(f"Unsupported job type: {job.job_type}")

    @staticmethod
    def _queue_name_for_job_type(job_type: str) -> str:
        if job_type == JOB_TYPE_CHAT_MESSAGE:
            return ARQ_QUEUE_CHAT
        if job_type == JOB_TYPE_DOCUMENT_DIFF_SUMMARY:
            return ARQ_QUEUE_DIFF
        if job_type == JOB_TYPE_DOCUMENT_INGEST:
            return ARQ_QUEUE_INGEST
        if job_type == JOB_TYPE_EVAL_RUN:
            return ARQ_QUEUE_EVAL
        raise RuntimeError(f"Unsupported job type: {job_type}")

    def _run_chat_message(self, job: OperationJob):
        actor = self._require_user(job.user_id)
        session_id = UUID(str(job.request_payload["session_id"]))
        payload = ChatMessageCreate(
            content=str(job.request_payload["content"]),
            top_k=int(job.request_payload.get("top_k") or 5),
            client_request_id=job.client_request_id,
        )
        return ChatService(self.session).create_message(actor, session_id, payload, allow_inflight_client_request=True)

    def _run_document_diff_summary(self, job: OperationJob):
        actor = self._require_user(job.user_id)
        document_id = UUID(str(job.request_payload["document_id"]))
        payload = DocumentDiffRequest(
            from_version_id=UUID(str(job.request_payload["from_version_id"])),
            to_version_id=UUID(str(job.request_payload["to_version_id"])),
            force_refresh=bool(job.request_payload.get("force_refresh", False)),
        )
        return DocumentDiffService(self.session).summarize_diff(actor, document_id, payload)

    def _run_document_ingest(self, job: OperationJob):
        actor = self._require_user(job.user_id)
        document_id = UUID(str(job.request_payload["document_id"]))
        payload = DocumentIngestRequest(version_id=UUID(str(job.request_payload["document_version_id"])))
        return DocumentIngestionService(self.session).ingest_document(actor, document_id, payload)

    def _run_eval_run(self, job: OperationJob):
        run_id = UUID(str(job.request_payload["run_id"]))
        return EvalService(self.session).execute_queued_run(run_id)

    def _create_job(
        self,
        actor: User,
        job_type: str,
        *,
        client_request_id: str | None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        request_payload: dict,
    ) -> tuple[OperationJob, bool]:
        job = OperationJob(
            job_type=job_type,
            status=JOB_STATUS_QUEUED,
            user_id=actor.id,
            client_request_id=client_request_id,
            resource_type=resource_type,
            resource_id=resource_id,
            request_payload=request_payload,
        )
        self.jobs.add(job)
        try:
            self.session.commit()
            self.session.refresh(job)
            return job, True
        except IntegrityError as exc:
            self.session.rollback()
            existing = self._find_existing_job(job_type, actor.id, client_request_id)
            if existing is not None:
                return existing, False
            raise exc

    def _find_existing_job(self, job_type: str, user_id: UUID, client_request_id: str | None) -> OperationJob | None:
        if not client_request_id:
            return None
        return self.jobs.find_by_client_request_id(job_type, user_id, client_request_id)

    def _serialize_existing_job_or_raise(
        self,
        job: OperationJob,
        *,
        resource_type: str,
        resource_id: str | None,
        expected_payload: dict,
    ) -> OperationJobRead:
        if job.resource_type != resource_type or str(job.resource_id or "") != str(resource_id or ""):
            raise self._client_request_conflict()
        actual_payload = job.request_payload or {}
        for key, expected_value in expected_payload.items():
            if actual_payload.get(key) != expected_value:
                raise self._client_request_conflict()
        return self._serialize(job)

    @staticmethod
    def _client_request_conflict() -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="client_request_id 已用于另一项后台任务请求。",
        )

    def _require_job(self, actor: User, job_id: UUID) -> OperationJob:
        job = self.jobs.get_by_id(job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Operation job not found.")
        if self._is_admin(actor):
            return job
        if job.user_id != actor.id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Operation job not found.")
        return job

    def _require_user(self, user_id: UUID) -> User:
        user = self.users.get_by_id(user_id)
        if user is None:
            raise RuntimeError(f"User {user_id} not found for operation job execution.")
        return user

    @staticmethod
    def _normalize_client_request_id(value: str | None) -> str | None:
        normalized = (value or "").strip()
        return normalized or None

    @staticmethod
    def _is_admin(actor: User) -> bool:
        return bool(actor.role and actor.role.name == RoleName.ADMIN)

    @staticmethod
    def _serialize(job: OperationJob) -> OperationJobRead:
        return OperationJobRead.model_validate(job)

    @staticmethod
    def _serialize_result(result) -> dict | None:
        if result is None:
            return None
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        if isinstance(result, dict):
            return result
        return {"value": result}

    @staticmethod
    def _map_eval_status(value: str) -> str:
        if value == "completed":
            return JOB_STATUS_COMPLETED
        if value == "failed":
            return JOB_STATUS_FAILED
        if value == "running":
            return JOB_STATUS_RUNNING
        return JOB_STATUS_QUEUED

    @staticmethod
    def _derive_completion_status(job_type: str, result_payload: dict | None) -> str:
        if job_type == JOB_TYPE_EVAL_RUN and isinstance(result_payload, dict):
            return OperationJobService._map_eval_status(str(result_payload.get("status") or "queued"))
        return JOB_STATUS_COMPLETED

    @staticmethod
    def _derive_result_error_text(job_type: str, result_payload: dict | None) -> str | None:
        if job_type == JOB_TYPE_EVAL_RUN and isinstance(result_payload, dict):
            error_text = result_payload.get("error_text")
            return str(error_text)[:1000] if error_text else None
        return None

    @staticmethod
    def _exc_message(exc: Exception) -> str:
        if isinstance(exc, HTTPException):
            detail = exc.detail
            return str(detail)[:1000] if detail else "Operation job failed."
        return str(exc)[:1000] if str(exc) else "Operation job failed."

    def _resolve_version(self, document: Document, version_id: UUID | None):
        document_service = DocumentIngestionService(self.session)
        return document_service._resolve_version(document, version_id)

    def _get_manageable_document(self, actor: User, document_id: UUID) -> Document:
        document_service = DocumentIngestionService(self.session)
        return document_service._get_manageable_document(actor, document_id)
