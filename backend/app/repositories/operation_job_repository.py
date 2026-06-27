from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.operation_job import OperationJob


class OperationJobRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, job: OperationJob) -> OperationJob:
        self.session.add(job)
        return job

    def get_by_id(self, job_id: UUID) -> OperationJob | None:
        statement = select(OperationJob).where(OperationJob.id == job_id)
        return self.session.scalar(statement)

    def find_by_client_request_id(self, job_type: str, user_id: UUID, client_request_id: str) -> OperationJob | None:
        statement = (
            select(OperationJob)
            .where(
                OperationJob.job_type == job_type,
                OperationJob.user_id == user_id,
                OperationJob.client_request_id == client_request_id,
            )
            .order_by(OperationJob.created_at.desc())
            .limit(1)
        )
        return self.session.scalar(statement)

