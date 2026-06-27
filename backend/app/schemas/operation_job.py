from __future__ import annotations

from datetime import datetime
from uuid import UUID

from app.schemas.base import ORMModel


class OperationJobRead(ORMModel):
    id: UUID
    job_type: str
    status: str
    user_id: UUID
    client_request_id: str | None = None
    arq_job_id: str | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    request_payload: dict
    result_payload: dict | None = None
    error_text: str | None = None
    retry_count: int
    queued_at: datetime
    running_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

