from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps.auth import get_current_user
from app.db.session import get_db_session
from app.models.user import User
from app.schemas.operation_job import OperationJobRead
from app.services.jobs.service import OperationJobService

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=OperationJobRead)
def get_operation_job(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> OperationJobRead:
    service = OperationJobService(session)
    return service.get_job(current_user, job_id)
