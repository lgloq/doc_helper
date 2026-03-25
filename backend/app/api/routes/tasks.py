from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps.auth import get_current_user
from app.db.session import get_db_session
from app.models.user import User
from app.schemas.workflow import TaskExtractRequest, TaskExtractResponse, TaskItemRead
from app.services.tasks.service import TaskService

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("/extract", response_model=TaskExtractResponse)
def extract_tasks(
    payload: TaskExtractRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> TaskExtractResponse:
    service = TaskService(session)
    return service.extract_tasks(current_user, payload)


@router.get("", response_model=list[TaskItemRead])
def list_tasks(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> list[TaskItemRead]:
    service = TaskService(session)
    return service.list_tasks(current_user)
