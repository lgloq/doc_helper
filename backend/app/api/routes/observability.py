from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps.auth import get_current_user
from app.db.session import get_db_session
from app.models.enums import RoleName
from app.models.user import User
from app.schemas.observability import TraceLogRead
from app.services.observability.service import ObservabilityService

router = APIRouter(prefix="/observability", tags=["observability"])


@router.get("/traces", response_model=list[TraceLogRead])
def list_traces(
    user_id: UUID | None = Query(default=None),
    session_id: UUID | None = Query(default=None),
    trace_type: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> list[TraceLogRead]:
    is_admin = current_user.role is not None and current_user.role.name == RoleName.ADMIN
    if user_id is not None and not is_admin and user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You can only view your own traces.")

    service = ObservabilityService(session)
    effective_user_id = user_id if is_admin else current_user.id
    return service.list_traces(user_id=effective_user_id, session_id=session_id, trace_type=trace_type, limit=limit)


@router.get("/traces/{trace_id}", response_model=TraceLogRead)
def get_trace(
    trace_id: UUID,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> TraceLogRead:
    service = ObservabilityService(session)
    trace = service.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trace not found.")

    is_admin = current_user.role is not None and current_user.role.name == RoleName.ADMIN
    if not is_admin and trace.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trace not found.")
    return trace
