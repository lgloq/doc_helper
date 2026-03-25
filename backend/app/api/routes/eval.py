from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps.auth import require_admin
from app.db.session import get_db_session
from app.models.user import User
from app.schemas.eval import EvalRunDetailRead, EvalRunRead, EvalRunRequest
from app.services.eval.service import EvalService

router = APIRouter(prefix="/eval", tags=["eval"])


@router.post("/run", response_model=EvalRunDetailRead)
def run_eval(
    payload: EvalRunRequest,
    current_user: User = Depends(require_admin),
    session: Session = Depends(get_db_session),
) -> EvalRunDetailRead:
    service = EvalService(session)
    return service.run_eval(current_user, payload)


@router.get("/runs", response_model=list[EvalRunRead])
def list_eval_runs(
    current_user: User = Depends(require_admin),
    session: Session = Depends(get_db_session),
) -> list[EvalRunRead]:
    service = EvalService(session)
    return service.list_runs(current_user)


@router.get("/runs/{run_id}", response_model=EvalRunDetailRead)
def get_eval_run(
    run_id: UUID,
    current_user: User = Depends(require_admin),
    session: Session = Depends(get_db_session),
) -> EvalRunDetailRead:
    service = EvalService(session)
    return service.get_run(current_user, run_id)
