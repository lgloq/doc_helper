from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps.auth import require_admin
from app.db.session import get_db_session
from app.models.user import User
from app.schemas.eval import EvalDashboardRead, EvalDatasetRead, EvalRunDetailRead, EvalRunRead, EvalRunRequest
from app.services.eval.async_service import AsyncEvalService
from app.services.eval.service import EvalService

router = APIRouter(prefix="/eval", tags=["eval"])


@router.get("/datasets", response_model=list[EvalDatasetRead])
def list_eval_datasets(
    current_user: User = Depends(require_admin),
    session: Session = Depends(get_db_session),
) -> list[EvalDatasetRead]:
    service = EvalService(session)
    return service.list_datasets(current_user)


@router.get("/dashboard", response_model=EvalDashboardRead)
def get_eval_dashboard(
    dataset_name: str | None = Query(default=None, max_length=128),
    limit: int = Query(default=8, ge=1, le=20),
    current_user: User = Depends(require_admin),
    session: Session = Depends(get_db_session),
) -> EvalDashboardRead:
    service = EvalService(session)
    return service.get_dashboard(current_user, dataset_name, limit=limit)

@router.post("/run", response_model=EvalRunDetailRead)
def run_eval(
    payload: EvalRunRequest,
    current_user: User = Depends(require_admin),
    session: Session = Depends(get_db_session),
) -> EvalRunDetailRead:
    service = EvalService(session)
    return service.run_eval(current_user, payload)


@router.post("/run/async", response_model=EvalRunDetailRead)
async def run_eval_async(
    payload: EvalRunRequest,
    current_user: User = Depends(require_admin),
    session: Session = Depends(get_db_session),
) -> EvalRunDetailRead:
    service = AsyncEvalService(session)
    return await service.enqueue_eval(current_user, payload)


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
