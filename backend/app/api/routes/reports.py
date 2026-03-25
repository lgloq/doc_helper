from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps.auth import get_current_user
from app.db.session import get_db_session
from app.models.user import User
from app.schemas.workflow import WeeklyReportDraftRead, WeeklyReportGenerateRequest, WeeklyReportGenerateResponse
from app.services.reports.service import WeeklyReportService

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post("/weekly", response_model=WeeklyReportGenerateResponse)
def generate_weekly_report(
    payload: WeeklyReportGenerateRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> WeeklyReportGenerateResponse:
    service = WeeklyReportService(session)
    return service.generate_report(current_user, payload)


@router.get("", response_model=list[WeeklyReportDraftRead])
def list_reports(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> list[WeeklyReportDraftRead]:
    service = WeeklyReportService(session)
    return service.list_reports(current_user)
