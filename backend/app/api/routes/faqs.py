from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps.auth import get_current_user
from app.db.session import get_db_session
from app.models.user import User
from app.schemas.workflow import FAQEntryRead, FAQGenerateRequest, FAQGenerateResponse
from app.services.faqs.service import FAQService

router = APIRouter(prefix="/faqs", tags=["faqs"])


@router.post("/generate", response_model=FAQGenerateResponse)
def generate_faqs(
    payload: FAQGenerateRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> FAQGenerateResponse:
    service = FAQService(session)
    return service.generate_faqs(current_user, payload)


@router.get("", response_model=list[FAQEntryRead])
def list_faqs(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> list[FAQEntryRead]:
    service = FAQService(session)
    return service.list_faqs(current_user)
