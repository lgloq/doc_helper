from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps.auth import get_current_user
from app.db.session import get_db_session
from app.models.user import User
from app.schemas.search import SearchRequest, SearchResponse
from app.services.retrieval.service import RetrievalService

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(
    payload: SearchRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> SearchResponse:
    service = RetrievalService(session)
    return service.search(current_user, payload)
