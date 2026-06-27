from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps.auth import get_current_user
from app.db.session import get_db_session
from app.models.user import User
from app.schemas.search import SearchRequest, SearchResponse
from app.services.observability.service import ObservabilityService
from app.services.retrieval.service import RetrievalService

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(
    payload: SearchRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> SearchResponse:
    service = RetrievalService(session)
    response = service.search(current_user, payload)
    if response.debug.permission_probe_early_stop_applied:
        try:
            ObservabilityService(session).record_permission_denied_retrieval(
                actor=current_user,
                query_text=payload.query,
                retrieval_response=response,
                source="search_api",
            )
        except Exception:
            pass
    return response
