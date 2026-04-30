from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session

from app.api.deps.auth import get_current_user
from app.db.session import get_db_session
from app.models.user import User
from app.schemas.chat import (
    ChatMessageCreate,
    ChatMessageCreateResponse,
    ChatSessionCreate,
    ChatSessionDetailRead,
    ChatSessionRead,
)
from app.services.chat.service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/sessions", response_model=ChatSessionRead)
def create_chat_session(
    payload: ChatSessionCreate | None = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> ChatSessionRead:
    service = ChatService(session)
    return service.create_session(current_user, payload)


@router.get("/sessions", response_model=list[ChatSessionRead])
def list_chat_sessions(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> list[ChatSessionRead]:
    service = ChatService(session)
    return service.list_sessions(current_user)


@router.get("/sessions/{session_id}", response_model=ChatSessionDetailRead)
def get_chat_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> ChatSessionDetailRead:
    service = ChatService(session)
    return service.get_session(current_user, session_id)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chat_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> Response:
    service = ChatService(session)
    service.delete_session(current_user, session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/sessions/{session_id}/messages", response_model=ChatMessageCreateResponse)
def create_chat_message(
    session_id: UUID,
    payload: ChatMessageCreate,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> ChatMessageCreateResponse:
    service = ChatService(session)
    return service.create_message(current_user, session_id, payload)
