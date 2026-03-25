from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models.chat import ChatMessage, ChatSession, MessageCitation


class ChatRepository:
    def __init__(self, session: Session):
        self.session = session

    def add_session(self, chat_session: ChatSession) -> ChatSession:
        self.session.add(chat_session)
        return chat_session

    def list_sessions_for_user(self, user_id: UUID) -> list[ChatSession]:
        statement = (
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc(), ChatSession.created_at.desc())
        )
        return list(self.session.scalars(statement).all())

    def get_session_for_user(self, session_id: UUID, user_id: UUID, include_messages: bool = False) -> ChatSession | None:
        statement = select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user_id)
        if include_messages:
            statement = statement.options(
                selectinload(ChatSession.messages).selectinload(ChatMessage.citations),
            )
        return self.session.scalar(statement)

    def list_messages_for_user(self, user_id: UUID, message_ids: Sequence[UUID]) -> list[ChatMessage]:
        if not message_ids:
            return []
        statement = (
            select(ChatMessage)
            .join(ChatSession, ChatSession.id == ChatMessage.session_id)
            .options(selectinload(ChatMessage.citations), selectinload(ChatMessage.session))
            .where(ChatMessage.id.in_(message_ids), ChatSession.user_id == user_id)
            .order_by(ChatMessage.created_at.asc())
        )
        return list(self.session.scalars(statement).all())

    def add_message(self, message: ChatMessage) -> ChatMessage:
        self.session.add(message)
        return message

    def add_citations(self, citations: list[MessageCitation]) -> None:
        for citation in citations:
            self.session.add(citation)

    def get_message(self, message_id: UUID) -> ChatMessage | None:
        statement = (
            select(ChatMessage)
            .options(selectinload(ChatMessage.citations))
            .where(ChatMessage.id == message_id)
        )
        return self.session.scalar(statement)
