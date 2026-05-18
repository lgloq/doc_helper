from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.models.chat import ChatMessage, ChatSession, MessageCitation
from app.models.enums import MessageRole


class ChatRepository:
    def __init__(self, session: Session):
        self.session = session

    def add_session(self, chat_session: ChatSession) -> ChatSession:
        self.session.add(chat_session)
        return chat_session

    def delete_session(self, chat_session: ChatSession) -> None:
        self.session.delete(chat_session)

    def list_sessions_for_user(self, user_id: UUID) -> list[ChatSession]:
        statement = (
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc(), ChatSession.created_at.desc())
        )
        return list(self.session.scalars(statement).all())

    def list_first_user_message_previews(self, session_ids: Sequence[UUID]) -> dict[UUID, str]:
        if not session_ids:
            return {}
        ranked_messages = (
            select(
                ChatMessage.session_id.label("session_id"),
                ChatMessage.content.label("content"),
                func.row_number()
                .over(
                    partition_by=ChatMessage.session_id,
                    order_by=(ChatMessage.created_at.asc(), ChatMessage.id.asc()),
                )
                .label("row_number"),
            )
            .where(
                ChatMessage.session_id.in_(session_ids),
                ChatMessage.role == MessageRole.USER,
            )
            .subquery()
        )
        statement = select(ranked_messages.c.session_id, ranked_messages.c.content).where(
            ranked_messages.c.row_number == 1
        )
        rows = self.session.execute(statement).all()
        return {row.session_id: row.content for row in rows}

    def list_first_message_previews_by_role(self, session_ids: Sequence[UUID]) -> dict[UUID, dict[str, str]]:
        if not session_ids:
            return {}
        statement = (
            select(ChatMessage.session_id, ChatMessage.role, ChatMessage.content)
            .where(ChatMessage.session_id.in_(session_ids))
            .order_by(ChatMessage.session_id.asc(), ChatMessage.created_at.asc(), ChatMessage.id.asc())
        )
        rows = self.session.execute(statement).all()
        previews: dict[UUID, dict[str, str]] = {}
        for row in rows:
            session_entry = previews.setdefault(row.session_id, {})
            role_key = str(row.role.value if hasattr(row.role, "value") else row.role)
            session_entry.setdefault(role_key, row.content)
        return previews

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
