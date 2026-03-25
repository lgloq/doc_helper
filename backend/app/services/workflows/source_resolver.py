from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.chat import ChatMessage, ChatSession, MessageCitation
from app.models.user import User
from app.repositories.chat_repository import ChatRepository
from app.schemas.workflow import SourceSelectionRequest


@dataclass
class SourceMaterialBundle:
    session: ChatSession | None
    messages: list[ChatMessage]


class SourceMaterialResolver:
    def __init__(self, session: Session):
        self.session = session
        self.chat_repository = ChatRepository(session)

    def resolve(self, actor: User, payload: SourceSelectionRequest) -> SourceMaterialBundle:
        if payload.session_id is not None:
            chat_session = self.chat_repository.get_session_for_user(payload.session_id, actor.id, include_messages=True)
            if chat_session is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.")
            return SourceMaterialBundle(session=chat_session, messages=list(chat_session.messages))

        messages = self.chat_repository.list_messages_for_user(actor.id, payload.message_ids)
        if len(messages) != len(payload.message_ids):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="One or more source messages were not found.")
        session_id = messages[0].session_id if messages else None
        same_session = session_id is not None and all(message.session_id == session_id for message in messages)
        source_session = messages[0].session if messages and same_session else None
        return SourceMaterialBundle(session=source_session, messages=messages)



def serialize_message_citation(citation: MessageCitation) -> dict:
    return {
        "message_citation_id": str(citation.id),
        "chunk_id": str(citation.chunk_id) if citation.chunk_id else None,
        "document_id": str(citation.document_id) if citation.document_id else None,
        "document_title": citation.document_title,
        "document_version_id": str(citation.document_version_id) if citation.document_version_id else None,
        "version_number": citation.version_number,
        "chunk_index": citation.chunk_index,
        "page_number_start": citation.page_number_start,
        "page_number_end": citation.page_number_end,
        "paragraph_start": citation.paragraph_start,
        "paragraph_end": citation.paragraph_end,
        "preview": citation.preview,
        "fused_score": citation.fused_score,
    }



def unique_serialized_citations(messages: list[ChatMessage], limit: int | None = None) -> list[dict]:
    seen: set[str] = set()
    items: list[dict] = []
    for message in messages:
        for citation in getattr(message, "citations", []):
            key = str(citation.id)
            if key in seen:
                continue
            seen.add(key)
            items.append(serialize_message_citation(citation))
            if limit is not None and len(items) >= limit:
                return items
    return items
