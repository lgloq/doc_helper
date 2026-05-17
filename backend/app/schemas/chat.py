from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.enums import MessageRole
from app.schemas.base import ORMModel
from app.schemas.search import SearchDebugInfo


class ChatCitationRead(ORMModel):
    id: UUID
    message_id: UUID
    chunk_id: UUID | None = None
    document_id: UUID
    document_title: str
    document_version_id: UUID
    version_number: int
    chunk_index: int
    page_number_start: int | None = None
    page_number_end: int | None = None
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    preview: str
    lexical_score: float | None = None
    vector_score: float | None = None
    fused_score: float | None = None
    rank: int
    citation_metadata: dict | None = None
    created_at: datetime


class ChatMessageRead(ORMModel):
    id: UUID
    session_id: UUID
    author_user_id: UUID | None = None
    role: MessageRole
    content: str
    model_name: str | None = None
    confidence: str | None = None
    insufficient_evidence: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int | None = None
    message_metadata: dict | None = None
    created_at: datetime
    citations: list[ChatCitationRead] = Field(default_factory=list)


class ChatSessionCreate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)


class ChatSessionRead(ORMModel):
    id: UUID
    user_id: UUID
    title: str
    display_title: str
    created_at: datetime
    updated_at: datetime


class ChatSessionDetailRead(ChatSessionRead):
    messages: list[ChatMessageRead] = Field(default_factory=list)


class ChatMessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=10)


class ChatMessageCreateResponse(BaseModel):
    session_id: UUID
    user_message: ChatMessageRead
    assistant_message: ChatMessageRead
    citations: list[ChatCitationRead] = Field(default_factory=list)
    retrieval_debug: SearchDebugInfo
