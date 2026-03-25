from __future__ import annotations

from datetime import datetime
from uuid import UUID

from app.schemas.base import ORMModel


class TraceLogRead(ORMModel):
    id: UUID
    trace_type: str
    user_id: UUID | None = None
    session_id: UUID | None = None
    user_message_id: UUID | None = None
    assistant_message_id: UUID | None = None
    query_text: str | None = None
    retrieved_chunks_json: list[dict]
    selected_citations_json: list[dict]
    model_name: str | None = None
    latency_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error_text: str | None = None
    trace_metadata: dict | None = None
    created_at: datetime
