from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.schemas.base import ORMModel


class SourceSelectionRequest(BaseModel):
    session_id: UUID | None = None
    message_ids: list[UUID] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_source(self):
        if self.session_id is None and not self.message_ids:
            raise ValueError("Either session_id or message_ids must be provided.")
        return self


class SourceCitationRead(BaseModel):
    message_citation_id: UUID | None = None
    chunk_id: UUID | None = None
    document_id: UUID | None = None
    document_title: str
    document_version_id: UUID | None = None
    version_number: int | None = None
    chunk_index: int | None = None
    page_number_start: int | None = None
    page_number_end: int | None = None
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    preview: str
    fused_score: float | None = None


class TaskItemRead(ORMModel):
    id: UUID
    created_by_user_id: UUID | None = None
    source_session_id: UUID | None = None
    source_message_id: UUID | None = None
    title: str
    description: str | None = None
    owner_name: str | None = None
    priority: str
    due_date: date | None = None
    status: str
    source_citations: list[SourceCitationRead] | None = None
    created_at: datetime
    updated_at: datetime


class TaskExtractRequest(SourceSelectionRequest):
    max_items: int = Field(default=8, ge=1, le=20)


class TaskExtractResponse(BaseModel):
    items: list[TaskItemRead] = Field(default_factory=list)


class WeeklyReportGenerateRequest(SourceSelectionRequest):
    title: str | None = Field(default=None, max_length=255)


class WeeklyReportDraftRead(ORMModel):
    id: UUID
    created_by_user_id: UUID | None = None
    source_session_id: UUID | None = None
    title: str
    summary: str | None = None
    completed_this_week: list[str] = Field(default_factory=list)
    risks_blockers: list[str] = Field(default_factory=list)
    next_week_plan: list[str] = Field(default_factory=list)
    reference_sources: list[SourceCitationRead] = Field(default_factory=list)
    source_message_ids: list[str] = Field(default_factory=list)
    status: str
    created_at: datetime
    updated_at: datetime


class WeeklyReportGenerateResponse(BaseModel):
    report: WeeklyReportDraftRead


class FAQGenerateRequest(SourceSelectionRequest):
    max_entries: int = Field(default=5, ge=1, le=10)


class FAQEntryRead(ORMModel):
    id: UUID
    created_by_user_id: UUID | None = None
    source_session_id: UUID | None = None
    source_message_id: UUID | None = None
    question: str
    answer: str
    quality: str
    status: str
    source_citations: list[SourceCitationRead] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class FAQGenerateResponse(BaseModel):
    entries: list[FAQEntryRead] = Field(default_factory=list)
