from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.base import ORMModel


class EvalCaseRead(ORMModel):
    id: UUID
    dataset_name: str
    case_name: str
    description: str | None = None
    acting_user_email: str
    question: str
    expected_document_titles: list[str] = Field(default_factory=list)
    forbidden_document_titles: list[str] = Field(default_factory=list)
    expected_answer_keywords: list[str] = Field(default_factory=list)
    notes: str | None = None
    is_demo_case: bool
    created_at: datetime
    updated_at: datetime


class EvalRunRequest(BaseModel):
    dataset_name: str = Field(default="demo_permission_eval", min_length=1, max_length=128)
    case_ids: list[UUID] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=10)
    seed_demo_cases: bool = True


class EvalResultRowRead(ORMModel):
    id: UUID
    run_id: UUID
    case_id: UUID
    acting_user_email: str
    retrieval_hit_rate: float
    citation_accuracy: float
    answer_faithfulness: float
    permission_isolation_correct: bool
    overall_pass: bool
    details_json: dict
    created_at: datetime


class EvalRunRead(ORMModel):
    id: UUID
    dataset_name: str
    status: str
    total_cases: int
    started_at: datetime | None = None
    finished_at: datetime | None = None
    summary_json: dict | None = None
    error_text: str | None = None
    created_at: datetime
    updated_at: datetime


class EvalRunDetailRead(EvalRunRead):
    results: list[EvalResultRowRead] = Field(default_factory=list)
