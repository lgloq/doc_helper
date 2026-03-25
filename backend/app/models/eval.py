from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class EvalCase(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "eval_cases"

    dataset_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    case_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    acting_user_email: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    expected_document_titles: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    forbidden_document_titles: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    expected_answer_keywords: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_demo_case: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class EvalRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "eval_runs"

    dataset_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", index=True)
    total_cases: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    summary_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)


class EvalResult(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "eval_results"

    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("eval_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("eval_cases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    acting_user_email: Mapped[str] = mapped_column(String(320), nullable=False)
    retrieval_hit_rate: Mapped[float] = mapped_column(Float, nullable=False)
    citation_accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    answer_faithfulness: Mapped[float] = mapped_column(Float, nullable=False)
    permission_isolation_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    overall_pass: Mapped[bool] = mapped_column(Boolean, nullable=False)
    details_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
