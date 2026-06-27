from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, JSON, String, Text, UniqueConstraint, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class OperationJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "operation_jobs"
    __table_args__ = (
        UniqueConstraint(
            "job_type",
            "user_id",
            "client_request_id",
            name="uq_operation_jobs_job_type_user_id_client_request_id",
        ),
        Index("ix_operation_jobs_job_type_status_created_at", "job_type", "status", "created_at"),
        Index("ix_operation_jobs_user_id_status_created_at", "user_id", "status", "created_at"),
        Index("ix_operation_jobs_client_request_id", "client_request_id"),
    )

    job_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued", index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    client_request_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    arq_job_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    resource_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    resource_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    request_payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    result_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(default=0, nullable=False)
    queued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    running_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
