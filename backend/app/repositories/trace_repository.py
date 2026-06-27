from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.observability import TraceLog


class TraceRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, trace: TraceLog) -> TraceLog:
        self.session.add(trace)
        return trace

    def list_traces(
        self,
        *,
        user_id: UUID | None = None,
        session_id: UUID | None = None,
        trace_type: str | None = None,
        limit: int = 50,
    ) -> list[TraceLog]:
        statement = select(TraceLog)
        if user_id is not None:
            statement = statement.where(TraceLog.user_id == user_id)
        if session_id is not None:
            statement = statement.where(TraceLog.session_id == session_id)
        if trace_type is not None:
            statement = statement.where(TraceLog.trace_type == trace_type)
        statement = statement.order_by(TraceLog.created_at.desc()).limit(limit)
        return list(self.session.scalars(statement).all())

    def get_by_id(self, trace_id: UUID) -> TraceLog | None:
        statement = select(TraceLog).where(TraceLog.id == trace_id)
        return self.session.scalar(statement)
