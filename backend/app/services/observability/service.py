from __future__ import annotations

import json
import logging
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.chat import ChatMessage, ChatSession
from app.models.observability import TraceLog
from app.models.user import User
from app.repositories.trace_repository import TraceRepository
from app.schemas.observability import TraceLogRead
from app.schemas.search import SearchResponse, SearchResultChunk
from app.services.observability.langfuse_adapter import LangfuseAdapter

logger = logging.getLogger("app.observability")


class ObservabilityService:
    def __init__(self, session: Session):
        self.session = session
        self.trace_repository = TraceRepository(session)
        self.langfuse_adapter = LangfuseAdapter()

    def record_trace(
        self,
        *,
        actor: User | None,
        chat_session: ChatSession | None,
        user_message: ChatMessage | None,
        assistant_message: ChatMessage | None,
        query_text: str,
        retrieval_response: SearchResponse,
        selected_chunks: list[SearchResultChunk],
        error_text: str | None = None,
        trace_type: str = "chat_qa",
        confidence: str | None = None,
        insufficient_evidence: bool | None = None,
        extra_metadata: dict | None = None,
        model_name: str | None = None,
        latency_ms: int | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> TraceLog:
        resolved_model_name = assistant_message.model_name if assistant_message else model_name
        resolved_latency_ms = assistant_message.latency_ms if assistant_message else latency_ms
        resolved_prompt_tokens = assistant_message.prompt_tokens if assistant_message else prompt_tokens
        resolved_completion_tokens = assistant_message.completion_tokens if assistant_message else completion_tokens

        payload = {
            "trace_type": trace_type,
            "user_id": str(actor.id) if actor else None,
            "session_id": str(chat_session.id) if chat_session else None,
            "query_text": query_text,
            "retrieved_chunks_json": [self._serialize_chunk(chunk) for chunk in retrieval_response.matched_chunks],
            "selected_citations_json": [self._serialize_chunk(chunk) for chunk in selected_chunks],
            "model_name": resolved_model_name,
            "latency_ms": resolved_latency_ms,
            "prompt_tokens": resolved_prompt_tokens,
            "completion_tokens": resolved_completion_tokens,
            "error_text": error_text,
            "trace_metadata": {
                "confidence": confidence,
                "insufficient_evidence": insufficient_evidence,
                "retrieval_debug": retrieval_response.debug.model_dump(),
                **(extra_metadata or {}),
            },
        }
        trace = TraceLog(
            trace_type=trace_type,
            user_id=actor.id if actor else None,
            session_id=chat_session.id if chat_session else None,
            user_message_id=user_message.id if user_message else None,
            assistant_message_id=assistant_message.id if assistant_message else None,
            query_text=query_text,
            retrieved_chunks_json=payload["retrieved_chunks_json"],
            selected_citations_json=payload["selected_citations_json"],
            model_name=payload["model_name"],
            latency_ms=payload["latency_ms"],
            prompt_tokens=payload["prompt_tokens"],
            completion_tokens=payload["completion_tokens"],
            error_text=error_text,
            trace_metadata=payload["trace_metadata"],
        )
        self.trace_repository.add(trace)
        self.session.commit()
        self.session.refresh(trace)
        logger.info("trace_recorded=%s", json.dumps({**payload, "trace_id": str(trace.id)}, ensure_ascii=False))
        self.langfuse_adapter.emit_trace({**payload, "trace_id": str(trace.id)})
        return trace

    def list_traces(
        self,
        *,
        user_id: UUID | None = None,
        session_id: UUID | None = None,
        trace_type: str | None = None,
        limit: int = 50,
    ) -> list[TraceLogRead]:
        traces = self.trace_repository.list_traces(
            user_id=user_id,
            session_id=session_id,
            trace_type=trace_type,
            limit=limit,
        )
        return [TraceLogRead.model_validate(trace) for trace in traces]

    def record_permission_denied_retrieval(
        self,
        *,
        actor: User,
        query_text: str,
        retrieval_response: SearchResponse,
        source: str,
    ) -> TraceLog:
        debug = retrieval_response.debug
        return self.record_trace(
            actor=actor,
            chat_session=None,
            user_message=None,
            assistant_message=None,
            query_text=query_text,
            retrieval_response=retrieval_response,
            selected_chunks=[],
            trace_type="permission_denied_retrieval",
            insufficient_evidence=True,
            extra_metadata={
                "source": source,
                "permission_refusal_reason_code": debug.permission_refusal_reason_code,
                "permission_refusal_reason": debug.permission_refusal_reason,
                "permission_probe_target_hint": debug.permission_probe_target_hint,
                "permission_probe_accessible_target_count": debug.permission_probe_accessible_target_count,
                "permission_probe_inaccessible_target_count": debug.permission_probe_inaccessible_target_count,
            },
        )

    def get_trace(self, trace_id: UUID) -> TraceLogRead | None:
        trace = self.trace_repository.get_by_id(trace_id)
        if trace is None:
            return None
        return TraceLogRead.model_validate(trace)

    @staticmethod
    def _serialize_chunk(chunk: SearchResultChunk) -> dict:
        return {
            "chunk_id": str(chunk.chunk_id),
            "document_id": str(chunk.document_id),
            "document_title": chunk.document_title,
            "document_version_id": str(chunk.document_version_id),
            "version_number": chunk.version_number,
            "chunk_index": chunk.chunk_index,
            "page_number_start": chunk.page_number_start,
            "page_number_end": chunk.page_number_end,
            "paragraph_start": chunk.paragraph_start,
            "paragraph_end": chunk.paragraph_end,
            "preview": chunk.preview,
            "score": chunk.score.model_dump(),
        }
