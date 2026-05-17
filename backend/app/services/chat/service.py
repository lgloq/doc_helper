from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.chat import ChatMessage, ChatSession, MessageCitation
from app.models.enums import MessageRole
from app.models.user import User
from app.repositories.chat_repository import ChatRepository
from app.schemas.chat import (
    ChatCitationRead,
    ChatMessageCreate,
    ChatMessageCreateResponse,
    ChatMessageRead,
    ChatSessionCreate,
    ChatSessionDetailRead,
    ChatSessionRead,
)
from app.services.llm.orchestrator import CopilotOrchestrator, CopilotRunResult
from app.services.observability.service import ObservabilityService

DEFAULT_CHAT_SESSION_TITLE = "新会话"
GENERIC_CHAT_SESSION_TITLES = {DEFAULT_CHAT_SESSION_TITLE, "New Chat"}


@dataclass
class PreparedChatAnswer:
    router_result: object
    tool_metadata: object
    answer_result: object
    retrieval_response: object
    candidate_chunks: list
    selected_chunks: list
    confidence: str
    structured_result: object
    agent_steps: list
    agent_run_trace: object | None = None


class ChatService:
    def __init__(self, session: Session):
        self.session = session
        self.chat_repository = ChatRepository(session)
        self.copilot_orchestrator = CopilotOrchestrator(session)
        self.observability_service = ObservabilityService(session)

    def create_session(self, actor: User, payload: ChatSessionCreate | None) -> ChatSessionRead:
        title = (payload.title.strip() if payload and payload.title else DEFAULT_CHAT_SESSION_TITLE) or DEFAULT_CHAT_SESSION_TITLE
        chat_session = ChatSession(user_id=actor.id, title=title)
        self.chat_repository.add_session(chat_session)
        self.session.commit()
        self.session.refresh(chat_session)
        return self._serialize_session(chat_session)

    def list_sessions(self, actor: User) -> list[ChatSessionRead]:
        sessions = self.chat_repository.list_sessions_for_user(actor.id)
        preview_by_session_id = self.chat_repository.list_first_user_message_previews([item.id for item in sessions])
        return [
            self._serialize_session(item, first_user_message=preview_by_session_id.get(item.id))
            for item in sessions
        ]

    def get_session(self, actor: User, session_id: UUID) -> ChatSessionDetailRead:
        chat_session = self._get_session_or_404(actor, session_id, include_messages=True)
        return ChatSessionDetailRead.model_validate(
            {
                **self._serialize_session(chat_session).model_dump(),
                "messages": [self._serialize_message(message) for message in chat_session.messages],
            }
        )

    def delete_session(self, actor: User, session_id: UUID) -> None:
        chat_session = self._get_session_or_404(actor, session_id, include_messages=False)
        self.chat_repository.delete_session(chat_session)
        self.session.commit()

    def preview_answer(self, actor: User, question: str, top_k: int = 5) -> PreparedChatAnswer:
        return self._prepare_answer(actor, question, top_k, existing_messages=[], session_id=None)

    def create_message(self, actor: User, session_id: UUID, payload: ChatMessageCreate) -> ChatMessageCreateResponse:
        chat_session = self._get_session_or_404(actor, session_id, include_messages=True)
        existing_messages = list(chat_session.messages)

        user_message = ChatMessage(
            session_id=chat_session.id,
            author_user_id=actor.id,
            role=MessageRole.USER,
            content=payload.content,
            message_metadata={"top_k": payload.top_k},
        )
        if not existing_messages and self._is_generic_session_title(chat_session.title):
            chat_session.title = self._truncate_session_title(payload.content)
        chat_session.updated_at = datetime.now(UTC)
        self.chat_repository.add_message(user_message)
        self.session.flush()
        user_message_id = user_message.id
        chat_session_id = chat_session.id

        prepared = self._prepare_answer(actor, payload.content, payload.top_k, existing_messages, session_id=chat_session.id)
        assistant_result = prepared.answer_result
        assistant_message = ChatMessage(
            session_id=chat_session.id,
            author_user_id=None,
            role=MessageRole.ASSISTANT,
            content=assistant_result.answer,
            model_name=assistant_result.model_name,
            confidence=prepared.confidence,
            insufficient_evidence=assistant_result.insufficient_evidence,
            prompt_tokens=assistant_result.prompt_tokens,
            completion_tokens=assistant_result.completion_tokens,
            latency_ms=assistant_result.latency_ms,
            message_metadata={
                "answer_provider": assistant_result.provider_name,
                "answer_basis": assistant_result.answer_basis,
                "evidence_conflict": assistant_result.evidence_conflict,
                "retrieval_debug": prepared.retrieval_response.debug.model_dump(),
                "used_chunk_ids": [str(item.chunk_id) for item in prepared.selected_chunks],
                "candidate_chunk_ids": [str(item.chunk_id) for item in prepared.candidate_chunks],
                "retrieved_chunk_ids": [str(item.chunk_id) for item in prepared.retrieval_response.matched_chunks],
                "router_decision": prepared.router_result.decision.model_dump(mode="json"),
                "tool_execution": prepared.tool_metadata.model_dump(mode="json"),
                "structured_result": prepared.structured_result.model_dump(mode="json"),
                "agent_steps": [item.model_dump(mode="json") for item in prepared.agent_steps],
                "agent_run_trace": prepared.agent_run_trace.model_dump(mode="json") if prepared.agent_run_trace else None,
                "raw_payload": assistant_result.raw_payload,
            },
        )
        chat_session.updated_at = datetime.now(UTC)
        self.chat_repository.add_message(assistant_message)
        self.session.flush()
        assistant_message_id = assistant_message.id

        citation_rows = self._build_citation_rows(assistant_message.id, prepared)
        self.chat_repository.add_citations(citation_rows)
        self.session.commit()

        hydrated_user_message = self.chat_repository.get_message(user_message_id)
        hydrated_assistant_message = self.chat_repository.get_message(assistant_message_id)
        if hydrated_user_message is None or hydrated_assistant_message is None:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to load chat messages.")

        try:
            self.observability_service.record_trace(
                actor=actor,
                chat_session=chat_session,
                user_message=hydrated_user_message,
                assistant_message=hydrated_assistant_message,
                query_text=payload.content,
                retrieval_response=prepared.retrieval_response,
                selected_chunks=prepared.selected_chunks,
                error_text=self._extract_trace_error(prepared),
                trace_type="chat_qa",
                confidence=prepared.confidence,
                insufficient_evidence=prepared.answer_result.insufficient_evidence,
                extra_metadata={
                    "router_decision": prepared.router_result.decision.model_dump(mode="json"),
                    "tool_execution": prepared.tool_metadata.model_dump(mode="json"),
                    "structured_result": prepared.structured_result.model_dump(mode="json"),
                    "agent_steps": [item.model_dump(mode="json") for item in prepared.agent_steps],
                    "agent_run_trace": prepared.agent_run_trace.model_dump(mode="json") if prepared.agent_run_trace else None,
                },
            )
        except Exception:
            pass

        assistant_citations = [self._serialize_citation(item) for item in hydrated_assistant_message.citations]
        return ChatMessageCreateResponse(
            session_id=chat_session_id,
            user_message=self._serialize_message(hydrated_user_message),
            assistant_message=self._serialize_message(hydrated_assistant_message),
            citations=assistant_citations,
            retrieval_debug=prepared.retrieval_response.debug,
        )

    def _prepare_answer(
        self,
        actor: User,
        question: str,
        top_k: int,
        existing_messages: list[ChatMessage],
        session_id: UUID | None,
    ) -> PreparedChatAnswer:
        effective_session_id = session_id or UUID(int=0)
        orchestrated: CopilotRunResult = self.copilot_orchestrator.run(
            actor=actor,
            question=question,
            session_id=effective_session_id,
            top_k=top_k,
            existing_messages=existing_messages,
        )
        return PreparedChatAnswer(
            router_result=orchestrated.router_result,
            tool_metadata=orchestrated.tool_metadata,
            answer_result=orchestrated.answer_result,
            retrieval_response=orchestrated.retrieval_response,
            candidate_chunks=orchestrated.candidate_chunks,
            selected_chunks=orchestrated.selected_chunks,
            confidence=orchestrated.confidence,
            structured_result=orchestrated.structured_result,
            agent_steps=orchestrated.agent_steps,
            agent_run_trace=orchestrated.agent_run_trace,
        )

    def _get_session_or_404(self, actor: User, session_id: UUID, include_messages: bool = False) -> ChatSession:
        chat_session = self.chat_repository.get_session_for_user(session_id, actor.id, include_messages=include_messages)
        if chat_session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.")
        return chat_session

    @staticmethod
    def _extract_trace_error(prepared: PreparedChatAnswer) -> str | None:
        raw_payload = prepared.answer_result.raw_payload or {}
        error_text = raw_payload.get("error_text")
        if error_text:
            return str(error_text)
        return None

    @staticmethod
    def _truncate_session_title(question: str, limit: int = 80) -> str:
        compact = " ".join(question.strip().split())
        if len(compact) <= limit:
            return compact or DEFAULT_CHAT_SESSION_TITLE
        return compact[: limit - 3].rstrip() + "..."

    @classmethod
    def _is_generic_session_title(cls, title: str | None) -> bool:
        return (title or "").strip() in GENERIC_CHAT_SESSION_TITLES

    @classmethod
    def _resolve_display_title(
        cls,
        chat_session: ChatSession,
        *,
        first_user_message: str | None = None,
    ) -> str:
        if cls._is_generic_session_title(chat_session.title) and first_user_message:
            return cls._truncate_session_title(first_user_message)
        return chat_session.title

    @classmethod
    def _serialize_session(
        cls,
        chat_session: ChatSession,
        *,
        first_user_message: str | None = None,
    ) -> ChatSessionRead:
        if first_user_message is None and "messages" in chat_session.__dict__:
            first_user_message = next(
                (message.content for message in chat_session.messages if message.role == MessageRole.USER),
                None,
            )
        payload = {
            "id": chat_session.id,
            "user_id": chat_session.user_id,
            "title": chat_session.title,
            "display_title": cls._resolve_display_title(chat_session, first_user_message=first_user_message),
            "created_at": chat_session.created_at,
            "updated_at": chat_session.updated_at,
        }
        return ChatSessionRead.model_validate(payload)

    def _serialize_message(self, message: ChatMessage) -> ChatMessageRead:
        return ChatMessageRead.model_validate(
            {
                "id": message.id,
                "session_id": message.session_id,
                "author_user_id": message.author_user_id,
                "role": message.role,
                "content": message.content,
                "model_name": message.model_name,
                "confidence": message.confidence,
                "insufficient_evidence": message.insufficient_evidence,
                "prompt_tokens": message.prompt_tokens,
                "completion_tokens": message.completion_tokens,
                "latency_ms": message.latency_ms,
                "message_metadata": message.message_metadata,
                "created_at": message.created_at,
                "citations": [self._serialize_citation(item) for item in getattr(message, "citations", [])],
            }
        )

    @staticmethod
    def _serialize_citation(citation: MessageCitation) -> ChatCitationRead:
        return ChatCitationRead.model_validate(
            {
                "id": citation.id,
                "message_id": citation.message_id,
                "chunk_id": citation.chunk_id,
                "document_id": citation.document_id,
                "document_title": citation.document_title,
                "document_version_id": citation.document_version_id,
                "version_number": citation.version_number,
                "chunk_index": citation.chunk_index,
                "page_number_start": citation.page_number_start,
                "page_number_end": citation.page_number_end,
                "paragraph_start": citation.paragraph_start,
                "paragraph_end": citation.paragraph_end,
                "preview": citation.preview,
                "lexical_score": citation.lexical_score,
                "vector_score": citation.vector_score,
                "fused_score": citation.fused_score,
                "rank": citation.rank,
                "citation_metadata": citation.citation_metadata,
                "created_at": citation.created_at,
            }
        )

    def _build_citation_rows(self, message_id: UUID, prepared: PreparedChatAnswer) -> list[MessageCitation]:
        if prepared.selected_chunks:
            return [
                MessageCitation(
                    message_id=message_id,
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    document_version_id=item.document_version_id,
                    document_title=item.document_title,
                    version_number=item.version_number,
                    chunk_index=item.chunk_index,
                    page_number_start=item.page_number_start,
                    page_number_end=item.page_number_end,
                    paragraph_start=item.paragraph_start,
                    paragraph_end=item.paragraph_end,
                    preview=item.preview,
                    lexical_score=item.score.lexical_raw,
                    vector_score=item.score.vector_raw,
                    fused_score=item.score.fused,
                    rank=index,
                    citation_metadata=item.citation_metadata,
                )
                for index, item in enumerate(prepared.selected_chunks, start=1)
            ]

        citations = getattr(prepared.structured_result, "citations", None)
        if not isinstance(citations, list) or not citations:
            return []

        rows: list[MessageCitation] = []
        seen: set[str] = set()
        for rank, citation in enumerate(citations, start=1):
            payload = citation.model_dump(mode="python") if hasattr(citation, "model_dump") else dict(citation)
            identity = self._citation_identity(payload)
            if identity in seen:
                continue
            seen.add(identity)
            rows.append(
                MessageCitation(
                    message_id=message_id,
                    chunk_id=payload.get("chunk_id"),
                    document_id=payload.get("document_id"),
                    document_version_id=payload.get("document_version_id"),
                    document_title=payload.get("document_title"),
                    version_number=payload.get("version_number"),
                    chunk_index=payload.get("chunk_index"),
                    page_number_start=payload.get("page_number_start"),
                    page_number_end=payload.get("page_number_end"),
                    paragraph_start=payload.get("paragraph_start"),
                    paragraph_end=payload.get("paragraph_end"),
                    preview=payload.get("preview"),
                    lexical_score=None,
                    vector_score=None,
                    fused_score=payload.get("fused_score"),
                    rank=len(rows) + 1,
                    citation_metadata=None,
                )
            )
        return rows

    @staticmethod
    def _citation_identity(payload: dict) -> str:
        chunk_id = payload.get("chunk_id")
        if chunk_id:
            return f"chunk:{chunk_id}"
        return "|".join(
            [
                str(payload.get("document_id") or ""),
                str(payload.get("document_version_id") or ""),
                str(payload.get("chunk_index") or ""),
                str(payload.get("page_number_start") or ""),
                str(payload.get("paragraph_start") or ""),
                str(payload.get("preview") or ""),
            ]
        )
