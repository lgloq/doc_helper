from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
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
from app.schemas.search import SearchRequest, SearchResponse, SearchResultChunk
from app.services.chat.generation import AnswerGenerationResult, AnswerGeneratorFactory
from app.services.chat.prompts import format_history_line, truncate_session_title, validate_used_chunk_ids
from app.services.observability.service import ObservabilityService
from app.services.retrieval.service import RetrievalService

DEFAULT_CHAT_SESSION_TITLE = "New Chat"


@dataclass
class PreparedChatAnswer:
    retrieval_response: SearchResponse
    answer_result: AnswerGenerationResult
    selected_chunks: list[SearchResultChunk]
    confidence: str


class ChatService:
    def __init__(self, session: Session):
        self.session = session
        self.settings = get_settings()
        self.chat_repository = ChatRepository(session)
        self.retrieval_service = RetrievalService(session)
        self.answer_generator = AnswerGeneratorFactory.create()
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
        return [self._serialize_session(item) for item in sessions]

    def get_session(self, actor: User, session_id: UUID) -> ChatSessionDetailRead:
        chat_session = self._get_session_or_404(actor, session_id, include_messages=True)
        return ChatSessionDetailRead.model_validate(
            {
                **self._serialize_session(chat_session).model_dump(),
                "messages": [self._serialize_message(message) for message in chat_session.messages],
            }
        )

    def preview_answer(self, actor: User, question: str, top_k: int = 5) -> PreparedChatAnswer:
        return self._prepare_answer(actor, question, top_k, existing_messages=[])

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
        if not existing_messages and chat_session.title == DEFAULT_CHAT_SESSION_TITLE:
            chat_session.title = truncate_session_title(payload.content)
        chat_session.updated_at = datetime.now(UTC)
        self.chat_repository.add_message(user_message)
        self.session.flush()

        prepared = self._prepare_answer(actor, payload.content, payload.top_k, existing_messages)
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
                "retrieved_chunk_ids": [str(item.chunk_id) for item in prepared.retrieval_response.matched_chunks],
                "raw_payload": assistant_result.raw_payload,
            },
        )
        chat_session.updated_at = datetime.now(UTC)
        self.chat_repository.add_message(assistant_message)
        self.session.flush()

        citation_rows = [
            MessageCitation(
                message_id=assistant_message.id,
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
        self.chat_repository.add_citations(citation_rows)
        self.session.commit()

        hydrated_user_message = self.chat_repository.get_message(user_message.id)
        hydrated_assistant_message = self.chat_repository.get_message(assistant_message.id)
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
                error_text=self._extract_trace_error(prepared.answer_result),
                trace_type="chat_qa",
                confidence=prepared.confidence,
                insufficient_evidence=prepared.answer_result.insufficient_evidence,
                extra_metadata={"answer_basis": prepared.answer_result.answer_basis},
            )
        except Exception:
            pass

        assistant_citations = [self._serialize_citation(item) for item in hydrated_assistant_message.citations]
        return ChatMessageCreateResponse(
            session_id=chat_session.id,
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
    ) -> PreparedChatAnswer:
        retrieval_response = self.retrieval_service.search(actor, SearchRequest(query=question, top_k=top_k))
        answer_result = self._generate_safe_answer(question, retrieval_response.matched_chunks, existing_messages)
        validated_chunk_ids = validate_used_chunk_ids(
            answer_result.used_chunk_ids,
            {str(item.chunk_id) for item in retrieval_response.matched_chunks},
        )
        selected_chunks = self._select_citation_chunks(retrieval_response.matched_chunks, validated_chunk_ids)
        if not answer_result.insufficient_evidence and not selected_chunks:
            answer_result = self._force_insufficient_result(
                answer_basis="invalid_or_missing_citations",
                original_answer=answer_result.answer,
                provider_name=answer_result.provider_name,
                model_name=answer_result.model_name,
                prompt_tokens=answer_result.prompt_tokens,
                completion_tokens=answer_result.completion_tokens,
                latency_ms=answer_result.latency_ms,
            )
        confidence = self._compute_confidence(selected_chunks, answer_result)
        return PreparedChatAnswer(
            retrieval_response=retrieval_response,
            answer_result=answer_result,
            selected_chunks=selected_chunks,
            confidence=confidence,
        )

    def _get_session_or_404(self, actor: User, session_id: UUID, include_messages: bool = False) -> ChatSession:
        chat_session = self.chat_repository.get_session_for_user(session_id, actor.id, include_messages=include_messages)
        if chat_session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.")
        return chat_session

    def _generate_safe_answer(
        self,
        question: str,
        retrieved_chunks: list[SearchResultChunk],
        existing_messages: list[ChatMessage],
    ) -> AnswerGenerationResult:
        history_limit = max(self.settings.chat_history_window, 0) * 2
        history_window = existing_messages[-history_limit:] if history_limit else []
        history_lines = [format_history_line(message.role.value, message.content) for message in history_window]
        if not retrieved_chunks:
            return self._force_insufficient_result(answer_basis="no_retrieval_hits")
        try:
            return self.answer_generator.generate(
                question=question,
                retrieved_chunks=retrieved_chunks,
                history_lines=history_lines,
            )
        except Exception as exc:
            return self._force_insufficient_result(
                answer_basis="model_failure",
                error_text=str(exc),
            )

    @staticmethod
    def _force_insufficient_result(
        *,
        answer_basis: str,
        original_answer: str | None = None,
        provider_name: str = "system",
        model_name: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        latency_ms: int | None = None,
        error_text: str | None = None,
    ) -> AnswerGenerationResult:
        if answer_basis == "model_failure":
            answer = "I could not complete a grounded answer because the answer generator failed. Please retry."
        elif answer_basis == "invalid_or_missing_citations":
            answer = (
                "I found related material, but I could not validate the citations for a grounded answer. "
                "Please inspect the retrieved evidence directly."
            )
        else:
            answer = (
                "I could not find enough evidence in the documents you can access to support a reliable answer."
            )
        metadata_payload = {"error_text": error_text, "original_answer": original_answer}
        return AnswerGenerationResult(
            answer=answer,
            insufficient_evidence=True,
            evidence_conflict=False,
            used_chunk_ids=[],
            answer_basis=answer_basis,
            provider_name=provider_name,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens or max(len(answer) // 4, 1),
            latency_ms=latency_ms,
            raw_payload=metadata_payload,
        )

    @staticmethod
    def _select_citation_chunks(
        matched_chunks: list[SearchResultChunk],
        validated_chunk_ids: list[UUID],
    ) -> list[SearchResultChunk]:
        if not validated_chunk_ids:
            return []
        position_map = {chunk_id: index for index, chunk_id in enumerate(validated_chunk_ids)}
        selected = [item for item in matched_chunks if item.chunk_id in position_map]
        selected.sort(key=lambda item: position_map[item.chunk_id])
        return selected[:3]

    @staticmethod
    def _compute_confidence(selected_chunks: list[SearchResultChunk], answer_result: AnswerGenerationResult) -> str:
        if answer_result.insufficient_evidence or not selected_chunks:
            return "insufficient"
        top_score = selected_chunks[0].score.fused
        average_score = sum(item.score.fused for item in selected_chunks) / len(selected_chunks)
        document_distribution = Counter(str(item.document_id) for item in selected_chunks)
        dominant_share = max(document_distribution.values()) / len(selected_chunks)

        if answer_result.evidence_conflict:
            if top_score >= 0.6:
                return "medium"
            return "low"
        if len(selected_chunks) >= 2 and top_score >= 0.7 and average_score >= 0.55 and dominant_share >= 0.5:
            return "high"
        if top_score >= 0.4:
            return "medium"
        return "low"

    @staticmethod
    def _extract_trace_error(answer_result: AnswerGenerationResult) -> str | None:
        if not answer_result.raw_payload:
            return None
        error_text = answer_result.raw_payload.get("error_text")
        if error_text:
            return str(error_text)
        return None

    @staticmethod
    def _serialize_session(chat_session: ChatSession) -> ChatSessionRead:
        return ChatSessionRead.model_validate(chat_session)

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
