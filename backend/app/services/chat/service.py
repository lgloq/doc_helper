from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from time import perf_counter
import re
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
from app.schemas.search import SearchDebugInfo
from app.services.diagnostics import build_trace_pipeline_diagnosis
from app.services.chat.evidence_audit import build_evidence_audit
from app.services.llm.orchestrator import CopilotOrchestrator, CopilotRunResult
from app.services.observability.service import ObservabilityService

DEFAULT_CHAT_SESSION_TITLE = "新会话"
GENERIC_CHAT_SESSION_TITLES = {DEFAULT_CHAT_SESSION_TITLE, "New Chat"}
CLIENT_REQUEST_STALE_AFTER_SECONDS = 180


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
        preview_by_session_id = self.chat_repository.list_first_message_previews_by_role([item.id for item in sessions])
        return [
            self._serialize_session(
                item,
                first_user_message=preview_by_session_id.get(item.id, {}).get("user"),
                first_assistant_message=preview_by_session_id.get(item.id, {}).get("assistant"),
            )
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

    def create_message(self, actor: User, session_id: UUID, payload: ChatMessageCreate, allow_inflight_client_request: bool = False) -> ChatMessageCreateResponse:
        chat_session = self._get_session_or_404(actor, session_id, include_messages=True)
        existing_messages = list(chat_session.messages)
        client_request_id = self._normalize_client_request_id(payload.client_request_id)

        if client_request_id:
            restored_response = self._restore_completed_client_request(chat_session, client_request_id)
            if restored_response:
                return restored_response

        user_message = None
        if client_request_id:
            user_message = self._find_client_request_user_message(chat_session, client_request_id)
            if user_message:
                if user_message.content.strip() != payload.content.strip():
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="client_request_id 已用于另一条问题。",
                    )
                if not self._is_stale_processing_request(user_message) and not allow_inflight_client_request:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="该问题仍在处理中，请稍后刷新会话查看结果。",
                    )
                existing_messages = [message for message in existing_messages if message.id != user_message.id]
                user_message.message_metadata = {
                    **(user_message.message_metadata or {}),
                    "top_k": payload.top_k,
                    "client_request_id": client_request_id,
                    "client_request_status": "processing",
                    "processing_started_at": datetime.now(UTC).isoformat(),
                    "retry_count": int((user_message.message_metadata or {}).get("retry_count") or 0) + 1,
                }
                chat_session.updated_at = datetime.now(UTC)
                self.session.commit()

        if user_message is None:
            user_metadata = {"top_k": payload.top_k}
            if client_request_id:
                user_metadata.update(
                    {
                        "client_request_id": client_request_id,
                        "client_request_status": "processing",
                        "processing_started_at": datetime.now(UTC).isoformat(),
                    }
                )
            user_message = ChatMessage(
                session_id=chat_session.id,
                author_user_id=actor.id,
                role=MessageRole.USER,
                content=payload.content,
                message_metadata=user_metadata,
            )
            if not existing_messages and self._is_generic_session_title(chat_session.title):
                chat_session.title = self._truncate_session_title(payload.content)
            chat_session.updated_at = datetime.now(UTC)
            self.chat_repository.add_message(user_message)
            self.session.flush()
            if client_request_id:
                self.session.commit()

        user_message_id = user_message.id
        chat_session_id = chat_session.id

        prepare_started = perf_counter()
        try:
            prepared = self._prepare_answer(actor, payload.content, payload.top_k, existing_messages, session_id=chat_session.id)
            request_latency_ms = int((perf_counter() - prepare_started) * 1000)
        except Exception as exc:
            if client_request_id:
                self.session.rollback()
                persisted_user_message = self.chat_repository.get_message(user_message_id)
                if persisted_user_message is not None:
                    persisted_user_message.message_metadata = {
                        **(persisted_user_message.message_metadata or {}),
                        "client_request_id": client_request_id,
                        "client_request_status": "failed",
                        "failed_at": datetime.now(UTC).isoformat(),
                        "error_text": str(exc),
                    }
                    self.session.commit()
            raise

        assistant_result = prepared.answer_result
        evidence_audit = build_evidence_audit(assistant_result.answer, prepared.selected_chunks)
        pipeline_diagnosis = build_trace_pipeline_diagnosis(
            retrieval_debug=prepared.retrieval_response.debug,
            selected_citation_count=len(prepared.selected_chunks),
            evidence_audit=evidence_audit,
            error_text=self._extract_trace_error(prepared),
            insufficient_evidence=assistant_result.insufficient_evidence,
        )
        latency_breakdown = self._build_latency_breakdown(prepared, request_latency_ms)
        assistant_metadata = {
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
            "evidence_audit": evidence_audit,
            "pipeline_diagnosis": pipeline_diagnosis,
            "latency_breakdown": latency_breakdown,
            "raw_payload": assistant_result.raw_payload,
        }
        if client_request_id:
            assistant_metadata.update(
                {
                    "client_request_id": client_request_id,
                    "client_request_status": "completed",
                    "completed_at": datetime.now(UTC).isoformat(),
                    "user_message_id": str(user_message_id),
                }
            )
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
            message_metadata=assistant_metadata,
        )
        chat_session.updated_at = datetime.now(UTC)
        self.chat_repository.add_message(assistant_message)
        self.session.flush()
        assistant_message_id = assistant_message.id

        if client_request_id:
            user_message.message_metadata = {
                **(user_message.message_metadata or {}),
                "client_request_id": client_request_id,
                "client_request_status": "completed",
                "assistant_message_id": str(assistant_message_id),
                "completed_at": datetime.now(UTC).isoformat(),
            }

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
                    "evidence_audit": evidence_audit,
                    "pipeline_diagnosis": pipeline_diagnosis,
                },
            )
        except Exception:
            pass

        if prepared.retrieval_response.debug.permission_probe_early_stop_applied:
            try:
                self.observability_service.record_permission_denied_retrieval(
                    actor=actor,
                    query_text=payload.content,
                    retrieval_response=prepared.retrieval_response,
                    source="chat_api",
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

    @staticmethod
    def _build_latency_breakdown(prepared: PreparedChatAnswer, total_latency_ms: int) -> dict[str, int | None]:
        router_latency_ms = prepared.router_result.latency_ms
        retrieval_latency_ms = prepared.retrieval_response.debug.search_total_latency_ms
        answer_total_latency_ms = prepared.answer_result.latency_ms
        generation_latency_ms = None
        if answer_total_latency_ms is not None:
            generation_latency_ms = max(0, int(answer_total_latency_ms) - int(router_latency_ms or 0))
        return {
            "router_latency_ms": router_latency_ms,
            "retrieval_latency_ms": retrieval_latency_ms,
            "answer_generation_latency_ms": generation_latency_ms,
            "answer_total_latency_ms": answer_total_latency_ms,
            "total_latency_ms": total_latency_ms,
        }

    def _restore_completed_client_request(
        self,
        chat_session: ChatSession,
        client_request_id: str,
    ) -> ChatMessageCreateResponse | None:
        user_message = self._find_client_request_user_message(chat_session, client_request_id)
        assistant_message = self._find_client_request_assistant_message(chat_session, client_request_id)
        if user_message is None or assistant_message is None:
            return None
        return self._build_message_create_response_from_messages(chat_session.id, user_message, assistant_message)

    def _build_message_create_response_from_messages(
        self,
        session_id: UUID,
        user_message: ChatMessage,
        assistant_message: ChatMessage,
    ) -> ChatMessageCreateResponse:
        hydrated_user_message = self.chat_repository.get_message(user_message.id) or user_message
        hydrated_assistant_message = self.chat_repository.get_message(assistant_message.id) or assistant_message
        assistant_citations = [self._serialize_citation(item) for item in getattr(hydrated_assistant_message, "citations", [])]
        retrieval_debug = self._read_retrieval_debug_from_message(hydrated_assistant_message)
        return ChatMessageCreateResponse(
            session_id=session_id,
            user_message=self._serialize_message(hydrated_user_message),
            assistant_message=self._serialize_message(hydrated_assistant_message),
            citations=assistant_citations,
            retrieval_debug=retrieval_debug,
        )

    @staticmethod
    def _read_retrieval_debug_from_message(message: ChatMessage) -> SearchDebugInfo:
        metadata = message.message_metadata or {}
        payload = metadata.get("retrieval_debug") if isinstance(metadata, dict) else None
        if isinstance(payload, dict):
            try:
                return SearchDebugInfo.model_validate(payload)
            except Exception:
                pass
        return SearchDebugInfo(
            accessible_document_count=0,
            lexical_candidate_count=0,
            vector_candidate_count=0,
            fusion_strategy="restored_without_debug",
        )

    @staticmethod
    def _normalize_client_request_id(value: str | None) -> str | None:
        normalized = (value or "").strip()
        return normalized or None

    @staticmethod
    def _find_client_request_user_message(chat_session: ChatSession, client_request_id: str) -> ChatMessage | None:
        for message in chat_session.messages:
            metadata = message.message_metadata or {}
            if (
                message.role == MessageRole.USER
                and isinstance(metadata, dict)
                and metadata.get("client_request_id") == client_request_id
            ):
                return message
        return None

    @staticmethod
    def _find_client_request_assistant_message(chat_session: ChatSession, client_request_id: str) -> ChatMessage | None:
        for message in chat_session.messages:
            metadata = message.message_metadata or {}
            if (
                message.role == MessageRole.ASSISTANT
                and isinstance(metadata, dict)
                and metadata.get("client_request_id") == client_request_id
            ):
                return message
        return None

    @staticmethod
    def _is_stale_processing_request(user_message: ChatMessage) -> bool:
        metadata = user_message.message_metadata or {}
        if not isinstance(metadata, dict):
            return True
        if metadata.get("client_request_status") in {"completed", "failed"}:
            return True
        started_at = metadata.get("processing_started_at")
        if not isinstance(started_at, str):
            return True
        try:
            parsed = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        except ValueError:
            return True
        return datetime.now(UTC) - parsed > timedelta(seconds=CLIENT_REQUEST_STALE_AFTER_SECONDS)

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
        first_assistant_message: str | None = None,
    ) -> str:
        is_auto_question_title = bool(
            first_user_message and chat_session.title == cls._truncate_session_title(first_user_message)
        )
        if cls._is_generic_session_title(chat_session.title) or is_auto_question_title:
            summarized_from_question = cls._summarize_session_title_from_question(first_user_message)
            if summarized_from_question:
                return summarized_from_question
            if first_assistant_message:
                return cls._summarize_session_title_from_answer(first_assistant_message)
        if cls._is_generic_session_title(chat_session.title) and first_user_message:
            return cls._truncate_session_title(first_user_message)
        return chat_session.title

    @classmethod
    def _serialize_session(
        cls,
        chat_session: ChatSession,
        *,
        first_user_message: str | None = None,
        first_assistant_message: str | None = None,
    ) -> ChatSessionRead:
        if "messages" in chat_session.__dict__:
            if first_user_message is None:
                first_user_message = next(
                    (message.content for message in chat_session.messages if message.role == MessageRole.USER),
                    None,
                )
            if first_assistant_message is None:
                first_assistant_message = next(
                    (message.content for message in chat_session.messages if message.role == MessageRole.ASSISTANT),
                    None,
                )
        payload = {
            "id": chat_session.id,
            "user_id": chat_session.user_id,
            "title": chat_session.title,
            "display_title": cls._resolve_display_title(
                chat_session,
                first_user_message=first_user_message,
                first_assistant_message=first_assistant_message,
            ),
            "created_at": chat_session.created_at,
            "updated_at": chat_session.updated_at,
        }
        return ChatSessionRead.model_validate(payload)

    @classmethod
    def _summarize_session_title_from_answer(cls, answer: str, limit: int = 34) -> str:
        compact = " ".join(answer.strip().split())
        if not compact:
            return DEFAULT_CHAT_SESSION_TITLE
        compact = re.sub(r"^(根据当前可访问文档中的证据[，,:： ]*|根据证据[，,:： ]*|当前可访问文档显示[，,:： ]*)", "", compact)
        sentence = re.split(r"[。！？!?；;\n]+", compact, maxsplit=1)[0].strip(" ，,：:；;")
        if not sentence:
            sentence = compact[:limit]
        if len(sentence) <= limit:
            return sentence
        return sentence[: limit - 3].rstrip() + "..."

    @classmethod
    def _summarize_session_title_from_question(cls, question: str | None, limit: int = 22) -> str | None:
        if not question:
            return None
        compact = " ".join(question.strip().split())
        if not compact:
            return None
        compact = re.sub(r"^《([^》]+)》里[，,:： ]*", "", compact)
        compact = compact.strip("？?。！!，,；;：:")
        normalized = re.sub(r"\s+", "", compact)

        if "节假日" in compact:
            return "节假日安排与值班要求"
        if "面向客户" in compact and "事故" in compact:
            return "客户事故响应要求"
        if "安全例外登记" in compact and "补偿控制" in compact and ("待办" in compact or "整理" in compact):
            return "安全例外补偿控制待办"
        if "安全例外登记" in compact:
            return "安全例外登记要求"
        if "临时高权限" in compact or "高权限访问" in compact:
            return "临时高权限访问审批"
        if "供应商" in compact and ("导出" in compact or "客户数据" in compact or "生产系统" in compact):
            return "供应商紧急接入与导出限制"
        if "客户" in compact and "手机号" in compact and "导出" in compact:
            return "客户数据导出审批与脱敏要求"
        if "客户" in compact and "数据导出" in compact and "审批" in compact:
            return "客户数据导出审批要求"
        if "数据导出" in compact and "审批" in compact:
            return "数据导出审批要求"
        if "工单" in compact and "首次响应时间" in compact:
            return "工单首次响应要求"
        if any(token in normalized for token in ("要注意的地方", "注意事项", "注意什么", "有什么要注意")):
            subject = cls._extract_question_subject(compact)
            if subject:
                return cls._truncate_phrase(f"{subject}注意事项", limit)
        if "检查清单" in compact and any(token in normalized for token in ("使用", "怎么用", "如何用", "注意")):
            subject = cls._extract_question_subject(compact)
            if subject:
                return cls._truncate_phrase(f"{subject}使用注意事项", limit)
        if "平台发布检查清单" in compact:
            return "平台发布检查清单要求"
        if "审批" in compact:
            subject = cls._extract_question_subject(compact)
            if subject:
                return cls._truncate_phrase(f"{subject}审批要求", limit)
        if "要求" in compact or "怎么" in compact or "什么" in compact or "多久" in compact or "哪些" in compact:
            subject = cls._extract_question_subject(compact)
            if subject:
                return cls._truncate_phrase(f"{subject}要求", limit)
        return cls._truncate_phrase(compact, limit)

    @staticmethod
    def _extract_question_subject(text: str) -> str | None:
        subject = text
        subject = re.sub(r"^(我想|我要|我需要|我想要|请问|想问一下|想问|帮我看下|帮我看看)", "", subject)
        subject = re.sub(r"^(使用|查看|阅读|了解|看下|看看)", "", subject)
        subject = re.sub(r"如果为了.*?[,，]", "", subject)
        subject = re.sub(r"(有什么要注意的地方|有什么要注意|有哪些注意事项|注意事项是什么|怎么用|如何用)$", "", subject)
        subject = re.sub(r"(由谁审批|谁来批|谁审批|处理时限.*|多久.*|哪些.*|什么.*|怎么.*)$", "", subject)
        subject = re.sub(r"(是什么样的|是什么|是怎样的)$", "", subject)
        subject = re.sub(r"(能不能|可不可以|是否)", "", subject)
        subject = re.sub(r"[，,]\s*有$", "", subject)
        subject = subject.strip(" ，,；;：:")
        if len(subject) < 2:
            return None
        return subject

    @staticmethod
    def _truncate_phrase(text: str, limit: int) -> str:
        compact = text.strip(" ，,；;：:")
        if len(compact) <= limit:
            return compact
        return compact[: limit - 1].rstrip() + "…"

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
