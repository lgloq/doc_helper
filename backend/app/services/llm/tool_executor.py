from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from app.models.chat import ChatMessage, MessageCitation
from app.models.enums import MessageRole
from app.models.user import User
from app.schemas.llm import CopilotExecutionMetadata, RouterDecisionResult, ToolAction, ToolObservation
from app.schemas.search import SearchDebugInfo, SearchResponse, SearchResultChunk
from app.services.chat.memory import ConversationMemory
from app.services.llm.tool_registry import DEFAULT_TOOL_REGISTRY, ToolRegistry
from app.services.llm.tools import CopilotToolService, VersionCompareToolResult, WorkflowToolResult


@dataclass
class ToolExecutionResult:
    observation: ToolObservation
    tool_metadata: CopilotExecutionMetadata
    retrieval_response: SearchResponse | None = None
    candidate_chunks: list[SearchResultChunk] = field(default_factory=list)
    target_document: str | None = None
    version_compare_result: VersionCompareToolResult | None = None
    workflow_result: WorkflowToolResult | None = None
    synthetic_messages: list[ChatMessage] = field(default_factory=list)
    refusal_reason: str | None = None
    answer_hint: str | None = None


class ToolExecutor:
    def __init__(self, tool_service: CopilotToolService, *, tool_registry: ToolRegistry | None = None) -> None:
        self.tools = tool_service
        self.tool_registry = tool_registry or DEFAULT_TOOL_REGISTRY

    def execute(
        self,
        *,
        actor: User,
        question: str,
        session_id: UUID | None,
        action: ToolAction,
        router_result: RouterDecisionResult,
        conversation_memory: ConversationMemory,
        working_messages: list[ChatMessage],
        top_k: int,
    ) -> ToolExecutionResult:
        tool_name = action.tool_name or ""
        if self.tool_registry.get(tool_name) is None:
            return self._failed_result(
                action=action,
                tool_name=tool_name or "unknown_tool",
                output_summary=f"工具 {tool_name or 'unknown_tool'} 不在允许列表中，已拒绝执行。",
                refusal_reason="unknown_tool_name",
            )
        action = action.model_copy(update={"tool_args": self.tool_registry.sanitize_args(tool_name, action.tool_args)})

        if tool_name == "search_docs":
            return self._execute_search_docs(
                actor=actor,
                question=question,
                session_id=session_id,
                action=action,
                router_result=router_result,
                top_k=top_k,
            )
        if tool_name == "compare_versions":
            return self._execute_compare_versions(
                actor=actor,
                question=question,
                session_id=session_id,
                action=action,
                router_result=router_result,
            )
        if tool_name == "extract_todos":
            return self._execute_extract_todos(
                actor=actor,
                action=action,
                session_id=session_id,
                working_messages=working_messages,
            )
        if tool_name == "generate_weekly_report":
            return self._execute_generate_weekly_report(
                actor=actor,
                action=action,
                session_id=session_id,
                working_messages=working_messages,
            )
        return self._execute_generate_faq(
            actor=actor,
            action=action,
            session_id=session_id,
            working_messages=working_messages,
        )

    def _execute_search_docs(
        self,
        *,
        actor: User,
        question: str,
        session_id: UUID | None,
        action: ToolAction,
        router_result: RouterDecisionResult,
        top_k: int,
    ) -> ToolExecutionResult:
        query = str(action.tool_args.get("query") or question).strip()
        target_document = _as_string(
            action.tool_args.get("target_document"),
            router_result.decision.target_document_title,
            router_result.decision.requested_document_name,
        )

        if target_document or router_result.decision.intent == "document_qa":
            tool_result = self.tools.get_document_context(actor, target_document, query, top_k)
            retrieval_response = tool_result.retrieval_response
            candidate_chunks = list(retrieval_response.matched_chunks)
            refusal_reason = tool_result.refusal_reason
            observation_status = "completed" if candidate_chunks and not refusal_reason else "failed"
            output_summary = _summarize_search_output(candidate_chunks, tool_result.document_title, refusal_reason)
            tool_metadata = CopilotExecutionMetadata(
                tool_name="search_docs",
                tool_input={"query": query, "top_k": top_k, "target_document": target_document},
                tool_output_summary={
                    "matched_chunks": len(candidate_chunks),
                    "document_title": tool_result.document_title,
                    "refusal_reason": refusal_reason,
                },
                retrieval_debug=retrieval_response.debug,
            )
            synthetic_messages = (
                self._build_search_messages(
                    session_id=session_id,
                    actor=actor,
                    question=query,
                    retrieval_response=retrieval_response,
                    target_document=tool_result.document_title or target_document,
                )
                if candidate_chunks
                else []
            )
            return ToolExecutionResult(
                observation=ToolObservation(
                    step_index=action.step_index,
                    tool_name="search_docs",
                    status=observation_status,  # type: ignore[arg-type]
                    output_summary=output_summary,
                    evidence_refs=_chunk_refs(candidate_chunks),
                    raw_output={
                        "matched_chunks": len(candidate_chunks),
                        "document_title": tool_result.document_title,
                        "refusal_reason": refusal_reason,
                    },
                ),
                tool_metadata=tool_metadata,
                retrieval_response=retrieval_response,
                candidate_chunks=candidate_chunks,
                target_document=tool_result.document_title or target_document,
                synthetic_messages=synthetic_messages,
                refusal_reason=refusal_reason,
            )

        retrieval_response = self.tools.search_accessible_documents(actor, query, top_k)
        candidate_chunks = list(retrieval_response.matched_chunks)
        output_summary = f"命中 {len(candidate_chunks)} 个候选分块。"
        tool_metadata = CopilotExecutionMetadata(
            tool_name="search_docs",
            tool_input={"query": query, "top_k": top_k},
            tool_output_summary={"matched_chunks": len(candidate_chunks)},
            retrieval_debug=retrieval_response.debug,
        )
        synthetic_messages = self._build_search_messages(
            session_id=session_id,
            actor=actor,
            question=query,
            retrieval_response=retrieval_response,
            target_document=None,
        )
        return ToolExecutionResult(
            observation=ToolObservation(
                step_index=action.step_index,
                tool_name="search_docs",
                status="completed" if candidate_chunks else "failed",
                output_summary=output_summary if candidate_chunks else "未检索到足够相关的可访问内容。",
                evidence_refs=_chunk_refs(candidate_chunks),
                raw_output={"matched_chunks": len(candidate_chunks)},
            ),
            tool_metadata=tool_metadata,
            retrieval_response=retrieval_response,
            candidate_chunks=candidate_chunks,
            synthetic_messages=synthetic_messages if candidate_chunks else [],
            refusal_reason=None if candidate_chunks else "insufficient_relevant_evidence",
        )

    def _execute_compare_versions(
        self,
        *,
        actor: User,
        question: str,
        session_id: UUID | None,
        action: ToolAction,
        router_result: RouterDecisionResult,
    ) -> ToolExecutionResult:
        target_document = _as_string(
            action.tool_args.get("target_document"),
            router_result.decision.target_document_title,
            router_result.decision.requested_document_name,
        )
        from_version_ref = _as_string(action.tool_args.get("from_version_ref"), router_result.decision.from_version_ref)
        to_version_ref = _as_string(action.tool_args.get("to_version_ref"), router_result.decision.to_version_ref)
        tool_result = self.tools.compare_document_versions(actor, target_document, from_version_ref, to_version_ref)
        refusal_reason = tool_result.refusal_reason
        observation_status = "completed" if tool_result.summary and not refusal_reason else "failed"
        output_summary = _summarize_compare_output(tool_result)
        tool_metadata = CopilotExecutionMetadata(
            tool_name="compare_versions",
            tool_input={
                "target_document": target_document,
                "from_version_ref": from_version_ref,
                "to_version_ref": to_version_ref,
            },
            tool_output_summary={
                "document_title": tool_result.document_title,
                "refusal_reason": refusal_reason,
                "has_summary": tool_result.summary is not None,
            },
        )
        synthetic_messages = (
            self._build_compare_messages(
                session_id=session_id,
                actor=actor,
                question=question,
                compare_result=tool_result,
            )
            if tool_result.summary and not refusal_reason
            else []
        )
        return ToolExecutionResult(
            observation=ToolObservation(
                step_index=action.step_index,
                tool_name="compare_versions",
                status=observation_status,  # type: ignore[arg-type]
                output_summary=output_summary,
                evidence_refs=_compare_refs(tool_result),
                raw_output={
                    "document_title": tool_result.document_title,
                    "refusal_reason": refusal_reason,
                    "summary": tool_result.summary.summary if tool_result.summary else None,
                },
            ),
            tool_metadata=tool_metadata,
            target_document=tool_result.document_title or target_document,
            version_compare_result=tool_result,
            synthetic_messages=synthetic_messages,
            refusal_reason=refusal_reason,
        )

    def _execute_extract_todos(
        self,
        *,
        actor: User,
        action: ToolAction,
        session_id: UUID | None,
        working_messages: list[ChatMessage],
    ) -> ToolExecutionResult:
        if not _has_grounded_context(working_messages):
            return self._insufficient_context_result(
                action=action,
                tool_name="extract_todos",
                output_summary="当前上下文缺少足够稳定的问答或检索结果，暂不直接提取待办。",
            )
        workflow_result = self.tools.generate_tasks_from_messages(actor, working_messages, session_id)
        item_count = len((workflow_result.structured_payload or {}).get("items", []))
        return ToolExecutionResult(
            observation=ToolObservation(
                step_index=action.step_index,
                tool_name="extract_todos",
                status="completed",
                output_summary=f"已提取 {item_count} 条待办事项。",
                evidence_refs=_workflow_refs(workflow_result),
                raw_output={"item_count": item_count},
            ),
            tool_metadata=CopilotExecutionMetadata(
                tool_name="extract_todos",
                tool_input={"session_id": str(session_id) if session_id else None},
                tool_output_summary={"artifact_type": "tasks", "item_count": item_count},
            ),
            workflow_result=workflow_result,
            answer_hint=f"已根据当前上下文整理出 {item_count} 条待办事项。",
        )

    def _execute_generate_weekly_report(
        self,
        *,
        actor: User,
        action: ToolAction,
        session_id: UUID | None,
        working_messages: list[ChatMessage],
    ) -> ToolExecutionResult:
        if not _has_grounded_context(working_messages):
            return self._insufficient_context_result(
                action=action,
                tool_name="generate_weekly_report",
                output_summary="当前上下文缺少足够稳定的问答结果，暂不生成周报。",
            )
        workflow_result = self.tools.generate_weekly_report_from_messages(actor, working_messages, session_id)
        report_title = ((workflow_result.structured_payload or {}).get("report") or {}).get("title") or "周报草稿"
        return ToolExecutionResult(
            observation=ToolObservation(
                step_index=action.step_index,
                tool_name="generate_weekly_report",
                status="completed",
                output_summary=f"已生成周报草稿《{report_title}》。",
                evidence_refs=_workflow_refs(workflow_result),
                raw_output={"report_title": report_title},
            ),
            tool_metadata=CopilotExecutionMetadata(
                tool_name="generate_weekly_report",
                tool_input={"session_id": str(session_id) if session_id else None},
                tool_output_summary={"artifact_type": "weekly_report", "report_title": report_title},
            ),
            workflow_result=workflow_result,
            answer_hint=f"已根据当前上下文生成周报草稿《{report_title}》。",
        )

    def _execute_generate_faq(
        self,
        *,
        actor: User,
        action: ToolAction,
        session_id: UUID | None,
        working_messages: list[ChatMessage],
    ) -> ToolExecutionResult:
        if not _has_grounded_context(working_messages):
            return self._insufficient_context_result(
                action=action,
                tool_name="generate_faq",
                output_summary="当前上下文缺少足够稳定的问答结果，暂不生成 FAQ。",
            )
        workflow_result = self.tools.generate_faq_from_messages(actor, working_messages, session_id)
        entry_count = len((workflow_result.structured_payload or {}).get("entries", []))
        return ToolExecutionResult(
            observation=ToolObservation(
                step_index=action.step_index,
                tool_name="generate_faq",
                status="completed",
                output_summary=f"已生成 {entry_count} 条 FAQ 草稿。",
                evidence_refs=_workflow_refs(workflow_result),
                raw_output={"entry_count": entry_count},
            ),
            tool_metadata=CopilotExecutionMetadata(
                tool_name="generate_faq",
                tool_input={"session_id": str(session_id) if session_id else None},
                tool_output_summary={"artifact_type": "faq", "entry_count": entry_count},
            ),
            workflow_result=workflow_result,
            answer_hint=f"已根据当前上下文生成 {entry_count} 条 FAQ 草稿。",
        )

    @staticmethod
    def _build_search_messages(
        *,
        session_id: UUID | None,
        actor: User,
        question: str,
        retrieval_response: SearchResponse,
        target_document: str | None,
    ) -> list[ChatMessage]:
        if not retrieval_response.matched_chunks:
            return []
        safe_session_id = session_id or UUID(int=0)
        user_message = ChatMessage(
            session_id=safe_session_id,
            author_user_id=actor.id,
            role=MessageRole.USER,
            content=question,
            message_metadata={"synthetic": True, "source": "agent_runner"},
        )
        assistant_message = ChatMessage(
            session_id=safe_session_id,
            author_user_id=None,
            role=MessageRole.ASSISTANT,
            content=_build_search_message_content(retrieval_response.matched_chunks, target_document),
            confidence="medium",
            insufficient_evidence=False,
            message_metadata={"synthetic": True, "source": "agent_runner", "target_document": target_document},
        )
        assistant_message.citations = [
            _build_message_citation(None, index, chunk)
            for index, chunk in enumerate(retrieval_response.matched_chunks[:3], start=1)
        ]
        return [user_message, assistant_message]

    @staticmethod
    def _build_compare_messages(
        *,
        session_id: UUID | None,
        actor: User,
        question: str,
        compare_result: VersionCompareToolResult,
    ) -> list[ChatMessage]:
        if compare_result.summary is None:
            return []
        safe_session_id = session_id or UUID(int=0)
        user_message = ChatMessage(
            session_id=safe_session_id,
            author_user_id=actor.id,
            role=MessageRole.USER,
            content=question,
            message_metadata={"synthetic": True, "source": "agent_runner"},
        )
        assistant_message = ChatMessage(
            session_id=safe_session_id,
            author_user_id=None,
            role=MessageRole.ASSISTANT,
            content=_build_compare_message_content(compare_result),
            confidence="high",
            insufficient_evidence=False,
            message_metadata={"synthetic": True, "source": "agent_runner", "target_document": compare_result.document_title},
        )
        return [user_message, assistant_message]

    @staticmethod
    def _failed_result(
        *,
        action: ToolAction,
        tool_name: str,
        output_summary: str,
        refusal_reason: str,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            observation=ToolObservation(
                step_index=action.step_index,
                tool_name=tool_name,
                status="failed",
                output_summary=output_summary,
                evidence_refs=[],
                raw_output={"refusal_reason": refusal_reason},
            ),
            tool_metadata=CopilotExecutionMetadata(
                tool_name=tool_name,
                tool_input=dict(action.tool_args),
                tool_output_summary={"refusal_reason": refusal_reason},
                retrieval_debug=_empty_debug_info(),
            ),
            refusal_reason=refusal_reason,
        )

    @staticmethod
    def _insufficient_context_result(
        *,
        action: ToolAction,
        tool_name: str,
        output_summary: str,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            observation=ToolObservation(
                step_index=action.step_index,
                tool_name=tool_name,
                status="insufficient_context",
                output_summary=output_summary,
                evidence_refs=[],
                raw_output={"refusal_reason": "insufficient_context"},
            ),
            tool_metadata=CopilotExecutionMetadata(
                tool_name=tool_name,
                tool_input=dict(action.tool_args),
                tool_output_summary={"status": "skipped_due_to_insufficient_context", "refusal_reason": "insufficient_context"},
                retrieval_debug=_empty_debug_info(),
            ),
            refusal_reason="insufficient_context",
        )


def _has_grounded_context(messages: list[ChatMessage]) -> bool:
    return any(message.role == MessageRole.ASSISTANT and not message.insufficient_evidence for message in messages)


def _as_string(*values: object) -> str | None:
    for value in values:
        if value is None:
            continue
        cleaned = str(value).strip()
        if cleaned:
            return cleaned
    return None


def _summarize_search_output(
    chunks: list[SearchResultChunk],
    document_title: str | None,
    refusal_reason: str | None,
) -> str:
    if refusal_reason == "target_document_not_accessible_or_not_found":
        return "目标文档不在当前可访问范围内，未执行有效检索。"
    if refusal_reason == "no_relevant_evidence_in_target_document":
        return f"已在“{document_title or '目标文档'}”内检索，但未找到足够相关的证据。"
    if not chunks:
        return "未检索到足够相关的可访问内容。"
    prefix = f"目标文档={document_title}；" if document_title else ""
    return f"{prefix}命中 {len(chunks)} 个候选分块。"


def _summarize_compare_output(compare_result: VersionCompareToolResult) -> str:
    if compare_result.refusal_reason == "target_document_not_accessible_or_not_found":
        return "目标文档不在当前可访问范围内，未执行版本对比。"
    if compare_result.refusal_reason == "insufficient_versions_for_compare":
        return "当前可访问范围内版本数量不足，无法完成版本对比。"
    if compare_result.refusal_reason == "unable_to_resolve_version_pair":
        return "无法解析需要比较的版本对。"
    if compare_result.summary is None:
        return "版本对比未返回可用摘要。"
    return (
        f"文档={compare_result.document_title}；"
        f"新增 {len(compare_result.summary.additions)} 项、修改 {len(compare_result.summary.modifications)} 项。"
    )


def _chunk_refs(chunks: list[SearchResultChunk]) -> list[str]:
    refs: list[str] = []
    for chunk in chunks[:3]:
        refs.append(f"{chunk.document_title} · 第 {chunk.chunk_index + 1} 段")
    return refs


def _compare_refs(compare_result: VersionCompareToolResult) -> list[str]:
    if compare_result.summary is None:
        return []
    return [
        f"{compare_result.document_title} {compare_result.summary.from_version_number}->{compare_result.summary.to_version_number}",
    ]


def _workflow_refs(workflow_result: WorkflowToolResult) -> list[str]:
    refs: list[str] = []
    for citation in workflow_result.citations[:3]:
        title = citation.get("document_title") if isinstance(citation, dict) else None
        preview = citation.get("preview") if isinstance(citation, dict) else None
        refs.append(str(title or preview or "会话上下文"))
    return refs


def _build_search_message_content(chunks: list[SearchResultChunk], target_document: str | None) -> str:
    prefix = f"已检索“{target_document}”中的相关内容：" if target_document else "已检索到以下相关内容："
    excerpts = [chunk.preview for chunk in chunks[:3]]
    return prefix + "\n" + "\n".join(excerpts)


def _build_compare_message_content(compare_result: VersionCompareToolResult) -> str:
    summary = compare_result.summary
    if summary is None:
        return "版本对比未返回可用结果。"
    parts = [f"{compare_result.document_title} 版本差异摘要：{summary.summary}"]
    if summary.additions:
        parts.append("新增内容：" + "；".join(summary.additions[:4]))
    if summary.modifications:
        parts.append("修改内容：" + "；".join(summary.modifications[:4]))
    return "\n".join(parts)


def _build_message_citation(message_id: UUID | None, rank: int, chunk: SearchResultChunk) -> MessageCitation:
    return MessageCitation(
        message_id=message_id or UUID(int=0),
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        document_version_id=chunk.document_version_id,
        document_title=chunk.document_title,
        version_number=chunk.version_number or 1,
        chunk_index=chunk.chunk_index or 0,
        page_number_start=chunk.page_number_start,
        page_number_end=chunk.page_number_end,
        paragraph_start=chunk.paragraph_start,
        paragraph_end=chunk.paragraph_end,
        preview=chunk.preview,
        lexical_score=chunk.score.lexical_raw,
        vector_score=chunk.score.vector_raw,
        fused_score=chunk.score.fused,
        rank=rank,
        citation_metadata=chunk.citation_metadata,
    )


def _empty_debug_info() -> SearchDebugInfo:
    return SearchDebugInfo(
        accessible_document_count=0,
        lexical_candidate_count=0,
        vector_candidate_count=0,
        fusion_strategy="min-max weighted sum",
    )
