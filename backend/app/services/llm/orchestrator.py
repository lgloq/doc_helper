from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from app.schemas.llm import (
    AgentStep,
    CopilotExecutionMetadata,
    QAAnswerResult,
    RouterDecision,
    RouterDecisionResult,
    ToolCitation,
    VersionCompareResult,
    WorkflowGenerationResult,
)
from app.services.chat.memory import ConversationMemory, build_conversation_memory, has_usable_workflow_context
from app.schemas.search import SearchDebugInfo, SearchResponse, SearchResultChunk
from app.services.chat.generation import AnswerGenerationResult, AnswerGeneratorFactory
from app.services.chat.prompts import validate_used_chunk_ids
from app.services.chat.reliability import should_abstain_from_answer
from app.services.llm.router import LLMRouterService
from app.services.llm.tools import CopilotToolService

TOOL_SEARCH_DOCS = "search_docs"
TOOL_COMPARE_VERSIONS = "compare_versions"
TOOL_EXTRACT_TODOS = "extract_todos"
TOOL_GENERATE_WEEKLY_REPORT = "generate_weekly_report"
TOOL_GENERATE_FAQ = "generate_faq"


@dataclass
class CopilotRunResult:
    router_result: RouterDecisionResult
    tool_metadata: CopilotExecutionMetadata
    answer_result: AnswerGenerationResult
    retrieval_response: SearchResponse
    candidate_chunks: list[SearchResultChunk]
    selected_chunks: list[SearchResultChunk]
    confidence: str
    structured_result: QAAnswerResult | VersionCompareResult | WorkflowGenerationResult
    agent_steps: list[AgentStep]


class CopilotOrchestrator:
    def __init__(self, session) -> None:
        self.router = LLMRouterService()
        self.tools = CopilotToolService(session)
        self.answer_generator = AnswerGeneratorFactory.create()

    def run(
        self,
        *,
        actor,
        question: str,
        session_id: UUID,
        top_k: int,
        existing_messages,
    ) -> CopilotRunResult:
        memory = build_conversation_memory(existing_messages)
        accessible_documents = self.tools.list_accessible_documents(actor)
        router_result = self.router.route(
            question=question,
            accessible_documents=accessible_documents,
            conversation_context=memory,
        )
        decision = router_result.decision

        if decision.intent == "document_qa":
            return self._run_document_qa(actor, question, existing_messages, memory, top_k, router_result)
        if decision.intent == "topic_qa":
            return self._run_topic_qa(actor, question, existing_messages, memory, top_k, router_result)
        if decision.intent == "version_compare":
            return self._run_version_compare(actor, question, memory, router_result)
        if decision.intent == "workflow_generation":
            if session_id.int == 0:
                return self._build_workflow_result(
                    question=question,
                    router_result=router_result,
                    conversation_memory=memory,
                    tool_metadata=CopilotExecutionMetadata(tool_name="none", tool_input={"session_id": None}),
                    answer="当前请求需要依赖已有会话上下文，预览模式下暂时无法直接生成派生结果。",
                    confidence="insufficient",
                    artifact_type=None,
                    structured_payload=None,
                    citations=[],
                    refusal_reason="missing_session_context",
                )
            return self._run_workflow_generation(actor, question, session_id, existing_messages, memory, router_result)
        return self._build_refusal_result(
            question=question,
            router_result=router_result,
            answer="我还不能可靠地理解这条请求。请更明确地说明你想查询的文档、主题，或直接说明要生成待办、周报、FAQ、版本差异。",
            refusal_reason="unsupported_or_unclear",
            conversation_memory=memory,
            tool_name="none",
            tool_input={"question": question},
            tool_output_summary={"status": "not_routed"},
        )

    def _run_document_qa(
        self,
        actor,
        question: str,
        existing_messages,
        conversation_memory: ConversationMemory,
        top_k: int,
        router_result: RouterDecisionResult,
    ) -> CopilotRunResult:
        decision = router_result.decision
        requested_document = decision.target_document_title or decision.target_document_id or decision.requested_document_name
        tool_result = self.tools.get_document_context(actor, requested_document, question, top_k)
        tool_metadata = CopilotExecutionMetadata(
            tool_name=TOOL_SEARCH_DOCS,
            tool_input={
                "query": question,
                "top_k": top_k,
                "target_document": str(requested_document) if requested_document else None,
            },
            tool_output_summary={
                "matched_chunks": len(tool_result.retrieval_response.matched_chunks),
                "document_title": tool_result.document_title,
                "refusal_reason": tool_result.refusal_reason,
            },
            retrieval_debug=tool_result.retrieval_response.debug,
        )
        if decision.should_refuse_if_inaccessible and not decision.target_document_title:
            return self._build_refusal_result(
                question=question,
                router_result=router_result,
                answer="当前可访问范围内未找到相关文档内容。该文档可能不存在，或你当前没有访问权限。",
                refusal_reason="target_document_not_accessible_or_not_found",
                conversation_memory=conversation_memory,
                tool_name=tool_metadata.tool_name,
                tool_input=tool_metadata.tool_input,
                tool_output_summary=tool_metadata.tool_output_summary,
                retrieval_response=tool_result.retrieval_response,
                target_document=decision.requested_document_name,
                intent="document_qa",
            )
        if tool_result.refusal_reason == "no_relevant_evidence_in_target_document" or not tool_result.retrieval_response.matched_chunks:
            target_name = tool_result.document_title or decision.requested_document_name or "该文档"
            return self._build_refusal_result(
                question=question,
                router_result=router_result,
                answer=f"在“{target_name}”当前可访问的内容中，未找到足够相关的证据来回答这个问题。",
                refusal_reason=tool_result.refusal_reason or "insufficient_relevant_evidence",
                conversation_memory=conversation_memory,
                tool_name=tool_metadata.tool_name,
                tool_input=tool_metadata.tool_input,
                tool_output_summary=tool_metadata.tool_output_summary,
                retrieval_response=tool_result.retrieval_response,
                target_document=target_name,
                intent="document_qa",
            )
        return self._generate_grounded_qa(
            question=question,
            existing_messages=existing_messages,
            conversation_memory=conversation_memory,
            router_result=router_result,
            retrieval_response=tool_result.retrieval_response,
            candidate_chunks=tool_result.retrieval_response.matched_chunks,
            tool_metadata=tool_metadata,
            target_document=tool_result.document_title,
            allow_low_score=True,
        )

    def _run_topic_qa(
        self,
        actor,
        question: str,
        existing_messages,
        conversation_memory: ConversationMemory,
        top_k: int,
        router_result: RouterDecisionResult,
    ) -> CopilotRunResult:
        retrieval_response = self.tools.search_accessible_documents(actor, question, top_k)
        tool_metadata = CopilotExecutionMetadata(
            tool_name=TOOL_SEARCH_DOCS,
            tool_input={"query": question, "top_k": top_k},
            tool_output_summary={"matched_chunks": len(retrieval_response.matched_chunks)},
            retrieval_debug=retrieval_response.debug,
        )
        abstain_decision = should_abstain_from_answer(question, retrieval_response.matched_chunks, None)
        candidate_chunks = list(abstain_decision.filtered_chunks or retrieval_response.matched_chunks)
        if abstain_decision.should_abstain or not candidate_chunks:
            return self._build_refusal_result(
                question=question,
                router_result=router_result,
                answer=abstain_decision.user_message or "未找到足够相关的可访问内容来支持可靠回答。",
                refusal_reason=abstain_decision.reason or "insufficient_relevant_evidence",
                conversation_memory=conversation_memory,
                tool_name=tool_metadata.tool_name,
                tool_input=tool_metadata.tool_input,
                tool_output_summary={
                    **tool_metadata.tool_output_summary,
                    "post_filter_candidates": len(candidate_chunks),
                },
                retrieval_response=retrieval_response,
                intent="topic_qa",
            )
        return self._generate_grounded_qa(
            question=question,
            existing_messages=existing_messages,
            conversation_memory=conversation_memory,
            router_result=router_result,
            retrieval_response=retrieval_response,
            candidate_chunks=candidate_chunks,
            tool_metadata=tool_metadata,
            target_document=None,
            allow_low_score=False,
        )

    def _run_version_compare(
        self,
        actor,
        question: str,
        conversation_memory: ConversationMemory,
        router_result: RouterDecisionResult,
    ) -> CopilotRunResult:
        decision = router_result.decision
        tool_result = self.tools.compare_document_versions(
            actor,
            decision.target_document_title or decision.target_document_id or decision.requested_document_name,
            decision.from_version_ref,
            decision.to_version_ref,
        )
        tool_metadata = CopilotExecutionMetadata(
            tool_name=TOOL_COMPARE_VERSIONS,
            tool_input={
                "target_document": decision.target_document_title or decision.requested_document_name,
                "from_version_ref": decision.from_version_ref,
                "to_version_ref": decision.to_version_ref,
            },
            tool_output_summary={
                "document_title": tool_result.document_title,
                "refusal_reason": tool_result.refusal_reason,
                "has_summary": tool_result.summary is not None,
            },
        )
        if tool_result.refusal_reason or tool_result.summary is None:
            requested_name = decision.target_document_title or decision.requested_document_name or "该文档"
            answer = "当前无法完成版本差异比较。"
            if tool_result.refusal_reason == "target_document_not_accessible_or_not_found":
                answer = "当前可访问范围内未找到相关文档内容，因此暂时无法进行版本对比。"
            elif tool_result.refusal_reason == "insufficient_versions_for_compare":
                answer = f"“{requested_name}”当前可访问范围内不足两个版本，暂时无法比较差异。"
            return self._build_version_compare_result(
                question=question,
                router_result=router_result,
                conversation_memory=conversation_memory,
                tool_metadata=tool_metadata,
                answer=answer,
                confidence="insufficient",
                target_document=requested_name,
                refusal_reason=tool_result.refusal_reason or "unable_to_compare_versions",
                summary=None,
            )
        summary = tool_result.summary
        answer = f"“{tool_result.document_title}”版本差异摘要：{summary.summary}"
        return self._build_version_compare_result(
            question=question,
            router_result=router_result,
            conversation_memory=conversation_memory,
            tool_metadata=tool_metadata,
            answer=answer,
            confidence="high",
            target_document=tool_result.document_title,
            refusal_reason=None,
            summary=tool_result.summary,
        )

    def _run_workflow_generation(
        self,
        actor,
        question: str,
        session_id: UUID,
        existing_messages,
        conversation_memory: ConversationMemory,
        router_result: RouterDecisionResult,
    ) -> CopilotRunResult:
        artifact_type = router_result.decision.artifact_type
        if not has_usable_workflow_context(existing_messages):
            return self._build_workflow_result(
                question=question,
                router_result=router_result,
                conversation_memory=conversation_memory,
                tool_metadata=CopilotExecutionMetadata(
                    tool_name=_workflow_tool_name(artifact_type),
                    tool_input={"session_id": str(session_id)},
                    tool_output_summary={"artifact_type": artifact_type, "status": "skipped_due_to_insufficient_context"},
                ),
                answer="当前会话里缺少足够稳定的问答结果，暂时不直接生成待办、周报或 FAQ。请先完成一次有证据支撑的问答，再继续生成结构化结果。",
                confidence="insufficient",
                artifact_type=None,
                structured_payload=None,
                citations=[],
                refusal_reason="insufficient_session_context_for_workflow",
            )
        if artifact_type == "tasks":
            tool_result = self.tools.generate_tasks_from_session(actor, session_id)
            answer = f"已根据当前会话提取 {len(tool_result.structured_payload.get('items', []))} 条待办事项，可前往“派生结果”查看。"
        elif artifact_type == "weekly_report":
            tool_result = self.tools.generate_weekly_report_from_session(actor, session_id)
            report_title = ((tool_result.structured_payload.get("report") or {}).get("title") or "周报草稿")
            answer = f"已根据当前会话生成周报草稿《{report_title}》。"
        elif artifact_type == "faq":
            tool_result = self.tools.generate_faq_from_session(actor, session_id)
            answer = f"已根据当前会话生成 {len(tool_result.structured_payload.get('entries', []))} 条 FAQ 草稿。"
        else:
            return self._build_workflow_result(
                question=question,
                router_result=router_result,
                conversation_memory=conversation_memory,
                tool_metadata=CopilotExecutionMetadata(tool_name="none", tool_input={"session_id": str(session_id)}),
                answer="我还不能确定你希望生成哪类结果。请明确说明是待办、周报，还是 FAQ 草稿。",
                confidence="insufficient",
                artifact_type=None,
                structured_payload=None,
                citations=[],
                refusal_reason="unsupported_or_unclear_workflow_request",
            )
        tool_metadata = CopilotExecutionMetadata(
            tool_name=_workflow_tool_name(tool_result.artifact_type),
            tool_input={"session_id": str(session_id)},
            tool_output_summary={
                "artifact_type": tool_result.artifact_type,
                "citation_count": len(tool_result.citations),
            },
        )
        return self._build_workflow_result(
            question=question,
            router_result=router_result,
            conversation_memory=conversation_memory,
            tool_metadata=tool_metadata,
            answer=answer,
            confidence="high",
            artifact_type=tool_result.artifact_type,
            structured_payload=tool_result.structured_payload,
            citations=tool_result.citations,
            refusal_reason=tool_result.refusal_reason,
        )

    def _generate_grounded_qa(
        self,
        *,
        question: str,
        existing_messages,
        conversation_memory: ConversationMemory,
        router_result: RouterDecisionResult,
        retrieval_response: SearchResponse,
        candidate_chunks: list[SearchResultChunk],
        tool_metadata: CopilotExecutionMetadata,
        target_document: str | None,
        allow_low_score: bool,
    ) -> CopilotRunResult:
        generation = self.answer_generator.generate(
            question=question,
            retrieved_chunks=candidate_chunks,
            history_lines=conversation_memory.history_lines,
            conversation_context=conversation_memory.to_answer_context(),
            allow_low_score=allow_low_score,
        )
        generation = self._merge_answer_metrics(router_result, generation)
        validated_ids = validate_used_chunk_ids(generation.used_chunk_ids, {str(item.chunk_id) for item in candidate_chunks})
        selected_chunks = self._select_citation_chunks(candidate_chunks, validated_ids)

        if not generation.insufficient_evidence and not selected_chunks:
            generation = self._refusal_generation_result(
                answer="系统未能稳定校验回答所依赖的引用来源，因此这次不输出正式答案。",
                refusal_reason="invalid_or_missing_citations",
                router_result=router_result,
            )

        confidence = self._compute_confidence(selected_chunks, generation)
        if generation.insufficient_evidence:
            qa_result = QAAnswerResult(
                answer_type="refusal",
                answer=generation.answer,
                confidence="insufficient",
                citations=[],
                refusal_reason=str((generation.raw_payload or {}).get("reason") or generation.answer_basis or "insufficient_relevant_evidence"),
                target_document=target_document,
                intent=router_result.decision.intent,  # type: ignore[arg-type]
            )
        else:
            qa_result = QAAnswerResult(
                answer_type="grounded_answer",
                answer=generation.answer,
                confidence=confidence,  # type: ignore[arg-type]
                citations=[self._to_tool_citation(item) for item in selected_chunks],
                refusal_reason=None,
                target_document=target_document,
                intent=router_result.decision.intent,  # type: ignore[arg-type]
            )

        tool_metadata.tool_output_summary = {
            **tool_metadata.tool_output_summary,
            "candidate_chunks": len(candidate_chunks),
            "selected_citations": len(selected_chunks),
            "insufficient_evidence": generation.insufficient_evidence,
        }
        agent_steps = self._build_agent_steps(
            question=question,
            conversation_memory=conversation_memory,
            router_result=router_result,
            tool_metadata=tool_metadata,
            retrieval_response=retrieval_response,
            candidate_chunks=candidate_chunks,
            selected_chunks=selected_chunks,
            answer_result=generation,
            structured_result=qa_result,
        )
        return CopilotRunResult(
            router_result=router_result,
            tool_metadata=tool_metadata,
            answer_result=generation,
            retrieval_response=retrieval_response,
            candidate_chunks=candidate_chunks,
            selected_chunks=selected_chunks,
            confidence=confidence,
            structured_result=qa_result,
            agent_steps=agent_steps,
        )

    def _build_refusal_result(
        self,
        *,
        question: str,
        router_result: RouterDecisionResult,
        answer: str,
        refusal_reason: str,
        conversation_memory: ConversationMemory,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output_summary: dict[str, Any],
        retrieval_response: SearchResponse | None = None,
        target_document: str | None = None,
        intent: str | None = None,
    ) -> CopilotRunResult:
        generation = self._refusal_generation_result(answer=answer, refusal_reason=refusal_reason, router_result=router_result)
        if retrieval_response is None:
            retrieval_response = _empty_search_response(answer)
        qa_result = QAAnswerResult(
            answer_type="refusal",
            answer=answer,
            confidence="insufficient",
            citations=[],
            refusal_reason=refusal_reason,
            target_document=target_document,
            intent=(intent or router_result.decision.intent),  # type: ignore[arg-type]
        )
        tool_metadata = CopilotExecutionMetadata(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output_summary=tool_output_summary,
            retrieval_debug=retrieval_response.debug,
        )
        agent_steps = self._build_agent_steps(
            question=question,
            conversation_memory=conversation_memory,
            router_result=router_result,
            tool_metadata=tool_metadata,
            retrieval_response=retrieval_response,
            candidate_chunks=[],
            selected_chunks=[],
            answer_result=generation,
            structured_result=qa_result,
        )
        return CopilotRunResult(
            router_result=router_result,
            tool_metadata=tool_metadata,
            answer_result=generation,
            retrieval_response=retrieval_response,
            candidate_chunks=[],
            selected_chunks=[],
            confidence="insufficient",
            structured_result=qa_result,
            agent_steps=agent_steps,
        )

    def _build_version_compare_result(
        self,
        *,
        question: str,
        router_result: RouterDecisionResult,
        conversation_memory: ConversationMemory,
        tool_metadata: CopilotExecutionMetadata,
        answer: str,
        confidence: str,
        target_document: str | None,
        refusal_reason: str | None,
        summary: Any,
    ) -> CopilotRunResult:
        answer_result = self._tool_answer_generation_result(
            answer=answer,
            router_result=router_result,
            provider_name="copilot-version-compare",
            model_name=_compose_model_name(router_result.model_name, summary.model_name if summary else None),
            insufficient_evidence=refusal_reason is not None,
            raw_payload={"refusal_reason": refusal_reason},
        )
        structured_result = VersionCompareResult(
            answer_type="refusal" if refusal_reason else "version_compare_result",
            answer=answer,
            confidence=confidence,  # type: ignore[arg-type]
            target_document=target_document,
            from_version=f"v{summary.from_version_number}" if summary else None,
            to_version=f"v{summary.to_version_number}" if summary else None,
            summary=summary.summary if summary else None,
            additions=list(summary.additions) if summary else [],
            deletions=list(summary.deletions) if summary else [],
            modifications=list(summary.modifications) if summary else [],
            impact_hints=list(summary.impact_hints) if summary else [],
            refusal_reason=refusal_reason,
        )
        agent_steps = self._build_agent_steps(
            question=question,
            conversation_memory=conversation_memory,
            router_result=router_result,
            tool_metadata=tool_metadata,
            retrieval_response=_empty_search_response(answer),
            candidate_chunks=[],
            selected_chunks=[],
            answer_result=answer_result,
            structured_result=structured_result,
        )
        return CopilotRunResult(
            router_result=router_result,
            tool_metadata=tool_metadata,
            answer_result=answer_result,
            retrieval_response=_empty_search_response(answer),
            candidate_chunks=[],
            selected_chunks=[],
            confidence=confidence,
            structured_result=structured_result,
            agent_steps=agent_steps,
        )

    def _build_workflow_result(
        self,
        *,
        question: str,
        router_result: RouterDecisionResult,
        conversation_memory: ConversationMemory,
        tool_metadata: CopilotExecutionMetadata,
        answer: str,
        confidence: str,
        artifact_type: str | None,
        structured_payload: dict[str, Any] | None,
        citations,
        refusal_reason: str | None,
    ) -> CopilotRunResult:
        answer_result = self._tool_answer_generation_result(
            answer=answer,
            router_result=router_result,
            provider_name="copilot-workflow",
            model_name=router_result.model_name,
            insufficient_evidence=refusal_reason is not None,
            raw_payload={"refusal_reason": refusal_reason, "artifact_type": artifact_type},
        )
        structured_result = WorkflowGenerationResult(
            answer_type="refusal" if refusal_reason else "workflow_result",
            answer=answer,
            confidence=confidence,  # type: ignore[arg-type]
            artifact_type=artifact_type,  # type: ignore[arg-type]
            structured_payload=structured_payload,
            citations=list(citations),
            refusal_reason=refusal_reason,
        )
        agent_steps = self._build_agent_steps(
            question=question,
            conversation_memory=conversation_memory,
            router_result=router_result,
            tool_metadata=tool_metadata,
            retrieval_response=_empty_search_response(answer),
            candidate_chunks=[],
            selected_chunks=[],
            answer_result=answer_result,
            structured_result=structured_result,
        )
        return CopilotRunResult(
            router_result=router_result,
            tool_metadata=tool_metadata,
            answer_result=answer_result,
            retrieval_response=_empty_search_response(answer),
            candidate_chunks=[],
            selected_chunks=[],
            confidence=confidence,
            structured_result=structured_result,
            agent_steps=agent_steps,
        )

    def _build_agent_steps(
        self,
        *,
        question: str,
        conversation_memory: ConversationMemory,
        router_result: RouterDecisionResult,
        tool_metadata: CopilotExecutionMetadata,
        retrieval_response: SearchResponse,
        candidate_chunks: list[SearchResultChunk],
        selected_chunks: list[SearchResultChunk],
        answer_result: AnswerGenerationResult,
        structured_result: QAAnswerResult | VersionCompareResult | WorkflowGenerationResult,
    ) -> list[AgentStep]:
        intent = router_result.decision.intent
        target_document = self._resolve_target_document(router_result.decision, structured_result)
        is_refusal = bool(answer_result.insufficient_evidence)
        tool_name = tool_metadata.tool_name if tool_metadata.tool_name != "none" else None

        query_output = f"识别为 {intent}"
        if target_document:
            query_output += f"，目标文档为“{target_document}”"
        if conversation_memory.previous_target_document and _looks_like_followup_question(question):
            query_output += "，并复用了上一轮对话上下文"

        tool_selection_output = "未选择工具，直接返回拒答。"
        if tool_name:
            tool_selection_output = f"选择工具 {tool_name}。"

        tool_execution_status = "skipped" if tool_name is None else "completed"
        if str(tool_metadata.tool_output_summary.get("status") or "").startswith("skipped"):
            tool_execution_status = "skipped"
        tool_execution_output = "未执行工具。"
        if tool_name:
            tool_execution_output = self._summarize_tool_execution(tool_metadata)

        evidence_status = "completed"
        evidence_output = self._summarize_evidence_review(intent, candidate_chunks, selected_chunks, answer_result, structured_result)
        if intent == "unsupported_or_unclear":
            evidence_status = "skipped"
            evidence_output = "当前请求未进入检索或工具结果校验。"
        elif intent == "workflow_generation" and is_refusal:
            evidence_status = "refused"

        answer_status = "refused" if is_refusal else "completed"
        answer_output = self._summarize_answer_generation(structured_result, answer_result)

        return [
            AgentStep(
                name="query_analysis",
                input_summary=_truncate(question, 140),
                output_summary=query_output,
                status="completed",
                tool_name=None,
                metadata={
                    "previous_target_document": conversation_memory.previous_target_document,
                    "previous_tool_name": conversation_memory.previous_tool_name,
                    "previous_artifact_type": conversation_memory.previous_artifact_type,
                    "previous_intent": conversation_memory.previous_intent,
                    "older_message_count": conversation_memory.older_message_count,
                },
            ),
            AgentStep(
                name="tool_selection",
                input_summary=f"intent={intent}",
                output_summary=tool_selection_output,
                status="completed",
                tool_name=tool_name,
                metadata={"reasoning_brief": router_result.decision.reasoning_brief},
            ),
            AgentStep(
                name="tool_execution",
                input_summary=self._summarize_tool_input(tool_metadata),
                output_summary=tool_execution_output,
                status=tool_execution_status,  # type: ignore[arg-type]
                tool_name=tool_name,
                metadata=tool_metadata.tool_output_summary,
            ),
            AgentStep(
                name="evidence_review",
                input_summary=f"candidate_chunks={len(candidate_chunks)}",
                output_summary=evidence_output,
                status=evidence_status,  # type: ignore[arg-type]
                tool_name=tool_name,
                metadata={
                    "selected_citations": len(selected_chunks),
                    "retrieval_debug": retrieval_response.debug.model_dump(),
                },
            ),
            AgentStep(
                name="answer_generation",
                input_summary=f"provider={answer_result.provider_name}",
                output_summary=answer_output,
                status=answer_status,  # type: ignore[arg-type]
                tool_name=tool_name,
                metadata={
                    "confidence": getattr(structured_result, "confidence", None),
                    "answer_basis": answer_result.answer_basis,
                    "artifact_type": getattr(structured_result, "artifact_type", None),
                    "refusal_reason": getattr(structured_result, "refusal_reason", None),
                },
            ),
        ]

    @staticmethod
    def _resolve_target_document(
        decision: RouterDecision,
        structured_result: QAAnswerResult | VersionCompareResult | WorkflowGenerationResult,
    ) -> str | None:
        return getattr(structured_result, "target_document", None) or decision.target_document_title or decision.requested_document_name

    @staticmethod
    def _summarize_tool_input(tool_metadata: CopilotExecutionMetadata) -> str:
        if not tool_metadata.tool_input:
            return "无额外输入参数。"
        parts = []
        for key, value in tool_metadata.tool_input.items():
            if value is None:
                continue
            parts.append(f"{key}={value}")
        return "，".join(parts) if parts else "无额外输入参数。"

    @staticmethod
    def _summarize_tool_execution(tool_metadata: CopilotExecutionMetadata) -> str:
        summary = tool_metadata.tool_output_summary
        parts: list[str] = []
        if "matched_chunks" in summary:
            parts.append(f"命中 {summary['matched_chunks']} 个分块")
        if "candidate_chunks" in summary:
            parts.append(f"候选 {summary['candidate_chunks']} 个")
        if "selected_citations" in summary:
            parts.append(f"选中 {summary['selected_citations']} 条引用")
        if "document_title" in summary and summary.get("document_title"):
            parts.append(f"目标文档={summary['document_title']}")
        if "artifact_type" in summary and summary.get("artifact_type"):
            parts.append(f"产物类型={summary['artifact_type']}")
        if "citation_count" in summary:
            parts.append(f"引用来源 {summary['citation_count']} 条")
        if "refusal_reason" in summary and summary.get("refusal_reason"):
            parts.append(f"返回拒答：{summary['refusal_reason']}")
        return "；".join(parts) if parts else "工具已执行。"

    @staticmethod
    def _summarize_evidence_review(
        intent: str,
        candidate_chunks: list[SearchResultChunk],
        selected_chunks: list[SearchResultChunk],
        answer_result: AnswerGenerationResult,
        structured_result: QAAnswerResult | VersionCompareResult | WorkflowGenerationResult,
    ) -> str:
        if intent in {"document_qa", "topic_qa"}:
            if answer_result.insufficient_evidence:
                return f"候选分块 {len(candidate_chunks)} 个，但证据不足，未确认正式引用。"
            return f"候选分块 {len(candidate_chunks)} 个，最终选中 {len(selected_chunks)} 条引用。"
        if intent == "version_compare":
            return "基于版本 diff 与摘要结果完成证据校验。"
        if intent == "workflow_generation":
            if getattr(structured_result, "refusal_reason", None):
                return "会话上下文不足，未继续生成结构化结果。"
            return "基于当前会话内容完成结构化结果生成前检查。"
        return "未进入证据校验阶段。"

    @staticmethod
    def _summarize_answer_generation(
        structured_result: QAAnswerResult | VersionCompareResult | WorkflowGenerationResult,
        answer_result: AnswerGenerationResult,
    ) -> str:
        answer_type = getattr(structured_result, "answer_type", "unknown")
        refusal_reason = getattr(structured_result, "refusal_reason", None)
        if refusal_reason:
            return f"生成 {answer_type}，原因：{refusal_reason}"
        return f"生成 {answer_type}，confidence={getattr(structured_result, 'confidence', None) or 'n/a'}"

    @staticmethod
    def _refusal_generation_result(*, answer: str, refusal_reason: str, router_result: RouterDecisionResult) -> AnswerGenerationResult:
        return AnswerGenerationResult(
            answer=answer,
            insufficient_evidence=True,
            evidence_conflict=False,
            used_chunk_ids=[],
            answer_basis=refusal_reason,
            provider_name=router_result.provider_name,
            model_name=router_result.model_name,
            prompt_tokens=router_result.prompt_tokens,
            completion_tokens=router_result.completion_tokens,
            latency_ms=router_result.latency_ms,
            raw_payload={"reason": refusal_reason, "router_payload": router_result.raw_payload},
        )

    @staticmethod
    def _tool_answer_generation_result(
        *,
        answer: str,
        router_result: RouterDecisionResult,
        provider_name: str,
        model_name: str | None,
        insufficient_evidence: bool,
        raw_payload: dict[str, Any] | None = None,
    ) -> AnswerGenerationResult:
        return AnswerGenerationResult(
            answer=answer,
            insufficient_evidence=insufficient_evidence,
            evidence_conflict=False,
            used_chunk_ids=[],
            answer_basis="tool_result",
            provider_name=provider_name,
            model_name=model_name,
            prompt_tokens=router_result.prompt_tokens,
            completion_tokens=max(len(answer) // 4, 1),
            latency_ms=router_result.latency_ms,
            raw_payload=raw_payload,
        )

    @staticmethod
    def _merge_answer_metrics(router_result: RouterDecisionResult, answer_result: AnswerGenerationResult) -> AnswerGenerationResult:
        return AnswerGenerationResult(
            answer=answer_result.answer,
            insufficient_evidence=answer_result.insufficient_evidence,
            evidence_conflict=answer_result.evidence_conflict,
            used_chunk_ids=answer_result.used_chunk_ids,
            answer_basis=answer_result.answer_basis,
            provider_name=answer_result.provider_name,
            model_name=_compose_model_name(router_result.model_name, answer_result.model_name),
            prompt_tokens=(router_result.prompt_tokens or 0) + (answer_result.prompt_tokens or 0),
            completion_tokens=(router_result.completion_tokens or 0) + (answer_result.completion_tokens or 0),
            latency_ms=(router_result.latency_ms or 0) + (answer_result.latency_ms or 0),
            raw_payload={
                "router_payload": router_result.raw_payload,
                "answer_payload": answer_result.raw_payload,
            },
        )

    @staticmethod
    def _select_citation_chunks(matched_chunks: list[SearchResultChunk], validated_chunk_ids: list[UUID]) -> list[SearchResultChunk]:
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
            return "medium" if top_score >= 0.6 else "low"
        if len(selected_chunks) >= 2 and top_score >= 0.7 and average_score >= 0.55 and dominant_share >= 0.5:
            return "high"
        if top_score >= 0.4:
            return "medium"
        return "low"

    @staticmethod
    def _to_tool_citation(chunk: SearchResultChunk) -> ToolCitation:
        return ToolCitation(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_title=chunk.document_title,
            document_version_id=chunk.document_version_id,
            version_number=chunk.version_number,
            chunk_index=chunk.chunk_index,
            page_number_start=chunk.page_number_start,
            page_number_end=chunk.page_number_end,
            paragraph_start=chunk.paragraph_start,
            paragraph_end=chunk.paragraph_end,
            preview=chunk.preview,
            fused_score=chunk.score.fused,
        )


def _workflow_tool_name(artifact_type: str | None) -> str:
    if artifact_type == "tasks":
        return TOOL_EXTRACT_TODOS
    if artifact_type == "weekly_report":
        return TOOL_GENERATE_WEEKLY_REPORT
    if artifact_type == "faq":
        return TOOL_GENERATE_FAQ
    return "none"


def _looks_like_followup_question(question: str) -> bool:
    lowered = question.casefold()
    markers = ("刚才", "上一个", "上一轮", "继续", "这个文档", "这份文档", "那份文档", "这个手册", "这份手册")
    return any(marker in lowered for marker in markers)


def _truncate(value: str, limit: int) -> str:
    compact = " ".join(value.strip().split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."



def _compose_model_name(primary: str | None, secondary: str | None) -> str | None:
    parts = [item for item in [primary, secondary] if item]
    if not parts:
        return None
    unique_parts: list[str] = []
    for item in parts:
        if item not in unique_parts:
            unique_parts.append(item)
    return " + ".join(unique_parts)



def _empty_search_response(query: str) -> SearchResponse:
    return SearchResponse(
        query=query,
        top_k=0,
        matched_chunks=[],
        debug=SearchDebugInfo(
            accessible_document_count=0,
            lexical_candidate_count=0,
            vector_candidate_count=0,
            fusion_strategy="min-max weighted sum",
        ),
    )


