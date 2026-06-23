from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any
from uuid import UUID

from app.schemas.llm import (
    AgentStep,
    AgentRunTrace,
    CopilotExecutionMetadata,
    QAAnswerResult,
    RouterDecision,
    RouterDecisionResult,
    ToolAction,
    ToolCitation,
    ToolObservation,
    ToolPlan,
    VersionCompareResult,
    WorkflowGenerationResult,
)
from app.services.chat.memory import ConversationMemory, build_conversation_memory, has_usable_workflow_context
from app.schemas.search import SearchDebugInfo, SearchResponse, SearchResultChunk
from app.services.chat.generation import AnswerGenerationResult, AnswerGeneratorFactory, DeterministicAnswerGenerator
from app.services.chat.prompts import validate_used_chunk_ids
from app.services.chat.reliability import should_abstain_from_answer
from app.services.llm.agent_runner import AgentRunner
from app.services.llm.planner import ActionPlannerFactory
from app.services.llm.router import LLMRouterService
from app.services.llm.tool_executor import ToolExecutor
from app.services.llm.tool_registry import DEFAULT_TOOL_REGISTRY
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
    agent_run_trace: AgentRunTrace | None = None


class CopilotOrchestrator:
    def __init__(self, session) -> None:
        self.router = LLMRouterService()
        self.tools = CopilotToolService(session)
        self.tool_registry = DEFAULT_TOOL_REGISTRY
        self.tool_executor = ToolExecutor(self.tools, tool_registry=self.tool_registry)
        self.answer_generator = AnswerGeneratorFactory.create()
        self.agent_runner_cls = AgentRunner
        self.action_planner = ActionPlannerFactory.create()

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

        if self._should_use_agent_runner(question, memory, router_result) and not (
            decision.intent == "workflow_generation" and session_id.int == 0
        ):
            return self._run_agent_workflow(
                actor=actor,
                question=question,
                session_id=session_id,
                top_k=top_k,
                existing_messages=existing_messages,
                conversation_memory=memory,
                router_result=router_result,
            )

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
        generation_candidates = self._expand_clause_generation_candidates(
            question=question,
            filtered_chunks=candidate_chunks,
            retrieval_chunks=retrieval_response.matched_chunks,
        )
        return self._generate_grounded_qa(
            question=question,
            existing_messages=existing_messages,
            conversation_memory=conversation_memory,
            router_result=router_result,
            retrieval_response=retrieval_response,
            candidate_chunks=generation_candidates,
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

    def _should_use_agent_runner(
        self,
        question: str,
        conversation_memory: ConversationMemory,
        router_result: RouterDecisionResult,
    ) -> bool:
        decision = router_result.decision
        if decision.intent == "unsupported_or_unclear":
            return False
        if decision.artifact_type is not None:
            return True
        return decision.intent in {"version_compare", "workflow_generation"}

    def _run_agent_workflow(
        self,
        *,
        actor,
        question: str,
        session_id: UUID,
        top_k: int,
        existing_messages,
        conversation_memory: ConversationMemory,
        router_result: RouterDecisionResult,
    ) -> CopilotRunResult:
        runner = self.agent_runner_cls(
            tool_executor=self.tool_executor,
            tool_registry=self.tool_registry,
            planner=self.action_planner,
            max_steps=3,
        )
        runner_result = runner.run(
            actor=actor,
            user_query=question,
            session_id=None if session_id.int == 0 else session_id,
            top_k=top_k,
            chat_context=conversation_memory,
            router_result=router_result,
            existing_messages=list(existing_messages),
        )

        execution_results = runner_result.execution_results
        latest_execution = execution_results[-1] if execution_results else None
        latest_search = next((item for item in reversed(execution_results) if item.retrieval_response is not None), None)
        latest_compare = next((item for item in reversed(execution_results) if item.version_compare_result is not None), None)
        latest_workflow = next((item for item in reversed(execution_results) if item.workflow_result is not None), None)

        if runner_result.final_action.action_type == "ask_clarification":
            clarification_answer = runner_result.final_action.reason or "我还需要你补充更明确的目标后，才能继续执行。"
            if router_result.decision.intent == "version_compare":
                return self._build_version_compare_result(
                    question=question,
                    router_result=router_result,
                    conversation_memory=conversation_memory,
                    tool_metadata=latest_execution.tool_metadata if latest_execution else CopilotExecutionMetadata(tool_name="none"),
                    answer=clarification_answer,
                    confidence="insufficient",
                    target_document=router_result.decision.target_document_title or router_result.decision.requested_document_name,
                    refusal_reason="clarification_required",
                    summary=None,
                    agent_run_trace=runner_result.run_trace,
                )
            if router_result.decision.intent == "workflow_generation" or router_result.decision.artifact_type is not None:
                return self._build_workflow_result(
                    question=question,
                    router_result=router_result,
                    conversation_memory=conversation_memory,
                    tool_metadata=latest_execution.tool_metadata if latest_execution else CopilotExecutionMetadata(tool_name="none"),
                    answer=clarification_answer,
                    confidence="insufficient",
                    artifact_type=None,
                    structured_payload=None,
                    citations=[],
                    refusal_reason="clarification_required",
                    agent_run_trace=runner_result.run_trace,
                )
            return self._build_refusal_result(
                question=question,
                router_result=router_result,
                answer=clarification_answer,
                refusal_reason="clarification_required",
                conversation_memory=conversation_memory,
                tool_name=latest_execution.tool_metadata.tool_name if latest_execution else "none",
                tool_input=latest_execution.tool_metadata.tool_input if latest_execution else {"question": question},
                tool_output_summary=latest_execution.tool_metadata.tool_output_summary if latest_execution else {"status": "not_executed"},
                retrieval_response=latest_execution.retrieval_response if latest_execution and latest_execution.retrieval_response else _empty_search_response(question),
                intent=router_result.decision.intent,
                agent_run_trace=runner_result.run_trace,
            )

        if latest_workflow and latest_workflow.workflow_result:
            workflow_result = latest_workflow.workflow_result
            refusal_reason = workflow_result.refusal_reason
            if latest_execution and latest_execution.refusal_reason == "insufficient_context":
                refusal_reason = "insufficient_session_context_for_workflow"
            if refusal_reason:
                return self._build_workflow_result(
                    question=question,
                    router_result=router_result,
                    conversation_memory=conversation_memory,
                    tool_metadata=latest_workflow.tool_metadata,
                    answer=latest_execution.observation.output_summary if latest_execution else "当前上下文不足，暂不生成结构化结果。",
                    confidence="insufficient",
                    artifact_type=None,
                    structured_payload=None,
                    citations=[],
                    refusal_reason=refusal_reason,
                    agent_run_trace=runner_result.run_trace,
                )

            answer = latest_workflow.answer_hint or _workflow_success_answer(workflow_result)
            return self._build_workflow_result(
                question=question,
                router_result=router_result,
                conversation_memory=conversation_memory,
                tool_metadata=latest_workflow.tool_metadata,
                answer=answer,
                confidence="high",
                artifact_type=workflow_result.artifact_type,
                structured_payload=workflow_result.structured_payload,
                citations=workflow_result.citations,
                refusal_reason=None,
                agent_run_trace=runner_result.run_trace,
            )

        if latest_compare and latest_compare.version_compare_result:
            compare_result = latest_compare.version_compare_result
            summary = compare_result.summary
            requested_name = (
                compare_result.document_title
                or router_result.decision.target_document_title
                or router_result.decision.requested_document_name
                or "该文档"
            )
            answer = "当前无法完成版本差异比较。"
            if compare_result.refusal_reason == "target_document_not_accessible_or_not_found":
                answer = "当前可访问范围内未找到相关文档内容，因此暂时无法进行版本对比。"
            elif compare_result.refusal_reason == "insufficient_versions_for_compare":
                answer = f"“{requested_name}”当前可访问范围内不足两个版本，暂时无法比较差异。"
            elif summary is not None:
                answer = f"“{compare_result.document_title}”版本差异摘要：{summary.summary}"
            return self._build_version_compare_result(
                question=question,
                router_result=router_result,
                conversation_memory=conversation_memory,
                tool_metadata=latest_compare.tool_metadata,
                answer=answer,
                confidence="insufficient" if compare_result.refusal_reason else "high",
                target_document=requested_name if compare_result.refusal_reason else compare_result.document_title,
                refusal_reason=compare_result.refusal_reason,
                summary=summary,
                agent_run_trace=runner_result.run_trace,
            )

        if latest_search and latest_search.retrieval_response is not None:
            if latest_search.refusal_reason == "target_document_not_accessible_or_not_found":
                return self._build_refusal_result(
                    question=question,
                    router_result=router_result,
                    answer="当前可访问范围内未找到相关文档内容。该文档可能不存在，或你当前没有访问权限。",
                    refusal_reason="target_document_not_accessible_or_not_found",
                    conversation_memory=conversation_memory,
                    tool_name=latest_search.tool_metadata.tool_name,
                    tool_input=latest_search.tool_metadata.tool_input,
                    tool_output_summary=latest_search.tool_metadata.tool_output_summary,
                    retrieval_response=latest_search.retrieval_response,
                    target_document=router_result.decision.requested_document_name,
                    intent=router_result.decision.intent,
                    agent_run_trace=runner_result.run_trace,
                )
            if latest_search.refusal_reason == "no_relevant_evidence_in_target_document" or not latest_search.candidate_chunks:
                target_name = latest_search.target_document or router_result.decision.requested_document_name or "该文档"
                return self._build_refusal_result(
                    question=question,
                    router_result=router_result,
                    answer=f"在“{target_name}”当前可访问的内容中，未找到足够相关的证据来回答这个问题。",
                    refusal_reason=latest_search.refusal_reason or "insufficient_relevant_evidence",
                    conversation_memory=conversation_memory,
                    tool_name=latest_search.tool_metadata.tool_name,
                    tool_input=latest_search.tool_metadata.tool_input,
                    tool_output_summary=latest_search.tool_metadata.tool_output_summary,
                    retrieval_response=latest_search.retrieval_response,
                    target_document=target_name,
                    intent=router_result.decision.intent,
                    agent_run_trace=runner_result.run_trace,
                )
            candidate_chunks = list(latest_search.candidate_chunks)
            if router_result.decision.intent == "topic_qa":
                abstain_decision = should_abstain_from_answer(question, latest_search.candidate_chunks, None)
                candidate_chunks = list(abstain_decision.filtered_chunks or latest_search.candidate_chunks)
                if abstain_decision.should_abstain or not candidate_chunks:
                    return self._build_refusal_result(
                        question=question,
                        router_result=router_result,
                        answer=abstain_decision.user_message or "未找到足够相关的可访问内容来支持可靠回答。",
                        refusal_reason=abstain_decision.reason or "insufficient_relevant_evidence",
                        conversation_memory=conversation_memory,
                        tool_name=latest_search.tool_metadata.tool_name,
                        tool_input=latest_search.tool_metadata.tool_input,
                        tool_output_summary={
                            **latest_search.tool_metadata.tool_output_summary,
                            "post_filter_candidates": len(candidate_chunks),
                        },
                        retrieval_response=latest_search.retrieval_response,
                        intent="topic_qa",
                        agent_run_trace=runner_result.run_trace,
                    )
                latest_search.tool_metadata.tool_output_summary = {
                    **latest_search.tool_metadata.tool_output_summary,
                    "post_filter_candidates": len(candidate_chunks),
                }
            return self._generate_grounded_qa(
                question=question,
                existing_messages=existing_messages,
                conversation_memory=conversation_memory,
                router_result=router_result,
                retrieval_response=latest_search.retrieval_response,
                candidate_chunks=candidate_chunks,
                tool_metadata=latest_search.tool_metadata,
                target_document=latest_search.target_document,
                allow_low_score=router_result.decision.intent == "document_qa",
                agent_run_trace=runner_result.run_trace,
            )

        refusal_reason = _resolve_runner_refusal_reason(runner_result)
        if router_result.decision.intent == "workflow_generation" or router_result.decision.artifact_type is not None:
            if refusal_reason == "insufficient_context":
                refusal_reason = "insufficient_session_context_for_workflow"
            tool_name = latest_execution.tool_metadata.tool_name if latest_execution else _workflow_tool_name(router_result.decision.artifact_type)
            tool_input = latest_execution.tool_metadata.tool_input if latest_execution else {"session_id": str(session_id)}
            tool_output_summary = latest_execution.tool_metadata.tool_output_summary if latest_execution else {"status": "not_executed"}
            return self._build_workflow_result(
                question=question,
                router_result=router_result,
                conversation_memory=conversation_memory,
                tool_metadata=CopilotExecutionMetadata(
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_output_summary=tool_output_summary,
                ),
                answer=latest_execution.observation.output_summary if latest_execution else "当前请求未形成可用 observation，暂不生成结构化结果。",
                confidence="insufficient",
                artifact_type=None,
                structured_payload=None,
                citations=[],
                refusal_reason=refusal_reason,
                agent_run_trace=runner_result.run_trace,
            )

        return self._build_refusal_result(
            question=question,
            router_result=router_result,
            answer="当前请求未形成足够稳定的 observation，暂时无法生成可靠回答。",
            refusal_reason=refusal_reason,
            conversation_memory=conversation_memory,
            tool_name=latest_execution.tool_metadata.tool_name if latest_execution else "none",
            tool_input=latest_execution.tool_metadata.tool_input if latest_execution else {"question": question},
            tool_output_summary=latest_execution.tool_metadata.tool_output_summary if latest_execution else {"status": "not_executed"},
            retrieval_response=latest_execution.retrieval_response if latest_execution and latest_execution.retrieval_response else _empty_search_response(question),
            intent=router_result.decision.intent,
            agent_run_trace=runner_result.run_trace,
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
        agent_run_trace: AgentRunTrace | None = None,
    ) -> CopilotRunResult:
        generation_chunks = self._focus_generation_chunks(question, candidate_chunks)
        structured_fastpath = self._try_structured_table_fastpath(
            question=question,
            candidate_chunks=generation_chunks,
            history_lines=conversation_memory.history_lines,
            conversation_context=conversation_memory.to_answer_context(),
            allow_low_score=allow_low_score,
        )
        if structured_fastpath is not None:
            generation = structured_fastpath
        elif self._should_prefer_fast_grounded_summary(
            question=question,
            router_result=router_result,
            candidate_chunks=generation_chunks,
            conversation_memory=conversation_memory,
        ):
            generation = DeterministicAnswerGenerator().generate(
                question=question,
                retrieved_chunks=generation_chunks,
                history_lines=conversation_memory.history_lines,
                conversation_context=conversation_memory.to_answer_context(),
                allow_low_score=allow_low_score,
            )
        else:
            generation = self.answer_generator.generate(
                question=question,
                retrieved_chunks=generation_chunks,
                history_lines=conversation_memory.history_lines,
                conversation_context=conversation_memory.to_answer_context(),
                allow_low_score=allow_low_score,
            )
        generation = self._merge_answer_metrics(router_result, generation)
        validated_ids = validate_used_chunk_ids(generation.used_chunk_ids, {str(item.chunk_id) for item in candidate_chunks})
        selected_chunks = self._select_citation_chunks(generation_chunks, validated_ids)
        if not selected_chunks:
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
        if agent_run_trace is None and tool_metadata.tool_name == TOOL_SEARCH_DOCS:
            agent_run_trace = self._build_direct_search_trace(
                router_result=router_result,
                conversation_memory=conversation_memory,
                tool_metadata=tool_metadata,
                final_status="refused" if generation.insufficient_evidence else "completed",
                final_reason=(
                    str((generation.raw_payload or {}).get("reason") or generation.answer_basis or "insufficient_relevant_evidence")
                    if generation.insufficient_evidence
                    else "direct grounded qa completed from search evidence"
                ),
            )
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
            agent_run_trace=agent_run_trace,
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
            agent_run_trace=agent_run_trace,
        )

    @staticmethod
    def _focus_generation_chunks(question: str, candidate_chunks: list[SearchResultChunk]) -> list[SearchResultChunk]:
        if not candidate_chunks:
            return []
        if _looks_like_structured_table_lookup(question):
            table_chunks = [_focus_table_rows_for_question(question, chunk) for chunk in candidate_chunks[:5] if "Table row:" in chunk.content]
            if table_chunks:
                return table_chunks[:3]
        ranked_chunks = sorted(
            candidate_chunks[:8],
            key=lambda chunk: (
                _generation_focus_score(question, chunk),
                chunk.score.rerank if chunk.score.rerank is not None else chunk.score.fused,
                chunk.score.fused,
            ),
            reverse=True,
        )
        selected: list[SearchResultChunk] = []
        per_document_count: dict[UUID, int] = {}
        evidence_hints = _extract_generation_evidence_hints(question)
        if len(evidence_hints) >= 2:
            seen_chunk_ids: set[UUID] = set()
            for hint in evidence_hints[:3]:
                matching_chunks = [
                    chunk
                    for chunk in ranked_chunks
                    if chunk.chunk_id not in seen_chunk_ids and _generation_chunk_matches_hint(chunk, hint)
                ]
                if not matching_chunks:
                    continue
                chunk = matching_chunks[0]
                selected.append(chunk)
                seen_chunk_ids.add(chunk.chunk_id)
                per_document_count[chunk.document_id] = per_document_count.get(chunk.document_id, 0) + 1
                if len(selected) >= 3:
                    break

        for chunk in ranked_chunks:
            if any(chunk.chunk_id == item.chunk_id for item in selected):
                continue
            document_count = per_document_count.get(chunk.document_id, 0)
            if document_count >= 2:
                continue
            selected.append(chunk)
            per_document_count[chunk.document_id] = document_count + 1
            if len(selected) >= 5:
                break
        return selected or candidate_chunks[:3]

    @staticmethod
    def _expand_clause_generation_candidates(
        *,
        question: str,
        filtered_chunks: list[SearchResultChunk],
        retrieval_chunks: list[SearchResultChunk],
    ) -> list[SearchResultChunk]:
        if not filtered_chunks or not retrieval_chunks:
            return filtered_chunks
        normalized_question = _normalize_for_focus(question)
        if not any(marker in normalized_question for marker in ("条", "规定", "合同", "承包", "个体工商户", "商标", "责任")):
            return filtered_chunks
        clause_chunks = [
            chunk
            for chunk in retrieval_chunks[:10]
            if "条款全称" in chunk.content or re.search(r"第[一二三四五六七八九十百千万零〇\d]+条", chunk.content)
        ]
        if not clause_chunks:
            return filtered_chunks
        seen: set[UUID] = {chunk.chunk_id for chunk in filtered_chunks}
        merged = list(filtered_chunks)
        for chunk in sorted(
            clause_chunks,
            key=lambda item: (
                _generation_focus_score(question, item),
                item.score.rerank if item.score.rerank is not None else item.score.fused,
                item.score.fused,
            ),
            reverse=True,
        ):
            if chunk.chunk_id in seen:
                continue
            merged.append(chunk)
            seen.add(chunk.chunk_id)
            if len(merged) >= 8:
                break
        return merged

    @staticmethod
    def _try_structured_table_fastpath(
        *,
        question: str,
        candidate_chunks: list[SearchResultChunk],
        history_lines: list[str],
        conversation_context: str | None,
        allow_low_score: bool,
    ) -> AnswerGenerationResult | None:
        if not candidate_chunks or not _looks_like_structured_table_lookup(question):
            return None
        top_document_id = candidate_chunks[0].document_id
        same_document_chunks = [chunk for chunk in candidate_chunks[:3] if chunk.document_id == top_document_id]
        if not same_document_chunks or not any("Table row:" in chunk.content for chunk in same_document_chunks):
            return None
        generation = DeterministicAnswerGenerator().generate(
            question=question,
            retrieved_chunks=same_document_chunks,
            history_lines=history_lines,
            conversation_context=conversation_context,
            allow_low_score=allow_low_score,
        )
        if generation.answer_basis in {"simple_table_lookup_answer", "structured_table_answer"} and not generation.insufficient_evidence:
            return generation
        return None

    @staticmethod
    def _should_prefer_fast_grounded_summary(
        *,
        question: str,
        router_result: RouterDecisionResult,
        candidate_chunks: list[SearchResultChunk],
        conversation_memory: ConversationMemory,
    ) -> bool:
        if router_result.decision.intent not in {"topic_qa", "document_qa"}:
            return False
        if conversation_memory.previous_tool_name or conversation_memory.previous_artifact_type:
            return False
        compact_question = question.strip()
        if len(compact_question) > 36:
            return False
        if any(marker in compact_question for marker in ("区别", "对比", "比较", "总结", "整理", "生成", "FAQ", "周报", "待办")):
            return False
        if not any(marker in compact_question for marker in ("什么", "怎么", "如何", "哪些", "谁", "多久", "多少", "安排", "要求")):
            return False
        if len(candidate_chunks) < 2:
            return False

        top_document_id = candidate_chunks[0].document_id
        top_two_chunks = candidate_chunks[:2]
        if any(chunk.document_id != top_document_id for chunk in top_two_chunks):
            return False

        top_chunk = top_two_chunks[0]
        lexical_signal = max(top_chunk.score.lexical_raw, top_chunk.score.lexical_normalized)
        fused_signal = max(top_chunk.score.fused, top_chunk.score.rerank or 0.0)
        return lexical_signal > 0 and fused_signal >= 0.4

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
        agent_run_trace: AgentRunTrace | None = None,
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
        if agent_run_trace is None and tool_name == TOOL_SEARCH_DOCS:
            agent_run_trace = self._build_direct_search_trace(
                router_result=router_result,
                conversation_memory=conversation_memory,
                tool_metadata=tool_metadata,
                final_status="refused",
                final_reason=refusal_reason,
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
            agent_run_trace=agent_run_trace,
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
            agent_run_trace=agent_run_trace,
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
        agent_run_trace: AgentRunTrace | None = None,
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
            agent_run_trace=agent_run_trace,
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
            agent_run_trace=agent_run_trace,
        )

    def _build_direct_search_trace(
        self,
        *,
        router_result: RouterDecisionResult,
        conversation_memory: ConversationMemory,
        tool_metadata: CopilotExecutionMetadata,
        final_status: str,
        final_reason: str | None,
    ) -> AgentRunTrace:
        final_action_type = "refuse" if final_status == "refused" else "final_answer"
        evidence_state = "insufficient" if final_status == "refused" else "sufficient"
        context_summary_parts: list[str] = []
        if conversation_memory.previous_target_document:
            context_summary_parts.append(f"previous_target_document={conversation_memory.previous_target_document}")
        if conversation_memory.previous_tool_name:
            context_summary_parts.append(f"previous_tool_name={conversation_memory.previous_tool_name}")
        if conversation_memory.previous_observation_summary:
            context_summary_parts.append(f"previous_observation={conversation_memory.previous_observation_summary}")

        return AgentRunTrace(
            tool_plan=ToolPlan(
                planner_name="DirectSearchPlan",
                available_tools=[TOOL_SEARCH_DOCS],
                max_steps=2,
                initial_intent=router_result.decision.intent,
                requested_artifact_type=router_result.decision.artifact_type,
                context_summary="；".join(context_summary_parts) or None,
            ),
            actions=[
                ToolAction(
                    step_index=1,
                    action_type="tool_call",
                    tool_name=TOOL_SEARCH_DOCS,
                    tool_args=dict(tool_metadata.tool_input),
                    reason="当前请求是单步问答，先直接检索可访问证据。",
                    evidence_state="none",
                    expected_next="根据检索结果直接生成回答或返回拒答。",
                ),
                ToolAction(
                    step_index=2,
                    action_type=final_action_type,  # type: ignore[arg-type]
                    reason="已有检索结果，可直接完成本轮问答。",
                    evidence_state=evidence_state,  # type: ignore[arg-type]
                    expected_next=None,
                    depends_on=[1],
                ),
            ],
            observations=[
                ToolObservation(
                    step_index=1,
                    tool_name=TOOL_SEARCH_DOCS,
                    status="completed",
                    output_summary=self._summarize_tool_execution(tool_metadata),
                    evidence_refs=[],
                    raw_output=dict(tool_metadata.tool_output_summary),
                )
            ],
            final_status=final_status,
            final_reason=final_reason,
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
        agent_run_trace: AgentRunTrace | None = None,
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
            agent_run_trace=agent_run_trace,
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
            agent_run_trace=agent_run_trace,
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
        agent_run_trace: AgentRunTrace | None = None,
    ) -> list[AgentStep]:
        intent = router_result.decision.intent
        target_document = self._resolve_target_document(router_result.decision, structured_result)
        is_refusal = bool(answer_result.insufficient_evidence)
        tool_name = tool_metadata.tool_name if tool_metadata.tool_name != "none" else None
        effective_candidate_count = self._effective_candidate_count(candidate_chunks, agent_run_trace)
        effective_selected_count = self._effective_selected_count(selected_chunks, structured_result)

        query_output = f"识别为 {intent}"
        if target_document:
            query_output += f"，目标文档为“{target_document}”"
        if conversation_memory.previous_target_document and _looks_like_followup_question(question):
            query_output += "，并复用了上一轮对话上下文"

        tool_selection_output = "未选择工具，直接返回拒答。"
        if agent_run_trace and agent_run_trace.actions:
            plan_parts = []
            for action in agent_run_trace.actions:
                action_label = action.tool_name if action.action_type == "tool_call" and action.tool_name else action.action_type
                plan_parts.append(f"{action_label}({action.evidence_state})")
            tool_selection_output = f"规划动作：{' -> '.join(plan_parts)}"
        elif tool_name:
            tool_selection_output = f"选择工具 {tool_name}。"

        tool_execution_status = "skipped" if tool_name is None else "completed"
        if str(tool_metadata.tool_output_summary.get("status") or "").startswith("skipped"):
            tool_execution_status = "skipped"
        if agent_run_trace and not agent_run_trace.observations and agent_run_trace.final_status in {"refused", "clarification_required"}:
            tool_execution_status = "skipped"
        if agent_run_trace and any(item.status == "failed" for item in agent_run_trace.observations):
            tool_execution_status = "refused"
        elif agent_run_trace and any(item.status == "insufficient_context" for item in agent_run_trace.observations):
            tool_execution_status = "skipped"
        tool_execution_output = "未执行工具。"
        if agent_run_trace and agent_run_trace.observations:
            tool_execution_output = "；".join(
                f"step {item.step_index}: {item.tool_name} -> {item.output_summary}" for item in agent_run_trace.observations
            )
        elif agent_run_trace and not agent_run_trace.observations and agent_run_trace.final_status in {"refused", "clarification_required"}:
            tool_execution_output = "planner 在执行前判定当前上下文不足，因此未继续调用工具。"
        elif tool_name:
            tool_execution_output = self._summarize_tool_execution(tool_metadata)

        evidence_status = "completed"
        evidence_output = self._summarize_evidence_review(
            intent,
            effective_candidate_count,
            effective_selected_count,
            answer_result,
            structured_result,
        )
        if intent == "unsupported_or_unclear":
            evidence_status = "skipped"
            evidence_output = "当前请求未进入检索或工具结果校验。"
        elif intent == "workflow_generation" and is_refusal:
            evidence_status = "refused"
        elif agent_run_trace and any(item.status == "insufficient_context" for item in agent_run_trace.observations):
            evidence_status = "refused"
            evidence_output = "planner 检测到上下文证据不足，未继续生成结构化结果。"

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
                    "agent_run_trace_available": agent_run_trace is not None,
                },
            ),
            AgentStep(
                name="tool_selection",
                input_summary=f"intent={intent}",
                output_summary=tool_selection_output,
                status="completed",
                tool_name=tool_name,
                metadata={
                    "reasoning_brief": router_result.decision.reasoning_brief,
                    "planned_actions": [item.model_dump(mode="json") for item in agent_run_trace.actions] if agent_run_trace else [],
                    "planner_name": agent_run_trace.tool_plan.planner_name if agent_run_trace else None,
                },
            ),
            AgentStep(
                name="tool_execution",
                input_summary=self._summarize_tool_input(tool_metadata),
                output_summary=tool_execution_output,
                status=tool_execution_status,  # type: ignore[arg-type]
                tool_name=tool_name,
                metadata={
                    **tool_metadata.tool_output_summary,
                    "observations": [item.model_dump(mode="json") for item in agent_run_trace.observations] if agent_run_trace else [],
                },
            ),
            AgentStep(
                name="evidence_review",
                input_summary=f"candidate_chunks={effective_candidate_count}",
                output_summary=evidence_output,
                status=evidence_status,  # type: ignore[arg-type]
                tool_name=tool_name,
                metadata={
                    "selected_citations": effective_selected_count,
                    "retrieval_debug": retrieval_response.debug.model_dump(),
                    "final_status": agent_run_trace.final_status if agent_run_trace else None,
                    "final_reason": agent_run_trace.final_reason if agent_run_trace else None,
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
        candidate_chunk_count: int,
        selected_citation_count: int,
        answer_result: AnswerGenerationResult,
        structured_result: QAAnswerResult | VersionCompareResult | WorkflowGenerationResult,
    ) -> str:
        if intent in {"document_qa", "topic_qa"}:
            if isinstance(structured_result, WorkflowGenerationResult):
                if getattr(structured_result, "refusal_reason", None):
                    return "当前请求未形成稳定结构化结果，未确认正式引用。"
                return f"检索候选分块 {candidate_chunk_count} 个，结构化结果复用了 {selected_citation_count} 条来源引用。"
            if answer_result.insufficient_evidence:
                return f"候选分块 {candidate_chunk_count} 个，但证据不足，未确认正式引用。"
            return f"候选分块 {candidate_chunk_count} 个，最终选中 {selected_citation_count} 条引用。"
        if intent == "version_compare":
            refusal_reason = getattr(structured_result, "refusal_reason", None)
            if refusal_reason == "insufficient_versions_for_compare":
                return "当前可访问范围内版本数量不足，未形成可用 diff 结果。"
            if refusal_reason == "unable_to_resolve_version_pair":
                return "待比较的版本范围不明确，未执行有效版本对比。"
            if refusal_reason == "target_document_not_accessible_or_not_found":
                return "目标文档不可访问，未执行有效版本对比。"
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
        unique_items: list[SearchResultChunk] = []
        seen: set[str] = set()
        for item in selected:
            key = CopilotOrchestrator._citation_identity_from_chunk(item)
            if key in seen:
                continue
            seen.add(key)
            unique_items.append(item)
            if len(unique_items) >= 3:
                break
        return unique_items

    @staticmethod
    def _citation_identity_from_chunk(chunk: SearchResultChunk) -> str:
        if chunk.chunk_id:
            return f"chunk:{chunk.chunk_id}"
        return "|".join(
            [
                str(chunk.document_id),
                str(chunk.document_version_id),
                str(chunk.chunk_index),
                str(chunk.page_number_start or ""),
                str(chunk.paragraph_start or ""),
                chunk.preview,
            ]
        )

    @staticmethod
    def _effective_candidate_count(candidate_chunks: list[SearchResultChunk], agent_run_trace: AgentRunTrace | None) -> int:
        if candidate_chunks:
            return len(candidate_chunks)
        if not agent_run_trace:
            return 0
        for observation in agent_run_trace.observations:
            raw_output = observation.raw_output or {}
            matched_chunks = raw_output.get("matched_chunks")
            if isinstance(matched_chunks, int):
                return matched_chunks
        return 0

    @staticmethod
    def _effective_selected_count(
        selected_chunks: list[SearchResultChunk],
        structured_result: QAAnswerResult | VersionCompareResult | WorkflowGenerationResult,
    ) -> int:
        if selected_chunks:
            return len(selected_chunks)
        citations = getattr(structured_result, "citations", None)
        if isinstance(citations, list):
            return len(citations)
        return 0

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
        if answer_result.answer_basis in {"simple_table_lookup_answer", "structured_table_answer"}:
            return "high"
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


def _workflow_success_answer(workflow_result) -> str:
    if workflow_result.artifact_type == "tasks":
        return f"已根据当前上下文提取 {len((workflow_result.structured_payload or {}).get('items', []))} 条待办事项。"
    if workflow_result.artifact_type == "weekly_report":
        report_title = ((workflow_result.structured_payload or {}).get("report") or {}).get("title") or "周报草稿"
        return f"已根据当前上下文生成周报草稿《{report_title}》。"
    if workflow_result.artifact_type == "faq":
        return f"已根据当前上下文生成 {len((workflow_result.structured_payload or {}).get('entries', []))} 条 FAQ 草稿。"
    return "已生成结构化结果。"


def _resolve_runner_refusal_reason(runner_result) -> str:
    if runner_result.run_trace.final_status == "max_steps_reached":
        return "max_steps_reached"
    if runner_result.run_trace.final_status == "clarification_required":
        return "clarification_required"
    if runner_result.run_trace.final_status == "refused" and not runner_result.execution_results:
        final_reason = runner_result.final_action.reason or runner_result.run_trace.final_reason or ""
        if any(marker in final_reason for marker in ("证据不足", "上下文", "有证据支撑的问答")):
            return "insufficient_session_context_for_workflow"
    if runner_result.execution_results:
        last_result = runner_result.execution_results[-1]
        if last_result.refusal_reason:
            return last_result.refusal_reason
        raw_output = last_result.observation.raw_output
        if isinstance(raw_output, dict):
            refusal_reason = raw_output.get("refusal_reason")
            if refusal_reason:
                return str(refusal_reason)
    return "agent_runner_no_final_result"


def _looks_like_followup_question(question: str) -> bool:
    lowered = question.casefold()
    markers = ("刚才", "上一个", "上一轮", "继续", "这个文档", "这份文档", "那份文档", "这个手册", "这份手册")
    return any(marker in lowered for marker in markers)


def _looks_like_structured_table_lookup(question: str) -> bool:
    lowered = question.casefold()
    markers = (
        "审批",
        "审批链路",
        "处理时限",
        "首次响应",
        "响应时间",
        "响应要求",
        "多久响应",
        "时限",
        "脱敏",
        "检查项",
        "是否必须",
        "负责人",
        "责任人",
        "完成时限",
        "复核周期",
        "退出要求",
        "有效期",
        "验收材料",
        "验收人",
        "保留",
        "l4",
        "高风险",
        "生产环境",
        "由谁",
        "哪些",
    )
    return any(marker in lowered for marker in markers)


def _generation_focus_score(question: str, chunk: SearchResultChunk) -> float:
    normalized_question = _normalize_for_focus(question)
    evidence_text = " ".join(
        part
        for part in [chunk.document_title, chunk.section_title or "", chunk.preview, chunk.content]
        if part
    )
    normalized_evidence = _normalize_for_focus(evidence_text)
    score = chunk.score.rerank if chunk.score.rerank is not None else chunk.score.fused
    score += chunk.score.lexical_normalized * 0.06

    for term in _focus_query_terms(normalized_question):
        if term in normalized_evidence:
            score += 0.08 if len(term) >= 4 else 0.04

    if "条款全称" in normalized_evidence and any(
        marker in normalized_question
        for marker in ("条", "规定", "债务", "合同", "承包", "个体工商户", "商标")
    ):
        score += 0.12

    domain_pairs = (
        ("债务", "债务"),
        ("偿还", "承担"),
        ("承担", "承担"),
        ("诉讼主体", "自然人从事工商业经营"),
        ("营业执照", "依法登记"),
        ("实际经营者", "个人经营"),
        ("实际经营者", "债务"),
        ("共同责任", "债务"),
        ("共同责任", "无法区分"),
        ("合伙", "合伙合同"),
        ("纠纷", "合伙合同"),
        ("无效", "民事法律行为无效"),
        ("合伙", "共享利益"),
        ("合伙", "共担风险"),
        ("纠纷", "虚假的意思表示"),
        ("纠纷", "恶意串通"),
        ("继承", "继承"),
        ("死亡", "死亡"),
        ("死亡", "承包收益"),
        ("死亡", "继续承包"),
        ("土地承包份额", "家庭成员"),
        ("土地承包份额", "平等享有"),
        ("收回承包地", "承包期"),
        ("收回承包地", "不得收回"),
        ("村集体", "不得收回"),
        ("本村以外", "本集体经济组织"),
        ("本村以外", "转让"),
        ("转让", "其他农户"),
        ("收回", "不得收回"),
        ("解除合同", "解除合同"),
        ("解除合同", "终止土地经营权流转合同"),
        ("解除合同", "当事人协商一致"),
        ("解除合同", "解除权人"),
        ("未按时缴纳费用", "严重违约"),
        ("个体工商户", "个体工商户"),
        ("土地承包", "土地承包"),
        ("发包方", "发包方"),
        ("承包方", "承包方"),
        ("民主议定", "村民会议"),
        ("三分之二", "三分之二"),
        ("鱼塘", "挖塘养鱼"),
        ("鱼塘", "基本农田保护区"),
        ("鱼塘", "农业用途"),
        ("基本农田", "基本农田"),
        ("正当使用", "正当使用"),
        ("正当使用", "通用名称"),
        ("先用权", "先于商标注册人使用"),
        ("先用权", "在原使用范围内继续使用"),
        ("抗辩", "在原使用范围内继续使用"),
        ("特许经营", "特许经营"),
        ("特许人", "特许人"),
        ("特许人", "企业以外的其他单位和个人不得作为特许人"),
    )
    for query_hint, evidence_hint in domain_pairs:
        if query_hint in normalized_question and evidence_hint in normalized_evidence:
            score += 0.16

    return score


def _focus_table_rows_for_question(question: str, chunk: SearchResultChunk) -> SearchResultChunk:
    rows = _extract_table_rows(chunk.content)
    if not rows:
        return chunk
    required_markers = _required_table_row_markers(question)
    if required_markers:
        strict_rows = [row for row in rows if any(marker in _normalize_for_focus(row) for marker in required_markers)]
        if strict_rows:
            rows = strict_rows

    priority_rows = _priority_table_rows_for_question(question, rows)
    if priority_rows:
        selected = priority_rows[:2]
    else:
        scored_rows = [
            (_table_row_focus_score(question, row), -index, row)
            for index, row in enumerate(rows)
        ]
        scored_rows.sort(reverse=True)
        best_score = scored_rows[0][0] if scored_rows else 0

        selected: list[str] = []
        for score, _, row in scored_rows:
            if score <= 0 and selected:
                continue
            if selected and best_score > 0 and score < best_score * 0.75:
                continue
            if row not in selected:
                selected.append(row)
            if len(selected) >= 2:
                break

        if not selected and scored_rows:
            selected = [scored_rows[0][2]]

    supporting_lines = _select_supporting_narrative_lines(question, chunk.content)
    prefix_parts = [part for part in [chunk.section_title, chunk.preview] if part and "Table row:" not in part]
    focused_content = "\n".join([*prefix_parts[:1], *supporting_lines, *selected]).strip()
    if not focused_content:
        return chunk
    return chunk.model_copy(update={"content": focused_content, "preview": focused_content[:500]})


def _priority_table_rows_for_question(question: str, rows: list[str]) -> list[str]:
    normalized_question = _normalize_for_focus(question)
    if not any(token in normalized_question for token in ("首次响应", "响应时间", "响应要求", "多久响应")):
        return []
    if not any(token in normalized_question for token in ("高优先级", "高优", "p1")):
        return []

    current_rows: list[str] = []
    history_rows: list[str] = []
    for row in rows:
        normalized_row = _normalize_for_focus(row)
        if "工单等级=p1" in normalized_row and "首次响应=" in normalized_row:
            current_rows.append(row)
        elif "问题类型=高优先级工单" in normalized_row and "历史响应时间=" in normalized_row:
            history_rows.append(row)

    return [*current_rows[:1], *history_rows[:1]]

def _extract_table_rows(content: str) -> list[str]:
    return [line.strip() for line in content.splitlines() if line.strip().startswith("Table row:")]


def _select_supporting_narrative_lines(question: str, content: str) -> list[str]:
    normalized_question = _normalize_for_focus(question)
    query_terms = _focus_query_terms(normalized_question)
    scored_lines: list[tuple[int, int, str]] = []
    for index, line in enumerate(content.splitlines()):
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("Table row:"):
            continue
        normalized_line = _normalize_for_focus(cleaned)
        score = 0
        for term in query_terms:
            if term in normalized_line:
                score += 3 if len(term) >= 4 else 1
        if any(token in normalized_question for token in ("先救火", "紧急", "最小权限", "最低权限")) and any(
            marker in normalized_line for marker in ("最小可用服务", "最低权限服务", "临时采购", "先行建立")
        ):
            score += 8
        if score > 0:
            scored_lines.append((score, -index, cleaned))

    scored_lines.sort(reverse=True)
    selected: list[str] = []
    for score, _, line in scored_lines:
        if line not in selected:
            selected.append(line)
        if len(selected) >= 2:
            break
    return selected


def _table_row_focus_score(question: str, row: str) -> int:
    normalized_question = _normalize_for_focus(question)
    normalized_row = _normalize_for_focus(row)
    score = 0

    query_terms = _focus_query_terms(normalized_question)
    for term in query_terms:
        if term in normalized_row:
            score += 4 if len(term) >= 4 else 2

    domain_pairs = (
        (("l4", "高风险"), ("准入等级=l4高风险", "l4高风险")),
        (("审批链路", "审批", "链路"), ("审批链路=",)),
        (("谁来批", "审批", "审批人"), ("必须审批人=", "审批人=", "审批链路=")),
        (("复核周期", "复核"), ("复核周期=",)),
        (("退出要求", "退出"), ("退出要求=",)),
        (("先救火", "紧急", "最小权限", "最低权限"), ("紧急场景=", "可先执行动作=", "最低权限服务", "最小可用服务")),
        (("补材料", "补齐材料"), ("事后补齐材料=", "最少材料=")),
        (("关账号", "关闭", "回收"), ("账号关闭时限=", "回收责任人=")),
        (("禁止发法", "禁止方式", "导出文件"), ("禁止方式=", "访问对象=数据导出文件")),
        (("生产环境",), ("访问对象=生产环境", "可访问生产环境")),
        (("允许方式", "允许"), ("允许方式=",)),
        (("有效期",), ("有效期=",)),
        (("回收责任人", "责任人"), ("回收责任人=",)),
        (("日志要求", "日志"), ("日志要求=",)),
        (("首次响应", "响应时间", "响应要求", "多久响应"), ("首次响应=",)),
        (("响应时间",), ("历史响应时间=",)),
        (("高优先级", "高优"), ("工单等级=p1", "问题类型=高优先级工单")),
        (("数据处理服务",), ("交付类型=数据处理服务",)),
        (("验收材料", "材料"), ("验收材料=",)),
        (("验收人",), ("验收人=",)),
        (("保留多久", "保留期限", "资料保留"), ("保留期限=",)),
    )
    for query_hints, row_hints in domain_pairs:
        if any(hint in normalized_question for hint in query_hints) and any(hint in normalized_row for hint in row_hints):
            score += 8

    if any(hint in normalized_question for hint in ("高优先级", "高优", "p1")):
        if "工单等级=p1" in normalized_row and "首次响应=" in normalized_row:
            score += 18
        elif "问题类型=高优先级工单" in normalized_row and "历史响应时间=" in normalized_row:
            score += 8

    return score


def _focus_query_terms(normalized_question: str) -> list[str]:
    stop_terms = {"什么", "哪些", "分别", "要求", "供应商", "文档", "规范", "需要"}
    terms: list[str] = []
    for term in re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9]{2,}", normalized_question):
        if term in stop_terms:
            continue
        if len(term) > 8:
            for size in (4, 5, 6):
                for index in range(0, max(len(term) - size + 1, 0)):
                    item = term[index : index + size]
                    if item not in stop_terms and item not in terms:
                        terms.append(item)
        elif term not in terms:
            terms.append(term)
    return terms


def _extract_generation_evidence_hints(question: str) -> list[str]:
    hints: list[str] = []
    for pattern in (r"“(?P<value>[^”]{2,260})”", r'"(?P<value>[^"]{2,260})"'):
        for match in re.finditer(pattern, question):
            cleaned = " ".join(match.group("value").split()).strip(" ：:；;，,、")
            if cleaned and cleaned not in hints:
                hints.append(cleaned)
    return hints


def _generation_chunk_matches_hint(chunk: SearchResultChunk, hint: str) -> bool:
    normalized_hint = _normalize_for_focus(hint)
    if len(normalized_hint) < 4:
        return False
    evidence_text = " ".join(
        part
        for part in [chunk.document_title, chunk.section_title or "", chunk.preview, chunk.content]
        if part
    )
    normalized_evidence = _normalize_for_focus(evidence_text)
    if normalized_hint in normalized_evidence:
        return True
    hint_terms = [term for term in _focus_query_terms(normalized_hint) if len(term) >= 3]
    if not hint_terms:
        return False
    matched = sum(1 for term in hint_terms if term in normalized_evidence)
    return matched >= max(2, len(hint_terms) - 1)


def _normalize_for_focus(value: str) -> str:
    return re.sub(r"\s+", "", value.casefold())


def _required_table_row_markers(question: str) -> tuple[str, ...]:
    normalized_question = _normalize_for_focus(question)
    if "l4" in normalized_question or "高风险" in normalized_question:
        return ("准入等级=l4", "l4高风险")
    if "数据处理服务" in normalized_question:
        return ("交付类型=数据处理服务",)
    if any(token in normalized_question for token in ("首次响应", "响应时间", "响应要求", "多久响应")):
        if any(token in normalized_question for token in ("高优先级", "高优", "p1")):
            return ("工单等级=p1", "问题类型=高优先级工单", "首次响应=")
        return ("首次响应=", "历史响应时间=")
    if any(token in normalized_question for token in ("先救火", "紧急", "最小权限", "最低权限", "补材料", "关账号", "禁止发法", "导出文件")):
        return ("紧急场景=", "可先执行动作=", "事后补齐材料=", "账号关闭时限=", "禁止方式=")
    if "生产环境" in normalized_question:
        return ("访问对象=生产环境", "可访问生产环境")
    if "客户手机号" in normalized_question or "手机号" in normalized_question:
        return ("客户手机号",)
    return ()


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


