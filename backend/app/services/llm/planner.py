from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from app.core.config import get_settings
from app.schemas.llm import PlannerDecision, RouterDecisionResult, ToolAction, ToolObservation
from app.services.chat.memory import ConversationMemory
from app.services.llm.openai_compatible import (
    create_openai_compatible_client,
    has_openai_compatible_credentials,
    uses_openai_compatible_provider,
)
from app.services.llm.tool_registry import ToolDefinition, ToolRegistry

PLANNER_SYSTEM_PROMPT = """You are the planning layer for an enterprise knowledge assistant.
Your job is to decide exactly one next action at a time.
You must NOT answer the user's business question directly.
Return valid JSON only.

Allowed action_type values:
- tool_call
- final_answer
- refuse
- ask_clarification

Allowed tool names:
- search_docs
- compare_versions
- extract_todos
- generate_weekly_report
- generate_faq

Rules:
- The router decision is only a coarse hint. Do not blindly follow it.
- Base every next step on the current user query, conversation context, previous actions, and previous observations.
- Prefer search_docs when the user first needs grounded document evidence.
- Prefer compare_versions when the user explicitly asks to compare versions or asks what changed between versions.
- Only use extract_todos, generate_weekly_report, or generate_faq after there is enough grounded evidence in previous observations or conversation context.
- Do not treat policy words such as 检查项 or 事项 as a request to call extract_todos unless the user explicitly asks to 提取待办, 整理成待办, 生成任务, 生成周报, or 生成 FAQ.
- If previous observations show missing evidence, inaccessible documents, unknown tools, or insufficient context, do not force another workflow tool. Use refuse or ask_clarification.
- If the user asks for a follow-up structured result based on previous context and there is already enough grounded evidence, it is acceptable to use extract_todos, generate_weekly_report, or generate_faq directly.
- Never invent tool names outside the whitelist.
- Keep reason short and concrete.
- evidence_state must be one of: none, partial, sufficient, insufficient.
- expected_next should briefly describe what may happen after this step, or null.
"""

WORKFLOW_TOOL_NAMES = {"extract_todos", "generate_weekly_report", "generate_faq"}


class ActionPlanner(Protocol):
    def plan_next_action(
        self,
        *,
        user_query: str,
        chat_context: ConversationMemory,
        router_result: RouterDecisionResult,
        available_tools: list[ToolDefinition],
        previous_actions: list[ToolAction],
        previous_observations: list[ToolObservation],
        remaining_steps: int,
    ) -> PlannerDecision: ...


class DeterministicFallbackPlanner:
    provider_name = "deterministic-planner"
    model_name = "planner-fallback-v1"

    def plan_next_action(
        self,
        *,
        user_query: str,
        chat_context: ConversationMemory,
        router_result: RouterDecisionResult,
        available_tools: list[ToolDefinition],
        previous_actions: list[ToolAction],
        previous_observations: list[ToolObservation],
        remaining_steps: int,
    ) -> PlannerDecision:
        allowed_tools = {tool.name for tool in available_tools}
        decision = router_result.decision
        artifact_type = decision.artifact_type
        last_observation = previous_observations[-1] if previous_observations else None
        successful_tools = [item.tool_name for item in previous_observations if item.status == "completed"]
        last_refusal_reason = None
        if last_observation and isinstance(last_observation.raw_output, dict):
            raw_reason = last_observation.raw_output.get("refusal_reason")
            if raw_reason:
                last_refusal_reason = str(raw_reason)

        terminal_decision = _terminal_decision_from_observation(
            last_observation=last_observation,
            last_refusal_reason=last_refusal_reason,
            artifact_type=artifact_type,
        )
        if terminal_decision is not None:
            return terminal_decision

        if last_observation:
            if last_observation.status in {"failed", "insufficient_context"}:
                return PlannerDecision(
                    action_type="refuse",
                    reason="上一步 observation 不足以继续完成当前请求。",
                    evidence_state="insufficient",
                    expected_next=None,
                )

            next_workflow_tool = _artifact_tool_name(artifact_type, allowed_tools)
            if next_workflow_tool and next_workflow_tool not in successful_tools:
                return PlannerDecision(
                    action_type="tool_call",
                    tool_name=next_workflow_tool,
                    tool_args={},
                    reason=f"已有前置证据，可继续执行 {next_workflow_tool}。",
                    evidence_state="sufficient",
                    expected_next="生成结构化结果后再决定是否直接答复。",
                )

            return PlannerDecision(
                action_type="final_answer",
                reason="当前已有足够 observation，可以结束规划并生成最终答复。",
                evidence_state="sufficient" if successful_tools else "partial",
                expected_next=None,
            )

        if decision.intent == "unsupported_or_unclear":
            return PlannerDecision(
                action_type="ask_clarification",
                reason="当前请求意图不够明确，请补充要查询的文档、主题或目标结果类型。",
                evidence_state="none",
                expected_next=None,
            )

        if decision.intent == "version_compare" and "compare_versions" in allowed_tools:
            return PlannerDecision(
                action_type="tool_call",
                tool_name="compare_versions",
                tool_args={
                    "target_document": decision.target_document_title or decision.requested_document_name,
                    "from_version_ref": decision.from_version_ref,
                    "to_version_ref": decision.to_version_ref,
                },
                reason="当前请求明确要求版本对比，先比较两个版本的差异。",
                evidence_state="none",
                expected_next="如差异内容充分，可继续提取待办或生成最终说明。",
            )

        if decision.intent in {"document_qa", "topic_qa"} and "search_docs" in allowed_tools:
            return PlannerDecision(
                action_type="tool_call",
                tool_name="search_docs",
                tool_args={
                    "query": user_query,
                    "target_document": decision.target_document_title or decision.requested_document_name or chat_context.previous_target_document,
                },
                reason="需要先获取 grounded 文档证据，再决定是否继续处理。",
                evidence_state="none",
                expected_next="检索后根据 observation 决定是直接回答还是继续生成结构化结果。",
            )

        workflow_tool = _artifact_tool_name(artifact_type, allowed_tools)
        if workflow_tool:
            if chat_context.previous_insufficient_evidence:
                return PlannerDecision(
                    action_type="refuse",
                    reason="上一轮证据不足，暂不直接生成结构化结果。",
                    evidence_state="insufficient",
                    expected_next="请先完成一次有证据支撑的问答。",
                )
            return PlannerDecision(
                action_type="tool_call",
                tool_name=workflow_tool,
                tool_args={},
                reason=f"当前请求主要依赖已有上下文，可尝试执行 {workflow_tool}。",
                evidence_state="partial" if chat_context.previous_tool_name else "none",
                expected_next="如果上下文不足，应转为拒绝或澄清。",
            )

        return PlannerDecision(
            action_type="ask_clarification",
            reason="暂时无法根据当前信息决定下一步工具，请补充更明确的目标。",
            evidence_state="none",
            expected_next=None,
        )


class LLMActionPlanner:
    provider_name = "openai-compatible-planner"

    def __init__(self, fallback: ActionPlanner | None = None) -> None:
        self.settings = get_settings()
        self.model_name = self.settings.effective_llm_router_model
        self.fallback = fallback or DeterministicFallbackPlanner()

    def plan_next_action(
        self,
        *,
        user_query: str,
        chat_context: ConversationMemory,
        router_result: RouterDecisionResult,
        available_tools: list[ToolDefinition],
        previous_actions: list[ToolAction],
        previous_observations: list[ToolObservation],
        remaining_steps: int,
    ) -> PlannerDecision:
        artifact_type = router_result.decision.artifact_type
        last_observation = previous_observations[-1] if previous_observations else None
        last_refusal_reason = None
        if last_observation and isinstance(last_observation.raw_output, dict):
            raw_reason = last_observation.raw_output.get("refusal_reason")
            if raw_reason:
                last_refusal_reason = str(raw_reason)

        terminal_decision = _terminal_decision_from_observation(
            last_observation=last_observation,
            last_refusal_reason=last_refusal_reason,
            artifact_type=artifact_type,
        )
        if terminal_decision is not None:
            return terminal_decision

        if not self._can_use_llm():
            return self.fallback.plan_next_action(
                user_query=user_query,
                chat_context=chat_context,
                router_result=router_result,
                available_tools=available_tools,
                previous_actions=previous_actions,
                previous_observations=previous_observations,
                remaining_steps=remaining_steps,
            )

        client = create_openai_compatible_client(self.settings)
        planner_state = {
            "user_query": user_query,
            "chat_context": _planner_context(chat_context),
            "router_hint": router_result.decision.model_dump(mode="json"),
            "available_tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "args_schema": tool.args_schema,
                    "safety_constraints": tool.safety_constraints,
                    "requires_evidence": tool.requires_evidence,
                    "output_type": tool.output_type,
                }
                for tool in available_tools
            ],
            "previous_actions": [action.model_dump(mode="json") for action in previous_actions],
            "previous_observations": [observation.model_dump(mode="json") for observation in previous_observations],
            "remaining_steps": remaining_steps,
        }
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Return the next planner decision as JSON.\n\n"
                            f"Planner state:\n{json.dumps(planner_state, ensure_ascii=False)}"
                        ),
                    },
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            _ = int((time.perf_counter() - started) * 1000)
            payload = _parse_json_payload(response.choices[0].message.content or "{}")
            decision = PlannerDecision.model_validate(payload)
        except Exception:
            return self.fallback.plan_next_action(
                user_query=user_query,
                chat_context=chat_context,
                router_result=router_result,
                available_tools=available_tools,
                previous_actions=previous_actions,
                previous_observations=previous_observations,
                remaining_steps=remaining_steps,
            )

        if decision.action_type == "tool_call":
            tool_args = dict(decision.tool_args)
            if decision.tool_name == "search_docs" and not tool_args.get("query"):
                tool_args["query"] = user_query
            decision = decision.model_copy(update={"tool_args": tool_args})
        decision = _sanitize_planner_decision(
            user_query=user_query,
            decision=decision,
            available_tools=available_tools,
            previous_actions=previous_actions,
            previous_observations=previous_observations,
            last_observation=last_observation,
            last_refusal_reason=last_refusal_reason,
            artifact_type=artifact_type,
        )
        return decision

    def _can_use_llm(self) -> bool:
        provider_candidates = [self.settings.router_provider, self.settings.answer_provider]
        return any(uses_openai_compatible_provider(item) for item in provider_candidates) and has_openai_compatible_credentials(
            self.settings
        )


class ActionPlannerFactory:
    @staticmethod
    def create() -> ActionPlanner:
        return LLMActionPlanner()


def _artifact_tool_name(artifact_type: str | None, allowed_tools: set[str]) -> str | None:
    mapping = {
        "tasks": "extract_todos",
        "weekly_report": "generate_weekly_report",
        "faq": "generate_faq",
    }
    tool_name = mapping.get(artifact_type or "")
    if tool_name in allowed_tools:
        return tool_name
    return None


def _planner_context(chat_context: ConversationMemory) -> dict[str, Any]:
    return {
        "older_summary": chat_context.older_summary,
        "history_lines": chat_context.history_lines,
        "previous_target_document": chat_context.previous_target_document,
        "previous_tool_name": chat_context.previous_tool_name,
        "previous_artifact_type": chat_context.previous_artifact_type,
        "previous_intent": chat_context.previous_intent,
        "previous_refusal_reason": chat_context.previous_refusal_reason,
        "previous_insufficient_evidence": chat_context.previous_insufficient_evidence,
        "previous_observation_summary": chat_context.previous_observation_summary,
        "recent_message_count": chat_context.recent_message_count,
        "older_message_count": chat_context.older_message_count,
    }


def _parse_json_payload(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        loaded = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid planner json payload") from exc
    if not isinstance(loaded, dict):
        raise ValueError("planner response must be an object")
    return loaded


def _terminal_decision_from_observation(
    *,
    last_observation: ToolObservation | None,
    last_refusal_reason: str | None,
    artifact_type: str | None,
) -> PlannerDecision | None:
    if last_observation is None:
        return None

    if last_refusal_reason == "unknown_tool_name":
        return PlannerDecision(
            action_type="refuse",
            reason="上一步请求了未知工具，已停止继续执行。",
            evidence_state="insufficient",
            expected_next=None,
        )
    if artifact_type and last_refusal_reason == "insufficient_context":
        return PlannerDecision(
            action_type="refuse",
            reason="当前上下文证据不足，暂不继续生成结构化结果。",
            evidence_state="insufficient",
            expected_next="请先完成一次有证据支撑的问答或版本对比。",
        )
    if last_refusal_reason == "insufficient_versions_for_compare":
        return PlannerDecision(
            action_type="refuse",
            reason="当前可访问范围内可比较的版本数量不足，暂时无法继续完成版本对比请求。",
            evidence_state="insufficient",
            expected_next="请确认该文档至少存在两个可访问版本后再比较。",
        )
    if last_refusal_reason == "unable_to_resolve_version_pair":
        return PlannerDecision(
            action_type="ask_clarification",
            reason="还需要更明确的版本范围，例如上一版、v1、v2 或最新版。",
            evidence_state="insufficient",
            expected_next="请补充要比较的两个版本。",
        )
    if last_refusal_reason == "target_document_not_accessible_or_not_found":
        return PlannerDecision(
            action_type="refuse",
            reason="目标文档不在当前可访问范围内，暂不继续调用后续工具。",
            evidence_state="insufficient",
            expected_next=None,
        )
    return None


def _sanitize_planner_decision(
    *,
    user_query: str,
    decision: PlannerDecision,
    available_tools: list[ToolDefinition],
    previous_actions: list[ToolAction],
    previous_observations: list[ToolObservation],
    last_observation: ToolObservation | None,
    last_refusal_reason: str | None,
    artifact_type: str | None,
) -> PlannerDecision:
    terminal_decision = _terminal_decision_from_observation(
        last_observation=last_observation,
        last_refusal_reason=last_refusal_reason,
        artifact_type=artifact_type,
    )
    if terminal_decision is not None and decision.action_type == "tool_call":
        return terminal_decision
    workflow_guard = _guard_unrequested_workflow_tool(
        user_query=user_query,
        decision=decision,
        available_tools=available_tools,
        previous_observations=previous_observations,
        artifact_type=artifact_type,
    )
    if workflow_guard is not None:
        return workflow_guard
    return _avoid_redundant_repeated_tool_call(
        decision=decision,
        available_tools=available_tools,
        previous_actions=previous_actions,
        previous_observations=previous_observations,
        artifact_type=artifact_type,
    )


def _guard_unrequested_workflow_tool(
    *,
    user_query: str,
    decision: PlannerDecision,
    available_tools: list[ToolDefinition],
    previous_observations: list[ToolObservation],
    artifact_type: str | None,
) -> PlannerDecision | None:
    if decision.action_type != "tool_call" or decision.tool_name not in WORKFLOW_TOOL_NAMES:
        return None
    if artifact_type or _has_explicit_artifact_request(user_query):
        return None
    if previous_observations:
        return PlannerDecision(
            action_type="final_answer",
            reason="用户是在询问文档规则，不是要求生成结构化产物；已有 observation 可直接回答。",
            evidence_state="sufficient",
            expected_next=None,
        )
    allowed_tools = {tool.name for tool in available_tools}
    if "search_docs" in allowed_tools:
        return PlannerDecision(
            action_type="tool_call",
            tool_name="search_docs",
            tool_args={"query": user_query},
            reason="用户是在询问文档规则，不是要求生成结构化产物；先检索证据。",
            evidence_state="none",
            expected_next="检索后根据证据生成回答。",
        )
    return PlannerDecision(
        action_type="ask_clarification",
        reason="当前请求不像结构化产物生成，请补充要查询的文档或主题。",
        evidence_state="none",
        expected_next=None,
    )


def _has_explicit_artifact_request(question: str) -> bool:
    normalized = re.sub(r"\s+", "", question.casefold())
    return any(
        marker in normalized
        for marker in (
            "整理成待办",
            "提取待办",
            "生成待办",
            "待办事项",
            "任务项",
            "actionitem",
            "todo",
            "生成周报",
            "周报草稿",
            "weeklyreport",
            "生成faq",
            "faq草稿",
            "常见问题",
        )
    )


def _avoid_redundant_repeated_tool_call(
    *,
    decision: PlannerDecision,
    available_tools: list[ToolDefinition],
    previous_actions: list[ToolAction],
    previous_observations: list[ToolObservation],
    artifact_type: str | None,
) -> PlannerDecision:
    if decision.action_type != "tool_call" or not previous_actions or not previous_observations:
        return decision

    previous_action = previous_actions[-1]
    last_observation = previous_observations[-1]
    if previous_action.action_type != "tool_call":
        return decision
    if previous_action.tool_name != decision.tool_name or previous_action.tool_args != decision.tool_args:
        return decision
    if last_observation.tool_name != decision.tool_name:
        return decision

    if last_observation.status == "completed":
        allowed_tools = {tool.name for tool in available_tools}
        workflow_tool = _artifact_tool_name(artifact_type, allowed_tools)
        if workflow_tool and workflow_tool != decision.tool_name:
            return PlannerDecision(
                action_type="tool_call",
                tool_name=workflow_tool,
                tool_args={},
                reason="已有同一轮检索或对比结果，继续执行结构化结果工具即可。",
                evidence_state="sufficient",
                expected_next="结构化结果生成后返回最终答复。",
            )
        return PlannerDecision(
            action_type="final_answer",
            reason="已有同一工具的有效 observation，无需重复调用相同工具。",
            evidence_state="sufficient",
            expected_next=None,
        )

    if last_observation.status in {"failed", "insufficient_context"}:
        return PlannerDecision(
            action_type="refuse",
            reason="上一轮同一工具未获得新证据，停止重复调用。",
            evidence_state="insufficient",
            expected_next=None,
        )
    return decision
