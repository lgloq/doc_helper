from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from app.models.chat import ChatMessage
from app.schemas.llm import AgentRunTrace, PlannerDecision, PublicToolName, RouterDecisionResult, ToolAction, ToolPlan
from app.services.chat.memory import ConversationMemory
from app.services.llm.planner import ActionPlanner, ActionPlannerFactory
from app.services.llm.tool_executor import ToolExecutionResult, ToolExecutor
from app.services.llm.tool_registry import DEFAULT_TOOL_REGISTRY, ToolDefinition, ToolRegistry

AVAILABLE_TOOLS: list[PublicToolName] = DEFAULT_TOOL_REGISTRY.names()


@dataclass
class AgentRunnerResult:
    run_trace: AgentRunTrace
    final_action: ToolAction
    execution_results: list[ToolExecutionResult] = field(default_factory=list)
    working_messages: list[ChatMessage] = field(default_factory=list)


class AgentRunner:
    def __init__(
        self,
        *,
        tool_executor: ToolExecutor,
        planner: ActionPlanner | None = None,
        tool_registry: ToolRegistry | None = None,
        max_steps: int = 3,
    ) -> None:
        self.tool_executor = tool_executor
        self.tool_registry = tool_registry or DEFAULT_TOOL_REGISTRY
        self.planner = planner or ActionPlannerFactory.create()
        self.max_steps = max_steps

    def run(
        self,
        *,
        actor,
        user_query: str,
        session_id: UUID | None,
        top_k: int,
        chat_context: ConversationMemory,
        router_result: RouterDecisionResult,
        existing_messages: list[ChatMessage],
        available_tools: list[PublicToolName] | None = None,
    ) -> AgentRunnerResult:
        tool_names = list(available_tools or AVAILABLE_TOOLS)
        tool_definitions = self.tool_registry.list_definitions(tool_names)
        working_messages = list(existing_messages)
        execution_results: list[ToolExecutionResult] = []
        actions: list[ToolAction] = []
        final_action: ToolAction | None = None
        final_status = "completed"
        final_reason: str | None = None

        tool_plan = ToolPlan(
            planner_name=type(self.planner).__name__,
            available_tools=[item.name for item in tool_definitions],
            max_steps=self.max_steps,
            initial_intent=router_result.decision.intent,
            requested_artifact_type=router_result.decision.artifact_type,
            context_summary=_build_context_summary(chat_context),
        )

        for step_index in range(1, self.max_steps + 1):
            decision = self.planner.plan_next_action(
                user_query=user_query,
                chat_context=chat_context,
                router_result=router_result,
                available_tools=tool_definitions,
                previous_actions=actions,
                previous_observations=[item.observation for item in execution_results],
                remaining_steps=self.max_steps - step_index + 1,
            )
            action = self._to_tool_action(step_index=step_index, decision=decision, execution_results=execution_results)
            actions.append(action)

            if action.action_type in {"final_answer", "refuse", "ask_clarification"}:
                final_action = action
                final_status = _final_status_from_action(action.action_type)
                final_reason = action.reason
                break

            execution_result = self.tool_executor.execute(
                actor=actor,
                question=user_query,
                session_id=session_id,
                action=action,
                router_result=router_result,
                conversation_memory=chat_context,
                working_messages=working_messages,
                top_k=top_k,
            )
            execution_results.append(execution_result)
            if execution_result.synthetic_messages:
                working_messages.extend(execution_result.synthetic_messages)

        if final_action is None:
            final_action, final_status, final_reason = self._finalize_after_limit(
                step_index=len(actions) + 1,
                execution_results=execution_results,
            )
            actions.append(final_action)

        run_trace = AgentRunTrace(
            tool_plan=tool_plan,
            actions=actions,
            observations=[item.observation for item in execution_results],
            final_status=final_status,
            final_reason=final_reason,
        )
        return AgentRunnerResult(
            run_trace=run_trace,
            final_action=final_action,
            execution_results=execution_results,
            working_messages=working_messages,
        )

    @staticmethod
    def _to_tool_action(
        *,
        step_index: int,
        decision: PlannerDecision,
        execution_results: list[ToolExecutionResult],
    ) -> ToolAction:
        return ToolAction(
            step_index=step_index,
            action_type=decision.action_type,
            tool_name=decision.tool_name,
            tool_args=dict(decision.tool_args),
            reason=decision.reason,
            evidence_state=decision.evidence_state,
            expected_next=decision.expected_next,
            depends_on=[item.observation.step_index for item in execution_results if item.observation.status == "completed"],
        )

    @staticmethod
    def _finalize_after_limit(
        *,
        step_index: int,
        execution_results: list[ToolExecutionResult],
    ) -> tuple[ToolAction, str, str]:
        successful_results = [item for item in execution_results if item.observation.status == "completed"]
        if successful_results:
            return (
                ToolAction(
                    step_index=step_index,
                    action_type="final_answer",
                    reason="已达到最大执行步数，基于已有 observation 结束本轮回答。",
                    evidence_state="partial",
                    expected_next=None,
                    depends_on=[item.observation.step_index for item in successful_results],
                ),
                "max_steps_reached",
                "agent runner reached max_steps and finalized from existing observations",
            )
        return (
            ToolAction(
                step_index=step_index,
                action_type="refuse",
                reason="已达到最大执行步数，且没有可用 observation 支撑最终回答。",
                evidence_state="insufficient",
                expected_next=None,
                depends_on=[item.observation.step_index for item in execution_results],
            ),
            "max_steps_reached",
            "agent runner reached max_steps without usable observations",
        )


def _build_context_summary(chat_context: ConversationMemory) -> str | None:
    parts: list[str] = []
    if chat_context.previous_target_document:
        parts.append(f"previous_target_document={chat_context.previous_target_document}")
    if chat_context.previous_tool_name:
        parts.append(f"previous_tool_name={chat_context.previous_tool_name}")
    if chat_context.previous_observation_summary:
        parts.append(f"previous_observation={chat_context.previous_observation_summary}")
    if chat_context.previous_artifact_type:
        parts.append(f"previous_artifact_type={chat_context.previous_artifact_type}")
    if chat_context.older_summary:
        parts.append("older_summary_available=true")
    if not parts:
        return None
    return "；".join(parts)


def _final_status_from_action(action_type: str) -> str:
    if action_type == "final_answer":
        return "completed"
    if action_type == "ask_clarification":
        return "clarification_required"
    return "refused"
