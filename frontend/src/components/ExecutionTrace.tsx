import { StatusBadge } from "./StatusBadge";
import {
  formatAgentStepName,
  formatAgentStepStatus,
  formatArtifactType,
  formatCopilotIntent,
  formatEvidenceState,
  formatToolActionType,
  formatToolName,
  formatToolObservationStatus,
} from "../lib/display";
import type { AgentRunTraceRead, AgentStepRead, ToolActionRead, ToolObservationRead } from "../types/api";

interface ExecutionTraceProps {
  steps: AgentStepRead[];
  runTrace?: AgentRunTraceRead | null;
}

export function ExecutionTrace({ steps, runTrace }: ExecutionTraceProps) {
  if (!steps.length && !runTrace) {
    return null;
  }

  const observationsByStep = new Map<number, ToolObservationRead>();
  runTrace?.observations.forEach((item) => observationsByStep.set(item.step_index, item));
  const finalAction = runTrace ? [...runTrace.actions].reverse().find((item) => item.action_type !== "tool_call") : undefined;
  const compactContextSummary = formatContextSummary(runTrace?.tool_plan.context_summary);

  return (
    <details className="execution-trace">
      <summary>处理轨迹</summary>
      <div className="execution-trace-list">
        {runTrace ? (
          <div className="execution-trace-item">
            <div className="execution-trace-topline">
              <strong>Tool Plan</strong>
              <StatusBadge tone="info">{runTrace.final_status}</StatusBadge>
            </div>
            <p className="muted">planner：{runTrace.tool_plan.planner_name}</p>
            <p className="muted">初始意图：{formatCopilotIntent(runTrace.tool_plan.initial_intent)}</p>
            {runTrace.tool_plan.requested_artifact_type ? (
              <p className="muted">目标结果：{formatArtifactType(runTrace.tool_plan.requested_artifact_type)}</p>
            ) : null}
            <p className="muted">最大步数：{runTrace.tool_plan.max_steps}</p>
            {compactContextSummary ? <p>上下文：{compactContextSummary}</p> : null}
            <p className="muted">可用工具：{runTrace.tool_plan.available_tools.map((name) => formatToolName(name)).join(" / ")}</p>
          </div>
        ) : null}
        {runTrace?.actions.map((action) => {
          const observation = observationsByStep.get(action.step_index);
          return (
            <TraceActionItem
              key={`action-${action.step_index}-${action.action_type}`}
              action={action}
              observation={observation}
            />
          );
        })}
        {runTrace?.final_reason ? (
          <div className="execution-trace-item">
            <div className="execution-trace-topline">
              <strong>Final Trace</strong>
              <StatusBadge tone={runTrace.final_status === "refused" ? "warning" : "info"}>{runTrace.final_status}</StatusBadge>
            </div>
            {finalAction ? <p className="muted">final action：{formatToolActionType(finalAction.action_type)}</p> : null}
            <p>{runTrace.final_reason}</p>
          </div>
        ) : null}
        {runTrace && steps.length ? (
          <details className="execution-trace-secondary">
            <summary>兼容摘要</summary>
            <div className="execution-trace-secondary-list">
              {steps.map((step) => (
                <LegacyStepItem key={`agent-step-${step.name}`} step={step} />
              ))}
            </div>
          </details>
        ) : null}
        {!runTrace
          ? steps.map((step) => <LegacyStepItem key={`agent-step-${step.name}`} step={step} />)
          : null}
      </div>
    </details>
  );
}

interface LegacyStepItemProps {
  step: AgentStepRead;
}

function LegacyStepItem({ step }: LegacyStepItemProps) {
  return (
    <div className="execution-trace-item">
      <div className="execution-trace-topline">
        <strong>{formatAgentStepName(step.name)}</strong>
        <StatusBadge tone={step.status === "completed" ? "success" : "warning"}>
          {formatAgentStepStatus(step.status)}
        </StatusBadge>
      </div>
      <p className="muted">输入：{step.input_summary}</p>
      <p>{step.output_summary}</p>
      {step.tool_name ? <p className="muted">工具：{formatToolName(step.tool_name)}</p> : null}
    </div>
  );
}

interface TraceActionItemProps {
  action: ToolActionRead;
  observation?: ToolObservationRead;
}

function TraceActionItem({ action, observation }: TraceActionItemProps) {
  const formattedToolArgs = formatToolArgs(action.tool_args);

  return (
    <div className="execution-trace-item">
      <div className="execution-trace-topline">
        <strong>Step {action.step_index}</strong>
        <StatusBadge tone={action.action_type === "tool_call" ? "info" : action.action_type === "final_answer" ? "success" : "warning"}>
          {formatToolActionType(action.action_type)}
        </StatusBadge>
      </div>
      <p className="muted">planner decision：{action.action_type}</p>
      <p className="muted">reason：{action.reason}</p>
      <p className="muted">evidence state：{formatEvidenceState(action.evidence_state)}</p>
      {action.tool_name ? <p>tool call：{formatToolName(action.tool_name)}</p> : null}
      {formattedToolArgs ? <p className="muted">tool args：{formattedToolArgs}</p> : null}
      {action.expected_next ? <p className="muted">expected next：{action.expected_next}</p> : null}
      {action.depends_on.length ? <p className="muted">depends on：{action.depends_on.join(", ")}</p> : null}
      {observation ? (
        <>
          <p>
            observation：
            <span className="muted">
              {formatToolObservationStatus(observation.status)}
            </span>
          </p>
          <p>{observation.output_summary}</p>
          {observation.evidence_refs.length ? (
            <p className="muted">evidence：{observation.evidence_refs.join("；")}</p>
          ) : null}
        </>
      ) : null}
    </div>
  );
}

const CONTEXT_KEY_LABELS: Record<string, string> = {
  previous_target_document: "上一目标文档",
  previous_tool_name: "上一工具",
  previous_artifact_type: "上一结果类型",
  previous_observation: "上一观察摘要",
  previous_observation_summary: "上一观察摘要",
  previous_refusal_reason: "上一拒答原因",
  older_summary_available: "更早历史摘要",
};

function formatContextSummary(summary: string | null | undefined): string | null {
  if (!summary) {
    return null;
  }

  const parts = summary
    .split(";")
    .map((item) => item.trim())
    .filter(Boolean);

  if (!parts.length) {
    return shortenText(summary, 72);
  }

  const displayed = parts.slice(0, 4).map(formatContextPart);
  const suffix = parts.length > 4 ? ` 等 ${parts.length} 项` : "";
  return `${displayed.join("；")}${suffix}`;
}

function formatContextPart(part: string): string {
  const [rawKey, ...rest] = part.split("=");
  if (!rest.length) {
    return shortenText(part, 28);
  }

  const key = rawKey.trim();
  const value = rest.join("=").trim();
  const label = CONTEXT_KEY_LABELS[key] ?? key;

  if (key === "previous_tool_name") {
    return `${label}=${formatToolName(value)}`;
  }

  if (key === "previous_artifact_type") {
    return `${label}=${formatArtifactType(value)}`;
  }

  if (key === "older_summary_available") {
    return `${label}=${value === "true" ? "有" : "无"}`;
  }

  return `${label}=${shortenText(value, 24)}`;
}

function formatToolArgs(toolArgs: Record<string, unknown>): string | null {
  const entries = Object.entries(toolArgs);
  if (!entries.length) {
    return null;
  }

  return entries
    .map(([key, value]) => `${key}=${formatArgValue(value)}`)
    .join("；");
}

function formatArgValue(value: unknown): string {
  if (value == null) {
    return "null";
  }

  if (typeof value === "string") {
    return shortenText(value, 36);
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }

  if (Array.isArray(value)) {
    return shortenText(value.join(" / "), 36);
  }

  return shortenText(JSON.stringify(value), 36);
}

function shortenText(value: string, maxLength: number): string {
  if (value.length <= maxLength) {
    return value;
  }
  return `${value.slice(0, maxLength - 1)}…`;
}
