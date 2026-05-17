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
import type { AgentRunTraceRead, AgentStepRead, SearchDebugInfo, ToolActionRead, ToolObservationRead } from "../types/api";

interface ExecutionTraceProps {
  steps: AgentStepRead[];
  runTrace?: AgentRunTraceRead | null;
  retrievalDebug?: SearchDebugInfo | null;
  originalQuery?: string | null;
}

export function ExecutionTrace({ steps, runTrace, retrievalDebug, originalQuery }: ExecutionTraceProps) {
  if (!steps.length && !runTrace && !retrievalDebug) {
    return null;
  }

  const observationsByStep = new Map<number, ToolObservationRead>();
  runTrace?.observations.forEach((item) => observationsByStep.set(item.step_index, item));
  const finalAction = runTrace ? [...runTrace.actions].reverse().find((item) => item.action_type !== "tool_call") : undefined;
  const compactContextSummary = formatContextSummary(runTrace?.tool_plan.context_summary);
  const retrievalSummary = retrievalDebug ? buildRetrievalSummary(retrievalDebug, originalQuery) : null;

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
        {retrievalDebug ? (
          <div className="execution-trace-item">
            <div className="execution-trace-topline">
              <strong>检索计划</strong>
              <StatusBadge tone={retrievalDebug.query_rewrite_applied ? "info" : "neutral"}>
                {retrievalDebug.query_rewrite_applied ? "已增强" : "原始查询"}
              </StatusBadge>
            </div>
            <p className="muted">检索语句：{retrievalSummary?.retrievalQuery ?? "-"}</p>
            <p className="muted">增强策略：{retrievalSummary?.strategySummary ?? "未改写"}</p>
            <p className="muted">候选结果：{retrievalSummary?.candidateSummary ?? "-"}</p>
            {typeof retrievalDebug.query_plan_candidate_count === "number" && retrievalDebug.query_plan_candidate_count > 1 ? (
              <p className="muted">
                候选方案：{retrievalDebug.query_plan_candidate_count} 个
                {retrievalDebug.query_plan_selected ? `，选中 ${retrievalDebug.query_plan_selected}` : ""}
              </p>
            ) : null}
            {retrievalDebug.query_plan_selection_reason ? (
              <p className="muted">选中原因：{retrievalDebug.query_plan_selection_reason}</p>
            ) : null}
            {retrievalSummary?.hasDetails ? (
              <details className="execution-trace-secondary execution-trace-retrieval-details">
                <summary>查看检索细节</summary>
                <div className="execution-trace-secondary-list">
                  {retrievalSummary.originalQuery ? <p className="muted">原始问题：{retrievalSummary.originalQuery}</p> : null}
                  {retrievalSummary.methodSummary ? <p className="muted">改写方式：{retrievalSummary.methodSummary}</p> : null}
                  {retrievalSummary.routeSummary ? <p className="muted">分路情况：{retrievalSummary.routeSummary}</p> : null}
                  {retrievalDebug.query_plan_probe_applied ? <p className="muted">方案选优：已执行低成本试探</p> : null}
                  {retrievalSummary.lexicalVariants.length ? (
                    <div className="execution-trace-detail-block">
                      <p className="muted">关键词检索变体：</p>
                      <ul className="execution-trace-detail-list">
                        {retrievalSummary.lexicalVariants.map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    </div>
                  ) : null}
                </div>
              </details>
            ) : null}
          </div>
        ) : null}
        {runTrace?.actions.map((action) => {
          const observation = observationsByStep.get(action.step_index);
          return (
            <TraceActionItem
              key={`action-${action.step_index}-${action.action_type}`}
              action={action}
              observation={observation}
              hideToolArgs={Boolean(retrievalDebug && action.tool_name === "search_docs")}
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
  hideToolArgs?: boolean;
}

function TraceActionItem({ action, observation, hideToolArgs = false }: TraceActionItemProps) {
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
      {formattedToolArgs && !hideToolArgs ? <p className="muted">tool args：{formattedToolArgs}</p> : null}
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

interface RetrievalSummary {
  retrievalQuery: string;
  originalQuery: string | null;
  strategySummary: string;
  candidateSummary: string;
  routeSummary: string | null;
  methodSummary: string | null;
  lexicalVariants: string[];
  hasDetails: boolean;
}

function buildRetrievalSummary(retrievalDebug: SearchDebugInfo, originalQuery?: string | null): RetrievalSummary {
  const retrievalQuery = retrievalDebug.retrieval_query ?? originalQuery ?? "-";
  const lexicalVariants = dedupeQueries(retrievalDebug.lexical_queries ?? []);
  const compactOriginalQuery =
    originalQuery && normalizeForCompare(originalQuery) !== normalizeForCompare(retrievalQuery) ? originalQuery : null;
  const strategySummary = retrievalDebug.query_rewrite_strategies?.length
    ? retrievalDebug.query_rewrite_strategies.map(formatRewriteStrategy).join(" / ")
    : "未改写";
  const routeSummary =
    typeof retrievalDebug.pre_rerank_count === "number" && typeof retrievalDebug.post_rerank_count === "number"
      ? `关键词检索 ${retrievalDebug.lexical_candidate_count} 个，向量检索 ${retrievalDebug.vector_candidate_count} 个，合并后 ${retrievalDebug.pre_rerank_count} 个，重排后保留 ${retrievalDebug.post_rerank_count} 个`
      : `关键词检索 ${retrievalDebug.lexical_candidate_count} 个，向量检索 ${retrievalDebug.vector_candidate_count} 个`;
  const candidateSummary =
    typeof retrievalDebug.pre_rerank_count === "number" && typeof retrievalDebug.post_rerank_count === "number"
      ? `召回 ${retrievalDebug.pre_rerank_count} 个候选，重排后保留 ${retrievalDebug.post_rerank_count} 个`
      : `关键词检索 ${retrievalDebug.lexical_candidate_count} 个，向量检索 ${retrievalDebug.vector_candidate_count} 个`;
  const methodSummary =
    retrievalDebug.query_rewrite_provider || retrievalDebug.query_rewrite_model
      ? `${retrievalDebug.query_rewrite_provider ?? "rules-only"}${retrievalDebug.query_rewrite_model ? ` · ${retrievalDebug.query_rewrite_model}` : ""}${typeof retrievalDebug.query_rewrite_latency_ms === "number" ? ` · ${retrievalDebug.query_rewrite_latency_ms} ms` : ""}`
      : null;
  const showRouteSummary = routeSummary !== candidateSummary;

  return {
    retrievalQuery,
    originalQuery: compactOriginalQuery,
    strategySummary,
    candidateSummary,
    routeSummary: showRouteSummary ? routeSummary : null,
    methodSummary,
    lexicalVariants,
    hasDetails: Boolean(compactOriginalQuery || methodSummary || lexicalVariants.length || showRouteSummary),
  };
}

function dedupeQueries(items: string[]): string[] {
  const seen = new Set<string>();
  const results: string[] = [];
  for (const item of items) {
    const trimmed = item.trim();
    if (!trimmed) {
      continue;
    }
    const key = normalizeForCompare(trimmed);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    results.push(trimmed);
  }
  return results;
}

function normalizeForCompare(value: string): string {
  return value.replace(/\s+/g, " ").trim().toLowerCase();
}

function formatRewriteStrategy(strategy: string): string {
  const labels: Record<string, string> = {
    normalize: "规范化",
    focus_keywords: "关键词聚焦",
    title_anchor: "标题锚定",
    llm_rewrite: "LLM 改写",
  };
  return labels[strategy] ?? strategy;
}
