import { StatusBadge } from "./StatusBadge";
import { formatAgentStepName, formatAgentStepStatus, formatToolName } from "../lib/display";
import type { AgentStepRead } from "../types/api";

interface ExecutionTraceProps {
  steps: AgentStepRead[];
}

export function ExecutionTrace({ steps }: ExecutionTraceProps) {
  if (!steps.length) {
    return null;
  }

  return (
    <details className="execution-trace">
      <summary>处理轨迹</summary>
      <div className="execution-trace-list">
        {steps.map((step) => (
          <div className="execution-trace-item" key={step.name}>
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
        ))}
      </div>
    </details>
  );
}
