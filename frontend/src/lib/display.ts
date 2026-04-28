import type { DocumentStatus, IngestStatus, MessageRole, PrincipalType, RoleName } from "../types/api";

const ROLE_LABELS: Record<RoleName, string> = {
  viewer: "普通员工",
  manager: "组长",
  admin: "管理员",
};

const DOCUMENT_STATUS_LABELS: Record<DocumentStatus, string> = {
  draft: "草稿",
  active: "启用中",
  archived: "已归档",
};

const INGEST_STATUS_LABELS: Record<IngestStatus, string> = {
  pending: "待处理",
  processing: "处理中",
  ready: "已就绪",
  failed: "失败",
};

const PRINCIPAL_TYPE_LABELS: Record<PrincipalType, string> = {
  public: "公开",
  user: "指定用户",
  role: "指定角色",
  team: "团队",
};

const MESSAGE_ROLE_LABELS: Record<MessageRole, string> = {
  user: "用户",
  assistant: "助手",
  system: "系统",
};

const CONFIDENCE_LABELS: Record<string, string> = {
  high: "高",
  medium: "中",
  low: "低",
  insufficient: "证据不足",
};

const PRIORITY_LABELS: Record<string, string> = {
  high: "高",
  medium: "中",
  low: "低",
};

const WORKFLOW_STATUS_LABELS: Record<string, string> = {
  open: "待处理",
  draft: "草稿",
  completed: "已完成",
  published: "已发布",
  pending: "待处理",
  processing: "处理中",
  running: "进行中",
  failed: "失败",
};

const TRACE_TYPE_LABELS: Record<string, string> = {
  chat_qa: "问答追踪",
  eval_case: "评测追踪",
};

const SUMMARY_PROVIDER_LABELS: Record<string, string> = {
  deterministic: "规则摘要",
  deterministic_fallback: "规则回退摘要",
  openai: "大模型摘要",
  "openai-compatible": "大模型摘要",
  system: "系统摘要",
};

const DIFF_CHANGE_TYPE_LABELS: Record<string, string> = {
  insert: "新增",
  delete: "删除",
  replace: "修改",
  equal: "未变更",
};

export function formatRoleName(value: RoleName | string | null | undefined, fallback = "未知角色"): string {
  if (!value) {
    return fallback;
  }
  return ROLE_LABELS[value as RoleName] ?? value;
}

export function formatDocumentStatus(value: DocumentStatus | string | null | undefined): string {
  if (!value) {
    return "未知状态";
  }
  return DOCUMENT_STATUS_LABELS[value as DocumentStatus] ?? value;
}

export function formatIngestStatus(value: IngestStatus | string | null | undefined): string {
  if (!value) {
    return "未知状态";
  }
  return INGEST_STATUS_LABELS[value as IngestStatus] ?? value;
}

export function formatPrincipalType(value: PrincipalType | string | null | undefined): string {
  if (!value) {
    return "未知类型";
  }
  return PRINCIPAL_TYPE_LABELS[value as PrincipalType] ?? value;
}

export function formatMessageRole(value: MessageRole | string | null | undefined): string {
  if (!value) {
    return "未知角色";
  }
  return MESSAGE_ROLE_LABELS[value as MessageRole] ?? value;
}

export function formatConfidence(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  return CONFIDENCE_LABELS[value] ?? value;
}

export function formatPriority(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  return PRIORITY_LABELS[value] ?? value;
}

export function formatWorkflowStatus(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  return WORKFLOW_STATUS_LABELS[value] ?? value;
}

export function formatTraceType(value: string | null | undefined): string {
  if (!value) {
    return "未知追踪";
  }
  return TRACE_TYPE_LABELS[value] ?? value;
}

export function formatSummaryProvider(value: string | null | undefined): string {
  if (!value) {
    return "未知来源";
  }
  return SUMMARY_PROVIDER_LABELS[value] ?? value;
}

export function formatDiffChangeType(value: string | null | undefined): string {
  if (!value) {
    return "未知变更";
  }
  return DIFF_CHANGE_TYPE_LABELS[value] ?? value;
}

export function formatBooleanFlag(value: boolean): string {
  return value ? "是" : "否";
}

export function formatSourceCount(count: number): string {
  return `${count} 条来源`;
}



const COPILOT_INTENT_LABELS: Record<string, string> = {
  document_qa: "文档问答",
  topic_qa: "主题问答",
  version_compare: "版本对比",
  workflow_generation: "派生生成",
  unsupported_or_unclear: "暂不支持/意图不清",
};

const ARTIFACT_TYPE_LABELS: Record<string, string> = {
  tasks: "待办事项",
  weekly_report: "周报草稿",
  faq: "FAQ 草稿",
};

const REFUSAL_REASON_LABELS: Record<string, string> = {
  target_document_not_accessible_or_not_found: "目标文档不可访问或不存在",
  no_relevant_evidence_in_target_document: "目标文档内缺少足够相关证据",
  insufficient_relevant_evidence: "未找到足够相关的可访问证据",
  invalid_or_missing_citations: "引用来源校验失败",
  unsupported_or_unclear: "问题不明确或暂不支持",
  unsupported_or_unclear_workflow_request: "无法判断要生成哪类结果",
  missing_session_context: "缺少可用于生成结果的会话上下文",
  insufficient_session_context_for_workflow: "当前会话缺少足够稳定的问答结果",
  insufficient_versions_for_compare: "可比较的版本数量不足",
  unable_to_resolve_version_pair: "无法解析要比较的版本",
  clarification_required: "需要补充更多上下文或约束",
};

const AGENT_STEP_LABELS: Record<string, string> = {
  query_analysis: "query_analysis",
  tool_selection: "tool_selection",
  tool_execution: "tool_execution",
  evidence_review: "evidence_review",
  answer_generation: "answer_generation",
};

const AGENT_STEP_STATUS_LABELS: Record<string, string> = {
  completed: "已完成",
  skipped: "已跳过",
  refused: "已拒绝",
};

const TOOL_NAME_LABELS: Record<string, string> = {
  search_docs: "search_docs",
  compare_versions: "compare_versions",
  extract_todos: "extract_todos",
  generate_weekly_report: "generate_weekly_report",
  generate_faq: "generate_faq",
  none: "none",
};

export function formatCopilotIntent(value: string | null | undefined): string {
  if (!value) {
    return "未知意图";
  }
  return COPILOT_INTENT_LABELS[value] ?? value;
}

export function formatArtifactType(value: string | null | undefined): string {
  if (!value) {
    return "未知类型";
  }
  return ARTIFACT_TYPE_LABELS[value] ?? value;
}

export function formatRefusalReason(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  return REFUSAL_REASON_LABELS[value] ?? value;
}

export function formatAgentStepName(value: string | null | undefined): string {
  if (!value) {
    return "unknown_step";
  }
  return AGENT_STEP_LABELS[value] ?? value;
}

export function formatAgentStepStatus(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  return AGENT_STEP_STATUS_LABELS[value] ?? value;
}

export function formatToolName(value: string | null | undefined): string {
  if (!value) {
    return "none";
  }
  return TOOL_NAME_LABELS[value] ?? value;
}

const TOOL_ACTION_LABELS: Record<string, string> = {
  tool_call: "调用工具",
  final_answer: "生成最终回答",
  refuse: "拒绝继续执行",
  ask_clarification: "请求补充信息",
};

const EVIDENCE_STATE_LABELS: Record<string, string> = {
  none: "尚无证据",
  partial: "证据部分具备",
  sufficient: "证据充分",
  insufficient: "证据不足",
};

const TOOL_OBSERVATION_STATUS_LABELS: Record<string, string> = {
  completed: "已完成",
  failed: "失败",
  skipped: "已跳过",
  insufficient_context: "上下文不足",
};

export function formatToolActionType(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  return TOOL_ACTION_LABELS[value] ?? value;
}

export function formatToolObservationStatus(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  return TOOL_OBSERVATION_STATUS_LABELS[value] ?? value;
}

export function formatEvidenceState(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  return EVIDENCE_STATE_LABELS[value] ?? value;
}

