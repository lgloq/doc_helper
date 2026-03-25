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
  openai: "大模型摘要",
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


