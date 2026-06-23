import type {
  ChatMessageCreateResponse,
  ChatSessionDetailRead,
  ChatSessionRead,
  ChunkRead,
  DepartmentRead,
  DocumentACLRead,
  DocumentAccessDebugRead,
  DocumentDiffRead,
  DocumentDiffSummaryRead,
  DocumentRead,
  DocumentUploadResponse,
  DocumentVersionRead,
  EvalDashboardRead,
  EvalDatasetRead,
  EvalRunDetailRead,
  EvalRunRead,
  FAQEntryRead,
  FAQGenerateResponse,
  IngestionResultRead,
  TaskExtractResponse,
  TaskItemRead,
  TokenResponse,
  UserCreatePayload,
  TraceLogRead,
  UserUpdatePayload,
  UserRead,
  WeeklyReportDraftRead,
  WeeklyReportGenerateResponse,
} from "../types/api";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "";
const DEFAULT_REQUEST_TIMEOUT_MS = 30_000;
const CHAT_REQUEST_TIMEOUT_MS = 90_000;
const EVAL_REQUEST_TIMEOUT_MS = 30_000;

export class ApiError extends Error {
  status: number;
  payload: unknown;

  constructor(message: string, status: number, payload: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

interface RequestOptions extends Omit<RequestInit, "body"> {
  token?: string | null;
  body?: BodyInit | object | null;
  timeoutMs?: number;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const headers = new Headers(options.headers);
  if (options.token) {
    headers.set("Authorization", `Bearer ${options.token}`);
  }

  let body = options.body;
  if (body && !(body instanceof FormData) && !(body instanceof Blob) && typeof body !== "string") {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(body);
  }

  const timeoutMs = options.timeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS;
  const controller = new AbortController();
  const abortFromCaller = () => controller.abort();
  if (options.signal?.aborted) {
    controller.abort();
  } else {
    options.signal?.addEventListener("abort", abortFromCaller, { once: true });
  }
  const timeoutHandle = timeoutMs > 0 ? window.setTimeout(() => controller.abort(), timeoutMs) : undefined;
  const { token: _token, body: _body, timeoutMs: _timeoutMs, signal: _signal, ...fetchOptions } = options;

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...fetchOptions,
      headers,
      body: body ?? undefined,
      signal: controller.signal,
    });
  } catch (error) {
    const isAbortError = error instanceof DOMException && error.name === "AbortError";
    throw new ApiError(
      isAbortError ? "请求超时，后端可能仍在处理，请稍后刷新会话查看结果。" : "网络请求失败，请确认前后端服务已启动。",
      0,
      error,
    );
  } finally {
    if (timeoutHandle !== undefined) {
      window.clearTimeout(timeoutHandle);
    }
    options.signal?.removeEventListener("abort", abortFromCaller);
  }

  const contentType = response.headers.get("content-type") ?? "";
  const responseText = response.status === 204 ? "" : await response.text();
  const payload =
    responseText && contentType.includes("application/json") ? JSON.parse(responseText) : responseText;

  if (!response.ok) {
    const detail =
      typeof payload === "object" && payload && "detail" in payload
        ? String((payload as { detail?: unknown }).detail)
        : response.statusText;
    throw new ApiError(detail || "请求失败，请稍后重试。", response.status, payload);
  }

  return payload as T;
}

export const api = {
  login(email: string, password: string) {
    return request<TokenResponse>("/api/v1/auth/login", {
      method: "POST",
      body: { email, password },
    });
  },
  getMe(token: string) {
    return request<UserRead>("/api/v1/auth/me", { token });
  },
  listDocuments(token: string) {
    return request<DocumentRead[]>("/api/v1/documents", { token });
  },
  getDocument(token: string, documentId: string) {
    return request<DocumentRead>(`/api/v1/documents/${documentId}`, { token });
  },
  uploadDocument(token: string, payload: { file: File; title?: string; description?: string; status?: string }) {
    const formData = new FormData();
    formData.set("file", payload.file);
    if (payload.title) formData.set("title", payload.title);
    if (payload.description) formData.set("description", payload.description);
    formData.set("status", payload.status ?? "active");
    return request<DocumentUploadResponse>("/api/v1/documents/upload", {
      method: "POST",
      token,
      body: formData,
    });
  },
  uploadDocumentVersion(token: string, documentId: string, file: File) {
    const formData = new FormData();
    formData.set("file", file);
    return request<DocumentUploadResponse>(`/api/v1/documents/${documentId}/versions/upload`, {
      method: "POST",
      token,
      body: formData,
    });
  },
  ingestDocument(token: string, documentId: string, versionId?: string) {
    return request<IngestionResultRead>(`/api/v1/documents/${documentId}/ingest`, {
      method: "POST",
      token,
      body: versionId ? { version_id: versionId } : {},
    });
  },
  listDocumentVersions(token: string, documentId: string) {
    return request<DocumentVersionRead[]>(`/api/v1/documents/${documentId}/versions`, { token });
  },
  getDocumentVersion(token: string, documentId: string, versionId: string) {
    return request<DocumentVersionRead>(`/api/v1/documents/${documentId}/versions/${versionId}`, { token });
  },
  listDocumentAcl(token: string, documentId: string) {
    return request<DocumentACLRead[]>(`/api/v1/documents/${documentId}/acl`, { token });
  },
  upsertDocumentAcl(
    token: string,
    documentId: string,
    payload: {
      principal_type: string;
      can_view: boolean;
      can_manage: boolean;
      department_id?: string;
      team_name?: string;
      role_name?: string;
      user_id?: string;
    },
  ) {
    return request<DocumentACLRead | null>(`/api/v1/documents/${documentId}/acl`, {
      method: "POST",
      token,
      body: payload,
    });
  },
  deleteDocumentAcl(token: string, documentId: string, aclEntryId: string) {
    return request<void>(`/api/v1/documents/${documentId}/acl/${aclEntryId}`, {
      method: "DELETE",
      token,
    });
  },
  debugDocumentAccess(token: string, documentId: string, userId: string) {
    return request<DocumentAccessDebugRead>(
      `/api/v1/documents/${documentId}/access-debug?user_id=${encodeURIComponent(userId)}`,
      { token },
    );
  },
  listDepartments(token: string) {
    return request<DepartmentRead[]>("/api/v1/departments", { token });
  },
  createDepartment(token: string, payload: { name: string; parent_id?: string | null }) {
    return request<DepartmentRead>("/api/v1/departments", {
      method: "POST",
      token,
      body: payload,
    });
  },
  updateDepartment(token: string, departmentId: string, payload: { name?: string; parent_id?: string | null }) {
    return request<DepartmentRead>(`/api/v1/departments/${departmentId}`, {
      method: "PUT",
      token,
      body: payload,
    });
  },
  deleteDepartment(token: string, departmentId: string) {
    return request<void>(`/api/v1/departments/${departmentId}`, {
      method: "DELETE",
      token,
    });
  },
  listUsers(token: string, filters?: { q?: string; is_active?: boolean | null; limit?: number | null }) {
    const params = new URLSearchParams();
    if (filters?.q?.trim()) {
      params.set("q", filters.q.trim());
    }
    if (filters?.is_active !== undefined && filters.is_active !== null) {
      params.set("is_active", String(filters.is_active));
    }
    if (filters?.limit !== undefined && filters.limit !== null) {
      params.set("limit", String(filters.limit));
    }
    const query = params.toString();
    return request<UserRead[]>(`/api/v1/users${query ? `?${query}` : ""}`, { token });
  },
  createUser(token: string, payload: UserCreatePayload) {
    return request<UserRead>("/api/v1/users", {
      method: "POST",
      token,
      body: payload,
    });
  },
  updateUser(token: string, userId: string, payload: UserUpdatePayload) {
    return request<UserRead>(`/api/v1/users/${userId}`, {
      method: "PUT",
      token,
      body: payload,
    });
  },
  deleteUser(token: string, userId: string) {
    return request<void>(`/api/v1/users/${userId}`, {
      method: "DELETE",
      token,
    });
  },
  updateUserDepartment(token: string, userId: string, departmentId: string | null) {
    return request<UserRead>(`/api/v1/users/${userId}/department`, {
      method: "PATCH",
      token,
      body: { department_id: departmentId },
    });
  },
  listChunks(token: string, documentId: string, versionId?: string) {
    const query = versionId ? `?version_id=${encodeURIComponent(versionId)}` : "";
    return request<ChunkRead[]>(`/api/v1/documents/${documentId}/chunks${query}`, { token });
  },
  getDocumentDiff(token: string, documentId: string, fromVersionId: string, toVersionId: string) {
    return request<DocumentDiffRead>(
      `/api/v1/documents/${documentId}/diff?from_version=${encodeURIComponent(fromVersionId)}&to_version=${encodeURIComponent(toVersionId)}`,
      { token },
    );
  },
  summarizeDocumentDiff(token: string, documentId: string, fromVersionId: string, toVersionId: string, forceRefresh = false) {
    return request<DocumentDiffSummaryRead>(`/api/v1/documents/${documentId}/diff/summary`, {
      method: "POST",
      token,
      body: { from_version_id: fromVersionId, to_version_id: toVersionId, force_refresh: forceRefresh },
    });
  },
  createChatSession(token: string, title?: string) {
    return request<ChatSessionRead>("/api/v1/chat/sessions", {
      method: "POST",
      token,
      body: title ? { title } : {},
    });
  },
  listChatSessions(token: string) {
    return request<ChatSessionRead[]>("/api/v1/chat/sessions", { token });
  },
  getChatSession(token: string, sessionId: string) {
    return request<ChatSessionDetailRead>(`/api/v1/chat/sessions/${sessionId}`, { token });
  },
  deleteChatSession(token: string, sessionId: string) {
    return request<void>(`/api/v1/chat/sessions/${sessionId}`, {
      method: "DELETE",
      token,
    });
  },
  sendChatMessage(token: string, sessionId: string, content: string, topK = 5, clientRequestId?: string | null) {
    return request<ChatMessageCreateResponse>(`/api/v1/chat/sessions/${sessionId}/messages`, {
      method: "POST",
      token,
      body: { content, top_k: topK, client_request_id: clientRequestId ?? undefined },
      timeoutMs: CHAT_REQUEST_TIMEOUT_MS,
    });
  },
  extractTasks(token: string, sessionId: string, maxItems = 8) {
    return request<TaskExtractResponse>("/api/v1/tasks/extract", {
      method: "POST",
      token,
      body: { session_id: sessionId, max_items: maxItems },
    });
  },
  listTasks(token: string) {
    return request<TaskItemRead[]>("/api/v1/tasks", { token });
  },
  generateWeeklyReport(token: string, sessionId: string, title?: string) {
    return request<WeeklyReportGenerateResponse>("/api/v1/reports/weekly", {
      method: "POST",
      token,
      body: { session_id: sessionId, title },
    });
  },
  listReports(token: string) {
    return request<WeeklyReportDraftRead[]>("/api/v1/reports", { token });
  },
  generateFaqs(token: string, sessionId: string, maxEntries = 5) {
    return request<FAQGenerateResponse>("/api/v1/faqs/generate", {
      method: "POST",
      token,
      body: { session_id: sessionId, max_entries: maxEntries },
    });
  },
  listFaqs(token: string) {
    return request<FAQEntryRead[]>("/api/v1/faqs", { token });
  },
  listEvalDatasets(token: string) {
    return request<EvalDatasetRead[]>("/api/v1/eval/datasets", { token });
  },
  getEvalDashboard(token: string, datasetName?: string | null, limit = 8) {
    const params = new URLSearchParams({ limit: String(limit) });
    if (datasetName) {
      params.set("dataset_name", datasetName);
    }
    return request<EvalDashboardRead>(`/api/v1/eval/dashboard?${params.toString()}`, { token });
  },
  runEval(
    token: string,
    payload?: { dataset_name?: string; top_k?: number; seed_demo_cases?: boolean; client_request_id?: string },
  ) {
    return request<EvalRunDetailRead>("/api/v1/eval/run/async", {
      method: "POST",
      token,
      body: {
        dataset_name: payload?.dataset_name ?? "demo_permission_eval",
        top_k: payload?.top_k ?? 5,
        seed_demo_cases: payload?.seed_demo_cases ?? true,
        client_request_id: payload?.client_request_id ?? undefined,
      },
      timeoutMs: EVAL_REQUEST_TIMEOUT_MS,
    });
  },
  listEvalRuns(token: string) {
    return request<EvalRunRead[]>("/api/v1/eval/runs", { token });
  },
  getEvalRun(token: string, runId: string) {
    return request<EvalRunDetailRead>(`/api/v1/eval/runs/${runId}`, { token });
  },
  listTraces(token: string, sessionId?: string) {
    const query = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "";
    return request<TraceLogRead[]>(`/api/v1/observability/traces${query}`, { token });
  },
  getTrace(token: string, traceId: string) {
    return request<TraceLogRead>(`/api/v1/observability/traces/${traceId}`, { token });
  },
};
