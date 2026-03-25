import type {
  ChatMessageCreateResponse,
  ChatSessionDetailRead,
  ChatSessionRead,
  ChunkRead,
  DocumentACLRead,
  DocumentDiffRead,
  DocumentDiffSummaryRead,
  DocumentRead,
  DocumentUploadResponse,
  DocumentVersionRead,
  EvalRunDetailRead,
  EvalRunRead,
  FAQEntryRead,
  FAQGenerateResponse,
  IngestionResultRead,
  TaskExtractResponse,
  TaskItemRead,
  TokenResponse,
  TraceLogRead,
  UserRead,
  WeeklyReportDraftRead,
  WeeklyReportGenerateResponse,
} from "../types/api";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "";

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
  body?: BodyInit | Record<string, unknown> | null;
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

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
    body: body ?? undefined,
  });

  const contentType = response.headers.get("content-type") ?? "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();

  if (!response.ok) {
    const detail =
      typeof payload === "object" && payload && "detail" in payload
        ? String((payload as { detail?: unknown }).detail)
        : response.statusText;
    throw new ApiError(detail || "Request failed", response.status, payload);
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
      team_name?: string;
      role_name?: string;
      user_id?: string;
    },
  ) {
    return request<DocumentACLRead>(`/api/v1/documents/${documentId}/acl`, {
      method: "POST",
      token,
      body: payload,
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
  summarizeDocumentDiff(token: string, documentId: string, fromVersionId: string, toVersionId: string) {
    return request<DocumentDiffSummaryRead>(`/api/v1/documents/${documentId}/diff/summary`, {
      method: "POST",
      token,
      body: { from_version_id: fromVersionId, to_version_id: toVersionId },
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
  sendChatMessage(token: string, sessionId: string, content: string, topK = 5) {
    return request<ChatMessageCreateResponse>(`/api/v1/chat/sessions/${sessionId}/messages`, {
      method: "POST",
      token,
      body: { content, top_k: topK },
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
  runEval(token: string, payload?: { dataset_name?: string; top_k?: number; seed_demo_cases?: boolean }) {
    return request<EvalRunDetailRead>("/api/v1/eval/run", {
      method: "POST",
      token,
      body: {
        dataset_name: payload?.dataset_name ?? "demo_permission_eval",
        top_k: payload?.top_k ?? 5,
        seed_demo_cases: payload?.seed_demo_cases ?? true,
      },
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
