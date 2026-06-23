import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { CitationList } from "../components/CitationList";
import { ErrorNotice } from "../components/ErrorNotice";
import { ExecutionTrace } from "../components/ExecutionTrace";
import { PageHeader } from "../components/PageHeader";
import { StatusBadge } from "../components/StatusBadge";
import { useAppContext } from "../context/AppContext";
import { ApiError, api } from "../lib/api";
import { formatArtifactType, formatConfidence, formatCopilotIntent, formatMessageRole, formatRefusalReason } from "../lib/display";
import { formatDateTime, locationLabel } from "../lib/format";
import {
  createPendingChatOperation,
  listPendingChatOperations,
  removePendingChatOperation,
  touchPendingChatOperation,
} from "../lib/pendingOperations";
import type {
  AgentRunTraceRead,
  AgentStepRead,
  AnswerClaimSupportRead,
  AnswerEvidenceAuditRead,
  ChatCitationRead,
  ChatMessageRead,
  ChatSessionDetailRead,
  ChatSessionRead,
  SearchDebugInfo,
} from "../types/api";

const GENERIC_SESSION_TITLES = new Set(["新会话", "New Chat"]);

export function ChatPage() {
  const { token, selectedSessionId, setSelectedSessionId } = useAppContext();
  const [sessions, setSessions] = useState<ChatSessionRead[]>([]);
  const [activeSession, setActiveSession] = useState<ChatSessionDetailRead | null>(null);
  const [messageDraft, setMessageDraft] = useState("");
  const [pendingSubmission, setPendingSubmission] = useState<{ sessionId: string; content: string; clientRequestId: string } | null>(null);
  const [selectedCitation, setSelectedCitation] = useState<ChatCitationRead | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [creatingSession, setCreatingSession] = useState(false);
  const [deletingSessionId, setDeletingSessionId] = useState<string | null>(null);
  const [artifactMessage, setArtifactMessage] = useState<string | null>(null);
  const threadRef = useRef<HTMLDivElement | null>(null);
  const sessionButtonRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  const selectedSessionIdRef = useRef<string | null>(selectedSessionId);
  const sessionListRequestRef = useRef(0);
  const sessionDetailRequestRef = useRef(0);
  selectedSessionIdRef.current = selectedSessionId;

  const loadSessions = useCallback(
    async (preferredSessionId?: string | null) => {
      if (!token) {
        setSessions([]);
        return null;
      }
      const requestId = ++sessionListRequestRef.current;
      const items = await api.listChatSessions(token);
      if (requestId !== sessionListRequestRef.current) {
        return null;
      }
      setSessions(items);
      const currentPreferredId = preferredSessionId ?? selectedSessionIdRef.current;
      const nextSessionId =
        currentPreferredId && items.some((item) => item.id === currentPreferredId)
          ? currentPreferredId
          : items[0]?.id ?? null;
      if (nextSessionId !== selectedSessionIdRef.current) {
        selectedSessionIdRef.current = nextSessionId;
        setSelectedSessionId(nextSessionId);
      }
      return items;
    },
    [setSelectedSessionId, token],
  );
  const recoverPendingChatRequests = useCallback(async () => {
    if (!token) {
      return;
    }
    const operations = listPendingChatOperations();
    if (!operations.length) {
      return;
    }

    for (const operation of operations) {
      try {
        const beforeSession = await api.getChatSession(token, operation.sessionId);
        if (chatSessionHasCompletedClientRequest(beforeSession, operation.id)) {
          removePendingChatOperation(operation.id);
          if (selectedSessionIdRef.current === operation.sessionId) {
            setActiveSession(beforeSession);
          }
          continue;
        }

        if (selectedSessionIdRef.current === operation.sessionId) {
          setPendingSubmission({
            sessionId: operation.sessionId,
            content: operation.content,
            clientRequestId: operation.id,
          });
          setActiveSession(beforeSession);
        }

        try {
          await api.sendChatMessage(token, operation.sessionId, operation.content, operation.topK, operation.id);
        } catch (nextError) {
          if (nextError instanceof ApiError && nextError.status === 409) {
            touchPendingChatOperation(operation.id, nextError.message);
            continue;
          }
          throw nextError;
        }

        const afterSession = await api.getChatSession(token, operation.sessionId);
        if (chatSessionHasCompletedClientRequest(afterSession, operation.id)) {
          removePendingChatOperation(operation.id);
          setPendingSubmission((current) => (current?.clientRequestId === operation.id ? null : current));
          if (selectedSessionIdRef.current === operation.sessionId) {
            setActiveSession(afterSession);
            const latestCitation = [...afterSession.messages]
              .reverse()
              .flatMap((message) => message.citations)[0] ?? null;
            setSelectedCitation(latestCitation);
          }
        } else {
          touchPendingChatOperation(operation.id);
        }
      } catch (nextError) {
        touchPendingChatOperation(operation.id, nextError instanceof Error ? nextError.message : "恢复请求失败。");
      }
    }

    await loadSessions(selectedSessionIdRef.current);
  }, [loadSessions, token]);

  useEffect(() => {
    void recoverPendingChatRequests();
  }, [recoverPendingChatRequests]);

  useEffect(() => {
    if (!token) {
      return;
    }
    setLoading(true);
    loadSessions()
      .then((items) => {
        if (items && !items.length) {
          setActiveSession(null);
        }
      })
      .catch((nextError) => setError(nextError instanceof Error ? nextError.message : "加载会话列表失败。"))
      .finally(() => setLoading(false));
  }, [loadSessions, token]);

  useEffect(() => {
    if (!token || !selectedSessionId) {
      setActiveSession(null);
      setSelectedCitation(null);
      return;
    }
    let cancelled = false;
    const sessionId = selectedSessionId;
    const requestId = ++sessionDetailRequestRef.current;
    setSelectedCitation(null);
    api
      .getChatSession(token, sessionId)
      .then((session) => {
        if (
          cancelled ||
          requestId !== sessionDetailRequestRef.current ||
          session.id !== sessionId ||
          selectedSessionIdRef.current !== sessionId
        ) {
          return;
        }
        setActiveSession(session);
        const latestCitation = [...session.messages]
          .reverse()
          .flatMap((message) => message.citations)[0] ?? null;
        setSelectedCitation(latestCitation);
      })
      .catch((nextError) => setError(nextError instanceof Error ? nextError.message : "加载会话详情失败。"));
    return () => {
      cancelled = true;
    };
  }, [selectedSessionId, token]);

  useEffect(() => {
    if (!selectedSessionId) {
      return;
    }
    const nextFrame = window.requestAnimationFrame(() => {
      sessionButtonRefs.current[selectedSessionId]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    });
    return () => window.cancelAnimationFrame(nextFrame);
  }, [selectedSessionId, sessions]);

  useEffect(() => {
    const hasPendingInActiveSession = Boolean(
      pendingSubmission && activeSession?.id && pendingSubmission.sessionId === activeSession.id,
    );
    if ((!activeSession?.messages.length && !hasPendingInActiveSession) || !threadRef.current) {
      return;
    }
    const nextFrame = window.requestAnimationFrame(() => {
      const thread = threadRef.current;
      if (!thread) {
        return;
      }
      thread.scrollTo({ top: thread.scrollHeight, behavior: "smooth" });
    });
    return () => window.cancelAnimationFrame(nextFrame);
  }, [activeSession?.id, activeSession?.messages.length, pendingSubmission]);

  async function handleCreateSession() {
    if (!token || creatingSession || sending || deletingSessionId) {
      return;
    }
    setCreatingSession(true);
    try {
      const session = await api.createChatSession(token);
      sessionListRequestRef.current += 1;
      sessionDetailRequestRef.current += 1;
      setSessions((current) => [session, ...current.filter((item) => item.id !== session.id)]);
      setActiveSession({ ...session, messages: [] });
      selectedSessionIdRef.current = session.id;
      setSelectedSessionId(session.id);
      setSelectedCitation(null);
      setArtifactMessage(null);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "创建会话失败。");
    } finally {
      setCreatingSession(false);
    }
  }

  async function handleSendMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (
      !token ||
      !selectedSessionId ||
      !messageDraft.trim() ||
      creatingSession ||
      deletingSessionId ||
      pendingSubmission
    ) {
      return;
    }
    const outgoingContent = messageDraft.trim();
    const targetSessionId = selectedSessionId;
    const pendingOperation = createPendingChatOperation({ sessionId: targetSessionId, content: outgoingContent, topK: 5 });
    let keepPendingOperation = false;
    setMessageDraft("");
    setPendingSubmission({ sessionId: targetSessionId, content: outgoingContent, clientRequestId: pendingOperation.id });
    setSelectedCitation(null);
    setSending(true);
    setError(null);
    try {
      await api.sendChatMessage(token, targetSessionId, outgoingContent, 5, pendingOperation.id);
      const refreshedSession = await api.getChatSession(token, targetSessionId);
      removePendingChatOperation(pendingOperation.id);
      setActiveSession((current) => {
        if (selectedSessionIdRef.current !== targetSessionId) {
          return current;
        }
        return refreshedSession;
      });
      if (selectedSessionIdRef.current === targetSessionId) {
        const latestCitation =
          [...refreshedSession.messages]
            .reverse()
            .flatMap((message) => message.citations)[0] ?? null;
        setSelectedCitation(latestCitation);
      }
      await loadSessions(targetSessionId);
    } catch (nextError) {
      keepPendingOperation = nextError instanceof ApiError && (nextError.status === 0 || nextError.status === 409);
      if (keepPendingOperation) {
        touchPendingChatOperation(pendingOperation.id, nextError instanceof Error ? nextError.message : undefined);
        setError("请求仍在后台处理，刷新或切换页面后会自动恢复结果。");
      } else {
        removePendingChatOperation(pendingOperation.id);
        setMessageDraft(outgoingContent);
        setError(nextError instanceof Error ? nextError.message : "发送问题失败。");
      }
    } finally {
      if (!keepPendingOperation) {
        setPendingSubmission((current) => (current?.clientRequestId === pendingOperation.id ? null : current));
      }
      setSending(false);
    }
  }
  async function handleArtifactAction(action: "tasks" | "report" | "faq") {
    if (!token || !selectedSessionId || deletingSessionId) {
      return;
    }
    setArtifactMessage(null);
    try {
      if (action === "tasks") {
        const response = await api.extractTasks(token, selectedSessionId);
        setArtifactMessage(`已生成 ${response.items.length} 条待办，可前往“派生结果”查看。`);
      } else if (action === "report") {
        const response = await api.generateWeeklyReport(token, selectedSessionId, "周报草稿");
        setArtifactMessage(`已生成周报草稿：${response.report.title}`);
      } else {
        const response = await api.generateFaqs(token, selectedSessionId);
        setArtifactMessage(`已生成 ${response.entries.length} 条 FAQ 草稿。`);
      }
    } catch (nextError) {
      setArtifactMessage(nextError instanceof Error ? nextError.message : "生成派生结果失败。");
    }
  }

  async function handleDeleteSession(sessionId: string) {
    if (!token || sending || creatingSession || deletingSessionId) {
      return;
    }
    const targetSession = sessions.find((item) => item.id === sessionId);
    const sessionLabel = targetSession?.display_title || targetSession?.title || "这个会话";
    if (!window.confirm(`确定删除“${sessionLabel}”吗？该会话下的问答记录和引用会一起移除。`)) {
      return;
    }

    const remainingSessions = sessions.filter((item) => item.id !== sessionId);
    const nextSessionId = selectedSessionId === sessionId ? (remainingSessions[0]?.id ?? null) : selectedSessionId;

    setDeletingSessionId(sessionId);
    setError(null);
    try {
      await api.deleteChatSession(token, sessionId);
      setSessions(remainingSessions);
      sessionDetailRequestRef.current += 1;
      if (selectedSessionId === sessionId) {
        selectedSessionIdRef.current = nextSessionId;
        setSelectedSessionId(nextSessionId);
        setSelectedCitation(null);
        setActiveSession(null);
      }
      await loadSessions(nextSessionId);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "删除会话失败。");
      await loadSessions(selectedSessionIdRef.current);
    } finally {
      setDeletingSessionId(null);
    }
  }

  const pendingSessionId = pendingSubmission?.sessionId ?? null;
  const pendingClientRequestId = pendingSubmission?.clientRequestId ?? null;
  const selectedSession = selectedSessionId ? sessions.find((item) => item.id === selectedSessionId) ?? null : null;
  const activeMessages =
    activeSession && (!selectedSessionId || activeSession.id === selectedSessionId) ? activeSession.messages : [];
  const pendingUserAlreadyVisible = Boolean(
    pendingClientRequestId &&
      activeMessages.some((message) => message.role === "user" && messageClientRequestId(message) === pendingClientRequestId),
  );
  const visibleSessionTitle =
    selectedSession?.display_title ??
    (activeSession && (!selectedSessionId || activeSession.id === selectedSessionId) ? activeSession.display_title : null) ??
    "当前会话";
  const pendingDraftContent = pendingSubmission?.content.trim() ?? "";
  const pendingMatchesSelectedSession = Boolean(
    pendingSessionId && selectedSessionId && pendingSessionId === selectedSessionId,
  );
  const pendingMatchesActiveSession = Boolean(
    pendingSessionId && activeSession?.id && pendingSessionId === activeSession.id,
  );
  const pendingFallbackForFreshSession = Boolean(
    pendingSessionId &&
      activeSession?.id === pendingSessionId &&
      activeMessages.length === 0 &&
      GENERIC_SESSION_TITLES.has(activeSession.title) &&
      pendingDraftContent,
  );
  const visibleCitations = resolveVisibleCitations(
    activeSession,
    selectedCitation,
    pendingSessionId,
    selectedSessionId,
  );
  const displayCitation = selectedCitation ?? visibleCitations[0] ?? null;
  const showPending = Boolean(
    pendingDraftContent && (pendingMatchesSelectedSession || pendingMatchesActiveSession || pendingFallbackForFreshSession),
  );

  return (
    <div className="page-stack">
      <PageHeader
        title="引用式问答"
        description="围绕可访问的企业文档发起 RAG 问答，查看引用来源，并沉淀为待办、周报或 FAQ。"
        actions={
          <div className="inline-actions">
            <button className="secondary-button" disabled={creatingSession || sending || Boolean(deletingSessionId)} onClick={handleCreateSession} type="button">
              {creatingSession ? "创建中..." : "新建会话"}
            </button>
            <Link className="secondary-button link-button" to="/artifacts">
              查看派生结果
            </Link>
          </div>
        }
      />
      <ErrorNotice message={error} />
      {artifactMessage ? <div className="info-block">{artifactMessage}</div> : null}

      <div className="page-grid chat-layout">
        <section className="panel stack session-panel">
          <div className="panel-header">
            <div className="panel-heading">
              <h3>会话列表</h3>
              <p>按最近更新时间展示会话，便于快速回到刚才的问答上下文。</p>
            </div>
            <StatusBadge tone="neutral">{sessions.length}</StatusBadge>
          </div>
          {loading ? <p className="muted">正在加载会话...</p> : null}
          <div className="session-list">
            {sessions.length ? (
              sessions.map((session) => (
                <div
                  key={session.id}
                  className={`list-card session-list-card ${selectedSessionId === session.id ? "is-selected" : ""}`}
                >
                  <button
                    className="session-select-button"
                    onClick={() => setSelectedSessionId(session.id)}
                    disabled={creatingSession || sending || Boolean(deletingSessionId)}
                    ref={(element) => {
                      sessionButtonRefs.current[session.id] = element;
                    }}
                    type="button"
                  >
                    <div className="list-card-topline">
                      <span className="session-card-title">{session.display_title}</span>
                    </div>
                    <p>{formatDateTime(session.updated_at)}</p>
                  </button>
                  <button
                    className="session-delete-button"
                    disabled={creatingSession || sending || Boolean(deletingSessionId)}
                    onClick={() => handleDeleteSession(session.id)}
                    aria-label={`删除会话：${session.display_title}`}
                    title={`删除会话：${session.display_title}`}
                    type="button"
                  >
                    {deletingSessionId === session.id ? "..." : "×"}
                  </button>
                </div>
              ))
            ) : (
              <p className="muted">暂无会话，点击右上角“新建会话”开始提问。</p>
            )}
          </div>
        </section>

        <section className="panel stack chat-conversation-panel">
          <div className="panel-header">
            <div className="panel-heading">
              <h3>{visibleSessionTitle}</h3>
              <p>仅基于检索证据作答；证据不足时会提示并保留引用来源。</p>
            </div>
            {selectedSessionId ? <StatusBadge tone="info">会话已关联</StatusBadge> : null}
          </div>
          <div className="chat-thread" ref={threadRef}>
            {activeMessages.length || showPending ? (
              <>
                {activeMessages.map((message) => (
                  <MessageBubble
                    key={message.id}
                    message={message}
                    onSelectCitation={(citation) => setSelectedCitation(citation)}
                  />
                ))}
                {showPending && !pendingUserAlreadyVisible ? (
                  <article className="message-bubble user is-pending">
                    <div className="message-meta">
                      <strong>用户</strong>
                      <span>正在发送...</span>
                    </div>
                    <p>{pendingDraftContent}</p>
                  </article>
                ) : null}
                {showPending ? (
                  <article className="message-bubble assistant is-pending">
                    <div className="message-meta">
                      <strong>助手</strong>
                      <span>正在整理引用并生成回答...</span>
                    </div>
                    <p>系统正在检索可访问文档并组织答案，请稍候。</p>
                  </article>
                ) : null}
              </>
            ) : (
              <div className="empty-state">
                <strong>暂无消息</strong>
                <p>先新建一个会话，再围绕你可访问的文档开始提问。</p>
              </div>
            )}
          </div>
          <form className="chat-composer" onSubmit={handleSendMessage}>
            <textarea
              placeholder="请输入一个基于内部文档的引用式问题..."
              rows={4}
              value={sending ? "" : messageDraft}
              onChange={(event) => setMessageDraft(event.target.value)}
              disabled={sending || creatingSession || Boolean(deletingSessionId)}
            />
            <div className="composer-actions">
              <div className="inline-actions">
                <button className="secondary-button" disabled={Boolean(deletingSessionId)} onClick={() => handleArtifactAction("tasks")} type="button">
                  提取待办
                </button>
                <button className="secondary-button" disabled={Boolean(deletingSessionId)} onClick={() => handleArtifactAction("report")} type="button">
                  周报草稿
                </button>
                <button className="secondary-button" disabled={Boolean(deletingSessionId)} onClick={() => handleArtifactAction("faq")} type="button">
                  FAQ 草稿
                </button>
              </div>
              <button className="primary-button" disabled={!selectedSessionId || sending || creatingSession || Boolean(deletingSessionId)} type="submit">
                {creatingSession ? "创建中..." : sending ? "回答中..." : "发送"}
              </button>
            </div>
          </form>
        </section>

        <div className="stack chat-side-panel">
          <CitationList
            citations={visibleCitations}
            onSelect={(citation) => setSelectedCitation(citation as ChatCitationRead)}
            selectedCitationId={displayCitation?.id}
            title="引用来源"
          />
          <section className="panel stack">
            <div className="panel-header">
              <div className="panel-heading">
                <h3>来源片段</h3>
                <p>展示当前选中引用的片段摘要，并支持跳转到文档页查看完整内容。</p>
              </div>
              {displayCitation ? (
                <Link className="secondary-button link-button" to={buildCitationDocumentLink(displayCitation)}>
                  查看完整文档
                </Link>
              ) : null}
            </div>
            {displayCitation ? (
              <>
                <div className="list-card-topline">
                  <strong>{displayCitation.document_title}</strong>
                  <span>
                    v{displayCitation.version_number} · {locationLabel(displayCitation)}
                  </span>
                </div>
                <p>{displayCitation.preview}</p>
                <div className="metadata-grid">
                  <span>分块 ID：{displayCitation.chunk_id ?? "-"}</span>
                  <span>融合分数：{displayCitation.fused_score?.toFixed(3) ?? "-"}</span>
                </div>
              </>
            ) : (
              <p className="muted">点击左侧引用可查看来源片段。</p>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}

interface MessageDebugInfo {
  intent: string | null;
  targetDocument: string | null;
  refusalReason: string | null;
  artifactType: string | null;
}

function resolveVisibleCitations(
  session: ChatSessionDetailRead | null,
  selectedCitation: ChatCitationRead | null,
  pendingSessionId: string | null,
  selectedSessionId: string | null,
): ChatCitationRead[] {
  if (pendingSessionId && selectedSessionId && pendingSessionId === selectedSessionId) {
    return [];
  }

  if (!session) {
    return [];
  }

  if (pendingSessionId && session.id === pendingSessionId) {
    return [];
  }

  if (selectedCitation) {
    const selectedMessage = session.messages.find((message) => message.id === selectedCitation.message_id);
    if (selectedMessage?.citations.length) {
      return dedupeCitations(selectedMessage.citations);
    }
  }

  const latestMessageWithCitations = [...session.messages]
    .reverse()
    .find((message) => message.citations.length);
  return latestMessageWithCitations ? dedupeCitations(latestMessageWithCitations.citations) : [];
}

function dedupeCitations(citations: ChatCitationRead[]): ChatCitationRead[] {
  const seen = new Set<string>();
  const items: ChatCitationRead[] = [];
  for (const citation of citations) {
    const key = citation.chunk_id
      ? `chunk:${citation.chunk_id}`
      : [
          citation.document_id,
          citation.document_version_id,
          citation.chunk_index ?? "",
          citation.page_number_start ?? "",
          citation.paragraph_start ?? "",
          citation.preview,
        ].join("|");
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    items.push(citation);
  }
  return items;
}

function readAgentSteps(message: ChatMessageRead): AgentStepRead[] {
  const metadata = message.message_metadata;
  if (!metadata || typeof metadata !== "object") {
    return [];
  }
  const steps = (metadata as Record<string, unknown>).agent_steps;
  if (!Array.isArray(steps)) {
    return [];
  }
  return steps.filter((item): item is AgentStepRead => {
    return Boolean(
      item &&
      typeof item === "object" &&
      typeof (item as Record<string, unknown>).name === "string" &&
      typeof (item as Record<string, unknown>).input_summary === "string" &&
      typeof (item as Record<string, unknown>).output_summary === "string",
    );
  });
}

function readAgentRunTrace(message: ChatMessageRead): AgentRunTraceRead | null {
  const metadata = message.message_metadata;
  if (!metadata || typeof metadata !== "object") {
    return null;
  }
  const trace = (metadata as Record<string, unknown>).agent_run_trace;
  if (!trace || typeof trace !== "object") {
    return null;
  }
  const traceRecord = trace as Record<string, unknown>;
  if (!traceRecord.tool_plan || !Array.isArray(traceRecord.actions) || !Array.isArray(traceRecord.observations)) {
    return null;
  }
  return trace as AgentRunTraceRead;
}

function readRetrievalDebug(message: ChatMessageRead): SearchDebugInfo | null {
  const metadata = message.message_metadata;
  if (!metadata || typeof metadata !== "object") {
    return null;
  }
  const retrievalDebug = (metadata as Record<string, unknown>).retrieval_debug;
  if (!retrievalDebug || typeof retrievalDebug !== "object") {
    return null;
  }
  return retrievalDebug as SearchDebugInfo;
}

function messageClientRequestId(message: ChatMessageRead): string | null {
  const metadata = message.message_metadata;
  if (!metadata || typeof metadata !== "object") {
    return null;
  }
  const value = (metadata as Record<string, unknown>).client_request_id;
  return typeof value === "string" ? value : null;
}

function chatSessionHasCompletedClientRequest(session: ChatSessionDetailRead, clientRequestId: string): boolean {
  return session.messages.some(
    (message) => message.role === "assistant" && messageClientRequestId(message) === clientRequestId,
  );
}
function readEvidenceAudit(message: ChatMessageRead): AnswerEvidenceAuditRead | null {
  const metadata = message.message_metadata;
  if (!metadata || typeof metadata !== "object") {
    return null;
  }
  const audit = (metadata as Record<string, unknown>).evidence_audit;
  if (!audit || typeof audit !== "object") {
    return null;
  }
  const auditRecord = audit as Record<string, unknown>;
  if (typeof auditRecord.claim_count !== "number" || !Array.isArray(auditRecord.claims)) {
    return null;
  }
  return audit as AnswerEvidenceAuditRead;
}

function evidenceAuditTone(status: string): "neutral" | "success" | "warning" | "danger" | "info" {
  if (status === "supported") {
    return "success";
  }
  if (status === "partial") {
    return "warning";
  }
  if (status === "needs_review") {
    return "danger";
  }
  return "neutral";
}

function evidenceAuditLabel(status: string): string {
  if (status === "supported") {
    return "已支撑";
  }
  if (status === "partial") {
    return "部分支撑";
  }
  if (status === "needs_review") {
    return "待核实";
  }
  return "未抽取";
}

function claimSupportTone(status: string): "neutral" | "success" | "warning" | "danger" | "info" {
  if (status === "supported") {
    return "success";
  }
  if (status === "partial") {
    return "warning";
  }
  return "danger";
}

function claimSupportLabel(status: string): string {
  if (status === "supported") {
    return "已支撑";
  }
  if (status === "partial") {
    return "部分支撑";
  }
  return "待核实";
}

function evidenceScoreText(value: number | null): string {
  if (typeof value !== "number") {
    return "-";
  }
  return `${Math.round(value * 100)}%`;
}

function resolveSupportCitation(
  support: AnswerClaimSupportRead["support_citations"][number],
  citations: ChatCitationRead[],
): ChatCitationRead | null {
  if (support.chunk_id) {
    const byChunk = citations.find((citation) => citation.chunk_id === support.chunk_id);
    if (byChunk) {
      return byChunk;
    }
  }
  return citations.find((citation) => citation.rank === support.rank) ?? null;
}

function readOriginalSearchQuery(message: ChatMessageRead): string | null {
  const metadata = message.message_metadata;
  if (!metadata || typeof metadata !== "object") {
    return null;
  }
  const toolExecution = (metadata as Record<string, unknown>).tool_execution;
  if (!toolExecution || typeof toolExecution !== "object") {
    return null;
  }
  const toolInput = (toolExecution as Record<string, unknown>).tool_input;
  if (!toolInput || typeof toolInput !== "object") {
    return null;
  }
  const query = (toolInput as Record<string, unknown>).query;
  return typeof query === "string" && query.trim() ? query : null;
}

function readMessageDebugInfo(message: ChatMessageRead): MessageDebugInfo | null {
  const metadata = message.message_metadata;
  if (!metadata || typeof metadata !== "object") {
    return null;
  }
  const metadataRecord = metadata as Record<string, unknown>;
  const routerDecision = metadataRecord.router_decision;
  const structuredResult = metadataRecord.structured_result;
  const routerRecord = routerDecision && typeof routerDecision === "object" ? (routerDecision as Record<string, unknown>) : null;
  const structuredRecord = structuredResult && typeof structuredResult === "object" ? (structuredResult as Record<string, unknown>) : null;
  if (!routerRecord && !structuredRecord) {
    return null;
  }
  return {
    intent: typeof routerRecord?.intent === "string" ? routerRecord.intent : null,
    targetDocument:
      typeof structuredRecord?.target_document === "string"
        ? structuredRecord.target_document
        : typeof routerRecord?.target_document_title === "string"
          ? routerRecord.target_document_title
          : null,
    refusalReason: typeof structuredRecord?.refusal_reason === "string" ? structuredRecord.refusal_reason : null,
    artifactType: typeof structuredRecord?.artifact_type === "string" ? structuredRecord.artifact_type : null,
  };
}
function buildCitationDocumentLink(citation: ChatCitationRead): string {
  const params = new URLSearchParams({
    documentId: citation.document_id,
    versionId: citation.document_version_id,
  });
  if (citation.chunk_id) {
    params.set("chunkId", citation.chunk_id);
  }
  return `/documents?${params.toString()}`;
}

interface EvidenceAuditPanelProps {
  audit: AnswerEvidenceAuditRead;
  citations: ChatCitationRead[];
  onSelectCitation: (citation: ChatCitationRead) => void;
}

function EvidenceAuditPanel({ audit, citations, onSelectCitation }: EvidenceAuditPanelProps) {
  const [showDetails, setShowDetails] = useState(false);
  const reviewClaims = audit.claims.filter((claim) => claim.support_status !== "supported");
  const visibleClaims = showDetails ? audit.claims : reviewClaims.slice(0, 4);
  const hasHiddenReviewClaims = !showDetails && reviewClaims.length > visibleClaims.length;
  const shouldShowClaimList = showDetails || reviewClaims.length > 0;
  const detailLabel = showDetails ? "收起明细" : reviewClaims.length ? "查看全部" : "查看明细";

  return (
    <div className="evidence-audit">
      <div className="evidence-audit-summary">
        <div>
          <strong>证据核验</strong>
          <span>
            已支撑 {audit.supported_count}/{audit.claim_count} · {evidenceScoreText(audit.score)}
          </span>
        </div>
        <div className="evidence-audit-actions">
          <StatusBadge tone={evidenceAuditTone(audit.status)}>{evidenceAuditLabel(audit.status)}</StatusBadge>
          <button className="evidence-audit-toggle" onClick={() => setShowDetails((current) => !current)} type="button">
            {detailLabel}
          </button>
        </div>
      </div>
      {shouldShowClaimList ? (
        <div className="evidence-claim-list">
          {visibleClaims.map((claim) => {
            const linkedCitations = claim.support_citations
              .map((support) => resolveSupportCitation(support, citations))
              .filter((citation): citation is ChatCitationRead => Boolean(citation));
            return (
              <div className="evidence-claim" key={`${claim.index}-${claim.text}`}>
                <div className="evidence-claim-topline">
                  <span>事实 {claim.index}</span>
                  <StatusBadge tone={claimSupportTone(claim.support_status)}>
                    {claimSupportLabel(claim.support_status)} · {evidenceScoreText(claim.support_score)}
                  </StatusBadge>
                </div>
                <p className="evidence-claim-text">{claim.text}</p>
                <div className="evidence-claim-citations">
                  {linkedCitations.length ? (
                    linkedCitations.map((citation) => (
                      <button
                        className="evidence-citation-link"
                        key={citation.id}
                        onClick={() => onSelectCitation(citation)}
                        type="button"
                      >
                        引用 {citation.rank} · {citation.document_title}
                      </button>
                    ))
                  ) : (
                    <span className="muted">未关联引用</span>
                  )}
                </div>
              </div>
            );
          })}
          {hasHiddenReviewClaims ? <p className="evidence-audit-note">还有 {reviewClaims.length - visibleClaims.length} 条待核验证据，展开后查看。</p> : null}
        </div>
      ) : (
        <p className="evidence-audit-note">答案中的事实均已关联引用，明细已收起。</p>
      )}
    </div>
  );
}

interface MessageBubbleProps {
  message: ChatMessageRead;
  onSelectCitation: (citation: ChatCitationRead) => void;
}

function MessageBubble({ message, onSelectCitation }: MessageBubbleProps) {
  const tone = message.role === "assistant" ? "assistant" : "user";
  const debugInfo = message.role === "assistant" ? readMessageDebugInfo(message) : null;
  const agentSteps = message.role === "assistant" ? readAgentSteps(message) : [];
  const agentRunTrace = message.role === "assistant" ? readAgentRunTrace(message) : null;
  const retrievalDebug = message.role === "assistant" ? readRetrievalDebug(message) : null;
  const originalSearchQuery = message.role === "assistant" ? readOriginalSearchQuery(message) : null;
  const evidenceAudit = message.role === "assistant" ? readEvidenceAudit(message) : null;
  return (
    <article className={`message-bubble ${tone}`}>
      <div className="message-meta">
        <strong>{formatMessageRole(message.role)}</strong>
        <span>{formatDateTime(message.created_at)}</span>
        {message.confidence ? (
          <StatusBadge tone={message.insufficient_evidence ? "warning" : "success"}>
            {formatConfidence(message.confidence)}
          </StatusBadge>
        ) : null}
      </div>
      <p>{message.content}</p>
      {evidenceAudit && evidenceAudit.claim_count > 0 ? (
        <EvidenceAuditPanel audit={evidenceAudit} citations={message.citations} onSelectCitation={onSelectCitation} />
      ) : null}
      {debugInfo ? (
        <div className="metadata-grid muted">
          <span>意图：{formatCopilotIntent(debugInfo.intent)}</span>
          {debugInfo.targetDocument ? <span>目标文档：{debugInfo.targetDocument}</span> : null}
          {debugInfo.artifactType ? <span>结果类型：{formatArtifactType(debugInfo.artifactType)}</span> : null}
          {debugInfo.refusalReason ? <span>拒答原因：{formatRefusalReason(debugInfo.refusalReason)}</span> : null}
        </div>
      ) : null}
      {agentSteps.length || agentRunTrace || retrievalDebug ? (
        <ExecutionTrace
          steps={agentSteps}
          runTrace={agentRunTrace}
          retrievalDebug={retrievalDebug}
          originalQuery={originalSearchQuery}
        />
      ) : null}
      {message.citations.length ? (
        <div className="message-citations">
          {message.citations.map((citation) => (
            <button className="citation-pill" key={citation.id} onClick={() => onSelectCitation(citation)} type="button">
              {citation.document_title} · {locationLabel(citation)}
            </button>
          ))}
        </div>
      ) : null}
    </article>
  );
}




