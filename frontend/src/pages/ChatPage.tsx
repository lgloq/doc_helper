import { FormEvent, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { CitationList } from "../components/CitationList";
import { ErrorNotice } from "../components/ErrorNotice";
import { ExecutionTrace } from "../components/ExecutionTrace";
import { PageHeader } from "../components/PageHeader";
import { StatusBadge } from "../components/StatusBadge";
import { useAppContext } from "../context/AppContext";
import { api } from "../lib/api";
import { formatArtifactType, formatConfidence, formatCopilotIntent, formatMessageRole, formatRefusalReason } from "../lib/display";
import { formatDateTime, locationLabel } from "../lib/format";
import type { AgentRunTraceRead, AgentStepRead, ChatCitationRead, ChatMessageRead, ChatSessionDetailRead, ChatSessionRead } from "../types/api";

export function ChatPage() {
  const { token, selectedSessionId, setSelectedSessionId } = useAppContext();
  const [sessions, setSessions] = useState<ChatSessionRead[]>([]);
  const [activeSession, setActiveSession] = useState<ChatSessionDetailRead | null>(null);
  const [messageDraft, setMessageDraft] = useState("");
  const [selectedCitation, setSelectedCitation] = useState<ChatCitationRead | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [artifactMessage, setArtifactMessage] = useState<string | null>(null);
  const threadRef = useRef<HTMLDivElement | null>(null);
  const sessionButtonRefs = useRef<Record<string, HTMLButtonElement | null>>({});

  useEffect(() => {
    if (!token) {
      return;
    }
    setLoading(true);
    api
      .listChatSessions(token)
      .then((items) => {
        setSessions(items);
        const nextSessionId = selectedSessionId && items.some((item) => item.id === selectedSessionId)
          ? selectedSessionId
          : items[0]?.id ?? null;
        if (nextSessionId) {
          setSelectedSessionId(nextSessionId);
        }
      })
      .catch((nextError) => setError(nextError instanceof Error ? nextError.message : "加载会话列表失败。"))
      .finally(() => setLoading(false));
  }, [selectedSessionId, setSelectedSessionId, token]);

  useEffect(() => {
    if (!token || !selectedSessionId) {
      setActiveSession(null);
      return;
    }
    api
      .getChatSession(token, selectedSessionId)
      .then((session) => {
        setActiveSession(session);
        const latestCitation = [...session.messages]
          .reverse()
          .flatMap((message) => message.citations)[0] ?? null;
        setSelectedCitation(latestCitation);
      })
      .catch((nextError) => setError(nextError instanceof Error ? nextError.message : "加载会话详情失败。"));
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
    if (!activeSession?.messages.length || !threadRef.current) {
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
  }, [activeSession?.id, activeSession?.messages.length]);

  async function handleCreateSession() {
    if (!token) {
      return;
    }
    try {
      const session = await api.createChatSession(token, "新会话");
      const nextSessions = await api.listChatSessions(token);
      setSessions(nextSessions);
      setSelectedSessionId(session.id);
      setArtifactMessage(null);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "创建会话失败。");
    }
  }

  async function handleSendMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!token || !selectedSessionId || !messageDraft.trim()) {
      return;
    }
    setSending(true);
    setError(null);
    try {
      const response = await api.sendChatMessage(token, selectedSessionId, messageDraft.trim(), 5);
      setMessageDraft("");
      setActiveSession((current) => {
        if (!current) {
          return current;
        }
        return {
          ...current,
          messages: [...current.messages, response.user_message, response.assistant_message],
          updated_at: response.assistant_message.created_at,
        };
      });
      setSelectedCitation(response.citations[0] ?? null);
      const nextSessions = await api.listChatSessions(token);
      setSessions(nextSessions);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "发送问题失败。");
    } finally {
      setSending(false);
    }
  }

  async function handleArtifactAction(action: "tasks" | "report" | "faq") {
    if (!token || !selectedSessionId) {
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

  const flattenedCitations = activeSession?.messages.flatMap((message) => message.citations) ?? [];

  return (
    <div className="page-stack">
      <PageHeader
        title="引用式问答"
        description="围绕可访问的企业文档发起 RAG 问答，查看引用来源，并沉淀为待办、周报或 FAQ。"
        actions={
          <div className="inline-actions">
            <button className="secondary-button" onClick={handleCreateSession} type="button">
              新建会话
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
                <button
                  key={session.id}
                  className={`list-card ${selectedSessionId === session.id ? "is-selected" : ""}`}
                  onClick={() => setSelectedSessionId(session.id)}
                  ref={(element) => {
                    sessionButtonRefs.current[session.id] = element;
                  }}
                  type="button"
                >
                  <div className="list-card-topline">
                    <strong>{session.title}</strong>
                  </div>
                  <p>{formatDateTime(session.updated_at)}</p>
                </button>
              ))
            ) : (
              <p className="muted">暂无会话，点击右上角“新建会话”开始提问。</p>
            )}
          </div>
        </section>

        <section className="panel stack chat-conversation-panel">
          <div className="panel-header">
            <div className="panel-heading">
              <h3>{activeSession?.title ?? "当前会话"}</h3>
              <p>仅基于检索证据作答；证据不足时会提示并保留引用来源。</p>
            </div>
            {activeSession ? <StatusBadge tone="info">会话已关联</StatusBadge> : null}
          </div>
          <div className="chat-thread" ref={threadRef}>
            {activeSession?.messages.length ? (
              <>
                {activeSession.messages.map((message) => (
                  <MessageBubble
                    key={message.id}
                    message={message}
                    onSelectCitation={(citation) => setSelectedCitation(citation)}
                  />
                ))}
                {sending ? (
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
              value={messageDraft}
              onChange={(event) => setMessageDraft(event.target.value)}
            />
            <div className="composer-actions">
              <div className="inline-actions">
                <button className="secondary-button" onClick={() => handleArtifactAction("tasks")} type="button">
                  提取待办
                </button>
                <button className="secondary-button" onClick={() => handleArtifactAction("report")} type="button">
                  周报草稿
                </button>
                <button className="secondary-button" onClick={() => handleArtifactAction("faq")} type="button">
                  FAQ 草稿
                </button>
              </div>
              <button className="primary-button" disabled={!selectedSessionId || sending} type="submit">
                {sending ? "回答中..." : "发送"}
              </button>
            </div>
          </form>
        </section>

        <div className="stack chat-side-panel">
          <CitationList
            citations={flattenedCitations}
            onSelect={(citation) => setSelectedCitation(citation as ChatCitationRead)}
            selectedCitationId={selectedCitation?.id}
            title="引用来源"
          />
          <section className="panel stack">
            <div className="panel-header">
              <div className="panel-heading">
                <h3>来源片段</h3>
                <p>展示当前选中引用的片段摘要，并支持跳转到文档页查看完整内容。</p>
              </div>
              {selectedCitation ? (
                <Link className="secondary-button link-button" to={buildCitationDocumentLink(selectedCitation)}>
                  查看完整文档
                </Link>
              ) : null}
            </div>
            {selectedCitation ? (
              <>
                <div className="list-card-topline">
                  <strong>{selectedCitation.document_title}</strong>
                  <span>
                    v{selectedCitation.version_number} · {locationLabel(selectedCitation)}
                  </span>
                </div>
                <p>{selectedCitation.preview}</p>
                <div className="metadata-grid">
                  <span>分块 ID：{selectedCitation.chunk_id ?? "-"}</span>
                  <span>融合分数：{selectedCitation.fused_score?.toFixed(3) ?? "-"}</span>
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

interface MessageBubbleProps {
  message: ChatMessageRead;
  onSelectCitation: (citation: ChatCitationRead) => void;
}

function MessageBubble({ message, onSelectCitation }: MessageBubbleProps) {
  const tone = message.role === "assistant" ? "assistant" : "user";
  const debugInfo = message.role === "assistant" ? readMessageDebugInfo(message) : null;
  const agentSteps = message.role === "assistant" ? readAgentSteps(message) : [];
  const agentRunTrace = message.role === "assistant" ? readAgentRunTrace(message) : null;
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
      {debugInfo ? (
        <div className="metadata-grid muted">
          <span>意图：{formatCopilotIntent(debugInfo.intent)}</span>
          {debugInfo.targetDocument ? <span>目标文档：{debugInfo.targetDocument}</span> : null}
          {debugInfo.artifactType ? <span>结果类型：{formatArtifactType(debugInfo.artifactType)}</span> : null}
          {debugInfo.refusalReason ? <span>拒答原因：{formatRefusalReason(debugInfo.refusalReason)}</span> : null}
        </div>
      ) : null}
      {agentSteps.length || agentRunTrace ? <ExecutionTrace steps={agentSteps} runTrace={agentRunTrace} /> : null}
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




