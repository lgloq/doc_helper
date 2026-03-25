import { FormEvent, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { CitationList } from "../components/CitationList";
import { ErrorNotice } from "../components/ErrorNotice";
import { PageHeader } from "../components/PageHeader";
import { StatusBadge } from "../components/StatusBadge";
import { useAppContext } from "../context/AppContext";
import { api } from "../lib/api";
import { formatConfidence, formatMessageRole } from "../lib/display";
import { formatDateTime, locationLabel } from "../lib/format";
import type { ChatCitationRead, ChatMessageRead, ChatSessionDetailRead, ChatSessionRead } from "../types/api";

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
        const firstCitation = session.messages.flatMap((message) => message.citations)[0] ?? null;
        setSelectedCitation(firstCitation);
      })
      .catch((nextError) => setError(nextError instanceof Error ? nextError.message : "加载会话详情失败。"));
  }, [selectedSessionId, token]);

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
        description="围绕你有权限访问的文档发起问答，查看引用来源，并将当前会话沉淀为待办、周报草稿或 FAQ 草稿。"
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
        <section className="panel stack">
          <div className="panel-header">
            <h3>会话列表</h3>
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
            <div>
              <h3>{activeSession?.title ?? "当前会话"}</h3>
              <p className="muted">回答只基于已检索证据生成；如果证据不足，系统会明确提示。</p>
            </div>
            {activeSession ? <StatusBadge tone="info">会话已关联</StatusBadge> : null}
          </div>
          <div className="chat-thread">
            {activeSession?.messages.length ? (
              activeSession.messages.map((message) => (
                <MessageBubble
                  key={message.id}
                  message={message}
                  onSelectCitation={(citation) => setSelectedCitation(citation)}
                />
              ))
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

        <div className="stack">
          <CitationList
            citations={flattenedCitations}
            onSelect={(citation) => setSelectedCitation(citation as ChatCitationRead)}
            selectedCitationId={selectedCitation?.id}
            title="引用来源"
          />
          <section className="panel stack">
            <div className="panel-header">
              <h3>来源片段</h3>
              {selectedCitation ? <StatusBadge tone="warning">可点击查看</StatusBadge> : null}
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

interface MessageBubbleProps {
  message: ChatMessageRead;
  onSelectCitation: (citation: ChatCitationRead) => void;
}

function MessageBubble({ message, onSelectCitation }: MessageBubbleProps) {
  const tone = message.role === "assistant" ? "assistant" : "user";
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


