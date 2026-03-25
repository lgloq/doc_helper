import { FormEvent, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { CitationList } from "../components/CitationList";
import { ErrorNotice } from "../components/ErrorNotice";
import { PageHeader } from "../components/PageHeader";
import { StatusBadge } from "../components/StatusBadge";
import { useAppContext } from "../context/AppContext";
import { api } from "../lib/api";
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
      .catch((nextError) => setError(nextError instanceof Error ? nextError.message : "Failed to load sessions."))
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
      .catch((nextError) => setError(nextError instanceof Error ? nextError.message : "Failed to load chat session."));
  }, [selectedSessionId, token]);

  async function handleCreateSession() {
    if (!token) {
      return;
    }
    try {
      const session = await api.createChatSession(token, "New Chat");
      const nextSessions = await api.listChatSessions(token);
      setSessions(nextSessions);
      setSelectedSessionId(session.id);
      setArtifactMessage(null);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to create session.");
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
      setError(nextError instanceof Error ? nextError.message : "Failed to send message.");
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
        setArtifactMessage(`Generated ${response.items.length} task items. Open Artifacts for details.`);
      } else if (action === "report") {
        const response = await api.generateWeeklyReport(token, selectedSessionId, "Weekly Report Draft");
        setArtifactMessage(`Generated report: ${response.report.title}`);
      } else {
        const response = await api.generateFaqs(token, selectedSessionId);
        setArtifactMessage(`Generated ${response.entries.length} FAQ draft entries.`);
      }
    } catch (nextError) {
      setArtifactMessage(nextError instanceof Error ? nextError.message : "Artifact generation failed.");
    }
  }

  const flattenedCitations = activeSession?.messages.flatMap((message) => message.citations) ?? [];

  return (
    <div className="page-stack">
      <PageHeader
        title="Grounded Chat"
        description="Ask a question, inspect grounded citations, and turn one session into tasks, weekly report drafts, or FAQ entries."
        actions={
          <div className="inline-actions">
            <button className="secondary-button" onClick={handleCreateSession} type="button">
              New session
            </button>
            <Link className="secondary-button link-button" to="/artifacts">
              Open artifacts
            </Link>
          </div>
        }
      />
      <ErrorNotice message={error} />
      {artifactMessage ? <div className="info-block">{artifactMessage}</div> : null}

      <div className="page-grid chat-layout">
        <section className="panel stack">
          <div className="panel-header">
            <h3>Sessions</h3>
            <StatusBadge tone="neutral">{sessions.length}</StatusBadge>
          </div>
          {loading ? <p className="muted">Loading sessions...</p> : null}
          <div className="session-list">
            {sessions.map((session) => (
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
            ))}
          </div>
        </section>

        <section className="panel stack chat-conversation-panel">
          <div className="panel-header">
            <div>
              <h3>{activeSession?.title ?? "Conversation"}</h3>
              <p className="muted">Grounded answering only. If evidence is weak, the assistant should stay cautious.</p>
            </div>
            {activeSession ? <StatusBadge tone="info">session linked</StatusBadge> : null}
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
                <strong>No messages yet</strong>
                <p>Start a new session and ask a question after uploading documents.</p>
              </div>
            )}
          </div>
          <form className="chat-composer" onSubmit={handleSendMessage}>
            <textarea
              placeholder="Ask a grounded question about accessible internal documents..."
              rows={4}
              value={messageDraft}
              onChange={(event) => setMessageDraft(event.target.value)}
            />
            <div className="composer-actions">
              <div className="inline-actions">
                <button className="secondary-button" onClick={() => handleArtifactAction("tasks")} type="button">
                  Extract tasks
                </button>
                <button className="secondary-button" onClick={() => handleArtifactAction("report")} type="button">
                  Weekly report
                </button>
                <button className="secondary-button" onClick={() => handleArtifactAction("faq")} type="button">
                  FAQ draft
                </button>
              </div>
              <button className="primary-button" disabled={!selectedSessionId || sending} type="submit">
                {sending ? "Answering..." : "Send"}
              </button>
            </div>
          </form>
        </section>

        <div className="stack">
          <CitationList
            citations={flattenedCitations}
            onSelect={(citation) => setSelectedCitation(citation as ChatCitationRead)}
            selectedCitationId={selectedCitation?.id}
            title="Grounding citations"
          />
          <section className="panel stack">
            <div className="panel-header">
              <h3>Source snippet</h3>
              {selectedCitation ? <StatusBadge tone="warning">click-through ready</StatusBadge> : null}
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
                  <span>chunk: {selectedCitation.chunk_id ?? "-"}</span>
                  <span>fused score: {selectedCitation.fused_score?.toFixed(3) ?? "-"}</span>
                </div>
              </>
            ) : (
              <p className="muted">Select a citation from the session to inspect the source preview.</p>
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
        <strong>{message.role}</strong>
        <span>{formatDateTime(message.created_at)}</span>
        {message.confidence ? <StatusBadge tone={message.insufficient_evidence ? "warning" : "success"}>{message.confidence}</StatusBadge> : null}
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
