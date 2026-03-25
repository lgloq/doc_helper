import { useEffect, useState } from "react";

import { CitationList } from "../components/CitationList";
import { ErrorNotice } from "../components/ErrorNotice";
import { PageHeader } from "../components/PageHeader";
import { StatusBadge } from "../components/StatusBadge";
import { useAppContext } from "../context/AppContext";
import { api } from "../lib/api";
import { formatDateTime } from "../lib/format";
import type { FAQEntryRead, SourceCitationRead, TaskItemRead, WeeklyReportDraftRead } from "../types/api";

export function ArtifactsPage() {
  const { token, selectedSessionId } = useAppContext();
  const [tasks, setTasks] = useState<TaskItemRead[]>([]);
  const [reports, setReports] = useState<WeeklyReportDraftRead[]>([]);
  const [faqs, setFaqs] = useState<FAQEntryRead[]>([]);
  const [sourceSessionId, setSourceSessionId] = useState(selectedSessionId ?? "");
  const [selectedSources, setSelectedSources] = useState<SourceCitationRead[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  useEffect(() => {
    setSourceSessionId(selectedSessionId ?? "");
  }, [selectedSessionId]);

  useEffect(() => {
    if (!token) {
      return;
    }
    Promise.all([api.listTasks(token), api.listReports(token), api.listFaqs(token)])
      .then(([nextTasks, nextReports, nextFaqs]) => {
        setTasks(nextTasks);
        setReports(nextReports);
        setFaqs(nextFaqs);
      })
      .catch((nextError) => setError(nextError instanceof Error ? nextError.message : "Failed to load artifacts."));
  }, [token]);

  function bindFirstSources(citations: SourceCitationRead[] | null | undefined) {
    setSelectedSources(citations ?? []);
  }

  async function handleGenerate(kind: "tasks" | "report" | "faq") {
    if (!token || !sourceSessionId.trim()) {
      setStatusMessage("Provide a chat session id or pick one from the chat page first.");
      return;
    }
    setStatusMessage(null);
    try {
      if (kind === "tasks") {
        const response = await api.extractTasks(token, sourceSessionId.trim());
        setTasks(await api.listTasks(token));
        bindFirstSources(response.items[0]?.source_citations ?? []);
        setStatusMessage(`Generated ${response.items.length} tasks.`);
      } else if (kind === "report") {
        const response = await api.generateWeeklyReport(token, sourceSessionId.trim(), "Weekly Report Draft");
        setReports(await api.listReports(token));
        bindFirstSources(response.report.reference_sources);
        setStatusMessage(`Generated report ${response.report.title}.`);
      } else {
        const response = await api.generateFaqs(token, sourceSessionId.trim());
        setFaqs(await api.listFaqs(token));
        bindFirstSources(response.entries[0]?.source_citations ?? []);
        setStatusMessage(`Generated ${response.entries.length} FAQ entries.`);
      }
    } catch (nextError) {
      setStatusMessage(nextError instanceof Error ? nextError.message : "Generation failed.");
    }
  }

  return (
    <div className="page-stack">
      <PageHeader
        title="Workflow Artifacts"
        description="Turn one grounded chat session into task items, weekly report drafts, and FAQ entries with source traceability."
      />
      <ErrorNotice message={error} />
      {statusMessage ? <div className="info-block">{statusMessage}</div> : null}

      <section className="panel stack">
        <div className="panel-header">
          <h3>Generate from session</h3>
          <StatusBadge tone="info">Structured outputs only</StatusBadge>
        </div>
        <label>
          <span>Source session id</span>
          <input value={sourceSessionId} onChange={(event) => setSourceSessionId(event.target.value)} />
        </label>
        <div className="inline-actions">
          <button className="secondary-button" onClick={() => handleGenerate("tasks")} type="button">
            Extract tasks
          </button>
          <button className="secondary-button" onClick={() => handleGenerate("report")} type="button">
            Generate weekly report
          </button>
          <button className="secondary-button" onClick={() => handleGenerate("faq")} type="button">
            Generate FAQ draft
          </button>
        </div>
      </section>

      <div className="page-grid artifacts-layout">
        <section className="panel stack">
          <div className="panel-header">
            <h3>Task items</h3>
            <StatusBadge tone="warning">{tasks.length}</StatusBadge>
          </div>
          <div className="stack dense-stack">
            {tasks.map((task) => (
              <button className="list-card text-left" key={task.id} onClick={() => bindFirstSources(task.source_citations)} type="button">
                <div className="list-card-topline">
                  <strong>{task.title}</strong>
                  <StatusBadge tone="info">{task.priority}</StatusBadge>
                </div>
                <p>{task.description ?? "No description"}</p>
                <p className="muted">{formatDateTime(task.created_at)}</p>
              </button>
            ))}
          </div>
        </section>

        <section className="panel stack">
          <div className="panel-header">
            <h3>Weekly reports</h3>
            <StatusBadge tone="success">{reports.length}</StatusBadge>
          </div>
          <div className="stack dense-stack">
            {reports.map((report) => (
              <button className="list-card text-left" key={report.id} onClick={() => bindFirstSources(report.reference_sources)} type="button">
                <div className="list-card-topline">
                  <strong>{report.title}</strong>
                  <StatusBadge tone="neutral">{report.status}</StatusBadge>
                </div>
                <p>{report.summary ?? "No summary"}</p>
                <div className="report-list-grid">
                  <span>Completed: {report.completed_this_week.length}</span>
                  <span>Risks: {report.risks_blockers.length}</span>
                  <span>Next week: {report.next_week_plan.length}</span>
                </div>
              </button>
            ))}
          </div>
        </section>

        <section className="panel stack">
          <div className="panel-header">
            <h3>FAQ entries</h3>
            <StatusBadge tone="neutral">{faqs.length}</StatusBadge>
          </div>
          <div className="stack dense-stack">
            {faqs.map((faq) => (
              <button className="list-card text-left" key={faq.id} onClick={() => bindFirstSources(faq.source_citations)} type="button">
                <div className="list-card-topline">
                  <strong>{faq.question}</strong>
                  <StatusBadge tone={faq.quality === "high" ? "success" : "warning"}>{faq.quality}</StatusBadge>
                </div>
                <p>{faq.answer}</p>
                <p className="muted">status: {faq.status}</p>
              </button>
            ))}
          </div>
        </section>
      </div>

      <CitationList citations={selectedSources} title="Artifact source citations" />
    </div>
  );
}
