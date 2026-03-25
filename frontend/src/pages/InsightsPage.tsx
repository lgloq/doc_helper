import { useEffect, useState } from "react";

import { ErrorNotice } from "../components/ErrorNotice";
import { PageHeader } from "../components/PageHeader";
import { StatusBadge } from "../components/StatusBadge";
import { useAppContext } from "../context/AppContext";
import { api } from "../lib/api";
import { asArray, formatDateTime, truncate } from "../lib/format";
import type { EvalRunDetailRead, EvalRunRead, TraceLogRead } from "../types/api";

export function InsightsPage() {
  const { token, user } = useAppContext();
  const [runs, setRuns] = useState<EvalRunRead[]>([]);
  const [selectedRun, setSelectedRun] = useState<EvalRunDetailRead | null>(null);
  const [traces, setTraces] = useState<TraceLogRead[]>([]);
  const [selectedTrace, setSelectedTrace] = useState<TraceLogRead | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const isAdmin = user?.role?.name === "admin";

  useEffect(() => {
    if (!token) {
      return;
    }
    api
      .listTraces(token)
      .then((items) => {
        setTraces(items);
        setSelectedTrace(items[0] ?? null);
      })
      .catch((nextError) => setError(nextError instanceof Error ? nextError.message : "Failed to load traces."));
  }, [token]);

  useEffect(() => {
    if (!token || !isAdmin) {
      return;
    }
    api
      .listEvalRuns(token)
      .then((items) => {
        setRuns(items);
        if (items[0]) {
          return api.getEvalRun(token, items[0].id);
        }
        return null;
      })
      .then((detail) => {
        if (detail) {
          setSelectedRun(detail);
        }
      })
      .catch((nextError) => setError(nextError instanceof Error ? nextError.message : "Failed to load eval runs."));
  }, [isAdmin, token]);

  async function handleRunEval() {
    if (!token) {
      return;
    }
    setStatusMessage(null);
    try {
      const run = await api.runEval(token, { dataset_name: "demo_permission_eval", top_k: 5, seed_demo_cases: true });
      setSelectedRun(run);
      setRuns(await api.listEvalRuns(token));
      setStatusMessage(`Eval run completed with ${run.results.length} cases.`);
    } catch (nextError) {
      setStatusMessage(nextError instanceof Error ? nextError.message : "Eval run failed.");
    }
  }

  async function handleSelectRun(runId: string) {
    if (!token) {
      return;
    }
    try {
      const detail = await api.getEvalRun(token, runId);
      setSelectedRun(detail);
    } catch (nextError) {
      setStatusMessage(nextError instanceof Error ? nextError.message : "Failed to load eval run.");
    }
  }

  async function handleSelectTrace(traceId: string) {
    if (!token) {
      return;
    }
    try {
      const detail = await api.getTrace(token, traceId);
      setSelectedTrace(detail);
    } catch (nextError) {
      setStatusMessage(nextError instanceof Error ? nextError.message : "Failed to load trace.");
    }
  }

  const retrievedChunks = asArray<Record<string, unknown>>(selectedTrace?.retrieved_chunks_json);
  const selectedCitations = asArray<Record<string, unknown>>(selectedTrace?.selected_citations_json);
  const summary = selectedRun?.summary_json ?? null;
  const summaryValue = (key: string, fallback = "-") => String(summary?.[key] ?? fallback);
  const totalCases = typeof summary?.total_cases === "number" ? summary.total_cases : selectedRun?.total_cases ?? 0;

  return (
    <div className="page-stack">
      <PageHeader
        title="Eval & Observability"
        description="Run a small permission-focused benchmark, inspect metric summaries, and review one grounded QA trace end to end."
        actions={
          isAdmin ? (
            <button className="primary-button" onClick={handleRunEval} type="button">
              Run demo eval
            </button>
          ) : null
        }
      />
      <ErrorNotice message={error} />
      {statusMessage ? <div className="info-block">{statusMessage}</div> : null}

      <div className="page-grid insights-layout">
        <section className="panel stack">
          <div className="panel-header">
            <h3>Eval runs</h3>
            <StatusBadge tone={isAdmin ? "warning" : "neutral"}>{isAdmin ? runs.length : "admin only"}</StatusBadge>
          </div>
          {isAdmin ? (
            <>
              <div className="stack dense-stack">
                {runs.map((run) => (
                  <button className="list-card text-left" key={run.id} onClick={() => handleSelectRun(run.id)} type="button">
                    <div className="list-card-topline">
                      <strong>{run.dataset_name}</strong>
                      <StatusBadge tone={run.status === "completed" ? "success" : "warning"}>{run.status}</StatusBadge>
                    </div>
                    <p>{formatDateTime(run.created_at)}</p>
                  </button>
                ))}
              </div>
              {selectedRun ? (
                <div className="stack">
                  <div className="subsection-header">
                    <h4>Selected run summary</h4>
                  </div>
                  <div className="metrics-grid">
                    <div className="metric-card">
                      <span>Total cases</span>
                      <strong>{totalCases}</strong>
                    </div>
                    <div className="metric-card">
                      <span>Retrieval hit avg</span>
                      <strong>{summaryValue("retrieval_hit_rate_avg")}</strong>
                    </div>
                    <div className="metric-card">
                      <span>Citation acc avg</span>
                      <strong>{summaryValue("citation_accuracy_avg")}</strong>
                    </div>
                    <div className="metric-card">
                      <span>Permission isolation</span>
                      <strong>{summaryValue("permission_isolation_pass_rate")}</strong>
                    </div>
                  </div>
                  <div className="stack dense-stack">
                    {selectedRun.results.map((result) => {
                      const details = result.details_json;
                      return (
                        <div className="list-card" key={result.id}>
                          <div className="list-card-topline">
                            <strong>{String(details.case_name ?? result.case_id)}</strong>
                            <StatusBadge tone={result.overall_pass ? "success" : "warning"}>
                              {result.overall_pass ? "pass" : "review"}
                            </StatusBadge>
                          </div>
                          <p>
                            retrieval {result.retrieval_hit_rate.toFixed(2)} · citation {result.citation_accuracy.toFixed(2)} ·
                            faithfulness {result.answer_faithfulness.toFixed(2)}
                          </p>
                          <p className="muted">{truncate(String(details.answer_excerpt ?? ""), 160)}</p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : (
                <p className="muted">No eval runs yet.</p>
              )}
            </>
          ) : (
            <p className="muted">Eval endpoints are restricted to admin users. Log in as admin to run the demo benchmark.</p>
          )}
        </section>

        <section className="panel stack">
          <div className="panel-header">
            <h3>Trace list</h3>
            <StatusBadge tone="info">{traces.length}</StatusBadge>
          </div>
          <div className="stack dense-stack">
            {traces.map((trace) => (
              <button className="list-card text-left" key={trace.id} onClick={() => handleSelectTrace(trace.id)} type="button">
                <div className="list-card-topline">
                  <strong>{trace.trace_type}</strong>
                  <span>{formatDateTime(trace.created_at)}</span>
                </div>
                <p>{truncate(trace.query_text ?? "No query text", 120)}</p>
              </button>
            ))}
          </div>
        </section>

        <section className="panel stack">
          <div className="panel-header">
            <h3>Trace detail</h3>
            {selectedTrace ? <StatusBadge tone="warning">trace</StatusBadge> : null}
          </div>
          {selectedTrace ? (
            <>
              <p>{selectedTrace.query_text}</p>
              <div className="metadata-grid">
                <span>model: {selectedTrace.model_name ?? "-"}</span>
                <span>latency: {selectedTrace.latency_ms ?? "-"} ms</span>
                <span>prompt tokens: {selectedTrace.prompt_tokens ?? "-"}</span>
                <span>completion tokens: {selectedTrace.completion_tokens ?? "-"}</span>
              </div>
              <div className="subsection-header">
                <h4>Retrieved chunks</h4>
              </div>
              <div className="stack dense-stack">
                {retrievedChunks.map((chunk, index) => (
                  <div className="list-card" key={`${String(chunk.chunk_id)}-${index}`}>
                    <div className="list-card-topline">
                      <strong>{String(chunk.document_title ?? "document")}</strong>
                      <span>chunk {String(chunk.chunk_index ?? "-")}</span>
                    </div>
                    <p>{truncate(String(chunk.preview ?? ""), 180)}</p>
                  </div>
                ))}
              </div>
              <div className="subsection-header">
                <h4>Selected citations</h4>
              </div>
              <div className="stack dense-stack">
                {selectedCitations.map((chunk, index) => (
                  <div className="list-card" key={`${String(chunk.chunk_id)}-selected-${index}`}>
                    <div className="list-card-topline">
                      <strong>{String(chunk.document_title ?? "document")}</strong>
                      <span>v{String(chunk.version_number ?? "-")}</span>
                    </div>
                    <p>{truncate(String(chunk.preview ?? ""), 180)}</p>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <p className="muted">Pick a trace to inspect query, retrieved chunks, citations, latency, and token usage.</p>
          )}
        </section>
      </div>
    </div>
  );
}
