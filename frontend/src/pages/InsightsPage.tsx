import { useEffect, useState } from "react";

import { ErrorNotice } from "../components/ErrorNotice";
import { PageHeader } from "../components/PageHeader";
import { StatusBadge } from "../components/StatusBadge";
import { useAppContext } from "../context/AppContext";
import { api } from "../lib/api";
import { formatTraceType, formatWorkflowStatus } from "../lib/display";
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
      .catch((nextError) => setError(nextError instanceof Error ? nextError.message : "加载追踪列表失败。"));
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
      .catch((nextError) => setError(nextError instanceof Error ? nextError.message : "加载评测记录失败。"));
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
      setStatusMessage(`评测完成，共运行 ${run.results.length} 条用例。`);
    } catch (nextError) {
      setStatusMessage(nextError instanceof Error ? nextError.message : "评测运行失败。");
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
      setStatusMessage(nextError instanceof Error ? nextError.message : "加载评测详情失败。");
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
      setStatusMessage(nextError instanceof Error ? nextError.message : "加载追踪详情失败。");
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
        title="评测与追踪"
        description="运行权限隔离导向的演示评测，并查看一次 RAG 问答的完整追踪信息。"
        actions={
          isAdmin ? (
            <button className="primary-button" onClick={handleRunEval} type="button">
              运行演示评测
            </button>
          ) : null
        }
      />
      <ErrorNotice message={error} />
      {statusMessage ? <div className="info-block">{statusMessage}</div> : null}

      <div className="page-grid insights-layout">
        <section className="panel stack">
          <div className="panel-header">
            <h3>评测记录</h3>
            <StatusBadge tone={isAdmin ? "warning" : "neutral"}>{isAdmin ? runs.length : "仅管理员"}</StatusBadge>
          </div>
          {isAdmin ? (
            <>
              <div className="stack dense-stack">
                {runs.length ? (
                  runs.map((run) => (
                    <button className="list-card text-left" key={run.id} onClick={() => handleSelectRun(run.id)} type="button">
                      <div className="list-card-topline">
                        <strong>{run.dataset_name}</strong>
                        <StatusBadge tone={run.status === "completed" ? "success" : "warning"}>
                          {formatWorkflowStatus(run.status)}
                        </StatusBadge>
                      </div>
                      <p>{formatDateTime(run.created_at)}</p>
                    </button>
                  ))
                ) : (
                  <p className="muted">暂无评测记录。</p>
                )}
              </div>
              {selectedRun ? (
                <div className="stack">
                  <div className="subsection-header">
                    <h4>当前运行摘要</h4>
                  </div>
                  <div className="metrics-grid">
                    <div className="metric-card">
                      <span>总用例数</span>
                      <strong>{totalCases}</strong>
                    </div>
                    <div className="metric-card">
                      <span>检索命中均值</span>
                      <strong>{summaryValue("retrieval_hit_rate_avg")}</strong>
                    </div>
                    <div className="metric-card">
                      <span>引用准确率均值</span>
                      <strong>{summaryValue("citation_accuracy_avg")}</strong>
                    </div>
                    <div className="metric-card">
                      <span>权限隔离通过率</span>
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
                              {result.overall_pass ? "通过" : "需复核"}
                            </StatusBadge>
                          </div>
                          <p>
                            检索 {result.retrieval_hit_rate.toFixed(2)} · 引用 {result.citation_accuracy.toFixed(2)} · 忠实性 {result.answer_faithfulness.toFixed(2)}
                          </p>
                          <p className="muted">{truncate(String(details.answer_excerpt ?? ""), 160)}</p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : null}
            </>
          ) : (
            <p className="muted">评测接口仅管理员可用，请使用管理员账号运行演示评测。</p>
          )}
        </section>

        <section className="panel stack">
          <div className="panel-header">
            <h3>追踪列表</h3>
            <StatusBadge tone="info">{traces.length}</StatusBadge>
          </div>
          <div className="stack dense-stack">
            {traces.length ? (
              traces.map((trace) => (
                <button className="list-card text-left" key={trace.id} onClick={() => handleSelectTrace(trace.id)} type="button">
                  <div className="list-card-topline">
                    <strong>{formatTraceType(trace.trace_type)}</strong>
                    <span>{formatDateTime(trace.created_at)}</span>
                  </div>
                  <p>{truncate(trace.query_text ?? "暂无查询内容", 120)}</p>
                </button>
              ))
            ) : (
              <p className="muted">暂无追踪记录，先去问答页发起一次提问。</p>
            )}
          </div>
        </section>

        <section className="panel stack">
          <div className="panel-header">
            <h3>追踪详情</h3>
            {selectedTrace ? <StatusBadge tone="warning">已选中</StatusBadge> : null}
          </div>
          {selectedTrace ? (
            <>
              <p>{selectedTrace.query_text}</p>
              <div className="metadata-grid">
                <span>模型：{selectedTrace.model_name ?? "-"}</span>
                <span>耗时：{selectedTrace.latency_ms ?? "-"} ms</span>
                <span>输入 Token：{selectedTrace.prompt_tokens ?? "-"}</span>
                <span>输出 Token：{selectedTrace.completion_tokens ?? "-"}</span>
              </div>
              <div className="subsection-header">
                <h4>召回分块</h4>
              </div>
              <div className="stack dense-stack">
                {retrievedChunks.length ? (
                  retrievedChunks.map((chunk, index) => (
                    <div className="list-card" key={`${String(chunk.chunk_id)}-${index}`}>
                      <div className="list-card-topline">
                        <strong>{String(chunk.document_title ?? "文档")}</strong>
                        <span>分块 {String(chunk.chunk_index ?? "-")}</span>
                      </div>
                      <p>{truncate(String(chunk.preview ?? ""), 180)}</p>
                    </div>
                  ))
                ) : (
                  <p className="muted">暂无召回分块记录。</p>
                )}
              </div>
              <div className="subsection-header">
                <h4>选中引用</h4>
              </div>
              <div className="stack dense-stack">
                {selectedCitations.length ? (
                  selectedCitations.map((chunk, index) => (
                    <div className="list-card" key={`${String(chunk.chunk_id)}-selected-${index}`}>
                      <div className="list-card-topline">
                        <strong>{String(chunk.document_title ?? "文档")}</strong>
                        <span>v{String(chunk.version_number ?? "-")}</span>
                      </div>
                      <p>{truncate(String(chunk.preview ?? ""), 180)}</p>
                    </div>
                  ))
                ) : (
                  <p className="muted">暂无引用记录。</p>
                )}
              </div>
            </>
          ) : (
            <p className="muted">请选择一条追踪，查看查询、召回分块、引用、延迟和 token 使用情况。</p>
          )}
        </section>
      </div>
    </div>
  );
}



