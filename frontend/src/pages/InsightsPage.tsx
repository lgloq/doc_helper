import { useEffect, useRef, useState } from "react";

import { ErrorNotice } from "../components/ErrorNotice";
import { PageHeader } from "../components/PageHeader";
import { StatusBadge } from "../components/StatusBadge";
import { useAppContext } from "../context/AppContext";
import { ApiError, api } from "../lib/api";
import { formatTraceType, formatWorkflowStatus } from "../lib/display";
import { asArray, formatDateTime, truncate } from "../lib/format";
import {
  createPendingEvalOperation,
  listPendingEvalOperations,
  removePendingEvalOperation,
  touchPendingEvalOperation,
} from "../lib/pendingOperations";
import type { PendingEvalOperation } from "../lib/pendingOperations";
import type { EvalDashboardRead, EvalDatasetRead, EvalResultRowRead, EvalRunDetailRead, EvalRunRead, TraceLogRead } from "../types/api";

const DEMO_EVAL_DATASET = "demo_access_matrix_eval";
const DEFAULT_VISIBLE_RUNS = 12;
const DEFAULT_VISIBLE_TRACES = 12;
const DEFAULT_VISIBLE_TRACE_ITEMS = 3;

type InsightsView = "eval" | "trace";
type EvalCaseFilter = "all" | "answer_expected" | "refusal_expected";
type EvalRunFilter = "latest_valid" | "connection_failures";

function formatEvalDatasetName(value: string | null | undefined): string {
  if (value === "demo_access_matrix_eval") {
    return "权限隔离演示评测";
  }
  if (value === "demo_permission_eval") {
    return "权限回归演示评测";
  }
  if (value === "zh_enterprise_v1_seed") {
    return "中文企业文档评测集";
  }
  return value ?? "-";
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value ? (value as Record<string, unknown>) : null;
}

function asStringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string" && item.length > 0) : [];
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" ? value : null;
}

function scoreText(value: unknown): string | null {
  const numeric = asNumber(value);
  return numeric === null ? null : numeric.toFixed(2);
}

function scoreFromRecord(record: Record<string, unknown> | null, key: string): string {
  const value = record?.[key];
  return typeof value === "number" ? value.toFixed(2) : "-";
}

function percentText(value: unknown): string {
  const numeric = asNumber(value);
  return numeric === null ? "-" : `${(numeric * 100).toFixed(1)}%`;
}

function deltaText(current: number | null | undefined, previous: number | null | undefined): string | null {
  if (typeof current !== "number" || typeof previous !== "number") {
    return null;
  }
  const delta = current - previous;
  if (Math.abs(delta) < 0.0001) {
    return "较上次持平";
  }
  const sign = delta > 0 ? "+" : "";
  return `较上次 ${sign}${(delta * 100).toFixed(1)} pct`;
}
function valueFromRecord(record: Record<string, unknown> | null, key: string, fallback = "-"): string {
  return String(record?.[key] ?? fallback);
}

function listText(values: string[], fallback = "无"): string {
  return values.length ? values.join("、") : fallback;
}

function expectedOutcomeText(value: unknown): string {
  return value === "refuse" ? "应拒答" : value === "answer" ? "应回答" : "未标注";
}

function caseTypeBucket(value: unknown): "answer_expected" | "refusal_expected" {
  return value === "refuse" ? "refusal_expected" : "answer_expected";
}

function isConnectionFailure(errorText: string | null | undefined): boolean {
  const value = (errorText ?? "").toLowerCase();
  return value.includes("connection error") || value.includes("ssl") || value.includes("timeout");
}

function evalRunStatusTone(
  status: string | null | undefined,
  errorText?: string | null,
): "success" | "warning" | "danger" | "neutral" {
  if (status === "completed") {
    return "success";
  }
  if (status === "failed") {
    return isConnectionFailure(errorText) ? "warning" : "danger";
  }
  if (status === "queued" || status === "running") {
    return "warning";
  }
  return "neutral";
}

function evalRunStatusLabel(status: string | null | undefined, errorText?: string | null): string {
  if (status === "failed") {
    return isConnectionFailure(errorText) ? "连接失败" : "运行失败";
  }
  return formatWorkflowStatus(status);
}

function formatEvalRunError(run: EvalRunDetailRead): string | null {
  if (!run.error_text) {
    return null;
  }
  if (!isConnectionFailure(run.error_text)) {
    return run.error_text;
  }
  const completedCases =
    typeof asRecord(run.summary_json)?.total_cases === "number"
      ? String(asRecord(run.summary_json)?.total_cases)
      : "0";
  return `本轮评测因上游模型连接失败提前结束。已落库用例：${completedCases} / ${run.total_cases}。`;
}

function caseStatusTone(result: EvalResultRowRead): "success" | "warning" {
  return result.overall_pass ? "success" : "warning";
}

function runSummaryRecord(run: EvalRunRead | EvalRunDetailRead): Record<string, unknown> | null {
  return asRecord(run.summary_json);
}

function runCompletedCaseCount(run: EvalRunRead | EvalRunDetailRead): number {
  const summary = runSummaryRecord(run);
  return typeof summary?.total_cases === "number" ? summary.total_cases : 0;
}

function runPassCount(run: EvalRunRead | EvalRunDetailRead): number | null {
  const summary = runSummaryRecord(run);
  return typeof summary?.pass_count === "number" ? summary.pass_count : null;
}

function isCompletedEvalRun(run: EvalRunRead | EvalRunDetailRead): boolean {
  return run.status === "completed";
}

function isFullPassEvalRun(run: EvalRunRead | EvalRunDetailRead): boolean {
  const passCount = runPassCount(run);
  return isCompletedEvalRun(run) && passCount !== null && passCount === run.total_cases;
}

function defaultEvalRun(items: EvalRunRead[]): EvalRunRead | null {
  return items.find((run) => isFullPassEvalRun(run)) ?? items.find((run) => isCompletedEvalRun(run)) ?? items[0] ?? null;
}

function runProgressText(run: EvalRunRead | EvalRunDetailRead): string {
  if (run.status === "completed") {
    const passCount = runPassCount(run);
    return passCount === null ? "已完成" : `${passCount} / ${run.total_cases} 通过`;
  }
  if (run.status === "failed" && isConnectionFailure(run.error_text)) {
    return `已完成 ${runCompletedCaseCount(run)} / ${run.total_cases}`;
  }
  return formatWorkflowStatus(run.status);
}
function evalRunClientRequestId(run: EvalRunRead | EvalRunDetailRead): string | null {
  const value = runSummaryRecord(run)?.client_request_id;
  return typeof value === "string" ? value : null;
}

function findEvalRunByClientRequestId(items: EvalRunRead[], clientRequestId: string): EvalRunRead | null {
  return items.find((run) => evalRunClientRequestId(run) === clientRequestId) ?? null;
}

export function InsightsPage() {
  const { token, user } = useAppContext();
  const [runs, setRuns] = useState<EvalRunRead[]>([]);
  const [evalDatasets, setEvalDatasets] = useState<EvalDatasetRead[]>([]);
  const [evalDashboard, setEvalDashboard] = useState<EvalDashboardRead | null>(null);
  const [selectedDatasetName, setSelectedDatasetName] = useState(DEMO_EVAL_DATASET);
  const [evalTopK, setEvalTopK] = useState(5);
  const [selectedRun, setSelectedRun] = useState<EvalRunDetailRead | null>(null);
  const [selectedResultId, setSelectedResultId] = useState<string | null>(null);
  const [traces, setTraces] = useState<TraceLogRead[]>([]);
  const [selectedTrace, setSelectedTrace] = useState<TraceLogRead | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [isRunningEval, setIsRunningEval] = useState(false);
  const [pendingEvalOperations, setPendingEvalOperations] = useState<PendingEvalOperation[]>([]);
  const [activeView, setActiveView] = useState<InsightsView>("eval");
  const [caseFilter, setCaseFilter] = useState<EvalCaseFilter>("all");
  const [runFilter, setRunFilter] = useState<EvalRunFilter>("latest_valid");
  const [showAllRuns, setShowAllRuns] = useState(false);
  const [showAllTraces, setShowAllTraces] = useState(false);
  const [showAllTraceChunks, setShowAllTraceChunks] = useState(false);
  const [showAllTraceCitations, setShowAllTraceCitations] = useState(false);
  const isAdmin = user?.role?.name === "admin";
  const evalRecoveryInFlightRef = useRef(false);
  async function refreshEvalOverview(datasetName: string, preferredRunId?: string) {
    if (!token || !isAdmin) {
      return;
    }
    const [nextDatasets, nextRuns, nextDashboard] = await Promise.all([
      api.listEvalDatasets(token),
      api.listEvalRuns(token),
      api.getEvalDashboard(token, datasetName, 8),
    ]);
    setEvalDatasets(nextDatasets);
    setRuns(nextRuns);
    setEvalDashboard(nextDashboard);

    const datasetRuns = nextRuns.filter((run) => run.dataset_name === datasetName);
    const targetRun = preferredRunId ? nextRuns.find((run) => run.id === preferredRunId) : defaultEvalRun(datasetRuns);
    if (targetRun) {
      setSelectedRun(await api.getEvalRun(token, targetRun.id));
    } else {
      setSelectedRun(null);
    }
  }

  async function handleSelectDataset(datasetName: string) {
    setSelectedDatasetName(datasetName);
    setRunFilter("latest_valid");
    setShowAllRuns(false);
    setError(null);
    try {
      await refreshEvalOverview(datasetName);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "加载评测数据集失败。");
    }
  }

  function syncPendingEvalOperations() {
    setPendingEvalOperations(listPendingEvalOperations());
  }

  function evalRunReturnedMessage(run: EvalRunDetailRead, prefix: string): string {
    if (run.status === "queued") {
      return `${prefix}：${formatEvalDatasetName(run.dataset_name)}已进入后台队列。`;
    }
    if (run.status === "running") {
      return `${prefix}：${formatEvalDatasetName(run.dataset_name)}仍在运行。`;
    }
    if (run.status === "failed") {
      return isConnectionFailure(run.error_text)
        ? `${prefix}：${formatEvalDatasetName(run.dataset_name)}因上游连接中断提前结束。`
        : `${prefix}：${formatEvalDatasetName(run.dataset_name)}运行失败。`;
    }
    return `${prefix}：${formatEvalDatasetName(run.dataset_name)}，共运行 ${run.results.length} 条用例。`;
  }

  async function acceptEvalRunResponse(operation: PendingEvalOperation, run: EvalRunDetailRead, prefix: string) {
    if (run.status === "queued" || run.status === "running") {
      touchPendingEvalOperation(operation.id);
      setSelectedRun(run);
      await refreshEvalOverview(run.dataset_name, run.id);
      setActiveView("eval");
      setStatusMessage(evalRunReturnedMessage(run, prefix));
      return;
    }
    removePendingEvalOperation(operation.id);
    setSelectedRun(run);
    await refreshEvalOverview(run.dataset_name, run.id);
    setActiveView("eval");
    setStatusMessage(evalRunReturnedMessage(run, prefix));
  }

  async function recoverPendingEvalRuns() {
    if (!token || !isAdmin || evalRecoveryInFlightRef.current) {
      return;
    }
    evalRecoveryInFlightRef.current = true;
    const operations = listPendingEvalOperations();
    setPendingEvalOperations(operations);
    if (!operations.length) {
      evalRecoveryInFlightRef.current = false;
      return;
    }

    try {
      for (const operation of operations) {
        try {
          const currentRuns = await api.listEvalRuns(token);
          const existingRun = findEvalRunByClientRequestId(currentRuns, operation.id);
          if (existingRun) {
            const detail = await api.getEvalRun(token, existingRun.id);
            await acceptEvalRunResponse(operation, detail, "已恢复评测结果");
            continue;
          }

          const run = await api.runEval(token, {
            dataset_name: operation.datasetName,
            top_k: operation.topK,
            seed_demo_cases: operation.seedDemoCases,
            client_request_id: operation.id,
          });
          await acceptEvalRunResponse(operation, run, "已恢复评测请求");
        } catch (nextError) {
          touchPendingEvalOperation(operation.id, nextError instanceof Error ? nextError.message : "恢复评测失败。");
          if (nextError instanceof ApiError && (nextError.status === 0 || nextError.status === 409)) {
            setStatusMessage(`评测仍在后台处理：${formatEvalDatasetName(operation.datasetName)}，刷新后会继续恢复。`);
          } else {
            setError(nextError instanceof Error ? nextError.message : "恢复评测失败。");
          }
        } finally {
          syncPendingEvalOperations();
        }
      }
    } finally {
      evalRecoveryInFlightRef.current = false;
      syncPendingEvalOperations();
    }
  }
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
    syncPendingEvalOperations();
    refreshEvalOverview(selectedDatasetName).catch((nextError) =>
      setError(nextError instanceof Error ? nextError.message : "加载评测记录失败。"),
    );
    recoverPendingEvalRuns().catch((nextError) =>
      setError(nextError instanceof Error ? nextError.message : "恢复评测请求失败。"),
    );
  }, [isAdmin, token]);

  useEffect(() => {
    setSelectedResultId(selectedRun?.results[0]?.id ?? null);
    setCaseFilter("all");
  }, [selectedRun?.id]);
  useEffect(() => {
    if (!token || !isAdmin || pendingEvalOperations.length === 0) {
      return;
    }
    const intervalId = window.setInterval(() => {
      recoverPendingEvalRuns().catch((nextError) =>
        setError(nextError instanceof Error ? nextError.message : "恢复评测请求失败。"),
      );
    }, 5000);
    return () => window.clearInterval(intervalId);
  }, [isAdmin, pendingEvalOperations.length, token]);

  useEffect(() => {
    setShowAllTraceChunks(false);
    setShowAllTraceCitations(false);
  }, [selectedTrace?.id]);

  async function handleRunEval() {
    const datasetName = selectedDatasetName;
    const existingPendingOperation = listPendingEvalOperations(datasetName)[0] ?? null;
    const existingActiveRun = runs.find(
      (run) => run.dataset_name === datasetName && (run.status === "queued" || run.status === "running"),
    );
    if (!token || isRunningEval || existingPendingOperation || existingActiveRun) {
      if (existingPendingOperation || existingActiveRun) {
        setStatusMessage(`评测仍在后台处理：${formatEvalDatasetName(datasetName)}，刷新后会继续恢复。`);
      }
      return;
    }

    const operation = createPendingEvalOperation({
      datasetName,
      topK: evalTopK,
      seedDemoCases: datasetName.startsWith("demo_"),
    });

    syncPendingEvalOperations();
    setError(null);
    setIsRunningEval(true);
    setStatusMessage(`正在运行${formatEvalDatasetName(datasetName)}...`);
    try {
      const run = await api.runEval(token, {
        dataset_name: datasetName,
        top_k: operation.topK,
        seed_demo_cases: operation.seedDemoCases,
        client_request_id: operation.id,
      });

      setError(null);
      await acceptEvalRunResponse(operation, run, run.status === "completed" ? "评测完成" : "评测已返回");
    } catch (nextError) {
      const keepPendingOperation = nextError instanceof ApiError && (nextError.status === 0 || nextError.status === 409);
      if (keepPendingOperation) {
        touchPendingEvalOperation(operation.id, nextError instanceof Error ? nextError.message : undefined);
        setStatusMessage(`评测仍在后台处理：${formatEvalDatasetName(datasetName)}，刷新后会自动恢复。`);
        setError(null);
      } else {
        removePendingEvalOperation(operation.id);
        setError(nextError instanceof Error ? nextError.message : "评测运行失败。");
        setStatusMessage(null);
      }
    } finally {
      syncPendingEvalOperations();
      setIsRunningEval(false);
    }
  }
  async function handleSelectRun(runId: string) {
    if (!token) {
      return;
    }
    try {
      const detail = await api.getEvalRun(token, runId);
      setSelectedRun(detail);
      setActiveView("eval");
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
      setActiveView("trace");
    } catch (nextError) {
      setStatusMessage(nextError instanceof Error ? nextError.message : "加载追踪详情失败。");
    }
  }

  const summary = selectedRun?.summary_json ?? null;
  const answerSummary = asRecord(asRecord(summary?.case_type_breakdown)?.answer_expected);
  const refusalSummary = asRecord(asRecord(summary?.case_type_breakdown)?.refusal_expected);
  const selectedRunInfraFailure = selectedRun ? selectedRun.status === "failed" && isConnectionFailure(selectedRun.error_text) : false;
  const selectedDataset = evalDatasets.find((dataset) => dataset.dataset_name === selectedDatasetName) ?? null;
  const selectedDatasetPendingEval =
    pendingEvalOperations.find((operation) => operation.datasetName === selectedDatasetName) ?? null;
  const datasetRuns = runs.filter((run) => run.dataset_name === selectedDatasetName);
  const latestCompletedRun = datasetRuns.find((run) => isCompletedEvalRun(run)) ?? null;
  const baselineRun = datasetRuns.find((run) => isFullPassEvalRun(run)) ?? latestCompletedRun;
  const baselineRunId = baselineRun?.id ?? null;
  const selectedRunIsBaseline =
    Boolean(baselineRunId) && selectedRun?.id === baselineRunId && selectedRun?.status === "completed";
  const plannedTotalCases = selectedRun?.total_cases ?? 0;
  const completedCases = typeof summary?.total_cases === "number" ? summary.total_cases : 0;
  const totalCases = selectedRunInfraFailure
    ? plannedTotalCases
    : typeof summary?.total_cases === "number"
      ? summary.total_cases
      : plannedTotalCases;
  const activeRuns = datasetRuns.filter((run) => run.status === "queued" || run.status === "running");
  const selectedDatasetActiveRun = activeRuns[0] ?? null;
  const validRuns = datasetRuns.filter((run) => isCompletedEvalRun(run));
  const connectionFailureRuns = datasetRuns.filter((run) => run.status === "failed" && isConnectionFailure(run.error_text));
  const filteredRuns = runFilter === "latest_valid" ? [...activeRuns, ...validRuns] : connectionFailureRuns;
  const visibleRuns = showAllRuns ? filteredRuns : filteredRuns.slice(0, DEFAULT_VISIBLE_RUNS);
  const hiddenRunCount = Math.max(0, filteredRuns.length - visibleRuns.length);
  const runFilterBadgeText =
    runFilter === "latest_valid"
      ? activeRuns.length
        ? `${activeRuns.length} 条处理中，${validRuns.length} 条完整`
        : `${validRuns.length} 条完整`
      : `${connectionFailureRuns.length} 条中断`;
  const visibleTraces = showAllTraces ? traces : traces.slice(0, DEFAULT_VISIBLE_TRACES);
  const hiddenTraceCount = Math.max(0, traces.length - visibleTraces.length);

  const filteredResults =
    selectedRun?.results.filter((result) => {
      if (caseFilter === "all") {
        return true;
      }
      const annotations = asRecord(result.details_json.case_annotations);
      return caseTypeBucket(annotations?.expected_outcome) === caseFilter;
    }) ?? [];

  const selectedResult =
    filteredResults.find((result) => result.id === selectedResultId) ??
    filteredResults[0] ??
    selectedRun?.results[0] ??
    null;

  useEffect(() => {
    if (!selectedResult) {
      setSelectedResultId(null);
      return;
    }
    if (selectedResult.id !== selectedResultId) {
      setSelectedResultId(selectedResult.id);
    }
  }, [selectedResult?.id]);

  const retrievedChunks = asArray<Record<string, unknown>>(selectedTrace?.retrieved_chunks_json);
  const selectedCitations = asArray<Record<string, unknown>>(selectedTrace?.selected_citations_json);
  const selectedTraceMetadata = asRecord(selectedTrace?.trace_metadata);
  const selectedTraceRetrievalDebug = asRecord(selectedTraceMetadata?.retrieval_debug);
  const traceLexicalQueries = asStringList(selectedTraceRetrievalDebug?.lexical_queries);
  const traceRewriteStrategies = asStringList(selectedTraceRetrievalDebug?.query_rewrite_strategies);
  const visibleRetrievedChunks = showAllTraceChunks ? retrievedChunks : retrievedChunks.slice(0, DEFAULT_VISIBLE_TRACE_ITEMS);
  const visibleSelectedCitations = showAllTraceCitations
    ? selectedCitations
    : selectedCitations.slice(0, DEFAULT_VISIBLE_TRACE_ITEMS);

  function renderSummaryCard(label: string, value: string, helper?: string) {
    return (
      <div className="metric-card">
        <span>{label}</span>
        <strong>{value}</strong>
        {helper ? <p className="muted">{helper}</p> : null}
      </div>
    );
  }

  const latestTrendPoint = evalDashboard?.trend.length ? evalDashboard.trend[evalDashboard.trend.length - 1] : null;
  const previousTrendPoint = evalDashboard && evalDashboard.trend.length > 1 ? evalDashboard.trend[evalDashboard.trend.length - 2] : null;

  function renderCaseTypeCard(bucket: Record<string, unknown> | null, fallbackTitle: string) {
    const title = typeof bucket?.label === "string" ? bucket.label : fallbackTitle;
    return (
      <div className="list-card">
        <div className="list-card-topline">
          <strong>{title}</strong>
          <StatusBadge tone="info">
            {valueFromRecord(bucket, "pass_count", "0")} / {valueFromRecord(bucket, "total_cases", "0")}
          </StatusBadge>
        </div>
        <div className="metadata-subline">
          <span>综合 {scoreFromRecord(bucket, "overall_score_avg")}</span>
          <span>权限 {scoreFromRecord(bucket, "permission_isolation_pass_rate")}</span>
        </div>
        <p className="muted">
          检索 {scoreFromRecord(bucket, "retrieval_hit_rate_avg")} · 引用 {scoreFromRecord(bucket, "citation_accuracy_avg")}
        </p>
      </div>
    );
  }

  function renderTrendOverview() {
    if (!latestTrendPoint) {
      return <div className="empty-state">当前数据集还没有完整评测记录。</div>;
    }
    return (
      <div className="eval-dashboard-grid">
        {renderSummaryCard(
          "趋势综合分",
          latestTrendPoint.overall_score_avg.toFixed(2),
          deltaText(latestTrendPoint.overall_score_avg, previousTrendPoint?.overall_score_avg) ?? undefined,
        )}
        {renderSummaryCard(
          "检索命中",
          percentText(latestTrendPoint.retrieval_hit_rate_avg),
          deltaText(latestTrendPoint.retrieval_hit_rate_avg, previousTrendPoint?.retrieval_hit_rate_avg) ?? undefined,
        )}
        {renderSummaryCard(
          "引用准确",
          percentText(latestTrendPoint.citation_accuracy_avg),
          deltaText(latestTrendPoint.citation_accuracy_avg, previousTrendPoint?.citation_accuracy_avg) ?? undefined,
        )}
        {renderSummaryCard(
          "答案支撑",
          percentText(latestTrendPoint.answer_faithfulness_avg),
          deltaText(latestTrendPoint.answer_faithfulness_avg, previousTrendPoint?.answer_faithfulness_avg) ?? undefined,
        )}
      </div>
    );
  }

  function renderTrendHistory() {
    const points = evalDashboard?.trend ?? [];
    if (!points.length) {
      return null;
    }
    return (
      <div className="list-card">
        <div className="list-card-topline">
          <strong>历史趋势</strong>
          <StatusBadge tone="info">{points.length}</StatusBadge>
        </div>
        <div className="eval-trend-list">
          {points.map((point) => (
            <div className="eval-trend-row" key={point.run_id}>
              <span>{formatDateTime(point.created_at)}</span>
              <strong>{point.overall_score_avg.toFixed(2)}</strong>
              <span>{point.pass_count} / {point.total_cases}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  function renderFailureModes() {
    const modes = evalDashboard?.failure_modes ?? [];
    return (
      <div className="list-card">
        <div className="list-card-topline">
          <strong>失败原因</strong>
          <StatusBadge tone={modes.length ? "warning" : "success"}>{modes.length ? `${modes.length} 类` : "无"}</StatusBadge>
        </div>
        {modes.length ? (
          <div className="stack dense-stack">
            {modes.map((mode) => (
              <div className="eval-failure-row" key={mode.key}>
                <div>
                  <strong>{mode.label}</strong>
                  <p className="muted">{mode.example_case_names.length ? mode.example_case_names.join("、") : "暂无样例"}</p>
                </div>
                <StatusBadge tone="warning">{mode.count}</StatusBadge>
              </div>
            ))}
          </div>
        ) : (
          <p className="muted">最近完整运行没有未通过样例。</p>
        )}
      </div>
    );
  }
  function renderCaseListItem(result: EvalResultRowRead) {
    const details = result.details_json;
    const annotations = asRecord(details.case_annotations);
    const overallBreakdown = asRecord(asRecord(details.metric_breakdown)?.overall);
    const permissionBreakdown = asRecord(asRecord(details.metric_breakdown)?.permission_isolation);
    return (
      <button
        className={`list-card text-left ${selectedResult?.id === result.id ? "is-selected" : ""}`}
        key={result.id}
        onClick={() => setSelectedResultId(result.id)}
        type="button"
      >
        <div className="list-card-topline">
          <strong>{String(details.case_name ?? result.case_id)}</strong>
          <StatusBadge tone={caseStatusTone(result)}>{result.overall_pass ? "通过" : "需复核"}</StatusBadge>
        </div>
        <p>{truncate(String(details.question ?? details.answer_excerpt ?? ""), 78)}</p>
        <div className="metadata-subline">
          <span>{expectedOutcomeText(annotations?.expected_outcome)}</span>
          <span>综合 {scoreText(overallBreakdown?.score) ?? "-"}</span>
          <span>权限 {scoreText(permissionBreakdown?.score) ?? "-"}</span>
        </div>
      </button>
    );
  }

  function renderSelectedCaseDetail(result: EvalResultRowRead | null) {
    if (!result) {
      return <div className="empty-state">当前筛选下没有可展示的样例。</div>;
    }

    const details = result.details_json;
    const caseAnnotations = asRecord(details.case_annotations);
    const metricBreakdownRecord = asRecord(details.metric_breakdown);
    const overallBreakdown = asRecord(metricBreakdownRecord?.overall);
    const permissionBreakdown = asRecord(metricBreakdownRecord?.permission_isolation);
    const retrievalBreakdown = asRecord(metricBreakdownRecord?.retrieval);
    const citationBreakdown = asRecord(metricBreakdownRecord?.citation);
    const faithfulnessBreakdown = asRecord(metricBreakdownRecord?.faithfulness);
    const humanReview = asRecord(details.human_review);
    const annotationSource =
      caseAnnotations?.source === "demo_annotations"
        ? "演示标注"
        : caseAnnotations?.source === "legacy_case_fields"
          ? "兼容字段"
          : "未标注";

    return (
      <div className="insights-detail-panel">
        <div className="panel-heading">
          <h3>{String(details.case_name ?? result.case_id)}</h3>
          <p>{String(details.question ?? "")}</p>
        </div>
        <div className="metadata-subline">
          <span>{expectedOutcomeText(caseAnnotations?.expected_outcome)}</span>
          <span>执行用户：{result.acting_user_email}</span>
          <span>来源：{annotationSource}</span>
        </div>
        <div className="insights-summary-grid insights-summary-grid-compact">
          {renderSummaryCard("检索", result.retrieval_hit_rate.toFixed(2))}
          {renderSummaryCard("引用", result.citation_accuracy.toFixed(2))}
          {renderSummaryCard("忠实性", result.answer_faithfulness.toFixed(2))}
          {renderSummaryCard("权限", scoreText(permissionBreakdown?.score) ?? "-")}
        </div>

        <div className="list-card">
          <div className="list-card-topline">
            <strong>系统判断</strong>
            <StatusBadge tone={caseStatusTone(result)}>{result.overall_pass ? "通过" : "需复核"}</StatusBadge>
          </div>
          <p>{String(overallBreakdown?.reason ?? "暂无综合判断说明。")}</p>
          <div className="metadata-subline">
            <span>综合分：{scoreText(overallBreakdown?.score) ?? "-"}</span>
            <span>人工复核：{humanReview?.recommended ? "建议复核" : "可直接通过"}</span>
          </div>
          {typeof humanReview?.reason === "string" && humanReview.reason ? <p className="muted">{humanReview.reason}</p> : null}
        </div>

        <div className="list-card">
          <div className="list-card-topline">
            <strong>答案摘要</strong>
          </div>
          <p className="insights-answer-preview">{truncate(String(details.answer_text ?? details.answer_excerpt ?? ""), 340)}</p>
        </div>

        <div className="list-card">
          <div className="list-card-topline">
            <strong>样例标注</strong>
          </div>
          <div className="metadata-subline">
            <span>期望证据：{listText(asStringList(caseAnnotations?.expected_evidence_titles))}</span>
          </div>
          <div className="metadata-subline">
            <span>标注事实：{listText(asStringList(caseAnnotations?.expected_key_facts))}</span>
          </div>
          <div className="metadata-subline">
            <span>受限事实：{listText(asStringList(caseAnnotations?.forbidden_key_facts))}</span>
          </div>
          {asStringList(details.forbidden_key_fact_hits).length ? (
            <div className="metadata-subline">
              <span>实际命中受限事实：{listText(asStringList(details.forbidden_key_fact_hits))}</span>
            </div>
          ) : null}
        </div>

        <details className="execution-trace-secondary">
          <summary>展开评分依据</summary>
          <div className="execution-trace-secondary-list">
            <div className="execution-trace-item">
              <div className="execution-trace-topline">
                <strong>检索与引用</strong>
              </div>
              <div className="metadata-subline">
                <span>召回率：{scoreText(retrievalBreakdown?.recall) ?? "-"}</span>
                <span>Precision：{scoreText(retrievalBreakdown?.precision) ?? "-"}</span>
                <span>AP@K：{scoreText(retrievalBreakdown?.average_precision) ?? "-"}</span>
                <span>nDCG：{scoreText(retrievalBreakdown?.ranking_score) ?? "-"}</span>
                <span>MRR：{scoreText(retrievalBreakdown?.mrr) ?? "-"}</span>
              </div>
              <div className="metadata-subline">
                <span>检索事实覆盖：{scoreText(retrievalBreakdown?.retrieved_fact_recall) ?? "-"}</span>
                <span>越权召回率：{scoreText(retrievalBreakdown?.unauthorized_retrieval_rate) ?? "-"}</span>
              </div>
              <div className="metadata-subline">
                <span>引用 Precision：{scoreText(citationBreakdown?.precision) ?? "-"}</span>
                <span>Recall：{scoreText(citationBreakdown?.recall) ?? "-"}</span>
                <span>F1：{scoreText(citationBreakdown?.f1) ?? "-"}</span>
                <span>引用事实覆盖：{scoreText(citationBreakdown?.evidence_fact_recall) ?? "-"}</span>
                <span>越权引用率：{scoreText(citationBreakdown?.unauthorized_citation_rate) ?? "-"}</span>
              </div>
              <div className="metadata-subline">
                <span>命中文档：{listText(asStringList(details.matched_expected_titles))}</span>
              </div>
              {asStringList(details.missing_expected_titles).length ? (
                <div className="metadata-subline">
                  <span>缺失文档：{listText(asStringList(details.missing_expected_titles))}</span>
                </div>
              ) : null}
              {typeof retrievalBreakdown?.formula === "string" ? (
                <p className="muted">口径：{retrievalBreakdown.formula}</p>
              ) : null}
            </div>

            <div className="execution-trace-item">
              <div className="execution-trace-topline">
                <strong>答案与隔离</strong>
              </div>
              {faithfulnessBreakdown?.mode === "refusal_expected" ? (
                <div className="metadata-subline">
                  <span>显式拒答：{faithfulnessBreakdown?.explicit_refusal ? "是" : "否"}</span>
                  <span>受限事实清洁度：{scoreText(faithfulnessBreakdown?.protected_fact_cleanliness) ?? "-"}</span>
                </div>
              ) : (
                <>
                  <div className="metadata-subline">
                    <span>答案事实覆盖：{scoreText(faithfulnessBreakdown?.answer_fact_recall) ?? "-"}</span>
                    <span>证据事实覆盖：{scoreText(faithfulnessBreakdown?.evidence_fact_recall) ?? "-"}</span>
                    <span>支撑 Precision：{scoreText(faithfulnessBreakdown?.supported_fact_precision) ?? "-"}</span>
                  </div>
                  <div className="metadata-subline">
                    <span>支撑 Recall：{scoreText(faithfulnessBreakdown?.supported_fact_recall) ?? "-"}</span>
                    <span>支撑 F1：{scoreText(faithfulnessBreakdown?.support_f1) ?? "-"}</span>
                  </div>
                </>
              )}
              <div className="metadata-subline">
                <span>召回泄漏：{scoreText(permissionBreakdown?.retrieval_leak_ratio) ?? "-"}</span>
                <span>引用泄漏：{scoreText(permissionBreakdown?.citation_leak_ratio) ?? "-"}</span>
                <span>答案泄漏：{scoreText(permissionBreakdown?.answer_leak_ratio) ?? "-"}</span>
              </div>
              {asStringList(details.matched_answer_facts).length ? (
                <div className="metadata-subline">
                  <span>命中事实：{listText(asStringList(details.matched_answer_facts))}</span>
                </div>
              ) : null}
              {asStringList(details.missing_answer_facts).length ? (
                <div className="metadata-subline">
                  <span>缺失事实：{listText(asStringList(details.missing_answer_facts))}</span>
                </div>
              ) : null}
              {asStringList(details.matched_evidence_facts).length ? (
                <div className="metadata-subline">
                  <span>证据命中事实：{listText(asStringList(details.matched_evidence_facts))}</span>
                </div>
              ) : null}
              {asStringList(details.supported_answer_facts).length ? (
                <div className="metadata-subline">
                  <span>已支撑事实：{listText(asStringList(details.supported_answer_facts))}</span>
                </div>
              ) : null}
              {asStringList(details.unsupported_answer_facts).length ? (
                <div className="metadata-subline">
                  <span>待核实事实：{listText(asStringList(details.unsupported_answer_facts))}</span>
                </div>
              ) : null}
              {typeof faithfulnessBreakdown?.formula === "string" ? (
                <p className="muted">口径：{faithfulnessBreakdown.formula}</p>
              ) : null}
              {typeof permissionBreakdown?.formula === "string" ? (
                <p className="muted">权限口径：{permissionBreakdown.formula}</p>
              ) : null}
            </div>
          </div>
        </details>
      </div>
    );
  }

  return (
    <div className="page-stack">
      <PageHeader
        title="评测与追踪"
        description="先看评测结果，再按需下钻样例和追踪细节。"
        actions={null}
      />
      <ErrorNotice message={error} />
      {statusMessage ? <div className="info-block">{statusMessage}</div> : null}

      <div className="segmented-control">
        <button
          className={`segmented-button ${activeView === "eval" ? "is-active" : ""}`}
          onClick={() => setActiveView("eval")}
          type="button"
        >
          评测
        </button>
        <button
          className={`segmented-button ${activeView === "trace" ? "is-active" : ""}`}
          onClick={() => setActiveView("trace")}
          type="button"
        >
          追踪
        </button>
      </div>

      {activeView === "eval" ? (
        <div className="page-grid insights-workspace">
          <section className="panel stack">
            <div className="panel-header">
              <h3>最近评测</h3>
              <StatusBadge tone={isAdmin ? "info" : "neutral"}>{isAdmin ? runFilterBadgeText : "仅管理员"}</StatusBadge>
            </div>
            {isAdmin ? (
              <div className="eval-control-panel">
                <label>
                  <span>评测集</span>
                  <select value={selectedDatasetName} onChange={(event) => handleSelectDataset(event.target.value)}>
                    {evalDatasets.length ? (
                      evalDatasets.map((dataset) => (
                        <option key={dataset.dataset_name} value={dataset.dataset_name}>
                          {dataset.display_name}（{dataset.case_count}）
                        </option>
                      ))
                    ) : (
                      <option value={selectedDatasetName}>{formatEvalDatasetName(selectedDatasetName)}</option>
                    )}
                  </select>
                </label>
                <label>
                  <span>Top-K</span>
                  <input
                    max={10}
                    min={1}
                    onChange={(event) => setEvalTopK(Math.min(10, Math.max(1, Number(event.target.value) || 1)))}
                    type="number"
                    value={evalTopK}
                  />
                </label>
                <button
                  className="primary-button"
                  disabled={isRunningEval || Boolean(selectedDatasetPendingEval) || Boolean(selectedDatasetActiveRun)}
                  onClick={handleRunEval}
                  type="button"
                >
                  {selectedDatasetPendingEval || selectedDatasetActiveRun ? "后台处理中" : isRunningEval ? "正在运行" : "运行评测"}
                </button>
                <p className="muted">
                  {selectedDatasetPendingEval
                    ? `已有评测请求在恢复中，ID：${selectedDatasetPendingEval.id}`
                    : selectedDatasetActiveRun
                      ? `已有评测在后台${evalRunStatusLabel(selectedDatasetActiveRun.status, selectedDatasetActiveRun.error_text)}`
                      : selectedDataset
                        ? `${selectedDataset.completed_run_count} 次完整运行，${selectedDataset.failed_run_count} 次中断`
                        : "暂无数据集统计"}
                </p>
              </div>
            ) : null}
            {isAdmin ? (
              <>
                <div className="run-filter-bar">
                  <p className="muted">
                    {runFilter === "latest_valid"
                      ? activeRuns.length
                        ? "运行中的评测会固定显示在顶部，完成后进入完整记录。"
                        : "这里只显示完整跑完的评测记录。"
                      : "这里只显示因上游连接中断而提前结束的记录。"}
                  </p>
                  <div className="run-filter-actions">
                    {runFilter === "latest_valid" && connectionFailureRuns.length ? (
                      <button className="inline-filter-button" onClick={() => setRunFilter("connection_failures")} type="button">
                        查看连接中断记录
                      </button>
                    ) : null}
                    {runFilter === "connection_failures" ? (
                      <button className="inline-filter-button" onClick={() => setRunFilter("latest_valid")} type="button">
                        返回有效结果
                      </button>
                    ) : null}
                  </div>
                </div>
                <div className="insights-scroll-list">
                  <div className="stack dense-stack">
                    {visibleRuns.length ? (
                      visibleRuns.map((run) => (
                        <button
                          className={`list-card text-left ${selectedRun?.id === run.id ? "is-selected" : ""}`}
                          key={run.id}
                          onClick={() => handleSelectRun(run.id)}
                          type="button"
                        >
                          <div className="list-card-topline">
                            <strong>{formatEvalDatasetName(run.dataset_name)}</strong>
                            <StatusBadge tone={evalRunStatusTone(run.status, run.error_text)}>
                              {evalRunStatusLabel(run.status, run.error_text)}
                            </StatusBadge>
                          </div>
                          <p>{formatDateTime(run.created_at)}</p>
                          <div className="metadata-subline">
                            <span>{runProgressText(run)}</span>
                            {baselineRunId === run.id ? <span>当前基线</span> : null}
                          </div>
                        </button>
                      ))
                    ) : (
                      <p className="muted">当前筛选下暂无评测记录。</p>
                    )}
                  </div>
                </div>
              </>
            ) : (
              <p className="muted">评测接口仅管理员可用，请使用管理员账号运行演示评测。</p>
            )}
            {isAdmin && hiddenRunCount ? (
              <button className="secondary-button" onClick={() => setShowAllRuns((value) => !value)} type="button">
                {showAllRuns ? "收起历史记录" : `显示其余 ${hiddenRunCount} 条记录`}
              </button>
            ) : null}
          </section>

          <section className="panel stack">
            {isAdmin && selectedRun ? (
              <>
                <div className="panel-heading">
                  <h3>{formatEvalDatasetName(selectedRun.dataset_name)}</h3>
                  <p>{formatDateTime(selectedRun.created_at)}</p>
                </div>
                {selectedRunIsBaseline ? (
                  <div className="metadata-subline">
                    <span>当前基线</span>
                    <span>
                      最近一轮完整成功运行（{runPassCount(selectedRun)} / {selectedRun.total_cases}）
                    </span>
                  </div>
                ) : null}
                {renderTrendOverview()}
                <div className="insights-summary-cards">
                  {renderTrendHistory()}
                  {renderFailureModes()}
                </div>
                <div className="insights-summary-grid">
                  {renderSummaryCard("总用例数", String(totalCases))}
                  {renderSummaryCard(
                    selectedRunInfraFailure ? "已完成用例" : "通过数",
                    selectedRunInfraFailure
                      ? `${completedCases} / ${plannedTotalCases}`
                      : `${valueFromRecord(summary, "pass_count", "0")} / ${totalCases}`,
                  )}
                  {renderSummaryCard(
                    "综合得分",
                    selectedRunInfraFailure ? "-" : scoreFromRecord(summary, "overall_score_avg"),
                  )}
                  {renderSummaryCard(
                    "权限通过率",
                    selectedRunInfraFailure ? "-" : scoreFromRecord(summary, "permission_isolation_pass_rate"),
                  )}
                </div>
                <div className="insights-summary-cards">
                  {renderCaseTypeCard(answerSummary, "回答型")}
                  {renderCaseTypeCard(refusalSummary, "拒答/权限型")}
                </div>
                {formatEvalRunError(selectedRun) ? <ErrorNotice message={formatEvalRunError(selectedRun)!} /> : null}

                <div className="segmented-control">
                  <button
                    className={`segmented-button ${caseFilter === "all" ? "is-active" : ""}`}
                    onClick={() => setCaseFilter("all")}
                    type="button"
                  >
                    全部样例
                  </button>
                  <button
                    className={`segmented-button ${caseFilter === "answer_expected" ? "is-active" : ""}`}
                    onClick={() => setCaseFilter("answer_expected")}
                    type="button"
                  >
                    回答型
                  </button>
                  <button
                    className={`segmented-button ${caseFilter === "refusal_expected" ? "is-active" : ""}`}
                    onClick={() => setCaseFilter("refusal_expected")}
                    type="button"
                  >
                    拒答/权限型
                  </button>
                </div>

                <div className="insights-main-split">
                  <div className="stack">
                    <div className="subsection-header">
                      <h4>样例列表</h4>
                      <span className="muted">{filteredResults.length} 条</span>
                    </div>
                    <div className="insights-scroll-list">
                      <div className="stack dense-stack">
                        {filteredResults.length ? (
                          filteredResults.map((result) => renderCaseListItem(result))
                        ) : (
                          <div className="empty-state">当前筛选下没有样例。</div>
                        )}
                      </div>
                    </div>
                  </div>
                  {renderSelectedCaseDetail(selectedResult)}
                </div>
              </>
            ) : (
              <div className="empty-state">先选择一条评测记录，再看这次评测的摘要和样例详情。</div>
            )}
          </section>
        </div>
      ) : (
        <div className="page-grid insights-workspace">
          <section className="panel stack">
            <div className="panel-header">
              <h3>最近追踪</h3>
              <StatusBadge tone="info">{traces.length}</StatusBadge>
            </div>
            <div className="insights-scroll-list">
              <div className="stack dense-stack">
                {visibleTraces.length ? (
                  visibleTraces.map((trace) => (
                    <button
                      className={`list-card text-left ${selectedTrace?.id === trace.id ? "is-selected" : ""}`}
                      key={trace.id}
                      onClick={() => handleSelectTrace(trace.id)}
                      type="button"
                    >
                      <div className="list-card-topline">
                        <strong>{formatTraceType(trace.trace_type)}</strong>
                        <span>{formatDateTime(trace.created_at)}</span>
                      </div>
                      <p>{truncate(trace.query_text ?? "暂无查询内容", 96)}</p>
                    </button>
                  ))
                ) : (
                  <p className="muted">暂无追踪记录，先去问答页发起一次提问。</p>
                )}
              </div>
            </div>
            {hiddenTraceCount ? (
              <button className="secondary-button" onClick={() => setShowAllTraces((value) => !value)} type="button">
                {showAllTraces ? "收起历史追踪" : `显示其余 ${hiddenTraceCount} 条追踪`}
              </button>
            ) : null}
          </section>

          <section className="panel stack">
            {selectedTrace ? (
              <>
                <div className="panel-heading">
                  <h3>{formatTraceType(selectedTrace.trace_type)}</h3>
                  <p>{selectedTrace.query_text ?? "暂无查询内容"}</p>
                </div>
                <div className="metadata-grid">
                  <span>模型：{selectedTrace.model_name ?? "-"}</span>
                  <span>耗时：{selectedTrace.latency_ms ?? "-"} ms</span>
                  <span>输入 Token：{selectedTrace.prompt_tokens ?? "-"}</span>
                  <span>输出 Token：{selectedTrace.completion_tokens ?? "-"}</span>
                </div>
                {selectedTrace.error_text ? <ErrorNotice message={selectedTrace.error_text} /> : null}

                {selectedTraceRetrievalDebug ? (
                  <div className="list-card">
                    <div className="list-card-topline">
                      <strong>查询增强与检索计划</strong>
                    </div>
                    <div className="metadata-subline">
                      <span>可访问文档：{String(selectedTraceRetrievalDebug.accessible_document_count ?? "-")}</span>
                      <span>关键词检索候选：{String(selectedTraceRetrievalDebug.lexical_candidate_count ?? "-")}</span>
                      <span>向量检索候选：{String(selectedTraceRetrievalDebug.vector_candidate_count ?? "-")}</span>
                    </div>
                    <div className="metadata-subline">
                      <span>重排前：{String(selectedTraceRetrievalDebug.pre_rerank_count ?? "-")}</span>
                      <span>重排后：{String(selectedTraceRetrievalDebug.post_rerank_count ?? "-")}</span>
                      <span>策略：{String(selectedTraceRetrievalDebug.rerank_strategy ?? "-")}</span>
                    </div>
                    <p className="muted">原始问题：{selectedTrace.query_text ?? "-"}</p>
                    <p className="muted">检索语句：{String(selectedTraceRetrievalDebug.retrieval_query ?? selectedTrace.query_text ?? "-")}</p>
                    <div className="metadata-subline">
                      <span>改写已应用：{selectedTraceRetrievalDebug.query_rewrite_applied ? "是" : "否"}</span>
                      <span>改写策略：{traceRewriteStrategies.length ? traceRewriteStrategies.join("、") : "无"}</span>
                    </div>
                    {typeof selectedTraceRetrievalDebug.query_plan_candidate_count === "number" &&
                    selectedTraceRetrievalDebug.query_plan_candidate_count > 1 ? (
                      <div className="metadata-subline">
                        <span>候选方案：{selectedTraceRetrievalDebug.query_plan_candidate_count} 个</span>
                        <span>选中方案：{String(selectedTraceRetrievalDebug.query_plan_selected ?? "-")}</span>
                      </div>
                    ) : null}
                    {selectedTraceRetrievalDebug.query_plan_selection_reason ? (
                      <p className="muted">选中原因：{String(selectedTraceRetrievalDebug.query_plan_selection_reason)}</p>
                    ) : null}
                    <div className="metadata-subline">
                      <span>改写方式：{String(selectedTraceRetrievalDebug.query_rewrite_provider ?? "rules-only")}</span>
                      <span>改写模型：{String(selectedTraceRetrievalDebug.query_rewrite_model ?? "-")}</span>
                      <span>改写耗时：{String(selectedTraceRetrievalDebug.query_rewrite_latency_ms ?? "-")} ms</span>
                    </div>
                    {traceLexicalQueries.length ? (
                      <div className="metadata-subline">
                        <span>关键词检索变体：{traceLexicalQueries.join(" ｜ ")}</span>
                      </div>
                    ) : null}
                    {typeof selectedTraceRetrievalDebug.fusion_strategy === "string" ? (
                      <p className="muted">融合口径：{selectedTraceRetrievalDebug.fusion_strategy}</p>
                    ) : null}
                  </div>
                ) : null}

                <details className="execution-trace-secondary">
                  <summary>召回分块（{retrievedChunks.length}）</summary>
                  <div className="execution-trace-secondary-list">
                    {visibleRetrievedChunks.length ? (
                      visibleRetrievedChunks.map((chunk, index) => (
                        <div className="execution-trace-item" key={`${String(chunk.chunk_id)}-${index}`}>
                          <div className="execution-trace-topline">
                            <strong>{String(chunk.document_title ?? "文档")}</strong>
                            <span>分块 {String(chunk.chunk_index ?? "-")}</span>
                          </div>
                          <p>{truncate(String(chunk.preview ?? ""), 220)}</p>
                        </div>
                      ))
                    ) : (
                      <p className="muted">暂无召回分块记录。</p>
                    )}
                    {retrievedChunks.length > DEFAULT_VISIBLE_TRACE_ITEMS ? (
                      <button className="secondary-button" onClick={() => setShowAllTraceChunks((value) => !value)} type="button">
                        {showAllTraceChunks ? "收起召回分块" : `展开其余 ${retrievedChunks.length - visibleRetrievedChunks.length} 条`}
                      </button>
                    ) : null}
                  </div>
                </details>

                <details className="execution-trace-secondary">
                  <summary>选中引用（{selectedCitations.length}）</summary>
                  <div className="execution-trace-secondary-list">
                    {visibleSelectedCitations.length ? (
                      visibleSelectedCitations.map((chunk, index) => (
                        <div className="execution-trace-item" key={`${String(chunk.chunk_id)}-selected-${index}`}>
                          <div className="execution-trace-topline">
                            <strong>{String(chunk.document_title ?? "文档")}</strong>
                            <span>v{String(chunk.version_number ?? "-")}</span>
                          </div>
                          <p>{truncate(String(chunk.preview ?? ""), 220)}</p>
                        </div>
                      ))
                    ) : (
                      <p className="muted">暂无引用记录。</p>
                    )}
                    {selectedCitations.length > DEFAULT_VISIBLE_TRACE_ITEMS ? (
                      <button
                        className="secondary-button"
                        onClick={() => setShowAllTraceCitations((value) => !value)}
                        type="button"
                      >
                        {showAllTraceCitations ? "收起引用记录" : `展开其余 ${selectedCitations.length - visibleSelectedCitations.length} 条`}
                      </button>
                    ) : null}
                  </div>
                </details>
              </>
            ) : (
              <div className="empty-state">先选择一条追踪，再看查询、证据和引用。</div>
            )}
          </section>
        </div>
      )}
    </div>
  );
}
