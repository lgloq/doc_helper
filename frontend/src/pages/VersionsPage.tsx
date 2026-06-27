import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { ErrorNotice } from "../components/ErrorNotice";
import { PageHeader } from "../components/PageHeader";
import { SelectField } from "../components/SelectField";
import { StatusBadge } from "../components/StatusBadge";
import { useAppContext } from "../context/AppContext";
import { ApiError, api } from "../lib/api";
import { formatDiffChangeType, formatIngestStatus, formatSummaryProvider } from "../lib/display";
import { truncate } from "../lib/format";
import {
  createPendingDiffSummaryOperation,
  listPendingDiffSummaryOperations,
  removePendingDiffSummaryOperation,
  setPendingOperationJob,
  touchPendingDiffSummaryOperation,
} from "../lib/pendingOperations";
import type { PendingDiffSummaryOperation } from "../lib/pendingOperations";
import type { DocumentDiffRead, DocumentDiffSummaryRead, DocumentRead, DocumentVersionRead } from "../types/api";

interface VersionsPageCache {
  documents: DocumentRead[];
  documentId: string;
  versions: DocumentVersionRead[];
  fromVersionId: string;
  toVersionId: string;
  diff: DocumentDiffRead | null;
  summary: DocumentDiffSummaryRead | null;
}

export function VersionsPage() {
  const { token, getPageCache, setPageCache } = useAppContext();
  const [searchParams, setSearchParams] = useSearchParams();
  const cachedPage = getPageCache<VersionsPageCache>("versions");
  const [documents, setDocuments] = useState<DocumentRead[]>(() => cachedPage?.documents ?? []);
  const [documentId, setDocumentId] = useState<string>(() => cachedPage?.documentId ?? "");
  const [versions, setVersions] = useState<DocumentVersionRead[]>(() => cachedPage?.versions ?? []);
  const [fromVersionId, setFromVersionId] = useState<string>(() => cachedPage?.fromVersionId ?? "");
  const [toVersionId, setToVersionId] = useState<string>(() => cachedPage?.toVersionId ?? "");
  const [diff, setDiff] = useState<DocumentDiffRead | null>(() => cachedPage?.diff ?? null);
  const [summary, setSummary] = useState<DocumentDiffSummaryRead | null>(() => cachedPage?.summary ?? null);
  const [isRawDiffCollapsed, setIsRawDiffCollapsed] = useState(false);
  const [isLoadingDiff, setIsLoadingDiff] = useState(false);
  const [isLoadingSummary, setIsLoadingSummary] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [isSummaryMenuOpen, setIsSummaryMenuOpen] = useState(false);
  const hasAutoLoadContext = useRef(false);
  const didRestoreCachedSelection = useRef(Boolean(cachedPage));
  const summaryCacheRef = useRef<Record<string, DocumentDiffSummaryRead>>({});
  const summaryMenuRef = useRef<HTMLDivElement | null>(null);
  const requestedDocumentId = searchParams.get("documentId");
  const requestedFromVersionId = searchParams.get("fromVersionId");
  const requestedToVersionId = searchParams.get("toVersionId");
  const summaryKey = documentId && fromVersionId && toVersionId ? `${documentId}:${fromVersionId}:${toVersionId}` : "";

  useEffect(() => {
    setPageCache<VersionsPageCache>("versions", {
      documents,
      documentId,
      versions,
      fromVersionId,
      toVersionId,
      diff,
      summary,
    });
  }, [diff, documentId, documents, fromVersionId, setPageCache, summary, toVersionId, versions]);

  function syncLocation(nextDocumentId: string, nextFromVersionId?: string, nextToVersionId?: string) {
    const params = new URLSearchParams(searchParams);
    if (nextDocumentId) {
      params.set("documentId", nextDocumentId);
    } else {
      params.delete("documentId");
    }
    if (nextFromVersionId) {
      params.set("fromVersionId", nextFromVersionId);
    } else {
      params.delete("fromVersionId");
    }
    if (nextToVersionId) {
      params.set("toVersionId", nextToVersionId);
    } else {
      params.delete("toVersionId");
    }
    setSearchParams(params, { replace: true });
  }

  function matchingPendingSummaryOperations(): PendingDiffSummaryOperation[] {
    if (!documentId || !fromVersionId || !toVersionId) {
      return [];
    }
    return listPendingDiffSummaryOperations(documentId).filter(
      (operation) => operation.fromVersionId === fromVersionId && operation.toVersionId === toVersionId,
    );
  }

  async function acceptPendingSummaryJob(operation: PendingDiffSummaryOperation, jobId: string, prefix: string) {
    if (!token) {
      return;
    }
    setPendingOperationJob(operation.id, jobId);
    const job = await api.getJob(token, jobId);
    if (job.status === "queued" || job.status === "running") {
      touchPendingDiffSummaryOperation(operation.id, job.error_text ?? undefined);
      setStatusMessage(`${prefix}：差异摘要仍在后台生成。`);
      return;
    }
    if (job.status === "failed") {
      removePendingDiffSummaryOperation(operation.id);
      setStatusMessage(job.error_text ?? "差异摘要生成失败。");
      return;
    }

    const nextSummary = job.result_payload as DocumentDiffSummaryRead | null;
    if (nextSummary) {
      const operationSummaryKey = `${operation.documentId}:${operation.fromVersionId}:${operation.toVersionId}`;
      summaryCacheRef.current[operationSummaryKey] = nextSummary;
      setSummary(nextSummary);
      setStatusMessage(
        nextSummary.cache_hit
          ? `${prefix}：已恢复缓存摘要。`
          : nextSummary.summary_provider === "deterministic_fallback"
            ? `${prefix}：大模型摘要不可用，已回退为规则摘要。`
            : `${prefix}：差异摘要已恢复。`,
      );
    } else {
      setStatusMessage(`${prefix}：差异摘要已完成。`);
    }
    removePendingDiffSummaryOperation(operation.id);
  }

  async function recoverPendingSummaryJobs() {
    if (!token) {
      return;
    }
    const operations = matchingPendingSummaryOperations();
    if (!operations.length) {
      return;
    }

    for (const operation of operations) {
      try {
        if (operation.jobId) {
          await acceptPendingSummaryJob(operation, operation.jobId, "已恢复差异摘要");
          continue;
        }
        const job = await api.summarizeDocumentDiffAsync(
          token,
          operation.documentId,
          operation.fromVersionId,
          operation.toVersionId,
          operation.forceRefresh,
          operation.id,
        );
        await acceptPendingSummaryJob(operation, job.id, "已恢复差异摘要请求");
      } catch (nextError) {
        touchPendingDiffSummaryOperation(operation.id, nextError instanceof Error ? nextError.message : "恢复差异摘要失败。");
        if (nextError instanceof ApiError && nextError.status === 0) {
          setStatusMessage("差异摘要仍在后台生成，刷新后会继续恢复。");
        } else {
          removePendingDiffSummaryOperation(operation.id);
          setStatusMessage(nextError instanceof Error ? nextError.message : "恢复差异摘要失败。");
        }
      }
    }
  }

  const pendingSummaryCount = matchingPendingSummaryOperations().length;

  useEffect(() => {
    if (!token) {
      return;
    }
    api
      .listDocuments(token)
      .then((items) => {
        setDocuments(items);
        const firstDocumentId = items[0]?.id ?? "";
        setDocumentId((current) =>
          requestedDocumentId && items.some((item) => item.id === requestedDocumentId)
            ? requestedDocumentId
            : current && items.some((item) => item.id === current)
              ? current
              : firstDocumentId,
        );
      })
      .catch((nextError) => setError(nextError instanceof Error ? nextError.message : "加载文档列表失败。"));
  }, [requestedDocumentId, token]);

  useEffect(() => {
    if (!token || !documentId) {
      setVersions([]);
      return;
    }
    api
      .listDocumentVersions(token, documentId)
      .then((items) => {
        setVersions(items);
        const hasVersion = (versionId: string) => items.some((version) => version.id === versionId);
        const nextToVersionId =
          requestedToVersionId && items.some((version) => version.id === requestedToVersionId)
            ? requestedToVersionId
            : toVersionId && hasVersion(toVersionId)
              ? toVersionId
            : items[0]?.id ?? "";
        const nextFromVersionId =
          requestedFromVersionId && items.some((version) => version.id === requestedFromVersionId)
            ? requestedFromVersionId
            : fromVersionId && fromVersionId !== nextToVersionId && hasVersion(fromVersionId)
              ? fromVersionId
            : items.find((version) => version.id !== nextToVersionId)?.id ?? items[0]?.id ?? "";
        setFromVersionId(nextFromVersionId);
        setToVersionId(nextToVersionId);
      })
      .catch((nextError) => setError(nextError instanceof Error ? nextError.message : "加载版本列表失败。"));
  }, [documentId, requestedFromVersionId, requestedToVersionId, token]);

  useEffect(() => {
    if (didRestoreCachedSelection.current) {
      didRestoreCachedSelection.current = false;
      return;
    }
    setDiff(null);
    setIsSummaryMenuOpen(false);
    hasAutoLoadContext.current = false;
    if (summaryKey && summaryCacheRef.current[summaryKey]) {
      setSummary(summaryCacheRef.current[summaryKey]);
      setStatusMessage("已恢复最近一次差异摘要。");
      return;
    }
    setSummary(null);
    setStatusMessage(null);
  }, [documentId, fromVersionId, summaryKey, toVersionId]);

  useEffect(() => {
    if (!documentId) {
      return;
    }
    syncLocation(documentId, fromVersionId || undefined, toVersionId || undefined);
  }, [documentId, fromVersionId, toVersionId]);

  useEffect(() => {
    if (!token || !documentId || !fromVersionId || !toVersionId) {
      return;
    }
    void recoverPendingSummaryJobs();
  }, [documentId, fromVersionId, toVersionId, token]);

  useEffect(() => {
    if (!token || pendingSummaryCount === 0) {
      return;
    }
    const intervalId = window.setInterval(() => {
      void recoverPendingSummaryJobs();
    }, 5000);
    return () => window.clearInterval(intervalId);
  }, [pendingSummaryCount, token, documentId, fromVersionId, toVersionId]);

  useEffect(() => {
    if (!isSummaryMenuOpen) {
      return;
    }
    function handleOutsideClick(event: MouseEvent) {
      if (summaryMenuRef.current && !summaryMenuRef.current.contains(event.target as Node)) {
        setIsSummaryMenuOpen(false);
      }
    }
    window.addEventListener("mousedown", handleOutsideClick);
    return () => window.removeEventListener("mousedown", handleOutsideClick);
  }, [isSummaryMenuOpen]);

  useEffect(() => {
    if (!token || !documentId || !fromVersionId || !toVersionId) {
      return;
    }
    if (!(requestedDocumentId || requestedFromVersionId || requestedToVersionId)) {
      return;
    }
    if (hasAutoLoadContext.current) {
      return;
    }
    hasAutoLoadContext.current = true;
    void handleLoadDiff();
  }, [documentId, fromVersionId, requestedDocumentId, requestedFromVersionId, requestedToVersionId, toVersionId, token]);

  async function handleLoadDiff() {
    if (!token || !documentId || !fromVersionId || !toVersionId) {
      setStatusMessage("请先选择文档和两个版本。");
      return;
    }
    setIsLoadingDiff(true);
    setStatusMessage("正在加载原始差异...");
    try {
      const nextDiff = await api.getDocumentDiff(token, documentId, fromVersionId, toVersionId);
      setDiff(nextDiff);
      setIsRawDiffCollapsed(false);
      setStatusMessage("已加载原始差异。");
    } catch (nextError) {
      setStatusMessage(nextError instanceof Error ? nextError.message : "生成差异失败。");
    } finally {
      setIsLoadingDiff(false);
    }
  }

  async function handleLoadSummary(forceRefresh = false) {
    if (!token || !documentId || !fromVersionId || !toVersionId) {
      setStatusMessage("请先选择文档和两个版本。");
      return;
    }
    const existingPendingOperation = matchingPendingSummaryOperations()[0] ?? null;
    if (existingPendingOperation) {
      setStatusMessage("差异摘要仍在后台生成，刷新后会继续恢复。");
      return;
    }
    if (!forceRefresh && summaryKey && summaryCacheRef.current[summaryKey]) {
      setSummary(summaryCacheRef.current[summaryKey]);
      setStatusMessage("已恢复最近一次差异摘要。");
      return;
    }
    if (forceRefresh && summaryKey) {
      delete summaryCacheRef.current[summaryKey];
    }
    setIsSummaryMenuOpen(false);
    setIsLoadingSummary(true);
    setStatusMessage(forceRefresh ? "正在提交强制重算任务..." : "正在提交差异摘要任务...");
    const operation = createPendingDiffSummaryOperation({
      documentId,
      fromVersionId,
      toVersionId,
      forceRefresh,
    });
    try {
      const hasMatchingDiff =
        diff && diff.document_id === documentId && diff.from_version_id === fromVersionId && diff.to_version_id === toVersionId;
      if (!hasMatchingDiff) {
        const nextDiff = await api.getDocumentDiff(token, documentId, fromVersionId, toVersionId);
        setDiff(nextDiff);
        setIsRawDiffCollapsed(false);
      }
      const job = await api.summarizeDocumentDiffAsync(
        token,
        documentId,
        fromVersionId,
        toVersionId,
        forceRefresh,
        operation.id,
      );
      await acceptPendingSummaryJob(operation, job.id, forceRefresh ? "已提交强制重算任务" : "已提交差异摘要任务");
    } catch (nextError) {
      const keepPendingOperation = nextError instanceof ApiError && nextError.status === 0;
      if (keepPendingOperation) {
        touchPendingDiffSummaryOperation(operation.id, nextError instanceof Error ? nextError.message : undefined);
        setStatusMessage("差异摘要仍在后台生成，刷新后会自动恢复。");
      } else {
        removePendingDiffSummaryOperation(operation.id);
        setStatusMessage(nextError instanceof Error ? nextError.message : "生成差异摘要失败。");
      }
    } finally {
      setIsLoadingSummary(false);
    }
  }

  return (
    <div className="page-stack">
      <PageHeader
        title="版本对比"
        description="查看两个版本之间的原始差异，并生成可解释的中文摘要。"
      />
      <ErrorNotice message={error} />
      {statusMessage ? <div className="info-block">{statusMessage}</div> : null}

      <section className="panel stack versions-controls-panel">
        <div className="panel-header">
          <div className="panel-heading">
            <h3>对比条件</h3>
            <p>先选择文档和两个版本，再加载原始差异或生成结构化摘要。</p>
          </div>
          {(fromVersionId && toVersionId) ? <StatusBadge tone="info">v 对比模式</StatusBadge> : null}
        </div>
        <div className="page-grid version-selector-grid">
          <label>
            <span>文档</span>
            <SelectField
              options={documents.map((document) => ({ value: document.id, label: document.title }))}
              value={documentId}
              onChange={setDocumentId}
            />
          </label>
          <label>
            <span>起始版本</span>
            <SelectField
              options={versions.map((version) => ({
                value: version.id,
                label: `v${version.version_number} · ${formatIngestStatus(version.ingest_status)}`,
              }))}
              value={fromVersionId}
              onChange={setFromVersionId}
            />
          </label>
          <label>
            <span>目标版本</span>
            <SelectField
              options={versions.map((version) => ({
                value: version.id,
                label: `v${version.version_number} · ${formatIngestStatus(version.ingest_status)}`,
              }))}
              value={toVersionId}
              onChange={setToVersionId}
            />
          </label>
        </div>
        <div className="inline-actions">
          <button className="secondary-button" disabled={isLoadingDiff || isLoadingSummary} onClick={handleLoadDiff} type="button">
            {isLoadingDiff ? "加载中..." : "加载原始差异"}
          </button>
          <div className="summary-action-group" ref={summaryMenuRef}>
            <button className="primary-button summary-action-main" disabled={isLoadingDiff || isLoadingSummary} onClick={() => void handleLoadSummary()} type="button">
              {isLoadingSummary ? "生成中..." : "生成差异摘要"}
            </button>
            <button
              aria-expanded={isSummaryMenuOpen}
              aria-haspopup="menu"
              className="summary-action-toggle"
              disabled={isLoadingDiff || isLoadingSummary}
              onClick={() => setIsSummaryMenuOpen((current) => !current)}
              type="button"
            >
              ▾
            </button>
            {isSummaryMenuOpen ? (
              <div className="summary-action-menu" role="menu">
                <button className="summary-action-menu-item" onClick={() => void handleLoadSummary(true)} role="menuitem" type="button">
                  强制重新生成
                </button>
              </div>
            ) : null}
          </div>
        </div>
      </section>

      <div className="page-grid versions-layout">
        <section className="panel stack versions-diff-panel">
          <div className="panel-header">
            <div className="panel-heading">
              <h3>原始差异</h3>
              <p>展示段落级 diff 和代表性变更，适合快速定位修改位置与内容。</p>
            </div>
            <div className="inline-actions">
              {diff ? (
                <StatusBadge tone="warning">
                  +{diff.added_count} / -{diff.deleted_count} / ~{diff.modified_count}
                </StatusBadge>
              ) : null}
              {diff ? (
                <button className="secondary-button" onClick={() => setIsRawDiffCollapsed((current) => !current)} type="button">
                  {isRawDiffCollapsed ? "展开原始差异" : "收起原始差异"}
                </button>
              ) : null}
            </div>
          </div>
          {diff ? (
            <>
              <div className="metadata-grid">
                <span>起始版本 v{diff.from_version_number}</span>
                <span>目标版本 v{diff.to_version_number}</span>
              </div>
              {!isRawDiffCollapsed ? <pre className="diff-block">{diff.unified_diff}</pre> : null}
              <div className="stack dense-stack">
                {diff.changes.slice(0, 8).map((change, index) => (
                  <div className="list-card" key={`${change.change_type}-${index}`}>
                    <div className="list-card-topline">
                      <strong>{formatDiffChangeType(change.change_type)}</strong>
                      <span>
                        {change.from_paragraph_start ?? "-"} → {change.to_paragraph_start ?? "-"}
                      </span>
                    </div>
                    <p>{truncate(change.new_text ?? change.old_text ?? "暂无文本", 240)}</p>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <p className="muted">请选择两个版本以查看段落级差异。</p>
          )}
        </section>

        <section className="panel stack versions-summary-panel">
          <div className="panel-header">
            <div className="panel-heading">
              <h3>差异摘要</h3>
              <p>基于差异结果总结新增、删除、修改和潜在影响，适合先看重点再回查原文。</p>
            </div>
            {summary ? <StatusBadge tone="success">{formatSummaryProvider(summary.summary_provider)}</StatusBadge> : null}
          </div>
          {isLoadingSummary ? (
            <p className="muted">正在生成差异摘要，请稍候...</p>
          ) : summary ? (
            <>
              <p>{summary.summary}</p>
              <div className="subsection-header">
                <h4>新增内容</h4>
              </div>
              <ul className="plain-list">
                {summary.additions.length ? (
                  summary.additions.map((item) => <li key={item}>{item}</li>)
                ) : (
                  <li>暂无新增重点。</li>
                )}
              </ul>
              <div className="subsection-header">
                <h4>删除内容</h4>
              </div>
              <ul className="plain-list">
                {summary.deletions.length ? (
                  summary.deletions.map((item) => <li key={item}>{item}</li>)
                ) : (
                  <li>暂无删除重点。</li>
                )}
              </ul>
              <div className="subsection-header">
                <h4>修改内容</h4>
              </div>
              <ul className="plain-list">
                {summary.modifications.length ? (
                  summary.modifications.map((item) => <li key={item}>{item}</li>)
                ) : (
                  <li>暂无修改重点。</li>
                )}
              </ul>
              <div className="subsection-header">
                <h4>潜在影响</h4>
              </div>
              <ul className="plain-list">
                {summary.impact_hints.length ? (
                  summary.impact_hints.map((item) => <li key={item}>{item}</li>)
                ) : (
                  <li>暂无明显影响提示。</li>
                )}
              </ul>
            </>
          ) : diff ? (
            <p className="muted">已加载原始差异，可以使用上方“生成差异摘要”按钮继续生成结果。</p>
          ) : (
            <p className="muted">生成差异摘要后，将在这里展示新增、删除、修改和潜在影响提示。</p>
          )}
        </section>
      </div>
    </div>
  );
}

