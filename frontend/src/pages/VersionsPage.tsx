import { useEffect, useState } from "react";

import { ErrorNotice } from "../components/ErrorNotice";
import { PageHeader } from "../components/PageHeader";
import { StatusBadge } from "../components/StatusBadge";
import { useAppContext } from "../context/AppContext";
import { api } from "../lib/api";
import { truncate } from "../lib/format";
import type { DocumentDiffRead, DocumentDiffSummaryRead, DocumentRead, DocumentVersionRead } from "../types/api";

export function VersionsPage() {
  const { token } = useAppContext();
  const [documents, setDocuments] = useState<DocumentRead[]>([]);
  const [documentId, setDocumentId] = useState<string>("");
  const [versions, setVersions] = useState<DocumentVersionRead[]>([]);
  const [fromVersionId, setFromVersionId] = useState<string>("");
  const [toVersionId, setToVersionId] = useState<string>("");
  const [diff, setDiff] = useState<DocumentDiffRead | null>(null);
  const [summary, setSummary] = useState<DocumentDiffSummaryRead | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!token) {
      return;
    }
    api
      .listDocuments(token)
      .then((items) => {
        setDocuments(items);
        const firstDocumentId = items[0]?.id ?? "";
        setDocumentId(firstDocumentId);
      })
      .catch((nextError) => setError(nextError instanceof Error ? nextError.message : "Failed to load documents."));
  }, [token]);

  useEffect(() => {
    if (!token || !documentId) {
      setVersions([]);
      return;
    }
    api
      .listDocumentVersions(token, documentId)
      .then((items) => {
        setVersions(items);
        setFromVersionId(items[1]?.id ?? items[0]?.id ?? "");
        setToVersionId(items[0]?.id ?? "");
      })
      .catch((nextError) => setError(nextError instanceof Error ? nextError.message : "Failed to load versions."));
  }, [documentId, token]);

  async function handleLoadDiff() {
    if (!token || !documentId || !fromVersionId || !toVersionId) {
      setStatusMessage("Select a document and two versions first.");
      return;
    }
    try {
      const nextDiff = await api.getDocumentDiff(token, documentId, fromVersionId, toVersionId);
      setDiff(nextDiff);
      setStatusMessage("Loaded raw diff.");
    } catch (nextError) {
      setStatusMessage(nextError instanceof Error ? nextError.message : "Failed to generate diff.");
    }
  }

  async function handleLoadSummary() {
    if (!token || !documentId || !fromVersionId || !toVersionId) {
      setStatusMessage("Select a document and two versions first.");
      return;
    }
    try {
      const nextSummary = await api.summarizeDocumentDiff(token, documentId, fromVersionId, toVersionId);
      setSummary(nextSummary);
      setStatusMessage("Generated summary and impact hints.");
    } catch (nextError) {
      setStatusMessage(nextError instanceof Error ? nextError.message : "Failed to summarize diff.");
    }
  }

  return (
    <div className="page-stack">
      <PageHeader
        title="Version Compare"
        description="Inspect raw text diff between versions, then summarize only from the diff output so change review stays grounded and explainable."
      />
      <ErrorNotice message={error} />
      {statusMessage ? <div className="info-block">{statusMessage}</div> : null}

      <section className="panel stack">
        <div className="page-grid version-selector-grid">
          <label>
            <span>Document</span>
            <select value={documentId} onChange={(event) => setDocumentId(event.target.value)}>
              {documents.map((document) => (
                <option key={document.id} value={document.id}>
                  {document.title}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>From version</span>
            <select value={fromVersionId} onChange={(event) => setFromVersionId(event.target.value)}>
              {versions.map((version) => (
                <option key={version.id} value={version.id}>
                  v{version.version_number} · {version.ingest_status}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>To version</span>
            <select value={toVersionId} onChange={(event) => setToVersionId(event.target.value)}>
              {versions.map((version) => (
                <option key={version.id} value={version.id}>
                  v{version.version_number} · {version.ingest_status}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="inline-actions">
          <button className="secondary-button" onClick={handleLoadDiff} type="button">
            Load raw diff
          </button>
          <button className="primary-button" onClick={handleLoadSummary} type="button">
            Summarize diff
          </button>
        </div>
      </section>

      <div className="page-grid versions-layout">
        <section className="panel stack">
          <div className="panel-header">
            <h3>Raw diff</h3>
            {diff ? (
              <StatusBadge tone="warning">
                +{diff.added_count} / -{diff.deleted_count} / ~{diff.modified_count}
              </StatusBadge>
            ) : null}
          </div>
          {diff ? (
            <>
              <div className="metadata-grid">
                <span>from v{diff.from_version_number}</span>
                <span>to v{diff.to_version_number}</span>
              </div>
              <pre className="diff-block">{diff.unified_diff}</pre>
              <div className="stack dense-stack">
                {diff.changes.slice(0, 8).map((change, index) => (
                  <div className="list-card" key={`${change.change_type}-${index}`}>
                    <div className="list-card-topline">
                      <strong>{change.change_type}</strong>
                      <span>
                        {change.from_paragraph_start ?? "-"} → {change.to_paragraph_start ?? "-"}
                      </span>
                    </div>
                    <p>{truncate(change.new_text ?? change.old_text ?? "No text", 240)}</p>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <p className="muted">Pick two versions to view paragraph-level diff.</p>
          )}
        </section>

        <section className="panel stack">
          <div className="panel-header">
            <h3>Grounded summary</h3>
            {summary ? <StatusBadge tone="success">{summary.summary_provider}</StatusBadge> : null}
          </div>
          {summary ? (
            <>
              <p>{summary.summary}</p>
              <div className="subsection-header">
                <h4>Additions</h4>
              </div>
              <ul className="plain-list">
                {summary.additions.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
              <div className="subsection-header">
                <h4>Deletions</h4>
              </div>
              <ul className="plain-list">
                {summary.deletions.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
              <div className="subsection-header">
                <h4>Modifications</h4>
              </div>
              <ul className="plain-list">
                {summary.modifications.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
              <div className="subsection-header">
                <h4>Impact hints</h4>
              </div>
              <ul className="plain-list">
                {summary.impact_hints.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </>
          ) : (
            <p className="muted">Run diff summary to highlight additions, deletions, modifications, and impact hints.</p>
          )}
        </section>
      </div>
    </div>
  );
}
