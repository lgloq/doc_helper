import { FormEvent, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { ErrorNotice } from "../components/ErrorNotice";
import { PageHeader } from "../components/PageHeader";
import { StatusBadge } from "../components/StatusBadge";
import { useAppContext } from "../context/AppContext";
import { api } from "../lib/api";
import { formatBytes, formatDateTime, truncate } from "../lib/format";
import type {
  ChunkRead,
  DocumentACLRead,
  DocumentRead,
  DocumentVersionRead,
  IngestionResultRead,
  PrincipalType,
  RoleName,
} from "../types/api";

interface AclFormState {
  principal_type: PrincipalType;
  team_name: string;
  role_name: RoleName;
  user_id: string;
  can_view: boolean;
  can_manage: boolean;
}

const defaultAclForm: AclFormState = {
  principal_type: "team",
  team_name: "platform",
  role_name: "viewer",
  user_id: "",
  can_view: true,
  can_manage: false,
};

export function DocumentsPage() {
  const { token } = useAppContext();
  const [documents, setDocuments] = useState<DocumentRead[]>([]);
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null);
  const [selectedDocument, setSelectedDocument] = useState<DocumentRead | null>(null);
  const [versions, setVersions] = useState<DocumentVersionRead[]>([]);
  const [aclEntries, setAclEntries] = useState<DocumentACLRead[]>([]);
  const [chunks, setChunks] = useState<ChunkRead[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [versionUploading, setVersionUploading] = useState(false);
  const [aclForm, setAclForm] = useState<AclFormState>(defaultAclForm);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [latestIngestion, setLatestIngestion] = useState<IngestionResultRead | null>(null);

  useEffect(() => {
    if (!token) {
      return;
    }
    setLoading(true);
    setError(null);
    api
      .listDocuments(token)
      .then((items) => {
        setDocuments(items);
        setSelectedDocumentId((current) => current ?? items[0]?.id ?? null);
      })
      .catch((nextError) => setError(nextError instanceof Error ? nextError.message : "Failed to load documents."))
      .finally(() => setLoading(false));
  }, [token]);

  useEffect(() => {
    if (!token || !selectedDocumentId) {
      setSelectedDocument(null);
      setVersions([]);
      setAclEntries([]);
      setChunks([]);
      return;
    }

    let isMounted = true;
    setError(null);
    Promise.all([
      api.getDocument(token, selectedDocumentId),
      api.listDocumentVersions(token, selectedDocumentId),
      api.listDocumentAcl(token, selectedDocumentId).catch(() => []),
      api.listChunks(token, selectedDocumentId).catch(() => []),
    ])
      .then(([document, nextVersions, nextAclEntries, nextChunks]) => {
        if (!isMounted) {
          return;
        }
        setSelectedDocument(document);
        setVersions(nextVersions);
        setAclEntries(nextAclEntries);
        setChunks(nextChunks);
      })
      .catch((nextError) => {
        if (isMounted) {
          setError(nextError instanceof Error ? nextError.message : "Failed to load document details.");
        }
      });

    return () => {
      isMounted = false;
    };
  }, [selectedDocumentId, token]);

  async function refreshSelectedDocument(documentId: string) {
    if (!token) {
      return;
    }
    const [document, nextVersions, nextAclEntries, nextChunks] = await Promise.all([
      api.getDocument(token, documentId),
      api.listDocumentVersions(token, documentId),
      api.listDocumentAcl(token, documentId).catch(() => []),
      api.listChunks(token, documentId).catch(() => []),
    ]);
    setSelectedDocument(document);
    setVersions(nextVersions);
    setAclEntries(nextAclEntries);
    setChunks(nextChunks);
  }

  async function handleUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!token) {
      return;
    }
    const form = new FormData(event.currentTarget);
    const file = form.get("file");
    if (!(file instanceof File) || !file.name) {
      setUploadError("Please choose a file to upload.");
      return;
    }

    setUploading(true);
    setUploadError(null);
    setActionMessage(null);
    try {
      const response = await api.uploadDocument(token, {
        file,
        title: String(form.get("title") || "").trim() || undefined,
        description: String(form.get("description") || "").trim() || undefined,
        status: String(form.get("status") || "active"),
      });
      const ingestion = await api.ingestDocument(token, response.document.id, response.version.id);
      setLatestIngestion(ingestion);
      const nextDocuments = await api.listDocuments(token);
      setDocuments(nextDocuments);
      setSelectedDocumentId(response.document.id);
      await refreshSelectedDocument(response.document.id);
      event.currentTarget.reset();
      setActionMessage(`Uploaded and ingested ${response.document.title}.`);
    } catch (nextError) {
      setUploadError(nextError instanceof Error ? nextError.message : "Upload failed.");
    } finally {
      setUploading(false);
    }
  }

  async function handleVersionUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!token || !selectedDocument) {
      return;
    }
    const form = new FormData(event.currentTarget);
    const file = form.get("file");
    if (!(file instanceof File) || !file.name) {
      setActionMessage("Please choose a new version file.");
      return;
    }

    setVersionUploading(true);
    setActionMessage(null);
    try {
      const response = await api.uploadDocumentVersion(token, selectedDocument.id, file);
      const ingestion = await api.ingestDocument(token, selectedDocument.id, response.version.id);
      setLatestIngestion(ingestion);
      await refreshSelectedDocument(selectedDocument.id);
      event.currentTarget.reset();
      setActionMessage(`Uploaded version v${response.version.version_number} and rebuilt chunks.`);
    } catch (nextError) {
      setActionMessage(nextError instanceof Error ? nextError.message : "Version upload failed.");
    } finally {
      setVersionUploading(false);
    }
  }

  async function handleAclSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!token || !selectedDocument) {
      return;
    }

    try {
      await api.upsertDocumentAcl(token, selectedDocument.id, {
        principal_type: aclForm.principal_type,
        can_view: aclForm.can_view,
        can_manage: aclForm.can_manage,
        team_name: aclForm.principal_type === "team" ? aclForm.team_name : undefined,
        role_name: aclForm.principal_type === "role" ? aclForm.role_name : undefined,
        user_id: aclForm.principal_type === "user" && aclForm.user_id ? aclForm.user_id : undefined,
      });
      setAclForm(defaultAclForm);
      await refreshSelectedDocument(selectedDocument.id);
      setActionMessage("ACL updated.");
    } catch (nextError) {
      setActionMessage(nextError instanceof Error ? nextError.message : "ACL update failed.");
    }
  }

  async function handleIngest(versionId?: string) {
    if (!token || !selectedDocument) {
      return;
    }
    try {
      const ingestion = await api.ingestDocument(token, selectedDocument.id, versionId);
      setLatestIngestion(ingestion);
      await refreshSelectedDocument(selectedDocument.id);
      setActionMessage(`Ingestion finished with ${ingestion.chunk_count} chunks.`);
    } catch (nextError) {
      setActionMessage(nextError instanceof Error ? nextError.message : "Ingestion failed.");
    }
  }

  return (
    <div className="page-stack">
      <PageHeader
        title="Document Management"
        description="Upload knowledge sources, inspect versions, preview chunks, and manage document-level ACL before retrieval."
      />
      <ErrorNotice message={error} />
      <div className="page-grid documents-layout">
        <section className="panel stack">
          <div className="panel-header">
            <h3>Upload document</h3>
            <StatusBadge tone="info">Phase 3-7 backend wired</StatusBadge>
          </div>
          <form className="stack" onSubmit={handleUpload}>
            <label>
              <span>Title</span>
              <input name="title" placeholder="Optional title override" />
            </label>
            <label>
              <span>Description</span>
              <textarea name="description" placeholder="Brief document summary" rows={3} />
            </label>
            <label>
              <span>Status</span>
              <select name="status" defaultValue="active">
                <option value="draft">draft</option>
                <option value="active">active</option>
                <option value="archived">archived</option>
              </select>
            </label>
            <label>
              <span>File</span>
              <input accept=".txt,.md,.markdown,.html,.htm,.pdf,.docx" name="file" type="file" required />
            </label>
            <ErrorNotice message={uploadError} />
            <button className="primary-button" disabled={uploading} type="submit">
              {uploading ? "Uploading..." : "Upload and ingest"}
            </button>
          </form>
          {latestIngestion ? (
            <div className="info-block">
              <strong>Last ingestion</strong>
              <p>
                status: {latestIngestion.ingest_status}, chunks: {latestIngestion.chunk_count}, pages:{" "}
                {latestIngestion.page_count ?? "-"}
              </p>
            </div>
          ) : null}
        </section>

        <section className="panel stack">
          <div className="panel-header">
            <h3>Visible documents</h3>
            <StatusBadge tone="neutral">{documents.length}</StatusBadge>
          </div>
          {loading ? <p className="muted">Loading documents...</p> : null}
          <div className="document-list">
            {documents.map((document) => (
              <button
                key={document.id}
                className={`list-card ${selectedDocumentId === document.id ? "is-selected" : ""}`}
                onClick={() => setSelectedDocumentId(document.id)}
                type="button"
              >
                <div className="list-card-topline">
                  <strong>{document.title}</strong>
                  <StatusBadge tone={document.current_user_can_manage ? "warning" : "neutral"}>
                    {document.status}
                  </StatusBadge>
                </div>
                <p>{truncate(document.description ?? "No description yet.", 120)}</p>
              </button>
            ))}
          </div>
        </section>
      </div>

      {selectedDocument ? (
        <div className="page-grid detail-layout">
          <section className="panel stack">
            <div className="panel-header">
              <div>
                <h3>{selectedDocument.title}</h3>
                <p className="muted">Owner {selectedDocument.owner_user_id}</p>
              </div>
              <Link className="secondary-button link-button" to="/versions">
                Open diff page
              </Link>
            </div>
            <p>{selectedDocument.description ?? "No description provided."}</p>
            {actionMessage ? <div className="info-block">{actionMessage}</div> : null}

            <div className="subsection-header">
              <h4>Versions</h4>
            </div>
            <div className="stack dense-stack">
              {versions.map((version) => (
                <div className="version-card" key={version.id}>
                  <div className="list-card-topline">
                    <strong>v{version.version_number}</strong>
                    <StatusBadge tone={version.is_current ? "success" : "neutral"}>
                      {version.ingest_status}
                    </StatusBadge>
                  </div>
                  <p>
                    {version.original_filename} · {formatBytes(version.file_size)} · {formatDateTime(version.created_at)}
                  </p>
                  <p className="muted">{version.storage_path}</p>
                  <div className="inline-actions">
                    <button className="secondary-button" onClick={() => handleIngest(version.id)} type="button">
                      Re-ingest
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {selectedDocument.current_user_can_manage ? (
              <form className="stack" onSubmit={handleVersionUpload}>
                <div className="subsection-header">
                  <h4>Upload new version</h4>
                </div>
                <input accept=".txt,.md,.markdown,.html,.htm,.pdf,.docx" name="file" type="file" required />
                <button className="primary-button" disabled={versionUploading} type="submit">
                  {versionUploading ? "Uploading..." : "Upload version"}
                </button>
              </form>
            ) : null}
          </section>

          <section className="panel stack">
            <div className="panel-header">
              <h3>Permissions</h3>
              <StatusBadge tone="info">Document ACL</StatusBadge>
            </div>
            <div className="stack dense-stack">
              {aclEntries.length ? (
                aclEntries.map((entry) => (
                  <div className="list-card" key={entry.id}>
                    <div className="list-card-topline">
                      <strong>{entry.principal_type}</strong>
                      <span>
                        {entry.user_email ?? entry.role_name ?? entry.team_name ?? "all users"}
                      </span>
                    </div>
                    <p>
                      view: {String(entry.can_view)} · manage: {String(entry.can_manage)}
                    </p>
                  </div>
                ))
              ) : (
                <p className="muted">No ACL entries yet. Owner/admin still retain access.</p>
              )}
            </div>

            {selectedDocument.current_user_can_manage ? (
              <form className="stack" onSubmit={handleAclSubmit}>
                <label>
                  <span>Principal type</span>
                  <select
                    value={aclForm.principal_type}
                    onChange={(event) =>
                      setAclForm((current) => ({ ...current, principal_type: event.target.value as PrincipalType }))
                    }
                  >
                    <option value="public">public</option>
                    <option value="team">team</option>
                    <option value="role">role</option>
                    <option value="user">user</option>
                  </select>
                </label>
                {aclForm.principal_type === "team" ? (
                  <label>
                    <span>Team name</span>
                    <input
                      value={aclForm.team_name}
                      onChange={(event) => setAclForm((current) => ({ ...current, team_name: event.target.value }))}
                    />
                  </label>
                ) : null}
                {aclForm.principal_type === "role" ? (
                  <label>
                    <span>Role</span>
                    <select
                      value={aclForm.role_name}
                      onChange={(event) =>
                        setAclForm((current) => ({ ...current, role_name: event.target.value as RoleName }))
                      }
                    >
                      <option value="viewer">viewer</option>
                      <option value="manager">manager</option>
                      <option value="admin">admin</option>
                    </select>
                  </label>
                ) : null}
                {aclForm.principal_type === "user" ? (
                  <label>
                    <span>User ID</span>
                    <input
                      placeholder="Paste a user UUID"
                      value={aclForm.user_id}
                      onChange={(event) => setAclForm((current) => ({ ...current, user_id: event.target.value }))}
                    />
                  </label>
                ) : null}
                <label className="inline-checkbox">
                  <input
                    checked={aclForm.can_view}
                    onChange={(event) => setAclForm((current) => ({ ...current, can_view: event.target.checked }))}
                    type="checkbox"
                  />
                  <span>can view</span>
                </label>
                <label className="inline-checkbox">
                  <input
                    checked={aclForm.can_manage}
                    onChange={(event) => setAclForm((current) => ({ ...current, can_manage: event.target.checked }))}
                    type="checkbox"
                  />
                  <span>can manage</span>
                </label>
                <button className="primary-button" type="submit">
                  Save ACL
                </button>
              </form>
            ) : null}
          </section>

          <section className="panel stack">
            <div className="panel-header">
              <h3>Chunk preview</h3>
              <StatusBadge tone="success">{chunks.length} chunks</StatusBadge>
            </div>
            <div className="chunk-list">
              {chunks.length ? (
                chunks.slice(0, 8).map((chunk) => (
                  <div className="chunk-card" key={chunk.id}>
                    <div className="list-card-topline">
                      <strong>{chunk.section_title ?? `Chunk ${chunk.chunk_index}`}</strong>
                      <span>
                        p.{chunk.page_number_start ?? "-"} · para.{chunk.paragraph_start ?? "-"}
                      </span>
                    </div>
                    <p>{truncate(chunk.preview, 220)}</p>
                  </div>
                ))
              ) : (
                <p className="muted">No chunks yet. Ingest a version to populate citation-ready chunk metadata.</p>
              )}
            </div>
          </section>
        </div>
      ) : null}
    </div>
  );
}
