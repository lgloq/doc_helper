import { FormEvent, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { ErrorNotice } from "../components/ErrorNotice";
import { PageHeader } from "../components/PageHeader";
import { StatusBadge } from "../components/StatusBadge";
import { useAppContext } from "../context/AppContext";
import {
  formatBooleanFlag,
  formatDocumentStatus,
  formatIngestStatus,
  formatPrincipalType,
  formatRoleName,
} from "../lib/display";
import { api } from "../lib/api";
import { formatBytes, formatDateTime, truncate } from "../lib/format";
import { canManageDocumentLibrary } from "../lib/permissions";
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
  const { token, user } = useAppContext();
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
  const canManageLibrary = canManageDocumentLibrary(user);

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
      .catch((nextError) => setError(nextError instanceof Error ? nextError.message : "加载文档列表失败。"))
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
          setError(nextError instanceof Error ? nextError.message : "加载文档详情失败。");
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
      setUploadError("请选择要上传的文件。");
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
      setActionMessage(`已完成上传并入库：${response.document.title}`);
    } catch (nextError) {
      setUploadError(nextError instanceof Error ? nextError.message : "上传失败，请稍后重试。");
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
      setActionMessage("请选择新的版本文件。");
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
      setActionMessage(`已上传新版本 v${response.version.version_number}，并完成重新入库。`);
    } catch (nextError) {
      setActionMessage(nextError instanceof Error ? nextError.message : "上传新版本失败。");
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
      setActionMessage("文档访问控制已更新。");
    } catch (nextError) {
      setActionMessage(nextError instanceof Error ? nextError.message : "更新权限失败。");
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
      setActionMessage(`入库完成，共生成 ${ingestion.chunk_count} 个分块。`);
    } catch (nextError) {
      setActionMessage(nextError instanceof Error ? nextError.message : "重新入库失败。");
    }
  }

  return (
    <div className="page-stack">
      <PageHeader title="文档管理" description="上传知识文档、查看版本与分块，并在进入检索前维护文档级访问控制。" />
      <ErrorNotice message={error} />
      <div className="page-grid documents-layout">
        <section className="panel stack">
          <div className="panel-header">
            <h3>{canManageLibrary ? "上传文档" : "当前角色"}</h3>
            <StatusBadge tone={canManageLibrary ? "info" : "neutral"}>{canManageLibrary ? "后端链路已接通" : "只读访问"}</StatusBadge>
          </div>
          {canManageLibrary ? (
            <>
              <form className="stack" onSubmit={handleUpload}>
                <label>
                  <span>文档标题</span>
                  <input name="title" placeholder="可选，留空则使用文件名" />
                </label>
                <label>
                  <span>文档说明</span>
                  <textarea name="description" placeholder="简要说明文档内容与用途" rows={3} />
                </label>
                <label>
                  <span>状态</span>
                  <select name="status" defaultValue="active">
                    <option value="draft">草稿</option>
                    <option value="active">启用中</option>
                    <option value="archived">已归档</option>
                  </select>
                </label>
                <label>
                  <span>文件</span>
                  <input accept=".txt,.md,.markdown,.html,.htm,.pdf,.docx" name="file" type="file" required />
                </label>
                <ErrorNotice message={uploadError} />
                <button className="primary-button" disabled={uploading} type="submit">
                  {uploading ? "上传中..." : "上传并入库"}
                </button>
              </form>
              {latestIngestion ? (
                <div className="info-block">
                  <strong>最近一次入库</strong>
                  <p>
                    状态：{formatIngestStatus(latestIngestion.ingest_status)}，分块数：{latestIngestion.chunk_count}，页数：
                    {latestIngestion.page_count ?? "-"}
                  </p>
                </div>
              ) : null}
            </>
          ) : (
            <div className="info-block">
              <strong>当前账号为只读角色</strong>
              <p>你可以查看可访问文档、版本差异、分块预览并发起问答；文档上传、版本维护、权限修改和重新入库仅管理员可用。</p>
            </div>
          )}
        </section>

        <section className="panel stack">
          <div className="panel-header">
            <h3>可见文档</h3>
            <StatusBadge tone="neutral">{documents.length}</StatusBadge>
          </div>
          {loading ? <p className="muted">正在加载文档列表...</p> : null}
          <div className="document-list">
            {documents.length ? (
              documents.map((document) => (
                <button
                  key={document.id}
                  className={`list-card ${selectedDocumentId === document.id ? "is-selected" : ""}`}
                  onClick={() => setSelectedDocumentId(document.id)}
                  type="button"
                >
                  <div className="list-card-topline">
                    <strong>{document.title}</strong>
                    <StatusBadge tone={canManageLibrary && document.current_user_can_manage ? "warning" : "neutral"}>
                      {formatDocumentStatus(document.status)}
                    </StatusBadge>
                  </div>
                  <p>{truncate(document.description ?? "暂无描述。", 120)}</p>
                </button>
              ))
            ) : !loading ? (
              <p className="muted">{canManageLibrary ? "暂无可见文档，可先上传一个文档开始体验。" : "暂无可见文档。"}</p>
            ) : null}
          </div>
        </section>
      </div>

      {selectedDocument ? (
        <div className="page-grid detail-layout">
          <section className="panel stack">
            <div className="panel-header">
              <div>
                <h3>{selectedDocument.title}</h3>
                <p className="muted">所有者 ID：{selectedDocument.owner_user_id}</p>
              </div>
              <Link className="secondary-button link-button" to="/versions">
                查看差异页
              </Link>
            </div>
            <p>{selectedDocument.description ?? "暂无文档说明。"}</p>
            {actionMessage ? <div className="info-block">{actionMessage.trim()}</div> : null}

            <div className="subsection-header">
              <h4>版本列表</h4>
            </div>
            <div className="stack dense-stack">
              {versions.length ? (
                versions.map((version) => (
                  <div className="version-card" key={version.id}>
                    <div className="list-card-topline">
                      <strong>v{version.version_number}</strong>
                      <StatusBadge tone={version.is_current ? "success" : "neutral"}>
                        {formatIngestStatus(version.ingest_status)}
                      </StatusBadge>
                    </div>
                    <p>
                      {version.original_filename} · {formatBytes(version.file_size)} · {formatDateTime(version.created_at)}
                    </p>
                    <p className="muted">{version.storage_path}</p>
                    {canManageLibrary && selectedDocument.current_user_can_manage ? (
                      <div className="inline-actions">
                        <button className="secondary-button" onClick={() => handleIngest(version.id)} type="button">
                          重新入库
                        </button>
                      </div>
                    ) : null}
                  </div>
                ))
              ) : (
                <p className="muted">当前文档还没有可用版本。</p>
              )}
            </div>

            {canManageLibrary && selectedDocument.current_user_can_manage ? (
              <form className="stack" onSubmit={handleVersionUpload}>
                <div className="subsection-header">
                  <h4>上传新版本</h4>
                </div>
                <input accept=".txt,.md,.markdown,.html,.htm,.pdf,.docx" name="file" type="file" required />
                <button className="primary-button" disabled={versionUploading} type="submit">
                  {versionUploading ? "上传中..." : "上传版本"}
                </button>
              </form>
            ) : null}
          </section>

          <section className="panel stack">
            <div className="panel-header">
              <h3>权限信息</h3>
              <StatusBadge tone="info">文档访问控制</StatusBadge>
            </div>
            {canManageLibrary ? (
              <>
                <div className="stack dense-stack">
                  {aclEntries.length ? (
                    aclEntries.map((entry) => (
                      <div className="list-card" key={entry.id}>
                        <div className="list-card-topline">
                          <strong>{formatPrincipalType(entry.principal_type)}</strong>
                          <span>
                            {entry.user_email ?? (entry.role_name ? formatRoleName(entry.role_name) : entry.team_name ?? "全部用户")}
                          </span>
                        </div>
                        <p>
                          可查看：{formatBooleanFlag(entry.can_view)} · 可管理：{formatBooleanFlag(entry.can_manage)}
                        </p>
                      </div>
                    ))
                  ) : (
                    <p className="muted">暂未配置显式 ACL。文档所有者和管理员仍然保留访问权限。</p>
                  )}
                </div>

                {selectedDocument.current_user_can_manage ? (
                  <form className="stack" onSubmit={handleAclSubmit}>
                    <label>
                      <span>授权主体</span>
                      <select
                        value={aclForm.principal_type}
                        onChange={(event) =>
                          setAclForm((current) => ({ ...current, principal_type: event.target.value as PrincipalType }))
                        }
                      >
                        <option value="public">公开</option>
                        <option value="team">团队</option>
                        <option value="role">角色</option>
                        <option value="user">指定用户</option>
                      </select>
                    </label>
                    {aclForm.principal_type === "team" ? (
                      <label>
                        <span>团队名称</span>
                        <input
                          value={aclForm.team_name}
                          onChange={(event) => setAclForm((current) => ({ ...current, team_name: event.target.value }))}
                        />
                      </label>
                    ) : null}
                    {aclForm.principal_type === "role" ? (
                      <label>
                        <span>角色</span>
                        <select
                          value={aclForm.role_name}
                          onChange={(event) =>
                            setAclForm((current) => ({ ...current, role_name: event.target.value as RoleName }))
                          }
                        >
                          <option value="viewer">普通员工</option>
                          <option value="manager">组长</option>
                          <option value="admin">管理员</option>
                        </select>
                      </label>
                    ) : null}
                    {aclForm.principal_type === "user" ? (
                      <label>
                        <span>用户 ID</span>
                        <input
                          placeholder="粘贴用户 UUID"
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
                      <span>可查看</span>
                    </label>
                    <label className="inline-checkbox">
                      <input
                        checked={aclForm.can_manage}
                        onChange={(event) => setAclForm((current) => ({ ...current, can_manage: event.target.checked }))}
                        type="checkbox"
                      />
                      <span>可管理</span>
                    </label>
                    <button className="primary-button" type="submit">
                      保存权限
                    </button>
                  </form>
                ) : null}
              </>
            ) : (
              <div className="info-block">
                <strong>当前账号不提供权限维护入口</strong>
                <p>文档访问控制仅管理员可查看和维护。如需新增授权或调整访问范围，请联系管理员处理。</p>
              </div>
            )}
          </section>

          <section className="panel stack">
            <div className="panel-header">
              <h3>分块预览</h3>
              <StatusBadge tone="success">{chunks.length} 个分块</StatusBadge>
            </div>
            <div className="chunk-list">
              {chunks.length ? (
                chunks.slice(0, 8).map((chunk) => (
                  <div className="chunk-card" key={chunk.id}>
                    <div className="list-card-topline">
                      <strong>{chunk.section_title ?? `分块 ${chunk.chunk_index}`}</strong>
                      <span>
                        {chunk.page_number_start != null ? `第 ${chunk.page_number_start} 页` : "页码未知"} · {chunk.paragraph_start != null ? `第 ${chunk.paragraph_start} 段` : "段落未知"}
                      </span>
                    </div>
                    <p>{truncate(chunk.preview, 220)}</p>
                  </div>
                ))
              ) : (
                <p className="muted">
                  {canManageLibrary ? "当前还没有分块数据，请先对版本执行入库处理。" : "当前还没有可显示的分块数据，如需补充入库请联系管理员。"}
                </p>
              )}
            </div>
          </section>
        </div>
      ) : null}
    </div>
  );
}
