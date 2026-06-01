import { useCallback, useEffect, useMemo, useState } from "react";

import { api } from "../lib/api";
import { formatRoleName } from "../lib/display";
import type { DocumentAccessDebugRead, UserRead } from "../types/api";

const USER_RESULT_LIMIT = 40;

interface AccessDebugPanelProps {
  token: string;
  documentId: string;
  documentTitle: string;
  initialUsers: UserRead[];
  onClose: () => void;
}

export function AccessDebugPanel({ token, documentId, documentTitle, initialUsers, onClose }: AccessDebugPanelProps) {
  const [users, setUsers] = useState<UserRead[]>(initialUsers);
  const [selectedUserId, setSelectedUserId] = useState<string>("");
  const [userQuery, setUserQuery] = useState("");
  const [userLoading, setUserLoading] = useState(false);
  const [result, setResult] = useState<DocumentAccessDebugRead | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const selectedUser = useMemo(
    () => users.find((user) => user.id === selectedUserId) ?? null,
    [selectedUserId, users],
  );

  const handleDebug = useCallback(async () => {
    if (!selectedUserId) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await api.debugDocumentAccess(token, documentId, selectedUserId);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "诊断失败");
    } finally {
      setLoading(false);
    }
  }, [token, documentId, selectedUserId]);

  useEffect(() => {
    setResult(null);
    setError(null);
  }, [selectedUserId]);

  useEffect(() => {
    if (selectedUserId && !selectedUser) {
      setSelectedUserId("");
    }
  }, [selectedUser, selectedUserId]);

  useEffect(() => {
    setUsers(initialUsers);
  }, [initialUsers]);

  useEffect(() => {
    let isMounted = true;
    const timer = window.setTimeout(() => {
      setUserLoading(true);
      api
        .listUsers(token, { q: userQuery, limit: USER_RESULT_LIMIT })
        .then((items) => {
          if (isMounted) {
            setUsers(items);
          }
        })
        .catch((err) => {
          if (isMounted) {
            setError(err instanceof Error ? err.message : "加载用户失败");
          }
        })
        .finally(() => {
          if (isMounted) {
            setUserLoading(false);
          }
        });
    }, 220);
    return () => {
      isMounted = false;
      window.clearTimeout(timer);
    };
  }, [token, userQuery]);

  return (
    <div className="access-debug-overlay">
      <div className="access-debug-panel panel stack">
        <div className="panel-header">
          <div className="panel-heading">
            <h3>权限诊断</h3>
            <p>{documentTitle}</p>
          </div>
          <button className="secondary-button compact-button" onClick={onClose} type="button">
            关闭
          </button>
        </div>

        <div className="access-debug-shell">
          <aside className="access-debug-user-panel">
            <label className="access-debug-search-label">
              <span>诊断用户</span>
              <input
                autoFocus
                className="access-debug-search-input"
                onChange={(event) => setUserQuery(event.target.value)}
                placeholder="搜索姓名、邮箱、部门"
                value={userQuery}
              />
            </label>

            <div className="access-debug-user-count">
              <span>{userLoading ? "搜索中..." : `显示 ${users.length} 个`}</span>
            </div>

            <div aria-label="诊断用户列表" className="access-debug-user-list" role="listbox">
              {users.map((user) => {
                const isSelected = selectedUserId === user.id;
                return (
                  <button
                    aria-selected={isSelected}
                    className={`access-debug-user-option ${isSelected ? "is-selected" : ""}`}
                    key={user.id}
                    onClick={() => setSelectedUserId(user.id)}
                    role="option"
                    type="button"
                  >
                    <span className="access-debug-user-main">
                      <strong>{user.full_name}</strong>
                      <span>{user.role?.name ? formatRoleName(user.role.name) : "未分配角色"}</span>
                    </span>
                    <span className="access-debug-user-email">{user.email}</span>
                    <span className="access-debug-user-path">{user.department?.path ?? "未设置部门"}</span>
                  </button>
                );
              })}
              {!userLoading && users.length === 0 ? (
                <div className="empty-state compact-empty-state">没有匹配的用户</div>
              ) : null}
            </div>

            <div className="access-debug-selected-user">
              {selectedUser ? (
                <>
                  <span>当前诊断对象</span>
                  <strong>{selectedUser.full_name}</strong>
                  <p>{selectedUser.department?.path ?? "未设置部门"}</p>
                </>
              ) : (
                <p>先从左侧选择一个用户。</p>
              )}
            </div>

            <button
              className="primary-button access-debug-run-button"
              disabled={!selectedUserId || loading}
              onClick={handleDebug}
              type="button"
            >
              {loading ? "诊断中..." : "开始诊断"}
            </button>
          </aside>

          <section className="access-debug-detail-panel">
            {error && <div className="error-notice">{error}</div>}

            {!result && !error ? (
              <div className="empty-state access-debug-empty-state">
                <strong>等待诊断</strong>
                <p>选择用户后点击“开始诊断”，这里会显示最终判定、命中规则和逐项检查结果。</p>
              </div>
            ) : null}

            {result && (
              <div className="access-debug-result stack">
                <div className="access-debug-verdict-card">
                  <div>
                    <span>判定结果</span>
                    <strong>{result.can_view ? "允许访问" : "拒绝访问"}</strong>
                  </div>
                  <div className="access-debug-verdict">
                    <span className={`access-debug-badge ${result.can_view ? "is-pass" : "is-fail"}`}>
                      {result.can_view ? "可查看" : "不可查看"}
                    </span>
                    <span className={`access-debug-badge ${result.can_manage ? "is-pass" : "is-fail"}`}>
                      {result.can_manage ? "可管理" : "不可管理"}
                    </span>
                  </div>
                </div>

                <p className="access-debug-reason">{result.reason}</p>

                {result.matched_rule && (
                  <div className="access-debug-rule-card">
                    <div className="list-card-topline">
                      <strong>命中规则</strong>
                      <span className="status-badge status-info">{result.matched_rule.source}</span>
                    </div>
                    {result.matched_rule.department_path && (
                      <p className="muted">部门：{result.matched_rule.department_path}</p>
                    )}
                    {result.matched_rule.match_type && (
                      <p className="muted">
                        匹配方式：
                        {result.matched_rule.match_type === "ancestor"
                          ? "继承命中"
                          : result.matched_rule.match_type === "direct"
                            ? "直接归属"
                            : result.matched_rule.match_type === "legacy"
                              ? "旧版团队"
                              : result.matched_rule.match_type}
                      </p>
                    )}
                    <p className="muted">
                      权限：{result.matched_rule.can_view ? "可查看" : "不可查看"} ·{" "}
                      {result.matched_rule.can_manage ? "可管理" : "不可管理"}
                    </p>
                  </div>
                )}

                <div className="subsection-header">
                  <h4>检查项</h4>
                </div>
                <div className="access-debug-check-list">
                  {result.checks.map((check, index) => (
                    <div
                      key={`${check.source}-${index}`}
                      className={`access-debug-check ${check.matched ? "is-matched" : "is-unmatched"}`}
                    >
                      <span className={`access-debug-check-icon ${check.matched ? "is-pass" : "is-fail"}`}>
                        {check.matched ? "✓" : "✗"}
                      </span>
                      <span>
                        <strong>{check.source}</strong>
                        <span className="muted"> — {check.message}</span>
                      </span>
                    </div>
                  ))}
                </div>

                {result.department_context.ancestor_department_paths.length > 0 && (
                  <div className="access-debug-context-card">
                    <div className="subsection-header">
                      <h4>部门上下文</h4>
                    </div>
                    <p>
                      <strong>用户部门：</strong>
                      {result.department_context.user_department_path ?? "未设置"}
                    </p>
                    <p className="muted">
                      祖先链：{result.department_context.ancestor_department_paths.join(" → ")}
                    </p>
                  </div>
                )}
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
