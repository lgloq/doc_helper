import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { DepartmentTreeSelect } from "../components/DepartmentTreeSelect";
import { ErrorNotice } from "../components/ErrorNotice";
import { PageHeader } from "../components/PageHeader";
import { StatusBadge } from "../components/StatusBadge";
import { useAppContext } from "../context/AppContext";
import { api } from "../lib/api";
import { formatRoleName } from "../lib/display";
import type { DepartmentRead, RoleName, UserRead } from "../types/api";

interface AssignmentEditingState {
  userId: string;
  departmentId: string | null;
}

interface UserDraft {
  email: string;
  full_name: string;
  password: string;
  role_name: RoleName;
  department_id: string | null;
  is_active: boolean;
}

type UsersPageTab = "manage" | "assign" | "departments";
type DepartmentFormMode = "detail" | "create";
type UserFormMode = "create" | "edit";
type UserStatusFilter = "all" | "active" | "inactive";

const ROLE_OPTIONS: Array<{ value: RoleName; label: string }> = [
  { value: "viewer", label: "普通员工" },
  { value: "manager", label: "组长" },
  { value: "admin", label: "管理员" },
];

function createEmptyUserDraft(): UserDraft {
  return {
    email: "",
    full_name: "",
    password: "",
    role_name: "viewer",
    department_id: null,
    is_active: true,
  };
}

function createUserDraftFromUser(user: UserRead): UserDraft {
  return {
    email: user.email,
    full_name: user.full_name,
    password: "",
    role_name: user.role?.name ?? "viewer",
    department_id: user.department_id,
    is_active: user.is_active,
  };
}

export function UsersPage() {
  const { token, user, refreshMe } = useAppContext();
  const navigate = useNavigate();

  const [users, setUsers] = useState<UserRead[]>([]);
  const [departments, setDepartments] = useState<DepartmentRead[]>([]);
  const [assignmentEditing, setAssignmentEditing] = useState<AssignmentEditingState | null>(null);
  const [saving, setSaving] = useState(false);
  const [departmentSaving, setDepartmentSaving] = useState(false);
  const [userSaving, setUserSaving] = useState(false);
  const [userDeleting, setUserDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<UsersPageTab>("manage");
  const [userSearch, setUserSearch] = useState("");
  const [userStatusFilter, setUserStatusFilter] = useState<UserStatusFilter>("all");
  const [managedUserId, setManagedUserId] = useState<string | null>(null);
  const [userFormMode, setUserFormMode] = useState<UserFormMode>("create");
  const [userDraft, setUserDraft] = useState<UserDraft>(() => createEmptyUserDraft());
  const [selectedManageDepartmentId, setSelectedManageDepartmentId] = useState<string | null>(null);
  const [departmentFormMode, setDepartmentFormMode] = useState<DepartmentFormMode>("detail");
  const [departmentNameDraft, setDepartmentNameDraft] = useState("");
  const [departmentParentDraft, setDepartmentParentDraft] = useState<string | null>(null);
  const [showCreateParentSelector, setShowCreateParentSelector] = useState(false);

  const isAdmin = user?.role?.name === "admin";
  const selectedAssignmentUser = useMemo(
    () => (assignmentEditing ? users.find((item) => item.id === assignmentEditing.userId) ?? null : null),
    [assignmentEditing, users],
  );
  const selectedAssignmentDepartment = useMemo(
    () =>
      assignmentEditing?.departmentId
        ? departments.find((item) => item.id === assignmentEditing.departmentId) ?? null
        : null,
    [departments, assignmentEditing],
  );
  const hasDepartmentChange = assignmentEditing
    ? assignmentEditing.departmentId !== (selectedAssignmentUser?.department_id ?? null)
    : false;
  const assignableUsers = useMemo(() => users.filter((item) => item.is_active), [users]);
  const filteredManagedUsers = useMemo(() => {
    const query = userSearch.trim().toLowerCase();
    return users.filter((item) => {
      const matchesQuery =
        !query ||
        item.email.toLowerCase().includes(query) ||
        item.full_name.toLowerCase().includes(query) ||
        (item.department?.path ?? "").toLowerCase().includes(query);
      const matchesStatus =
        userStatusFilter === "all" ||
        (userStatusFilter === "active" && item.is_active) ||
        (userStatusFilter === "inactive" && !item.is_active);
      return matchesQuery && matchesStatus;
    });
  }, [userSearch, userStatusFilter, users]);
  const selectedManagedUser = useMemo(
    () => (managedUserId ? users.find((item) => item.id === managedUserId) ?? null : null),
    [managedUserId, users],
  );
  const selectedUserFormDepartment = useMemo(
    () => (userDraft.department_id ? departments.find((item) => item.id === userDraft.department_id) ?? null : null),
    [departments, userDraft.department_id],
  );
  const isManagingCurrentUser = selectedManagedUser?.id === user?.id;
  const selectedManageDepartment = useMemo(
    () =>
      selectedManageDepartmentId
        ? departments.find((department) => department.id === selectedManageDepartmentId) ?? null
        : null,
    [departments, selectedManageDepartmentId],
  );
  const selectedManageParent = useMemo(
    () =>
      selectedManageDepartment?.parent_id
        ? departments.find((department) => department.id === selectedManageDepartment.parent_id) ?? null
        : null,
    [departments, selectedManageDepartment],
  );
  const departmentChildCount = useMemo(
    () =>
      selectedManageDepartment
        ? departments.filter((department) => department.parent_id === selectedManageDepartment.id).length
        : 0,
    [departments, selectedManageDepartment],
  );
  const movableParentDepartments = useMemo(() => {
    if (!selectedManageDepartment) {
      return departments;
    }
    return departments.filter(
      (department) =>
        department.id !== selectedManageDepartment.id &&
        !department.id_path.startsWith(`${selectedManageDepartment.id_path}/`),
    );
  }, [departments, selectedManageDepartment]);

  useEffect(() => {
    if (!isAdmin) {
      navigate("/documents", { replace: true });
    }
  }, [isAdmin, navigate]);

  const loadData = useCallback(async () => {
    if (!token) return null;
    try {
      const [usersData, departmentsData] = await Promise.all([api.listUsers(token), api.listDepartments(token)]);
      setUsers(usersData);
      setDepartments(departmentsData);
      return { usersData, departmentsData };
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载数据失败");
      return null;
    }
  }, [token]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const clearFeedback = () => {
    setError(null);
    setStatusMessage(null);
  };

  const updateUserDraft = <K extends keyof UserDraft>(key: K, value: UserDraft[K]) => {
    setUserDraft((current) => ({ ...current, [key]: value }));
  };

  const startCreateUser = () => {
    setUserFormMode("create");
    setManagedUserId(null);
    setUserDraft(createEmptyUserDraft());
    setActiveTab("manage");
    clearFeedback();
  };

  const startManageUser = (targetUser: UserRead) => {
    setUserFormMode("edit");
    setManagedUserId(targetUser.id);
    setUserDraft(createUserDraftFromUser(targetUser));
    setActiveTab("manage");
    clearFeedback();
  };

  const startAssignmentEditing = (targetUser: UserRead) => {
    setAssignmentEditing({ userId: targetUser.id, departmentId: targetUser.department_id });
    setActiveTab("assign");
    clearFeedback();
  };

  const selectManageDepartment = (departmentId: string | null) => {
    setSelectedManageDepartmentId(departmentId);
    setDepartmentFormMode("detail");
    clearFeedback();
    const department = departmentId ? departments.find((item) => item.id === departmentId) ?? null : null;
    setDepartmentNameDraft(department?.name ?? "");
    setDepartmentParentDraft(department?.parent_id ?? null);
  };

  const startCreateDepartment = (parentId: string | null) => {
    setDepartmentFormMode("create");
    setDepartmentNameDraft("");
    setDepartmentParentDraft(parentId);
    setShowCreateParentSelector(parentId !== null);
    clearFeedback();
  };

  const handleUserSubmit = async () => {
    if (!token) return;

    const email = userDraft.email.trim().toLowerCase();
    const fullName = userDraft.full_name.trim();
    const password = userDraft.password.trim();
    if (!email || !email.includes("@")) {
      setError("邮箱格式不正确");
      return;
    }
    if (!fullName) {
      setError("姓名不能为空");
      return;
    }
    if (userFormMode === "create" && password.length < 6) {
      setError("新用户密码至少 6 位");
      return;
    }
    if (userFormMode === "edit" && password && password.length < 6) {
      setError("新密码至少 6 位");
      return;
    }
    if (userFormMode === "edit" && isManagingCurrentUser && (userDraft.role_name !== "admin" || !userDraft.is_active)) {
      setError("不能停用或移除自己的管理员权限");
      return;
    }

    setUserSaving(true);
    clearFeedback();
    try {
      const payload = {
        email,
        full_name: fullName,
        role_name: userDraft.role_name,
        department_id: userDraft.department_id,
        is_active: userDraft.is_active,
      };
      const savedUser =
        userFormMode === "create"
          ? await api.createUser(token, { ...payload, password })
          : selectedManagedUser
            ? await api.updateUser(token, selectedManagedUser.id, {
                ...payload,
                ...(password ? { password } : {}),
              })
            : null;
      if (!savedUser) return;

      await loadData();
      setUserFormMode("edit");
      setManagedUserId(savedUser.id);
      setUserDraft(createUserDraftFromUser(savedUser));
      if (savedUser.id === user?.id) {
        await refreshMe();
      }
      setStatusMessage(userFormMode === "create" ? "用户已创建。" : "用户已更新。");
    } catch (err) {
      setError(err instanceof Error ? err.message : "用户保存失败");
    } finally {
      setUserSaving(false);
    }
  };

  const handleUserDeactivate = async (targetUser: UserRead | null = selectedManagedUser) => {
    if (!token || !targetUser) return;
    if (targetUser.id === user?.id) {
      setError("不能停用当前登录用户");
      return;
    }
    const confirmed = window.confirm(`确定停用用户「${targetUser.full_name}」吗？停用后该用户不能再登录。`);
    if (!confirmed) return;

    setUserDeleting(true);
    clearFeedback();
    try {
      await api.deleteUser(token, targetUser.id);
      const loaded = await loadData();
      const deactivated = loaded?.usersData.find((item) => item.id === targetUser.id);
      if (deactivated) {
        setUserFormMode("edit");
        setManagedUserId(deactivated.id);
        setUserDraft(createUserDraftFromUser(deactivated));
      }
      setStatusMessage("用户已停用。");
    } catch (err) {
      setError(err instanceof Error ? err.message : "用户停用失败");
    } finally {
      setUserDeleting(false);
    }
  };

  const handleAssignmentSave = async () => {
    if (!token || !assignmentEditing) return;
    setSaving(true);
    clearFeedback();
    try {
      await api.updateUserDepartment(token, assignmentEditing.userId, assignmentEditing.departmentId);
      setAssignmentEditing(null);
      await loadData();
      await refreshMe();
      setStatusMessage("部门已更新。");
    } catch (err) {
      setError(err instanceof Error ? err.message : "保存失败");
    } finally {
      setSaving(false);
    }
  };

  const handleDepartmentSubmit = async () => {
    if (!token) return;
    const name = departmentNameDraft.trim();
    if (!name) {
      setError("部门名称不能为空");
      return;
    }

    setDepartmentSaving(true);
    clearFeedback();
    try {
      if (departmentFormMode === "create") {
        const created = await api.createDepartment(token, { name, parent_id: departmentParentDraft });
        await loadData();
        setSelectedManageDepartmentId(created.id);
        setDepartmentFormMode("detail");
        setDepartmentNameDraft(created.name);
        setDepartmentParentDraft(created.parent_id);
        setStatusMessage("部门已创建。");
      } else if (selectedManageDepartment) {
        const updated = await api.updateDepartment(token, selectedManageDepartment.id, {
          name,
          parent_id: departmentParentDraft,
        });
        await loadData();
        setSelectedManageDepartmentId(updated.id);
        setDepartmentNameDraft(updated.name);
        setDepartmentParentDraft(updated.parent_id);
        setStatusMessage("部门已更新。");
      }
      await refreshMe();
    } catch (err) {
      setError(err instanceof Error ? err.message : "部门保存失败");
    } finally {
      setDepartmentSaving(false);
    }
  };

  const handleDepartmentDelete = async () => {
    if (!token || !selectedManageDepartment) return;
    const confirmed = window.confirm(`确定删除部门「${selectedManageDepartment.path}」吗？`);
    if (!confirmed) return;

    setDepartmentSaving(true);
    clearFeedback();
    try {
      await api.deleteDepartment(token, selectedManageDepartment.id);
      await loadData();
      setSelectedManageDepartmentId(null);
      setDepartmentNameDraft("");
      setDepartmentParentDraft(null);
      setStatusMessage("部门已删除。");
    } catch (err) {
      setError(err instanceof Error ? err.message : "部门删除失败");
    } finally {
      setDepartmentSaving(false);
    }
  };

  if (!isAdmin) {
    return null;
  }

  return (
    <div className="page-stack">
      <PageHeader title="用户与部门管理" description="维护用户账号、角色、部门归属和部门层级。" />

      <ErrorNotice message={error} />
      {statusMessage ? <div className="info-block">{statusMessage}</div> : null}

      <div className="segmented-control users-mode-tabs">
        <button
          className={`segmented-button ${activeTab === "manage" ? "is-active" : ""}`}
          onClick={() => setActiveTab("manage")}
          type="button"
        >
          用户管理
        </button>
        <button
          className={`segmented-button ${activeTab === "assign" ? "is-active" : ""}`}
          onClick={() => setActiveTab("assign")}
          type="button"
        >
          用户分配
        </button>
        <button
          className={`segmented-button ${activeTab === "departments" ? "is-active" : ""}`}
          onClick={() => setActiveTab("departments")}
          type="button"
        >
          部门管理
        </button>
      </div>

      {activeTab === "manage" ? (
        <div className="page-grid user-management-layout">
          <section className="panel stack users-list-panel">
            <div className="panel-header">
              <div className="panel-heading">
                <h3>用户列表</h3>
                <p>查询用户，编辑账号、角色、部门和启停状态。</p>
              </div>
              <StatusBadge tone="info">{filteredManagedUsers.length} / {users.length} 位</StatusBadge>
            </div>

            <div className="users-toolbar">
              <input
                aria-label="搜索用户"
                onChange={(event) => setUserSearch(event.target.value)}
                placeholder="搜索姓名、邮箱或部门路径"
                value={userSearch}
              />
              <select
                aria-label="用户状态筛选"
                onChange={(event) => setUserStatusFilter(event.target.value as UserStatusFilter)}
                value={userStatusFilter}
              >
                <option value="all">全部状态</option>
                <option value="active">启用</option>
                <option value="inactive">停用</option>
              </select>
              <button className="primary-button" onClick={startCreateUser} type="button">
                新增用户
              </button>
            </div>

            <div className="users-list-scroll">
              {filteredManagedUsers.length === 0 ? (
                <div className="empty-state compact-empty-state">没有匹配的用户</div>
              ) : null}
              {filteredManagedUsers.map((item) => {
                const isCurrentUser = item.id === user?.id;
                const isSelected = managedUserId === item.id && userFormMode === "edit";

                return (
                  <div
                    key={item.id}
                    className={`list-card users-list-card user-management-card ${isSelected ? "is-selected" : ""}`}
                  >
                    <div className="users-card-main">
                      <div className="users-card-name">
                        <strong>{item.full_name}</strong>
                        {isCurrentUser ? <StatusBadge tone="info">当前用户</StatusBadge> : null}
                        {!item.is_active ? <StatusBadge tone="danger">已停用</StatusBadge> : null}
                      </div>
                      <p className="muted">{item.email}</p>
                      <div className="user-card-meta">
                        <span>
                          <span className="muted">角色</span>
                          <strong>{formatRoleName(item.role?.name)}</strong>
                        </span>
                        <span>
                          <span className="muted">部门</span>
                          <strong>{item.department?.path ?? "未设置"}</strong>
                        </span>
                      </div>
                    </div>
                    <div className="users-card-side user-management-card-actions">
                      <StatusBadge tone={item.is_active ? "success" : "neutral"}>
                        {item.is_active ? "启用" : "停用"}
                      </StatusBadge>
                      <button className="secondary-button compact-button" onClick={() => startManageUser(item)} type="button">
                        编辑
                      </button>
                      {item.is_active ? (
                        <button
                          className="secondary-button danger-button compact-button"
                          disabled={isCurrentUser || userDeleting}
                          onClick={() => handleUserDeactivate(item)}
                          type="button"
                        >
                          停用
                        </button>
                      ) : null}
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          <section className="panel stack user-form-panel">
            <div className="panel-header">
              <div className="panel-heading">
                <h3>{userFormMode === "create" ? "新增用户" : `编辑用户 - ${selectedManagedUser?.full_name ?? ""}`}</h3>
                <p>{userFormMode === "create" ? "创建后即可登录系统。" : "密码留空时不会修改原密码。"}</p>
              </div>
              <StatusBadge tone={userFormMode === "create" ? "info" : userDraft.is_active ? "success" : "warning"}>
                {userFormMode === "create" ? "创建" : userDraft.is_active ? "启用" : "停用"}
              </StatusBadge>
            </div>

            <div className="user-form-grid">
              <label>
                <span>姓名</span>
                <input
                  onChange={(event) => updateUserDraft("full_name", event.target.value)}
                  placeholder="输入姓名"
                  value={userDraft.full_name}
                />
              </label>
              <label>
                <span>邮箱</span>
                <input
                  onChange={(event) => updateUserDraft("email", event.target.value)}
                  placeholder="name@example.com"
                  type="email"
                  value={userDraft.email}
                />
              </label>
              <label>
                <span>角色</span>
                <select
                  disabled={userFormMode === "edit" && isManagingCurrentUser}
                  onChange={(event) => updateUserDraft("role_name", event.target.value as RoleName)}
                  value={userDraft.role_name}
                >
                  {ROLE_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>状态</span>
                <select
                  disabled={userFormMode === "edit" && isManagingCurrentUser}
                  onChange={(event) => updateUserDraft("is_active", event.target.value === "active")}
                  value={userDraft.is_active ? "active" : "inactive"}
                >
                  <option value="active">启用</option>
                  <option value="inactive">停用</option>
                </select>
              </label>
              <label className="user-form-full">
                <span>{userFormMode === "create" ? "初始密码" : "重置密码"}</span>
                <input
                  autoComplete="new-password"
                  onChange={(event) => updateUserDraft("password", event.target.value)}
                  placeholder={userFormMode === "create" ? "至少 6 位" : "留空则不修改"}
                  type="password"
                  value={userDraft.password}
                />
              </label>
            </div>

            <div className="users-selected-summary">
              <div>
                <span>当前选择部门</span>
                <strong>{selectedUserFormDepartment?.path ?? "未设置"}</strong>
              </div>
              <div>
                <span>角色权限</span>
                <strong>{formatRoleName(userDraft.role_name)}</strong>
              </div>
            </div>

            <DepartmentTreeSelect
              className="user-form-department-tree"
              departments={departments}
              emptyDescription="该用户不继承部门 ACL"
              emptyLabel="未设置部门"
              selectedId={userDraft.department_id}
              onSelect={(id) => updateUserDraft("department_id", id)}
            />

            <div className="inline-actions users-edit-actions">
              <button className="primary-button" disabled={userSaving} onClick={handleUserSubmit} type="button">
                {userSaving ? "保存中..." : userFormMode === "create" ? "创建用户" : "保存修改"}
              </button>
              {userFormMode === "edit" && selectedManagedUser?.is_active ? (
                <button
                  className="secondary-button danger-button"
                  disabled={userDeleting || isManagingCurrentUser}
                  onClick={() => handleUserDeactivate()}
                  type="button"
                >
                  {userDeleting ? "停用中..." : "停用用户"}
                </button>
              ) : null}
              <button className="secondary-button" disabled={userSaving || userDeleting} onClick={startCreateUser} type="button">
                清空表单
              </button>
            </div>
          </section>
        </div>
      ) : null}

      {activeTab === "assign" ? (
        <div className="page-grid users-layout">
          <section className="panel stack users-list-panel">
            <div className="panel-header">
              <div className="panel-heading">
                <h3>用户部门分配</h3>
                <p>点击「设置部门」进入快速分配模式。</p>
              </div>
              <StatusBadge tone="info">{assignableUsers.length} 位启用用户</StatusBadge>
            </div>
            <div className="users-list-scroll">
              {assignableUsers.map((item) => {
                const isCurrentUser = item.id === user?.id;
                const isEditing = assignmentEditing?.userId === item.id;

                return (
                  <div key={item.id} className={`list-card users-list-card ${isEditing ? "is-selected" : ""}`}>
                    <div className="users-card-main">
                      <div className="users-card-name">
                        <strong>{item.full_name}</strong>
                        {isCurrentUser ? <StatusBadge tone="info">当前用户</StatusBadge> : null}
                      </div>
                      <p className="muted">{item.email}</p>
                      <div className="metadata-subline">
                        <span className="muted">部门</span>
                        <span className="users-dept-path">{item.department?.path ?? "未设置"}</span>
                      </div>
                    </div>
                    <div className="users-card-side">
                      <StatusBadge tone={item.role?.name === "admin" ? "warning" : "neutral"}>
                        {formatRoleName(item.role?.name)}
                      </StatusBadge>
                      {!isEditing ? (
                        <button
                          className="secondary-button users-card-edit-button"
                          onClick={() => startAssignmentEditing(item)}
                          type="button"
                        >
                          设置部门
                        </button>
                      ) : (
                        <StatusBadge tone="info">编辑中</StatusBadge>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          <section className="panel stack users-dept-panel">
            {assignmentEditing && selectedAssignmentUser ? (
              <>
                <div className="panel-header">
                  <div className="panel-heading">
                    <h3>编辑部门 - {selectedAssignmentUser.full_name}</h3>
                    <p>选择目标部门后保存，部门权限会按层级继承。</p>
                  </div>
                  <StatusBadge tone={assignmentEditing.departmentId ? "success" : "warning"}>
                    {assignmentEditing.departmentId ? "已选择" : "未设置"}
                  </StatusBadge>
                </div>

                <div className="users-selected-summary">
                  <div>
                    <span>当前部门</span>
                    <strong>{selectedAssignmentUser.department?.path ?? "未设置"}</strong>
                  </div>
                  <div>
                    <span>目标部门</span>
                    <strong>{selectedAssignmentDepartment?.path ?? "未设置"}</strong>
                  </div>
                </div>

                <DepartmentTreeSelect
                  departments={departments}
                  selectedId={assignmentEditing.departmentId}
                  onSelect={(id) => setAssignmentEditing({ ...assignmentEditing, departmentId: id })}
                />

                <div className="inline-actions users-edit-actions">
                  <button
                    className="primary-button"
                    onClick={handleAssignmentSave}
                    disabled={saving || !hasDepartmentChange}
                    type="button"
                  >
                    {saving ? "保存中..." : "保存"}
                  </button>
                  <button
                    className="secondary-button"
                    onClick={() => setAssignmentEditing(null)}
                    disabled={saving}
                    type="button"
                  >
                    取消
                  </button>
                </div>
              </>
            ) : (
              <div className="users-assignment-empty">
                <strong>选择一个用户</strong>
                <p className="muted">点击左侧用户卡片里的「设置部门」，然后在这里选择目标部门。</p>
              </div>
            )}
          </section>
        </div>
      ) : null}

      {activeTab === "departments" ? (
        <div className="page-grid department-management-layout">
          <section className="panel stack department-management-tree-panel">
            <div className="panel-header">
              <div className="panel-heading">
                <h3>部门树</h3>
                <p>选择部门后查看详情，或创建新的一级部门。</p>
              </div>
              <button className="secondary-button" onClick={() => startCreateDepartment(null)} type="button">
                新增一级部门
              </button>
            </div>
            <DepartmentTreeSelect
              className="department-management-tree"
              departments={departments}
              emptyDescription="选择部门后查看详情"
              emptyLabel="未选择部门"
              selectedId={selectedManageDepartmentId}
              onSelect={selectManageDepartment}
            />
          </section>

          <section className="panel stack department-management-detail-panel">
            {departmentFormMode === "create" ? (
              <>
                <div className="panel-header">
                  <div className="panel-heading">
                    <h3>{departmentParentDraft ? "新增子部门" : "新增一级部门"}</h3>
                    <p>部门名称会生成中文展示路径，系统会自动生成组织编号。</p>
                  </div>
                  <StatusBadge tone="info">创建</StatusBadge>
                </div>
                <div className="department-detail-grid">
                  <div>
                    <span>父部门</span>
                    <strong>
                      {departmentParentDraft
                        ? departments.find((department) => department.id === departmentParentDraft)?.path ?? "未知部门"
                        : "顶层部门"}
                    </strong>
                  </div>
                </div>
                <label>
                  <span>部门名称</span>
                  <input
                    onChange={(event) => setDepartmentNameDraft(event.target.value)}
                    placeholder="输入部门名称"
                    value={departmentNameDraft}
                  />
                </label>
                {showCreateParentSelector ? (
                  <div className="department-management-subsection">
                    <div className="subsection-header">
                      <div className="panel-heading">
                        <h4>父部门位置</h4>
                        <p>选择父部门，或选择顶层部门。</p>
                      </div>
                    </div>
                    <DepartmentTreeSelect
                      className="department-parent-tree"
                      departments={departments}
                      emptyDescription="创建为一级部门"
                      emptyLabel="顶层部门"
                      selectedId={departmentParentDraft}
                      onSelect={setDepartmentParentDraft}
                    />
                  </div>
                ) : null}
                <div className="inline-actions users-edit-actions">
                  <button
                    className="primary-button"
                    disabled={departmentSaving || !departmentNameDraft.trim()}
                    onClick={handleDepartmentSubmit}
                    type="button"
                  >
                    {departmentSaving ? "保存中..." : "创建部门"}
                  </button>
                  <button
                    className="secondary-button"
                    disabled={departmentSaving}
                    onClick={() => setDepartmentFormMode("detail")}
                    type="button"
                  >
                    取消
                  </button>
                </div>
              </>
            ) : selectedManageDepartment ? (
              <>
                <div className="panel-header">
                  <div className="panel-heading">
                    <h3 className="department-title-with-code">
                      <span>{selectedManageDepartment.name}</span>
                      <span className="dept-code-badge">{selectedManageDepartment.org_code}</span>
                    </h3>
                    <p>重命名只改变中文路径；移动部门会同步更新中文路径和组织编号路径。</p>
                  </div>
                  <StatusBadge tone="success">已选择</StatusBadge>
                </div>

                <div className="department-detail-grid">
                  <div>
                    <span>中文路径</span>
                    <strong>{selectedManageDepartment.path}</strong>
                  </div>
                  <div>
                    <span>组织编号路径</span>
                    <strong>{selectedManageDepartment.org_code_path}</strong>
                  </div>
                  <div>
                    <span>父部门</span>
                    <strong>{selectedManageParent?.path ?? "顶层部门"}</strong>
                  </div>
                  <div>
                    <span>子部门</span>
                    <strong>{departmentChildCount} 个</strong>
                  </div>
                </div>

                <label>
                  <span>部门名称</span>
                  <input
                    onChange={(event) => setDepartmentNameDraft(event.target.value)}
                    placeholder="输入部门名称"
                    value={departmentNameDraft}
                  />
                </label>

                <div className="department-management-subsection">
                  <div className="subsection-header">
                    <div className="panel-heading">
                      <h4>移动位置</h4>
                      <p>不能移动到自身或子部门下，后端也会再次校验。</p>
                    </div>
                  </div>
                  <DepartmentTreeSelect
                    className="department-parent-tree"
                    departments={movableParentDepartments}
                    emptyDescription="移动到组织根层级"
                    emptyLabel="顶层部门"
                    selectedId={departmentParentDraft}
                    onSelect={setDepartmentParentDraft}
                  />
                </div>

                <div className="inline-actions users-edit-actions department-management-actions">
                  <button
                    className="primary-button"
                    disabled={departmentSaving || !departmentNameDraft.trim()}
                    onClick={handleDepartmentSubmit}
                    type="button"
                  >
                    {departmentSaving ? "保存中..." : "保存修改"}
                  </button>
                  <button
                    className="secondary-button"
                    disabled={departmentSaving}
                    onClick={() => startCreateDepartment(selectedManageDepartment.id)}
                    type="button"
                  >
                    新增子部门
                  </button>
                  <button
                    className="secondary-button danger-button"
                    disabled={departmentSaving}
                    onClick={handleDepartmentDelete}
                    type="button"
                  >
                    删除部门
                  </button>
                </div>
              </>
            ) : (
              <div className="users-assignment-empty">
                <strong>选择一个部门</strong>
                <p className="muted">在左侧部门树中选择部门查看详情、重命名、移动或删除。</p>
              </div>
            )}
          </section>
        </div>
      ) : null}
    </div>
  );
}
