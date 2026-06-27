import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { DepartmentTreeSelect } from "../components/DepartmentTreeSelect";
import { ErrorNotice } from "../components/ErrorNotice";
import { PageHeader } from "../components/PageHeader";
import { SelectField } from "../components/SelectField";
import { StatusBadge } from "../components/StatusBadge";
import { useAppContext } from "../context/AppContext";
import { api } from "../lib/api";
import { formatDocumentStatus, formatRoleName } from "../lib/display";
import { formatDateTime } from "../lib/format";
import type { DepartmentRead, RoleName, UserRead, UserVisibleScopeRead } from "../types/api";

interface UserDraft {
  email: string;
  full_name: string;
  password: string;
  role_name: RoleName;
  department_id: string | null;
  is_active: boolean;
}

interface OverviewDepartmentNode {
  department: DepartmentRead;
  children: OverviewDepartmentNode[];
}

interface DepartmentOverviewStats {
  directUsers: UserRead[];
  subtreeUsers: UserRead[];
  activeCount: number;
  inactiveCount: number;
  roleCounts: Record<RoleName, number>;
}

interface UsersPageCache {
  users: UserRead[];
  departments: DepartmentRead[];
  activeTab: UsersPageTab;
  managedUserId: string | null;
  userFormMode: UserFormMode;
  userDraft: UserDraft;
  overviewSelectedId: string | null;
  selectedManageDepartmentId: string | null;
}

type UsersPageTab = "manage" | "overview" | "departments";
type DepartmentFormMode = "detail" | "create";
type UserFormMode = "create" | "edit";
type UserStatusFilter = "all" | "active" | "inactive";

const UNASSIGNED_SELECTION = "__unassigned__";

const ROLE_OPTIONS: Array<{ value: RoleName; label: string }> = [
  { value: "viewer", label: "普通员工" },
  { value: "manager", label: "组长" },
  { value: "admin", label: "管理员" },
];

const USER_STATUS_FILTER_OPTIONS: Array<{ value: UserStatusFilter; label: string }> = [
  { value: "all", label: "全部状态" },
  { value: "active", label: "启用" },
  { value: "inactive", label: "停用" },
];

const USER_ACTIVE_OPTIONS: Array<{ value: "active" | "inactive"; label: string }> = [
  { value: "active", label: "启用" },
  { value: "inactive", label: "停用" },
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

function buildOverviewTree(departments: DepartmentRead[]): OverviewDepartmentNode[] {
  const map = new Map<string, OverviewDepartmentNode>();
  for (const department of departments) {
    map.set(department.id, { department, children: [] });
  }

  const roots: OverviewDepartmentNode[] = [];
  for (const node of map.values()) {
    if (node.department.parent_id && map.has(node.department.parent_id)) {
      map.get(node.department.parent_id)!.children.push(node);
    } else {
      roots.push(node);
    }
  }

  const sortNodes = (nodes: OverviewDepartmentNode[]) => {
    nodes.sort((a, b) => a.department.name.localeCompare(b.department.name, "zh-CN"));
    for (const node of nodes) {
      sortNodes(node.children);
    }
  };
  sortNodes(roots);
  return roots;
}

function collectBranchIds(nodes: OverviewDepartmentNode[]): string[] {
  const ids: string[] = [];
  for (const node of nodes) {
    if (node.children.length > 0) {
      ids.push(node.department.id, ...collectBranchIds(node.children));
    }
  }
  return ids;
}

function filterOverviewTree(
  nodes: OverviewDepartmentNode[],
  query: string,
  statsByDepartment: Map<string, DepartmentOverviewStats>,
): OverviewDepartmentNode[] {
  const normalizedQuery = query.trim().toLowerCase();
  if (!normalizedQuery) {
    return nodes;
  }

  return nodes.flatMap((node) => {
    const stats = statsByDepartment.get(node.department.id);
    const selfMatches =
      node.department.name.toLowerCase().includes(normalizedQuery) ||
      node.department.path.toLowerCase().includes(normalizedQuery) ||
      node.department.org_code.toLowerCase().includes(normalizedQuery) ||
      node.department.org_code_path.toLowerCase().includes(normalizedQuery) ||
      stats?.directUsers.some(
        (item) =>
          item.full_name.toLowerCase().includes(normalizedQuery) ||
          item.email.toLowerCase().includes(normalizedQuery),
      );
    const filteredChildren = filterOverviewTree(node.children, query, statsByDepartment);
    if (!selfMatches && filteredChildren.length === 0) {
      return [];
    }
    return [{ department: node.department, children: selfMatches ? node.children : filteredChildren }];
  });
}

function isUserInDepartmentSubtree(user: UserRead, department: DepartmentRead): boolean {
  const userDepartmentPath = user.department?.id_path;
  return Boolean(
    user.department_id === department.id ||
      userDepartmentPath === department.id_path ||
      userDepartmentPath?.startsWith(`${department.id_path}/`),
  );
}

function createRoleCounts(users: UserRead[]): Record<RoleName, number> {
  return ROLE_OPTIONS.reduce(
    (acc, option) => {
      acc[option.value] = users.filter((item) => item.role?.name === option.value).length;
      return acc;
    },
    { viewer: 0, manager: 0, admin: 0 } as Record<RoleName, number>,
  );
}

function isDepartmentOptionalUser(user: UserRead): boolean {
  return user.role?.name === "admin";
}

function getUserDepartmentLabel(user: UserRead): string {
  if (user.department) {
    return user.department.path;
  }
  return isDepartmentOptionalUser(user) ? "管理员免分配" : "未设置";
}

function OrgOverviewNode({
  node,
  expanded,
  selectedId,
  statsByDepartment,
  onSelect,
  onToggle,
}: {
  node: OverviewDepartmentNode;
  expanded: Set<string>;
  selectedId: string | null;
  statsByDepartment: Map<string, DepartmentOverviewStats>;
  onSelect: (id: string) => void;
  onToggle: (id: string) => void;
}) {
  const hasChildren = node.children.length > 0;
  const isExpanded = expanded.has(node.department.id);
  const isSelected = selectedId === node.department.id;
  const stats = statsByDepartment.get(node.department.id);
  const directCount = stats?.directUsers.length ?? 0;
  const subtreeCount = stats?.subtreeUsers.length ?? 0;
  const nonZeroRoles = ROLE_OPTIONS.filter((option) => (stats?.roleCounts[option.value] ?? 0) > 0);

  return (
    <div className="org-tree-branch">
      <div className={`org-tree-node ${isSelected ? "is-selected" : ""}`}>
        {hasChildren ? (
          <button
            aria-label={isExpanded ? "收起部门" : "展开部门"}
            className="org-tree-toggle"
            onClick={() => onToggle(node.department.id)}
            type="button"
          >
            {isExpanded ? "▾" : "▸"}
          </button>
        ) : (
          <span className="org-tree-spacer" />
        )}
        <button className="org-tree-card" onClick={() => onSelect(node.department.id)} type="button">
          <span className="org-tree-mainline">
            <span className="org-tree-name">{node.department.name}</span>
            <span className="dept-code-badge">{node.department.org_code}</span>
          </span>
          <span className="org-tree-path">{node.department.path}</span>
          <span className="org-tree-counts">
            <span>直属 {directCount}</span>
            <span>总计 {subtreeCount}</span>
            <span>子部门 {node.children.length}</span>
          </span>
          <span className="org-role-chips">
            {nonZeroRoles.length > 0 ? (
              nonZeroRoles.map((option) => (
                <span key={option.value}>
                  {option.label} {stats?.roleCounts[option.value] ?? 0}
                </span>
              ))
            ) : (
              <span>暂无成员</span>
            )}
          </span>
        </button>
      </div>
      {hasChildren && isExpanded ? (
        <div className="org-tree-children">
          {node.children.map((child) => (
            <OrgOverviewNode
              key={child.department.id}
              node={child}
              expanded={expanded}
              selectedId={selectedId}
              statsByDepartment={statsByDepartment}
              onSelect={onSelect}
              onToggle={onToggle}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}

export function UsersPage() {
  const { token, user, refreshMe, getPageCache, setPageCache } = useAppContext();
  const navigate = useNavigate();
  const cachedPage = getPageCache<UsersPageCache>("users");

  const [users, setUsers] = useState<UserRead[]>(() => cachedPage?.users ?? []);
  const [departments, setDepartments] = useState<DepartmentRead[]>(() => cachedPage?.departments ?? []);
  const [departmentSaving, setDepartmentSaving] = useState(false);
  const [userSaving, setUserSaving] = useState(false);
  const [userDeleting, setUserDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<UsersPageTab>(() => cachedPage?.activeTab ?? "manage");
  const [userSearch, setUserSearch] = useState("");
  const [userStatusFilter, setUserStatusFilter] = useState<UserStatusFilter>("all");
  const [managedUserId, setManagedUserId] = useState<string | null>(() => cachedPage?.managedUserId ?? null);
  const [userFormMode, setUserFormMode] = useState<UserFormMode>(() => cachedPage?.userFormMode ?? "create");
  const [userDraft, setUserDraft] = useState<UserDraft>(() => cachedPage?.userDraft ?? createEmptyUserDraft());
  const [userScope, setUserScope] = useState<UserVisibleScopeRead | null>(null);
  const [userScopeLoading, setUserScopeLoading] = useState(false);
  const [userScopeError, setUserScopeError] = useState<string | null>(null);
  const [overviewSelectedId, setOverviewSelectedId] = useState<string | null>(
    () => cachedPage?.overviewSelectedId ?? UNASSIGNED_SELECTION,
  );
  const [overviewQuery, setOverviewQuery] = useState("");
  const [overviewExpanded, setOverviewExpanded] = useState<Set<string>>(new Set());
  const [selectedManageDepartmentId, setSelectedManageDepartmentId] = useState<string | null>(
    () => cachedPage?.selectedManageDepartmentId ?? null,
  );
  const [departmentFormMode, setDepartmentFormMode] = useState<DepartmentFormMode>("detail");
  const [departmentNameDraft, setDepartmentNameDraft] = useState("");
  const [departmentParentDraft, setDepartmentParentDraft] = useState<string | null>(null);
  const [showCreateParentSelector, setShowCreateParentSelector] = useState(false);

  const isAdmin = user?.role?.name === "admin";
  const activeUserCount = useMemo(() => users.filter((item) => item.is_active).length, [users]);
  const inactiveUserCount = users.length - activeUserCount;
  const unassignedRequiredUsers = useMemo(
    () => users.filter((item) => !item.department_id && !isDepartmentOptionalUser(item)),
    [users],
  );
  const departmentOptionalUsers = useMemo(
    () => users.filter((item) => !item.department_id && isDepartmentOptionalUser(item)),
    [users],
  );
  const departmentById = useMemo(() => {
    const map = new Map<string, DepartmentRead>();
    for (const department of departments) {
      map.set(department.id, department);
    }
    return map;
  }, [departments]);
  const overviewTree = useMemo(() => buildOverviewTree(departments), [departments]);
  const statsByDepartment = useMemo(() => {
    const map = new Map<string, DepartmentOverviewStats>();
    for (const department of departments) {
      const directUsers = users.filter((item) => item.department_id === department.id);
      const subtreeUsers = users.filter((item) => isUserInDepartmentSubtree(item, department));
      map.set(department.id, {
        directUsers,
        subtreeUsers,
        activeCount: subtreeUsers.filter((item) => item.is_active).length,
        inactiveCount: subtreeUsers.filter((item) => !item.is_active).length,
        roleCounts: createRoleCounts(subtreeUsers),
      });
    }
    return map;
  }, [departments, users]);
  const filteredOverviewTree = useMemo(
    () => filterOverviewTree(overviewTree, overviewQuery, statsByDepartment),
    [overviewQuery, overviewTree, statsByDepartment],
  );
  const allOverviewBranchIds = useMemo(() => collectBranchIds(overviewTree), [overviewTree]);
  const visibleOverviewBranchIds = useMemo(() => collectBranchIds(filteredOverviewTree), [filteredOverviewTree]);
  const overviewSelectedAncestorIds = useMemo(() => {
    const ids = new Set<string>();
    let cursor =
      overviewSelectedId && overviewSelectedId !== UNASSIGNED_SELECTION
        ? departmentById.get(overviewSelectedId)
        : null;
    while (cursor?.parent_id) {
      ids.add(cursor.parent_id);
      cursor = departmentById.get(cursor.parent_id);
    }
    return ids;
  }, [departmentById, overviewSelectedId]);
  const emptyDepartmentCount = useMemo(
    () => departments.filter((department) => (statsByDepartment.get(department.id)?.subtreeUsers.length ?? 0) === 0).length,
    [departments, statsByDepartment],
  );
  const largestDepartment = useMemo(() => {
    return departments.reduce<DepartmentRead | null>((largest, department) => {
      if (!largest) return department;
      const currentCount = statsByDepartment.get(department.id)?.subtreeUsers.length ?? 0;
      const largestCount = statsByDepartment.get(largest.id)?.subtreeUsers.length ?? 0;
      return currentCount > largestCount ? department : largest;
    }, null);
  }, [departments, statsByDepartment]);
  const selectedManagedUser = useMemo(
    () => (managedUserId ? users.find((item) => item.id === managedUserId) ?? null : null),
    [managedUserId, users],
  );
  const selectedUserFormDepartment = useMemo(
    () => (userDraft.department_id ? departments.find((item) => item.id === userDraft.department_id) ?? null : null),
    [departments, userDraft.department_id],
  );
  const isManagingCurrentUser = selectedManagedUser?.id === user?.id;
  const selectedOverviewDepartment =
    overviewSelectedId && overviewSelectedId !== UNASSIGNED_SELECTION ? departmentById.get(overviewSelectedId) ?? null : null;
  const selectedOverviewStats = selectedOverviewDepartment
    ? statsByDepartment.get(selectedOverviewDepartment.id) ?? null
    : null;
  const overviewDetailUsers =
    overviewSelectedId === UNASSIGNED_SELECTION ? unassignedRequiredUsers : selectedOverviewStats?.directUsers ?? [];
  const overviewDistributionUsers =
    overviewSelectedId === UNASSIGNED_SELECTION ? unassignedRequiredUsers : selectedOverviewStats?.subtreeUsers ?? users;
  const overviewRoleCounts = useMemo(() => createRoleCounts(overviewDistributionUsers), [overviewDistributionUsers]);
  const overviewMaxRoleCount = Math.max(1, ...ROLE_OPTIONS.map((option) => overviewRoleCounts[option.value]));
  const selectedOverviewChildren = selectedOverviewDepartment
    ? departments.filter((department) => department.parent_id === selectedOverviewDepartment.id)
    : [];
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

  useEffect(() => {
    setPageCache<UsersPageCache>("users", {
      users,
      departments,
      activeTab,
      managedUserId,
      userFormMode,
      userDraft,
      overviewSelectedId,
      selectedManageDepartmentId,
    });
  }, [
    activeTab,
    departments,
    managedUserId,
    overviewSelectedId,
    selectedManageDepartmentId,
    setPageCache,
    userDraft,
    userFormMode,
    users,
  ]);

  useEffect(() => {
    if (!isAdmin) {
      navigate("/documents", { replace: true });
    }
  }, [isAdmin, navigate]);

  useEffect(() => {
    setOverviewExpanded((current) => {
      const next = new Set(current);
      for (const root of overviewTree) {
        next.add(root.department.id);
      }
      for (const id of overviewSelectedAncestorIds) {
        next.add(id);
      }
      if (overviewQuery.trim()) {
        for (const id of visibleOverviewBranchIds) {
          next.add(id);
        }
      }
      return next;
    });
  }, [overviewQuery, overviewSelectedAncestorIds, overviewTree, visibleOverviewBranchIds]);

  useEffect(() => {
    if (overviewSelectedId === UNASSIGNED_SELECTION) return;
    if (overviewSelectedId && departmentById.has(overviewSelectedId)) return;
    setOverviewSelectedId(unassignedRequiredUsers.length > 0 ? UNASSIGNED_SELECTION : departments[0]?.id ?? null);
  }, [departmentById, departments, overviewSelectedId, unassignedRequiredUsers.length]);

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

  useEffect(() => {
    if (!token || userFormMode !== "edit" || !selectedManagedUser) {
      setUserScope(null);
      setUserScopeError(null);
      setUserScopeLoading(false);
      return;
    }

    let cancelled = false;
    setUserScopeLoading(true);
    setUserScopeError(null);
    api
      .getUserVisibleScope(token, selectedManagedUser.id, 12)
      .then((scope) => {
        if (!cancelled) {
          setUserScope(scope);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setUserScope(null);
          setUserScopeError(err instanceof Error ? err.message : "加载用户可见范围失败");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setUserScopeLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selectedManagedUser, token, userFormMode]);

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

  const copyUserId = async (targetUser: UserRead) => {
    try {
      await navigator.clipboard.writeText(targetUser.id);
      setStatusMessage("用户 ID 已复制。");
    } catch {
      setError("复制用户 ID 失败，请手动复制。");
    }
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

  const handleUserActiveToggle = async (targetUser: UserRead | null = selectedManagedUser) => {
    if (!token || !targetUser) return;
    if (targetUser.is_active && targetUser.id === user?.id) {
      setError("不能停用当前登录用户");
      return;
    }
    const confirmed = window.confirm(
      targetUser.is_active
        ? `确定停用用户「${targetUser.full_name}」吗？停用后该用户不能再登录。`
        : `确定启用用户「${targetUser.full_name}」吗？启用后该用户可以重新登录。`,
    );
    if (!confirmed) return;

    setUserDeleting(true);
    clearFeedback();
    try {
      if (targetUser.is_active) {
        await api.deleteUser(token, targetUser.id);
      } else {
        await api.updateUser(token, targetUser.id, { is_active: true });
      }
      const loaded = await loadData();
      const updated = loaded?.usersData.find((item) => item.id === targetUser.id);
      if (updated) {
        setUserFormMode("edit");
        setManagedUserId(updated.id);
        setUserDraft(createUserDraftFromUser(updated));
      }
      setStatusMessage(targetUser.is_active ? "用户已停用。" : "用户已启用。");
    } catch (err) {
      setError(err instanceof Error ? err.message : "用户状态更新失败");
    } finally {
      setUserDeleting(false);
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

  const handleDepartmentMove = async (departmentId: string, parentId: string | null) => {
    if (!token) return;
    const department = departments.find((item) => item.id === departmentId);
    if (!department || department.parent_id === parentId) return;

    setDepartmentSaving(true);
    clearFeedback();
    try {
      const updated = await api.updateDepartment(token, departmentId, { parent_id: parentId });
      await loadData();
      setSelectedManageDepartmentId(updated.id);
      setDepartmentFormMode("detail");
      setDepartmentNameDraft(updated.name);
      setDepartmentParentDraft(updated.parent_id);
      setStatusMessage(`部门已移动到 ${updated.parent_id ? updated.path : "顶层"}。`);
      await refreshMe();
    } catch (err) {
      setError(err instanceof Error ? err.message : "部门移动失败");
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
          className={`segmented-button ${activeTab === "overview" ? "is-active" : ""}`}
          onClick={() => setActiveTab("overview")}
          type="button"
        >
          组织概览
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
              <StatusBadge tone="info">
                {filteredManagedUsers.length} / {users.length} 位
              </StatusBadge>
            </div>

            <div className="users-toolbar">
              <input
                aria-label="搜索用户"
                onChange={(event) => setUserSearch(event.target.value)}
                placeholder="搜索姓名、邮箱或部门路径"
                value={userSearch}
              />
              <SelectField
                aria-label="用户状态筛选"
                options={USER_STATUS_FILTER_OPTIONS}
                onChange={setUserStatusFilter}
                value={userStatusFilter}
              />
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
                        <StatusBadge tone={item.is_active ? "success" : "danger"}>
                          {item.is_active ? "启用" : "停用"}
                        </StatusBadge>
                      </div>
                      <div className="users-card-email-row">
                        <p className="muted">{item.email}</p>
                        <button className="user-id-copy-button" onClick={() => copyUserId(item)} type="button">
                          复制ID
                        </button>
                      </div>
                      <div className="user-card-meta user-management-meta">
                        <span>
                          <span className="muted">部门</span>
                          <strong>{getUserDepartmentLabel(item)}</strong>
                        </span>
                      </div>
                    </div>
                    <div className="users-card-side user-management-card-actions">
                      <StatusBadge tone={item.role?.name === "admin" ? "warning" : "neutral"}>
                        {formatRoleName(item.role?.name)}
                      </StatusBadge>
                      {isSelected ? (
                        <StatusBadge tone="info">编辑中</StatusBadge>
                      ) : (
                        <button
                          className="secondary-button users-card-edit-button user-management-edit-button"
                          onClick={() => startManageUser(item)}
                          type="button"
                        >
                          编辑用户
                        </button>
                      )}
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
              <div className="user-form-header-actions">
                {userFormMode === "edit" && selectedManagedUser ? (
                  <button
                    className="user-id-chip"
                    onClick={() => copyUserId(selectedManagedUser)}
                    title={selectedManagedUser.id}
                    type="button"
                  >
                    复制ID
                  </button>
                ) : null}
                <StatusBadge tone={userFormMode === "create" ? "info" : userDraft.is_active ? "success" : "warning"}>
                  {userFormMode === "create" ? "创建" : userDraft.is_active ? "启用" : "停用"}
                </StatusBadge>
              </div>
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
                <SelectField
                  disabled={userFormMode === "edit" && isManagingCurrentUser}
                  options={ROLE_OPTIONS}
                  onChange={(value) => updateUserDraft("role_name", value)}
                  value={userDraft.role_name}
                />
              </label>
              <label>
                <span>状态</span>
                <SelectField
                  disabled={userFormMode === "edit" && isManagingCurrentUser}
                  options={USER_ACTIVE_OPTIONS}
                  onChange={(value) => updateUserDraft("is_active", value === "active")}
                  value={userDraft.is_active ? "active" : "inactive"}
                />
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
                <strong>{selectedUserFormDepartment?.path ?? (userDraft.role_name === "admin" ? "管理员免分配" : "未设置")}</strong>
              </div>
              <div>
                <span>角色权限</span>
                <strong>{formatRoleName(userDraft.role_name)}</strong>
              </div>
            </div>

            {userFormMode === "edit" ? (
              <div className="info-block permission-scope-panel">
                <div className="list-card-topline">
                  <strong>可见文档范围</strong>
                  {userScopeLoading ? <StatusBadge tone="neutral">加载中</StatusBadge> : null}
                  {userScope ? <StatusBadge tone="info">{userScope.visible_document_count} 份可见</StatusBadge> : null}
                </div>
                {userScopeError ? <p className="muted">{userScopeError}</p> : null}
                {userScope ? (
                  <>
                    <div className="metadata-grid permission-scope-stats">
                      <span>可管理：{userScope.manageable_document_count}</span>
                      <span>所有者命中：{userScope.permission_summary.owner_count}</span>
                      <span>ACL 命中：{userScope.permission_summary.acl_count}</span>
                      <span>公开：{userScope.permission_summary.public_acl_count}</span>
                      <span>部门：{userScope.permission_summary.department_acl_count}</span>
                      <span>角色：{userScope.permission_summary.role_acl_count}</span>
                      <span>指定用户：{userScope.permission_summary.user_acl_count}</span>
                    </div>
                    <div className="permission-scope-documents">
                      {userScope.visible_documents.length === 0 ? (
                        <p className="muted">该用户暂无可见文档。</p>
                      ) : null}
                      {userScope.visible_documents.map((document) => (
                        <div className="list-card compact-list-card" key={document.id}>
                          <div className="list-card-topline">
                            <strong>{document.title}</strong>
                            <StatusBadge tone={document.can_manage ? "warning" : "neutral"}>
                              {document.can_manage ? "可管理" : "可查看"}
                            </StatusBadge>
                          </div>
                          <p className="muted">
                            {formatDocumentStatus(document.status)} · {document.reason} · {formatDateTime(document.updated_at)}
                          </p>
                        </div>
                      ))}
                    </div>
                  </>
                ) : !userScopeLoading && !userScopeError ? (
                  <p className="muted">选择用户后查看该用户可访问的企业文档范围。</p>
                ) : null}
              </div>
            ) : null}

            <DepartmentTreeSelect
              className="user-form-department-tree"
              departments={departments}
              emptyDescription={
                userDraft.role_name === "admin" ? "管理员可不加入部门，仍保留全局管理权限" : "该用户不继承部门 ACL"
              }
              emptyLabel={userDraft.role_name === "admin" ? "管理员免分配" : "未设置部门"}
              selectedId={userDraft.department_id}
              onSelect={(id) => updateUserDraft("department_id", id)}
            />

            <div className="inline-actions users-edit-actions">
              <button className="primary-button" disabled={userSaving} onClick={handleUserSubmit} type="button">
                {userSaving ? "保存中..." : userFormMode === "create" ? "创建用户" : "保存修改"}
              </button>
              {userFormMode === "edit" && selectedManagedUser ? (
                <button
                  className={`secondary-button ${selectedManagedUser.is_active ? "danger-button" : ""}`}
                  disabled={userDeleting || (selectedManagedUser.is_active && isManagingCurrentUser)}
                  onClick={() => handleUserActiveToggle()}
                  type="button"
                >
                  {userDeleting ? "更新中..." : selectedManagedUser.is_active ? "停用用户" : "启用用户"}
                </button>
              ) : null}
              <button className="secondary-button" disabled={userSaving || userDeleting} onClick={startCreateUser} type="button">
                清空表单
              </button>
            </div>
          </section>
        </div>
      ) : null}

      {activeTab === "overview" ? (
        <div className="org-overview-page">
          <div className="org-overview-stats">
            <div className="org-stat-card">
              <span>总用户</span>
              <strong>{users.length}</strong>
              <small>覆盖全部账号</small>
            </div>
            <div className="org-stat-card">
              <span>启用用户</span>
              <strong>{activeUserCount}</strong>
              <small>{inactiveUserCount} 位停用</small>
            </div>
            <div className="org-stat-card org-stat-warning">
              <span>未设置部门</span>
              <strong>{unassignedRequiredUsers.length}</strong>
              <small>
                {departmentOptionalUsers.length > 0
                  ? `管理员免分配 ${departmentOptionalUsers.length} 位`
                  : "普通/组长需补齐"}
              </small>
            </div>
            <div className="org-stat-card">
              <span>部门数量</span>
              <strong>{departments.length}</strong>
              <small>{emptyDepartmentCount} 个空部门</small>
            </div>
          </div>

          <div className="page-grid org-overview-layout">
            <section className="panel stack org-tree-panel">
              <div className="panel-header">
                <div className="panel-heading">
                  <h3>组织树</h3>
                  <p>
                    最大部门：{largestDepartment?.path ?? "暂无"}，
                    {largestDepartment ? statsByDepartment.get(largestDepartment.id)?.subtreeUsers.length ?? 0 : 0} 位用户
                  </p>
                </div>
                <StatusBadge tone="info">{departments.length} 个部门</StatusBadge>
              </div>
              <div className="org-overview-toolbar">
                <input
                  aria-label="搜索组织"
                  onChange={(event) => setOverviewQuery(event.target.value)}
                  placeholder="搜索部门、编号、路径或用户"
                  value={overviewQuery}
                />
                <div className="dept-tree-toolbar-actions">
                  <button
                    className="secondary-button compact-button"
                    onClick={() => setOverviewExpanded(new Set(allOverviewBranchIds))}
                    type="button"
                  >
                    展开全部
                  </button>
                  <button
                    className="secondary-button compact-button"
                    onClick={() => setOverviewExpanded(new Set())}
                    type="button"
                  >
                    收起全部
                  </button>
                </div>
              </div>
              <div className="org-tree-scroll">
                <button
                  className={`org-unassigned-card ${overviewSelectedId === UNASSIGNED_SELECTION ? "is-selected" : ""}`}
                  onClick={() => setOverviewSelectedId(UNASSIGNED_SELECTION)}
                  type="button"
                >
                  <span>
                    <strong>未设置部门</strong>
                    <small>普通/组长用户需设置部门，管理员可免分配</small>
                  </span>
                  <StatusBadge tone={unassignedRequiredUsers.length > 0 ? "warning" : "success"}>
                    {unassignedRequiredUsers.length} 人
                  </StatusBadge>
                </button>
                {filteredOverviewTree.length === 0 ? (
                  <div className="empty-state compact-empty-state">没有匹配的部门</div>
                ) : null}
                {filteredOverviewTree.map((node) => (
                  <OrgOverviewNode
                    key={node.department.id}
                    node={node}
                    expanded={overviewExpanded}
                    selectedId={overviewSelectedId}
                    statsByDepartment={statsByDepartment}
                    onSelect={setOverviewSelectedId}
                    onToggle={(id) =>
                      setOverviewExpanded((current) => {
                        const next = new Set(current);
                        if (next.has(id)) {
                          next.delete(id);
                        } else {
                          next.add(id);
                        }
                        return next;
                      })
                    }
                  />
                ))}
              </div>
            </section>

            <section className="panel stack org-detail-panel">
              <div className="panel-header">
                <div className="panel-heading">
                  <h3 className="department-title-with-code">
                    <span>{selectedOverviewDepartment?.name ?? "未设置部门"}</span>
                    {selectedOverviewDepartment ? (
                      <span className="dept-code-badge">{selectedOverviewDepartment.org_code}</span>
                    ) : null}
                  </h3>
                  <p>{selectedOverviewDepartment?.path ?? "普通/组长用户需要部门归属；管理员可以不设置部门。"}</p>
                </div>
                <StatusBadge tone={overviewSelectedId === UNASSIGNED_SELECTION ? "warning" : "success"}>
                  {overviewDistributionUsers.length} 人
                </StatusBadge>
              </div>

              <div className="org-detail-grid">
                <div>
                  <span>{overviewSelectedId === UNASSIGNED_SELECTION ? "需分配用户" : "直属用户"}</span>
                  <strong>{overviewDetailUsers.length}</strong>
                </div>
                <div>
                  <span>{overviewSelectedId === UNASSIGNED_SELECTION ? "待处理" : "子树用户"}</span>
                  <strong>{overviewDistributionUsers.length}</strong>
                </div>
                <div>
                  <span>启用</span>
                  <strong>{overviewDistributionUsers.filter((item) => item.is_active).length}</strong>
                </div>
                <div>
                  <span>停用</span>
                  <strong>{overviewDistributionUsers.filter((item) => !item.is_active).length}</strong>
                </div>
              </div>

              <div className="org-role-distribution">
                <div className="subsection-header">
                  <div className="panel-heading">
                    <h4>角色分布</h4>
                    <p>按当前节点子树统计。</p>
                  </div>
                </div>
                {ROLE_OPTIONS.map((option) => (
                  <div key={option.value} className="org-role-row">
                    <span>{option.label}</span>
                    <div className="org-role-track">
                      <span style={{ width: `${(overviewRoleCounts[option.value] / overviewMaxRoleCount) * 100}%` }} />
                    </div>
                    <strong>{overviewRoleCounts[option.value]}</strong>
                  </div>
                ))}
              </div>

              {selectedOverviewDepartment ? (
                <div className="org-child-summary">
                  <div className="subsection-header">
                    <div className="panel-heading">
                      <h4>直属子部门</h4>
                      <p>{selectedOverviewChildren.length} 个下级部门。</p>
                    </div>
                  </div>
                  <div className="org-child-list">
                    {selectedOverviewChildren.length === 0 ? <span className="muted">暂无直属子部门</span> : null}
                    {selectedOverviewChildren.map((department) => (
                      <button key={department.id} onClick={() => setOverviewSelectedId(department.id)} type="button">
                        <span>
                          <strong>{department.name}</strong>
                          <small>{department.path}</small>
                        </span>
                        <StatusBadge tone="neutral">
                          {statsByDepartment.get(department.id)?.subtreeUsers.length ?? 0} 人
                        </StatusBadge>
                      </button>
                    ))}
                  </div>
                </div>
              ) : null}

              <div className="org-user-list-section">
                <div className="subsection-header">
                  <div className="panel-heading">
                    <h4>{overviewSelectedId === UNASSIGNED_SELECTION ? "未分配用户" : "直属用户"}</h4>
                    <p>点击用户可进入编辑。</p>
                  </div>
                </div>
                <div className="org-user-list">
                  {overviewDetailUsers.length === 0 ? (
                    <div className="empty-state compact-empty-state">
                      {overviewSelectedId === UNASSIGNED_SELECTION ? "所有需分配用户都已设置部门" : "该部门暂无直属用户"}
                    </div>
                  ) : null}
                  {overviewDetailUsers.map((item) => (
                    <button key={item.id} onClick={() => startManageUser(item)} type="button">
                      <span>
                        <strong>{item.full_name}</strong>
                        <small>{item.email}</small>
                      </span>
                      <span className="org-user-badges">
                        <StatusBadge tone={item.role?.name === "admin" ? "warning" : "neutral"}>
                          {formatRoleName(item.role?.name)}
                        </StatusBadge>
                        <StatusBadge tone={item.is_active ? "success" : "danger"}>
                          {item.is_active ? "启用" : "停用"}
                        </StatusBadge>
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            </section>
          </div>
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
              enableDragMove
              emptyDescription="选择部门后查看详情"
              emptyLabel="未选择部门"
              selectedId={selectedManageDepartmentId}
              onMove={handleDepartmentMove}
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
