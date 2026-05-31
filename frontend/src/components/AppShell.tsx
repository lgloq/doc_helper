import { NavLink, Outlet } from "react-router-dom";

import { useAppContext } from "../context/AppContext";
import { formatRoleName } from "../lib/display";
import { StatusBadge } from "./StatusBadge";

const navItems = [
  { to: "/documents", label: "文档" },
  { to: "/chat", label: "问答" },
  { to: "/artifacts", label: "派生结果" },
  { to: "/versions", label: "版本对比" },
  { to: "/insights", label: "评测与追踪" },
];

export function AppShell() {
  const { user, logout } = useAppContext();
  const isAdmin = user?.role?.name === "admin";
  const departmentLabel = user?.department?.path ?? (isAdmin ? "管理员免分配" : "未设置");

  const visibleNavItems = isAdmin ? [...navItems, { to: "/users", label: "用户与部门管理" }] : navItems;

  return (
    <div className="app-shell">
      <aside className="app-sidebar">
        <div className="sidebar-top">
          <div>
            <div className="brand-mark">RAG</div>
            <h2>权限感知 RAG 文档知识助手</h2>
            <p className="muted">面向企业知识库的权限感知检索、引用溯源、版本对比与结构化工作流生成。</p>
          </div>
          <nav className="sidebar-nav">
            {visibleNavItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) => `nav-link ${isActive ? "is-active" : ""}`}
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
        <div className="sidebar-footer">
          <div className="user-card">
            <div>
              <strong>{user?.full_name ?? "未知用户"}</strong>
              <p>{user?.email}</p>
              <p className="muted">部门：{departmentLabel}</p>
            </div>
            <StatusBadge tone={isAdmin ? "warning" : "neutral"}>
              {formatRoleName(user?.role?.name, "访客")}
            </StatusBadge>
          </div>
          <button className="secondary-button" onClick={logout} type="button">
            退出登录
          </button>
        </div>
      </aside>
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  );
}


