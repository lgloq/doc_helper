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

  return (
    <div className="app-shell">
      <aside className="app-sidebar">
        <div>
          <div className="brand-mark">EKA</div>
          <h2>企业知识助手</h2>
          <p className="muted">权限感知问答、派生结果、版本差异、评测与追踪。</p>
        </div>
        <nav className="sidebar-nav">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => `nav-link ${isActive ? "is-active" : ""}`}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className="user-card">
            <div>
              <strong>{user?.full_name ?? "未知用户"}</strong>
              <p>{user?.email}</p>
            </div>
            <StatusBadge tone={user?.role?.name === "admin" ? "warning" : "neutral"}>
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

