import { NavLink, Outlet } from "react-router-dom";

import { useAppContext } from "../context/AppContext";
import { StatusBadge } from "./StatusBadge";

const navItems = [
  { to: "/documents", label: "Documents" },
  { to: "/chat", label: "Chat" },
  { to: "/artifacts", label: "Artifacts" },
  { to: "/versions", label: "Versions" },
  { to: "/insights", label: "Eval & Trace" },
];

export function AppShell() {
  const { user, logout } = useAppContext();

  return (
    <div className="app-shell">
      <aside className="app-sidebar">
        <div>
          <div className="brand-mark">EKA</div>
          <h2>Enterprise Knowledge Assistant</h2>
          <p className="muted">Permission-aware QA, task workflows, version diff, eval and traces.</p>
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
              <strong>{user?.full_name ?? "Unknown user"}</strong>
              <p>{user?.email}</p>
            </div>
            <StatusBadge tone={user?.role?.name === "admin" ? "warning" : "neutral"}>
              {user?.role?.name ?? "guest"}
            </StatusBadge>
          </div>
          <button className="secondary-button" onClick={logout} type="button">
            Logout
          </button>
        </div>
      </aside>
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  );
}
