import { FormEvent, useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";

import { ErrorNotice } from "../components/ErrorNotice";
import { useAppContext } from "../context/AppContext";

const demoAccounts = [
  { label: "Viewer", email: "viewer@local.test", password: "viewer123" },
  { label: "Manager", email: "manager@local.test", password: "manager123" },
  { label: "Admin", email: "admin@local.test", password: "admin123" },
];

export function LoginPage() {
  const { token, login } = useAppContext();
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState(demoAccounts[2].email);
  const [password, setPassword] = useState(demoAccounts[2].password);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  if (token) {
    return <Navigate to="/documents" replace />;
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await login(email, password);
      const nextPath = (location.state as { from?: { pathname?: string } } | null)?.from?.pathname ?? "/documents";
      navigate(nextPath, { replace: true });
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Login failed.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="login-page">
      <div className="login-card">
        <div className="login-copy">
          <span className="eyebrow">Phase 9 Frontend</span>
          <h1>Enterprise Knowledge Assistant</h1>
          <p>
            A minimal operator console for permission-aware retrieval, grounded QA, workflow artifacts, document
            version diff, and eval visibility.
          </p>
        </div>
        <form className="stack" onSubmit={handleSubmit}>
          <label>
            <span>Email</span>
            <input value={email} onChange={(event) => setEmail(event.target.value)} type="email" required />
          </label>
          <label>
            <span>Password</span>
            <input value={password} onChange={(event) => setPassword(event.target.value)} type="password" required />
          </label>
          <ErrorNotice message={error} />
          <button className="primary-button" disabled={submitting} type="submit">
            {submitting ? "Signing in..." : "Sign in"}
          </button>
        </form>
        <div className="demo-account-grid">
          {demoAccounts.map((account) => (
            <button
              key={account.label}
              className="secondary-button"
              onClick={() => {
                setEmail(account.email);
                setPassword(account.password);
              }}
              type="button"
            >
              Use {account.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
