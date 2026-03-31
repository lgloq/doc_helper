import { FormEvent, useEffect, useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";

import { ErrorNotice } from "../components/ErrorNotice";
import { useAppContext } from "../context/AppContext";

const demoAccounts = [
  { label: "普通员工", email: "viewer@local.test", password: "viewer123" },
  { label: "组长", email: "manager@local.test", password: "manager123" },
  { label: "管理员", email: "admin@local.test", password: "admin123" },
];

export function LoginPage() {
  const { token, login } = useAppContext();
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState(demoAccounts[2].email);
  const [password, setPassword] = useState(demoAccounts[2].password);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    document.title = "登录 · 权限感知 RAG 文档知识助手";
  }, []);

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
      setError(nextError instanceof Error ? nextError.message : "登录失败，请检查账号或密码。");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="login-page">
      <div className="login-card">
        <div className="login-copy">
          <span className="eyebrow">企业文档知识场景演示</span>
          <h1>权限感知的 RAG 企业文档知识助手</h1>
          <p>面向企业知识库的权限感知 RAG 应用，支持多角色访问控制、引用溯源、版本对比与结构化工作流生成。</p>
        </div>
        <form className="stack" onSubmit={handleSubmit}>
          <label>
            <span>邮箱</span>
            <input
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              type="email"
              placeholder="请输入登录邮箱"
              required
            />
          </label>
          <label>
            <span>密码</span>
            <input
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              type="password"
              placeholder="请输入密码"
              required
            />
          </label>
          <ErrorNotice message={error} />
          <button className="primary-button" disabled={submitting} type="submit">
            {submitting ? "登录中..." : "登录"}
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
              使用{account.label}账号
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}


