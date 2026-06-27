import { useEffect, useState } from "react";

import { CitationList } from "../components/CitationList";
import { ErrorNotice } from "../components/ErrorNotice";
import { PageHeader } from "../components/PageHeader";
import { StatusBadge } from "../components/StatusBadge";
import { useAppContext } from "../context/AppContext";
import { api } from "../lib/api";
import { formatConfidence, formatPriority, formatWorkflowStatus } from "../lib/display";
import { formatDateTime } from "../lib/format";
import type { FAQEntryRead, SourceCitationRead, TaskItemRead, WeeklyReportDraftRead } from "../types/api";

interface ArtifactsPageCache {
  tasks: TaskItemRead[];
  reports: WeeklyReportDraftRead[];
  faqs: FAQEntryRead[];
  sourceSessionId: string;
  selectedSources: SourceCitationRead[];
}

export function ArtifactsPage() {
  const { token, selectedSessionId, getPageCache, setPageCache } = useAppContext();
  const cachedPage = getPageCache<ArtifactsPageCache>("artifacts");
  const [tasks, setTasks] = useState<TaskItemRead[]>(() => cachedPage?.tasks ?? []);
  const [reports, setReports] = useState<WeeklyReportDraftRead[]>(() => cachedPage?.reports ?? []);
  const [faqs, setFaqs] = useState<FAQEntryRead[]>(() => cachedPage?.faqs ?? []);
  const [sourceSessionId, setSourceSessionId] = useState(cachedPage?.sourceSessionId ?? selectedSessionId ?? "");
  const [selectedSources, setSelectedSources] = useState<SourceCitationRead[]>(() => cachedPage?.selectedSources ?? []);
  const [error, setError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  useEffect(() => {
    if (selectedSessionId) {
      setSourceSessionId(selectedSessionId);
    }
  }, [selectedSessionId]);

  useEffect(() => {
    setPageCache<ArtifactsPageCache>("artifacts", {
      tasks,
      reports,
      faqs,
      sourceSessionId,
      selectedSources,
    });
  }, [faqs, reports, selectedSources, setPageCache, sourceSessionId, tasks]);

  useEffect(() => {
    if (!token) {
      return;
    }
    Promise.all([api.listTasks(token), api.listReports(token), api.listFaqs(token)])
      .then(([nextTasks, nextReports, nextFaqs]) => {
        setTasks(nextTasks);
        setReports(nextReports);
        setFaqs(nextFaqs);
      })
      .catch((nextError) => setError(nextError instanceof Error ? nextError.message : "加载派生结果失败。"));
  }, [token]);

  function bindFirstSources(citations: SourceCitationRead[] | null | undefined) {
    setSelectedSources(citations ?? []);
  }

  async function handleGenerate(kind: "tasks" | "report" | "faq") {
    if (!token || !sourceSessionId.trim()) {
      setStatusMessage("请先提供会话 ID，或先在问答页选中一个会话。");
      return;
    }
    setStatusMessage(null);
    try {
      if (kind === "tasks") {
        const response = await api.extractTasks(token, sourceSessionId.trim());
        setTasks(await api.listTasks(token));
        bindFirstSources(response.items[0]?.source_citations ?? []);
        setStatusMessage(`已生成 ${response.items.length} 条待办。`);
      } else if (kind === "report") {
        const response = await api.generateWeeklyReport(token, sourceSessionId.trim(), "周报草稿");
        setReports(await api.listReports(token));
        bindFirstSources(response.report.reference_sources);
        setStatusMessage(`已生成周报草稿：${response.report.title}`);
      } else {
        const response = await api.generateFaqs(token, sourceSessionId.trim());
        setFaqs(await api.listFaqs(token));
        bindFirstSources(response.entries[0]?.source_citations ?? []);
        setStatusMessage(`已生成 ${response.entries.length} 条 FAQ 草稿。`);
      }
    } catch (nextError) {
      setStatusMessage(nextError instanceof Error ? nextError.message : "生成失败，请稍后重试。");
    }
  }

  return (
    <div className="page-stack">
      <PageHeader
        title="派生结果"
        description="把有引用依据的问答会话转为待办、周报草稿和 FAQ 草稿，并保留来源追溯。"
      />
      <ErrorNotice message={error} />
      {statusMessage ? <div className="info-block">{statusMessage}</div> : null}

      <section className="panel stack">
        <div className="panel-header">
          <h3>从会话生成</h3>
          <StatusBadge tone="info">仅展示结构化结果</StatusBadge>
        </div>
        <label>
          <span>来源会话 ID</span>
          <input
            placeholder="请输入问答页中的会话 ID"
            value={sourceSessionId}
            onChange={(event) => setSourceSessionId(event.target.value)}
          />
        </label>
        <div className="inline-actions">
          <button className="secondary-button" onClick={() => handleGenerate("tasks")} type="button">
            提取待办
          </button>
          <button className="secondary-button" onClick={() => handleGenerate("report")} type="button">
            生成周报草稿
          </button>
          <button className="secondary-button" onClick={() => handleGenerate("faq")} type="button">
            生成 FAQ 草稿
          </button>
        </div>
      </section>

      <div className="page-grid artifacts-layout">
        <section className="panel stack">
          <div className="panel-header">
            <h3>待办事项</h3>
            <StatusBadge tone="warning">{tasks.length}</StatusBadge>
          </div>
          <div className="stack dense-stack">
            {tasks.length ? (
              tasks.map((task) => (
                <button className="list-card text-left" key={task.id} onClick={() => bindFirstSources(task.source_citations)} type="button">
                  <div className="list-card-topline">
                    <strong>{task.title}</strong>
                    <StatusBadge tone="info">{formatPriority(task.priority)}</StatusBadge>
                  </div>
                  <p>{task.description ?? "暂无描述"}</p>
                  <p className="muted">{formatDateTime(task.created_at)}</p>
                </button>
              ))
            ) : (
              <p className="muted">暂无待办，可从一条引用式问答中提取。</p>
            )}
          </div>
        </section>

        <section className="panel stack">
          <div className="panel-header">
            <h3>周报草稿</h3>
            <StatusBadge tone="success">{reports.length}</StatusBadge>
          </div>
          <div className="stack dense-stack">
            {reports.length ? (
              reports.map((report) => (
                <button className="list-card text-left" key={report.id} onClick={() => bindFirstSources(report.reference_sources)} type="button">
                  <div className="list-card-topline">
                    <strong>{report.title}</strong>
                    <StatusBadge tone="neutral">{formatWorkflowStatus(report.status)}</StatusBadge>
                  </div>
                  <p>{report.summary ?? "暂无摘要"}</p>
                  <div className="report-list-grid">
                    <span>本周完成：{report.completed_this_week.length}</span>
                    <span>风险阻塞：{report.risks_blockers.length}</span>
                    <span>下周计划：{report.next_week_plan.length}</span>
                  </div>
                </button>
              ))
            ) : (
              <p className="muted">暂无周报草稿，可从最近一次会话生成。</p>
            )}
          </div>
        </section>

        <section className="panel stack">
          <div className="panel-header">
            <h3>FAQ 草稿</h3>
            <StatusBadge tone="neutral">{faqs.length}</StatusBadge>
          </div>
          <div className="stack dense-stack">
            {faqs.length ? (
              faqs.map((faq) => (
                <button className="list-card text-left" key={faq.id} onClick={() => bindFirstSources(faq.source_citations)} type="button">
                  <div className="list-card-topline">
                    <strong>{faq.question}</strong>
                    <StatusBadge tone={faq.quality === "high" ? "success" : "warning"}>{formatConfidence(faq.quality)}</StatusBadge>
                  </div>
                  <p>{faq.answer}</p>
                  <p className="muted">状态：{formatWorkflowStatus(faq.status)}</p>
                </button>
              ))
            ) : (
              <p className="muted">暂无 FAQ 草稿，可从高质量问答中沉淀。</p>
            )}
          </div>
        </section>
      </div>

      <CitationList citations={selectedSources} title="结果引用来源" />
    </div>
  );
}



