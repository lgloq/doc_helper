import { useState } from "react";

import { StatusBadge } from "./StatusBadge";
import type { AnswerEvidenceAuditRead, AnswerSupportCitationRead } from "../types/api";

interface FactEvidencePanelProps {
  audit: AnswerEvidenceAuditRead;
  title?: string;
  onSelectCitation?: (support: AnswerSupportCitationRead) => void;
}

export function FactEvidencePanel({
  audit,
  title = "事实级证据",
  onSelectCitation,
}: FactEvidencePanelProps) {
  const [expanded, setExpanded] = useState(false);
  const coverageCount = audit.supported_count + audit.partial_count;
  const coverageRatio = audit.claim_count ? coverageCount / audit.claim_count : null;
  const visibleClaims = expanded ? audit.claims : [];
  const claimCount = audit.claims.length;

  return (
    <div className="fact-evidence-panel">
      <div className="fact-evidence-header">
        <div>
          <strong>{title}</strong>
          <p className="muted">
            覆盖 {coverageCount}/{audit.claim_count}
            {coverageRatio !== null ? ` · ${percentText(coverageRatio)}` : ""}
          </p>
        </div>
        <StatusBadge tone={auditTone(audit.status)}>{auditLabel(audit.status)}</StatusBadge>
      </div>

      <div className="fact-evidence-metrics">
        <span>完全支撑 {audit.supported_count}</span>
        <span>部分支撑 {audit.partial_count}</span>
        <span>待核实 {audit.unsupported_count}</span>
        <span>综合 {scoreText(audit.score)}</span>
      </div>

      {expanded ? (
        <div className="fact-evidence-list">
          {visibleClaims.map((claim) => (
            <div className={`fact-evidence-item tone-${claimTone(claim.support_status)}`} key={`${claim.index}-${claim.text}`}>
              <div className="fact-evidence-item-topline">
                <span>事实 {claim.index}</span>
                <StatusBadge tone={claimTone(claim.support_status)}>
                  {claimLabel(claim.support_status)} · {scoreText(claim.support_score)}
                </StatusBadge>
              </div>
              <p className="fact-evidence-claim-text">{claim.text}</p>
              {claim.support_citations.length ? (
                <div className="fact-evidence-support-list">
                  {claim.support_citations.map((support, index) => (
                    <div className="fact-evidence-support" key={`${claim.index}-${support.rank}-${support.chunk_id ?? index}`}>
                      <div className="fact-evidence-support-topline">
                        <strong>{support.document_title ?? `引用 ${support.rank}`}</strong>
                        <span>
                          引用 {support.rank}
                          {support.location ? ` · ${support.location}` : ""}
                        </span>
                      </div>
                      <p className="fact-evidence-support-excerpt">
                        {support.evidence_excerpt?.trim() ? support.evidence_excerpt : "当前没有可展示的支撑片段。"}
                      </p>
                      {onSelectCitation ? (
                        <button className="fact-evidence-support-action" onClick={() => onSelectCitation(support)} type="button">
                          定位到引用
                        </button>
                      ) : null}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="fact-evidence-unsupported-note">未找到可支撑这条事实的已选证据片段。</p>
              )}
            </div>
          ))}
        </div>
      ) : null}

      {claimCount ? (
        <button className="fact-evidence-toggle" aria-expanded={expanded} onClick={() => setExpanded((current) => !current)} type="button">
          {expanded ? "收起事实明细" : `展开 ${claimCount} 条事实明细`}
        </button>
      ) : null}
    </div>
  );
}

function auditTone(status: string): "neutral" | "success" | "warning" | "danger" | "info" {
  if (status === "supported") {
    return "success";
  }
  if (status === "partial") {
    return "warning";
  }
  if (status === "needs_review") {
    return "danger";
  }
  return "neutral";
}

function auditLabel(status: string): string {
  if (status === "supported") {
    return "整体已支撑";
  }
  if (status === "partial") {
    return "整体部分支撑";
  }
  if (status === "needs_review") {
    return "存在待核实事实";
  }
  return "未抽取";
}

function claimTone(status: string): "neutral" | "success" | "warning" | "danger" | "info" {
  if (status === "supported") {
    return "success";
  }
  if (status === "partial") {
    return "warning";
  }
  return "danger";
}

function claimLabel(status: string): string {
  if (status === "supported") {
    return "完全支撑";
  }
  if (status === "partial") {
    return "部分支撑";
  }
  return "待核实";
}

function scoreText(value: number | null): string {
  if (typeof value !== "number") {
    return "-";
  }
  return `${Math.round(value * 100)}%`;
}

function percentText(value: number): string {
  return `${Math.round(value * 100)}%`;
}
