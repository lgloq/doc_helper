from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.user import User
from app.schemas.search import SearchDebugInfo
from app.services.permissions.service import PermissionFilterBuilder

PIPELINE_STAGE_LABELS = {
    "passed": "通过",
    "permission_filter": "权限过滤",
    "candidate_recall": "候选召回",
    "candidate_selection": "候选选择",
    "citation_coverage": "引用覆盖",
    "answer_generation": "答案生成",
}

PIPELINE_REASON_LABELS = {
    "passed": "未发现明显问题",
    "unauthorized_content_leaked": "出现越权内容泄漏",
    "expected_documents_not_accessible": "期望文档不在当前用户可见范围",
    "permission_probe_blocked_target": "权限探测判定目标文档不可见",
    "no_candidates_recalled": "没有召回到候选内容",
    "expected_documents_not_recalled": "期望文档未进入召回结果",
    "expected_facts_not_recalled": "期望事实未进入召回结果",
    "expected_evidence_not_selected": "召回到了相关候选，但最终没有选中期望证据",
    "selected_citations_missing_required_facts": "已选引用未覆盖所需事实",
    "selected_citations_do_not_support_answer": "已选引用不足以支撑答案",
    "answered_instead_of_refusing": "应该拒答但模型仍给出回答",
    "refused_despite_available_evidence": "已有足够证据但模型仍拒答",
    "unsupported_answer_claims": "答案包含未被证据支撑的事实",
    "answer_did_not_follow_evidence": "答案没有按证据作答",
    "overall_score_below_threshold": "综合评分未达到阈值",
    "generation_failed": "生成阶段报错",
    "no_citations_selected": "生成前未选出引用证据",
    "insufficient_evidence_after_selection": "已选引用后仍证据不足",
}


def build_eval_pipeline_diagnosis(
    *,
    session: Session,
    actor: User,
    expected_titles: list[str],
    expected_outcome: str,
    overall_pass: bool,
    retrieval_breakdown: dict[str, Any],
    citation_breakdown: dict[str, Any],
    faithfulness_breakdown: dict[str, Any],
    permission_breakdown: dict[str, Any],
    permission_checks: dict[str, Any],
    retrieval_debug: SearchDebugInfo,
    matched_expected_titles: list[str],
    missing_expected_titles: list[str],
    matched_citation_titles: list[str],
    missing_citation_titles: list[str],
    unsupported_answer_facts: list[str],
    unsupported_answer_claims: list[str],
    insufficient_evidence: bool,
) -> dict[str, Any]:
    accessible_expected_titles, inaccessible_expected_titles = _split_expected_titles_by_access(
        session,
        actor,
        expected_titles,
    )
    retrieval_score = _score(retrieval_breakdown)
    citation_score = _score(citation_breakdown)
    faithfulness_score = _score(faithfulness_breakdown)
    evidence_fact_recall = _safe_float(citation_breakdown.get("evidence_fact_recall"), default=0.0)
    retrieved_fact_recall = _safe_float(retrieval_breakdown.get("retrieved_fact_recall"), default=0.0)
    claim_support_score = _safe_float(faithfulness_breakdown.get("claim_support_score"), default=faithfulness_score)
    forbidden_in_retrieval = _string_list(permission_checks.get("forbidden_in_retrieval"))
    forbidden_in_citations = _string_list(permission_checks.get("forbidden_in_citations"))
    forbidden_in_answer = _string_list(permission_checks.get("forbidden_in_answer"))
    leak_detected = (
        not bool(permission_breakdown.get("passed", True))
        or bool(forbidden_in_retrieval)
        or bool(forbidden_in_citations)
        or bool(forbidden_in_answer)
    )

    if overall_pass:
        return _diagnosis(
            status="passed",
            stage="passed",
            reason_code="passed",
            summary="当前样例没有检测到明显的检索或生成失败信号。",
            signals={
                "retrieval_score": retrieval_score,
                "citation_score": citation_score,
                "faithfulness_score": faithfulness_score,
            },
        )

    if leak_detected:
        return _diagnosis(
            status="failed",
            stage="permission_filter",
            reason_code="unauthorized_content_leaked",
            summary="受限文档或受限事实出现在召回、引用或答案中。",
            signals={
                "forbidden_in_retrieval": forbidden_in_retrieval,
                "forbidden_in_citations": forbidden_in_citations,
                "forbidden_in_answer": forbidden_in_answer,
            },
        )

    if expected_outcome == "refuse":
        if not insufficient_evidence:
            return _diagnosis(
                status="failed",
                stage="answer_generation",
                reason_code="answered_instead_of_refusing",
                summary="该样例预期拒答，但模型仍生成了回答。",
                signals={"expected_outcome": expected_outcome},
            )
        return _diagnosis(
            status="failed",
            stage="answer_generation",
            reason_code="overall_score_below_threshold",
            summary="该拒答样例未通过，但未检测到更具体的权限泄漏信号。",
            signals={"expected_outcome": expected_outcome},
        )

    if retrieval_debug.permission_probe_early_stop_applied and retrieval_debug.permission_probe_inaccessible_target_count > 0:
        return _diagnosis(
            status="failed",
            stage="permission_filter",
            reason_code="permission_probe_blocked_target",
            summary="查询命中了权限探测早停，目标文档在当前用户可见范围外。",
            signals={
                "permission_probe_target_hint": retrieval_debug.permission_probe_target_hint,
                "inaccessible_target_count": retrieval_debug.permission_probe_inaccessible_target_count,
            },
        )

    if inaccessible_expected_titles and not accessible_expected_titles:
        return _diagnosis(
            status="failed",
            stage="permission_filter",
            reason_code="expected_documents_not_accessible",
            summary="样例期望文档不在当前执行用户的可访问范围内。",
            signals={
                "accessible_expected_titles": accessible_expected_titles,
                "inaccessible_expected_titles": inaccessible_expected_titles,
            },
        )

    if retrieval_score < 0.5 or (missing_expected_titles and not matched_expected_titles):
        if _safe_int(retrieval_debug.pre_rerank_count) <= 0 and _safe_int(retrieval_debug.post_rerank_count) <= 0:
            reason_code = "no_candidates_recalled"
            summary = "检索链路没有召回到候选内容。"
        elif retrieved_fact_recall <= 0.0:
            reason_code = "expected_facts_not_recalled"
            summary = "召回结果没有覆盖样例要求的关键事实。"
        else:
            reason_code = "expected_documents_not_recalled"
            summary = "样例期望文档没有进入召回结果。"
        return _diagnosis(
            status="failed",
            stage="candidate_recall",
            reason_code=reason_code,
            summary=summary,
            signals={
                "matched_expected_titles": matched_expected_titles,
                "missing_expected_titles": missing_expected_titles,
                "retrieved_fact_recall": retrieved_fact_recall,
                "pre_rerank_count": retrieval_debug.pre_rerank_count,
                "post_rerank_count": retrieval_debug.post_rerank_count,
            },
        )

    if citation_score < 0.5:
        if matched_citation_titles or evidence_fact_recall > 0.0:
            return _diagnosis(
                status="failed",
                stage="citation_coverage",
                reason_code="selected_citations_missing_required_facts",
                summary="已经选中了部分引用，但引用内容仍未覆盖样例要求的关键事实。",
                signals={
                    "matched_citation_titles": matched_citation_titles,
                    "missing_citation_titles": missing_citation_titles,
                    "evidence_fact_recall": evidence_fact_recall,
                },
            )
        return _diagnosis(
            status="failed",
            stage="candidate_selection",
            reason_code="expected_evidence_not_selected",
            summary="召回阶段已有相关文档，但最终没有选中期望证据作为引用。",
            signals={
                "matched_expected_titles": matched_expected_titles,
                "matched_citation_titles": matched_citation_titles,
                "missing_citation_titles": missing_citation_titles,
            },
        )

    if faithfulness_score < 0.7:
        if unsupported_answer_claims:
            return _diagnosis(
                status="failed",
                stage="answer_generation",
                reason_code="unsupported_answer_claims",
                summary="答案中包含未被已选证据支撑的事实性表述。",
                signals={
                    "unsupported_answer_claims": unsupported_answer_claims,
                    "claim_support_score": claim_support_score,
                },
            )
        if insufficient_evidence:
            return _diagnosis(
                status="failed",
                stage="answer_generation",
                reason_code="refused_despite_available_evidence",
                summary="引用和检索结果已有一定覆盖，但模型仍以证据不足作答。",
                signals={
                    "citation_score": citation_score,
                    "evidence_fact_recall": evidence_fact_recall,
                },
            )
        if unsupported_answer_facts or evidence_fact_recall < 1.0:
            return _diagnosis(
                status="failed",
                stage="citation_coverage",
                reason_code="selected_citations_do_not_support_answer",
                summary="已选引用不足以完整支撑答案中的关键事实。",
                signals={
                    "unsupported_answer_facts": unsupported_answer_facts,
                    "evidence_fact_recall": evidence_fact_recall,
                },
            )
        return _diagnosis(
            status="failed",
            stage="answer_generation",
            reason_code="answer_did_not_follow_evidence",
            summary="检索和引用基本到位，但最终答案没有按证据正确表达。",
            signals={"faithfulness_score": faithfulness_score},
        )

    return _diagnosis(
        status="failed",
        stage="answer_generation",
        reason_code="overall_score_below_threshold",
        summary="样例未通过，但没有命中更具体的失败归因规则。",
        signals={
            "retrieval_score": retrieval_score,
            "citation_score": citation_score,
            "faithfulness_score": faithfulness_score,
        },
    )


def build_trace_pipeline_diagnosis(
    *,
    retrieval_debug: SearchDebugInfo,
    selected_citation_count: int,
    evidence_audit: dict[str, Any] | None,
    error_text: str | None,
    insufficient_evidence: bool | None,
) -> dict[str, Any]:
    if error_text:
        return _diagnosis(
            status="failed",
            stage="answer_generation",
            reason_code="generation_failed",
            summary="生成阶段返回了错误，未形成稳定答案。",
            signals={"error_text": error_text[:300]},
        )
    if retrieval_debug.permission_probe_early_stop_applied or retrieval_debug.accessible_document_count <= 0:
        reason_code = (
            "permission_probe_blocked_target"
            if retrieval_debug.permission_probe_early_stop_applied
            else "expected_documents_not_accessible"
        )
        summary = (
            "查询在权限探测阶段被提前拦截。"
            if retrieval_debug.permission_probe_early_stop_applied
            else "当前用户没有可访问文档或目标文档不在可见范围内。"
        )
        return _diagnosis(
            status="warning",
            stage="permission_filter",
            reason_code=reason_code,
            summary=summary,
            signals={
                "accessible_document_count": retrieval_debug.accessible_document_count,
                "permission_probe_target_hint": retrieval_debug.permission_probe_target_hint,
            },
        )
    if _safe_int(retrieval_debug.pre_rerank_count) <= 0 and _safe_int(retrieval_debug.post_rerank_count) <= 0:
        return _diagnosis(
            status="warning",
            stage="candidate_recall",
            reason_code="no_candidates_recalled",
            summary="检索链路没有召回到候选内容。",
            signals={
                "lexical_candidate_count": retrieval_debug.lexical_candidate_count,
                "vector_candidate_count": retrieval_debug.vector_candidate_count,
            },
        )
    if selected_citation_count <= 0:
        return _diagnosis(
            status="warning",
            stage="candidate_selection",
            reason_code="no_citations_selected",
            summary="检索返回了候选，但在生成前没有选出最终引用证据。",
            signals={
                "pre_rerank_count": retrieval_debug.pre_rerank_count,
                "post_rerank_count": retrieval_debug.post_rerank_count,
            },
        )
    unsupported_claim_count = _safe_int((evidence_audit or {}).get("unsupported_count"))
    if unsupported_claim_count > 0:
        return _diagnosis(
            status="warning",
            stage="answer_generation",
            reason_code="unsupported_answer_claims",
            summary="答案存在未被已选引用支撑的事实，建议人工复核。",
            signals={
                "unsupported_claim_count": unsupported_claim_count,
                "evidence_audit_status": (evidence_audit or {}).get("status"),
            },
        )
    if insufficient_evidence:
        return _diagnosis(
            status="warning",
            stage="citation_coverage",
            reason_code="insufficient_evidence_after_selection",
            summary="已经选出引用，但当前证据仍不足以稳定回答问题。",
            signals={"selected_citation_count": selected_citation_count},
        )
    return _diagnosis(
        status="passed",
        stage="passed",
        reason_code="passed",
        summary="当前追踪没有暴露明显的阶段性异常。",
        signals={"selected_citation_count": selected_citation_count},
    )


def _split_expected_titles_by_access(session: Session, actor: User, expected_titles: list[str]) -> tuple[list[str], list[str]]:
    normalized_titles = [item.strip() for item in expected_titles if isinstance(item, str) and item.strip()]
    if not normalized_titles:
        return [], []
    accessible_ids = set(PermissionFilterBuilder().resolve_accessible_document_ids(session, actor, require_manage=False))
    documents = list(session.scalars(select(Document).where(Document.title.in_(normalized_titles))).all())
    accessible_titles = sorted({item.title for item in documents if item.id in accessible_ids})
    accessible_title_keys = {_normalize(item) for item in accessible_titles}
    inaccessible_titles = sorted(
        {
            item
            for item in normalized_titles
            if _normalize(item) not in accessible_title_keys
        }
    )
    return accessible_titles, inaccessible_titles


def _diagnosis(
    *,
    status: str,
    stage: str,
    reason_code: str,
    summary: str,
    signals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "stage": stage,
        "stage_label": PIPELINE_STAGE_LABELS.get(stage, stage),
        "reason_code": reason_code,
        "reason_label": PIPELINE_REASON_LABELS.get(reason_code, reason_code),
        "summary": summary,
        "signals": signals or {},
    }


def _score(payload: dict[str, Any] | None) -> float:
    value = payload.get("score") if isinstance(payload, dict) else None
    return _safe_float(value, default=0.0)


def _safe_float(value: Any, *, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return 0


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def _normalize(value: str) -> str:
    return " ".join(value.split()).casefold()
