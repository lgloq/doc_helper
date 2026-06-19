from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT_DIR / "backend" / "data" / "benchmark_raw" / "zh_enterprise" / "v1_case_manifest_v1.json"
DEFAULT_OUTPUT = ROOT_DIR / "backend" / "data" / "eval_outputs" / "zh-enterprise-v1-case-quality-audit-local.json"
DEFAULT_MARKDOWN_OUTPUT = ROOT_DIR / "backend" / "data" / "eval_outputs" / "zh-enterprise-v1-case-quality-audit-local.md"
DEFAULT_ANCHOR_OCCURRENCE_THRESHOLD = 5

LOW_SIGNAL_TOKENS = {
    "公司",
    "有限",
    "有限公司",
    "集团",
    "材料",
    "文件",
    "相关",
    "事项",
    "原文",
    "依据",
    "需要",
    "确认",
    "处理",
    "口径",
    "请指",
    "指出",
    "具体",
    "怎么",
    "如何",
    "投研",
    "团队",
    "底稿",
    "业务",
    "同事",
    "同时",
    "分别",
    "引用",
    "中的",
    "两个",
    "核对",
    "披露",
}

TABLE_CASE_TYPES = {"table_structured", "table_numeric", "structured_evidence", "table"}
PERMISSION_CASE_TYPES = {"permission"}
LOW_OVERLAP_SCENARIO_CASE_TYPE = "low_overlap_enterprise_scenario"
ANCHOR_RELIANT_CASE_TYPES = {
    "single_fact",
    "version_temporal",
    "multi_evidence_same_document",
    "multi_evidence_cross_document",
    *TABLE_CASE_TYPES,
}
STRICT_EXCLUDED_METRIC_GROUPS = {"broad_document_discovery", "anchor_quality_review"}


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = audit_manifest_case_quality(
        manifest,
        manifest_path=manifest_path,
        weak_overlap_threshold=args.weak_overlap_threshold,
        very_weak_overlap_threshold=args.very_weak_overlap_threshold,
        anchor_occurrence_threshold=args.anchor_occurrence_threshold,
        marker_presence_issues=load_marker_presence_issues(args.root_cause_report),
        document_texts=load_document_texts(args.document_text_json),
    )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown_output:
        markdown_path = Path(args.markdown_output).resolve()
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
    if args.annotated_manifest_output:
        annotated_path = Path(args.annotated_manifest_output).resolve()
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        annotated_path.write_text(json.dumps(build_annotated_manifest(report, manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    if args.strict_manifest_output:
        strict_path = Path(args.strict_manifest_output).resolve()
        strict_path.parent.mkdir(parents=True, exist_ok=True)
        strict_path.write_text(json.dumps(build_strict_manifest(report, manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_text(report))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit benchmark case question/evidence alignment and metric suitability.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    parser.add_argument("--annotated-manifest-output", help="Optional manifest copy with per-case quality_audit metadata.")
    parser.add_argument(
        "--root-cause-report",
        action="append",
        default=None,
        help=(
            "Optional retrieval root-cause audit JSON. Cases with markers classified as "
            "marker_not_found_in_expected_document_chunks are excluded from strict evidence metrics."
        ),
    )
    parser.add_argument(
        "--strict-manifest-output",
        help="Optional manifest copy excluding broad-document-discovery cases from strict evidence evaluation.",
    )
    parser.add_argument("--weak-overlap-threshold", type=float, default=0.20)
    parser.add_argument("--very-weak-overlap-threshold", type=float, default=0.10)
    parser.add_argument(
        "--document-text-json",
        help=(
            "Optional JSON mapping document id/title to extracted text, or {'documents': [{'id','title','text'}]}. "
            "Used only to count repeated low-information anchors inside their expected documents."
        ),
    )
    parser.add_argument("--anchor-occurrence-threshold", type=int, default=DEFAULT_ANCHOR_OCCURRENCE_THRESHOLD)
    return parser


def audit_manifest_case_quality(
    manifest: dict[str, Any],
    *,
    manifest_path: Path | None = None,
    weak_overlap_threshold: float = 0.20,
    very_weak_overlap_threshold: float = 0.10,
    anchor_occurrence_threshold: int = DEFAULT_ANCHOR_OCCURRENCE_THRESHOLD,
    marker_presence_issues: dict[str, dict[str, Any]] | None = None,
    document_texts: dict[str, str] | None = None,
) -> dict[str, Any]:
    marker_presence_issues = marker_presence_issues or {}
    document_texts = document_texts or {}
    case_reports = [
        audit_case_quality(
            case,
            weak_overlap_threshold=weak_overlap_threshold,
            very_weak_overlap_threshold=very_weak_overlap_threshold,
            anchor_occurrence_threshold=anchor_occurrence_threshold,
            marker_presence_issue=marker_presence_issues.get(str(case.get("case_name"))),
            document_texts=document_texts,
        )
        for case in manifest.get("cases", [])
    ]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in case_reports:
        by_type[item["case_type"]].append(item)

    return {
        "manifest_path": str(manifest_path) if manifest_path else None,
        "dataset_name": manifest.get("dataset_name"),
        "case_count": len(case_reports),
        "thresholds": {
            "weak_overlap_threshold": weak_overlap_threshold,
            "very_weak_overlap_threshold": very_weak_overlap_threshold,
            "anchor_occurrence_threshold": anchor_occurrence_threshold,
        },
        "marker_presence_issue_source_count": len(marker_presence_issues),
        "summary": summarize_cases(case_reports),
        "case_type_summary": {
            case_type: summarize_cases(items)
            for case_type, items in sorted(by_type.items())
        },
        "weak_cases": [
            item
            for item in case_reports
            if "weak_question_evidence_overlap" in item["flags"]
            or "broad_question_with_exact_gold" in item["flags"]
            or "expected_marker_not_found_in_chunks" in item["flags"]
            or "unreliable_question_anchor" in item["flags"]
            or "under_specified_question_anchor" in item["flags"]
        ],
        "cases": case_reports,
    }


def audit_case_quality(
    case: dict[str, Any],
    *,
    weak_overlap_threshold: float,
    very_weak_overlap_threshold: float,
    anchor_occurrence_threshold: int,
    marker_presence_issue: dict[str, Any] | None = None,
    document_texts: dict[str, str] | None = None,
) -> dict[str, Any]:
    case_type = str((case.get("metadata") or {}).get("case_type") or "unknown")
    expected_outcome = str(case.get("expected_outcome") or "answer")
    question = str(case.get("question") or "")
    evidence_text = expected_evidence_text(case)
    overlap = question_evidence_overlap(question, evidence_text)
    anchor_report = audit_question_anchor_quality(
        question,
        case=case,
        document_texts=document_texts,
        anchor_occurrence_threshold=anchor_occurrence_threshold,
    )
    flags: list[str] = []
    metric_group = "strict_exact_evidence"

    if expected_outcome == "refuse" or case_type in PERMISSION_CASE_TYPES:
        metric_group = "permission"
    elif case_type in TABLE_CASE_TYPES:
        metric_group = "table_key_value"
    elif case_type == LOW_OVERLAP_SCENARIO_CASE_TYPE and not has_matching_evidence_anchor(question, evidence_text):
        metric_group = "broad_document_discovery"
        flags.append("broad_question_with_exact_gold")
        if not explicit_evidence_anchor(question):
            flags.append("missing_concrete_evidence_anchor")
        else:
            flags.append("concrete_evidence_anchor_not_supported")

    if (
        expected_outcome == "answer"
        and metric_group not in {"permission", "broad_document_discovery"}
        and case_type in ANCHOR_RELIANT_CASE_TYPES
        and anchor_report["unreliable_anchor_count"]
    ):
        metric_group = "anchor_quality_review"
        flags.append("unreliable_question_anchor")
        if anchor_report.get("under_specified_anchor_count"):
            flags.append("under_specified_question_anchor")
        flags.extend(anchor_report["flag_reasons"])

    if expected_outcome == "answer" and marker_presence_issue:
        metric_group = "anchor_quality_review"
        flags.append("expected_marker_not_found_in_chunks")

    if expected_outcome == "answer" and evidence_text:
        if overlap < weak_overlap_threshold:
            flags.append("weak_question_evidence_overlap")
        if overlap < very_weak_overlap_threshold:
            flags.append("very_weak_question_evidence_overlap")

    return {
        "case_name": case.get("case_name"),
        "case_type": case_type,
        "expected_outcome": expected_outcome,
        "metric_group": metric_group,
        "question_evidence_overlap": round(overlap, 6),
        "flags": flags,
        "question_anchors": anchor_report["anchors"],
        "unreliable_question_anchors": anchor_report["unreliable_anchors"],
        "anchor_occurrence_stats": anchor_report["anchor_occurrence_stats"],
        "missing_gold_discriminators": anchor_report["missing_gold_discriminators"],
        "marker_presence_issue": marker_presence_issue or {},
        "question_preview": question[:180],
        "evidence_preview": evidence_text[:220],
    }


def load_document_texts(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "documents" not in raw:
        return {str(key): str(value or "") for key, value in raw.items() if str(value or "")}
    documents = raw.get("documents") if isinstance(raw, dict) else raw
    if not isinstance(documents, list):
        raise ValueError("document text JSON must be a mapping or a list/documents list")
    texts: dict[str, str] = {}
    for document in documents:
        if not isinstance(document, dict):
            continue
        text = str(document.get("text") or document.get("content") or "")
        if not text:
            continue
        for key_name in ("id", "document_id", "title", "document_title"):
            key = str(document.get(key_name) or "").strip()
            if key:
                texts[key] = text
    return texts


def load_marker_presence_issues(paths: list[str] | None) -> dict[str, dict[str, Any]]:
    issues: dict[str, dict[str, Any]] = {}
    for raw_path in paths or []:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"root-cause report not found: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        for item in report.get("cases", []):
            missing_markers = [
                str(marker.get("label") or "")
                for marker in item.get("markers", [])
                if marker.get("root_cause") == "marker_not_found_in_expected_document_chunks"
            ]
            if not missing_markers:
                continue
            case_name = str(item.get("case_name") or "")
            if not case_name:
                continue
            current = issues.setdefault(
                case_name,
                {
                    "root_cause": "marker_not_found_in_expected_document_chunks",
                    "missing_marker_count": 0,
                    "missing_marker_preview": [],
                    "source_reports": [],
                },
            )
            current["missing_marker_count"] += len(missing_markers)
            remaining_preview_slots = max(0, 5 - len(current["missing_marker_preview"]))
            current["missing_marker_preview"].extend(missing_markers[:remaining_preview_slots])
            current["source_reports"].append(str(path))
    return issues


def expected_evidence_text(case: dict[str, Any]) -> str:
    parts: list[str] = []
    for marker in case.get("expected_evidence_markers") or []:
        if isinstance(marker, dict):
            parts.append(str(marker.get("label") or ""))
            parts.extend(str(alias) for alias in marker.get("aliases") or [])
        else:
            parts.append(str(marker))
    return " ".join(part for part in parts if part)


def explicit_evidence_anchor(question: str) -> str:
    match = re.search(r"重点核对[“\"']([^”\"']{2,120})[”\"']", question)
    if not match:
        return ""
    return match.group(1).strip()


def has_matching_evidence_anchor(question: str, evidence_text: str) -> bool:
    anchor = explicit_evidence_anchor(question)
    if not anchor:
        return False
    if anchor in evidence_text:
        return True
    return question_evidence_overlap(anchor, evidence_text) >= 0.5


def audit_question_anchor_quality(
    question: str,
    *,
    case: dict[str, Any] | None = None,
    document_texts: dict[str, str] | None = None,
    anchor_occurrence_threshold: int = DEFAULT_ANCHOR_OCCURRENCE_THRESHOLD,
) -> dict[str, Any]:
    anchors = question_anchors(question)
    unreliable: list[dict[str, Any]] = []
    flag_reasons: list[str] = []
    occurrence_stats: list[dict[str, Any]] = []
    missing_discriminators: list[dict[str, Any]] = []
    under_specified_count = 0
    for anchor in anchors:
        reasons = weak_anchor_reasons(anchor)
        occurrence = anchor_occurrence_stats(anchor, case or {}, document_texts or {})
        if occurrence["max_document_count"] >= max(1, int(anchor_occurrence_threshold)):
            reasons.append("repeated_in_expected_document")
        discriminators = missing_gold_discriminators_for_anchor(question, anchor, case or {})
        low_information_anchor = bool(reasons)
        if discriminators and low_information_anchor:
            reasons.append("missing_gold_discriminator")
            missing_discriminators.extend(discriminators)
        if not reasons:
            continue
        deduped_reasons = sorted(set(reasons))
        under_specified_count += 1
        occurrence_stats.append({"anchor": anchor, **occurrence})
        unreliable.append(
            {
                "anchor": anchor,
                "reasons": deduped_reasons,
                "anchor_occurrence": occurrence,
                "missing_gold_discriminators": discriminators,
            }
        )
        flag_reasons.extend(f"weak_question_anchor:{reason}" for reason in deduped_reasons)
    return {
        "anchors": anchors,
        "unreliable_anchors": unreliable,
        "unreliable_anchor_count": len(unreliable),
        "under_specified_anchor_count": under_specified_count,
        "anchor_occurrence_stats": occurrence_stats,
        "missing_gold_discriminators": missing_discriminators,
        "flag_reasons": sorted(set(flag_reasons)),
    }


def question_anchors(question: str) -> list[str]:
    return [item.strip() for item in re.findall(r"“([^”]{1,160})”", question) if item.strip()]


def weak_anchor_reasons(anchor: str) -> list[str]:
    compact = re.sub(r"\s+", "", anchor)
    alnum_cjk = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9.%％]+", "", compact)
    reasons: list[str] = []
    if len(alnum_cjk) < 6:
        reasons.append("too_short")
    if re.fullmatch(r"(?:截至)?\d{4}年\d{1,2}月\d{1,2}日", compact):
        reasons.append("date_only")
    if re.fullmatch(r"(?:截至)?\d{4}年(?:年)?(?:末|底|初|度|上半年|下半年|半年度|[一二三四1-4]季度)", compact):
        reasons.append("period_only")
    if re.fullmatch(r"序号为\d+", compact):
        reasons.append("row_number_only")
    if re.fullmatch(r"\d+(?:\.\d+)?(?:万?元|亿元|%|％|倍|条)?", compact):
        reasons.append("numeric_only")
    if re.fullmatch(r"\d+(?:\.\d+)?条", compact):
        reasons.append("article_fragment")
    if "募集说明书人作为" in compact or "说明书人作为" in compact:
        reasons.append("pdf_extraction_fragment")
    return reasons


def anchor_occurrence_stats(anchor: str, case: dict[str, Any], document_texts: dict[str, str]) -> dict[str, Any]:
    if not document_texts:
        return {"document_count": 0, "max_document_count": 0, "total_count": 0, "documents_checked": 0}
    document_keys = expected_document_keys(case)
    counts: list[dict[str, Any]] = []
    for key in document_keys:
        text = document_texts.get(key)
        if not text:
            continue
        count = count_compact_occurrences(text, anchor)
        counts.append({"document_key": key, "count": count})
    if not counts:
        return {"document_count": 0, "max_document_count": 0, "total_count": 0, "documents_checked": len(document_keys)}
    positive_counts = [item for item in counts if item["count"] > 0]
    return {
        "document_count": len(positive_counts),
        "max_document_count": max((int(item["count"]) for item in counts), default=0),
        "total_count": sum(int(item["count"]) for item in counts),
        "documents_checked": len(counts),
        "documents_preview": positive_counts[:5],
    }


def expected_document_keys(case: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for key_name in ("expected_document_ids", "expected_retrieval_titles"):
        keys.extend(str(value).strip() for value in case.get(key_name) or [] if str(value).strip())
    for marker in case.get("expected_evidence_markers") or []:
        if not isinstance(marker, dict):
            continue
        for key_name in ("document_id", "document_title"):
            value = str(marker.get(key_name) or "").strip()
            if value:
                keys.append(value)
    return dedupe(keys)


def count_compact_occurrences(text: str, needle: str) -> int:
    compact_text = compact_for_anchor_matching(text)
    compact_needle = compact_for_anchor_matching(needle)
    if not compact_text or not compact_needle:
        return 0
    return compact_text.count(compact_needle)


def compact_for_anchor_matching(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def missing_gold_discriminators_for_anchor(question: str, anchor: str, case: dict[str, Any]) -> list[dict[str, Any]]:
    question_compact = compact_for_anchor_matching(question)
    rows: list[dict[str, Any]] = []
    for marker in case.get("expected_evidence_markers") or []:
        if isinstance(marker, dict):
            marker_texts = dedupe([str(marker.get("label") or ""), *[str(alias) for alias in marker.get("aliases") or []]])
            document_title = str(marker.get("document_title") or "")
        else:
            marker_texts = [str(marker)]
            document_title = ""
        for marker_text in marker_texts:
            phrase = discriminator_phrase_after_anchor(marker_text, anchor)
            if not phrase:
                continue
            if compact_for_anchor_matching(phrase) in question_compact:
                continue
            rows.append(
                {
                    "anchor": anchor,
                    "missing_phrase": phrase,
                    "document_title": document_title,
                    "marker_preview": marker_text[:120],
                }
            )
    return rows[:5]


def discriminator_phrase_after_anchor(marker_text: str, anchor: str) -> str:
    marker = str(marker_text or "")
    exact_index = marker.find(anchor)
    if exact_index >= 0:
        after = marker[exact_index + len(anchor) :]
    else:
        compact_marker = compact_for_anchor_matching(marker)
        compact_anchor = compact_for_anchor_matching(anchor)
        compact_index = compact_marker.find(compact_anchor)
        if compact_index < 0:
            return ""
        after = compact_marker[compact_index + len(compact_anchor) :]
    after = re.sub(r"^[\s，,。；;：:、]+", "", after)
    after = re.sub(r"^(?:发行人|公司|本公司|该公司)", "", after)
    after = re.sub(r"^[^\u4e00-\u9fffA-Za-z0-9]+", "", after)
    match = re.match(r"[\u4e00-\u9fffA-Za-z0-9（）()、]{6,40}", after)
    if not match:
        return ""
    phrase = match.group(0)
    phrase = re.split(r"[，,。；;：:]", phrase, maxsplit=1)[0]
    phrase = re.sub(r"^(?:发行人|公司|本公司|该公司)", "", phrase)
    if len(compact_for_anchor_matching(phrase)) < 6:
        return ""
    return phrase[:28]


def dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip()
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def question_evidence_overlap(question: str, evidence_text: str) -> float:
    question_terms = quality_terms(question)
    if not question_terms:
        return 0.0
    evidence_terms = quality_terms(evidence_text)
    return len(question_terms & evidence_terms) / len(question_terms)


def quality_terms(value: str) -> set[str]:
    normalized = str(value or "").casefold()
    terms: set[str] = set(re.findall(r"[a-z0-9]+", normalized))
    for cjk_text in re.findall(r"[\u4e00-\u9fff]+", normalized):
        for size in (2, 3, 4):
            if len(cjk_text) < size:
                continue
            for index in range(len(cjk_text) - size + 1):
                terms.add(cjk_text[index : index + size])
    return {
        term
        for term in terms
        if len(term) > 1 and term not in LOW_SIGNAL_TOKENS and not any(fragment in term for fragment in LOW_SIGNAL_TOKENS if len(fragment) >= 3)
    }


def summarize_cases(case_reports: list[dict[str, Any]]) -> dict[str, Any]:
    overlaps = [float(item["question_evidence_overlap"]) for item in case_reports if item["expected_outcome"] == "answer"]
    flag_counts = Counter(flag for item in case_reports for flag in item["flags"])
    metric_group_counts = Counter(str(item["metric_group"]) for item in case_reports)
    if not overlaps:
        overlap_summary = {
            "answer_case_count": 0,
            "avg_question_evidence_overlap": None,
            "median_question_evidence_overlap": None,
        }
    else:
        overlap_summary = {
            "answer_case_count": len(overlaps),
            "avg_question_evidence_overlap": round(mean(overlaps), 6),
            "median_question_evidence_overlap": round(median(overlaps), 6),
            "min_question_evidence_overlap": round(min(overlaps), 6),
            "max_question_evidence_overlap": round(max(overlaps), 6),
        }
    return {
        "case_count": len(case_reports),
        **overlap_summary,
        "metric_group_counts": dict(sorted(metric_group_counts.items())),
        "flag_counts": dict(sorted(flag_counts.items())),
    }


def build_annotated_manifest(report: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    quality_by_case_name = {
        item["case_name"]: {
            "metric_group": item["metric_group"],
            "question_evidence_overlap": item["question_evidence_overlap"],
            "flags": item["flags"],
            "anchor_occurrence_stats": item.get("anchor_occurrence_stats") or [],
            "missing_gold_discriminators": item.get("missing_gold_discriminators") or [],
            "marker_presence_issue": item.get("marker_presence_issue") or {},
        }
        for item in report["cases"]
    }
    cases = []
    for case in manifest.get("cases", []):
        updated_case = {**case}
        metadata = dict(updated_case.get("metadata") or {})
        metadata["quality_audit"] = quality_by_case_name.get(case.get("case_name"), {})
        updated_case["metadata"] = metadata
        cases.append(updated_case)
    return {
        **manifest,
        "cases": cases,
        "case_quality_audit": {
            "source_manifest": report.get("manifest_path"),
            "summary": report["summary"],
            "case_type_summary": report["case_type_summary"],
            "strict_manifest_rule": "exclude broad document-discovery and unreliable-anchor cases from strict exact-evidence benchmark",
        },
    }


def build_strict_manifest(report: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    quality_by_case_name = {item["case_name"]: item for item in report["cases"]}
    strict_cases = []
    excluded_case_names = []
    for case in manifest.get("cases", []):
        quality = quality_by_case_name.get(case.get("case_name"))
        if quality and quality["metric_group"] in STRICT_EXCLUDED_METRIC_GROUPS:
            excluded_case_names.append(case.get("case_name"))
            continue
        updated_case = {**case}
        metadata = dict(updated_case.get("metadata") or {})
        if quality:
            metadata["quality_audit"] = {
                "metric_group": quality["metric_group"],
                "question_evidence_overlap": quality["question_evidence_overlap"],
                "flags": quality["flags"],
                "anchor_occurrence_stats": quality.get("anchor_occurrence_stats") or [],
                "missing_gold_discriminators": quality.get("missing_gold_discriminators") or [],
                "marker_presence_issue": quality.get("marker_presence_issue") or {},
            }
        updated_case["metadata"] = metadata
        strict_cases.append(updated_case)
    strict_case_reports = [
        quality_by_case_name[case["case_name"]]
        for case in strict_cases
        if case.get("case_name") in quality_by_case_name
    ]
    return {
        **manifest,
        "benchmark_version": f"{manifest.get('benchmark_version', 'unknown')}-strict-evidence",
        "description": (
            f"{manifest.get('description', '').rstrip()} "
            "Quality-filtered copy for strict exact-evidence retrieval; broad document-discovery cases are excluded."
        ).strip(),
        "cases": strict_cases,
        "case_quality_audit": {
            "source_manifest": report.get("manifest_path"),
            "source_case_count": len(manifest.get("cases", [])),
            "strict_case_count": len(strict_cases),
            "excluded_case_count": len(excluded_case_names),
            "excluded_metric_groups": sorted(STRICT_EXCLUDED_METRIC_GROUPS),
            "excluded_case_names_preview": excluded_case_names[:50],
            "summary": summarize_cases(strict_case_reports),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Case Quality Audit",
        "",
        f"- Dataset: `{report.get('dataset_name')}`",
        f"- Manifest: `{report.get('manifest_path')}`",
        f"- Cases: `{report.get('case_count')}`",
        "",
        "## Overall",
        "",
        render_summary_table({"all": report["summary"]}),
        "",
        "## Case Types",
        "",
        render_summary_table(report["case_type_summary"]),
        "",
        "## Weak Case Preview",
        "",
    ]
    weak_cases = report.get("weak_cases") or []
    if not weak_cases:
        lines.append("No weak cases found.")
    else:
        lines.extend(["| Case | Type | Overlap | Metric group | Flags |", "| --- | --- | ---: | --- | --- |"])
        for item in weak_cases[:30]:
            lines.append(
                f"| `{item['case_name']}` | `{item['case_type']}` | {item['question_evidence_overlap']} | "
                f"`{item['metric_group']}` | {', '.join(item['flags'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def render_summary_table(summary_by_key: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Group | Cases | Answer cases | Avg overlap | Median overlap | Metric groups | Flags |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for key, summary in summary_by_key.items():
        metric_groups = ", ".join(f"{name}:{count}" for name, count in summary.get("metric_group_counts", {}).items()) or "none"
        flags = ", ".join(f"{name}:{count}" for name, count in summary.get("flag_counts", {}).items()) or "none"
        lines.append(
            f"| `{key}` | {summary.get('case_count')} | {summary.get('answer_case_count')} | "
            f"{summary.get('avg_question_evidence_overlap')} | {summary.get('median_question_evidence_overlap')} | "
            f"{metric_groups} | {flags} |"
        )
    return "\n".join(lines)


def summary_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    flags = ", ".join(f"{name}={count}" for name, count in summary.get("flag_counts", {}).items()) or "none"
    groups = ", ".join(f"{name}={count}" for name, count in summary.get("metric_group_counts", {}).items()) or "none"
    return (
        f"cases={summary['case_count']} answer_cases={summary.get('answer_case_count')} "
        f"avg_overlap={summary.get('avg_question_evidence_overlap')} metric_groups={groups} flags={flags}"
    )


if __name__ == "__main__":
    main()
