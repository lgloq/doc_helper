from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "backend" / "data" / "eval_outputs"


def main() -> None:
    args = build_parser().parse_args()
    report = build_report(
        manifest_path=Path(args.manifest).resolve(),
        diagnostics_paths=[Path(path).resolve() for path in args.diagnostics],
        low_overlap_threshold=args.low_overlap_threshold,
        saturation_pass_rate=args.saturation_pass_rate,
    )
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {output_path}")
    if args.markdown_output:
        markdown_path = Path(args.markdown_output).resolve()
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        print(f"Wrote {markdown_path}")
    print(summary_text(report))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report whether a benchmark is hard enough to guide RAG-chain optimization. "
            "This is an offline analysis over a manifest plus retrieval diagnostics artifacts."
        )
    )
    parser.add_argument("--manifest", required=True, help="Benchmark manifest JSON.")
    parser.add_argument(
        "--diagnostics",
        action="append",
        default=[],
        help="Retrieval diagnostics JSON. Can be repeated.",
    )
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--markdown-output", help="Optional Markdown output path.")
    parser.add_argument(
        "--low-overlap-threshold",
        type=float,
        default=0.35,
        help="Case-average query/evidence lexical overlap below this value is considered low-overlap.",
    )
    parser.add_argument(
        "--saturation-pass-rate",
        type=float,
        default=0.98,
        help="Ablations at or above this pass rate are considered saturated.",
    )
    return parser


def build_report(
    *,
    manifest_path: Path,
    diagnostics_paths: list[Path],
    low_overlap_threshold: float,
    saturation_pass_rate: float,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    manifest_summary = summarize_manifest(manifest, low_overlap_threshold=low_overlap_threshold)
    diagnostics = [
        summarize_diagnostics(path, saturation_pass_rate=saturation_pass_rate)
        for path in diagnostics_paths
        if path.exists()
    ]
    return {
        "manifest_path": str(manifest_path),
        "dataset_name": manifest.get("dataset_name"),
        "hardness_thresholds": {
            "low_overlap_threshold": low_overlap_threshold,
            "saturation_pass_rate": saturation_pass_rate,
            "min_recommended_documents": 20,
            "max_recommended_cases_per_document": 8,
        },
        "manifest": manifest_summary,
        "diagnostics": diagnostics,
        "recommendations": recommendations(manifest_summary, diagnostics),
    }


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def summarize_manifest(manifest: dict[str, Any], *, low_overlap_threshold: float) -> dict[str, Any]:
    documents = list(manifest.get("documents") or [])
    cases = list(manifest.get("cases") or [])
    document_count = len(documents)
    case_count = len(cases)
    case_type_counts = Counter(case_type(case) for case in cases)
    domain_counts = Counter(str((doc.get("metadata") or {}).get("domain") or "unknown") for doc in documents)
    source_org_counts = Counter(str((doc.get("metadata") or {}).get("source_org") or "unknown") for doc in documents)
    expected_doc_counts = [len(case.get("expected_document_ids") or []) for case in cases]
    evidence_marker_counts = [len(case.get("expected_evidence_markers") or []) for case in cases]
    overlap_cases = [case_overlap_summary(case, low_overlap_threshold=low_overlap_threshold) for case in cases]
    overlap_values = [item["avg_overlap"] for item in overlap_cases if item["marker_count"] > 0]
    overlap_by_type = summarize_overlap_by_type(overlap_cases)
    evidence_overlap_cases = [item for item in overlap_cases if item["marker_count"] > 0]
    low_overlap_cases = [item for item in evidence_overlap_cases if item["low_overlap"]]

    return {
        "document_count": document_count,
        "case_count": case_count,
        "cases_per_document": round(case_count / document_count, 4) if document_count else None,
        "case_type_counts": dict(sorted(case_type_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "source_org_counts": dict(sorted(source_org_counts.items())),
        "expected_document_count": {
            "avg": round(sum(expected_doc_counts) / len(expected_doc_counts), 4) if expected_doc_counts else 0.0,
            "multi_document_cases": sum(1 for value in expected_doc_counts if value > 1),
            "zero_expected_document_cases": sum(1 for value in expected_doc_counts if value == 0),
        },
        "evidence_marker_count": {
            "avg": round(sum(evidence_marker_counts) / len(evidence_marker_counts), 4) if evidence_marker_counts else 0.0,
            "multi_evidence_cases": sum(1 for value in evidence_marker_counts if value > 1),
        },
        "query_evidence_overlap": {
            "avg": round(sum(overlap_values) / len(overlap_values), 6) if overlap_values else 0.0,
            "median": round(median(overlap_values), 6) if overlap_values else 0.0,
            "p25": round(percentile(overlap_values, 0.25), 6) if overlap_values else 0.0,
            "p75": round(percentile(overlap_values, 0.75), 6) if overlap_values else 0.0,
            "low_overlap_count": len(low_overlap_cases),
            "low_overlap_rate": round(len(low_overlap_cases) / len(evidence_overlap_cases), 6) if evidence_overlap_cases else 0.0,
            "by_case_type": overlap_by_type,
            "lowest_overlap_cases": sorted(evidence_overlap_cases, key=lambda item: item["avg_overlap"])[:10],
        },
        "risk_flags": manifest_risk_flags(
            document_count=document_count,
            case_count=case_count,
            cases_per_document=case_count / document_count if document_count else math.inf,
            low_overlap_rate=(len(low_overlap_cases) / len(evidence_overlap_cases) if evidence_overlap_cases else 0.0),
        ),
    }


def case_type(case: dict[str, Any]) -> str:
    metadata = case.get("metadata") or {}
    if metadata.get("case_type"):
        return str(metadata["case_type"])
    if metadata.get("permission_variant"):
        return "permission"
    if case.get("expected_outcome") == "refuse":
        return "permission"
    return "single_fact"


def case_overlap_summary(case: dict[str, Any], *, low_overlap_threshold: float) -> dict[str, Any]:
    question = str(case.get("question") or "")
    question_tokens = set(tokenize_text(question))
    marker_scores = []
    for marker in case.get("expected_evidence_markers") or []:
        aliases = marker_aliases(marker)
        best = 0.0
        best_alias = ""
        for alias in aliases:
            alias_tokens = set(tokenize_text(alias))
            if not question_tokens:
                score = 0.0
            else:
                score = len(question_tokens.intersection(alias_tokens)) / len(question_tokens)
            if score > best:
                best = score
                best_alias = alias
        marker_scores.append({"overlap": round(best, 6), "best_alias": best_alias[:120]})
    scores = [item["overlap"] for item in marker_scores]
    avg_overlap = sum(scores) / len(scores) if scores else 0.0
    return {
        "case_name": case.get("case_name"),
        "case_type": case_type(case),
        "question": question,
        "expected_document_count": len(case.get("expected_document_ids") or []),
        "marker_count": len(marker_scores),
        "question_token_count": len(question_tokens),
        "avg_overlap": round(avg_overlap, 6),
        "min_marker_overlap": round(min(scores), 6) if scores else 0.0,
        "max_marker_overlap": round(max(scores), 6) if scores else 0.0,
        "low_overlap": bool(scores) and avg_overlap < low_overlap_threshold,
    }


def marker_aliases(marker: Any) -> list[str]:
    if isinstance(marker, str):
        return [marker]
    if not isinstance(marker, dict):
        return []
    aliases = []
    label = str(marker.get("label") or "").strip()
    if label:
        aliases.append(label)
    aliases.extend(str(alias).strip() for alias in marker.get("aliases") or [] if str(alias).strip())
    return list(dict.fromkeys(aliases))


def tokenize_text(text: str) -> list[str]:
    compact = re.sub(r"\s+", "", text.casefold())
    tokens: list[str] = []
    tokens.extend(re.findall(r"[a-z0-9]+", compact))
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", compact)
    tokens.extend(cjk_chars)
    tokens.extend("".join(cjk_chars[index : index + 2]) for index in range(max(0, len(cjk_chars) - 1)))
    return [token for token in tokens if token not in STOP_TOKENS and len(token) > 0]


STOP_TOKENS = {
    "的",
    "了",
    "和",
    "及",
    "或",
    "与",
    "在",
    "中",
    "时",
    "应",
    "要",
    "能",
    "哪些",
}


def summarize_overlap_by_type(overlap_cases: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    low_counts: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for item in overlap_cases:
        case_type_name = item["case_type"]
        counts[case_type_name] += 1
        grouped[case_type_name].append(item["avg_overlap"])
        if item["low_overlap"]:
            low_counts[case_type_name] += 1
    return {
        case_type_name: {
            "count": counts[case_type_name],
            "avg_overlap": round(sum(values) / len(values), 6) if values else 0.0,
            "low_overlap_count": low_counts[case_type_name],
            "low_overlap_rate": round(low_counts[case_type_name] / counts[case_type_name], 6) if counts[case_type_name] else 0.0,
        }
        for case_type_name, values in sorted(grouped.items())
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def manifest_risk_flags(*, document_count: int, case_count: int, cases_per_document: float, low_overlap_rate: float) -> list[str]:
    flags = []
    if document_count < 20:
        flags.append("small_document_pool")
    if cases_per_document > 8:
        flags.append("high_cases_per_document")
    if low_overlap_rate < 0.25 and case_count >= 20:
        flags.append("low_overlap_underrepresented")
    return flags


def summarize_diagnostics(path: Path, *, saturation_pass_rate: float) -> dict[str, Any]:
    payload = load_json(path)
    ablations = [summarize_ablation(item) for item in payload.get("ablations") or []]
    saturated = [item for item in ablations if item["pass_rate"] is not None and item["pass_rate"] >= saturation_pass_rate]
    return {
        "path": str(path),
        "dataset_name": payload.get("dataset_name"),
        "case_count": payload.get("case_count"),
        "document_scope": payload.get("document_scope"),
        "case_filters": payload.get("case_filters"),
        "ablation_count": len(ablations),
        "ablations": ablations,
        "saturation": {
            "saturation_pass_rate": saturation_pass_rate,
            "saturated_ablation_count": len(saturated),
            "all_ablations_saturated": bool(ablations) and len(saturated) == len(ablations),
            "saturated_ablations": [item["name"] for item in saturated],
        },
        "risk_flags": diagnostics_risk_flags(ablations, saturation_pass_rate=saturation_pass_rate),
    }


def summarize_ablation(item: dict[str, Any]) -> dict[str, Any]:
    ablation = item.get("ablation") or {}
    summary = item.get("summary") or {}
    stage_metrics = summary.get("stage_metrics") or {}
    lexical = stage_metrics.get("lexical") or {}
    final = stage_metrics.get("final") or {}
    return {
        "name": ablation.get("name"),
        "total_cases": summary.get("total_cases"),
        "pass_count": summary.get("pass_count"),
        "pass_rate": summary.get("pass_rate"),
        "recall_at_k_avg": summary.get("recall_at_k_avg"),
        "evidence_recall_at_k_avg": summary.get("evidence_recall_at_k_avg"),
        "mrr_avg": summary.get("mrr_avg"),
        "evidence_mrr_avg": summary.get("evidence_mrr_avg"),
        "failure_mode_counts": summary.get("failure_mode_counts") or {},
        "stage_loss_counts": summary.get("stage_loss_counts") or {},
        "lexical_evidence_recall_avg": lexical.get("evidence_recall_avg"),
        "final_evidence_recall_avg": final.get("evidence_recall_avg"),
    }


def diagnostics_risk_flags(ablations: list[dict[str, Any]], *, saturation_pass_rate: float) -> list[str]:
    flags = []
    if len(ablations) > 1 and all((item["pass_rate"] or 0.0) >= saturation_pass_rate for item in ablations):
        flags.append("ablation_saturated")
    default = next((item for item in ablations if item["name"] == "full_local"), ablations[0] if ablations else None)
    lexical_only = next((item for item in ablations if item["name"] == "lexical_only"), None)
    no_expansion = next((item for item in ablations if item["name"] == "no_expansion"), None)
    if default and lexical_only and lexical_only["pass_rate"] == default["pass_rate"]:
        flags.append("lexical_only_matches_default")
    if default and no_expansion and no_expansion["pass_rate"] == default["pass_rate"]:
        flags.append("expansion_not_required_on_this_slice")
    return flags


def recommendations(manifest: dict[str, Any], diagnostics: list[dict[str, Any]]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    risk_flags = set(manifest.get("risk_flags") or [])
    diagnostic_flags = {flag for diagnostic in diagnostics for flag in diagnostic.get("risk_flags") or []}
    if "small_document_pool" in risk_flags:
        items.append(
            {
                "priority": "P0",
                "action": "Add real same-domain distractor documents",
                "reason": "The benchmark has fewer than 20 documents, so document-level recall is too easy to saturate.",
            }
        )
    if "high_cases_per_document" in risk_flags:
        items.append(
            {
                "priority": "P0",
                "action": "Reduce cases-per-document pressure by adding documents, not by splitting sources",
                "reason": "Many questions point to the same small document set; this does not model enterprise search across a broad corpus.",
            }
        )
    if "ablation_saturated" in diagnostic_flags:
        items.append(
            {
                "priority": "P0",
                "action": "Do not tune retrieval parameters from this gate",
                "reason": "Multiple ablations reach the saturation pass threshold, so the gate cannot distinguish chain improvements.",
            }
        )
    if "lexical_only_matches_default" in diagnostic_flags:
        items.append(
            {
                "priority": "P1",
                "action": "Add harder low-overlap cases or semantic evaluation before changing rerank/fusion",
                "reason": "Lexical-only performance matches the default on this slice, so semantic/candidate-generation weaknesses are not exposed.",
            }
        )
    if not items:
        items.append(
            {
                "priority": "P1",
                "action": "Use diagnostics failure buckets to choose the next chain stage",
                "reason": "The current artifacts do not show benchmark saturation as the dominant issue.",
            }
        )
    return items


def render_markdown(report: dict[str, Any]) -> str:
    manifest = report["manifest"]
    lines = [
        f"# Benchmark Hardness Report: {report.get('dataset_name')}",
        "",
        "## Manifest",
        "",
        f"- Documents: `{manifest['document_count']}`",
        f"- Cases: `{manifest['case_count']}`",
        f"- Cases per document: `{manifest['cases_per_document']}`",
        f"- Risk flags: `{', '.join(manifest['risk_flags']) or 'none'}`",
        "",
        "### Case Types",
        "",
        "| Case type | Count |",
        "| --- | ---: |",
    ]
    for name, count in manifest["case_type_counts"].items():
        lines.append(f"| `{name}` | {count} |")
    overlap = manifest["query_evidence_overlap"]
    lines.extend(
        [
            "",
            "### Query/Evidence Overlap",
            "",
            f"- Average overlap: `{overlap['avg']}`",
            f"- Median overlap: `{overlap['median']}`",
            f"- Low-overlap cases: `{overlap['low_overlap_count']}` (`{overlap['low_overlap_rate']}`)",
            "",
            "| Case | Type | Avg overlap | Markers |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for item in overlap["lowest_overlap_cases"][:10]:
        lines.append(f"| `{item['case_name']}` | `{item['case_type']}` | {item['avg_overlap']} | {item['marker_count']} |")
    for diagnostic in report["diagnostics"]:
        lines.extend(
            [
                "",
                f"## Diagnostics: `{Path(diagnostic['path']).name}`",
                "",
                f"- Cases: `{diagnostic['case_count']}`",
                f"- Risk flags: `{', '.join(diagnostic['risk_flags']) or 'none'}`",
                f"- Saturated ablations: `{', '.join(diagnostic['saturation']['saturated_ablations']) or 'none'}`",
                "",
                "| Ablation | Cases | Pass rate | Recall@10 | Evidence recall@10 |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for ablation in diagnostic["ablations"]:
            lines.append(
                "| `{}` | {} | {} | {} | {} |".format(
                    ablation["name"],
                    ablation["total_cases"],
                    ablation["pass_rate"],
                    ablation["recall_at_k_avg"],
                    ablation["evidence_recall_at_k_avg"],
                )
            )
    lines.extend(["", "## Recommendations", "", "| Priority | Action | Reason |", "| --- | --- | --- |"])
    for item in report["recommendations"]:
        lines.append(f"| {item['priority']} | {item['action']} | {item['reason']} |")
    lines.append("")
    return "\n".join(lines)


def summary_text(report: dict[str, Any]) -> str:
    manifest = report["manifest"]
    parts = [
        f"dataset={report.get('dataset_name')}",
        f"documents={manifest['document_count']}",
        f"cases={manifest['case_count']}",
        f"low_overlap={manifest['query_evidence_overlap']['low_overlap_count']}",
        f"risk_flags={','.join(manifest['risk_flags']) or 'none'}",
    ]
    for diagnostic in report["diagnostics"]:
        parts.append(
            f"diagnostic={Path(diagnostic['path']).name}:flags={','.join(diagnostic['risk_flags']) or 'none'}"
        )
    return " ".join(parts)


if __name__ == "__main__":
    main()
