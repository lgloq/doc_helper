from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "backend" / "data" / "eval_outputs" / "zh-enterprise-v1-verified234-full-after-exact-evidence-local.json"
DEFAULT_OUTPUT = ROOT_DIR / "backend" / "data" / "eval_outputs" / "zh-enterprise-v1-verified234-latency-outliers-local.json"
DEFAULT_MARKDOWN_OUTPUT = ROOT_DIR / "backend" / "data" / "eval_outputs" / "zh-enterprise-v1-verified234-latency-outliers-local.md"

TOTAL_LATENCY_KEYS = {"search_total_latency_ms"}


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input).resolve()
    report = json.loads(input_path.read_text(encoding="utf-8"))
    audit = audit_latency_outliers(
        report,
        source_report=input_path,
        slow_threshold_ms=args.slow_threshold_ms,
        top_n=args.top_n,
    )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown_output:
        markdown_path = Path(args.markdown_output).resolve()
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(audit), encoding="utf-8")
    print(summary_text(audit))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit retrieval benchmark latency outliers. This parser does not rerun retrieval; "
            "it uses saved per-case retrieval_debug when present and reports replay gaps otherwise."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="run_retrieval_benchmark or ablation benchmark JSON.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    parser.add_argument("--slow-threshold-ms", type=int, default=8000)
    parser.add_argument("--top-n", type=int, default=10)
    return parser


def audit_latency_outliers(
    report: dict[str, Any],
    *,
    source_report: Path | str | None = None,
    slow_threshold_ms: int = 8000,
    top_n: int = 10,
) -> dict[str, Any]:
    ablations = [audit_ablation_latency(item, slow_threshold_ms=slow_threshold_ms, top_n=top_n) for item in report_ablations(report)]
    outliers = [case for ablation in ablations for case in ablation["outliers"]]
    debug_missing = [case for case in outliers if case["debug_status"] == "missing"]
    stage_counts = Counter(case["dominant_stage"] for case in outliers if case["dominant_stage"])
    recommended_replay_cases = recommended_replays(debug_missing)
    return {
        "source_report": str(source_report) if source_report else None,
        "dataset_name": report.get("dataset_name"),
        "manifest": report.get("manifest"),
        "slow_threshold_ms": slow_threshold_ms,
        "top_n": top_n,
        "summary": {
            "ablation_count": len(ablations),
            "outlier_count": len(outliers),
            "debug_available_count": sum(1 for case in outliers if case["debug_status"] == "available"),
            "debug_missing_count": len(debug_missing),
            "dominant_stage_counts": dict(sorted(stage_counts.items())),
            "recommended_replay_case_count": len(recommended_replay_cases),
        },
        "ablations": ablations,
        "recommended_replay_cases": recommended_replay_cases,
    }


def report_ablations(report: dict[str, Any]) -> list[dict[str, Any]]:
    ablations = report.get("ablations")
    if isinstance(ablations, list) and ablations:
        return ablations
    name = str(report.get("ablation", {}).get("name") or report.get("retrieval_domain_profile") or "single_run")
    return [
        {
            "ablation": {"name": name},
            "summary": report.get("summary", {}),
            "latency": report.get("latency") or (report.get("summary") or {}).get("latency_seconds") or {},
            "slowest_cases": report.get("slowest_cases", []),
            "cases": report.get("cases", []),
            "failure_cases": report.get("failure_cases", []),
        }
    ]


def audit_ablation_latency(item: dict[str, Any], *, slow_threshold_ms: int, top_n: int) -> dict[str, Any]:
    ablation_name = str((item.get("ablation") or {}).get("name") or "unknown")
    rows_by_case = {
        str(row.get("case_name")): row
        for row in [*(item.get("cases") or []), *(item.get("failure_cases") or [])]
        if row.get("case_name")
    }
    outlier_refs = collect_outlier_refs(item, rows_by_case, slow_threshold_ms=slow_threshold_ms, top_n=top_n)
    outliers = [audit_outlier_case(ref, rows_by_case.get(str(ref.get("case_name")))) for ref in outlier_refs]
    return {
        "ablation_name": ablation_name,
        "latency": item.get("latency") or (item.get("summary") or {}).get("latency_seconds") or {},
        "summary": {
            "outlier_count": len(outliers),
            "debug_available_count": sum(1 for case in outliers if case["debug_status"] == "available"),
            "debug_missing_count": sum(1 for case in outliers if case["debug_status"] == "missing"),
            "dominant_stage_counts": dict(sorted(Counter(case["dominant_stage"] for case in outliers).items())),
        },
        "outliers": outliers,
    }


def collect_outlier_refs(
    item: dict[str, Any],
    rows_by_case: dict[str, dict[str, Any]],
    *,
    slow_threshold_ms: int,
    top_n: int,
) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows_by_case.values():
        if case_latency_ms(row) >= slow_threshold_ms:
            selected[str(row["case_name"])] = row
    for row in item.get("slowest_cases") or []:
        if not row.get("case_name"):
            continue
        if case_latency_ms(row) >= slow_threshold_ms:
            selected.setdefault(str(row["case_name"]), row)
    return sorted(selected.values(), key=case_latency_ms, reverse=True)[:top_n]


def audit_outlier_case(ref: dict[str, Any], full_row: dict[str, Any] | None) -> dict[str, Any]:
    row = {**ref, **(full_row or {})}
    debug = row.get("retrieval_debug") if isinstance(row.get("retrieval_debug"), dict) else {}
    stage_latencies = stage_latency_breakdown(debug)
    dominant_stage = stage_latencies[0]["stage"] if stage_latencies else "debug_missing"
    return {
        "case_name": row.get("case_name"),
        "question_preview": str(row.get("question") or "")[:160],
        "passed": row.get("passed"),
        "failure_mode": row.get("failure_mode"),
        "elapsed_seconds": row.get("elapsed_seconds"),
        "search_total_latency_ms": case_latency_ms(row) or None,
        "debug_status": "available" if debug else "missing",
        "dominant_stage": dominant_stage,
        "stage_latencies_ms": stage_latencies,
        "candidate_counts": candidate_count_breakdown(debug),
        "query_decomposition_applied": debug.get("query_decomposition_applied"),
        "subquery_count": debug.get("subquery_count"),
        "subquery_timeout_count": debug.get("subquery_timeout_count"),
        "action_hint": action_hint(dominant_stage, bool(debug)),
    }


def case_latency_ms(row: dict[str, Any]) -> int:
    debug = row.get("retrieval_debug") if isinstance(row.get("retrieval_debug"), dict) else {}
    for value in (row.get("search_total_latency_ms"), debug.get("search_total_latency_ms")):
        if value is not None:
            return int(float(value))
    elapsed = row.get("elapsed_seconds")
    if elapsed is not None:
        return int(float(elapsed) * 1000)
    return 0


def stage_latency_breakdown(debug: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, value in debug.items():
        if key in TOTAL_LATENCY_KEYS or not key.endswith("_latency_ms") or value is None:
            continue
        latency_ms = int(float(value))
        if latency_ms <= 0:
            continue
        rows.append({"stage": key.removesuffix("_latency_ms"), "latency_ms": latency_ms})
    return sorted(rows, key=lambda item: item["latency_ms"], reverse=True)


def candidate_count_breakdown(debug: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, value in debug.items():
        if not key.endswith("_candidate_count") or value is None:
            continue
        count = int(float(value))
        if count <= 0:
            continue
        rows.append({"stage": key.removesuffix("_candidate_count"), "count": count})
    return sorted(rows, key=lambda item: item["count"], reverse=True)[:10]


def action_hint(dominant_stage: str, debug_available: bool) -> str:
    if not debug_available:
        return "replay_target_case_with_full_retrieval_debug"
    if dominant_stage in {"lexical_retrieval", "indexed_sparse_retrieval"}:
        return "inspect_sparse_query_breadth_cjk_fallback_and_statement_timeout"
    if dominant_stage in {"subquery_document_evidence", "document_evidence_sweep"}:
        return "inspect_within_document_evidence_scan_fanout"
    if dominant_stage == "rerank":
        return "inspect_rerank_candidate_limit_or_provider_latency"
    return "inspect_stage_specific_latency"


def recommended_replays(debug_missing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    rows = []
    for case in debug_missing:
        case_name = str(case.get("case_name") or "")
        if not case_name:
            continue
        key = (case_name, str(case.get("dominant_stage") or ""))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "case_name": case_name,
                "search_total_latency_ms": case.get("search_total_latency_ms"),
                "reason": "slowest_case_missing_retrieval_debug",
                "suggested_command_scope": "--case-name " + case_name,
            }
        )
    return sorted(rows, key=lambda item: int(item.get("search_total_latency_ms") or 0), reverse=True)


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Retrieval Latency Outlier Audit",
        "",
        f"- Dataset: `{audit.get('dataset_name')}`",
        f"- Source report: `{audit.get('source_report')}`",
        f"- Slow threshold: `{audit.get('slow_threshold_ms')} ms`",
        "",
        "## Summary",
        "",
        "| Outliers | Debug available | Debug missing | Replay cases | Dominant stages |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    summary = audit["summary"]
    stage_counts = ", ".join(f"{name}:{count}" for name, count in summary.get("dominant_stage_counts", {}).items()) or "none"
    lines.append(
        f"| {summary['outlier_count']} | {summary['debug_available_count']} | "
        f"{summary['debug_missing_count']} | {summary['recommended_replay_case_count']} | {stage_counts} |"
    )
    for ablation in audit.get("ablations") or []:
        lines.extend(
            [
                "",
                f"## {ablation['ablation_name']}",
                "",
                "| Case | Search ms | Debug | Dominant stage | Top stage latencies | Action |",
                "| --- | ---: | --- | --- | --- | --- |",
            ]
        )
        for case in ablation.get("outliers") or []:
            top_stages = ", ".join(
                f"{item['stage']}:{item['latency_ms']}" for item in (case.get("stage_latencies_ms") or [])[:3]
            ) or "n/a"
            lines.append(
                f"| `{case.get('case_name')}` | {case.get('search_total_latency_ms')} | "
                f"{case.get('debug_status')} | `{case.get('dominant_stage')}` | {top_stages} | "
                f"{case.get('action_hint')} |"
            )
    replay_cases = audit.get("recommended_replay_cases") or []
    if replay_cases:
        lines.extend(["", "## Recommended Targeted Replays", "", "| Case | Search ms | Reason |", "| --- | ---: | --- |"])
        for case in replay_cases[:20]:
            lines.append(f"| `{case['case_name']}` | {case.get('search_total_latency_ms')} | {case.get('reason')} |")
    lines.append("")
    return "\n".join(lines)


def summary_text(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    stages = ", ".join(f"{name}={count}" for name, count in summary.get("dominant_stage_counts", {}).items()) or "none"
    return (
        f"outliers={summary['outlier_count']} debug_available={summary['debug_available_count']} "
        f"debug_missing={summary['debug_missing_count']} replay_cases={summary['recommended_replay_case_count']} "
        f"dominant_stages={stages}"
    )


if __name__ == "__main__":
    main()
