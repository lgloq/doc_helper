from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
BACKEND_DIR = ROOT_DIR / "backend"
for import_path in (SCRIPTS_DIR, BACKEND_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_retrieval_benchmark as benchmark
import run_retrieval_diagnostics as diagnostics


DEFAULT_MANIFEST = BACKEND_DIR / "data" / "benchmark_raw" / "zh_enterprise" / "v1_case_manifest_v1.json"
DEFAULT_OUTPUT = BACKEND_DIR / "data" / "eval_outputs" / "zh-enterprise-v1-case-retrieval-ablation-balanced-local.json"
DEFAULT_MARKDOWN_OUTPUT = BACKEND_DIR / "data" / "eval_outputs" / "zh-enterprise-v1-case-retrieval-ablation-balanced-local.md"
DEFAULT_ABLATIONS = "full_local,no_expansion,lexical_only,structural_only,lexical_structural"


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.case_name:
        selected_case_names = list(dict.fromkeys(args.case_name))
        sample_summary = {
            "selected_case_count": len(selected_case_names),
            "selected_case_type_counts": dict(Counter(case_type_from_name(name, manifest) for name in selected_case_names)),
            "selection": "case_name",
        }
    else:
        selected_case_names, sample_summary = select_balanced_case_names(
            manifest,
            total_limit=args.balanced_limit,
            per_type_limit=args.per_type_limit,
        )
    if not selected_case_names:
        raise SystemExit("No cases selected from manifest.")

    benchmark.apply_local_baseline_env(args.domain_profile)
    benchmark.apply_retrieval_ablation_overrides(
        cjk_python_fallback_mode=args.cjk_python_fallback_mode,
        cjk_python_scorer=args.cjk_python_scorer,
    )
    ablations = diagnostics.parse_ablations(args.ablations)

    reports = []
    for ablation in ablations:
        if args.progress:
            print(f"[ablation] start {ablation.name}", file=sys.stderr, flush=True)
        diagnostics.apply_ablation_env(ablation)
        report = benchmark.run_retrieval_benchmark(
            dataset_name=args.dataset,
            top_k=args.top_k,
            limit=None,
            case_names=selected_case_names,
            case_name_contains=None,
            document_title_prefix=args.document_title_prefix,
            manifest_scope=str(manifest_path),
            case_statement_timeout_ms=args.case_statement_timeout_ms,
            progress=args.progress,
        )
        if args.progress:
            print(f"[ablation] done {ablation.name}", file=sys.stderr, flush=True)
        reports.append(
            {
                "ablation": ablation.__dict__,
                "summary": report["summary"],
                "latency": report.get("latency") or report["summary"].get("latency_seconds", {}),
                "slowest_cases": report.get("slowest_cases", []),
                "document_scope": report.get("document_scope"),
                "failure_mode_counts": report["summary"].get("failure_mode_counts", {}),
                "case_count": report["case_count"],
                "failure_case_count": len(report["failure_cases"]),
                "failure_cases": report["failure_cases"][: args.failure_case_preview],
            }
        )

    output = {
        "dataset_name": args.dataset,
        "manifest": str(manifest_path),
        "generated_at": datetime.now(UTC).isoformat(),
        "top_k": args.top_k,
        "sample": sample_summary,
        "document_scope": reports[0].get("document_scope") if reports else None,
        "ablations": reports,
        "notes": [
            "Balanced sample is drawn from the case-bearing manifest by case_type.",
            "This runner reuses retrieval-only scoring and avoids per-stage diagnostics trace for speed.",
            "Use run_retrieval_diagnostics.py for deeper root-cause traces on selected failures.",
        ],
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown_output:
        markdown_path = Path(args.markdown_output).resolve()
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(output), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(summary_text(output))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run lightweight retrieval ablations on a balanced manifest sample.")
    parser.add_argument("--dataset", default="zh_enterprise_v1_seed")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--case-name",
        action="append",
        default=None,
        help="Exact eval case name to include. Can be repeated. Overrides balanced sampling.",
    )
    parser.add_argument("--balanced-limit", type=int, default=42)
    parser.add_argument("--per-type-limit", type=int, default=6)
    parser.add_argument("--ablations", default=DEFAULT_ABLATIONS)
    parser.add_argument("--document-title-prefix", default=None)
    parser.add_argument("--domain-profile", choices=["enterprise", "legal_benchmark"], default="enterprise")
    parser.add_argument("--cjk-python-fallback-mode", choices=["auto", "always", "off"], default=None)
    parser.add_argument("--cjk-python-scorer", choices=["weighted", "bm25"], default=None)
    parser.add_argument("--failure-case-preview", type=int, default=20)
    parser.add_argument("--case-statement-timeout-ms", type=int, default=None)
    parser.add_argument("--progress", action="store_true", help="Print ablation and per-case progress to stderr.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    return parser


def select_balanced_case_names(
    manifest: dict[str, Any],
    *,
    total_limit: int,
    per_type_limit: int,
) -> tuple[list[str], dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for case in manifest.get("cases", []):
        metadata = case.get("metadata") or {}
        case_type = str(metadata.get("case_type") or "unknown")
        if len(grouped[case_type]) >= per_type_limit:
            continue
        grouped[case_type].append(str(case["case_name"]))

    selected: list[str] = []
    indexes = Counter()
    case_types = sorted(grouped)
    while len(selected) < total_limit:
        advanced = False
        for case_type in case_types:
            index = indexes[case_type]
            if index >= len(grouped[case_type]):
                continue
            selected.append(grouped[case_type][index])
            indexes[case_type] += 1
            advanced = True
            if len(selected) >= total_limit:
                break
        if not advanced:
            break
    return selected, {
        "selected_case_count": len(selected),
        "selected_case_type_counts": dict(Counter(case_type_from_name(name, manifest) for name in selected)),
        "per_type_limit": per_type_limit,
        "balanced_limit": total_limit,
    }


def case_type_from_name(case_name: str, manifest: dict[str, Any]) -> str:
    for case in manifest.get("cases", []):
        if case.get("case_name") == case_name:
            return str((case.get("metadata") or {}).get("case_type") or "unknown")
    return "unknown"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Retrieval Ablation Benchmark",
        "",
        f"- Dataset: `{report['dataset_name']}`",
        f"- Manifest: `{report['manifest']}`",
        f"- Sample cases: `{report['sample']['selected_case_count']}`",
        f"- Top k: `{report['top_k']}`",
        "",
        "## Sample",
        "",
        "| Case type | Count |",
        "| --- | ---: |",
    ]
    for case_type, count in sorted(report["sample"]["selected_case_type_counts"].items()):
        lines.append(f"| `{case_type}` | {count} |")
    lines.extend(
        [
            "",
            "## Ablations",
            "",
            "| Ablation | Pass | Recall@k | Evidence Recall@k | Permission | P95 latency (s) | Max latency (s) | Failures |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in report["ablations"]:
        summary = item["summary"]
        latency = item.get("latency") or summary.get("latency_seconds") or {}
        failures = ", ".join(f"{key}:{value}" for key, value in sorted(item["failure_mode_counts"].items())) or "none"
        lines.append(
            f"| `{item['ablation']['name']}` | {summary.get('pass_rate')} | {summary.get('recall_at_k_avg')} | "
            f"{summary.get('evidence_recall_at_k_avg')} | {summary.get('permission_isolation_pass_rate')} | "
            f"{latency.get('p95')} | {latency.get('max')} | {failures} |"
        )
    for item in report["ablations"]:
        slowest = item.get("slowest_cases") or []
        if not slowest:
            continue
        lines.extend(["", f"## Slowest Cases: {item['ablation']['name']}", "", "| Case | Seconds | Status | Failure | Search latency ms |", "| --- | ---: | --- | --- | ---: |"])
        for case in slowest[:10]:
            status = "pass" if case.get("passed") else "fail"
            lines.append(
                f"| `{case.get('case_name')}` | {case.get('elapsed_seconds')} | {status} | "
                f"{case.get('failure_mode')} | {case.get('search_total_latency_ms')} |"
            )
    lines.append("")
    return "\n".join(lines)


def summary_text(report: dict[str, Any]) -> str:
    parts = [f"dataset={report['dataset_name']} sample_cases={report['sample']['selected_case_count']}"]
    for item in report["ablations"]:
        summary = item["summary"]
        latency = item.get("latency") or summary.get("latency_seconds") or {}
        parts.append(
            f"{item['ablation']['name']}:pass={summary.get('pass_rate')} "
            f"evidence={summary.get('evidence_recall_at_k_avg')} "
            f"p95={latency.get('p95')} max={latency.get('max')}"
        )
    return "\n".join(parts)


if __name__ == "__main__":
    main()
