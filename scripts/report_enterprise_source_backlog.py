from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BACKLOG = ROOT_DIR / "backend" / "data" / "benchmark_raw" / "zh_enterprise" / "source_backlog_v1.json"


def main() -> None:
    args = build_parser().parse_args()
    report = build_report(Path(args.backlog).resolve())
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
    parser = argparse.ArgumentParser(description="Summarize and sanity-check the enterprise benchmark source backlog.")
    parser.add_argument("--backlog", default=str(DEFAULT_BACKLOG))
    parser.add_argument("--output")
    parser.add_argument("--markdown-output")
    return parser


def build_report(backlog_path: Path) -> dict[str, Any]:
    backlog = json.loads(backlog_path.read_text(encoding="utf-8"))
    collections = list(backlog.get("source_collections") or [])
    target = dict(backlog.get("target") or {})
    target_documents = sum(int(item.get("target_documents") or 0) for item in collections if item.get("priority") != "P2")
    target_cases = sum(int(item.get("target_cases") or 0) for item in collections if item.get("priority") != "P2")
    priority_counts = Counter(str(item.get("priority") or "unknown") for item in collections)
    domain_counts: Counter[str] = Counter()
    query_count = 0
    for item in collections:
        domain_counts.update(str(domain) for domain in item.get("domains") or [])
        query_count += len(item.get("seed_queries") or [])

    checks = {
        "target_documents_met": target_documents >= int(target.get("documents_min") or 0),
        "target_cases_met": target_cases >= int(target.get("cases_min") or 0),
        "has_acceptance_rules": len(backlog.get("acceptance_rules") or []) >= 6,
        "has_seed_queries": query_count >= 15,
        "has_ablation_gate": bool((backlog.get("ablation_gate") or {}).get("required_ablations")),
        "has_case_mix": bool(backlog.get("case_mix")),
    }
    risks = []
    if not checks["target_documents_met"]:
        risks.append("target_documents_below_minimum")
    if not checks["target_cases_met"]:
        risks.append("target_cases_below_minimum")
    if domain_counts.get("governance", 0) + domain_counts.get("compliance", 0) > max(len(domain_counts), 1):
        risks.append("policy_governance_may_dominate")
    if "format_coverage_only" not in {item.get("id") for item in collections}:
        risks.append("format_coverage_not_separated")
    if not checks["has_ablation_gate"]:
        risks.append("missing_ablation_gate")

    return {
        "backlog_path": str(backlog_path),
        "name": backlog.get("name"),
        "version": backlog.get("version"),
        "target": target,
        "summary": {
            "collection_count": len(collections),
            "target_documents": target_documents,
            "target_cases": target_cases,
            "seed_query_count": query_count,
            "priority_counts": dict(sorted(priority_counts.items())),
            "domain_counts": dict(sorted(domain_counts.items())),
        },
        "checks": checks,
        "risks": risks,
        "collections": [
            {
                "id": item.get("id"),
                "priority": item.get("priority"),
                "target_documents": item.get("target_documents"),
                "target_cases": item.get("target_cases"),
                "source_type": item.get("source_type"),
                "seed_query_count": len(item.get("seed_queries") or []),
            }
            for item in collections
        ],
    }


def summary_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    risks = ",".join(report["risks"]) if report["risks"] else "none"
    return (
        f"source_backlog={report.get('name')} collections={summary['collection_count']} "
        f"target_documents={summary['target_documents']} target_cases={summary['target_cases']} "
        f"seed_queries={summary['seed_query_count']} risks={risks}"
    )


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"# Enterprise Benchmark Source Backlog: {report.get('name')}",
        "",
        f"- Version: `{report.get('version')}`",
        f"- Target documents: `{summary['target_documents']}`",
        f"- Target cases: `{summary['target_cases']}`",
        f"- Seed queries: `{summary['seed_query_count']}`",
        f"- Risks: `{', '.join(report['risks']) if report['risks'] else 'none'}`",
        "",
        "## Checks",
        "",
        "| Check | Passed |",
        "| --- | ---: |",
    ]
    for name, passed in report["checks"].items():
        lines.append(f"| `{name}` | `{str(passed).lower()}` |")
    lines.extend(
        [
            "",
            "## Collections",
            "",
            "| ID | Priority | Target docs | Target cases | Seed queries | Source type |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for item in report["collections"]:
        lines.append(
            f"| `{item['id']}` | `{item['priority']}` | {item['target_documents']} | "
            f"{item['target_cases']} | {item['seed_query_count']} | {item['source_type']} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
