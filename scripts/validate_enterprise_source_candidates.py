from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = (
    ROOT_DIR / "backend" / "data" / "benchmark_raw" / "zh_enterprise" / "source_candidates_v1.json"
)
OFFICIAL_HOST_SUFFIXES = (
    "cninfo.com.cn",
    "static.cninfo.com.cn",
    "sse.com.cn",
    "szse.cn",
    "bse.cn",
    "hkexnews.hk",
    "sasac.gov.cn",
    "mof.gov.cn",
    "miit.gov.cn",
    "cac.gov.cn",
    "gov.cn",
    "nafmii.org.cn",
    "chinamoney.com.cn",
    "chinamoney.org.cn",
    "shclearing.com.cn",
    "bidding.csg.cn",
    "ecp.sgcc.com.cn",
    "bid.powerchina.cn",
    "bidding.epec.com",
    "b2b.10086.cn",
    "huawei.com",
    "tencent.com",
    "tencent.net.cn",
    "10086.cn",
)
REQUIRED_FIELDS = (
    "candidate_id",
    "collection_id",
    "title",
    "source_org",
    "source_url",
    "source_platform",
    "doc_type",
    "source_format",
    "retrieval_method",
    "stability_status",
    "benchmark_role",
    "selection_status",
    "expected_case_types",
    "risk_notes",
)
DIRECT_DOWNLOAD_HINTS = (".pdf", ".PDF", "download", "fileDownLoad", "download.jsp")


def main() -> None:
    args = build_parser().parse_args()
    report = build_report(
        Path(args.candidates).resolve(),
        min_candidates=args.min_candidates,
        min_collections=args.min_collections,
        min_direct_download=args.min_direct_download,
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
    if report["errors"] and not args.no_fail:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate executable source candidates for the enterprise benchmark.")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--output")
    parser.add_argument("--markdown-output")
    parser.add_argument("--no-fail", action="store_true")
    parser.add_argument("--min-candidates", type=int, default=30)
    parser.add_argument("--min-collections", type=int, default=5)
    parser.add_argument("--min-direct-download", type=int, default=15)
    return parser


def build_report(
    candidates_path: Path,
    *,
    min_candidates: int,
    min_collections: int,
    min_direct_download: int,
) -> dict[str, Any]:
    payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidates = list(payload.get("candidate_sources") or [])
    errors: list[str] = []
    warnings: list[str] = []
    collection_counts: Counter[str] = Counter()
    doc_type_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    url_counts: Counter[str] = Counter()
    official_count = 0
    direct_download_count = 0
    ready_to_download_count = 0

    for index, candidate in enumerate(candidates):
        candidate_id = str(candidate.get("candidate_id") or f"index-{index}")
        for field in REQUIRED_FIELDS:
            if field not in candidate or candidate.get(field) in (None, "", []):
                errors.append(f"candidate:{candidate_id}:missing_field:{field}")
        collection_id = str(candidate.get("collection_id") or "")
        doc_type = str(candidate.get("doc_type") or "")
        role = str(candidate.get("benchmark_role") or "")
        status = str(candidate.get("selection_status") or "")
        collection_counts[collection_id] += 1
        doc_type_counts[doc_type] += 1
        role_counts[role] += 1
        status_counts[status] += 1

        source_url = str(candidate.get("source_url") or "").strip()
        if source_url:
            url_counts[source_url] += 1
            if is_official_source_url(source_url):
                official_count += 1
            else:
                errors.append(f"candidate:{candidate_id}:source_url_not_official:{source_url}")
            if is_direct_download_url(source_url):
                direct_download_count += 1

        if (
            source_url
            and is_official_source_url(source_url)
            and is_direct_download_url(source_url)
            and status in {"screened", "accepted", "ready_to_download"}
        ):
            ready_to_download_count += 1
        if role == "effect" and status == "accepted":
            warnings.append(f"candidate:{candidate_id}:accepted_before_checksum_download")

    for url, count in url_counts.items():
        if count > 1:
            errors.append(f"duplicate_candidate_url:{url}")
    if len(candidates) < min_candidates:
        errors.append(f"candidate_count_below_min:{len(candidates)}<{min_candidates}")
    if len(collection_counts) < min_collections:
        errors.append(f"collection_count_below_min:{len(collection_counts)}<{min_collections}")
    if direct_download_count < min_direct_download:
        errors.append(f"direct_download_count_below_min:{direct_download_count}<{min_direct_download}")
    if ready_to_download_count < min_direct_download:
        errors.append(f"ready_to_download_count_below_min:{ready_to_download_count}<{min_direct_download}")

    return {
        "candidates_path": str(candidates_path),
        "name": payload.get("name"),
        "version": payload.get("version"),
        "thresholds": {
            "min_candidates": min_candidates,
            "min_collections": min_collections,
            "min_direct_download": min_direct_download,
        },
        "summary": {
            "candidate_count": len(candidates),
            "collection_count": len(collection_counts),
            "official_source_count": official_count,
            "direct_download_count": direct_download_count,
            "ready_to_download_count": ready_to_download_count,
            "collection_counts": dict(sorted(collection_counts.items())),
            "doc_type_counts": dict(sorted(doc_type_counts.items())),
            "role_counts": dict(sorted(role_counts.items())),
            "selection_status_counts": dict(sorted(status_counts.items())),
        },
        "errors": errors,
        "warnings": warnings,
        "passed": not errors,
    }


def is_official_source_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in OFFICIAL_HOST_SUFFIXES)


def is_direct_download_url(url: str) -> bool:
    return any(hint in url for hint in DIRECT_DOWNLOAD_HINTS)


def summary_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    return (
        f"source_candidates={report.get('name')} passed={report['passed']} "
        f"candidates={summary['candidate_count']} collections={summary['collection_count']} "
        f"official={summary['official_source_count']} direct_download={summary['direct_download_count']} "
        f"ready={summary['ready_to_download_count']} errors={len(report['errors'])}"
    )


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"# Enterprise Source Candidates: {report.get('name')}",
        "",
        f"- Version: `{report.get('version')}`",
        f"- Passed: `{str(report['passed']).lower()}`",
        f"- Candidates: `{summary['candidate_count']}`",
        f"- Collections: `{summary['collection_count']}`",
        f"- Official sources: `{summary['official_source_count']}`",
        f"- Direct downloads: `{summary['direct_download_count']}`",
        f"- Ready to download: `{summary['ready_to_download_count']}`",
        f"- Errors: `{len(report['errors'])}`",
        f"- Warnings: `{len(report['warnings'])}`",
        "",
        "## Collection Counts",
        "",
        "| Collection | Count |",
        "| --- | ---: |",
    ]
    for collection, count in summary["collection_counts"].items():
        lines.append(f"| `{collection}` | {count} |")
    lines.extend(["", "## Errors", ""])
    lines.extend(f"- `{item}`" for item in report["errors"][:100])
    if not report["errors"]:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- `{item}`" for item in report["warnings"][:100])
    if not report["warnings"]:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
