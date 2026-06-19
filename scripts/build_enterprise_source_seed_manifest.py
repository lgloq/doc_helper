from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = (
    ROOT_DIR / "backend" / "data" / "benchmark_raw" / "zh_enterprise" / "source_candidates_combined_v2.json"
)
DEFAULT_DOWNLOAD_REPORT = (
    ROOT_DIR / "backend" / "data" / "eval_outputs" / "zh-enterprise-source-download-combined-v2-local.json"
)
DEFAULT_QUALITY_REPORT = (
    ROOT_DIR / "backend" / "data" / "eval_outputs" / "zh-enterprise-source-file-quality-combined-v2-local.json"
)
DEFAULT_OUTPUT = ROOT_DIR / "backend" / "data" / "benchmark_raw" / "zh_enterprise" / "v1_seed_manifest.json"
DEFAULT_METADATA_OVERRIDES = (
    ROOT_DIR / "backend" / "data" / "benchmark_raw" / "zh_enterprise" / "source_metadata_overrides_v1.json"
)
MANIFEST_SCHEMA_VERSION = "zh-enterprise-benchmark-manifest-v1"
DEFAULT_ACCEPT_STATUSES = {"accepted_effect_long"}
PLACEHOLDER_ORG = "待下载后从文档首页确认"


def main() -> None:
    args = build_parser().parse_args()
    accept_statuses = set(args.accept_status)
    if args.include_review:
        accept_statuses.add("needs_case_review_short_but_usable")
    manifest = build_manifest(
        candidates_path=Path(args.candidates).resolve(),
        download_report_path=Path(args.download_report).resolve(),
        quality_report_path=Path(args.quality_report).resolve(),
        metadata_overrides_path=Path(args.metadata_overrides).resolve(),
        dataset_name=args.dataset_name,
        accept_statuses=accept_statuses,
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"dataset={manifest['dataset_name']} documents={len(manifest['documents'])} cases={len(manifest['cases'])}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a document-only seed manifest from accepted enterprise source files.")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--download-report", default=str(DEFAULT_DOWNLOAD_REPORT))
    parser.add_argument("--quality-report", default=str(DEFAULT_QUALITY_REPORT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--metadata-overrides", default=str(DEFAULT_METADATA_OVERRIDES))
    parser.add_argument("--dataset-name", default="zh_enterprise_v1_seed")
    parser.add_argument("--accept-status", action="append", default=sorted(DEFAULT_ACCEPT_STATUSES))
    parser.add_argument("--include-review", action="store_true")
    return parser


def build_manifest(
    *,
    candidates_path: Path,
    download_report_path: Path,
    quality_report_path: Path,
    dataset_name: str,
    accept_statuses: set[str],
    metadata_overrides_path: Path | None = DEFAULT_METADATA_OVERRIDES,
) -> dict[str, Any]:
    candidates = {
        str(item.get("candidate_id")): item
        for item in json.loads(candidates_path.read_text(encoding="utf-8")).get("candidate_sources", [])
    }
    downloads = {
        str(item.get("candidate_id")): item
        for item in json.loads(download_report_path.read_text(encoding="utf-8")).get("downloads", [])
        if item.get("status") == "downloaded"
    }
    metadata_overrides = load_metadata_overrides(metadata_overrides_path)
    quality_rows = [
        item
        for item in json.loads(quality_report_path.read_text(encoding="utf-8")).get("sources", [])
        if item.get("quality_status") in accept_statuses
    ]
    documents = []
    seen_titles: set[str] = set()
    for row in sorted(quality_rows, key=lambda item: str(item.get("candidate_id"))):
        candidate_id = str(row["candidate_id"])
        candidate = candidates.get(candidate_id) or {}
        download = downloads.get(candidate_id) or {}
        override = metadata_overrides.get(candidate_id) or {}
        document_id = candidate_id.replace("-", "_")
        source_org = resolved_value(
            override.get("source_org"),
            candidate.get("source_org"),
            row.get("source_platform"),
            default="unknown_source_org",
        )
        source_title = resolved_value(override.get("title"), row.get("title"), candidate.get("title"), default=candidate_id)
        title = f"{dataset_name}:{row.get('domain') or candidate.get('collection_id')}:{source_org}:{source_title}"
        if title in seen_titles:
            title = f"{title}:{candidate_id}"
        seen_titles.add(title)
        target_path = Path(str(row["target_path"]))
        documents.append(
            {
                "id": document_id,
                "title": title,
                "path": relative_to_source_root(target_path),
                "description": f"Downloaded official source candidate `{candidate_id}` for Chinese enterprise benchmark V1.",
                "status": "active",
                "acl": [{"principal_type": "public"}],
                "metadata": {
                    "source_url": row.get("source_url"),
                    "source_org": source_org,
                    "source_org_evidence": override.get("evidence"),
                    "metadata_override_applied": bool(override),
                    "language": "zh",
                    "file_sha256": row.get("actual_sha256") or row.get("expected_sha256"),
                    "retrieved_at": download.get("retrieved_at"),
                    "doc_type": row.get("doc_type"),
                    "domain": row.get("domain") or candidate.get("collection_id"),
                    "benchmark_role": candidate.get("benchmark_role") or "effect",
                    "source_platform": row.get("source_platform"),
                    "source_format": candidate.get("source_format"),
                    "source_candidate_id": candidate_id,
                    "quality_status": row.get("quality_status"),
                    "page_count": row.get("page_count"),
                    "cjk_chars": row.get("cjk_chars"),
                },
            }
        )
    return {
        "dataset_name": dataset_name,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "benchmark_version": "2026-06-08-source-seed",
        "description": "Document-only seed manifest for accepted downloaded Chinese enterprise V1 source candidates. Cases are intentionally empty until evidence design is complete.",
        "documents": documents,
        "cases": [],
    }


def load_metadata_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    overrides = payload.get("overrides", payload)
    return {str(candidate_id): dict(values or {}) for candidate_id, values in overrides.items()}


def resolved_value(*values: Any, default: str) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text != PLACEHOLDER_ORG:
            return text
    return default


def relative_to_source_root(path: Path) -> str:
    source_root = ROOT_DIR / "backend" / "data" / "benchmark_raw" / "zh_enterprise"
    try:
        return path.resolve().relative_to(source_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    main()
