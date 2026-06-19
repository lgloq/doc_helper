from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LONG_MANIFEST = ROOT_DIR / "backend" / "data" / "benchmark_raw" / "zh_enterprise" / "v1_seed_manifest.json"
DEFAULT_REVIEW_MANIFEST = (
    ROOT_DIR / "backend" / "data" / "benchmark_raw" / "zh_enterprise" / "v1_seed_manifest_with_review.json"
)
DEFAULT_INGESTION_REPORT = (
    ROOT_DIR
    / "backend"
    / "data"
    / "eval_outputs"
    / "zh-enterprise-v1-seed-with-review-ingestion-quality-require-embeddings-local.json"
)
DEFAULT_OUTPUT = ROOT_DIR / "backend" / "data" / "eval_outputs" / "zh-enterprise-review-promotion-local.json"
DEFAULT_PROMOTED_MANIFEST = ROOT_DIR / "backend" / "data" / "benchmark_raw" / "zh_enterprise" / "v1_seed_manifest_promoted_review.json"
PLACEHOLDER_ORG = "待下载后从文档首页确认"
MIN_PROMOTION_CJK = 4000
MIN_PROMOTION_CHUNKS = 8


def main() -> None:
    args = build_parser().parse_args()
    report, promoted_manifest = build_review_promotion_report(
        long_manifest_path=Path(args.long_manifest).resolve(),
        review_manifest_path=Path(args.review_manifest).resolve(),
        ingestion_report_path=Path(args.ingestion_report).resolve(),
        promoted_dataset_name=args.promoted_dataset_name,
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    if args.markdown_output:
        markdown_path = Path(args.markdown_output).resolve()
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        print(f"Wrote {markdown_path}")
    if args.promoted_manifest:
        manifest_path = Path(args.promoted_manifest).resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(promoted_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {manifest_path}")
    print(summary_text(report))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Promote review-quality enterprise source documents into a pilot manifest.")
    parser.add_argument("--long-manifest", default=str(DEFAULT_LONG_MANIFEST))
    parser.add_argument("--review-manifest", default=str(DEFAULT_REVIEW_MANIFEST))
    parser.add_argument("--ingestion-report", default=str(DEFAULT_INGESTION_REPORT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--markdown-output")
    parser.add_argument("--promoted-manifest", default=str(DEFAULT_PROMOTED_MANIFEST))
    parser.add_argument(
        "--promoted-dataset-name",
        default=None,
        help="Optional dataset_name/title prefix rewrite. Defaults to the review manifest dataset name so imported DB documents can be reused.",
    )
    return parser


def build_review_promotion_report(
    *,
    long_manifest_path: Path,
    review_manifest_path: Path,
    ingestion_report_path: Path,
    promoted_dataset_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    long_manifest = json.loads(long_manifest_path.read_text(encoding="utf-8"))
    review_manifest = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    ingestion_report = json.loads(ingestion_report_path.read_text(encoding="utf-8"))
    promoted_dataset_name = promoted_dataset_name or str(review_manifest.get("dataset_name") or "zh_enterprise_v1_seed")
    long_ids = {document["id"] for document in long_manifest.get("documents", [])}
    ingestion_by_id = {item.get("manifest_id"): item for item in ingestion_report.get("documents", [])}
    decisions = []

    promoted_documents = []
    for document in review_manifest.get("documents", []):
        if document["id"] in long_ids:
            promoted_documents.append(with_dataset_name(document, promoted_dataset_name))
            continue
        ingestion = ingestion_by_id.get(document["id"]) or {}
        decision = classify_review_document(document, ingestion)
        decisions.append(decision)
        if decision["decision"] == "promote_to_pilot_effect":
            promoted_documents.append(with_dataset_name(mark_promoted(document, decision), promoted_dataset_name))

    summary = build_summary(long_manifest, review_manifest, decisions, promoted_documents)
    report = {
        "long_manifest_path": str(long_manifest_path),
        "review_manifest_path": str(review_manifest_path),
        "ingestion_report_path": str(ingestion_report_path),
        "criteria": {
            "min_promotion_cjk": MIN_PROMOTION_CJK,
            "min_promotion_chunks": MIN_PROMOTION_CHUNKS,
            "reject_source_org_placeholder": True,
            "reject_title_truncated": True,
            "requires_ingestion_passed": True,
        },
        "summary": summary,
        "decisions": decisions,
        "passed": summary["promoted_pilot_document_count"] > summary["long_document_count"],
    }
    promoted_manifest = {
        "dataset_name": promoted_dataset_name,
        "schema_version": review_manifest.get("schema_version"),
        "benchmark_version": "2026-06-08-source-seed-promoted-review",
        "description": (
            "Document-only pilot manifest containing accepted long documents plus review documents that passed "
            "source/ingestion promotion checks. Cases are intentionally empty until evidence design is complete."
        ),
        "documents": promoted_documents,
        "cases": [],
    }
    return report, promoted_manifest


def classify_review_document(document: dict[str, Any], ingestion: dict[str, Any]) -> dict[str, Any]:
    metadata = document.get("metadata") or {}
    flags = promotion_flags(document, ingestion)
    blocking_flags = {"source_org_placeholder", "title_truncated", "ingestion_gate_failed", "few_chunks", "noise"}
    decision = "keep_review" if set(flags) & blocking_flags else "promote_to_pilot_effect"
    return {
        "document_id": document.get("id"),
        "title": document.get("title"),
        "domain": metadata.get("domain"),
        "doc_type": metadata.get("doc_type"),
        "source_format": metadata.get("source_format"),
        "quality_status": metadata.get("quality_status"),
        "source_org": metadata.get("source_org"),
        "cjk_chars": metadata.get("cjk_chars"),
        "page_count": metadata.get("page_count"),
        "chunk_count": ingestion.get("chunk_count"),
        "table_signal_chunk_count": ingestion.get("table_signal_chunk_count"),
        "ingestion_passed": ingestion.get("passed"),
        "flags": flags,
        "decision": decision,
    }


def promotion_flags(document: dict[str, Any], ingestion: dict[str, Any]) -> list[str]:
    metadata = document.get("metadata") or {}
    flags = []
    title = str(document.get("title") or "")
    source_org = str(metadata.get("source_org") or "")
    cjk_chars = int(metadata.get("cjk_chars") or 0)
    chunk_count = int(ingestion.get("chunk_count") or 0)
    if source_org == PLACEHOLDER_ORG or PLACEHOLDER_ORG in title:
        flags.append("source_org_placeholder")
    if "..." in title or "…" in title:
        flags.append("title_truncated")
    if metadata.get("source_format") in {"html", "html_with_pdf_attachment"}:
        flags.append("html_source")
    if cjk_chars < MIN_PROMOTION_CJK:
        flags.append("below_4k_cjk")
    if chunk_count < MIN_PROMOTION_CHUNKS:
        flags.append("few_chunks")
    if not ingestion.get("passed"):
        flags.append("ingestion_gate_failed")
    if int(ingestion.get("strict_noise_count") or 0) > 0 or int(ingestion.get("noisy_chunk_count") or 0) > 0:
        flags.append("noise")
    return flags


def mark_promoted(document: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    promoted = json.loads(json.dumps(document, ensure_ascii=False))
    metadata = promoted.setdefault("metadata", {})
    metadata["quality_status"] = "promoted_review_pilot"
    metadata["promotion_decision"] = decision["decision"]
    metadata["promotion_flags"] = decision["flags"]
    metadata["promotion_note"] = (
        "Promoted from review seed for pilot effect use after passing backend ingestion and metadata checks. "
        "This is not an accepted-long V1 completion signal."
    )
    return promoted


def with_dataset_name(document: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    cloned = json.loads(json.dumps(document, ensure_ascii=False))
    title = str(cloned.get("title") or "")
    original_dataset = title.split(":", 1)[0] if ":" in title else None
    if original_dataset and original_dataset != dataset_name:
        cloned["title"] = f"{dataset_name}:{title.split(':', 1)[1]}"
    return cloned


def build_summary(
    long_manifest: dict[str, Any],
    review_manifest: dict[str, Any],
    decisions: list[dict[str, Any]],
    promoted_documents: list[dict[str, Any]],
) -> dict[str, Any]:
    decision_counts = Counter(item["decision"] for item in decisions)
    domain_counts = Counter(item.get("domain") for item in decisions if item["decision"] == "promote_to_pilot_effect")
    flag_counts = Counter(flag for item in decisions for flag in item["flags"])
    return {
        "long_document_count": len(long_manifest.get("documents") or []),
        "review_total_document_count": len(review_manifest.get("documents") or []),
        "review_only_document_count": len(decisions),
        "promoted_review_document_count": decision_counts["promote_to_pilot_effect"],
        "kept_review_document_count": decision_counts["keep_review"],
        "promoted_pilot_document_count": len(promoted_documents),
        "promoted_review_domain_counts": dict(sorted(domain_counts.items())),
        "flag_counts": dict(sorted(flag_counts.items())),
        "remaining_to_v1_min_80": max(0, 80 - len(promoted_documents)),
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Enterprise Review Promotion",
        "",
        f"- Long seed documents: `{summary['long_document_count']}`",
        f"- Review-only documents: `{summary['review_only_document_count']}`",
        f"- Promoted review documents: `{summary['promoted_review_document_count']}`",
        f"- Kept for review: `{summary['kept_review_document_count']}`",
        f"- Promoted pilot manifest documents: `{summary['promoted_pilot_document_count']}`",
        f"- Remaining to 80-document V1 floor: `{summary['remaining_to_v1_min_80']}`",
        "",
        "## Decisions",
        "",
        "| Decision | Document | Domain | CJK | Chunks | Flags |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for decision in report["decisions"]:
        flags = ", ".join(decision["flags"]) if decision["flags"] else "none"
        lines.append(
            f"| `{decision['decision']}` | `{decision['document_id']}` | `{decision.get('domain')}` | "
            f"{decision.get('cjk_chars') or ''} | {decision.get('chunk_count') or ''} | {flags} |"
        )
    lines.append("")
    return "\n".join(lines)


def summary_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    return (
        f"review_promotion passed={report['passed']} long={summary['long_document_count']} "
        f"review_only={summary['review_only_document_count']} promoted={summary['promoted_review_document_count']} "
        f"kept_review={summary['kept_review_document_count']} pilot_docs={summary['promoted_pilot_document_count']} "
        f"remaining_to_80={summary['remaining_to_v1_min_80']}"
    )


if __name__ == "__main__":
    main()
