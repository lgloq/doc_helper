from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import func, select

from app.db.session import SessionLocal
from app.models.chunk import Chunk
from app.models.document import Document, DocumentVersion
from app.models.eval import EvalCase, EvalRun


DEFAULT_DATASETS = [
    "stard_zh_law_docs_small",
]


def main() -> None:
    args = build_parser().parse_args()
    datasets = args.dataset or DEFAULT_DATASETS
    report = build_report(datasets, include_format_coverage=args.include_format_coverage)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {output_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report imported benchmark dataset and latest eval summaries.")
    parser.add_argument("--dataset", action="append", help="Dataset name to include. Defaults to the real benchmark suite.")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--include-format-coverage", action="store_true", help="Include format_coverage:* ingestion rows.")
    return parser


def build_report(datasets: list[str], *, include_format_coverage: bool) -> dict[str, Any]:
    session = SessionLocal()
    try:
        report = {
            "datasets": [dataset_report(session, dataset) for dataset in datasets],
        }
        if include_format_coverage:
            report["format_coverage"] = format_coverage_report(session)
        return report
    finally:
        session.close()


def dataset_report(session, dataset_name: str) -> dict[str, Any]:
    case_count = session.scalar(select(func.count(EvalCase.id)).where(EvalCase.dataset_name == dataset_name)) or 0
    latest_run = session.scalar(
        select(EvalRun)
        .where(EvalRun.dataset_name == dataset_name)
        .order_by(EvalRun.created_at.desc())
        .limit(1)
    )
    return {
        "dataset_name": dataset_name,
        "eval_cases": int(case_count),
        "latest_run": serialize_run(latest_run) if latest_run else None,
        "latest_retrieval_report": latest_retrieval_report(dataset_name),
    }


def serialize_run(run: EvalRun) -> dict[str, Any]:
    return {
        "id": str(run.id),
        "status": run.status,
        "total_cases": run.total_cases,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "summary_json": run.summary_json,
        "error_text": run.error_text,
    }


def latest_retrieval_report(dataset_name: str) -> dict[str, Any] | None:
    output_dir = BACKEND_DIR / "data" / "eval_outputs"
    if not output_dir.exists():
        return None
    candidates = []
    dataset_slug = slugify_dataset_name(dataset_name)
    for path in output_dir.glob("*retrieval*.json"):
        payload = load_json_report(path)
        if not payload or payload.get("dataset_name") != dataset_name:
            continue
        if dataset_slug not in path.name and dataset_name not in path.name:
            continue
        candidates.append((path.stat().st_mtime, path, payload))
    if not candidates:
        return None
    _, path, payload = max(candidates, key=lambda item: item[0])
    return {
        "path": str(path),
        "generated_at": payload.get("generated_at"),
        "top_k": payload.get("top_k"),
        "case_count": payload.get("case_count"),
        "document_scope": payload.get("document_scope"),
        "summary": payload.get("summary"),
        "failure_case_count": len(payload.get("failure_cases") or []),
    }


def load_json_report(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def slugify_dataset_name(dataset_name: str) -> str:
    return dataset_name.replace("_", "-")


def format_coverage_report(session) -> list[dict[str, Any]]:
    rows = session.execute(
        select(
            Document.title,
            DocumentVersion.original_filename,
            DocumentVersion.mime_type,
            DocumentVersion.ingest_status,
            func.length(DocumentVersion.extracted_text).label("text_length"),
            func.count(Chunk.id).label("chunk_count"),
            DocumentVersion.page_count,
            DocumentVersion.ingest_error,
        )
        .join(DocumentVersion, DocumentVersion.id == Document.current_version_id)
        .outerjoin(Chunk, Chunk.document_version_id == DocumentVersion.id)
        .where(Document.title.like("format_coverage:%"))
        .group_by(
            Document.title,
            DocumentVersion.original_filename,
            DocumentVersion.mime_type,
            DocumentVersion.ingest_status,
            DocumentVersion.extracted_text,
            DocumentVersion.page_count,
            DocumentVersion.ingest_error,
        )
        .order_by(DocumentVersion.original_filename)
    ).all()
    return [
        {
            "title": row.title,
            "original_filename": row.original_filename,
            "mime_type": row.mime_type,
            "ingest_status": row.ingest_status.value if hasattr(row.ingest_status, "value") else str(row.ingest_status),
            "text_length": int(row.text_length or 0),
            "chunk_count": int(row.chunk_count or 0),
            "page_count": row.page_count,
            "ingest_error": row.ingest_error,
        }
        for row in rows
    ]


if __name__ == "__main__":
    main()
