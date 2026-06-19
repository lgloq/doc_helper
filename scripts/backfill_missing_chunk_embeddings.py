from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import func, select

from app.db.session import SessionLocal
from app.models.chunk import Chunk
from app.models.document import Document
from app.services.ingestion.embeddings import EmbeddingProviderFactory


DEFAULT_TITLE_PREFIX = "zh_enterprise_v1_seed:%"


def main() -> None:
    args = build_parser().parse_args()
    report = backfill_embeddings(
        title_prefix=args.title_prefix,
        manifest_path=Path(args.manifest).resolve() if args.manifest else None,
        batch_size=args.batch_size,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {output_path}")
    print(summary_text(report))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill missing embeddings for current-version chunks.")
    parser.add_argument("--title-prefix", default=DEFAULT_TITLE_PREFIX, help="SQL LIKE prefix for scoped documents.")
    parser.add_argument("--manifest", help="Optional manifest; scopes backfill to exact manifest document titles.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="Maximum missing chunks to update in this run.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser


def backfill_embeddings(
    *,
    title_prefix: str,
    manifest_path: Path | None,
    batch_size: int,
    limit: int | None,
    dry_run: bool,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    scoped_titles = load_manifest_titles(manifest_path) if manifest_path else None
    session = SessionLocal()
    try:
        missing_before = count_missing_chunks(session, title_prefix=title_prefix, scoped_titles=scoped_titles)
        if dry_run:
            return build_report(
                title_prefix=title_prefix,
                manifest_path=manifest_path,
                dry_run=True,
                missing_before=missing_before,
                updated_chunks=0,
                missing_after=missing_before,
            )

        provider = EmbeddingProviderFactory.create()
        updated_chunks = 0
        remaining_budget = limit
        while True:
            current_limit = batch_size if remaining_budget is None else min(batch_size, remaining_budget)
            if current_limit <= 0:
                break
            chunks = load_missing_chunk_batch(
                session,
                title_prefix=title_prefix,
                scoped_titles=scoped_titles,
                limit=current_limit,
            )
            if not chunks:
                break
            embeddings = provider.embed_texts([chunk.content for chunk in chunks])
            for index, chunk in enumerate(chunks):
                if index < len(embeddings):
                    chunk.embedding = embeddings[index]
            session.commit()
            updated_chunks += len(chunks)
            if remaining_budget is not None:
                remaining_budget -= len(chunks)

        missing_after = count_missing_chunks(session, title_prefix=title_prefix, scoped_titles=scoped_titles)
        return build_report(
            title_prefix=title_prefix,
            manifest_path=manifest_path,
            dry_run=False,
            missing_before=missing_before,
            updated_chunks=updated_chunks,
            missing_after=missing_after,
        )
    finally:
        session.close()


def load_manifest_titles(manifest_path: Path | None) -> list[str] | None:
    if manifest_path is None:
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [str(item["title"]) for item in payload.get("documents") or []]


def missing_chunk_statement(*, title_prefix: str, scoped_titles: list[str] | None):
    statement = (
        select(Chunk)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.document_version_id == Document.current_version_id)
        .where(Chunk.embedding.is_(None))
        .order_by(Document.title, Chunk.chunk_index)
    )
    if scoped_titles is not None:
        return statement.where(Document.title.in_(scoped_titles))
    return statement.where(Document.title.like(title_prefix))


def count_missing_chunks(session, *, title_prefix: str, scoped_titles: list[str] | None) -> int:
    subquery = missing_chunk_statement(title_prefix=title_prefix, scoped_titles=scoped_titles).subquery()
    return int(session.scalar(select(func.count()).select_from(subquery)) or 0)


def load_missing_chunk_batch(session, *, title_prefix: str, scoped_titles: list[str] | None, limit: int) -> list[Chunk]:
    statement = missing_chunk_statement(title_prefix=title_prefix, scoped_titles=scoped_titles).limit(limit)
    return list(session.scalars(statement).all())


def build_report(
    *,
    title_prefix: str,
    manifest_path: Path | None,
    dry_run: bool,
    missing_before: int,
    updated_chunks: int,
    missing_after: int,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "title_prefix": title_prefix,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "dry_run": dry_run,
        "missing_before": missing_before,
        "updated_chunks": updated_chunks,
        "missing_after": missing_after,
        "passed": missing_after == 0,
    }


def summary_text(report: dict[str, Any]) -> str:
    return (
        f"embedding_backfill passed={report['passed']} dry_run={report['dry_run']} "
        f"missing_before={report['missing_before']} updated={report['updated_chunks']} "
        f"missing_after={report['missing_after']}"
    )


if __name__ == "__main__":
    main()
