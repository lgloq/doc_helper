from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import select

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.db.session import SessionLocal
from app.models.chunk import Chunk
from app.models.document import Document, DocumentVersion
from app.services.ingestion.search_index import build_lexical_search_text


def main() -> None:
    args = build_parser().parse_args()
    updated = backfill_lexical_search_text(
        document_title_prefix=args.document_title_prefix,
        batch_size=args.batch_size,
        only_missing=not args.rebuild_all,
    )
    print(f"updated_chunks={updated}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill persisted weighted lexical search text for chunks.")
    parser.add_argument("--document-title-prefix", default=None)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--rebuild-all", action="store_true")
    return parser


def backfill_lexical_search_text(
    *,
    document_title_prefix: str | None,
    batch_size: int,
    only_missing: bool,
) -> int:
    session = SessionLocal()
    try:
        statement = (
            select(Chunk, Document.title)
            .join(Document, Document.id == Chunk.document_id)
            .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
            .where(Document.current_version_id == Chunk.document_version_id)
            .order_by(Document.title.asc(), Chunk.chunk_index.asc())
        )
        if document_title_prefix:
            statement = statement.where(Document.title.like(f"{document_title_prefix}%"))
        if only_missing:
            statement = statement.where(Chunk.lexical_search_text.is_(None))

        updated = 0
        for chunk, document_title in session.execute(statement).yield_per(batch_size):
            chunk.lexical_search_text = build_lexical_search_text(
                document_title=document_title,
                section_title=chunk.section_title,
                clause_full_name=chunk.clause_full_name,
                article_number=chunk.article_number,
                heading_path=chunk.heading_path,
                structural_search_text=chunk.structural_search_text,
                content=chunk.content,
            )
            updated += 1
            if updated % batch_size == 0:
                session.commit()
        session.commit()
        return updated
    finally:
        session.close()


if __name__ == "__main__":
    main()
