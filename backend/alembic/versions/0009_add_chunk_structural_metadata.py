"""add chunk structural metadata

Revision ID: 0009_chunk_structure
Revises: 0008_add_department_short_codes
Create Date: 2026-06-02 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0009_chunk_structure"
down_revision = "0008_add_department_short_codes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("clause_full_name", sa.String(length=255), nullable=True))
    op.add_column("chunks", sa.Column("article_number", sa.String(length=64), nullable=True))
    op.add_column("chunks", sa.Column("chunk_type", sa.String(length=32), nullable=True))
    op.add_column("chunks", sa.Column("heading_path", sa.String(length=1024), nullable=True))
    op.add_column("chunks", sa.Column("structural_search_text", sa.Text(), nullable=True))
    op.create_index("ix_chunks_clause_full_name", "chunks", ["clause_full_name"])
    op.create_index("ix_chunks_article_number", "chunks", ["article_number"])
    op.create_index("ix_chunks_chunk_type", "chunks", ["chunk_type"])
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            "CREATE INDEX ix_chunks_structural_search_fts "
            "ON chunks USING GIN (to_tsvector('simple', coalesce(structural_search_text, '')))"
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_chunks_structural_search_fts")
    op.drop_index("ix_chunks_chunk_type", table_name="chunks")
    op.drop_index("ix_chunks_article_number", table_name="chunks")
    op.drop_index("ix_chunks_clause_full_name", table_name="chunks")
    op.drop_column("chunks", "structural_search_text")
    op.drop_column("chunks", "heading_path")
    op.drop_column("chunks", "chunk_type")
    op.drop_column("chunks", "article_number")
    op.drop_column("chunks", "clause_full_name")
