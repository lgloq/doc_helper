"""add chunk lexical search text

Revision ID: 0010_chunk_lexical_search
Revises: 0009_chunk_structure
Create Date: 2026-06-03 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0010_chunk_lexical_search"
down_revision = "0009_chunk_structure"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("lexical_search_text", sa.Text(), nullable=True))
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            "CREATE INDEX ix_chunks_lexical_search_fts "
            "ON chunks USING GIN (to_tsvector('simple', coalesce(lexical_search_text, '')))"
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_chunks_lexical_search_fts")
    op.drop_column("chunks", "lexical_search_text")
