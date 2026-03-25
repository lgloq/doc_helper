"""add chunks and ingestion metadata

Revision ID: 0002_add_chunks_and_ingestion_metadata
Revises: 0001_initial_schema
Create Date: 2026-03-24 00:30:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "0002_add_chunks_and_ingestion_metadata"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.add_column("document_versions", sa.Column("page_count", sa.Integer(), nullable=True))

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("section_title", sa.String(length=255), nullable=True),
        sa.Column("page_number_start", sa.Integer(), nullable=True),
        sa.Column("page_number_end", sa.Integer(), nullable=True),
        sa.Column("paragraph_start", sa.Integer(), nullable=True),
        sa.Column("paragraph_end", sa.Integer(), nullable=True),
        sa.Column("char_start", sa.Integer(), nullable=True),
        sa.Column("char_end", sa.Integer(), nullable=True),
        sa.Column("citation_metadata", sa.JSON(), nullable=True),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["document_version_id"], ["document_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_version_id", "chunk_index", name="uq_chunks_document_version_id_chunk_index"),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_document_version_id", "chunks", ["document_version_id"])
    if bind.dialect.name == "postgresql":
        op.execute("CREATE INDEX ix_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops)")
        op.execute("CREATE INDEX ix_chunks_content_fts ON chunks USING GIN (to_tsvector('simple', content))")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_chunks_content_fts")
        op.execute("DROP INDEX IF EXISTS ix_chunks_embedding")
    op.drop_index("ix_chunks_document_version_id", table_name="chunks")
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_column("document_versions", "page_count")
