"""add operation jobs table

Revision ID: 0011_operation_jobs
Revises: 0010_chunk_lexical_search
Create Date: 2026-06-25 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0011_operation_jobs"
down_revision = "0010_chunk_lexical_search"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "operation_jobs",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("job_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default=sa.text("'queued'")),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("client_request_id", sa.String(length=80), nullable=True),
        sa.Column("arq_job_id", sa.String(length=128), nullable=True),
        sa.Column("resource_type", sa.String(length=64), nullable=True),
        sa.Column("resource_id", sa.String(length=64), nullable=True),
        sa.Column("request_payload", sa.JSON(), nullable=False),
        sa.Column("result_payload", sa.JSON(), nullable=True),
        sa.Column("error_text", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("running_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.UniqueConstraint("job_type", "user_id", "client_request_id", name="uq_operation_jobs_job_type_user_id_client_request_id"),
    )
    op.create_index("ix_operation_jobs_job_type_status_created_at", "operation_jobs", ["job_type", "status", "created_at"], unique=False)
    op.create_index("ix_operation_jobs_user_id_status_created_at", "operation_jobs", ["user_id", "status", "created_at"], unique=False)
    op.create_index("ix_operation_jobs_client_request_id", "operation_jobs", ["client_request_id"], unique=False)
    op.create_index("ix_operation_jobs_arq_job_id", "operation_jobs", ["arq_job_id"], unique=False)
    op.create_index("ix_operation_jobs_resource_type", "operation_jobs", ["resource_type"], unique=False)
    op.create_index("ix_operation_jobs_resource_id", "operation_jobs", ["resource_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_operation_jobs_resource_id", table_name="operation_jobs")
    op.drop_index("ix_operation_jobs_resource_type", table_name="operation_jobs")
    op.drop_index("ix_operation_jobs_arq_job_id", table_name="operation_jobs")
    op.drop_index("ix_operation_jobs_client_request_id", table_name="operation_jobs")
    op.drop_index("ix_operation_jobs_user_id_status_created_at", table_name="operation_jobs")
    op.drop_index("ix_operation_jobs_job_type_status_created_at", table_name="operation_jobs")
    op.drop_table("operation_jobs")
