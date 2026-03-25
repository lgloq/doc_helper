"""add workflow artifacts

Revision ID: 0004_add_workflow_artifacts
Revises: 0003_add_message_citations
Create Date: 2026-03-25 01:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0004_add_workflow_artifacts"
down_revision = "0003_add_message_citations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "task_items",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("source_session_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("source_message_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("owner_name", sa.String(length=255), nullable=True),
        sa.Column("priority", sa.String(length=32), nullable=False, server_default="medium"),
        sa.Column("due_date", sa.Date(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="open"),
        sa.Column("source_citations", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["source_message_id"], ["chat_messages.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["source_session_id"], ["chat_sessions.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_task_items_created_by_user_id", "task_items", ["created_by_user_id"])
    op.create_index("ix_task_items_source_session_id", "task_items", ["source_session_id"])
    op.create_index("ix_task_items_source_message_id", "task_items", ["source_message_id"])

    op.create_table(
        "weekly_report_drafts",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("source_session_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("completed_this_week", sa.JSON(), nullable=False),
        sa.Column("risks_blockers", sa.JSON(), nullable=False),
        sa.Column("next_week_plan", sa.JSON(), nullable=False),
        sa.Column("reference_sources", sa.JSON(), nullable=False),
        sa.Column("source_message_ids", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="draft"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["source_session_id"], ["chat_sessions.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_weekly_report_drafts_created_by_user_id", "weekly_report_drafts", ["created_by_user_id"])
    op.create_index("ix_weekly_report_drafts_source_session_id", "weekly_report_drafts", ["source_session_id"])

    op.create_table(
        "faq_entries",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("source_session_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("source_message_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("quality", sa.String(length=32), nullable=False, server_default="medium"),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="draft"),
        sa.Column("source_citations", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["source_message_id"], ["chat_messages.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["source_session_id"], ["chat_sessions.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_faq_entries_created_by_user_id", "faq_entries", ["created_by_user_id"])
    op.create_index("ix_faq_entries_source_session_id", "faq_entries", ["source_session_id"])
    op.create_index("ix_faq_entries_source_message_id", "faq_entries", ["source_message_id"])


def downgrade() -> None:
    op.drop_index("ix_faq_entries_source_message_id", table_name="faq_entries")
    op.drop_index("ix_faq_entries_source_session_id", table_name="faq_entries")
    op.drop_index("ix_faq_entries_created_by_user_id", table_name="faq_entries")
    op.drop_table("faq_entries")

    op.drop_index("ix_weekly_report_drafts_source_session_id", table_name="weekly_report_drafts")
    op.drop_index("ix_weekly_report_drafts_created_by_user_id", table_name="weekly_report_drafts")
    op.drop_table("weekly_report_drafts")

    op.drop_index("ix_task_items_source_message_id", table_name="task_items")
    op.drop_index("ix_task_items_source_session_id", table_name="task_items")
    op.drop_index("ix_task_items_created_by_user_id", table_name="task_items")
    op.drop_table("task_items")
