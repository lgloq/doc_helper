"""add eval and observability tables

Revision ID: 0005_eval_observability
Revises: 0004_add_workflow_artifacts
Create Date: 2026-03-25 02:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0005_eval_observability"    # 0005_add_eval_and_observability
down_revision = "0004_add_workflow_artifacts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "eval_cases",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("dataset_name", sa.String(length=128), nullable=False),
        sa.Column("case_name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("acting_user_email", sa.String(length=320), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("expected_document_titles", sa.JSON(), nullable=False),
        sa.Column("forbidden_document_titles", sa.JSON(), nullable=False),
        sa.Column("expected_answer_keywords", sa.JSON(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("is_demo_case", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_eval_cases_dataset_name", "eval_cases", ["dataset_name"])
    op.create_index("ix_eval_cases_acting_user_email", "eval_cases", ["acting_user_email"])

    op.create_table(
        "eval_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("dataset_name", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("total_cases", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("summary_json", sa.JSON(), nullable=True),
        sa.Column("error_text", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_eval_runs_dataset_name", "eval_runs", ["dataset_name"])
    op.create_index("ix_eval_runs_status", "eval_runs", ["status"])

    op.create_table(
        "eval_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("case_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("acting_user_email", sa.String(length=320), nullable=False),
        sa.Column("retrieval_hit_rate", sa.Float(), nullable=False),
        sa.Column("citation_accuracy", sa.Float(), nullable=False),
        sa.Column("answer_faithfulness", sa.Float(), nullable=False),
        sa.Column("permission_isolation_correct", sa.Boolean(), nullable=False),
        sa.Column("overall_pass", sa.Boolean(), nullable=False),
        sa.Column("details_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["case_id"], ["eval_cases.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["eval_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_eval_results_run_id", "eval_results", ["run_id"])
    op.create_index("ix_eval_results_case_id", "eval_results", ["case_id"])

    op.create_table(
        "trace_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("trace_type", sa.String(length=64), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("user_message_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("assistant_message_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("query_text", sa.Text(), nullable=True),
        sa.Column("retrieved_chunks_json", sa.JSON(), nullable=False),
        sa.Column("selected_citations_json", sa.JSON(), nullable=False),
        sa.Column("model_name", sa.String(length=128), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("error_text", sa.Text(), nullable=True),
        sa.Column("trace_metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["assistant_message_id"], ["chat_messages.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_message_id"], ["chat_messages.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trace_logs_trace_type", "trace_logs", ["trace_type"])
    op.create_index("ix_trace_logs_user_id", "trace_logs", ["user_id"])
    op.create_index("ix_trace_logs_session_id", "trace_logs", ["session_id"])
    op.create_index("ix_trace_logs_user_message_id", "trace_logs", ["user_message_id"])
    op.create_index("ix_trace_logs_assistant_message_id", "trace_logs", ["assistant_message_id"])


def downgrade() -> None:
    op.drop_index("ix_trace_logs_assistant_message_id", table_name="trace_logs")
    op.drop_index("ix_trace_logs_user_message_id", table_name="trace_logs")
    op.drop_index("ix_trace_logs_session_id", table_name="trace_logs")
    op.drop_index("ix_trace_logs_user_id", table_name="trace_logs")
    op.drop_index("ix_trace_logs_trace_type", table_name="trace_logs")
    op.drop_table("trace_logs")

    op.drop_index("ix_eval_results_case_id", table_name="eval_results")
    op.drop_index("ix_eval_results_run_id", table_name="eval_results")
    op.drop_table("eval_results")

    op.drop_index("ix_eval_runs_status", table_name="eval_runs")
    op.drop_index("ix_eval_runs_dataset_name", table_name="eval_runs")
    op.drop_table("eval_runs")

    op.drop_index("ix_eval_cases_acting_user_email", table_name="eval_cases")
    op.drop_index("ix_eval_cases_dataset_name", table_name="eval_cases")
    op.drop_table("eval_cases")
