"""add stable department id_path

Revision ID: 0007_add_department_id_path
Revises: 0006_add_departments
Create Date: 2026-05-29 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0007_add_department_id_path"
down_revision = "0006_add_departments"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("departments", sa.Column("id_path", sa.String(length=2048), nullable=True))
    _backfill_id_path()
    op.alter_column("departments", "id_path", existing_type=sa.String(length=2048), nullable=False)
    op.create_index("uq_departments_id_path", "departments", ["id_path"], unique=True)


def _backfill_id_path() -> None:
    bind = op.get_bind()
    rows = bind.execute(sa.text("SELECT id, parent_id, depth FROM departments ORDER BY depth ASC, path ASC")).mappings()

    id_to_path: dict[str, str] = {}
    for row in rows:
        department_id = str(row["id"])
        parent_id = str(row["parent_id"]) if row["parent_id"] is not None else None
        parent_path = id_to_path.get(parent_id) if parent_id else None
        id_path = f"{parent_path}/{department_id}" if parent_path else f"/{department_id}"
        id_to_path[department_id] = id_path
        bind.execute(
            sa.text("UPDATE departments SET id_path = :id_path WHERE id = :department_id"),
            {"id_path": id_path, "department_id": row["id"]},
        )


def downgrade() -> None:
    op.drop_index("uq_departments_id_path", table_name="departments")
    op.drop_column("departments", "id_path")
