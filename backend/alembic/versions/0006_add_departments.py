"""add departments table and department_id columns

Revision ID: 0006_add_departments
Revises: 0005_eval_observability
Create Date: 2026-05-29 00:00:00.000000
"""

from __future__ import annotations

import uuid

import sqlalchemy as sa

from alembic import op

revision = "0006_add_departments"
down_revision = "0005_eval_observability"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 第一步：建 departments 表
    op.create_table(
        "departments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("parent_id", sa.Uuid(), nullable=True),
        sa.Column("path", sa.String(length=512), nullable=False),
        sa.Column("depth", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("uq_departments_path", "departments", ["path"], unique=True)
    op.create_index(
        "uq_departments_top_name",
        "departments",
        ["name"],
        unique=True,
        postgresql_where=sa.text("parent_id IS NULL"),
    )
    op.create_index(
        "uq_departments_sibling_name",
        "departments",
        ["parent_id", "name"],
        unique=True,
        postgresql_where=sa.text("parent_id IS NOT NULL"),
    )
    op.create_index("idx_departments_parent_id", "departments", ["parent_id"])

    op.create_foreign_key(
        "fk_departments_parent_id_departments",
        "departments",
        "departments",
        ["parent_id"],
        ["id"],
        ondelete="RESTRICT",
    )

    # 第二步：users 新增 department_id（不删 team_name）
    op.add_column("users", sa.Column("department_id", sa.Uuid(), nullable=True))
    op.create_index("idx_users_department_id", "users", ["department_id"])
    op.create_foreign_key(
        "fk_users_department_id_departments",
        "users",
        "departments",
        ["department_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # 第三步：document_acls 新增 department_id（不删 team_name）
    op.add_column("document_acls", sa.Column("department_id", sa.Uuid(), nullable=True))
    op.create_index("idx_document_acls_department_id", "document_acls", ["department_id"])
    op.create_foreign_key(
        "fk_document_acls_department_id_departments",
        "document_acls",
        "departments",
        ["department_id"],
        ["id"],
        ondelete="RESTRICT",
    )

    # 第四步：Python 侧数据回填
    _backfill_departments()


def _backfill_departments() -> None:
    """从 users.team_name 和 document_acls.team_name 取并集，创建部门并回填。"""
    bind = op.get_bind()

    # 取所有非空 team_name 的并集
    if bind.dialect.name == "sqlite":
        rows = bind.execute(
            sa.text(
                "SELECT DISTINCT name FROM ("
                "  SELECT trim(team_name) AS name FROM users WHERE team_name IS NOT NULL AND trim(team_name) != ''"
                "  UNION"
                "  SELECT trim(team_name) AS name FROM document_acls WHERE team_name IS NOT NULL AND trim(team_name) != ''"
                ")"
            )
        ).fetchall()
    else:
        rows = bind.execute(
            sa.text(
                "SELECT DISTINCT name FROM ("
                "  SELECT trim(team_name) AS name FROM users WHERE team_name IS NOT NULL AND trim(team_name) != ''"
                "  UNION"
                "  SELECT trim(team_name) AS name FROM document_acls WHERE team_name IS NOT NULL AND trim(team_name) != ''"
                ") AS combined"
            )
        ).fetchall()

    name_to_id: dict[str, str] = {}
    name_to_raws: dict[str, list[str]] = {}
    used_names: set[str] = set()
    for (raw_name,) in rows:
        # 规范化：strip，/ 替换为 _
        name = raw_name.strip().replace("/", "_")
        if not name:
            continue

        # 碰撞处理：如 tech/infra 和 tech_infra 同时存在，加后缀避免静默合并权限
        if name in used_names:
            counter = 2
            while f"{name}_{counter}" in used_names:
                counter += 1
            name = f"{name}_{counter}"
        used_names.add(name)

        # 记录原始名称 → 规范化名称的映射（回填时用原始名称匹配）
        name_to_raws.setdefault(name, []).append(raw_name.strip())

        dept_id = str(uuid.uuid4())
        name_to_id[name] = dept_id
        bind.execute(
            sa.text("INSERT INTO departments (id, name, path, depth) VALUES (:id, :name, :path, 0)"),
            {"id": dept_id, "name": name, "path": f"/{name}"},
        )

    # 回填 users.department_id（用原始 team_name 匹配）
    for name, dept_id in name_to_id.items():
        raw_names = name_to_raws.get(name, [name])
        for raw in set(raw_names):
            bind.execute(
                sa.text("UPDATE users SET department_id = :dept_id WHERE trim(team_name) = :raw"),
                {"dept_id": dept_id, "raw": raw},
            )

    # 回填 document_acls.department_id（用原始 team_name 匹配）
    for name, dept_id in name_to_id.items():
        raw_names = name_to_raws.get(name, [name])
        for raw in set(raw_names):
            bind.execute(
                sa.text("UPDATE document_acls SET department_id = :dept_id WHERE trim(team_name) = :raw"),
                {"dept_id": dept_id, "raw": raw},
            )


def downgrade() -> None:
    op.drop_constraint("fk_document_acls_department_id_departments", "document_acls", type_="foreignkey")
    op.drop_index("idx_document_acls_department_id", table_name="document_acls")
    op.drop_column("document_acls", "department_id")

    op.drop_constraint("fk_users_department_id_departments", "users", type_="foreignkey")
    op.drop_index("idx_users_department_id", table_name="users")
    op.drop_column("users", "department_id")

    op.drop_constraint("fk_departments_parent_id_departments", "departments", type_="foreignkey")
    op.drop_index("idx_departments_parent_id", table_name="departments")
    op.drop_index("uq_departments_sibling_name", table_name="departments")
    op.drop_index("uq_departments_top_name", table_name="departments")
    op.drop_index("uq_departments_path", table_name="departments")
    op.drop_table("departments")
