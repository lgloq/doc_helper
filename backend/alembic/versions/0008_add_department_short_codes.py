"""add department short codes

Revision ID: 0008_add_department_short_codes
Revises: 0007_add_department_id_path
Create Date: 2026-05-30 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0008_add_department_short_codes"
down_revision = "0007_add_department_id_path"
branch_labels = None
depends_on = None

ORG_CODE_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ORG_CODE_ROOT_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def upgrade() -> None:
    op.add_column("departments", sa.Column("stable_code", sa.String(length=5), nullable=True))
    op.add_column("departments", sa.Column("org_code", sa.String(length=64), nullable=True))
    op.add_column("departments", sa.Column("org_code_path", sa.String(length=1024), nullable=True))
    _backfill_short_codes()
    op.alter_column("departments", "stable_code", existing_type=sa.String(length=5), nullable=False)
    op.alter_column("departments", "org_code", existing_type=sa.String(length=64), nullable=False)
    op.alter_column("departments", "org_code_path", existing_type=sa.String(length=1024), nullable=False)
    op.create_index("uq_departments_stable_code", "departments", ["stable_code"], unique=True)
    op.create_index("uq_departments_org_code_path", "departments", ["org_code_path"], unique=True)
    op.create_index("uq_departments_parent_org_code", "departments", ["parent_id", "org_code"], unique=True)


def _backfill_short_codes() -> None:
    bind = op.get_bind()
    rows = bind.execute(sa.text("SELECT id, parent_id, depth FROM departments ORDER BY depth ASC, path ASC")).mappings()

    id_to_org_code: dict[str, str] = {}
    id_to_org_code_path: dict[str, str] = {}
    parent_to_used_suffixes: dict[str | None, set[str]] = {}

    for index, row in enumerate(rows, start=1):
        department_id = str(row["id"])
        parent_id = str(row["parent_id"]) if row["parent_id"] is not None else None
        stable_code = _stable_code(index)

        if parent_id:
            parent_org_code = id_to_org_code[parent_id]
            suffix = _next_suffix(parent_to_used_suffixes.setdefault(parent_id, set()))
            org_code = f"{parent_org_code}{suffix}"
            parent_org_code_path = id_to_org_code_path[parent_id]
            org_code_path = f"{parent_org_code_path}/{org_code}"
        else:
            org_code = _next_root_code(parent_to_used_suffixes.setdefault(None, set()))
            org_code_path = f"/{org_code}"

        id_to_org_code[department_id] = org_code
        id_to_org_code_path[department_id] = org_code_path
        bind.execute(
            sa.text(
                "UPDATE departments "
                "SET stable_code = :stable_code, org_code = :org_code, org_code_path = :org_code_path "
                "WHERE id = :department_id"
            ),
            {
                "stable_code": stable_code,
                "org_code": org_code,
                "org_code_path": org_code_path,
                "department_id": row["id"],
            },
        )


def _stable_code(index: int) -> str:
    return "S" + _to_base36(index).rjust(4, "0")[-4:]


def _to_base36(value: int) -> str:
    if value == 0:
        return "0"
    digits: list[str] = []
    while value:
        value, remainder = divmod(value, len(ORG_CODE_CHARS))
        digits.append(ORG_CODE_CHARS[remainder])
    return "".join(reversed(digits))


def _next_root_code(used_codes: set[str]) -> str:
    for letter in ORG_CODE_ROOT_LETTERS:
        if not any(code.startswith(letter) for code in used_codes):
            code = f"{letter}0"
            used_codes.add(code)
            return code

    for letter in ORG_CODE_ROOT_LETTERS:
        for suffix in ORG_CODE_CHARS:
            code = f"{letter}{suffix}"
            if code not in used_codes:
                used_codes.add(code)
                return code
    raise RuntimeError("Root department code space exhausted")


def _next_suffix(used_suffixes: set[str]) -> str:
    for suffix in ORG_CODE_CHARS:
        if suffix not in used_suffixes:
            used_suffixes.add(suffix)
            return suffix
    raise RuntimeError("Sibling department code space exhausted")


def downgrade() -> None:
    op.drop_index("uq_departments_parent_org_code", table_name="departments")
    op.drop_index("uq_departments_org_code_path", table_name="departments")
    op.drop_index("uq_departments_stable_code", table_name="departments")
    op.drop_column("departments", "org_code_path")
    op.drop_column("departments", "org_code")
    op.drop_column("departments", "stable_code")
