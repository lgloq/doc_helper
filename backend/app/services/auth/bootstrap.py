from __future__ import annotations

import logging
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import get_settings
from app.core.security import hash_password
from app.db.session import SessionLocal
from app.models.department import Department
from app.models.enums import RoleName
from app.models.role import Role
from app.models.user import User
from app.services.departments.codes import (
    build_org_code_path,
    generate_child_org_code,
    generate_root_org_code,
    generate_stable_code,
)

logger = logging.getLogger(__name__)

_DEPARTMENT_TREE: list[tuple[str, str | None, str]] = [
    # (path, parent_path, name)
    ("/sales", None, "sales"),
    ("/platform", None, "platform"),
    ("/测试总部", None, "测试总部"),
    ("/测试总部/测试技术部", "/测试总部", "测试技术部"),
    ("/测试总部/测试技术部/测试后端组", "/测试总部/测试技术部", "测试后端组"),
    ("/测试总部/测试技术部/测试前端组", "/测试总部/测试技术部", "测试前端组"),
    ("/测试总部/测试市场部", "/测试总部", "测试市场部"),
]

DEFAULT_USERS = [
    {
        "email": "viewer@local.test",
        "full_name": "Default Viewer",
        "password": "viewer123",
        "team_name": "sales",
        "department_path": "/sales",
        "role_name": RoleName.VIEWER,
    },
    {
        "email": "viewer2@local.test",
        "full_name": "Platform Viewer",
        "password": "viewer123",
        "team_name": "platform",
        "department_path": "/platform",
        "role_name": RoleName.VIEWER,
    },
    {
        "email": "manager@local.test",
        "full_name": "Default Manager",
        "password": "manager123",
        "team_name": "platform",
        "department_path": "/platform",
        "role_name": RoleName.MANAGER,
    },
    {
        "email": "admin@local.test",
        "full_name": "Default Admin",
        "password": "admin123",
        "team_name": None,
        "department_path": None,
        "role_name": RoleName.ADMIN,
    },
]


def seed_mock_data() -> None:
    settings = get_settings()
    if not settings.seed_mock_data:
        return

    session = SessionLocal()
    try:
        roles = _ensure_roles(session)
        departments = _ensure_departments(session)
        _ensure_users(session, roles, departments)
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        logger.exception("Unable to seed mock auth data.")
    finally:
        session.close()


def _ensure_roles(session) -> dict[RoleName, Role]:
    roles: dict[RoleName, Role] = {}
    for role_name in RoleName:
        role = session.scalar(select(Role).where(Role.name == role_name))
        if role is None:
            role = Role(name=role_name, description=f"Default {role_name.value} role")
            session.add(role)
            session.flush()
        roles[role_name] = role
    return roles


def _ensure_departments(session) -> dict[str, Department]:
    departments: dict[str, Department] = {}
    stable_codes = {code for code in session.scalars(select(Department.stable_code)).all() if code}
    for path, parent_path, name in _DEPARTMENT_TREE:
        existing = session.scalar(select(Department).where(Department.path == path))
        parent = departments.get(parent_path) if parent_path else None
        if existing:
            if not existing.id_path:
                existing.id_path = f"{parent.id_path}/{existing.id}" if parent else f"/{existing.id}"
            _ensure_department_codes(session, existing, parent, stable_codes)
            departments[path] = existing
            continue
        depth = parent.depth + 1 if parent else 0
        dept_id = uuid4()
        stable_code = generate_stable_code(stable_codes)
        stable_codes.add(stable_code)
        org_code = _generate_org_code_for_parent(session, parent)
        dept = Department(
            id=dept_id,
            name=name,
            parent_id=parent.id if parent else None,
            path=path,
            id_path=f"{parent.id_path}/{dept_id}" if parent else f"/{dept_id}",
            stable_code=stable_code,
            org_code=org_code,
            org_code_path=build_org_code_path(parent.org_code_path if parent else None, org_code),
            depth=depth,
        )
        session.add(dept)
        session.flush()
        departments[path] = dept
    return departments


def _ensure_department_codes(
    session,
    department: Department,
    parent: Department | None,
    stable_codes: set[str],
) -> None:
    if not department.stable_code:
        department.stable_code = generate_stable_code(stable_codes)
    stable_codes.add(department.stable_code)

    if not department.org_code:
        department.org_code = _generate_org_code_for_parent(session, parent)
    if not department.org_code_path:
        department.org_code_path = build_org_code_path(parent.org_code_path if parent else None, department.org_code)


def _generate_org_code_for_parent(session, parent: Department | None) -> str:
    if parent is None:
        existing_codes = {
            code
            for code in session.scalars(select(Department.org_code).where(Department.parent_id.is_(None))).all()
            if code
        }
        return generate_root_org_code(existing_codes)

    existing_codes = {
        code
        for code in session.scalars(select(Department.org_code).where(Department.parent_id == parent.id)).all()
        if code
    }
    return generate_child_org_code(parent.org_code, existing_codes)


def _ensure_users(session, roles: dict[RoleName, Role], departments: dict[str, Department]) -> None:
    for item in DEFAULT_USERS:
        existing = session.scalar(select(User).where(User.email == item["email"]))
        dept = departments.get(item["department_path"]) if item.get("department_path") else None
        if existing is not None:
            desired_role_id = roles[item["role_name"]].id
            existing.full_name = item["full_name"]
            existing.team_name = item["team_name"]
            existing.department_id = dept.id if dept else None
            existing.is_active = True
            existing.role_id = desired_role_id
            continue
        session.add(
            User(
                email=item["email"],
                full_name=item["full_name"],
                password_hash=hash_password(item["password"]),
                team_name=item["team_name"],
                department_id=dept.id if dept else None,
                is_active=True,
                role_id=roles[item["role_name"]].id,
            )
        )
