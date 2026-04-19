from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import get_settings
from app.core.security import hash_password
from app.db.session import SessionLocal
from app.models.enums import RoleName
from app.models.role import Role
from app.models.user import User

logger = logging.getLogger(__name__)

DEFAULT_USERS = [
    {
        "email": "viewer@local.test",
        "full_name": "Default Viewer",
        "password": "viewer123",
        "team_name": "sales",
        "role_name": RoleName.VIEWER,
    },
    {
        "email": "viewer2@local.test",
        "full_name": "Platform Viewer",
        "password": "viewer123",
        "team_name": "platform",
        "role_name": RoleName.VIEWER,
    },
    {
        "email": "manager@local.test",
        "full_name": "Default Manager",
        "password": "manager123",
        "team_name": "platform",
        "role_name": RoleName.MANAGER,
    },
    {
        "email": "admin@local.test",
        "full_name": "Default Admin",
        "password": "admin123",
        "team_name": None,
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
        _ensure_users(session, roles)
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


def _ensure_users(session, roles: dict[RoleName, Role]) -> None:
    for item in DEFAULT_USERS:
        existing = session.scalar(select(User).where(User.email == item["email"]))
        if existing is not None:
            desired_role_id = roles[item["role_name"]].id
            existing.full_name = item["full_name"]
            existing.team_name = item["team_name"]
            existing.is_active = True
            existing.role_id = desired_role_id
            continue
        session.add(
            User(
                email=item["email"],
                full_name=item["full_name"],
                password_hash=hash_password(item["password"]),
                team_name=item["team_name"],
                is_active=True,
                role_id=roles[item["role_name"]].id,
            )
        )
