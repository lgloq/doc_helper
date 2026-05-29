from __future__ import annotations

from sqlalchemy import select

from app.models.enums import RoleName
from app.models.role import Role
from app.models.user import User
from app.services.auth.bootstrap import _ensure_users


def test_ensure_users_updates_existing_default_user_team(db_session) -> None:
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    manager_role = Role(name=RoleName.MANAGER, description="Manager")
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add_all([viewer_role, manager_role, admin_role])
    db_session.flush()

    viewer = User(
        email="viewer@local.test",
        full_name="Default Viewer",
        password_hash="hashed",
        team_name="platform",
        is_active=True,
        role_id=viewer_role.id,
    )
    db_session.add(viewer)
    db_session.flush()

    _ensure_users(
        db_session,
        {
            RoleName.VIEWER: viewer_role,
            RoleName.MANAGER: manager_role,
            RoleName.ADMIN: admin_role,
        },
        {},
    )

    assert viewer.team_name == "sales"


def test_ensure_users_creates_platform_viewer_default_user(db_session) -> None:
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    manager_role = Role(name=RoleName.MANAGER, description="Manager")
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add_all([viewer_role, manager_role, admin_role])
    db_session.flush()

    _ensure_users(
        db_session,
        {
            RoleName.VIEWER: viewer_role,
            RoleName.MANAGER: manager_role,
            RoleName.ADMIN: admin_role,
        },
        {},
    )
    db_session.flush()

    platform_viewer = db_session.scalar(select(User).where(User.email == "viewer2@local.test"))
    assert platform_viewer is not None
    assert platform_viewer.team_name == "platform"
    assert platform_viewer.role_id == viewer_role.id
