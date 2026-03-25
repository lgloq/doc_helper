from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.document import Document, DocumentACL
from app.models.enums import DocumentStatus, PrincipalType, RoleName
from app.models.role import Role
from app.models.user import User
from app.services.permissions.service import PermissionFilterBuilder


def test_permission_filter_builder_respects_manage_scope(db_session: Session) -> None:
    manager_role = Role(name=RoleName.MANAGER, description="Manager")
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add_all([manager_role, admin_role])
    db_session.flush()

    manager = User(
        email="manager@example.com",
        full_name="Manager",
        password_hash=hash_password("manager-pass"),
        team_name="platform",
        is_active=True,
        role_id=manager_role.id,
    )
    admin = User(
        email="admin@example.com",
        full_name="Admin",
        password_hash=hash_password("admin-pass"),
        team_name=None,
        is_active=True,
        role_id=admin_role.id,
    )
    db_session.add_all([manager, admin])
    db_session.flush()

    team_view_doc = Document(title="Team View", description=None, status=DocumentStatus.ACTIVE, owner_user_id=admin.id)
    team_manage_doc = Document(title="Team Manage", description=None, status=DocumentStatus.ACTIVE, owner_user_id=admin.id)
    owned_doc = Document(title="Owned By Manager", description=None, status=DocumentStatus.ACTIVE, owner_user_id=manager.id)
    db_session.add_all([team_view_doc, team_manage_doc, owned_doc])
    db_session.flush()

    db_session.add_all(
        [
            DocumentACL(document_id=team_view_doc.id, principal_type=PrincipalType.TEAM, team_name="platform", can_view=True, can_manage=False),
            DocumentACL(document_id=team_manage_doc.id, principal_type=PrincipalType.TEAM, team_name="platform", can_view=True, can_manage=True),
        ]
    )
    db_session.commit()
    db_session.refresh(manager)
    manager.role = manager_role

    builder = PermissionFilterBuilder()
    visible_ids = set(db_session.scalars(builder.build_accessible_document_ids_query(manager, require_manage=False)).all())
    manageable_ids = set(db_session.scalars(builder.build_accessible_document_ids_query(manager, require_manage=True)).all())

    assert visible_ids == {team_view_doc.id, team_manage_doc.id, owned_doc.id}
    assert manageable_ids == {team_manage_doc.id, owned_doc.id}
