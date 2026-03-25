from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.user import User


class PermissionDecision:
    def __init__(self, can_view: bool, can_manage: bool):
        self.can_view = can_view
        self.can_manage = can_manage


class PermissionFilterBuilder:
    def build_accessible_document_ids_query(self, user: User, require_manage: bool = False):
        from uuid import UUID

        from sqlalchemy import Select, exists, literal, or_, select
        from sqlalchemy.sql.elements import ColumnElement

        from app.models.document import Document, DocumentACL
        from app.models.enums import PrincipalType, RoleName

        if user.role and user.role.name == RoleName.ADMIN:
            return select(Document.id)

        permission_clause = DocumentACL.can_manage.is_(True) if require_manage else or_(
            DocumentACL.can_view.is_(True),
            DocumentACL.can_manage.is_(True),
        )

        principal_matches: list[ColumnElement[bool]] = [DocumentACL.principal_type == PrincipalType.PUBLIC]
        principal_matches.append(
            (DocumentACL.principal_type == PrincipalType.USER) & (DocumentACL.user_id == user.id)
        )
        principal_matches.append(
            (DocumentACL.principal_type == PrincipalType.ROLE) & (DocumentACL.role_id == user.role_id)
        )
        if user.team_name:
            principal_matches.append(
                (DocumentACL.principal_type == PrincipalType.TEAM) & (DocumentACL.team_name == user.team_name)
            )

        acl_exists = exists(
            select(literal(1)).where(
                DocumentACL.document_id == Document.id,
                permission_clause,
                or_(*principal_matches),
            )
        )

        ownership_or_acl = or_(Document.owner_user_id == user.id, acl_exists)
        return select(Document.id).where(ownership_or_acl)

    def resolve_accessible_document_ids(self, session: Session, user: User, require_manage: bool = False) -> list:
        query = self.build_accessible_document_ids_query(user, require_manage=require_manage)
        return list(session.scalars(query).all())

    def build_document_id_filter(self, user: User, document_id_column, require_manage: bool = False):
        return document_id_column.in_(self.build_accessible_document_ids_query(user, require_manage=require_manage))

    def get_document_decision(self, user: User, document) -> PermissionDecision:
        from app.models.enums import PrincipalType, RoleName

        if user.role and user.role.name == RoleName.ADMIN:
            return PermissionDecision(can_view=True, can_manage=True)

        if document.owner_user_id == user.id:
            return PermissionDecision(can_view=True, can_manage=True)

        can_view = False
        can_manage = False
        for acl in document.acl_entries:
            principal_matches = (
                acl.principal_type == PrincipalType.PUBLIC
                or (acl.principal_type == PrincipalType.USER and acl.user_id == user.id)
                or (acl.principal_type == PrincipalType.ROLE and acl.role_id == user.role_id)
                or (acl.principal_type == PrincipalType.TEAM and user.team_name and acl.team_name == user.team_name)
            )
            if not principal_matches:
                continue
            can_view = can_view or acl.can_view or acl.can_manage
            can_manage = can_manage or acl.can_manage

        return PermissionDecision(can_view=can_view, can_manage=can_manage)
