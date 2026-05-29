from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from app.models.user import User


class PermissionDecision:
    def __init__(self, can_view: bool, can_manage: bool):
        self.can_view = can_view
        self.can_manage = can_manage


class PermissionFilterBuilder:
    def build_accessible_document_ids_query(self, session: Session, user: User, require_manage: bool = False):
        from sqlalchemy import exists, literal, or_, select
        from sqlalchemy.sql.elements import ColumnElement

        from app.models.document import Document, DocumentACL
        from app.models.enums import PrincipalType, RoleName

        if user.role and user.role.name == RoleName.ADMIN:
            return select(Document.id)

        permission_clause = (
            DocumentACL.can_manage.is_(True)
            if require_manage
            else or_(
                DocumentACL.can_view.is_(True),
                DocumentACL.can_manage.is_(True),
            )
        )

        ancestor_ids, legacy_team_name = self._resolve_user_dept_context(session, user)

        principal_matches: list[ColumnElement[bool]] = [DocumentACL.principal_type == PrincipalType.PUBLIC]
        principal_matches.append((DocumentACL.principal_type == PrincipalType.USER) & (DocumentACL.user_id == user.id))
        principal_matches.append(
            (DocumentACL.principal_type == PrincipalType.ROLE) & (DocumentACL.role_id == user.role_id)
        )

        # 部门层级继承（新 department_id）+ 旧 team_name fallback
        team_conditions: list[ColumnElement[bool]] = []
        if ancestor_ids:
            team_conditions.append(
                (DocumentACL.principal_type == PrincipalType.TEAM) & DocumentACL.department_id.in_(ancestor_ids)
            )
        if legacy_team_name:
            team_conditions.append(
                (DocumentACL.principal_type == PrincipalType.TEAM) & (DocumentACL.team_name == legacy_team_name)
            )
        if team_conditions:
            principal_matches.append(or_(*team_conditions))

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
        query = self.build_accessible_document_ids_query(session, user, require_manage=require_manage)
        return list(session.scalars(query).all())

    def build_document_id_filter(self, session: Session, user: User, document_id_column, require_manage: bool = False):
        return document_id_column.in_(
            self.build_accessible_document_ids_query(session, user, require_manage=require_manage)
        )

    def get_document_decision(self, session: Session, user: User, document) -> PermissionDecision:
        from app.models.enums import PrincipalType, RoleName

        if user.role and user.role.name == RoleName.ADMIN:
            return PermissionDecision(can_view=True, can_manage=True)

        if document.owner_user_id == user.id:
            return PermissionDecision(can_view=True, can_manage=True)

        ancestor_ids, legacy_team_name = self._resolve_user_dept_context(session, user)

        can_view = False
        can_manage = False
        for acl in document.acl_entries:
            principal_matches = acl.principal_type == PrincipalType.PUBLIC
            if not principal_matches:
                principal_matches = (acl.principal_type == PrincipalType.USER and acl.user_id == user.id) or (
                    acl.principal_type == PrincipalType.ROLE and acl.role_id == user.role_id
                )
            if (
                not principal_matches
                and acl.principal_type == PrincipalType.TEAM
                and (
                    (acl.department_id and acl.department_id in ancestor_ids)
                    or (acl.team_name and legacy_team_name and acl.team_name == legacy_team_name)
                )
            ):
                principal_matches = True
            if not principal_matches:
                continue
            can_view = can_view or acl.can_view or acl.can_manage
            can_manage = can_manage or acl.can_manage

        return PermissionDecision(can_view=can_view, can_manage=can_manage)

    @staticmethod
    def _resolve_user_dept_context(session: Session, user: User) -> tuple[set[UUID], str | None]:
        """一次性解析用户部门祖先 ID 集合和旧 team_name。"""
        from app.models.department import Department
        from app.repositories.department_repository import DepartmentRepository

        ancestor_ids: set[UUID] = set()
        if user.department_id:
            user_dept = session.get(Department, user.department_id)
            if user_dept:
                repo = DepartmentRepository(session)
                ancestor_ids = set(repo.get_ancestor_ids(user_dept.id_path))

        legacy_team_name = user.team_name if user.team_name else None
        return ancestor_ids, legacy_team_name
