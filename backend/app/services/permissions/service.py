from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.user import User

if TYPE_CHECKING:
    from app.models.department import Department
    from app.models.document import DocumentACL


class PermissionDecision:
    def __init__(self, can_view: bool, can_manage: bool):
        self.can_view = can_view
        self.can_manage = can_manage


@dataclass
class _DepartmentContext:
    ancestor_ids: set[UUID]
    ancestor_departments: list[Department]
    legacy_team_name: str | None
    user_department_path: str | None


@dataclass
class _AclEvaluation:
    acl: DocumentACL
    principal_matches: bool
    source_label: str
    message: str
    match_type: str | None = None
    department_path: str | None = None

    @property
    def effective_view(self) -> bool:
        return bool(self.acl.can_view or self.acl.can_manage)

    @property
    def effective_manage(self) -> bool:
        return bool(self.acl.can_manage)

    @property
    def has_effective_permission(self) -> bool:
        return self.principal_matches and (self.effective_view or self.effective_manage)


@dataclass
class _DocumentAccessEvaluation:
    can_view: bool
    can_manage: bool
    system_match: str | None
    department_context: _DepartmentContext
    acl_evaluations: list[_AclEvaluation]
    effective_acl_matches: list[_AclEvaluation]


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

    def _evaluate_document_access(self, session: Session, user: User, document) -> _DocumentAccessEvaluation:
        from app.models.enums import RoleName

        department_context = self._build_department_context(session, user)

        if user.role and user.role.name == RoleName.ADMIN:
            return _DocumentAccessEvaluation(
                can_view=True,
                can_manage=True,
                system_match="admin",
                department_context=department_context,
                acl_evaluations=[],
                effective_acl_matches=[],
            )

        if document.owner_user_id == user.id:
            return _DocumentAccessEvaluation(
                can_view=True,
                can_manage=True,
                system_match="owner",
                department_context=department_context,
                acl_evaluations=[],
                effective_acl_matches=[],
            )

        acl_evaluations = [self._evaluate_acl(session, user, acl, department_context) for acl in document.acl_entries]
        effective_acl_matches = [acl_eval for acl_eval in acl_evaluations if acl_eval.has_effective_permission]

        return _DocumentAccessEvaluation(
            can_view=bool(effective_acl_matches),
            can_manage=any(acl_eval.effective_manage for acl_eval in effective_acl_matches),
            system_match=None,
            department_context=department_context,
            acl_evaluations=acl_evaluations,
            effective_acl_matches=effective_acl_matches,
        )

    def _build_department_context(self, session: Session, user: User) -> _DepartmentContext:
        from sqlalchemy import select

        from app.models.department import Department
        from app.repositories.department_repository import DepartmentRepository

        ancestor_ids: set[UUID] = set()
        ancestor_departments: list[Department] = []
        user_department_path = None

        if user.department_id:
            user_dept = session.get(Department, user.department_id)
            if user_dept:
                user_department_path = user_dept.path
                repo = DepartmentRepository(session)
                ancestor_ids = set(repo.get_ancestor_ids(user_dept.id_path))
                if ancestor_ids:
                    statement = select(Department).where(Department.id.in_(ancestor_ids))
                    ancestor_departments = list(session.scalars(statement).all())

        legacy_team_name = user.team_name if user.team_name else None
        return _DepartmentContext(
            ancestor_ids=ancestor_ids,
            ancestor_departments=ancestor_departments,
            legacy_team_name=legacy_team_name,
            user_department_path=user_department_path,
        )

    def _evaluate_acl(
        self,
        session: Session,
        user: User,
        acl: DocumentACL,
        department_context: _DepartmentContext,
    ) -> _AclEvaluation:
        from app.models.department import Department
        from app.models.enums import PrincipalType

        principal_matches = False
        match_type = None
        department_path = None
        source_label = acl.principal_type.value
        message = "ACL 主体不匹配。"

        if acl.principal_type == PrincipalType.PUBLIC:
            principal_matches = True
            message = "公共 ACL 匹配所有用户。"
        elif acl.principal_type == PrincipalType.USER:
            principal_matches = acl.user_id == user.id
            message = "用户 ACL 匹配当前用户。" if principal_matches else "用户 ACL 不匹配当前用户。"
        elif acl.principal_type == PrincipalType.ROLE:
            principal_matches = acl.role_id == user.role_id
            message = "角色 ACL 匹配当前用户角色。" if principal_matches else "角色 ACL 不匹配当前用户角色。"
        elif acl.principal_type == PrincipalType.TEAM:
            if acl.department_id and acl.department_id in department_context.ancestor_ids:
                principal_matches = True
                match_type = "direct" if acl.department_id == user.department_id else "ancestor"
                acl_department = session.get(Department, acl.department_id)
                department_path = acl_department.path if acl_department else None
                message = "部门 ACL 直接匹配当前用户部门。" if match_type == "direct" else "部门 ACL 匹配用户祖先部门。"
            elif acl.team_name and department_context.legacy_team_name == acl.team_name:
                principal_matches = True
                match_type = "legacy"
                message = "旧版团队 ACL 匹配当前用户团队。"
            else:
                message = "部门 ACL 不匹配当前用户部门或团队。"

        acl_eval = _AclEvaluation(
            acl=acl,
            principal_matches=principal_matches,
            source_label=source_label,
            message=message,
            match_type=match_type,
            department_path=department_path,
        )
        if principal_matches and not acl_eval.has_effective_permission:
            acl_eval.message = f"{message} 但该 ACL 未授予查看或管理权限。"
        return acl_eval

    def get_document_decision(self, session: Session, user: User, document) -> PermissionDecision:
        evaluation = self._evaluate_document_access(session, user, document)
        return PermissionDecision(can_view=evaluation.can_view, can_manage=evaluation.can_manage)

    def get_document_access_debug(self, session: Session, user_id: UUID, document_id: UUID):
        from app.repositories.document_repository import DocumentRepository
        from app.repositories.user_repository import UserRepository
        from app.schemas.document import (
            DocumentAccessCheckRead,
            DocumentAccessDebugDocumentRead,
            DocumentAccessDebugRead,
            DocumentAccessDebugUserRead,
            DocumentAccessDepartmentContextRead,
            DocumentAccessMatchedRuleRead,
        )

        user_repo = UserRepository(session)
        target_user = user_repo.get_by_id(user_id)
        if target_user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在。")

        document = DocumentRepository(session).get_by_id(document_id)
        if document is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在。")

        evaluation = self._evaluate_document_access(session, target_user, document)

        # 构建部门上下文
        ancestor_depts_sorted = sorted(evaluation.department_context.ancestor_departments, key=lambda d: d.depth)
        department_context = DocumentAccessDepartmentContextRead(
            user_department_id=target_user.department_id,
            user_department_path=evaluation.department_context.user_department_path,
            ancestor_department_ids=[d.id for d in ancestor_depts_sorted],
            ancestor_department_paths=[d.path for d in ancestor_depts_sorted],
        )

        # 构建用户信息
        evaluated_user = DocumentAccessDebugUserRead(
            id=target_user.id,
            email=target_user.email,
            full_name=target_user.full_name,
            role_name=target_user.role.name.value if target_user.role else None,
            department_id=target_user.department_id,
            department_path=evaluation.department_context.user_department_path,
        )

        # 构建文档信息
        evaluated_document = DocumentAccessDebugDocumentRead(
            id=document.id,
            title=document.title,
            owner_user_id=document.owner_user_id,
        )

        checks: list[DocumentAccessCheckRead] = []

        checks.append(
            DocumentAccessCheckRead(
                source="admin",
                matched=evaluation.system_match == "admin",
                message="用户是管理员，拥有全部权限。" if evaluation.system_match == "admin" else "用户不是管理员。",
            )
        )
        if evaluation.system_match == "admin":
            return DocumentAccessDebugRead(
                document_id=document.id,
                user_id=target_user.id,
                can_view=evaluation.can_view,
                can_manage=evaluation.can_manage,
                reason="用户是管理员，拥有全部权限。",
                matched_rule=DocumentAccessMatchedRuleRead(source="admin", can_view=True, can_manage=True),
                evaluated_user=evaluated_user,
                evaluated_document=evaluated_document,
                department_context=department_context,
                checks=checks,
            )

        checks.append(
            DocumentAccessCheckRead(
                source="owner",
                matched=evaluation.system_match == "owner",
                message="用户是文档所有者，拥有全部权限。"
                if evaluation.system_match == "owner"
                else "用户不是文档所有者。",
            )
        )
        if evaluation.system_match == "owner":
            return DocumentAccessDebugRead(
                document_id=document.id,
                user_id=target_user.id,
                can_view=evaluation.can_view,
                can_manage=evaluation.can_manage,
                reason="用户是文档所有者，拥有全部权限。",
                matched_rule=DocumentAccessMatchedRuleRead(source="owner", can_view=True, can_manage=True),
                evaluated_user=evaluated_user,
                evaluated_document=evaluated_document,
                department_context=department_context,
                checks=checks,
            )

        for acl_eval in evaluation.acl_evaluations:
            checks.append(
                DocumentAccessCheckRead(
                    source=acl_eval.source_label,
                    matched=acl_eval.has_effective_permission,
                    message=acl_eval.message,
                )
            )

        matched_acl = evaluation.effective_acl_matches[0] if evaluation.effective_acl_matches else None
        matched_rule = (
            DocumentAccessMatchedRuleRead(
                source=matched_acl.acl.principal_type.value,
                acl_id=matched_acl.acl.id,
                principal_type=matched_acl.acl.principal_type,
                department_id=matched_acl.acl.department_id,
                department_path=matched_acl.department_path,
                match_type=matched_acl.match_type,
                can_view=matched_acl.effective_view,
                can_manage=matched_acl.effective_manage,
            )
            if matched_acl
            else None
        )

        if evaluation.can_view:
            reason = (
                "命中 ACL 规则，用户拥有查看和管理权限。"
                if evaluation.can_manage
                else "命中 ACL 规则，用户拥有查看权限。"
            )
            if matched_rule and matched_rule.match_type == "ancestor":
                reason = "用户所在部门继承了父部门 ACL 的查看权限。"
        else:
            reason = "未命中任何权限规则，用户无权访问此文档。"

        return DocumentAccessDebugRead(
            document_id=document.id,
            user_id=target_user.id,
            can_view=evaluation.can_view,
            can_manage=evaluation.can_manage,
            reason=reason,
            matched_rule=matched_rule,
            evaluated_user=evaluated_user,
            evaluated_document=evaluated_document,
            department_context=department_context,
            checks=checks,
        )

    def _resolve_user_dept_context(self, session: Session, user: User) -> tuple[set[UUID], str | None]:
        """一次性解析用户部门祖先 ID 集合和旧 team_name。"""
        context = self._build_department_context(session, user)
        return context.ancestor_ids, context.legacy_team_name
