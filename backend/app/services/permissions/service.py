from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload
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


@dataclass
class _ResolvedACLTarget:
    user_id: UUID | None = None
    user_email: str | None = None
    user_full_name: str | None = None
    role_id: UUID | None = None
    role_name: object | None = None
    team_name: str | None = None
    department_id: UUID | None = None
    department_path: str | None = None


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

    def get_user_visible_scope(self, session: Session, user_id: UUID, *, limit: int = 50):
        from app.models.document import Document
        from app.repositories.user_repository import UserRepository
        from app.schemas.document import DocumentAccessDebugUserRead
        from app.schemas.permission import PermissionScopeDocumentRead, PermissionScopeSummaryRead, UserVisibleScopeRead

        target_user = UserRepository(session).get_by_id(user_id)
        if target_user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在。")

        documents = list(
            session.scalars(
                select(Document)
                .options(
                    selectinload(Document.acl_entries),
                    selectinload(Document.current_version),
                )
                .order_by(Document.created_at.desc())
            ).all()
        )
        department_context = self._serialize_department_context(target_user, self._build_department_context(session, target_user))
        evaluated_user = DocumentAccessDebugUserRead(
            id=target_user.id,
            email=target_user.email,
            full_name=target_user.full_name,
            role_name=target_user.role.name.value if target_user.role else None,
            department_id=target_user.department_id,
            department_path=department_context.user_department_path,
        )

        visible_document_count = 0
        manageable_document_count = 0
        summary = PermissionScopeSummaryRead()
        visible_documents: list[PermissionScopeDocumentRead] = []
        normalized_limit = max(1, min(int(limit), 200))

        for document in documents:
            evaluation = self._evaluate_document_access(session, target_user, document)
            if not evaluation.can_view:
                continue
            visible_document_count += 1
            if evaluation.can_manage:
                manageable_document_count += 1
            self._accumulate_scope_summary(summary, evaluation)
            reason, matched_rule = self._summarize_access_evaluation(evaluation)
            if len(visible_documents) < normalized_limit:
                visible_documents.append(
                    PermissionScopeDocumentRead(
                        id=document.id,
                        title=document.title,
                        status=document.status,
                        owner_user_id=document.owner_user_id,
                        current_version_id=document.current_version_id,
                        can_manage=evaluation.can_manage,
                        reason=reason,
                        matched_rule=matched_rule,
                        created_at=document.created_at,
                        updated_at=document.updated_at,
                    )
                )

        return UserVisibleScopeRead(
            evaluated_user=evaluated_user,
            department_context=department_context,
            visible_document_count=visible_document_count,
            manageable_document_count=manageable_document_count,
            returned_document_count=len(visible_documents),
            limit=normalized_limit,
            permission_summary=summary,
            visible_documents=visible_documents,
        )

    def analyze_acl_impact(self, session: Session, document_id: UUID, payload, *, preview_limit: int = 30):
        from app.models.document import Document, DocumentACL
        from app.repositories.document_repository import DocumentRepository
        from app.repositories.user_repository import UserRepository
        from app.schemas.permission import PermissionACLImpactRead, PermissionImpactUserRead

        document = DocumentRepository(session).get_by_id(document_id)
        if document is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在。")

        resolved_target = self._resolve_acl_target(session, payload)
        existing_acl = DocumentRepository(session).find_acl_entry(
            document_id=document.id,
            principal_type=payload.principal_type,
            user_id=resolved_target.user_id,
            role_id=resolved_target.role_id,
            team_name=resolved_target.team_name,
            department_id=resolved_target.department_id,
        )
        proposed_acl = DocumentACL(
            document_id=document.id,
            principal_type=payload.principal_type,
            user_id=resolved_target.user_id,
            role_id=resolved_target.role_id,
            team_name=resolved_target.team_name,
            department_id=resolved_target.department_id,
            can_view=payload.can_view,
            can_manage=payload.can_manage,
        )

        affected_user_count = 0
        newly_visible_user_count = 0
        no_longer_visible_user_count = 0
        newly_manageable_user_count = 0
        no_longer_manageable_user_count = 0
        users_preview: list[PermissionImpactUserRead] = []
        normalized_preview_limit = max(1, min(int(preview_limit), 100))

        for user in UserRepository(session).list_all(is_active=True):
            before = self._evaluate_document_access(session, user, document)
            after = self._evaluate_document_access_with_acl_change(
                session,
                user,
                document,
                proposed_acl=proposed_acl,
                existing_acl_id=existing_acl.id if existing_acl else None,
            )
            if before.can_view == after.can_view and before.can_manage == after.can_manage:
                continue

            affected_user_count += 1
            if not before.can_view and after.can_view:
                newly_visible_user_count += 1
            if before.can_view and not after.can_view:
                no_longer_visible_user_count += 1
            if not before.can_manage and after.can_manage:
                newly_manageable_user_count += 1
            if before.can_manage and not after.can_manage:
                no_longer_manageable_user_count += 1

            if len(users_preview) < normalized_preview_limit:
                reason, matched_rule = self._summarize_access_evaluation(after)
                impact = self._classify_permission_impact(before, after)
                department_context = self._build_department_context(session, user)
                users_preview.append(
                    PermissionImpactUserRead(
                        id=user.id,
                        email=user.email,
                        full_name=user.full_name,
                        role_name=user.role.name.value if user.role else None,
                        department_id=user.department_id,
                        department_path=department_context.user_department_path,
                        before_can_view=before.can_view,
                        after_can_view=after.can_view,
                        before_can_manage=before.can_manage,
                        after_can_manage=after.can_manage,
                        impact=impact,
                        reason=reason,
                        matched_rule=matched_rule,
                    )
                )

        operation = self._classify_acl_operation(existing_acl, payload)
        return PermissionACLImpactRead(
            document_id=document.id,
            document_title=document.title,
            existing_acl_id=existing_acl.id if existing_acl else None,
            operation=operation,
            proposed_acl=self._serialize_proposed_acl(payload, resolved_target),
            affected_user_count=affected_user_count,
            newly_visible_user_count=newly_visible_user_count,
            no_longer_visible_user_count=no_longer_visible_user_count,
            newly_manageable_user_count=newly_manageable_user_count,
            no_longer_manageable_user_count=no_longer_manageable_user_count,
            preview_user_count=len(users_preview),
            users_preview=users_preview,
        )

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
        department_context = self._serialize_department_context(target_user, evaluation.department_context)

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

    def _evaluate_document_access_with_acl_change(
        self,
        session: Session,
        user: User,
        document,
        *,
        proposed_acl,
        existing_acl_id: UUID | None,
    ) -> _DocumentAccessEvaluation:
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

        acl_entries = [
            acl for acl in document.acl_entries
            if existing_acl_id is None or acl.id != existing_acl_id
        ]
        if proposed_acl.can_view or proposed_acl.can_manage:
            acl_entries.append(proposed_acl)
        acl_evaluations = [self._evaluate_acl(session, user, acl, department_context) for acl in acl_entries]
        effective_acl_matches = [acl_eval for acl_eval in acl_evaluations if acl_eval.has_effective_permission]
        return _DocumentAccessEvaluation(
            can_view=bool(effective_acl_matches),
            can_manage=any(acl_eval.effective_manage for acl_eval in effective_acl_matches),
            system_match=None,
            department_context=department_context,
            acl_evaluations=acl_evaluations,
            effective_acl_matches=effective_acl_matches,
        )

    @staticmethod
    def _classify_permission_impact(before: _DocumentAccessEvaluation, after: _DocumentAccessEvaluation) -> str:
        if not before.can_view and after.can_view:
            return "newly_visible"
        if before.can_view and not after.can_view:
            return "no_longer_visible"
        if not before.can_manage and after.can_manage:
            return "newly_manageable"
        if before.can_manage and not after.can_manage:
            return "no_longer_manageable"
        return "changed"

    @staticmethod
    def _classify_acl_operation(existing_acl, payload) -> str:
        grants_permission = bool(payload.can_view or payload.can_manage)
        if existing_acl is None and grants_permission:
            return "create"
        if existing_acl is None and not grants_permission:
            return "noop"
        if not grants_permission:
            return "revoke"
        if existing_acl.can_view == payload.can_view and existing_acl.can_manage == payload.can_manage:
            return "unchanged"
        return "update"

    def _resolve_acl_target(self, session: Session, payload) -> _ResolvedACLTarget:
        from app.models.enums import PrincipalType
        from app.repositories.department_repository import DepartmentRepository
        from app.repositories.role_repository import RoleRepository
        from app.repositories.user_repository import UserRepository

        if payload.principal_type == PrincipalType.USER:
            target_user = UserRepository(session).get_by_id(payload.user_id)
            if target_user is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="目标用户不存在。")
            return _ResolvedACLTarget(
                user_id=target_user.id,
                user_email=target_user.email,
                user_full_name=target_user.full_name,
            )
        if payload.principal_type == PrincipalType.ROLE:
            target_role = RoleRepository(session).get_by_name(payload.role_name)
            if target_role is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="目标角色不存在。")
            return _ResolvedACLTarget(role_id=target_role.id, role_name=target_role.name)
        if payload.principal_type == PrincipalType.TEAM:
            department_path = None
            if payload.department_id is not None:
                department = DepartmentRepository(session).get_by_id(payload.department_id)
                if department is None:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="目标部门不存在。")
                department_path = department.path
            return _ResolvedACLTarget(
                team_name=payload.team_name,
                department_id=payload.department_id,
                department_path=department_path,
            )
        return _ResolvedACLTarget()

    @staticmethod
    def _serialize_proposed_acl(payload, resolved_target: _ResolvedACLTarget):
        from app.schemas.permission import ProposedACLRead

        return ProposedACLRead(
            principal_type=payload.principal_type,
            user_id=resolved_target.user_id,
            user_email=resolved_target.user_email,
            user_full_name=resolved_target.user_full_name,
            role_id=resolved_target.role_id,
            role_name=resolved_target.role_name,
            team_name=resolved_target.team_name,
            department_id=resolved_target.department_id,
            department_path=resolved_target.department_path,
            can_view=payload.can_view,
            can_manage=payload.can_manage,
        )

    @staticmethod
    def _serialize_department_context(user: User, department_context: _DepartmentContext):
        from app.schemas.document import DocumentAccessDepartmentContextRead

        ancestor_depts_sorted = sorted(department_context.ancestor_departments, key=lambda d: d.depth)
        return DocumentAccessDepartmentContextRead(
            user_department_id=user.department_id,
            user_department_path=department_context.user_department_path,
            ancestor_department_ids=[d.id for d in ancestor_depts_sorted],
            ancestor_department_paths=[d.path for d in ancestor_depts_sorted],
        )

    @staticmethod
    def _accumulate_scope_summary(summary, evaluation: _DocumentAccessEvaluation) -> None:
        if evaluation.system_match == "admin":
            summary.admin_count += 1
            return
        if evaluation.system_match == "owner":
            summary.owner_count += 1
            return
        matched_acl = evaluation.effective_acl_matches[0] if evaluation.effective_acl_matches else None
        if matched_acl is None:
            return
        summary.acl_count += 1
        source = matched_acl.acl.principal_type.value
        if source == "public":
            summary.public_acl_count += 1
        elif source == "user":
            summary.user_acl_count += 1
        elif source == "role":
            summary.role_acl_count += 1
        elif source == "team":
            summary.department_acl_count += 1

    @staticmethod
    def _summarize_access_evaluation(evaluation: _DocumentAccessEvaluation):
        from app.schemas.document import DocumentAccessMatchedRuleRead

        if evaluation.system_match == "admin":
            return "用户是管理员，拥有全部权限。", DocumentAccessMatchedRuleRead(
                source="admin",
                can_view=True,
                can_manage=True,
            )
        if evaluation.system_match == "owner":
            return "用户是文档所有者，拥有全部权限。", DocumentAccessMatchedRuleRead(
                source="owner",
                can_view=True,
                can_manage=True,
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
            return reason, matched_rule
        return "未命中任何权限规则，用户无权访问此文档。", None
