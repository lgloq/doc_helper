from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.enums import DocumentStatus, PrincipalType, RoleName
from app.schemas.document import (
    DocumentAccessDebugUserRead,
    DocumentAccessDepartmentContextRead,
    DocumentAccessMatchedRuleRead,
)


class PermissionScopeDocumentRead(BaseModel):
    id: UUID
    title: str
    status: DocumentStatus
    owner_user_id: UUID
    current_version_id: UUID | None = None
    can_manage: bool = False
    reason: str
    matched_rule: DocumentAccessMatchedRuleRead | None = None
    created_at: datetime
    updated_at: datetime


class PermissionScopeSummaryRead(BaseModel):
    admin_count: int = 0
    owner_count: int = 0
    acl_count: int = 0
    public_acl_count: int = 0
    user_acl_count: int = 0
    role_acl_count: int = 0
    department_acl_count: int = 0


class UserVisibleScopeRead(BaseModel):
    evaluated_user: DocumentAccessDebugUserRead
    department_context: DocumentAccessDepartmentContextRead
    visible_document_count: int
    manageable_document_count: int
    returned_document_count: int
    limit: int
    permission_summary: PermissionScopeSummaryRead
    visible_documents: list[PermissionScopeDocumentRead] = Field(default_factory=list)


class ProposedACLRead(BaseModel):
    principal_type: PrincipalType
    user_id: UUID | None = None
    user_email: str | None = None
    user_full_name: str | None = None
    role_id: UUID | None = None
    role_name: RoleName | None = None
    team_name: str | None = None
    department_id: UUID | None = None
    department_path: str | None = None
    can_view: bool
    can_manage: bool


class PermissionImpactUserRead(BaseModel):
    id: UUID
    email: str
    full_name: str
    role_name: str | None = None
    department_id: UUID | None = None
    department_path: str | None = None
    before_can_view: bool
    after_can_view: bool
    before_can_manage: bool
    after_can_manage: bool
    impact: str
    reason: str
    matched_rule: DocumentAccessMatchedRuleRead | None = None


class PermissionACLImpactRead(BaseModel):
    document_id: UUID
    document_title: str
    existing_acl_id: UUID | None = None
    operation: str
    proposed_acl: ProposedACLRead
    affected_user_count: int
    newly_visible_user_count: int
    no_longer_visible_user_count: int
    newly_manageable_user_count: int
    no_longer_manageable_user_count: int
    preview_user_count: int
    users_preview: list[PermissionImpactUserRead] = Field(default_factory=list)
