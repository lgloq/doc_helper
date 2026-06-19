from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.models.enums import DocumentStatus, IngestStatus, PrincipalType, RoleName
from app.schemas.base import ORMModel


class DocumentCreate(BaseModel):
    title: str
    description: str | None = None
    status: DocumentStatus = DocumentStatus.DRAFT


class DocumentVersionRead(ORMModel):
    id: UUID
    document_id: UUID
    version_number: int
    original_filename: str
    mime_type: str
    file_size: int
    storage_path: str
    checksum_sha256: str
    extracted_text: str | None = None
    ingest_status: IngestStatus
    ingest_error: str | None = None
    page_count: int | None = None
    created_at: datetime


class DocumentVersionDetailRead(DocumentVersionRead):
    is_current: bool = False


class DocumentIngestRequest(BaseModel):
    version_id: UUID | None = None


class DocumentDiffRequest(BaseModel):
    from_version_id: UUID
    to_version_id: UUID
    force_refresh: bool = False


class DocumentDiffChangeRead(BaseModel):
    change_type: str
    from_paragraph_start: int | None = None
    from_paragraph_end: int | None = None
    to_paragraph_start: int | None = None
    to_paragraph_end: int | None = None
    old_text: str | None = None
    new_text: str | None = None


class DocumentDiffRead(BaseModel):
    document_id: UUID
    from_version_id: UUID
    to_version_id: UUID
    from_version_number: int
    to_version_number: int
    added_count: int
    deleted_count: int
    modified_count: int
    unified_diff: str
    changes: list[DocumentDiffChangeRead] = Field(default_factory=list)
    impact_hints: list[str] = Field(default_factory=list)


class DocumentDiffSummaryRead(BaseModel):
    document_id: UUID
    from_version_id: UUID
    to_version_id: UUID
    from_version_number: int
    to_version_number: int
    summary: str
    additions: list[str] = Field(default_factory=list)
    deletions: list[str] = Field(default_factory=list)
    modifications: list[str] = Field(default_factory=list)
    impact_hints: list[str] = Field(default_factory=list)
    summary_provider: str
    model_name: str | None = None
    cache_hit: bool = False


class DocumentACLCreate(BaseModel):
    principal_type: PrincipalType
    user_id: UUID | None = None
    role_name: RoleName | None = None
    team_name: str | None = None  # deprecated fallback
    department_id: UUID | None = None
    can_view: bool = True
    can_manage: bool = False

    @model_validator(mode="after")
    def validate_principal_fields(self):
        if self.principal_type == PrincipalType.USER and self.user_id is None:
            raise ValueError("user_id is required when principal_type is user")
        if self.principal_type == PrincipalType.ROLE and self.role_name is None:
            raise ValueError("role_name is required when principal_type is role")
        if self.principal_type == PrincipalType.TEAM:
            if not self.department_id and not self.team_name:
                raise ValueError("department_id or team_name is required when principal_type is team")
            if self.department_id and self.team_name:
                raise ValueError("department_id and team_name cannot be used together")
        if self.principal_type == PrincipalType.PUBLIC and any(
            [self.user_id, self.role_name, self.team_name, self.department_id]
        ):
            raise ValueError("public ACL cannot target a user, role, team or department")
        if self.principal_type in {PrincipalType.USER, PrincipalType.ROLE} and (self.department_id or self.team_name):
            raise ValueError("user or role ACL should not have department_id or team_name")
        return self


class DocumentACLRead(ORMModel):
    id: UUID
    document_id: UUID
    principal_type: PrincipalType
    user_id: UUID | None = None
    user_email: str | None = None
    user_full_name: str | None = None
    role_id: UUID | None = None
    role_name: RoleName | None = None
    team_name: str | None = None
    department_id: UUID | None = None
    can_view: bool
    can_manage: bool
    created_at: datetime


class DocumentRead(ORMModel):
    id: UUID
    title: str
    description: str | None = None
    status: DocumentStatus
    owner_user_id: UUID
    current_version_id: UUID | None = None
    current_user_can_manage: bool = False
    created_at: datetime
    updated_at: datetime


class DocumentUploadResponse(BaseModel):
    document: DocumentRead
    version: DocumentVersionRead


class IngestionResultRead(BaseModel):
    document_id: UUID
    document_version_id: UUID
    ingest_status: IngestStatus
    chunk_count: int
    page_count: int | None = None


class AsyncIngestResponse(BaseModel):
    document_id: UUID
    document_version_id: UUID
    job_id: str
    ingest_status: IngestStatus
    message: str


class DocumentAccessCheckRead(BaseModel):
    source: str
    matched: bool
    message: str


class DocumentAccessMatchedRuleRead(BaseModel):
    source: str
    acl_id: UUID | None = None
    principal_type: PrincipalType | None = None
    department_id: UUID | None = None
    department_path: str | None = None
    match_type: str | None = None
    can_view: bool = False
    can_manage: bool = False


class DocumentAccessDebugUserRead(BaseModel):
    id: UUID
    email: str
    full_name: str
    role_name: str | None = None
    department_id: UUID | None = None
    department_path: str | None = None


class DocumentAccessDebugDocumentRead(BaseModel):
    id: UUID
    title: str
    owner_user_id: UUID


class DocumentAccessDepartmentContextRead(BaseModel):
    user_department_id: UUID | None = None
    user_department_path: str | None = None
    ancestor_department_ids: list[UUID] = Field(default_factory=list)
    ancestor_department_paths: list[str] = Field(default_factory=list)


class DocumentAccessDebugRead(BaseModel):
    document_id: UUID
    user_id: UUID
    can_view: bool
    can_manage: bool
    reason: str
    matched_rule: DocumentAccessMatchedRuleRead | None = None
    evaluated_user: DocumentAccessDebugUserRead
    evaluated_document: DocumentAccessDebugDocumentRead
    department_context: DocumentAccessDepartmentContextRead
    checks: list[DocumentAccessCheckRead] = Field(default_factory=list)


class ChunkRead(ORMModel):
    id: UUID
    document_id: UUID
    document_version_id: UUID
    chunk_index: int
    content: str
    preview: str
    token_count: int
    section_title: str | None = None
    page_number_start: int | None = None
    page_number_end: int | None = None
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    clause_full_name: str | None = None
    article_number: str | None = None
    chunk_type: str | None = None
    heading_path: str | None = None
    citation_metadata: dict | None = None
    created_at: datetime
