from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps.auth import require_admin
from app.db.session import get_db_session
from app.models.user import User
from app.schemas.document import DocumentACLCreate
from app.schemas.observability import TraceLogRead
from app.schemas.permission import PermissionACLImpactRead, UserVisibleScopeRead
from app.services.observability.service import ObservabilityService
from app.services.permissions.service import PermissionFilterBuilder

router = APIRouter(prefix="/permissions", tags=["permissions"])


@router.get("/users/{user_id}/scope", response_model=UserVisibleScopeRead)
def get_user_visible_scope(
    user_id: UUID,
    limit: int = Query(default=50, ge=1, le=200),
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_db_session),
) -> UserVisibleScopeRead:
    builder = PermissionFilterBuilder()
    return builder.get_user_visible_scope(session, user_id, limit=limit)


@router.post("/documents/{document_id}/acl/impact", response_model=PermissionACLImpactRead)
def analyze_document_acl_impact(
    document_id: UUID,
    payload: DocumentACLCreate,
    preview_limit: int = Query(default=30, ge=1, le=100),
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_db_session),
) -> PermissionACLImpactRead:
    builder = PermissionFilterBuilder()
    return builder.analyze_acl_impact(session, document_id, payload, preview_limit=preview_limit)


@router.get("/audit/traces", response_model=list[TraceLogRead])
def list_permission_audit_traces(
    user_id: UUID | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_db_session),
) -> list[TraceLogRead]:
    service = ObservabilityService(session)
    return service.list_traces(
        user_id=user_id,
        trace_type="permission_denied_retrieval",
        limit=limit,
    )
