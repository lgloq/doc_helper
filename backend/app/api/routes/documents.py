from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.orm import Session

from app.api.deps.auth import get_current_user, require_admin
from app.db.session import get_db_session
from app.models.enums import DocumentStatus
from app.models.user import User
from app.schemas.document import (
    AsyncIngestResponse,
    ChunkRead,
    DocumentACLCreate,
    DocumentACLRead,
    DocumentCreate,
    DocumentDiffRead,
    DocumentDiffRequest,
    DocumentDiffSummaryRead,
    DocumentIngestRequest,
    DocumentRead,
    DocumentUploadResponse,
    DocumentVersionDetailRead,
    DocumentVersionRead,
    IngestionResultRead,
)
from app.services.diff.service import DocumentDiffService
from app.services.documents.service import DocumentService
from app.services.ingestion.async_service import AsyncIngestionService
from app.services.ingestion.service import DocumentIngestionService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentRead)
def create_document(
    payload: DocumentCreate,
    current_user: User = Depends(require_admin),
    session: Session = Depends(get_db_session),
) -> DocumentRead:
    service = DocumentService(session)
    return service.create_document(current_user, payload)


@router.post("/upload", response_model=DocumentUploadResponse)
def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    description: str | None = Form(None),
    status: DocumentStatus = Form(DocumentStatus.DRAFT),
    current_user: User = Depends(require_admin),
    session: Session = Depends(get_db_session),
) -> DocumentUploadResponse:
    service = DocumentIngestionService(session)
    return service.upload_document(current_user, file, title, description, status)


@router.get("", response_model=list[DocumentRead])
def list_documents(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> list[DocumentRead]:
    service = DocumentService(session)
    return service.list_visible_documents(current_user)


@router.post("/{document_id}/versions/upload", response_model=DocumentUploadResponse)
def upload_document_version(
    document_id: UUID,
    file: UploadFile = File(...),
    current_user: User = Depends(require_admin),
    session: Session = Depends(get_db_session),
) -> DocumentUploadResponse:
    service = DocumentIngestionService(session)
    return service.upload_document_version(current_user, document_id, file)


@router.get("/{document_id}/versions", response_model=list[DocumentVersionRead])
def list_document_versions(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> list[DocumentVersionRead]:
    service = DocumentIngestionService(session)
    return service.list_versions(current_user, document_id)


@router.get("/{document_id}/versions/{version_id}", response_model=DocumentVersionDetailRead)
def get_document_version(
    document_id: UUID,
    version_id: UUID,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> DocumentVersionDetailRead:
    service = DocumentIngestionService(session)
    return service.get_version_detail(current_user, document_id, version_id)


@router.get("/{document_id}/diff", response_model=DocumentDiffRead)
def get_document_diff(
    document_id: UUID,
    from_version: UUID = Query(...),
    to_version: UUID = Query(...),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> DocumentDiffRead:
    service = DocumentDiffService(session)
    return service.get_diff(current_user, document_id, from_version, to_version)


@router.post("/{document_id}/diff/summary", response_model=DocumentDiffSummaryRead)
def summarize_document_diff(
    document_id: UUID,
    payload: DocumentDiffRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> DocumentDiffSummaryRead:
    service = DocumentDiffService(session)
    return service.summarize_diff(current_user, document_id, payload)


@router.get("/{document_id}", response_model=DocumentRead)
def get_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> DocumentRead:
    service = DocumentService(session)
    return service.get_visible_document(current_user, document_id)


@router.post("/{document_id}/ingest", response_model=IngestionResultRead)
def ingest_document(
    document_id: UUID,
    payload: DocumentIngestRequest | None = None,
    current_user: User = Depends(require_admin),
    session: Session = Depends(get_db_session),
) -> IngestionResultRead:
    service = DocumentIngestionService(session)
    return service.ingest_document(current_user, document_id, payload)


@router.post("/{document_id}/ingest/async", response_model=AsyncIngestResponse)
async def ingest_document_async(
    document_id: UUID,
    payload: DocumentIngestRequest | None = None,
    current_user: User = Depends(require_admin),
    session: Session = Depends(get_db_session),
) -> AsyncIngestResponse:
    """异步入库：提交后台任务后立即返回，通过轮询版本状态获取进度。"""
    service = AsyncIngestionService(session)
    version_id = payload.version_id if payload else None
    result = await service.enqueue_ingest(current_user, document_id, version_id)
    return AsyncIngestResponse.model_validate(result)


@router.get("/{document_id}/chunks", response_model=list[ChunkRead])
def list_document_chunks(
    document_id: UUID,
    version_id: UUID | None = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> list[ChunkRead]:
    service = DocumentIngestionService(session)
    return service.list_chunks(current_user, document_id, version_id)


@router.post("/{document_id}/acl", response_model=DocumentACLRead)
def upsert_document_acl(
    document_id: UUID,
    payload: DocumentACLCreate,
    current_user: User = Depends(require_admin),
    session: Session = Depends(get_db_session),
) -> DocumentACLRead:
    service = DocumentService(session)
    return service.upsert_acl_entry(current_user, document_id, payload)


@router.get("/{document_id}/acl", response_model=list[DocumentACLRead])
def list_document_acls(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> list[DocumentACLRead]:
    service = DocumentService(session)
    return service.list_acl_entries(current_user, document_id)
