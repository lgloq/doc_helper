from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.models.chunk import Chunk
from app.models.document import Document, DocumentVersion
from app.models.enums import DocumentStatus, IngestStatus
from app.models.user import User
from app.repositories.document_repository import ChunkRepository, DocumentRepository
from app.schemas.document import (
    ChunkRead,
    DocumentIngestRequest,
    DocumentUploadResponse,
    DocumentVersionDetailRead,
    DocumentVersionRead,
    IngestionResultRead,
)
from app.services.ingestion.chunking import SemanticChunker
from app.services.ingestion.embeddings import EmbeddingProviderFactory
from app.services.ingestion.file_storage import LocalDocumentStorage, UploadInspection
from app.services.ingestion.parsers import DocumentParser
from app.services.permissions.service import PermissionFilterBuilder


class DocumentIngestionService:
    def __init__(self, session: Session):
        self.session = session
        self.document_repository = DocumentRepository(session)
        self.chunk_repository = ChunkRepository(session)
        self.permission_builder = PermissionFilterBuilder()
        self.storage = LocalDocumentStorage()
        self.parser = DocumentParser()
        self.chunker = SemanticChunker()
        self.embedding_provider = EmbeddingProviderFactory.create()

    def upload_document(
        self,
        actor: User,
        file: UploadFile,
        title: str | None,
        description: str | None,
        status_value: DocumentStatus,
    ) -> DocumentUploadResponse:
        document_title = title or Path(file.filename or "document").stem or "Untitled Document"
        upload_inspection = self._inspect_upload(file)
        existing_version = self.document_repository.find_version_by_checksum_and_title(
            checksum_sha256=upload_inspection.checksum_sha256,
            title=document_title,
        )
        if existing_version is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"检测到同标题且内容完全相同的文档已存在："
                    f"《{existing_version.document.title}》v{existing_version.version_number}。"
                    "若要更新现有文档，请使用“上传新版本”；若只需重建索引，请对现有版本执行重新入库。"
                ),
            )

        document = Document(
            title=document_title,
            description=description,
            status=status_value,
            owner_user_id=actor.id,
        )
        self.document_repository.add(document)
        self.session.flush()

        version = self._create_version_record(actor, document, file, version_number=1, upload_inspection=upload_inspection)
        self.document_repository.set_current_version(document, version.id)
        self.session.commit()
        self.session.refresh(document)
        self.session.refresh(version)
        return DocumentUploadResponse(
            document=self._serialize_document(document, current_user_can_manage=True),
            version=DocumentVersionRead.model_validate(version),
        )

    def upload_document_version(self, actor: User, document_id: UUID, file: UploadFile) -> DocumentUploadResponse:
        document = self._get_manageable_document(actor, document_id)
        upload_inspection = self._inspect_upload(file)
        existing_version = self.document_repository.find_version_by_checksum_in_document(
            document_id=document.id,
            checksum_sha256=upload_inspection.checksum_sha256,
        )
        if existing_version is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"该文件与文档《{document.title}》的 v{existing_version.version_number} 完全一致，"
                    "无需重复上传相同版本。"
                ),
            )
        version_number = self.document_repository.get_next_version_number(document.id)
        version = self._create_version_record(
            actor,
            document,
            file,
            version_number=version_number,
            upload_inspection=upload_inspection,
        )
        self.session.commit()
        self.session.refresh(document)
        self.session.refresh(version)
        current_user_can_manage = self.permission_builder.get_document_decision(self.session, actor, document).can_manage
        return DocumentUploadResponse(
            document=self._serialize_document(document, current_user_can_manage=current_user_can_manage),
            version=DocumentVersionRead.model_validate(version),
        )

    def ingest_document(self, actor: User, document_id: UUID, payload: DocumentIngestRequest | None = None) -> IngestionResultRead:
        document = self._get_manageable_document(actor, document_id)
        version = self._resolve_version(document, payload.version_id if payload else None)

        version.ingest_status = IngestStatus.PROCESSING
        version.ingest_error = None
        self.session.commit()

        try:
            parsed_document = self.parser.parse(self.storage.resolve_path(version.storage_path))
            chunk_payloads = self.chunker.chunk_document(parsed_document)
            embeddings = self.embedding_provider.embed_texts([chunk.content for chunk in chunk_payloads]) if chunk_payloads else []
            chunk_models = [
                Chunk(
                    document_id=document.id,
                    document_version_id=version.id,
                    chunk_index=chunk_payload.chunk_index,
                    content=chunk_payload.content,
                    token_count=chunk_payload.token_count,
                    section_title=chunk_payload.section_title,
                    page_number_start=chunk_payload.page_number_start,
                    page_number_end=chunk_payload.page_number_end,
                    paragraph_start=chunk_payload.paragraph_start,
                    paragraph_end=chunk_payload.paragraph_end,
                    char_start=chunk_payload.char_start,
                    char_end=chunk_payload.char_end,
                    citation_metadata=chunk_payload.citation_metadata,
                    embedding=embeddings[index] if index < len(embeddings) else None,
                )
                for index, chunk_payload in enumerate(chunk_payloads)
            ]

            version.extracted_text = parsed_document.normalized_text
            version.page_count = parsed_document.page_count
            version.ingest_status = IngestStatus.READY
            version.ingest_error = None
            self.chunk_repository.replace_version_chunks(version, chunk_models)
            self.document_repository.set_current_version(document, version.id)
            self.session.commit()
            self.session.refresh(version)

            return IngestionResultRead(
                document_id=document.id,
                document_version_id=version.id,
                ingest_status=version.ingest_status,
                chunk_count=len(chunk_models),
                page_count=version.page_count,
            )
        except Exception as exc:
            self.session.rollback()
            version = self.document_repository.get_version(document.id, version.id)
            if version is not None:
                version.ingest_status = IngestStatus.FAILED
                version.ingest_error = str(exc)
                self.session.commit()
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    def list_versions(self, actor: User, document_id: UUID) -> list[DocumentVersionRead]:
        document = self._get_viewable_document(actor, document_id)
        versions = self.document_repository.list_versions(document_id)
        return [self._serialize_version(version, document.current_version_id == version.id) for version in versions]

    def get_version_detail(self, actor: User, document_id: UUID, version_id: UUID) -> DocumentVersionDetailRead:
        document = self._get_viewable_document(actor, document_id)
        version = self.document_repository.get_version(document.id, version_id)
        if version is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document version not found.")
        return self._serialize_version(version, document.current_version_id == version.id)

    def list_chunks(self, actor: User, document_id: UUID, version_id: UUID | None = None) -> list[ChunkRead]:
        document = self._get_viewable_document(actor, document_id)
        version = self._resolve_version(document, version_id)
        chunks = self.chunk_repository.list_by_document(document.id, version.id)
        return [self._serialize_chunk(chunk) for chunk in chunks]

    def _get_viewable_document(self, actor: User, document_id: UUID) -> Document:
        visibility_query = self.permission_builder.build_accessible_document_ids_query(self.session, actor, require_manage=False)
        document = self.document_repository.get_visible_by_id(document_id, visibility_query)
        if document is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
        return document

    def _get_manageable_document(self, actor: User, document_id: UUID) -> Document:
        visibility_query = self.permission_builder.build_accessible_document_ids_query(self.session, actor, require_manage=True)
        document = self.document_repository.get_visible_by_id(document_id, visibility_query)
        if document is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
        return document

    def _resolve_version(self, document: Document, version_id: UUID | None) -> DocumentVersion:
        version = None
        if version_id is not None:
            version = self.document_repository.get_version(document.id, version_id)
        elif document.current_version_id is not None:
            version = self.document_repository.get_version(document.id, document.current_version_id)
        if version is None:
            version = self.document_repository.get_latest_version(document.id)
        if version is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document version not found.")
        return version

    def _create_version_record(
        self,
        actor: User,
        document: Document,
        file: UploadFile,
        version_number: int,
        *,
        upload_inspection: UploadInspection | None = None,
    ) -> DocumentVersion:
        try:
            stored_file = self.storage.save_upload(document.id, version_number, file, inspection=upload_inspection)
        except ValueError as exc:
            self.session.rollback()
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

        version = DocumentVersion(
            document_id=document.id,
            version_number=version_number,
            original_filename=stored_file.original_filename,
            mime_type=stored_file.mime_type,
            file_size=stored_file.file_size,
            storage_path=stored_file.relative_path,
            checksum_sha256=stored_file.checksum_sha256,
            created_by_user_id=actor.id,
            ingest_status=IngestStatus.PENDING,
        )
        self.document_repository.add_version(version)
        self.session.flush()
        return version

    def _inspect_upload(self, file: UploadFile) -> UploadInspection:
        try:
            return self.storage.inspect_upload(file)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    @staticmethod
    def _serialize_document(document: Document, current_user_can_manage: bool):
        from app.services.documents.service import DocumentService

        return DocumentService._serialize_document(document, current_user_can_manage)

    @staticmethod
    def _serialize_chunk(chunk: Chunk) -> ChunkRead:
        return ChunkRead.model_validate(
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "document_version_id": chunk.document_version_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "preview": chunk.content[:240],
                "token_count": chunk.token_count,
                "section_title": chunk.section_title,
                "page_number_start": chunk.page_number_start,
                "page_number_end": chunk.page_number_end,
                "paragraph_start": chunk.paragraph_start,
                "paragraph_end": chunk.paragraph_end,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "citation_metadata": chunk.citation_metadata,
                "created_at": chunk.created_at,
            }
        )

    @staticmethod
    def _serialize_version(version: DocumentVersion, is_current: bool) -> DocumentVersionDetailRead:
        return DocumentVersionDetailRead.model_validate(
            {
                "id": version.id,
                "document_id": version.document_id,
                "version_number": version.version_number,
                "original_filename": version.original_filename,
                "mime_type": version.mime_type,
                "file_size": version.file_size,
                "storage_path": version.storage_path,
                "checksum_sha256": version.checksum_sha256,
                "extracted_text": version.extracted_text,
                "ingest_status": version.ingest_status,
                "ingest_error": version.ingest_error,
                "page_count": version.page_count,
                "created_at": version.created_at,
                "is_current": is_current,
            }
        )
