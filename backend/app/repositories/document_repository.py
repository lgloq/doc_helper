from __future__ import annotations

from typing import Sequence
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session, selectinload

from app.models.chunk import Chunk
from app.models.document import Document, DocumentACL, DocumentVersion


class DocumentRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, document: Document) -> Document:
        self.session.add(document)
        return document

    def add_version(self, version: DocumentVersion) -> DocumentVersion:
        self.session.add(version)
        return version

    def get_by_id(self, document_id: UUID) -> Document | None:
        statement = (
            select(Document)
            .options(selectinload(Document.acl_entries), selectinload(Document.current_version))
            .where(Document.id == document_id)
        )
        return self.session.scalar(statement)

    def get_visible_by_id(self, document_id: UUID, visibility_query) -> Document | None:
        statement = (
            select(Document)
            .options(
                selectinload(Document.acl_entries),
                selectinload(Document.current_version),
                selectinload(Document.versions),
            )
            .where(Document.id == document_id)
            .where(Document.id.in_(visibility_query))
        )
        return self.session.scalar(statement)

    def list_visible(self, visibility_query) -> Sequence[Document]:
        statement = (
            select(Document)
            .options(selectinload(Document.acl_entries), selectinload(Document.current_version))
            .where(Document.id.in_(visibility_query))
            .order_by(Document.created_at.desc())
        )
        return list(self.session.scalars(statement).all())

    def list_versions(self, document_id: UUID) -> Sequence[DocumentVersion]:
        statement = (
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.version_number.desc())
        )
        return list(self.session.scalars(statement).all())

    def get_version(self, document_id: UUID, version_id: UUID) -> DocumentVersion | None:
        statement = select(DocumentVersion).where(
            DocumentVersion.document_id == document_id,
            DocumentVersion.id == version_id,
        )
        return self.session.scalar(statement)

    def get_latest_version(self, document_id: UUID) -> DocumentVersion | None:
        statement = (
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.version_number.desc())
            .limit(1)
        )
        return self.session.scalar(statement)

    def find_version_by_checksum_and_title(self, *, checksum_sha256: str, title: str) -> DocumentVersion | None:
        statement = (
            select(DocumentVersion)
            .join(Document, Document.id == DocumentVersion.document_id)
            .options(selectinload(DocumentVersion.document))
            .where(DocumentVersion.checksum_sha256 == checksum_sha256)
            .where(func.lower(Document.title) == title.casefold())
            .order_by(DocumentVersion.created_at.desc())
            .limit(1)
        )
        return self.session.scalar(statement)

    def find_version_by_checksum_in_document(self, *, document_id: UUID, checksum_sha256: str) -> DocumentVersion | None:
        statement = (
            select(DocumentVersion)
            .options(selectinload(DocumentVersion.document))
            .where(DocumentVersion.document_id == document_id)
            .where(DocumentVersion.checksum_sha256 == checksum_sha256)
            .order_by(DocumentVersion.version_number.desc())
            .limit(1)
        )
        return self.session.scalar(statement)

    def get_next_version_number(self, document_id: UUID) -> int:
        statement = select(func.coalesce(func.max(DocumentVersion.version_number), 0)).where(DocumentVersion.document_id == document_id)
        current = self.session.scalar(statement) or 0
        return int(current) + 1

    def get_acl_entries(self, document_id: UUID) -> Sequence[DocumentACL]:
        statement = (
            select(DocumentACL)
            .options(selectinload(DocumentACL.user), selectinload(DocumentACL.role))
            .where(DocumentACL.document_id == document_id)
            .order_by(DocumentACL.created_at.asc())
        )
        return list(self.session.scalars(statement).all())

    def get_acl_entry_by_id(self, acl_entry_id: UUID) -> DocumentACL | None:
        statement = (
            select(DocumentACL)
            .options(selectinload(DocumentACL.user), selectinload(DocumentACL.role))
            .where(DocumentACL.id == acl_entry_id)
        )
        return self.session.scalar(statement)

    def find_acl_entry(
        self,
        document_id: UUID,
        principal_type,
        user_id: UUID | None = None,
        role_id: UUID | None = None,
        team_name: str | None = None,
    ) -> DocumentACL | None:
        statement = select(DocumentACL).where(
            and_(
                DocumentACL.document_id == document_id,
                DocumentACL.principal_type == principal_type,
                DocumentACL.user_id == user_id,
                DocumentACL.role_id == role_id,
                DocumentACL.team_name == team_name,
            )
        )
        return self.session.scalar(statement)

    def add_acl_entry(self, acl_entry: DocumentACL) -> DocumentACL:
        self.session.add(acl_entry)
        return acl_entry

    def set_current_version(self, document: Document, version_id: UUID) -> None:
        document.current_version_id = version_id


class ChunkRepository:
    def __init__(self, session: Session):
        self.session = session

    def replace_version_chunks(self, version: DocumentVersion, chunks: Sequence[Chunk]) -> None:
        self.session.query(Chunk).filter(Chunk.document_version_id == version.id).delete()
        for chunk in chunks:
            self.session.add(chunk)

    def list_by_document(self, document_id: UUID, version_id: UUID | None = None) -> Sequence[Chunk]:
        statement = select(Chunk).where(Chunk.document_id == document_id)
        if version_id is not None:
            statement = statement.where(Chunk.document_version_id == version_id)
        statement = statement.order_by(Chunk.chunk_index.asc())
        return list(self.session.scalars(statement).all())
