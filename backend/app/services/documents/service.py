from __future__ import annotations

from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.document import Document, DocumentACL
from app.models.user import User
from app.repositories.document_repository import DocumentRepository
from app.repositories.role_repository import RoleRepository
from app.repositories.user_repository import UserRepository
from app.schemas.document import DocumentACLCreate, DocumentACLRead, DocumentCreate, DocumentRead
from app.services.permissions.service import PermissionFilterBuilder


class DocumentService:
    def __init__(self, session: Session):
        self.session = session
        self.document_repository = DocumentRepository(session)
        self.user_repository = UserRepository(session)
        self.role_repository = RoleRepository(session)
        self.permission_builder = PermissionFilterBuilder()

    def create_document(self, actor: User, payload: DocumentCreate) -> DocumentRead:
        document = Document(
            title=payload.title,
            description=payload.description,
            status=payload.status,
            owner_user_id=actor.id,
        )
        self.document_repository.add(document)
        self.session.commit()
        self.session.refresh(document)
        return self._serialize_document(document, current_user_can_manage=True)

    def list_visible_documents(self, actor: User) -> list[DocumentRead]:
        visibility_query = self.permission_builder.build_accessible_document_ids_query(actor, require_manage=False)
        documents = self.document_repository.list_visible(visibility_query)
        return [
            self._serialize_document(document, self.permission_builder.get_document_decision(actor, document).can_manage)
            for document in documents
        ]

    def get_visible_document(self, actor: User, document_id: UUID) -> DocumentRead:
        visibility_query = self.permission_builder.build_accessible_document_ids_query(actor, require_manage=False)
        document = self.document_repository.get_visible_by_id(document_id, visibility_query)
        if document is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

        decision = self.permission_builder.get_document_decision(actor, document)
        return self._serialize_document(document, decision.can_manage)

    def list_acl_entries(self, actor: User, document_id: UUID) -> list[DocumentACLRead]:
        document = self._get_manageable_document(actor, document_id)
        acl_entries = self.document_repository.get_acl_entries(document.id)
        return [self._serialize_acl_entry(entry) for entry in acl_entries]

    def upsert_acl_entry(self, actor: User, document_id: UUID, payload: DocumentACLCreate) -> DocumentACLRead:
        document = self._get_manageable_document(actor, document_id)

        resolved_user = None
        resolved_role = None
        resolved_team_name = None

        if payload.principal_type.value == "user":
            resolved_user = self.user_repository.get_by_id(payload.user_id)
            if resolved_user is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target user not found.")
        elif payload.principal_type.value == "role":
            resolved_role = self.role_repository.get_by_name(payload.role_name)
            if resolved_role is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target role not found.")
        elif payload.principal_type.value == "team":
            resolved_team_name = payload.team_name

        existing = self.document_repository.find_acl_entry(
            document_id=document.id,
            principal_type=payload.principal_type,
            user_id=resolved_user.id if resolved_user else None,
            role_id=resolved_role.id if resolved_role else None,
            team_name=resolved_team_name,
        )
        if existing is None:
            existing = DocumentACL(
                document_id=document.id,
                principal_type=payload.principal_type,
                user_id=resolved_user.id if resolved_user else None,
                role_id=resolved_role.id if resolved_role else None,
                team_name=resolved_team_name,
                can_view=payload.can_view,
                can_manage=payload.can_manage,
            )
            if resolved_user is not None:
                existing.user = resolved_user
            if resolved_role is not None:
                existing.role = resolved_role
            self.document_repository.add_acl_entry(existing)
        else:
            existing.can_view = payload.can_view
            existing.can_manage = payload.can_manage

        self.session.commit()
        hydrated = self.document_repository.get_acl_entry_by_id(existing.id)
        if hydrated is None:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to load ACL entry.")
        return self._serialize_acl_entry(hydrated)

    def _get_manageable_document(self, actor: User, document_id: UUID) -> Document:
        visibility_query = self.permission_builder.build_accessible_document_ids_query(actor, require_manage=True)
        document = self.document_repository.get_visible_by_id(document_id, visibility_query)
        if document is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
        return document

    @staticmethod
    def _serialize_document(document: Document, current_user_can_manage: bool) -> DocumentRead:
        return DocumentRead.model_validate(
            {
                "id": document.id,
                "title": document.title,
                "description": document.description,
                "status": document.status,
                "owner_user_id": document.owner_user_id,
                "current_version_id": document.current_version_id,
                "created_at": document.created_at,
                "updated_at": document.updated_at,
                "current_user_can_manage": current_user_can_manage,
            }
        )

    @staticmethod
    def _serialize_acl_entry(entry: DocumentACL) -> DocumentACLRead:
        role_name = entry.role.name if entry.role else None
        user_email = entry.user.email if entry.user else None
        return DocumentACLRead.model_validate(
            {
                "id": entry.id,
                "document_id": entry.document_id,
                "principal_type": entry.principal_type,
                "user_id": entry.user_id,
                "user_email": user_email,
                "role_id": entry.role_id,
                "role_name": role_name,
                "team_name": entry.team_name,
                "can_view": entry.can_view,
                "can_manage": entry.can_manage,
                "created_at": entry.created_at,
            }
        )
