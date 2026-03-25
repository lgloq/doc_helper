from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.document import Document, DocumentVersion
from app.models.user import User
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import DocumentDiffChangeRead, DocumentDiffRead, DocumentDiffRequest, DocumentDiffSummaryRead
from app.services.diff.generation import DiffSummaryGeneratorFactory
from app.services.ingestion.file_storage import LocalDocumentStorage
from app.services.ingestion.parsers import DocumentParser
from app.services.permissions.service import PermissionFilterBuilder


KEYWORD_HINTS = {
    "security": "Security-related wording changed. Recheck access controls, security guidance, and incident procedures.",
    "access": "Access or permission language changed. Verify onboarding, privileged access, and approval workflows.",
    "deadline": "Deadline-related wording changed. Confirm project timelines and downstream commitments.",
    "sla": "SLA or service expectation wording changed. Notify support and operations stakeholders.",
    "rollback": "Rollback or recovery guidance changed. Revalidate release and incident runbooks.",
    "release": "Release process wording changed. Review deployment checklists and communication plans.",
    "approval": "Approval or sign-off wording changed. Check ownership and escalation paths.",
    "policy": "Policy wording changed. Review linked FAQs, training, and compliance-facing materials.",
    "compliance": "Compliance wording changed. Confirm regulatory documentation and audit evidence remain aligned.",
    "owner": "Ownership or contact wording changed. Verify responsible teams and support contacts.",
}


@dataclass
class ComputedDiff:
    document: Document
    from_version: DocumentVersion
    to_version: DocumentVersion
    changes: list[dict[str, Any]]
    unified_diff: str
    added_count: int
    deleted_count: int
    modified_count: int
    impact_hints: list[str]


class DocumentDiffService:
    def __init__(self, session: Session):
        self.session = session
        self.document_repository = DocumentRepository(session)
        self.permission_builder = PermissionFilterBuilder()
        self.storage = LocalDocumentStorage()
        self.parser = DocumentParser()
        self.summary_generator = DiffSummaryGeneratorFactory.create()

    def get_diff(self, actor: User, document_id: UUID, from_version_id: UUID, to_version_id: UUID) -> DocumentDiffRead:
        diff = self._compute_diff(actor, document_id, from_version_id, to_version_id)
        return self._serialize_diff(diff)

    def summarize_diff(self, actor: User, document_id: UUID, payload: DocumentDiffRequest) -> DocumentDiffSummaryRead:
        diff = self._compute_diff(actor, document_id, payload.from_version_id, payload.to_version_id)
        summary = self.summary_generator.generate(
            from_version_number=diff.from_version.version_number,
            to_version_number=diff.to_version.version_number,
            diff_changes=diff.changes,
            unified_diff=diff.unified_diff,
        )
        return DocumentDiffSummaryRead(
            document_id=diff.document.id,
            from_version_id=diff.from_version.id,
            to_version_id=diff.to_version.id,
            from_version_number=diff.from_version.version_number,
            to_version_number=diff.to_version.version_number,
            summary=summary.summary,
            additions=summary.additions,
            deletions=summary.deletions,
            modifications=summary.modifications,
            impact_hints=diff.impact_hints,
            summary_provider=summary.provider_name,
            model_name=summary.model_name,
        )

    def _compute_diff(self, actor: User, document_id: UUID, from_version_id: UUID, to_version_id: UUID) -> ComputedDiff:
        document = self._get_viewable_document(actor, document_id)
        from_version = self.document_repository.get_version(document.id, from_version_id)
        to_version = self.document_repository.get_version(document.id, to_version_id)
        if from_version is None or to_version is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document version not found.")

        from_paragraphs = self._extract_paragraphs(from_version)
        to_paragraphs = self._extract_paragraphs(to_version)
        changes, added_count, deleted_count, modified_count = self._build_change_set(from_paragraphs, to_paragraphs)
        unified_diff = self._build_unified_diff(from_paragraphs, to_paragraphs, from_version.version_number, to_version.version_number)
        impact_hints = self._infer_impact_hints(changes, added_count, deleted_count, modified_count)
        return ComputedDiff(
            document=document,
            from_version=from_version,
            to_version=to_version,
            changes=changes,
            unified_diff=unified_diff,
            added_count=added_count,
            deleted_count=deleted_count,
            modified_count=modified_count,
            impact_hints=impact_hints,
        )

    def _get_viewable_document(self, actor: User, document_id: UUID) -> Document:
        visibility_query = self.permission_builder.build_accessible_document_ids_query(actor, require_manage=False)
        document = self.document_repository.get_visible_by_id(document_id, visibility_query)
        if document is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
        return document

    def _extract_paragraphs(self, version: DocumentVersion) -> list[str]:
        text = version.extracted_text
        if not text:
            parsed = self.parser.parse(self.storage.resolve_path(version.storage_path))
            text = parsed.normalized_text
        normalized = self._normalize_text(text or "")
        paragraphs = [segment.strip() for segment in re.split(r"\n{2,}", normalized) if segment.strip()]
        if paragraphs:
            return paragraphs
        single_line = normalized.strip()
        return [single_line] if single_line else []

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _build_change_set(from_paragraphs: list[str], to_paragraphs: list[str]) -> tuple[list[dict[str, Any]], int, int, int]:
        matcher = difflib.SequenceMatcher(a=from_paragraphs, b=to_paragraphs)
        changes: list[dict[str, Any]] = []
        added_count = 0
        deleted_count = 0
        modified_count = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            old_text = "\n\n".join(from_paragraphs[i1:i2]) or None
            new_text = "\n\n".join(to_paragraphs[j1:j2]) or None
            if tag == "insert":
                added_count += max(j2 - j1, 1)
                change_type = "insert"
            elif tag == "delete":
                deleted_count += max(i2 - i1, 1)
                change_type = "delete"
            else:
                modified_count += max(i2 - i1, j2 - j1, 1)
                change_type = "replace"
            changes.append(
                {
                    "change_type": change_type,
                    "from_paragraph_start": i1 + 1 if i1 != i2 else None,
                    "from_paragraph_end": i2 if i1 != i2 else None,
                    "to_paragraph_start": j1 + 1 if j1 != j2 else None,
                    "to_paragraph_end": j2 if j1 != j2 else None,
                    "old_text": old_text,
                    "new_text": new_text,
                }
            )
        return changes, added_count, deleted_count, modified_count

    @staticmethod
    def _build_unified_diff(from_paragraphs: list[str], to_paragraphs: list[str], from_version_number: int, to_version_number: int) -> str:
        diff_lines = list(
            difflib.unified_diff(
                from_paragraphs,
                to_paragraphs,
                fromfile=f"version_{from_version_number}",
                tofile=f"version_{to_version_number}",
                lineterm="",
                n=2,
            )
        )
        return "\n".join(diff_lines)

    def _infer_impact_hints(
        self,
        changes: list[dict[str, Any]],
        added_count: int,
        deleted_count: int,
        modified_count: int,
    ) -> list[str]:
        changed_text = " ".join(filter(None, [change.get("old_text") for change in changes] + [change.get("new_text") for change in changes])).lower()
        hints: list[str] = []
        for keyword, hint in KEYWORD_HINTS.items():
            if keyword in changed_text:
                hints.append(hint)
        if modified_count > 0:
            hints.append("Existing guidance changed. Recheck downstream FAQs, runbooks, and cached answers.")
        if added_count >= 3:
            hints.append("Several new paragraphs were added. Verify whether onboarding or operating procedures need updating.")
        if deleted_count >= 3:
            hints.append("Several paragraphs were removed. Confirm no dependent checklist or communication still references deleted guidance.")
        if not hints:
            hints.append("No high-risk keywords were detected, but consumers should still review the highlighted additions, deletions, and modifications.")
        deduped: list[str] = []
        seen: set[str] = set()
        for hint in hints:
            if hint in seen:
                continue
            seen.add(hint)
            deduped.append(hint)
        return deduped[:6]

    @staticmethod
    def _serialize_diff(diff: ComputedDiff) -> DocumentDiffRead:
        return DocumentDiffRead(
            document_id=diff.document.id,
            from_version_id=diff.from_version.id,
            to_version_id=diff.to_version.id,
            from_version_number=diff.from_version.version_number,
            to_version_number=diff.to_version.version_number,
            added_count=diff.added_count,
            deleted_count=diff.deleted_count,
            modified_count=diff.modified_count,
            unified_diff=diff.unified_diff,
            changes=[DocumentDiffChangeRead(**change) for change in diff.changes],
            impact_hints=diff.impact_hints,
        )
