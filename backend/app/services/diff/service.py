from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from fastapi import HTTPException, status
from redis.exceptions import RedisError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.redis import get_redis_client
from app.models.document import Document, DocumentVersion
from app.models.user import User
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import DocumentDiffChangeRead, DocumentDiffRead, DocumentDiffRequest, DocumentDiffSummaryRead
from app.services.diff.generation import DiffSummaryGeneratorFactory
from app.services.ingestion.file_storage import LocalDocumentStorage
from app.services.ingestion.parsers import DocumentParser
from app.services.permissions.service import PermissionFilterBuilder


KEYWORD_HINTS = {
    "security": "安全相关表述发生变化。建议重新检查访问控制、安全指引和事故处理流程。",
    "access": "访问或权限相关表述发生变化。建议复核入职开通、特权权限和审批流程。",
    "deadline": "时限相关表述发生变化。建议确认项目时间表和下游承诺是否需要调整。",
    "sla": "服务等级或服务预期发生变化。建议同步支持、运维和业务相关方。",
    "rollback": "回滚或恢复指引发生变化。建议重新核对发布和事故处理手册。",
    "release": "发布流程相关表述发生变化。建议重新检查发布清单和沟通计划。",
    "approval": "审批或签字要求发生变化。建议复核负责人和升级路径。",
    "policy": "制度或政策表述发生变化。建议同步检查 FAQ、培训材料和合规说明。",
    "compliance": "合规相关表述发生变化。建议确认制度文件与审计证据仍然一致。",
    "owner": "负责人或联系人信息发生变化。建议确认责任团队和支持联系方式。",
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
        self.settings = get_settings()

    def get_diff(self, actor: User, document_id: UUID, from_version_id: UUID, to_version_id: UUID) -> DocumentDiffRead:
        diff = self._compute_diff(actor, document_id, from_version_id, to_version_id)
        return self._serialize_diff(diff)

    def summarize_diff(self, actor: User, document_id: UUID, payload: DocumentDiffRequest) -> DocumentDiffSummaryRead:
        document = self._get_viewable_document(actor, document_id)
        from_version, to_version = self._get_versions(document, payload.from_version_id, payload.to_version_id)
        cache_key = self._build_summary_cache_key(document.id, from_version.id, to_version.id)
        if not payload.force_refresh:
            cached_summary = self._get_cached_summary(cache_key)
            if cached_summary is not None:
                return cached_summary

        diff = self._compute_diff_for_versions(document, from_version, to_version)
        summary = self.summary_generator.generate(
            from_version_number=diff.from_version.version_number,
            to_version_number=diff.to_version.version_number,
            diff_changes=diff.changes,
            unified_diff=diff.unified_diff,
        )
        result = DocumentDiffSummaryRead(
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
            cache_hit=False,
        )
        self._set_cached_summary(cache_key, result)
        return result

    def _compute_diff(self, actor: User, document_id: UUID, from_version_id: UUID, to_version_id: UUID) -> ComputedDiff:
        document = self._get_viewable_document(actor, document_id)
        from_version, to_version = self._get_versions(document, from_version_id, to_version_id)
        return self._compute_diff_for_versions(document, from_version, to_version)

    def _compute_diff_for_versions(
        self,
        document: Document,
        from_version: DocumentVersion,
        to_version: DocumentVersion,
    ) -> ComputedDiff:
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

    def _get_versions(self, document: Document, from_version_id: UUID, to_version_id: UUID) -> tuple[DocumentVersion, DocumentVersion]:
        from_version = self.document_repository.get_version(document.id, from_version_id)
        to_version = self.document_repository.get_version(document.id, to_version_id)
        if from_version is None or to_version is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document version not found.")
        return from_version, to_version

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
            hints.append("已有指引内容发生修改。建议同步复核相关 FAQ、运行手册和已缓存答案。")
        if added_count >= 3:
            hints.append("本次新增了较多段落。建议确认入职说明、流程规范或操作手册是否需要同步更新。")
        if deleted_count >= 3:
            hints.append("本次删除了较多段落。建议确认是否仍有清单、FAQ 或沟通材料引用了已删除内容。")
        if not hints:
            hints.append("未检测到明显高风险关键词，但仍建议结合新增、删除和修改内容进行人工复核。")
        deduped: list[str] = []
        seen: set[str] = set()
        for hint in hints:
            if hint in seen:
                continue
            seen.add(hint)
            deduped.append(hint)
        return deduped[:6]

    def _build_summary_cache_key(self, document_id: UUID, from_version_id: UUID, to_version_id: UUID) -> str:
        provider_name = getattr(self.summary_generator, "provider_name", "unknown")
        model_name = getattr(self.summary_generator, "model_name", None) or "default"
        return f"diff_summary:{document_id}:{from_version_id}:{to_version_id}:{provider_name}:{model_name}"

    def _get_cached_summary(self, cache_key: str) -> DocumentDiffSummaryRead | None:
        try:
            cached_payload = get_redis_client().get(cache_key)
        except RedisError:
            return None
        if not cached_payload:
            return None
        try:
            payload = json.loads(cached_payload)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        payload["cache_hit"] = True
        try:
            return DocumentDiffSummaryRead(**payload)
        except Exception:
            return None

    def _set_cached_summary(self, cache_key: str, summary: DocumentDiffSummaryRead) -> None:
        try:
            get_redis_client().setex(
                cache_key,
                self.settings.diff_summary_cache_ttl_seconds,
                summary.model_dump_json(),
            )
        except RedisError:
            return None

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
