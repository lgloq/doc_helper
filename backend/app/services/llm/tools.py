from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

from app.models.chat import ChatMessage
from app.models.user import User
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import DocumentDiffRead, DocumentDiffRequest, DocumentDiffSummaryRead
from app.schemas.llm import RouterAccessibleDocument
from app.schemas.search import SearchDebugInfo, SearchRequest, SearchResponse
from app.schemas.workflow import FAQGenerateRequest, SourceCitationRead, TaskExtractRequest, WeeklyReportGenerateRequest
from app.services.diff.service import DocumentDiffService
from app.services.documents.service import DocumentService
from app.services.faqs.service import FAQService
from app.services.reports.service import WeeklyReportService
from app.services.retrieval.service import RetrievalService
from app.services.tasks.service import TaskService


@dataclass
class DocumentContextToolResult:
    document_id: UUID | None
    document_title: str | None
    retrieval_response: SearchResponse
    refusal_reason: str | None = None


@dataclass
class VersionCompareToolResult:
    document_id: UUID | None
    document_title: str | None
    diff: DocumentDiffRead | None
    summary: DocumentDiffSummaryRead | None
    refusal_reason: str | None = None


@dataclass
class WorkflowToolResult:
    artifact_type: str
    structured_payload: dict
    citations: list[SourceCitationRead]
    refusal_reason: str | None = None


class CopilotToolService:
    def __init__(self, session) -> None:
        self.session = session
        self.document_repository = DocumentRepository(session)
        self.document_service = DocumentService(session)
        self.retrieval_service = RetrievalService(session)
        self.diff_service = DocumentDiffService(session)
        self.task_service = TaskService(session)
        self.report_service = WeeklyReportService(session)
        self.faq_service = FAQService(session)

    def list_accessible_documents(self, actor: User) -> list[RouterAccessibleDocument]:
        documents = self.document_service.list_visible_documents(actor)
        return [RouterAccessibleDocument(document_id=document.id, title=document.title) for document in documents]

    def search_accessible_documents(self, actor: User, query: str, top_k: int) -> SearchResponse:
        return self.retrieval_service.search(actor, SearchRequest(query=query, top_k=top_k))

    def get_document_context(self, actor: User, document_title_or_id: str | UUID | None, query: str, top_k: int) -> DocumentContextToolResult:
        resolved = self._resolve_accessible_document(actor, document_title_or_id)
        if resolved is None:
            return DocumentContextToolResult(
                document_id=None,
                document_title=None,
                retrieval_response=self._empty_search_response(query, top_k),
                refusal_reason="target_document_not_accessible_or_not_found",
            )
        retrieval_response = self.retrieval_service.search(
            actor,
            SearchRequest(query=query, top_k=top_k),
            scoped_document_ids=[resolved.document_id],
        )
        if not retrieval_response.matched_chunks:
            return DocumentContextToolResult(
                document_id=resolved.document_id,
                document_title=resolved.title,
                retrieval_response=retrieval_response,
                refusal_reason="no_relevant_evidence_in_target_document",
            )
        return DocumentContextToolResult(
            document_id=resolved.document_id,
            document_title=resolved.title,
            retrieval_response=retrieval_response,
        )

    def compare_document_versions(
        self,
        actor: User,
        document_title_or_id: str | UUID | None,
        from_version_ref: str | None,
        to_version_ref: str | None,
    ) -> VersionCompareToolResult:
        resolved = self._resolve_accessible_document(actor, document_title_or_id)
        if resolved is None:
            return VersionCompareToolResult(
                document_id=None,
                document_title=None,
                diff=None,
                summary=None,
                refusal_reason="target_document_not_accessible_or_not_found",
            )

        versions = list(self.document_repository.list_versions(resolved.document_id))
        if len(versions) < 2:
            return VersionCompareToolResult(
                document_id=resolved.document_id,
                document_title=resolved.title,
                diff=None,
                summary=None,
                refusal_reason="insufficient_versions_for_compare",
            )

        from_version, to_version = self._resolve_version_pair(versions, from_version_ref, to_version_ref)
        if from_version is None or to_version is None or from_version.id == to_version.id:
            return VersionCompareToolResult(
                document_id=resolved.document_id,
                document_title=resolved.title,
                diff=None,
                summary=None,
                refusal_reason="unable_to_resolve_version_pair",
            )

        diff = self.diff_service.get_diff(actor, resolved.document_id, from_version.id, to_version.id)
        summary = self.diff_service.summarize_diff(
            actor,
            resolved.document_id,
            DocumentDiffRequest(from_version_id=from_version.id, to_version_id=to_version.id),
        )
        return VersionCompareToolResult(
            document_id=resolved.document_id,
            document_title=resolved.title,
            diff=diff,
            summary=summary,
        )

    def generate_tasks_from_session(self, actor: User, session_id: UUID) -> WorkflowToolResult:
        response = self.task_service.extract_tasks(actor, TaskExtractRequest(session_id=session_id))
        citations = [citation for item in response.items for citation in (item.source_citations or [])]
        return WorkflowToolResult(
            artifact_type="tasks",
            structured_payload=response.model_dump(mode="json"),
            citations=citations,
        )

    def generate_weekly_report_from_session(self, actor: User, session_id: UUID) -> WorkflowToolResult:
        response = self.report_service.generate_report(actor, WeeklyReportGenerateRequest(session_id=session_id))
        return WorkflowToolResult(
            artifact_type="weekly_report",
            structured_payload=response.model_dump(mode="json"),
            citations=list(response.report.reference_sources),
        )

    def generate_faq_from_session(self, actor: User, session_id: UUID) -> WorkflowToolResult:
        response = self.faq_service.generate_faqs(actor, FAQGenerateRequest(session_id=session_id))
        citations = [citation for entry in response.entries for citation in entry.source_citations]
        return WorkflowToolResult(
            artifact_type="faq",
            structured_payload=response.model_dump(mode="json"),
            citations=citations,
        )

    def generate_tasks_from_messages(
        self,
        actor: User,
        messages: list[ChatMessage],
        source_session_id: UUID | None,
        max_items: int = 8,
    ) -> WorkflowToolResult:
        items = self.task_service._build_task_items(actor, messages, source_session_id, max_items)
        self.task_service.artifact_repository.add_task_items(items)
        self.session.flush()
        citations = [citation for item in items for citation in (item.source_citations or [])]
        return WorkflowToolResult(
            artifact_type="tasks",
            structured_payload={
                "items": [self.task_service._serialize_task_item(item).model_dump(mode="json") for item in items],
            },
            citations=citations,
        )

    def generate_weekly_report_from_messages(
        self,
        actor: User,
        messages: list[ChatMessage],
        source_session_id: UUID | None,
        title: str | None = None,
    ) -> WorkflowToolResult:
        report = self.report_service._build_report(actor, messages, source_session_id, title)
        self.report_service.artifact_repository.add_weekly_report(report)
        self.session.flush()
        return WorkflowToolResult(
            artifact_type="weekly_report",
            structured_payload={
                "report": self.report_service._serialize_report(report).model_dump(mode="json"),
            },
            citations=list(report.reference_sources),
        )

    def generate_faq_from_messages(
        self,
        actor: User,
        messages: list[ChatMessage],
        source_session_id: UUID | None,
        max_entries: int = 5,
    ) -> WorkflowToolResult:
        entries = self.faq_service._build_entries(actor, messages, source_session_id, max_entries)
        self.faq_service.artifact_repository.add_faq_entries(entries)
        self.session.flush()
        citations = [citation for entry in entries for citation in entry.source_citations]
        return WorkflowToolResult(
            artifact_type="faq",
            structured_payload={
                "entries": [self.faq_service._serialize_entry(entry).model_dump(mode="json") for entry in entries],
            },
            citations=citations,
        )

    def _resolve_accessible_document(self, actor: User, document_title_or_id: str | UUID | None) -> RouterAccessibleDocument | None:
        if document_title_or_id is None:
            return None
        accessible_documents = self.list_accessible_documents(actor)
        value = str(document_title_or_id).strip()
        if not value:
            return None

        by_id = {str(item.document_id): item for item in accessible_documents}
        if value in by_id:
            return by_id[value]

        lowered = value.casefold()
        normalized = _normalize_title(value)
        exact = next((item for item in accessible_documents if item.title.casefold() == lowered), None)
        if exact is not None:
            return exact

        normalized_matches = [item for item in accessible_documents if _normalize_title(item.title) == normalized]
        if normalized_matches:
            return normalized_matches[0]

        contains_matches = [item for item in accessible_documents if item.title.casefold() in lowered or lowered in item.title.casefold()]
        if contains_matches:
            return contains_matches[0]
        return None

    @staticmethod
    def _resolve_version_pair(versions, from_version_ref: str | None, to_version_ref: str | None):
        ordered = sorted(versions, key=lambda item: item.version_number, reverse=True)
        latest = ordered[0]
        previous = ordered[1] if len(ordered) > 1 else None

        from_version = _resolve_version_ref(ordered, from_version_ref)
        to_version = _resolve_version_ref(ordered, to_version_ref)

        if from_version is None and to_version is None:
            return previous, latest
        if from_version is None:
            if to_version and to_version.id == latest.id and previous is not None:
                return previous, latest
            return previous or latest, to_version
        if to_version is None:
            if from_version.id == latest.id and previous is not None:
                return previous, latest
            return from_version, latest
        return from_version, to_version

    @staticmethod
    def _empty_search_response(query: str, top_k: int) -> SearchResponse:
        return SearchResponse(
            query=query,
            top_k=top_k,
            matched_chunks=[],
            debug=SearchDebugInfo(
                accessible_document_count=0,
                lexical_candidate_count=0,
                vector_candidate_count=0,
                fusion_strategy="min-max weighted sum",
            ),
        )



def _normalize_title(value: str) -> str:
    lowered = value.casefold()
    lowered = re.sub(r"[《》\-_/,.?？!！:：;；'\"\(\)\[\]]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    for suffix in (" document", " doc", " handbook", " guide", " runbook", " policy", " 文档", " 手册", " 指南", " 登记"):
        if lowered.endswith(suffix):
            lowered = lowered[: -len(suffix)].strip()
    return lowered



def _resolve_version_ref(versions, version_ref: str | None):
    if not version_ref:
        return None
    lowered = version_ref.strip().casefold()
    if lowered in {"latest", "newest", "current", "最新", "当前"}:
        return versions[0] if versions else None
    if lowered in {"previous", "prior", "上一版", "前一版"}:
        return versions[1] if len(versions) > 1 else None
    match = re.match(r"v\s*(\d+)", lowered)
    if match:
        target_number = int(match.group(1))
        return next((item for item in versions if item.version_number == target_number), None)
    if lowered.isdigit():
        target_number = int(lowered)
        return next((item for item in versions if item.version_number == target_number), None)
    return None
