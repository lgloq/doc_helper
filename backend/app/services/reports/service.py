from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.models.chat import ChatMessage
from app.models.enums import MessageRole
from app.models.user import User
from app.models.workflow import WeeklyReportDraft
from app.repositories.artifact_repository import ArtifactRepository
from app.schemas.workflow import WeeklyReportDraftRead, WeeklyReportGenerateRequest, WeeklyReportGenerateResponse
from app.services.workflows.source_resolver import SourceMaterialResolver, unique_serialized_citations
from app.services.workflows.utils import (
    compact_text,
    infer_priority,
    is_actionable_sentence,
    iter_question_answer_pairs,
    looks_like_risk,
    normalize_title,
    split_into_sentences,
    unique_strings,
)


class WeeklyReportService:
    def __init__(self, session: Session):
        self.session = session
        self.artifact_repository = ArtifactRepository(session)
        self.source_resolver = SourceMaterialResolver(session)

    def generate_report(self, actor: User, payload: WeeklyReportGenerateRequest) -> WeeklyReportGenerateResponse:
        bundle = self.source_resolver.resolve(actor, payload)
        report = self._build_report(actor, bundle.messages, bundle.session.id if bundle.session else None, payload.title)
        self.artifact_repository.add_weekly_report(report)
        self.session.commit()
        self.session.refresh(report)
        return WeeklyReportGenerateResponse(report=self._serialize_report(report))

    def list_reports(self, actor: User) -> list[WeeklyReportDraftRead]:
        reports = self.artifact_repository.list_weekly_reports_for_user(actor.id)
        return [self._serialize_report(report) for report in reports]

    def _build_report(self, actor: User, messages: list[ChatMessage], source_session_id, title: str | None) -> WeeklyReportDraft:
        qa_pairs = iter_question_answer_pairs(messages)
        completed_items: list[str] = []
        risks_blockers: list[str] = []

        for user_message, assistant_message in qa_pairs:
            if assistant_message.insufficient_evidence:
                risks_blockers.append(f"Need more evidence for: {compact_text(user_message.content, 120)}")
                continue
            completed_items.append(f"Clarified: {compact_text(user_message.content, 140)}")
            if assistant_message.message_metadata and assistant_message.message_metadata.get("evidence_conflict"):
                risks_blockers.append(f"Evidence conflict needs review for: {compact_text(user_message.content, 120)}")
            if looks_like_risk(assistant_message.content):
                risks_blockers.append(compact_text(assistant_message.content, 180))

        next_week_plan = self._build_next_week_plan(messages)
        reference_sources = unique_serialized_citations([message for message in messages if message.role == MessageRole.ASSISTANT], limit=12)

        if not completed_items:
            completed_items = ["No grounded completed items were identified from the selected Q&A."]
        if not risks_blockers:
            risks_blockers = ["No explicit blockers were detected from the selected Q&A."]
        if not next_week_plan:
            next_week_plan = ["Review recent Q&A and convert any pending decisions into concrete follow-up actions."]

        summary = (
            f"Generated from {len(messages)} messages with {len(reference_sources)} reference sources. "
            f"Completed items: {len(completed_items)}. Risks: {len(risks_blockers)}. Next steps: {len(next_week_plan)}."
        )
        report_title = title or f"Weekly Report Draft {datetime.now().date().isoformat()}"
        return WeeklyReportDraft(
            created_by_user_id=actor.id,
            source_session_id=source_session_id,
            title=report_title,
            summary=summary,
            completed_this_week=unique_strings(completed_items),
            risks_blockers=unique_strings(risks_blockers),
            next_week_plan=unique_strings(next_week_plan),
            reference_sources=reference_sources,
            source_message_ids=[str(message.id) for message in messages],
            status="draft",
        )

    @staticmethod
    def _build_next_week_plan(messages: list[ChatMessage]) -> list[str]:
        suggestions: list[str] = []
        for message in messages:
            if message.role != MessageRole.ASSISTANT or message.insufficient_evidence:
                continue
            for sentence in split_into_sentences(message.content):
                if not is_actionable_sentence(sentence):
                    continue
                priority = infer_priority(sentence)
                suggestions.append(f"[{priority}] {normalize_title(sentence)}")
            for citation in getattr(message, "citations", []):
                for sentence in split_into_sentences(citation.preview):
                    if not is_actionable_sentence(sentence):
                        continue
                    priority = infer_priority(sentence)
                    suggestions.append(f"[{priority}] {normalize_title(sentence)}")
        return unique_strings(suggestions)[:8]

    @staticmethod
    def _serialize_report(report: WeeklyReportDraft) -> WeeklyReportDraftRead:
        return WeeklyReportDraftRead.model_validate(report)
