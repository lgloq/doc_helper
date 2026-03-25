from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.user import User
from app.models.workflow import FAQEntry
from app.repositories.artifact_repository import ArtifactRepository
from app.schemas.workflow import FAQEntryRead, FAQGenerateRequest, FAQGenerateResponse
from app.services.workflows.source_resolver import SourceMaterialResolver, serialize_message_citation
from app.services.workflows.utils import compact_text, iter_question_answer_pairs


class FAQService:
    def __init__(self, session: Session):
        self.session = session
        self.artifact_repository = ArtifactRepository(session)
        self.source_resolver = SourceMaterialResolver(session)

    def generate_faqs(self, actor: User, payload: FAQGenerateRequest) -> FAQGenerateResponse:
        bundle = self.source_resolver.resolve(actor, payload)
        entries = self._build_entries(actor, bundle.messages, bundle.session.id if bundle.session else None, payload.max_entries)
        self.artifact_repository.add_faq_entries(entries)
        self.session.commit()
        return FAQGenerateResponse(entries=[self._serialize_entry(entry) for entry in entries])

    def list_faqs(self, actor: User) -> list[FAQEntryRead]:
        entries = self.artifact_repository.list_faq_entries_for_user(actor.id)
        return [self._serialize_entry(entry) for entry in entries]

    @staticmethod
    def _build_entries(actor: User, messages, source_session_id, max_entries: int) -> list[FAQEntry]:
        entries: list[FAQEntry] = []
        for user_message, assistant_message in iter_question_answer_pairs(messages):
            if assistant_message.insufficient_evidence:
                continue
            if not getattr(assistant_message, "citations", []):
                continue
            if (assistant_message.confidence or "").lower() not in {"high", "medium", "low"}:
                continue
            citations = [serialize_message_citation(citation) for citation in assistant_message.citations[:4]]
            entries.append(
                FAQEntry(
                    created_by_user_id=actor.id,
                    source_session_id=source_session_id,
                    source_message_id=assistant_message.id,
                    question=compact_text(user_message.content, 500),
                    answer=compact_text(assistant_message.content, 1200),
                    quality=assistant_message.confidence or "medium",
                    status="draft",
                    source_citations=citations,
                )
            )
            if len(entries) >= max_entries:
                break
        return entries

    @staticmethod
    def _serialize_entry(entry: FAQEntry) -> FAQEntryRead:
        return FAQEntryRead.model_validate(entry)
