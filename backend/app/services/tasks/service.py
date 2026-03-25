from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.chat import ChatMessage
from app.models.enums import MessageRole
from app.models.user import User
from app.models.workflow import TaskItem
from app.repositories.artifact_repository import ArtifactRepository
from app.schemas.workflow import TaskExtractRequest, TaskExtractResponse, TaskItemRead
from app.services.workflows.source_resolver import SourceMaterialResolver, serialize_message_citation
from app.services.workflows.utils import compact_text, infer_priority, is_actionable_sentence, normalize_title, split_into_sentences


class TaskService:
    def __init__(self, session: Session):
        self.session = session
        self.artifact_repository = ArtifactRepository(session)
        self.source_resolver = SourceMaterialResolver(session)

    def extract_tasks(self, actor: User, payload: TaskExtractRequest) -> TaskExtractResponse:
        bundle = self.source_resolver.resolve(actor, payload)
        items = self._build_task_items(actor, bundle.messages, bundle.session.id if bundle.session else None, payload.max_items)
        self.artifact_repository.add_task_items(items)
        self.session.commit()
        return TaskExtractResponse(items=[self._serialize_task_item(item) for item in items])

    def list_tasks(self, actor: User) -> list[TaskItemRead]:
        items = self.artifact_repository.list_task_items_for_user(actor.id)
        return [self._serialize_task_item(item) for item in items]

    def _build_task_items(
        self,
        actor: User,
        messages: list[ChatMessage],
        source_session_id,
        max_items: int,
    ) -> list[TaskItem]:
        assistant_messages = [message for message in messages if message.role == MessageRole.ASSISTANT and not message.insufficient_evidence]
        task_items: list[TaskItem] = []
        seen_titles: set[str] = set()

        for message in assistant_messages:
            candidate_sentences: list[str] = []
            candidate_sentences.extend(split_into_sentences(message.content))
            for citation in getattr(message, "citations", []):
                candidate_sentences.extend(split_into_sentences(citation.preview))

            for sentence in candidate_sentences:
                if not is_actionable_sentence(sentence):
                    continue
                title = normalize_title(sentence)
                dedupe_key = title.lower()
                if dedupe_key in seen_titles:
                    continue
                seen_titles.add(dedupe_key)
                citations = [serialize_message_citation(citation) for citation in getattr(message, "citations", [])[:3]]
                task_items.append(
                    TaskItem(
                        created_by_user_id=actor.id,
                        source_session_id=source_session_id,
                        source_message_id=message.id,
                        title=title,
                        description=compact_text(sentence, limit=400),
                        owner_name=None,
                        priority=infer_priority(sentence),
                        due_date=None,
                        status="open",
                        source_citations=citations or None,
                    )
                )
                if len(task_items) >= max_items:
                    return task_items

        if not task_items:
            latest_user_message = next((message for message in reversed(messages) if message.role == MessageRole.USER), None)
            if latest_user_message is None:
                return []
            fallback_title = normalize_title(f"Review follow-up for: {latest_user_message.content}")
            task_items.append(
                TaskItem(
                    created_by_user_id=actor.id,
                    source_session_id=source_session_id,
                    source_message_id=latest_user_message.id,
                    title=fallback_title,
                    description="No explicit action sentence was found, so this task was created as a manual follow-up placeholder.",
                    owner_name=None,
                    priority="low",
                    due_date=None,
                    status="open",
                    source_citations=None,
                )
            )
        return task_items

    @staticmethod
    def _serialize_task_item(item: TaskItem) -> TaskItemRead:
        return TaskItemRead.model_validate(item)
