from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.workflow import FAQEntry, TaskItem, WeeklyReportDraft


class ArtifactRepository:
    def __init__(self, session: Session):
        self.session = session

    def add_task_items(self, items: list[TaskItem]) -> None:
        for item in items:
            self.session.add(item)

    def list_task_items_for_user(self, user_id) -> list[TaskItem]:
        statement = (
            select(TaskItem)
            .where(TaskItem.created_by_user_id == user_id)
            .order_by(TaskItem.created_at.desc(), TaskItem.updated_at.desc())
        )
        return list(self.session.scalars(statement).all())

    def add_weekly_report(self, report: WeeklyReportDraft) -> WeeklyReportDraft:
        self.session.add(report)
        return report

    def list_weekly_reports_for_user(self, user_id) -> list[WeeklyReportDraft]:
        statement = (
            select(WeeklyReportDraft)
            .where(WeeklyReportDraft.created_by_user_id == user_id)
            .order_by(WeeklyReportDraft.created_at.desc(), WeeklyReportDraft.updated_at.desc())
        )
        return list(self.session.scalars(statement).all())

    def add_faq_entries(self, entries: list[FAQEntry]) -> None:
        for entry in entries:
            self.session.add(entry)

    def list_faq_entries_for_user(self, user_id) -> list[FAQEntry]:
        statement = (
            select(FAQEntry)
            .where(FAQEntry.created_by_user_id == user_id)
            .order_by(FAQEntry.created_at.desc(), FAQEntry.updated_at.desc())
        )
        return list(self.session.scalars(statement).all())
