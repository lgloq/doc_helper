from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.eval import EvalCase, EvalResult, EvalRun


class EvalRepository:
    def __init__(self, session: Session):
        self.session = session

    def add_case(self, case: EvalCase) -> EvalCase:
        self.session.add(case)
        return case

    def list_cases(self, dataset_name: str | None = None) -> list[EvalCase]:
        statement = select(EvalCase)
        if dataset_name:
            statement = statement.where(EvalCase.dataset_name == dataset_name)
        statement = statement.order_by(EvalCase.dataset_name.asc(), EvalCase.case_name.asc())
        return list(self.session.scalars(statement).all())

    def get_cases_by_ids(self, case_ids: list[UUID]) -> list[EvalCase]:
        if not case_ids:
            return []
        statement = select(EvalCase).where(EvalCase.id.in_(case_ids)).order_by(EvalCase.case_name.asc())
        return list(self.session.scalars(statement).all())

    def count_demo_cases(self, dataset_name: str) -> int:
        statement = select(EvalCase).where(EvalCase.dataset_name == dataset_name, EvalCase.is_demo_case.is_(True))
        return len(list(self.session.scalars(statement).all()))

    def add_run(self, run: EvalRun) -> EvalRun:
        self.session.add(run)
        return run

    def add_results(self, results: list[EvalResult]) -> None:
        for result in results:
            self.session.add(result)

    def list_runs(self) -> list[EvalRun]:
        statement = select(EvalRun).order_by(EvalRun.created_at.desc())
        return list(self.session.scalars(statement).all())

    def get_run(self, run_id: UUID) -> EvalRun | None:
        statement = select(EvalRun).where(EvalRun.id == run_id)
        return self.session.scalar(statement)

    def list_results_for_run(self, run_id: UUID) -> list[EvalResult]:
        statement = select(EvalResult).where(EvalResult.run_id == run_id).order_by(EvalResult.created_at.asc())
        return list(self.session.scalars(statement).all())
