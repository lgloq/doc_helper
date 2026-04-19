from __future__ import annotations

import logging

from sqlalchemy.exc import SQLAlchemyError

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.models.eval import EvalCase
from app.repositories.eval_repository import EvalRepository
from app.services.eval.demo_cases import DEMO_EVAL_CASES

logger = logging.getLogger(__name__)


def seed_demo_eval_cases() -> None:
    settings = get_settings()
    if not settings.seed_demo_eval_cases:
        return

    session = SessionLocal()
    try:
        repository = EvalRepository(session)
        dataset_target_names: dict[str, set[str]] = {}
        for item in DEMO_EVAL_CASES:
            dataset_target_names.setdefault(item["dataset_name"], set()).add(item["case_name"])
            candidate_case_names = [item["case_name"], *item.get("legacy_case_names", [])]
            existing = next(
                (
                    case
                    for case in repository.list_cases(item["dataset_name"])
                    if case.is_demo_case and case.case_name in candidate_case_names
                ),
                None,
            )
            if existing is None:
                repository.add_case(
                    EvalCase(
                        dataset_name=item["dataset_name"],
                        case_name=item["case_name"],
                        description=item.get("description"),
                        acting_user_email=item["acting_user_email"],
                        question=item["question"],
                        expected_document_titles=item.get("expected_document_titles", []),
                        forbidden_document_titles=item.get("forbidden_document_titles", []),
                        expected_answer_keywords=item.get("expected_answer_keywords", []),
                        notes=item.get("notes"),
                        is_demo_case=True,
                    )
                )
                continue

            existing.case_name = item["case_name"]
            existing.description = item.get("description")
            existing.acting_user_email = item["acting_user_email"]
            existing.question = item["question"]
            existing.expected_document_titles = item.get("expected_document_titles", [])
            existing.forbidden_document_titles = item.get("forbidden_document_titles", [])
            existing.expected_answer_keywords = item.get("expected_answer_keywords", [])
            existing.notes = item.get("notes")
            existing.is_demo_case = True

        for dataset_name, target_case_names in dataset_target_names.items():
            for case in repository.list_cases(dataset_name):
                if case.is_demo_case and case.case_name not in target_case_names:
                    session.delete(case)

        session.commit()
    except SQLAlchemyError:
        session.rollback()
        logger.exception("Unable to seed demo eval cases.")
    finally:
        session.close()
