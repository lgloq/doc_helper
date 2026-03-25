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
        for item in DEMO_EVAL_CASES:
            existing = [case for case in repository.list_cases(item["dataset_name"]) if case.case_name == item["case_name"]]
            if existing:
                continue
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
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        logger.exception("Unable to seed demo eval cases.")
    finally:
        session.close()
