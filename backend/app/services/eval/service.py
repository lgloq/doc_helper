from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.eval import EvalResult, EvalRun
from app.models.enums import RoleName
from app.models.user import User
from app.repositories.eval_repository import EvalRepository
from app.repositories.user_repository import UserRepository
from app.schemas.eval import EvalRunDetailRead, EvalRunRead, EvalRunRequest, EvalResultRowRead
from app.services.chat.service import ChatService, PreparedChatAnswer
from app.services.eval.bootstrap import seed_demo_eval_cases
from app.services.observability.service import ObservabilityService


class EvalService:
    def __init__(self, session: Session):
        self.session = session
        self.eval_repository = EvalRepository(session)
        self.user_repository = UserRepository(session)
        self.chat_service = ChatService(session)
        self.observability_service = ObservabilityService(session)

    def run_eval(self, actor: User, payload: EvalRunRequest) -> EvalRunDetailRead:
        self._ensure_admin(actor)
        if payload.seed_demo_cases:
            seed_demo_eval_cases()

        cases = self._select_cases(payload)
        if not cases:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No eval cases found for the requested dataset.")

        run = EvalRun(
            dataset_name=payload.dataset_name,
            status="running",
            total_cases=len(cases),
            started_at=datetime.now(UTC),
        )
        self.eval_repository.add_run(run)
        self.session.commit()
        self.session.refresh(run)

        results: list[EvalResult] = []
        for case in cases:
            results.append(self._evaluate_case(run, case, payload.top_k))

        self.eval_repository.add_results(results)
        run.status = "completed"
        run.finished_at = datetime.now(UTC)
        run.summary_json = self._build_summary(results)
        self.session.commit()
        self.session.refresh(run)

        return self.get_run(actor, run.id)

    def list_runs(self, actor: User) -> list[EvalRunRead]:
        self._ensure_admin(actor)
        runs = self.eval_repository.list_runs()
        return [EvalRunRead.model_validate(item) for item in runs]

    def get_run(self, actor: User, run_id: UUID) -> EvalRunDetailRead:
        self._ensure_admin(actor)
        run = self.eval_repository.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Eval run not found.")
        results = self.eval_repository.list_results_for_run(run_id)
        return EvalRunDetailRead.model_validate(
            {
                **EvalRunRead.model_validate(run).model_dump(),
                "results": [EvalResultRowRead.model_validate(item) for item in results],
            }
        )

    def _select_cases(self, payload: EvalRunRequest):
        if payload.case_ids:
            return self.eval_repository.get_cases_by_ids(payload.case_ids)
        return self.eval_repository.list_cases(payload.dataset_name)

    def _evaluate_case(self, run: EvalRun, case, top_k: int) -> EvalResult:
        actor = self.user_repository.get_by_email(case.acting_user_email)
        if actor is None:
            details = {
                "case_name": case.case_name,
                "dataset_name": case.dataset_name,
                "error": f"acting user {case.acting_user_email} was not found",
                "expected_document_titles": case.expected_document_titles,
                "forbidden_document_titles": case.forbidden_document_titles,
                "human_review": {
                    "recommended": True,
                    "reason": "eval case references a missing acting user",
                },
            }
            return EvalResult(
                run_id=run.id,
                case_id=case.id,
                acting_user_email=case.acting_user_email,
                retrieval_hit_rate=0.0,
                citation_accuracy=0.0,
                answer_faithfulness=0.0,
                permission_isolation_correct=False,
                overall_pass=False,
                details_json=details,
            )

        prepared = self.chat_service.preview_answer(actor, case.question, top_k=top_k)
        metrics = self._compute_metrics(case, prepared)
        trace_id = None
        try:
            trace = self.observability_service.record_trace(
                actor=actor,
                chat_session=None,
                user_message=None,
                assistant_message=None,
                query_text=case.question,
                retrieval_response=prepared.retrieval_response,
                selected_chunks=prepared.selected_chunks,
                error_text=self._extract_eval_error(prepared),
                trace_type="eval_case",
                confidence=prepared.confidence,
                insufficient_evidence=prepared.answer_result.insufficient_evidence,
                extra_metadata={
                    "eval_run_id": str(run.id),
                    "eval_case_id": str(case.id),
                    "case_name": case.case_name,
                    "dataset_name": case.dataset_name,
                    "permission_isolation_correct": metrics["permission_isolation_correct"],
                },
                model_name=prepared.answer_result.model_name,
                latency_ms=prepared.answer_result.latency_ms,
                prompt_tokens=prepared.answer_result.prompt_tokens,
                completion_tokens=prepared.answer_result.completion_tokens,
            )
            trace_id = str(trace.id)
        except Exception:
            trace_id = None

        details = {
            "case_name": case.case_name,
            "dataset_name": case.dataset_name,
            "description": case.description,
            "question": case.question,
            "expected_document_titles": case.expected_document_titles,
            "forbidden_document_titles": case.forbidden_document_titles,
            "expected_answer_keywords": case.expected_answer_keywords,
            "retrieved_document_titles": metrics["retrieved_document_titles"],
            "citation_document_titles": metrics["citation_document_titles"],
            "matched_answer_keywords": metrics["matched_answer_keywords"],
            "missing_answer_keywords": metrics["missing_answer_keywords"],
            "answer_text": prepared.answer_result.answer,
            "answer_excerpt": prepared.answer_result.answer[:400],
            "confidence": prepared.confidence,
            "insufficient_evidence": prepared.answer_result.insufficient_evidence,
            "evidence_conflict": prepared.answer_result.evidence_conflict,
            "permission_checks": metrics["permission_checks"],
            "retrieval_debug": prepared.retrieval_response.debug.model_dump(),
            "trace_id": trace_id,
            "human_review": {
                "recommended": metrics["human_review_recommended"],
                "reason": metrics["human_review_reason"],
            },
        }
        return EvalResult(
            run_id=run.id,
            case_id=case.id,
            acting_user_email=case.acting_user_email,
            retrieval_hit_rate=metrics["retrieval_hit_rate"],
            citation_accuracy=metrics["citation_accuracy"],
            answer_faithfulness=metrics["answer_faithfulness"],
            permission_isolation_correct=metrics["permission_isolation_correct"],
            overall_pass=metrics["overall_pass"],
            details_json=details,
        )

    def _compute_metrics(self, case, prepared: PreparedChatAnswer) -> dict:
        expected_titles = self._normalize_strings(case.expected_document_titles)
        forbidden_titles = self._normalize_strings(case.forbidden_document_titles)
        expected_keywords = self._normalize_strings(case.expected_answer_keywords)
        retrieved_titles = self._normalize_strings(item.document_title for item in prepared.retrieval_response.matched_chunks)
        citation_titles = self._normalize_strings(item.document_title for item in prepared.selected_chunks)
        answer_text = prepared.answer_result.answer or ""
        answer_text_lower = answer_text.lower()
        matched_keywords = sorted(keyword for keyword in expected_keywords if keyword in answer_text_lower)
        missing_keywords = sorted(keyword for keyword in expected_keywords if keyword not in answer_text_lower)

        forbidden_in_retrieval = sorted(forbidden_titles.intersection(retrieved_titles))
        forbidden_in_citations = sorted(forbidden_titles.intersection(citation_titles))
        forbidden_in_answer = sorted(
            title for title in forbidden_titles if title and title in answer_text_lower
        )
        permission_isolation_correct = not forbidden_in_retrieval and not forbidden_in_citations and not forbidden_in_answer

        retrieval_hit_rate = self._compute_retrieval_hit_rate(expected_titles, forbidden_titles, retrieved_titles)
        citation_accuracy = self._compute_citation_accuracy(expected_titles, forbidden_titles, citation_titles)
        answer_faithfulness = self._compute_answer_faithfulness(
            expected_titles=expected_titles,
            expected_keywords=expected_keywords,
            matched_keywords=matched_keywords,
            prepared=prepared,
            permission_isolation_correct=permission_isolation_correct,
        )
        overall_pass = self._compute_overall_pass(
            expected_titles=expected_titles,
            retrieval_hit_rate=retrieval_hit_rate,
            citation_accuracy=citation_accuracy,
            answer_faithfulness=answer_faithfulness,
            permission_isolation_correct=permission_isolation_correct,
        )

        human_review_recommended = (
            not overall_pass
            or prepared.answer_result.evidence_conflict
            or prepared.confidence in {"low", "insufficient"}
        )
        if not permission_isolation_correct:
            human_review_reason = "forbidden content appeared in retrieval, citations, or answer text"
        elif prepared.answer_result.evidence_conflict:
            human_review_reason = "top evidence conflicts across sources"
        elif prepared.confidence in {"low", "insufficient"}:
            human_review_reason = "confidence is low or evidence is insufficient"
        else:
            human_review_reason = "no immediate review required"

        return {
            "retrieval_hit_rate": retrieval_hit_rate,
            "citation_accuracy": citation_accuracy,
            "answer_faithfulness": answer_faithfulness,
            "permission_isolation_correct": permission_isolation_correct,
            "overall_pass": overall_pass,
            "retrieved_document_titles": sorted(retrieved_titles),
            "citation_document_titles": sorted(citation_titles),
            "matched_answer_keywords": matched_keywords,
            "missing_answer_keywords": missing_keywords,
            "permission_checks": {
                "forbidden_in_retrieval": forbidden_in_retrieval,
                "forbidden_in_citations": forbidden_in_citations,
                "forbidden_in_answer": forbidden_in_answer,
            },
            "human_review_recommended": human_review_recommended,
            "human_review_reason": human_review_reason,
        }

    @staticmethod
    def _compute_retrieval_hit_rate(expected_titles: set[str], forbidden_titles: set[str], retrieved_titles: set[str]) -> float:
        if not expected_titles:
            return 1.0 if not forbidden_titles.intersection(retrieved_titles) else 0.0
        matched = expected_titles.intersection(retrieved_titles)
        return len(matched) / len(expected_titles)

    @staticmethod
    def _compute_citation_accuracy(expected_titles: set[str], forbidden_titles: set[str], citation_titles: set[str]) -> float:
        if forbidden_titles.intersection(citation_titles):
            return 0.0
        if not expected_titles:
            return 1.0 if not citation_titles else 0.0
        if not citation_titles:
            return 0.0
        matched = expected_titles.intersection(citation_titles)
        return len(matched) / len(citation_titles)

    @staticmethod
    def _compute_answer_faithfulness(
        *,
        expected_titles: set[str],
        expected_keywords: set[str],
        matched_keywords: list[str],
        prepared: PreparedChatAnswer,
        permission_isolation_correct: bool,
    ) -> float:
        if not permission_isolation_correct:
            return 0.0
        if not expected_titles:
            return 1.0 if prepared.answer_result.insufficient_evidence else 0.35

        citation_support = 1.0 if prepared.selected_chunks else 0.0
        keyword_coverage = 1.0 if not expected_keywords else len(matched_keywords) / len(expected_keywords)
        confidence_bonus = 0.2 if prepared.confidence == "high" else 0.1 if prepared.confidence == "medium" else 0.0
        score = (0.5 * citation_support) + (0.3 * keyword_coverage) + confidence_bonus
        if prepared.answer_result.insufficient_evidence:
            score -= 0.35
        if prepared.answer_result.evidence_conflict:
            score -= 0.15
        return max(0.0, min(score, 1.0))

    @staticmethod
    def _compute_overall_pass(
        *,
        expected_titles: set[str],
        retrieval_hit_rate: float,
        citation_accuracy: float,
        answer_faithfulness: float,
        permission_isolation_correct: bool,
    ) -> bool:
        if not permission_isolation_correct:
            return False
        if not expected_titles:
            return answer_faithfulness >= 0.9 and citation_accuracy >= 1.0
        return retrieval_hit_rate >= 0.5 and citation_accuracy >= 0.5 and answer_faithfulness >= 0.55

    @staticmethod
    def _build_summary(results: list[EvalResult]) -> dict:
        if not results:
            return {
                "total_cases": 0,
                "pass_count": 0,
                "retrieval_hit_rate_avg": 0.0,
                "citation_accuracy_avg": 0.0,
                "answer_faithfulness_avg": 0.0,
                "permission_isolation_pass_rate": 0.0,
            }
        total_cases = len(results)
        pass_count = sum(1 for item in results if item.overall_pass)
        permission_pass_count = sum(1 for item in results if item.permission_isolation_correct)
        return {
            "total_cases": total_cases,
            "pass_count": pass_count,
            "retrieval_hit_rate_avg": round(sum(item.retrieval_hit_rate for item in results) / total_cases, 4),
            "citation_accuracy_avg": round(sum(item.citation_accuracy for item in results) / total_cases, 4),
            "answer_faithfulness_avg": round(sum(item.answer_faithfulness for item in results) / total_cases, 4),
            "permission_isolation_pass_rate": round(permission_pass_count / total_cases, 4),
        }

    @staticmethod
    def _extract_eval_error(prepared: PreparedChatAnswer) -> str | None:
        raw_payload = prepared.answer_result.raw_payload or {}
        error_text = raw_payload.get("error_text")
        return str(error_text) if error_text else None

    @staticmethod
    def _normalize_strings(values) -> set[str]:
        normalized: set[str] = set()
        for item in values:
            if item is None:
                continue
            cleaned = str(item).strip().lower()
            if cleaned:
                normalized.add(cleaned)
        return normalized

    @staticmethod
    def _ensure_admin(actor: User) -> None:
        if actor.role is None or actor.role.name != RoleName.ADMIN:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin permission required.")
