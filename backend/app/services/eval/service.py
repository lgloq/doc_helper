from __future__ import annotations

import math
import re
import time
import unicodedata
from datetime import UTC, datetime, timedelta
from uuid import UUID

from fastapi import HTTPException, status
from openai import APIConnectionError, APITimeoutError
from sqlalchemy.orm import Session

from app.models.eval import EvalResult, EvalRun
from app.models.enums import RoleName
from app.models.user import User
from app.repositories.eval_repository import EvalRepository
from app.repositories.user_repository import UserRepository
from app.schemas.eval import EvalRunDetailRead, EvalRunRead, EvalRunRequest, EvalResultRowRead
from app.services.chat.service import ChatService, PreparedChatAnswer
from app.services.eval.bootstrap import seed_demo_eval_cases
from app.services.eval.demo_cases import resolve_demo_eval_annotation
from app.services.observability.service import ObservabilityService

STALE_EVAL_RUN_THRESHOLD = timedelta(minutes=5)


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
        try:
            for case in cases:
                result = self._evaluate_case_with_retry(run, case, payload.top_k)
                results.append(result)
                self.eval_repository.add_results([result])
                self.session.commit()

            run.status = "completed"
            run.finished_at = datetime.now(UTC)
            run.summary_json = self._build_summary(results)
            run.error_text = None
            self.session.commit()
            self.session.refresh(run)
        except Exception as exc:
            run.status = "failed"
            run.finished_at = datetime.now(UTC)
            run.summary_json = self._build_summary(results)
            run.error_text = str(exc)[:1000] if str(exc) else "Eval run failed before completion."
            self.session.commit()
            self.session.refresh(run)
            return self.get_run(actor, run.id)

        return self.get_run(actor, run.id)

    def list_runs(self, actor: User) -> list[EvalRunRead]:
        self._ensure_admin(actor)
        self._reconcile_stale_runs()
        runs = self.eval_repository.list_runs()
        return [EvalRunRead.model_validate(item) for item in runs]

    def get_run(self, actor: User, run_id: UUID) -> EvalRunDetailRead:
        self._ensure_admin(actor)
        self._reconcile_stale_runs()
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

    def _evaluate_case_with_retry(self, run: EvalRun, case, top_k: int, *, max_attempts: int = 2) -> EvalResult:
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return self._evaluate_case(run, case, top_k)
            except (APIConnectionError, APITimeoutError) as error:
                last_error = error
                if attempt >= max_attempts:
                    raise
                time.sleep(1.5 * attempt)
        if last_error is not None:
            raise last_error
        raise RuntimeError("eval case retry loop exited without result")

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
                    "router_decision": prepared.router_result.decision.model_dump(mode="json"),
                    "tool_execution": prepared.tool_metadata.model_dump(mode="json"),
                    "structured_result": prepared.structured_result.model_dump(mode="json"),
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
            "case_annotations": metrics["case_annotations"],
            "expected_document_titles": case.expected_document_titles,
            "forbidden_document_titles": case.forbidden_document_titles,
            "expected_answer_keywords": case.expected_answer_keywords,
            "retrieved_document_titles": metrics["retrieved_document_titles"],
            "citation_document_titles": metrics["citation_document_titles"],
            "matched_expected_titles": metrics["matched_expected_titles"],
            "missing_expected_titles": metrics["missing_expected_titles"],
            "matched_citation_titles": metrics["matched_citation_titles"],
            "missing_citation_titles": metrics["missing_citation_titles"],
            "matched_answer_keywords": metrics["matched_answer_keywords"],
            "missing_answer_keywords": metrics["missing_answer_keywords"],
            "metric_breakdown": metrics["metric_breakdown"],
            "answer_text": prepared.answer_result.answer,
            "answer_excerpt": prepared.answer_result.answer[:400],
            "confidence": prepared.confidence,
            "insufficient_evidence": prepared.answer_result.insufficient_evidence,
            "evidence_conflict": prepared.answer_result.evidence_conflict,
            "permission_checks": metrics["permission_checks"],
            "retrieval_debug": prepared.retrieval_response.debug.model_dump(),
            "router_decision": prepared.router_result.decision.model_dump(mode="json"),
            "tool_execution": prepared.tool_metadata.model_dump(mode="json"),
            "structured_result": prepared.structured_result.model_dump(mode="json"),
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
        case_annotations = self._resolve_case_annotations(case)
        expected_titles = self._normalize_strings(case_annotations["expected_retrieval_titles"])
        expected_citation_titles = self._normalize_strings(case_annotations["expected_evidence_titles"])
        forbidden_titles = self._normalize_strings(case.forbidden_document_titles)
        expected_fact_specs = list(case_annotations.get("expected_key_fact_specs", []))
        forbidden_fact_specs = list(case_annotations.get("forbidden_key_fact_specs", []))
        refusal_expected = case_annotations["expected_outcome"] == "refuse"
        retrieved_titles = self._normalize_strings(item.document_title for item in prepared.retrieval_response.matched_chunks)
        citation_titles = self._normalize_strings(item.document_title for item in prepared.selected_chunks)
        ranked_retrieved_titles = self._ordered_normalized_strings(
            item.document_title for item in prepared.retrieval_response.matched_chunks
        )
        ranked_citation_titles = self._ordered_normalized_strings(item.document_title for item in prepared.selected_chunks)
        answer_text = prepared.answer_result.answer or ""
        answer_text_lower = answer_text.casefold()
        expected_fact_matches = self._match_fact_specs(expected_fact_specs, answer_text_lower)
        retrieval_fact_matches = self._match_fact_specs_in_chunks(
            expected_fact_specs,
            prepared.retrieval_response.matched_chunks,
        )
        evidence_fact_matches = self._match_fact_specs_in_chunks(expected_fact_specs, prepared.selected_chunks)
        forbidden_fact_matches = self._match_fact_specs(forbidden_fact_specs, answer_text_lower)
        if not expected_fact_specs:
            expected_fact_matches["coverage"] = 1.0
            evidence_fact_matches["coverage"] = 1.0
        supported_fact_breakdown = self._compute_fact_support_metrics(
            expected_fact_specs=expected_fact_specs,
            answer_fact_matches=expected_fact_matches,
            evidence_fact_matches=evidence_fact_matches,
        )
        matched_keywords = expected_fact_matches["matched_labels"]
        missing_keywords = expected_fact_matches["missing_labels"]
        matched_expected_titles = sorted(expected_titles.intersection(retrieved_titles))
        missing_expected_titles = sorted(expected_titles.difference(retrieved_titles))
        matched_citation_titles = sorted(expected_citation_titles.intersection(citation_titles))
        missing_citation_titles = sorted(expected_citation_titles.difference(citation_titles))
        forbidden_key_fact_hits = forbidden_fact_matches["matched_labels"]

        forbidden_in_retrieval = sorted(forbidden_titles.intersection(retrieved_titles))
        forbidden_in_citations = sorted(forbidden_titles.intersection(citation_titles))
        forbidden_in_answer = sorted(
            title for title in forbidden_titles if title and title in answer_text_lower
        )

        retrieval_breakdown = self._compute_retrieval_metrics(
            expected_titles=expected_titles,
            ranked_retrieved_titles=ranked_retrieved_titles,
            matched_expected_titles=matched_expected_titles,
            missing_expected_titles=missing_expected_titles,
            forbidden_titles=forbidden_titles,
            retrieved_fact_recall=retrieval_fact_matches["coverage"],
            matched_retrieval_fact_labels=retrieval_fact_matches["matched_labels"],
            missing_retrieval_fact_labels=retrieval_fact_matches["missing_labels"],
        )
        citation_breakdown = self._compute_citation_metrics(
            expected_titles=expected_citation_titles,
            ranked_citation_titles=ranked_citation_titles,
            matched_citation_titles=matched_citation_titles,
            missing_citation_titles=missing_citation_titles,
            forbidden_titles=forbidden_titles,
            evidence_fact_recall=evidence_fact_matches["coverage"],
            matched_evidence_fact_labels=evidence_fact_matches["matched_labels"],
            missing_evidence_fact_labels=evidence_fact_matches["missing_labels"],
        )
        permission_breakdown = self._compute_permission_isolation_metrics(
            forbidden_titles=forbidden_titles,
            forbidden_in_retrieval=forbidden_in_retrieval,
            forbidden_in_citations=forbidden_in_citations,
            forbidden_in_answer=forbidden_in_answer,
            forbidden_fact_leak_ratio=forbidden_fact_matches["coverage"],
        )
        permission_isolation_correct = permission_breakdown["passed"]
        faithfulness_breakdown = self._compute_answer_faithfulness(
            answer_fact_recall=expected_fact_matches["coverage"],
            evidence_fact_recall=evidence_fact_matches["coverage"],
            matched_fact_labels=expected_fact_matches["matched_labels"],
            missing_fact_labels=expected_fact_matches["missing_labels"],
            supported_fact_labels=supported_fact_breakdown["supported_labels"],
            unsupported_answer_fact_labels=supported_fact_breakdown["unsupported_answer_labels"],
            evidence_only_fact_labels=supported_fact_breakdown["evidence_only_labels"],
            prepared=prepared,
            refusal_expected=refusal_expected,
            support_precision=supported_fact_breakdown["support_precision"],
            supported_fact_recall=supported_fact_breakdown["support_recall"],
            forbidden_fact_leak_ratio=forbidden_fact_matches["coverage"],
        )
        retrieval_hit_rate = retrieval_breakdown["score"]
        citation_accuracy = citation_breakdown["score"]
        answer_faithfulness = faithfulness_breakdown["score"]
        overall_pass = self._compute_overall_pass(
            refusal_expected=refusal_expected,
            retrieval_hit_rate=retrieval_hit_rate,
            citation_accuracy=citation_accuracy,
            answer_faithfulness=answer_faithfulness,
            permission_isolation_correct=permission_isolation_correct,
            explicit_refusal=prepared.answer_result.insufficient_evidence,
            answer_reported_insufficient=prepared.answer_result.insufficient_evidence,
            evidence_conflict=prepared.answer_result.evidence_conflict,
        )
        overall_breakdown = self._compute_overall_score(
            retrieval_hit_rate=retrieval_hit_rate,
            citation_accuracy=citation_accuracy,
            answer_faithfulness=answer_faithfulness,
            permission_isolation_score=permission_breakdown["score"],
            permission_isolation_correct=permission_isolation_correct,
            refusal_expected=refusal_expected,
            overall_pass=overall_pass,
        )

        human_review_recommended = (
            not overall_pass
            or bool(forbidden_key_fact_hits)
            or bool(supported_fact_breakdown["unsupported_answer_labels"])
            or prepared.answer_result.evidence_conflict
            or (
                not refusal_expected
                and prepared.answer_result.insufficient_evidence
            )
        )
        if not permission_isolation_correct:
            human_review_reason = "forbidden documents or protected facts appeared in retrieval, citations, or answer text"
        elif forbidden_key_fact_hits:
            human_review_reason = "forbidden key facts appeared in answer text"
        elif supported_fact_breakdown["unsupported_answer_labels"]:
            human_review_reason = "answer mentioned expected facts that were not found in selected evidence"
        elif prepared.answer_result.evidence_conflict:
            human_review_reason = "top evidence conflicts across sources"
        elif not refusal_expected and prepared.answer_result.insufficient_evidence:
            human_review_reason = "answer expected, but model still reported insufficient evidence"
        elif not overall_pass:
            human_review_reason = "fact-to-evidence support fell below the current threshold"
        else:
            human_review_reason = "no immediate review required"

        return {
            "retrieval_hit_rate": retrieval_hit_rate,
            "citation_accuracy": citation_accuracy,
            "answer_faithfulness": answer_faithfulness,
            "permission_isolation_correct": permission_isolation_correct,
            "overall_pass": overall_pass,
            "case_annotations": case_annotations,
            "retrieved_document_titles": sorted(retrieved_titles),
            "citation_document_titles": sorted(citation_titles),
            "matched_expected_titles": matched_expected_titles,
            "missing_expected_titles": missing_expected_titles,
            "matched_citation_titles": matched_citation_titles,
            "missing_citation_titles": missing_citation_titles,
            "matched_answer_keywords": matched_keywords,
            "missing_answer_keywords": missing_keywords,
            "matched_answer_facts": matched_keywords,
            "missing_answer_facts": missing_keywords,
            "matched_retrieval_facts": retrieval_fact_matches["matched_labels"],
            "missing_retrieval_facts": retrieval_fact_matches["missing_labels"],
            "matched_evidence_facts": evidence_fact_matches["matched_labels"],
            "missing_evidence_facts": evidence_fact_matches["missing_labels"],
            "supported_answer_facts": supported_fact_breakdown["supported_labels"],
            "unsupported_answer_facts": supported_fact_breakdown["unsupported_answer_labels"],
            "evidence_only_facts": supported_fact_breakdown["evidence_only_labels"],
            "forbidden_key_fact_hits": forbidden_key_fact_hits,
            "permission_checks": {
                "forbidden_in_retrieval": forbidden_in_retrieval,
                "forbidden_in_citations": forbidden_in_citations,
                "forbidden_in_answer": forbidden_in_answer,
                "forbidden_answer_fact_hits": forbidden_key_fact_hits,
            },
            "metric_breakdown": {
                "retrieval": retrieval_breakdown,
                "citation": citation_breakdown,
                "faithfulness": faithfulness_breakdown,
                "permission_isolation": permission_breakdown,
                "overall": overall_breakdown,
            },
            "human_review_recommended": human_review_recommended,
            "human_review_reason": human_review_reason,
        }

    @staticmethod
    def _compute_retrieval_metrics(
        *,
        expected_titles: set[str],
        ranked_retrieved_titles: list[str],
        matched_expected_titles: list[str],
        missing_expected_titles: list[str],
        forbidden_titles: set[str],
        retrieved_fact_recall: float,
        matched_retrieval_fact_labels: list[str],
        missing_retrieval_fact_labels: list[str],
    ) -> dict:
        retrieved_title_set = set(ranked_retrieved_titles)
        forbidden_rate = (
            len(forbidden_titles.intersection(retrieved_title_set)) / len(forbidden_titles)
            if forbidden_titles
            else 0.0
        )
        if expected_titles:
            recall = len(matched_expected_titles) / len(expected_titles)
            precision = len(matched_expected_titles) / len(retrieved_title_set) if retrieved_title_set else 0.0
            average_precision = EvalService._compute_average_precision(ranked_retrieved_titles, expected_titles)
            ranking_score = EvalService._compute_binary_ndcg(ranked_retrieved_titles, expected_titles)
            mrr = EvalService._compute_mrr(ranked_retrieved_titles, expected_titles)
            score = EvalService._clamp_score(
                (recall + precision + ranking_score + retrieved_fact_recall) / 4
            )
        else:
            recall = 1.0 - forbidden_rate
            precision = 1.0 - forbidden_rate
            average_precision = 1.0 - forbidden_rate
            ranking_score = 1.0 - forbidden_rate
            mrr = 1.0 - forbidden_rate
            score = EvalService._clamp_score(1.0 - forbidden_rate)

        return {
            "score": score,
            "mode": "answer_expected" if expected_titles else "refusal_expected",
            "formula": (
                "mean(context_recall, context_precision, ndcg@k, retrieved_fact_recall)"
                if expected_titles
                else "1 - unauthorized_retrieval_rate"
            ),
            "recall": EvalService._round_score(recall),
            "precision": EvalService._round_score(precision),
            "average_precision": EvalService._round_score(average_precision),
            "ranking_score": EvalService._round_score(ranking_score),
            "mrr": EvalService._round_score(mrr),
            "retrieved_fact_recall": EvalService._round_score(retrieved_fact_recall),
            "unauthorized_retrieval_rate": EvalService._round_score(forbidden_rate),
            "retrieved_unique_titles": ranked_retrieved_titles,
            "matched_expected_titles": matched_expected_titles,
            "missing_expected_titles": missing_expected_titles,
            "matched_retrieval_facts": matched_retrieval_fact_labels,
            "missing_retrieval_facts": missing_retrieval_fact_labels,
        }

    @staticmethod
    def _compute_citation_metrics(
        *,
        expected_titles: set[str],
        ranked_citation_titles: list[str],
        matched_citation_titles: list[str],
        missing_citation_titles: list[str],
        forbidden_titles: set[str],
        evidence_fact_recall: float,
        matched_evidence_fact_labels: list[str],
        missing_evidence_fact_labels: list[str],
    ) -> dict:
        citation_title_set = set(ranked_citation_titles)
        forbidden_rate = (
            len(forbidden_titles.intersection(citation_title_set)) / len(forbidden_titles)
            if forbidden_titles
            else 0.0
        )

        if expected_titles:
            precision = len(matched_citation_titles) / len(citation_title_set) if citation_title_set else 0.0
            recall = len(matched_citation_titles) / len(expected_titles)
            f1 = EvalService._f1_score(precision, recall)
            score = EvalService._clamp_score((f1 + evidence_fact_recall) / 2)
        else:
            precision = 1.0 - forbidden_rate
            recall = 1.0 - forbidden_rate
            f1 = 1.0 - forbidden_rate
            score = EvalService._clamp_score(1.0 - forbidden_rate)

        return {
            "score": score,
            "mode": "answer_expected" if expected_titles else "refusal_expected",
            "formula": (
                "mean(citation_f1, evidence_fact_recall)"
                if expected_titles
                else "1 - unauthorized_citation_rate"
            ),
            "precision": EvalService._round_score(precision),
            "recall": EvalService._round_score(recall),
            "f1": EvalService._round_score(f1),
            "evidence_fact_recall": EvalService._round_score(evidence_fact_recall),
            "unauthorized_citation_rate": EvalService._round_score(forbidden_rate),
            "citation_unique_titles": ranked_citation_titles,
            "matched_expected_titles": matched_citation_titles,
            "missing_expected_titles": missing_citation_titles,
            "matched_evidence_facts": matched_evidence_fact_labels,
            "missing_evidence_facts": missing_evidence_fact_labels,
        }

    @staticmethod
    def _compute_answer_faithfulness(
        *,
        answer_fact_recall: float,
        evidence_fact_recall: float,
        matched_fact_labels: list[str],
        missing_fact_labels: list[str],
        supported_fact_labels: list[str],
        unsupported_answer_fact_labels: list[str],
        evidence_only_fact_labels: list[str],
        prepared: PreparedChatAnswer,
        refusal_expected: bool,
        support_precision: float,
        supported_fact_recall: float,
        forbidden_fact_leak_ratio: float,
    ) -> dict:
        explicit_refusal = prepared.answer_result.insufficient_evidence
        forbidden_fact_penalty = forbidden_fact_leak_ratio

        if refusal_expected:
            protected_fact_cleanliness = 1.0 - forbidden_fact_penalty
            score = EvalService._clamp_score(protected_fact_cleanliness)
            return {
                "score": score,
                "mode": "refusal_expected",
                "formula": "1 - protected_fact_leak_rate",
                "explicit_refusal": explicit_refusal,
                "answer_fact_recall": 1.0,
                "evidence_fact_recall": 1.0,
                "matched_expected_facts": matched_fact_labels,
                "missing_expected_facts": missing_fact_labels,
                "protected_fact_cleanliness": EvalService._round_score(protected_fact_cleanliness),
                "forbidden_fact_penalty": EvalService._round_score(forbidden_fact_penalty),
            }

        support_f1 = EvalService._f1_score(support_precision, supported_fact_recall)
        score = EvalService._clamp_score(
            max(0.0, support_f1 - forbidden_fact_penalty)
        )
        return {
            "score": score,
            "mode": "answer_expected",
            "formula": "support_f1(supported_fact_precision, supported_fact_recall) - forbidden_fact_leak_rate",
            "answer_fact_recall": EvalService._round_score(answer_fact_recall),
            "evidence_fact_recall": EvalService._round_score(evidence_fact_recall),
            "matched_expected_facts": matched_fact_labels,
            "missing_expected_facts": missing_fact_labels,
            "supported_facts": supported_fact_labels,
            "unsupported_answer_facts": unsupported_answer_fact_labels,
            "evidence_only_facts": evidence_only_fact_labels,
            "supported_fact_precision": EvalService._round_score(support_precision),
            "supported_fact_recall": EvalService._round_score(supported_fact_recall),
            "support_f1": EvalService._round_score(support_f1),
            "explicit_refusal": explicit_refusal,
            "evidence_conflict": prepared.answer_result.evidence_conflict,
            "insufficient_evidence": prepared.answer_result.insufficient_evidence,
            "forbidden_fact_penalty": EvalService._round_score(forbidden_fact_penalty),
        }

    @staticmethod
    def _compute_permission_isolation_metrics(
        *,
        forbidden_titles: set[str],
        forbidden_in_retrieval: list[str],
        forbidden_in_citations: list[str],
        forbidden_in_answer: list[str],
        forbidden_fact_leak_ratio: float,
    ) -> dict:
        if not forbidden_titles:
            retrieval_leak_ratio = 0.0
            citation_leak_ratio = 0.0
            answer_title_leak_ratio = 0.0
        else:
            denominator = len(forbidden_titles)
            retrieval_leak_ratio = len(forbidden_in_retrieval) / denominator
            citation_leak_ratio = len(forbidden_in_citations) / denominator
            answer_title_leak_ratio = len(forbidden_in_answer) / denominator

        answer_leak_ratio = max(answer_title_leak_ratio, forbidden_fact_leak_ratio)
        score = EvalService._clamp_score(1.0 - max(retrieval_leak_ratio, citation_leak_ratio, answer_leak_ratio))
        return {
            "score": score,
            "formula": "1 - max(unauthorized_retrieval_rate, unauthorized_citation_rate, unauthorized_answer_rate)",
            "retrieval_leak_ratio": EvalService._round_score(retrieval_leak_ratio),
            "citation_leak_ratio": EvalService._round_score(citation_leak_ratio),
            "answer_title_leak_ratio": EvalService._round_score(answer_title_leak_ratio),
            "forbidden_fact_leak_ratio": EvalService._round_score(forbidden_fact_leak_ratio),
            "answer_leak_ratio": EvalService._round_score(answer_leak_ratio),
            "passed": not (forbidden_in_retrieval or forbidden_in_citations or answer_leak_ratio > 0.0),
        }

    @staticmethod
    def _compute_overall_pass(
        *,
        refusal_expected: bool,
        retrieval_hit_rate: float,
        citation_accuracy: float,
        answer_faithfulness: float,
        permission_isolation_correct: bool,
        explicit_refusal: bool,
        answer_reported_insufficient: bool,
        evidence_conflict: bool,
    ) -> bool:
        if not permission_isolation_correct:
            return False
        if refusal_expected:
            return explicit_refusal and answer_faithfulness >= 1.0
        return (
            retrieval_hit_rate >= 0.5
            and citation_accuracy >= 0.5
            and answer_faithfulness >= 0.5
            and not answer_reported_insufficient
            and not evidence_conflict
        )

    @staticmethod
    def _compute_overall_score(
        *,
        retrieval_hit_rate: float,
        citation_accuracy: float,
        answer_faithfulness: float,
        permission_isolation_score: float,
        permission_isolation_correct: bool,
        refusal_expected: bool,
        overall_pass: bool,
    ) -> dict:
        score = EvalService._clamp_score(
            (retrieval_hit_rate + citation_accuracy + answer_faithfulness + permission_isolation_score) / 4
        )

        if not permission_isolation_correct:
            blocking_reason = "权限隔离失败"
        elif overall_pass:
            blocking_reason = "已满足当前阈值"
        elif refusal_expected:
            blocking_reason = "拒答型样例未达到拒答/引用阈值"
        else:
            blocking_reason = "检索、引用或答案得分未达到阈值"

        return {
            "score": score,
            "formula": "mean(retrieval, citation, faithfulness, permission)",
            "threshold_profile": "refusal_expected" if refusal_expected else "answer_expected",
            "permission_blocker": not permission_isolation_correct,
            "pass": overall_pass,
            "reason": blocking_reason,
        }

    @staticmethod
    def _build_summary(results: list[EvalResult]) -> dict:
        summary = EvalService._build_result_summary(results)
        answer_results = [item for item in results if EvalService._case_type_key(item) == "answer_expected"]
        refusal_results = [item for item in results if EvalService._case_type_key(item) == "refusal_expected"]
        summary["case_type_breakdown"] = {
            "answer_expected": EvalService._build_result_summary(
                answer_results,
                profile="answer_expected",
                label="回答型",
            ),
            "refusal_expected": EvalService._build_result_summary(
                refusal_results,
                profile="refusal_expected",
                label="拒答/权限型",
            ),
        }
        return summary

    @staticmethod
    def _build_result_summary(
        results: list[EvalResult],
        *,
        profile: str | None = None,
        label: str | None = None,
    ) -> dict:
        if not results:
            summary = {
                "total_cases": 0,
                "pass_count": 0,
                "pass_rate": 0.0,
                "retrieval_hit_rate_avg": 0.0,
                "citation_accuracy_avg": 0.0,
                "answer_faithfulness_avg": 0.0,
                "overall_score_avg": 0.0,
                "permission_isolation_pass_rate": 0.0,
                "permission_isolation_score_avg": 0.0,
            }
        else:
            total_cases = len(results)
            pass_count = sum(1 for item in results if item.overall_pass)
            permission_pass_count = sum(1 for item in results if item.permission_isolation_correct)
            overall_score_sum = sum(
                float((item.details_json or {}).get("metric_breakdown", {}).get("overall", {}).get("score", 0.0))
                for item in results
            )
            permission_score_sum = sum(
                float((item.details_json or {}).get("metric_breakdown", {}).get("permission_isolation", {}).get("score", 0.0))
                for item in results
            )
            summary = {
                "total_cases": total_cases,
                "pass_count": pass_count,
                "pass_rate": round(pass_count / total_cases, 4),
                "retrieval_hit_rate_avg": round(sum(item.retrieval_hit_rate for item in results) / total_cases, 4),
                "citation_accuracy_avg": round(sum(item.citation_accuracy for item in results) / total_cases, 4),
                "answer_faithfulness_avg": round(sum(item.answer_faithfulness for item in results) / total_cases, 4),
                "overall_score_avg": round(overall_score_sum / total_cases, 4),
                "permission_isolation_pass_rate": round(permission_pass_count / total_cases, 4),
                "permission_isolation_score_avg": round(permission_score_sum / total_cases, 4),
            }

        if profile:
            summary["profile"] = profile
        if label:
            summary["label"] = label
        return summary

    @staticmethod
    def _case_type_key(result: EvalResult) -> str:
        case_annotations = (result.details_json or {}).get("case_annotations", {})
        if isinstance(case_annotations, dict) and case_annotations.get("expected_outcome") == "refuse":
            return "refusal_expected"
        return "answer_expected"

    def _reconcile_stale_runs(self) -> None:
        now = datetime.now(UTC)
        stale_runs: list[EvalRun] = []
        for run in self.eval_repository.list_runs():
            if run.status != "running" or run.finished_at is not None:
                continue
            reference_time = run.started_at or run.created_at
            if reference_time is None:
                continue
            if now - reference_time < STALE_EVAL_RUN_THRESHOLD:
                continue
            run.status = "failed"
            run.finished_at = now
            if not run.error_text:
                run.error_text = "Eval run did not complete and was automatically marked as failed."
            stale_runs.append(run)
        if stale_runs:
            self.session.commit()

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
    def _ordered_normalized_strings(values) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for item in values:
            if item is None:
                continue
            cleaned = str(item).strip().lower()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            ordered.append(cleaned)
        return ordered

    @staticmethod
    def _normalize_fact_specs(values) -> list[dict]:
        normalized: list[dict] = []
        for item in values or []:
            if isinstance(item, str):
                label = item.strip()
                aliases = [label]
                weight = 1.0
            elif isinstance(item, dict):
                label = str(item.get("label") or "").strip()
                aliases = [
                    str(alias).strip()
                    for alias in item.get("aliases", [])
                    if str(alias).strip()
                ]
                weight = float(item.get("weight", 1.0) or 1.0)
                if not label and aliases:
                    label = aliases[0]
            else:
                continue

            if not label:
                continue

            normalized_aliases = EvalService._ordered_normalized_strings([label, *aliases])
            normalized.append(
                {
                    "label": label,
                    "aliases": normalized_aliases,
                    "weight": max(weight, 0.0) or 1.0,
                }
            )
        return normalized

    @staticmethod
    def _match_fact_specs(fact_specs: list[dict], answer_text_lower: str) -> dict:
        normalized_text = EvalService._normalize_match_text(answer_text_lower)
        total_weight = sum(float(item.get("weight", 1.0)) for item in fact_specs)
        if total_weight <= 0:
            return {
                "matched_labels": [],
                "missing_labels": [],
                "matched_weight": 0.0,
                "coverage": 0.0,
            }

        matched_labels: list[str] = []
        missing_labels: list[str] = []
        matched_weight = 0.0
        for item in fact_specs:
            label = str(item.get("label", "")).strip()
            aliases = [alias for alias in item.get("aliases", []) if isinstance(alias, str)]
            weight = float(item.get("weight", 1.0))
            if any(
                EvalService._alias_matches_text(alias, answer_text_lower, normalized_text)
                for alias in aliases
            ):
                matched_labels.append(label)
                matched_weight += weight
            else:
                missing_labels.append(label)

        return {
            "matched_labels": matched_labels,
            "missing_labels": missing_labels,
            "matched_weight": EvalService._round_score(matched_weight),
            "coverage": EvalService._clamp_score(matched_weight / total_weight),
        }

    @staticmethod
    def _normalize_match_text(text: str) -> str:
        return "".join(
            char
            for char in text.casefold()
            if not char.isspace() and unicodedata.category(char)[0] not in {"P", "S"}
        )

    @staticmethod
    def _alias_matches_text(alias: str, raw_text: str, normalized_text: str) -> bool:
        normalized_alias = EvalService._normalize_match_text(alias)
        if alias.casefold() in raw_text or normalized_alias in normalized_text:
            return True

        ordered_parts = EvalService._ordered_alias_parts(alias)
        if len(ordered_parts) < 2:
            return False

        cursor = 0
        for part in ordered_parts:
            found_at = normalized_text.find(part, cursor)
            if found_at < 0:
                return False
            cursor = found_at + len(part)
        return True

    @staticmethod
    def _ordered_alias_parts(alias: str) -> list[str]:
        parts = [
            EvalService._normalize_match_text(part)
            for part in re.split(r"(?:并且|并|且|同时|然后|以及|需要|必须|应当|应|要|需|，|。|；|;|：|:|、|\(|\)|（|）)", alias.casefold())
        ]
        return [part for part in parts if len(part) >= 4]

    @staticmethod
    def _match_fact_specs_in_chunks(fact_specs: list[dict], chunks: list) -> dict:
        chunk_text_parts: list[str] = []
        for chunk in chunks:
            for key in ("document_title", "section_title", "preview", "content"):
                if isinstance(chunk, dict):
                    value = chunk.get(key)
                else:
                    value = getattr(chunk, key, None)
                if isinstance(value, str) and value.strip():
                    chunk_text_parts.append(value)
        return EvalService._match_fact_specs(fact_specs, " ".join(chunk_text_parts).casefold())

    @staticmethod
    def _compute_fact_support_metrics(
        *,
        expected_fact_specs: list[dict],
        answer_fact_matches: dict,
        evidence_fact_matches: dict,
    ) -> dict:
        total_weight = sum(float(item.get("weight", 1.0)) for item in expected_fact_specs)
        if total_weight <= 0:
            return {
                "supported_labels": [],
                "unsupported_answer_labels": [],
                "evidence_only_labels": [],
                "supported_weight": 0.0,
                "support_precision": 1.0,
                "support_recall": 1.0,
                "support_f1": 1.0,
            }

        answer_labels = set(answer_fact_matches.get("matched_labels", []))
        evidence_labels = set(evidence_fact_matches.get("matched_labels", []))

        supported_labels: list[str] = []
        unsupported_answer_labels: list[str] = []
        evidence_only_labels: list[str] = []
        supported_weight = 0.0
        matched_answer_weight = 0.0

        for item in expected_fact_specs:
            label = str(item.get("label", "")).strip()
            weight = float(item.get("weight", 1.0))
            in_answer = label in answer_labels
            in_evidence = label in evidence_labels
            if in_answer:
                matched_answer_weight += weight
            if in_answer and in_evidence:
                supported_labels.append(label)
                supported_weight += weight
            elif in_answer:
                unsupported_answer_labels.append(label)
            elif in_evidence:
                evidence_only_labels.append(label)

        support_precision = supported_weight / matched_answer_weight if matched_answer_weight > 0 else 0.0
        support_recall = supported_weight / total_weight
        support_f1 = EvalService._f1_score(support_precision, support_recall)

        return {
            "supported_labels": supported_labels,
            "unsupported_answer_labels": unsupported_answer_labels,
            "evidence_only_labels": evidence_only_labels,
            "supported_weight": EvalService._round_score(supported_weight),
            "support_precision": EvalService._clamp_score(support_precision),
            "support_recall": EvalService._clamp_score(support_recall),
            "support_f1": EvalService._clamp_score(support_f1),
        }

    @staticmethod
    def _compute_binary_ndcg(ranked_titles: list[str], relevant_titles: set[str]) -> float:
        if not relevant_titles:
            return 1.0
        gains = [1.0 if title in relevant_titles else 0.0 for title in ranked_titles]
        dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
        ideal_hits = min(len(relevant_titles), len(ranked_titles))
        if ideal_hits == 0:
            return 0.0
        ideal_dcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
        if ideal_dcg == 0:
            return 0.0
        return dcg / ideal_dcg

    @staticmethod
    def _compute_average_precision(ranked_titles: list[str], relevant_titles: set[str]) -> float:
        if not relevant_titles:
            return 1.0

        precision_sum = 0.0
        true_positives = 0
        for index, title in enumerate(ranked_titles, start=1):
            if title not in relevant_titles:
                continue
            true_positives += 1
            precision_sum += true_positives / index

        return precision_sum / len(relevant_titles)

    @staticmethod
    def _compute_mrr(ranked_titles: list[str], relevant_titles: set[str]) -> float:
        if not relevant_titles:
            return 1.0

        for index, title in enumerate(ranked_titles, start=1):
            if title in relevant_titles:
                return 1.0 / index
        return 0.0

    @staticmethod
    def _f1_score(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)

    @staticmethod
    def _clamp_score(value: float) -> float:
        return round(max(0.0, min(value, 1.0)), 4)

    @staticmethod
    def _round_score(value: float) -> float:
        return round(value, 4)

    @staticmethod
    def _resolve_case_annotations(case) -> dict:
        demo_annotation = resolve_demo_eval_annotation(case.dataset_name, case.case_name) if case.is_demo_case else None
        expected_document_titles = list(case.expected_document_titles or [])
        expected_answer_keywords = list(case.expected_answer_keywords or [])

        expected_outcome = (
            demo_annotation.get("expected_outcome")
            if demo_annotation and demo_annotation.get("expected_outcome")
            else "refuse" if not expected_document_titles else "answer"
        )
        expected_retrieval_titles = (
            list(demo_annotation.get("expected_retrieval_titles", []))
            if demo_annotation and demo_annotation.get("expected_retrieval_titles")
            else expected_document_titles
        )
        expected_evidence_titles = (
            list(demo_annotation.get("expected_evidence_titles", []))
            if demo_annotation and demo_annotation.get("expected_evidence_titles")
            else expected_document_titles
        )
        raw_expected_key_facts = (
            list(demo_annotation.get("expected_key_facts", []))
            if demo_annotation and demo_annotation.get("expected_key_facts")
            else expected_answer_keywords
        )
        raw_forbidden_key_facts = (
            list(demo_annotation.get("forbidden_key_facts", []))
            if demo_annotation and demo_annotation.get("forbidden_key_facts")
            else []
        )
        expected_key_fact_specs = EvalService._normalize_fact_specs(raw_expected_key_facts)
        forbidden_key_fact_specs = EvalService._normalize_fact_specs(raw_forbidden_key_facts)
        scoring_notes = (
            str(demo_annotation.get("scoring_notes"))
            if demo_annotation and demo_annotation.get("scoring_notes")
            else None
        )
        return {
            "source": "demo_annotations" if demo_annotation else "legacy_case_fields",
            "expected_outcome": expected_outcome,
            "expected_retrieval_titles": expected_retrieval_titles,
            "expected_evidence_titles": expected_evidence_titles,
            "expected_key_facts": [item["label"] for item in expected_key_fact_specs],
            "forbidden_key_facts": [item["label"] for item in forbidden_key_fact_specs],
            "expected_key_fact_specs": expected_key_fact_specs,
            "forbidden_key_fact_specs": forbidden_key_fact_specs,
            "scoring_notes": scoring_notes,
        }

    @staticmethod
    def _ensure_admin(actor: User) -> None:
        if actor.role is None or actor.role.name != RoleName.ADMIN:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin permission required.")

