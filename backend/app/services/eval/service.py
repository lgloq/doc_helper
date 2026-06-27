from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
from uuid import UUID

from fastapi import HTTPException, status
from openai import APIConnectionError, APITimeoutError
from sqlalchemy.orm import Session

from app.models.eval import EvalResult, EvalRun
from app.models.enums import RoleName
from app.models.user import User
from app.repositories.eval_repository import EvalRepository
from app.repositories.user_repository import UserRepository
from app.schemas.eval import (
    EvalDashboardRead,
    EvalDatasetRead,
    EvalFailureModeRead,
    EvalRunDetailRead,
    EvalRunRead,
    EvalRunRequest,
    EvalResultRowRead,
    EvalTrendPointRead,
)
from app.services.diagnostics import PIPELINE_STAGE_LABELS, build_eval_pipeline_diagnosis
from app.services.chat.service import ChatService, PreparedChatAnswer
from app.services.eval.bootstrap import seed_demo_eval_cases
from app.services.eval.demo_cases import resolve_demo_eval_annotation
from app.services.observability.service import ObservabilityService

STALE_QUEUED_EVAL_RUN_THRESHOLD = timedelta(minutes=5)
STALE_RUNNING_EVAL_RUN_THRESHOLD = timedelta(minutes=15)


class EvalService:
    def __init__(self, session: Session):
        self.session = session
        self.eval_repository = EvalRepository(session)
        self.user_repository = UserRepository(session)
        self.chat_service = ChatService(session)
        self.observability_service = ObservabilityService(session)

    def run_eval(self, actor: User, payload: EvalRunRequest) -> EvalRunDetailRead:
        self._ensure_admin(actor)
        client_request_id = self._normalize_client_request_id(payload.client_request_id)
        if client_request_id:
            existing_run = self._find_client_request_run(client_request_id)
            if existing_run is not None:
                return self.get_run(actor, existing_run.id)

        if payload.seed_demo_cases:
            seed_demo_eval_cases()

        cases = self._select_cases(payload)
        if not cases:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No eval cases found for the requested dataset.")

        request_metadata = self._build_client_request_metadata(
            payload,
            client_request_id,
            status="running",
            selected_case_ids=[case.id for case in cases],
        )
        run = self._create_eval_run(
            payload,
            cases,
            status="running",
            request_metadata=request_metadata,
            started_at=datetime.now(UTC),
        )
        return self._execute_run_cases(run, cases, payload.top_k, request_metadata, actor=actor)

    def create_queued_run(self, actor: User, payload: EvalRunRequest) -> tuple[EvalRunDetailRead, bool]:
        self._ensure_admin(actor)
        client_request_id = self._normalize_client_request_id(payload.client_request_id)
        if client_request_id:
            existing_run = self._find_client_request_run(client_request_id)
            if existing_run is not None:
                return self.get_run(actor, existing_run.id), False

        if payload.seed_demo_cases:
            seed_demo_eval_cases()

        cases = self._select_cases(payload)
        if not cases:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No eval cases found for the requested dataset.")

        request_metadata = self._build_client_request_metadata(
            payload,
            client_request_id,
            status="queued",
            selected_case_ids=[case.id for case in cases],
            force=True,
        )
        run = self._create_eval_run(payload, cases, status="queued", request_metadata=request_metadata)
        return self._build_run_detail(run), True

    def attach_job_to_run(self, run_id: UUID, job_id: str) -> EvalRunDetailRead:
        run = self.eval_repository.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Eval run not found.")
        summary = run.summary_json if isinstance(run.summary_json, dict) else {}
        run.summary_json = {**summary, "job_id": job_id}
        self.session.commit()
        self.session.refresh(run)
        return self._build_run_detail(run)

    def mark_run_enqueue_failed(self, run_id: UUID, error_text: str) -> EvalRunDetailRead:
        run = self.eval_repository.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Eval run not found.")
        run.status = "failed"
        run.finished_at = datetime.now(UTC)
        run.error_text = error_text[:1000]
        summary = run.summary_json if isinstance(run.summary_json, dict) else {}
        run.summary_json = {
            **summary,
            "client_request_status": "failed",
        }
        self.session.commit()
        self.session.refresh(run)
        return self._build_run_detail(run)

    def discard_unstarted_queued_run(self, run_id: UUID) -> None:
        run = self.eval_repository.get_run(run_id)
        if run is None:
            return
        if run.status != "queued" or run.started_at is not None or self.eval_repository.list_results_for_run(run.id):
            return
        self.session.delete(run)
        self.session.commit()

    def execute_queued_run(self, run_id: UUID) -> EvalRunDetailRead:
        run = self.eval_repository.get_run(run_id)
        if run is None:
            raise RuntimeError(f"Eval run {run_id} was not found.")
        if run.status in {"completed", "failed"}:
            return self._build_run_detail(run)
        if run.status == "running" and run.started_at is not None:
            return self._build_run_detail(run)

        summary = run.summary_json if isinstance(run.summary_json, dict) else {}
        case_ids = self._selected_case_ids_from_summary(summary)
        cases = self.eval_repository.get_cases_by_ids(case_ids) if case_ids else self.eval_repository.list_cases(run.dataset_name)
        if not cases:
            run.status = "failed"
            run.finished_at = datetime.now(UTC)
            run.error_text = "No eval cases found for the queued run."
            run.summary_json = {**summary, "client_request_status": "failed"}
            self.session.commit()
            self.session.refresh(run)
            return self._build_run_detail(run)

        top_k = self._top_k_from_summary(summary)
        run.status = "running"
        run.started_at = run.started_at or datetime.now(UTC)
        run.total_cases = len(cases)
        run.summary_json = {**summary, "client_request_status": "running"}
        self.session.commit()
        self.session.refresh(run)
        return self._execute_run_cases(run, cases, top_k, run.summary_json or {})

    def _create_eval_run(
        self,
        payload: EvalRunRequest,
        cases,
        *,
        status: str,
        request_metadata: dict,
        started_at: datetime | None = None,
    ) -> EvalRun:
        run = EvalRun(
            dataset_name=payload.dataset_name,
            status=status,
            total_cases=len(cases),
            started_at=started_at,
            summary_json=request_metadata or None,
        )
        self.eval_repository.add_run(run)
        self.session.commit()
        self.session.refresh(run)
        return run

    def _execute_run_cases(
        self,
        run: EvalRun,
        cases,
        top_k: int,
        request_metadata: dict,
        *,
        actor: User | None = None,
    ) -> EvalRunDetailRead:
        results: list[EvalResult] = []
        try:
            for case in cases:
                result = self._evaluate_case_with_retry(run, case, top_k)
                results.append(result)
                self.eval_repository.add_results([result])
                run.summary_json = self._with_client_request_metadata(
                    self._build_running_progress_summary(len(results), len(cases)),
                    request_metadata,
                    status="running",
                )
                self.session.commit()

            run.status = "completed"
            run.finished_at = datetime.now(UTC)
            run.summary_json = self._with_client_request_metadata(
                self._build_summary(results),
                request_metadata,
                status="completed",
            )
            run.error_text = None
            self.session.commit()
            self.session.refresh(run)
        except Exception as exc:
            run.status = "failed"
            run.finished_at = datetime.now(UTC)
            run.summary_json = self._with_client_request_metadata(
                self._build_summary(results),
                request_metadata,
                status="failed",
            )
            run.error_text = str(exc)[:1000] if str(exc) else "Eval run failed before completion."
            self.session.commit()
            self.session.refresh(run)

        if actor is not None:
            return self.get_run(actor, run.id)
        return self._build_run_detail(run)

    @staticmethod
    def _normalize_client_request_id(value: str | None) -> str | None:
        normalized = (value or "").strip()
        return normalized or None

    def _find_client_request_run(self, client_request_id: str) -> EvalRun | None:
        for run in self.eval_repository.list_runs():
            summary = run.summary_json if isinstance(run.summary_json, dict) else {}
            if summary.get("client_request_id") == client_request_id:
                return run
        return None

    @staticmethod
    def _build_client_request_metadata(
        payload: EvalRunRequest,
        client_request_id: str | None,
        *,
        status: str,
        selected_case_ids: list[UUID] | None = None,
        force: bool = False,
    ) -> dict:
        if not client_request_id and not force:
            return {}
        return {
            "client_request_id": client_request_id,
            "client_request_status": status,
            "requested_dataset_name": payload.dataset_name,
            "requested_top_k": payload.top_k,
            "requested_case_ids": [str(item) for item in payload.case_ids],
            "requested_seed_demo_cases": payload.seed_demo_cases,
            "selected_case_ids": [str(item) for item in selected_case_ids or []],
        }

    @staticmethod
    def _with_client_request_metadata(summary: dict, metadata: dict, *, status: str) -> dict:
        if not metadata:
            return summary
        return {
            **summary,
            **metadata,
            "client_request_status": status,
        }

    @staticmethod
    def _selected_case_ids_from_summary(summary: dict) -> list[UUID]:
        values = summary.get("selected_case_ids") or summary.get("requested_case_ids") or []
        if not isinstance(values, list):
            return []
        case_ids: list[UUID] = []
        for value in values:
            try:
                case_ids.append(UUID(str(value)))
            except (TypeError, ValueError):
                continue
        return case_ids

    @staticmethod
    def _top_k_from_summary(summary: dict) -> int:
        value = summary.get("requested_top_k")
        if isinstance(value, int):
            return min(10, max(1, value))
        return 5

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
        return self._build_run_detail(run)

    def get_reconciled_run_detail(self, run_id: UUID) -> EvalRunDetailRead | None:
        self._reconcile_stale_runs()
        run = self.eval_repository.get_run(run_id)
        if run is None:
            return None
        return self._build_run_detail(run)

    def _build_run_detail(self, run: EvalRun) -> EvalRunDetailRead:
        results = self.eval_repository.list_results_for_run(run.id)
        return EvalRunDetailRead.model_validate(
            {
                **EvalRunRead.model_validate(run).model_dump(),
                "results": [EvalResultRowRead.model_validate(item) for item in results],
            }
        )

    def list_datasets(self, actor: User) -> list[EvalDatasetRead]:
        self._ensure_admin(actor)
        self._reconcile_stale_runs()
        datasets: list[EvalDatasetRead] = []
        for dataset_name in self.eval_repository.list_dataset_names():
            cases = self.eval_repository.list_cases(dataset_name)
            runs = self.eval_repository.list_runs(dataset_name=dataset_name)
            completed_count = sum(1 for run in runs if run.status == "completed")
            failed_count = sum(1 for run in runs if run.status == "failed")
            datasets.append(
                EvalDatasetRead(
                    dataset_name=dataset_name,
                    display_name=self._display_dataset_name(dataset_name),
                    case_count=len(cases),
                    demo_case_count=sum(1 for case in cases if case.is_demo_case),
                    completed_run_count=completed_count,
                    failed_run_count=failed_count,
                    latest_run=EvalRunRead.model_validate(runs[0]) if runs else None,
                )
            )
        return datasets

    def get_dashboard(self, actor: User, dataset_name: str | None = None, *, limit: int = 8) -> EvalDashboardRead:
        self._ensure_admin(actor)
        self._reconcile_stale_runs()
        selected_dataset = dataset_name or self._default_dashboard_dataset_name()
        if not selected_dataset:
            return EvalDashboardRead()

        completed_runs = self.eval_repository.list_runs(
            dataset_name=selected_dataset,
            statuses=["completed"],
            limit=max(1, limit),
        )
        trend = [self._trend_point_from_run(run) for run in reversed(completed_runs)]
        latest_completed_run = completed_runs[0] if completed_runs else None
        failure_modes: list[EvalFailureModeRead] = []
        if latest_completed_run is not None:
            failure_modes = self._failure_modes_for_results(self.eval_repository.list_results_for_run(latest_completed_run.id))

        return EvalDashboardRead(
            dataset_name=selected_dataset,
            display_name=self._display_dataset_name(selected_dataset),
            trend=trend,
            failure_modes=failure_modes,
            latest_completed_run=EvalRunRead.model_validate(latest_completed_run) if latest_completed_run else None,
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
        pipeline_diagnosis = build_eval_pipeline_diagnosis(
            session=self.session,
            actor=actor,
            expected_titles=list(metrics["case_annotations"]["expected_retrieval_titles"]),
            expected_outcome=str(metrics["case_annotations"]["expected_outcome"]),
            overall_pass=metrics["overall_pass"],
            retrieval_breakdown=metrics["metric_breakdown"]["retrieval"],
            citation_breakdown=metrics["metric_breakdown"]["citation"],
            faithfulness_breakdown=metrics["metric_breakdown"]["faithfulness"],
            permission_breakdown=metrics["metric_breakdown"]["permission_isolation"],
            permission_checks=metrics["permission_checks"],
            retrieval_debug=prepared.retrieval_response.debug,
            matched_expected_titles=metrics["matched_expected_titles"],
            missing_expected_titles=metrics["missing_expected_titles"],
            matched_citation_titles=metrics["matched_citation_titles"],
            missing_citation_titles=metrics["missing_citation_titles"],
            unsupported_answer_facts=metrics["unsupported_answer_facts"],
            unsupported_answer_claims=metrics["unsupported_answer_claims"],
            insufficient_evidence=prepared.answer_result.insufficient_evidence,
        )
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
                    "pipeline_diagnosis": pipeline_diagnosis,
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
            "pipeline_diagnosis": pipeline_diagnosis,
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
        unsupported_claims = list(faithfulness_breakdown.get("unsupported_claims", []))
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
            or bool(unsupported_claims)
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
        elif unsupported_claims:
            human_review_reason = "answer contained claims that were not supported by selected evidence"
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
            "unsupported_answer_claims": unsupported_claims,
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
        claim_support = EvalService._compute_claim_support_metrics(
            answer_text=getattr(prepared.answer_result, "answer", "") or "",
            selected_chunks=getattr(prepared, "selected_chunks", []) or [],
        )
        claim_support_score = claim_support["score"]
        score = EvalService._clamp_score(
            max(0.0, claim_support_score - forbidden_fact_penalty)
        )
        return {
            "score": score,
            "mode": "answer_expected",
            "formula": "mean(max_claim_support_by_selected_evidence) - forbidden_fact_leak_rate",
            "claim_support_score": EvalService._round_score(claim_support_score),
            "claim_count": claim_support["claim_count"],
            "answer_claims": claim_support["claims"],
            "unsupported_claims": claim_support["unsupported_claims"],
            "claim_extraction_method": claim_support["extraction_method"],
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
    def _compute_claim_support_metrics(answer_text: str, selected_chunks: list) -> dict:
        claims = EvalService._extract_answer_claims(answer_text)
        evidence_payloads = EvalService._evidence_payloads(selected_chunks)
        if not claims:
            return {
                "score": 0.0,
                "claim_count": 0,
                "claims": [],
                "unsupported_claims": [],
                "extraction_method": "deterministic_sentence_clause_split",
            }

        scored_claims: list[dict] = []
        for claim in claims:
            best = {
                "score": 0.0,
                "chunk_id": None,
                "document_title": None,
                "reasons": ["no selected citation evidence"],
            }
            for payload in evidence_payloads:
                candidate = EvalService._score_claim_against_evidence(claim["text"], payload["text"])
                if candidate["score"] > best["score"]:
                    best = {
                        "score": candidate["score"],
                        "chunk_id": payload["chunk_id"],
                        "document_title": payload["document_title"],
                        "reasons": candidate["reasons"],
                    }
            scored_claims.append(
                {
                    **claim,
                    "support_score": EvalService._round_score(best["score"]),
                    "support_evidence_chunk_id": best["chunk_id"],
                    "support_document_title": best["document_title"],
                    "support_reasons": best["reasons"],
                }
            )

        score = sum(item["support_score"] for item in scored_claims) / len(scored_claims)
        return {
            "score": EvalService._clamp_score(score),
            "claim_count": len(scored_claims),
            "claims": scored_claims,
            "unsupported_claims": [
                item["text"]
                for item in scored_claims
                if item["support_score"] < 0.5
            ],
            "extraction_method": "deterministic_sentence_clause_split",
        }

    @staticmethod
    def _extract_answer_claims(answer_text: str) -> list[dict]:
        cleaned = EvalService._strip_citation_markers(answer_text or "")
        cleaned = re.sub(r"\s+", " ", cleaned.replace("\r", "\n")).strip()
        if not cleaned:
            return []

        cleaned = re.sub(
            r"(?:^|[，,；;。]\s*)(第一|第二|第三|第四|第五|第六|第七|第八|其一|其二|其三|一是|二是|三是)\s*[，,:：]",
            "；",
            cleaned,
        )
        parts = [
            part.strip()
            for part in re.split(r"[。！？!?；;\n]+", cleaned)
            if part.strip()
        ]

        claims: list[dict] = []
        seen: set[str] = set()
        for raw_part in parts:
            claim_text = EvalService._clean_claim_candidate(raw_part)
            if not EvalService._looks_like_factual_claim(claim_text):
                continue
            normalized = EvalService._normalize_match_text(claim_text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            claims.append(
                {
                    "text": claim_text,
                    "normalized": normalized,
                    "length": len(normalized),
                }
            )
        return claims

    @staticmethod
    def _strip_citation_markers(text: str) -> str:
        cleaned = re.sub(r"\[[0-9,\s]+\]", "", text)
        cleaned = re.sub(r"【[^】]{1,80}】", "", cleaned)
        cleaned = re.sub(r"\(来源[:：][^)）]{1,120}[)）]", "", cleaned)
        return cleaned

    @staticmethod
    def _clean_claim_candidate(text: str) -> str:
        cleaned = text.strip(" \t\r\n，,。；;：:")
        cleaned = re.sub(r"^根据当前可访问文档中的证据[，,]?", "", cleaned)
        cleaned = re.sub(r"^当前可访问文档中的证据[，,]?", "", cleaned)
        cleaned = re.sub(r"^《[^》]{1,120}》(?:里|中)?", "", cleaned)
        cleaned = re.sub(r"^[^：:]{0,160}（[^）]{1,120}）提到[:：]", "", cleaned)
        cleaned = re.sub(r"^[^：:]{0,160}(?:提到|显示|说明|指出)[:：]", "", cleaned)
        cleaned = re.sub(
            r"^[^：:]{0,160}(?:主要有[一二三四五六七八九十两0-9]+点|主要说明|直接相关的要求是|要求是|要求为)[:：]",
            "",
            cleaned,
        )
        cleaned = re.sub(r"^(?:第一|第二|第三|第四|第五|第六|第七|第八|其一|其二|其三|一是|二是|三是)[，,:：]?", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ，,。；;：:")
        return cleaned

    @staticmethod
    def _looks_like_factual_claim(text: str) -> bool:
        if not text:
            return False
        normalized = EvalService._normalize_match_text(text)
        if len(normalized) < 6 and not EvalService._extract_numeric_constraints(text):
            return False
        low_information_markers = (
            "未找到足够相关",
            "证据不足",
            "暂时无法",
            "请换个问法",
            "建议结合引用片段",
            "建议你继续",
            "建议你把问题",
            "我已经检索到",
            "进一步确认",
            "当前可访问范围内未找到",
            "文档可能不存在",
            "没有访问权限",
        )
        if any(marker in text for marker in low_information_markers):
            return False
        factual_markers = (
            "为",
            "是",
            "包括",
            "包含",
            "需要",
            "必须",
            "应",
            "不得",
            "禁止",
            "负责",
            "审批",
            "处理",
            "时限",
            "要求",
            "同步",
            "补齐",
            "关闭",
            "回收",
            "导出",
            "升级",
            "建立",
            "明确",
            "可以",
            "先",
        )
        return len(normalized) >= 14 or any(marker in text for marker in factual_markers)

    @staticmethod
    def _evidence_payloads(chunks: list) -> list[dict]:
        payloads: list[dict] = []
        for chunk in chunks:
            parts: list[str] = []
            for key in (
                "document_title",
                "section_title",
                "heading_path",
                "clause_full_name",
                "article_number",
                "preview",
                "content",
            ):
                value = chunk.get(key) if isinstance(chunk, dict) else getattr(chunk, key, None)
                if isinstance(value, str) and value.strip():
                    parts.append(value)
            chunk_id = chunk.get("chunk_id") if isinstance(chunk, dict) else getattr(chunk, "chunk_id", None)
            document_title = chunk.get("document_title") if isinstance(chunk, dict) else getattr(chunk, "document_title", None)
            payloads.append(
                {
                    "chunk_id": str(chunk_id) if chunk_id is not None else None,
                    "document_title": document_title,
                    "text": " ".join(parts),
                }
            )
        return payloads

    @staticmethod
    def _score_claim_against_evidence(claim_text: str, evidence_text: str) -> dict:
        claim_norm = EvalService._normalize_match_text(claim_text)
        evidence_norm = EvalService._normalize_match_text(evidence_text)
        if not claim_norm or not evidence_norm:
            return {"score": 0.0, "reasons": ["empty claim or evidence"]}
        if claim_norm in evidence_norm:
            return {"score": 1.0, "reasons": ["normalized claim is contained in citation evidence"]}

        structured_score, structured_reasons = EvalService._score_structured_claim_support(claim_text, evidence_text)
        claim_constraints = EvalService._extract_numeric_constraints(claim_text)
        missing_constraints = [
            item
            for item in claim_constraints
            if EvalService._normalize_match_text(item) not in evidence_norm
        ]

        token_recall = EvalService._token_recall(
            EvalService._extract_support_tokens(claim_text),
            EvalService._extract_support_tokens(evidence_text),
        )
        ordered_ratio = EvalService._ordered_part_support_ratio(claim_text, evidence_norm)
        longest_ratio = EvalService._longest_common_match_ratio(claim_norm, evidence_norm)
        score = max(
            structured_score,
            (0.55 * token_recall) + (0.25 * ordered_ratio) + (0.20 * longest_ratio),
        )

        reasons = [
            f"token_recall={EvalService._round_score(token_recall)}",
            f"ordered_part_ratio={EvalService._round_score(ordered_ratio)}",
            f"longest_common_ratio={EvalService._round_score(longest_ratio)}",
        ]
        if structured_reasons:
            reasons.extend(structured_reasons)
        if missing_constraints:
            cap = 0.35 if len(missing_constraints) == len(claim_constraints) else 0.65
            score = min(score, cap)
            reasons.append(f"missing_numeric_or_date_constraints={missing_constraints}")

        relation_tokens = EvalService._extract_relation_tokens(claim_text)
        if relation_tokens:
            evidence_relation_tokens = EvalService._extract_relation_tokens(evidence_text)
            if not set(relation_tokens).intersection(evidence_relation_tokens):
                score = min(score, 0.65)
                reasons.append("claim relation/action tokens not found in evidence")

        if structured_score <= 0 and token_recall < 0.28 and ordered_ratio < 0.34:
            score = min(score, 0.35)
            reasons.append("only weak topical overlap")

        return {
            "score": EvalService._clamp_score(score),
            "reasons": reasons,
        }

    @staticmethod
    def _score_structured_claim_support(claim_text: str, evidence_text: str) -> tuple[float, list[str]]:
        claim_pairs = EvalService._extract_key_value_pairs(claim_text)
        if not claim_pairs:
            return 0.0, []

        evidence_pairs = EvalService._extract_key_value_pairs(evidence_text)
        if not evidence_pairs:
            return 0.0, ["structured_claim_without_key_value_evidence"]

        pair_scores: list[float] = []
        for claim_key, claim_value in claim_pairs:
            best_pair_score = 0.0
            claim_key_tokens = EvalService._extract_support_tokens(claim_key)
            claim_value_tokens = EvalService._extract_support_tokens(claim_value)
            for evidence_key, evidence_value in evidence_pairs:
                key_score = EvalService._token_recall(claim_key_tokens, EvalService._extract_support_tokens(evidence_key))
                value_norm = EvalService._normalize_match_text(claim_value)
                evidence_value_norm = EvalService._normalize_match_text(evidence_value)
                if value_norm and value_norm in evidence_value_norm:
                    value_score = 1.0
                else:
                    value_score = EvalService._token_recall(
                        claim_value_tokens,
                        EvalService._extract_support_tokens(evidence_value),
                    )
                best_pair_score = max(best_pair_score, (0.35 * key_score) + (0.65 * value_score))
            pair_scores.append(best_pair_score)

        score = sum(pair_scores) / len(pair_scores)
        return EvalService._clamp_score(score), [f"structured_pair_support={EvalService._round_score(score)}"]

    @staticmethod
    def _extract_key_value_pairs(text: str) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        normalized = text.replace("：", "=").replace(":", "=")
        matches = list(re.finditer(r"(?P<key>[^=;；。,\n]{2,32}?)=", normalized))
        for index, match in enumerate(matches):
            key = match.group("key").strip(" ，,；;。")
            value_start = match.end()
            value_end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
            value = normalized[value_start:value_end].strip(" ，,；;。")
            if key and value and len(EvalService._normalize_match_text(value)) >= 2:
                pairs.append((key, value))

        natural_patterns = (
            r"(?P<key>[\u4e00-\u9fffA-Za-z0-9]{2,18}?)(?:为|是|包括|包含)(?P<value>[^；;。,.，]{2,60})",
            r"(?P<key>账号)(?:应在|需要在)(?P<value>[^；;。,.，]{2,60})",
            r"(?P<key>导出文件|文件)(?:禁止通过|不得通过)(?P<value>[^；;。,.，]{2,60})",
            r"(?P<key>紧急场景|临时场景)(?:下)?可以先(?P<value>[^；;。,.，]{2,60})",
            r"(?P<key>先同步|同步)(?P<value>[^；;。,.，]{2,60})",
        )
        for pattern in natural_patterns:
            for match in re.finditer(pattern, text):
                key = match.group("key").strip()
                value = match.group("value").strip(" ，,；;。")
                if key and value:
                    pairs.append((key, value))

        unique: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for key, value in pairs:
            identity = (EvalService._normalize_match_text(key), EvalService._normalize_match_text(value))
            if identity in seen:
                continue
            seen.add(identity)
            unique.append((key, value))
        return unique

    @staticmethod
    def _extract_numeric_constraints(text: str) -> list[str]:
        patterns = (
            r"\d{4}[-/年]\d{1,2}(?:[-/月]\d{1,2}日?)?",
            r"\d+(?:\.\d+)?\s*(?:%|％|个工作日|工作日|分钟内|小时内|日内|天内|分钟|小时|个月|月|天|日|年|万元|亿元|元|人|次|条|份|级|类)",
            r"[一二两三四五六七八九十百千]+(?:个)?(?:工作日|分钟内|小时内|日内|天内|分钟|小时|个月|月|天|日|年)",
        )
        found: list[str] = []
        for pattern in patterns:
            found.extend(match.group(0).strip() for match in re.finditer(pattern, text))
        return list(dict.fromkeys(item for item in found if item))

    @staticmethod
    def _extract_support_tokens(text: str) -> list[str]:
        normalized = text.casefold()
        tokens: list[str] = []
        tokens.extend(EvalService._normalize_match_text(item) for item in EvalService._extract_numeric_constraints(text))
        tokens.extend(re.findall(r"[a-z0-9]{2,}", normalized))
        stop_terms = {
            "根据",
            "当前",
            "可访问",
            "文档",
            "证据",
            "主要",
            "这个",
            "场景",
            "相关",
            "要求",
            "说明",
            "包括",
            "包含",
            "以及",
            "同时",
            "其中",
            "应当",
            "需要",
            "必须",
            "可以",
            "进行",
            "通过",
            "如果",
            "对于",
            "里面",
            "条款",
        }
        for segment in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
            parts = [
                part
                for part in re.split(r"的|了|在|和|与|及|或|并|对|中|里|为|是|由|将|把|向|于", segment)
                if len(part) >= 2 and part not in stop_terms
            ]
            for part in parts:
                tokens.append(part)
                if len(part) > 4:
                    for size in (2, 3, 4):
                        tokens.extend(part[index : index + size] for index in range(0, len(part) - size + 1))
        return [token for token in dict.fromkeys(tokens) if token and token not in stop_terms]

    @staticmethod
    def _extract_relation_tokens(text: str) -> list[str]:
        relation_markers = (
            "需要",
            "必须",
            "应",
            "不得",
            "禁止",
            "负责",
            "审批",
            "提交",
            "保留",
            "关闭",
            "回收",
            "升级",
            "建立",
            "明确",
            "同步",
            "补齐",
            "脱敏",
            "导出",
            "执行",
            "处理",
            "验收",
            "复核",
            "通知",
        )
        return [marker for marker in relation_markers if marker in text]

    @staticmethod
    def _token_recall(claim_tokens: list[str], evidence_tokens: list[str]) -> float:
        if not claim_tokens:
            return 0.0
        evidence_set = set(evidence_tokens)
        if not evidence_set:
            return 0.0
        matched = sum(1 for token in claim_tokens if token in evidence_set)
        return matched / len(claim_tokens)

    @staticmethod
    def _ordered_part_support_ratio(claim_text: str, evidence_norm: str) -> float:
        parts = EvalService._ordered_alias_parts(claim_text)
        if not parts:
            return 0.0
        matched = 0
        cursor = 0
        for part in parts:
            found_at = evidence_norm.find(part, cursor)
            if found_at < 0:
                continue
            matched += 1
            cursor = found_at + len(part)
        return matched / len(parts)

    @staticmethod
    def _longest_common_match_ratio(claim_norm: str, evidence_norm: str) -> float:
        if not claim_norm or not evidence_norm:
            return 0.0
        evidence_window = evidence_norm[:5000]
        match = SequenceMatcher(None, claim_norm, evidence_window, autojunk=False).find_longest_match(
            0,
            len(claim_norm),
            0,
            len(evidence_window),
        )
        return match.size / len(claim_norm) if claim_norm else 0.0

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

    def _default_dashboard_dataset_name(self) -> str | None:
        completed_runs = self.eval_repository.list_runs(statuses=["completed"], limit=1)
        if completed_runs:
            return completed_runs[0].dataset_name
        dataset_names = self.eval_repository.list_dataset_names()
        return dataset_names[0] if dataset_names else None

    @staticmethod
    def _display_dataset_name(dataset_name: str) -> str:
        labels = {
            "demo_access_matrix_eval": "权限隔离演示评测",
            "demo_permission_eval": "权限回归演示评测",
            "zh_enterprise_v1_seed": "中文企业文档评测集",
            "format_coverage_zh_parser": "格式覆盖回归集",
        }
        if dataset_name in labels:
            return labels[dataset_name]
        return dataset_name.replace("_", " ")

    @staticmethod
    def _trend_point_from_run(run: EvalRun) -> EvalTrendPointRead:
        summary = run.summary_json if isinstance(run.summary_json, dict) else {}
        return EvalTrendPointRead(
            run_id=run.id,
            dataset_name=run.dataset_name,
            created_at=run.created_at,
            status=run.status,
            total_cases=run.total_cases,
            pass_count=EvalService._summary_int(summary, "pass_count"),
            pass_rate=EvalService._summary_float(summary, "pass_rate"),
            retrieval_hit_rate_avg=EvalService._summary_float(summary, "retrieval_hit_rate_avg"),
            citation_accuracy_avg=EvalService._summary_float(summary, "citation_accuracy_avg"),
            answer_faithfulness_avg=EvalService._summary_float(summary, "answer_faithfulness_avg"),
            permission_isolation_pass_rate=EvalService._summary_float(summary, "permission_isolation_pass_rate"),
            overall_score_avg=EvalService._summary_float(summary, "overall_score_avg"),
        )

    @staticmethod
    def _failure_modes_for_results(results: list[EvalResult]) -> list[EvalFailureModeRead]:
        labels = {
            "permission_failure": "权限隔离问题",
            "retrieval_failure": "检索命中不足",
            "citation_failure": "引用证据不足",
            "answer_faithfulness_failure": "答案支撑不足",
            "overall_failure": "综合分未达标",
        }
        counts: dict[str, int] = {}
        examples: dict[str, list[str]] = {}
        payload_by_key: dict[str, dict] = {}
        for result in results:
            key = EvalService._failure_mode_key(result)
            if key is None:
                continue
            counts[key] = counts.get(key, 0) + 1
            diagnosis = EvalService._failure_diagnosis_payload(result)
            if diagnosis is not None and key not in payload_by_key:
                payload_by_key[key] = diagnosis
            case_name = str((result.details_json or {}).get("case_name") or result.case_id)
            examples.setdefault(key, [])
            if case_name not in examples[key] and len(examples[key]) < 3:
                examples[key].append(case_name)
        return [
            EvalFailureModeRead(
                key=key,
                label=(
                    str(payload_by_key.get(key, {}).get("reason_label"))
                    if payload_by_key.get(key, {}).get("reason_label")
                    else labels.get(key, key)
                ),
                count=count,
                stage=(
                    str(payload_by_key.get(key, {}).get("stage"))
                    if payload_by_key.get(key, {}).get("stage")
                    else None
                ),
                stage_label=(
                    str(payload_by_key.get(key, {}).get("stage_label"))
                    if payload_by_key.get(key, {}).get("stage_label")
                    else None
                ),
                example_case_names=examples.get(key, []),
            )
            for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ]

    @staticmethod
    def _failure_mode_key(result: EvalResult) -> str | None:
        if result.overall_pass:
            return None
        diagnosis = EvalService._failure_diagnosis_payload(result)
        if diagnosis and isinstance(diagnosis.get("reason_code"), str) and diagnosis.get("reason_code"):
            return str(diagnosis["reason_code"])
        details = result.details_json or {}
        metric_breakdown = details.get("metric_breakdown") if isinstance(details, dict) else {}
        metric_breakdown = metric_breakdown if isinstance(metric_breakdown, dict) else {}
        permission = metric_breakdown.get("permission_isolation") if isinstance(metric_breakdown.get("permission_isolation"), dict) else {}
        retrieval = metric_breakdown.get("retrieval") if isinstance(metric_breakdown.get("retrieval"), dict) else {}
        citation = metric_breakdown.get("citation") if isinstance(metric_breakdown.get("citation"), dict) else {}
        faithfulness = metric_breakdown.get("faithfulness") if isinstance(metric_breakdown.get("faithfulness"), dict) else {}

        if not result.permission_isolation_correct or permission.get("passed") is False:
            return "permission_failure"
        if EvalService._metric_score(retrieval, result.retrieval_hit_rate) < 0.5:
            return "retrieval_failure"
        if EvalService._metric_score(citation, result.citation_accuracy) < 0.5:
            return "citation_failure"
        if EvalService._metric_score(faithfulness, result.answer_faithfulness) < 0.7:
            return "answer_faithfulness_failure"
        return "overall_failure"

    @staticmethod
    def _failure_diagnosis_payload(result: EvalResult) -> dict | None:
        details = result.details_json or {}
        payload = details.get("pipeline_diagnosis") if isinstance(details, dict) else None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _metric_score(payload: dict, fallback: float) -> float:
        value = payload.get("score") if isinstance(payload, dict) else None
        if isinstance(value, (int, float)):
            return float(value)
        return float(fallback or 0.0)

    @staticmethod
    def _summary_float(summary: dict, key: str) -> float:
        value = summary.get(key)
        return round(float(value), 4) if isinstance(value, (int, float)) else 0.0

    @staticmethod
    def _summary_int(summary: dict, key: str) -> int:
        value = summary.get(key)
        return int(value) if isinstance(value, int) else 0

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
            if run.status not in {"queued", "running"} or run.finished_at is not None:
                continue
            reference_time = self._stale_reference_time(run)
            if reference_time is None:
                continue
            threshold = (
                STALE_QUEUED_EVAL_RUN_THRESHOLD
                if run.status == "queued"
                else STALE_RUNNING_EVAL_RUN_THRESHOLD
            )
            if now - reference_time < threshold:
                continue
            run.status = "failed"
            run.finished_at = now
            if not run.error_text:
                run.error_text = "Eval run did not complete and was automatically marked as failed."
            if isinstance(run.summary_json, dict):
                run.summary_json = {
                    **run.summary_json,
                    "client_request_status": "failed",
                }
            stale_runs.append(run)
        if stale_runs:
            self.session.commit()

    @staticmethod
    def _build_running_progress_summary(completed_cases: int, total_cases: int) -> dict:
        return {
            "completed_cases": completed_cases,
            "total_cases": total_cases,
            "last_progress_at": datetime.now(UTC).isoformat(),
        }

    @staticmethod
    def _stale_reference_time(run: EvalRun) -> datetime | None:
        if run.status == "queued":
            return EvalService._ensure_aware_datetime(run.created_at)

        summary = run.summary_json if isinstance(run.summary_json, dict) else {}
        progress_time = EvalService._parse_summary_datetime(summary.get("last_progress_at"))
        return progress_time or EvalService._ensure_aware_datetime(run.started_at) or EvalService._ensure_aware_datetime(run.created_at)

    @staticmethod
    def _parse_summary_datetime(value: object) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            return EvalService._ensure_aware_datetime(datetime.fromisoformat(value))
        except ValueError:
            return None

    @staticmethod
    def _ensure_aware_datetime(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

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
        external_annotation = EvalService._parse_external_case_annotation(getattr(case, "notes", None))
        expected_document_titles = list(case.expected_document_titles or [])
        expected_answer_keywords = list(case.expected_answer_keywords or [])

        expected_outcome = (
            demo_annotation.get("expected_outcome")
            if demo_annotation and demo_annotation.get("expected_outcome")
            else external_annotation.get("expected_outcome")
            if external_annotation and external_annotation.get("expected_outcome")
            else "refuse" if not expected_document_titles else "answer"
        )
        expected_retrieval_titles = (
            list(demo_annotation.get("expected_retrieval_titles", []))
            if demo_annotation and demo_annotation.get("expected_retrieval_titles")
            else list(external_annotation.get("expected_retrieval_titles", []))
            if external_annotation and external_annotation.get("expected_retrieval_titles")
            else expected_document_titles
        )
        expected_evidence_titles = (
            list(demo_annotation.get("expected_evidence_titles", []))
            if demo_annotation and demo_annotation.get("expected_evidence_titles")
            else list(external_annotation.get("expected_evidence_titles", []))
            if external_annotation and external_annotation.get("expected_evidence_titles")
            else expected_document_titles
        )
        raw_expected_key_facts = (
            list(demo_annotation.get("expected_key_facts", []))
            if demo_annotation and demo_annotation.get("expected_key_facts")
            else list(external_annotation.get("expected_key_facts", []))
            if external_annotation and external_annotation.get("expected_key_facts")
            else expected_answer_keywords
        )
        raw_forbidden_key_facts = (
            list(demo_annotation.get("forbidden_key_facts", []))
            if demo_annotation and demo_annotation.get("forbidden_key_facts")
            else list(external_annotation.get("forbidden_key_facts", []))
            if external_annotation and external_annotation.get("forbidden_key_facts")
            else []
        )
        expected_key_fact_specs = EvalService._normalize_fact_specs(raw_expected_key_facts)
        forbidden_key_fact_specs = EvalService._normalize_fact_specs(raw_forbidden_key_facts)
        scoring_notes = (
            str(demo_annotation.get("scoring_notes"))
            if demo_annotation and demo_annotation.get("scoring_notes")
            else str(external_annotation.get("scoring_notes"))
            if external_annotation and external_annotation.get("scoring_notes")
            else None
        )
        return {
            "source": "demo_annotations" if demo_annotation else "external_notes" if external_annotation else "legacy_case_fields",
            "expected_outcome": expected_outcome,
            "expected_retrieval_titles": expected_retrieval_titles,
            "expected_evidence_titles": expected_evidence_titles,
            "expected_key_facts": [item["label"] for item in expected_key_fact_specs],
            "expected_evidence_markers": (
                list(demo_annotation.get("expected_evidence_markers", []))
                if demo_annotation and demo_annotation.get("expected_evidence_markers")
                else list(external_annotation.get("expected_evidence_markers", []))
                if external_annotation and external_annotation.get("expected_evidence_markers")
                else []
            ),
            "forbidden_key_facts": [item["label"] for item in forbidden_key_fact_specs],
            "expected_key_fact_specs": expected_key_fact_specs,
            "forbidden_key_fact_specs": forbidden_key_fact_specs,
            "scoring_notes": scoring_notes,
        }

    @staticmethod
    def _parse_external_case_annotation(notes: str | None) -> dict | None:
        if not notes:
            return None

        try:
            payload = json.loads(notes)
        except (TypeError, ValueError):
            return None

        if not isinstance(payload, dict):
            return None

        annotation = payload.get("benchmark_annotation", payload)
        if not isinstance(annotation, dict):
            return None

        allowed_keys = {
            "expected_outcome",
            "expected_retrieval_titles",
            "expected_evidence_titles",
            "expected_key_facts",
            "expected_evidence_markers",
            "forbidden_key_facts",
            "scoring_notes",
        }
        return {key: value for key, value in annotation.items() if key in allowed_keys}

    @staticmethod
    def _ensure_admin(actor: User) -> None:
        if actor.role is None or actor.role.name != RoleName.ADMIN:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin permission required.")

