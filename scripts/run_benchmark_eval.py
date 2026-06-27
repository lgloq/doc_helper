from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.db.session import SessionLocal
from app.models.eval import EvalRun
from app.repositories.user_repository import UserRepository
from app.schemas.eval import EvalRunRequest
from app.services.auth.bootstrap import seed_mock_data
from app.services.eval.service import EvalService


def main() -> None:
    args = build_parser().parse_args()
    if args.local_baseline:
        apply_local_baseline_env()
    if args.retrieval_ablation:
        apply_retrieval_ablation_env(args.retrieval_ablation)
    manifest_case_names = load_manifest_case_names(args.manifest) if args.manifest else None
    if args.from_run:
        report = report_existing_run(run_id=args.from_run)
    else:
        report = run_benchmark_eval(
            dataset_name=args.dataset,
            top_k=args.top_k,
            admin_email=args.admin_email,
            seed_demo_cases=args.seed_demo_cases,
            limit=args.limit,
            case_name_contains=args.case_name_contains,
            case_name=args.case_name,
            manifest_case_names=manifest_case_names,
            manifest_path=args.manifest,
        )

    output_path = Path(args.output) if args.output else default_output_path(args.dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {output_path}")
    print(summary_text(report))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a benchmark eval dataset and write a detailed report.")
    parser.add_argument(
        "--dataset",
        default="stard_zh_law_docs_small",
        help="Eval dataset name, e.g. stard_zh_law_docs_small.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k context size to use for evaluation.")
    parser.add_argument("--admin-email", default="admin@local.test", help="Admin user used to launch the eval.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum cases to run from the selected dataset.")
    parser.add_argument("--case-name", action="append", help="Exact case_name to run. Can be repeated.")
    parser.add_argument("--case-name-contains", help="Only run cases whose case_name contains this substring.")
    parser.add_argument(
        "--manifest",
        help="Optional benchmark manifest JSON. When provided, only cases listed in manifest.cases are evaluated.",
    )
    parser.add_argument(
        "--retrieval-ablation",
        choices=["full_indexed_sparse", "indexed_sparse_only"],
        help="Apply the same retrieval source switches used by run_retrieval_ablation_benchmark.py.",
    )
    parser.add_argument("--from-run", help="Do not run eval; build a report from an existing eval run id.")
    parser.add_argument(
        "--local-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use deterministic local providers for reproducible benchmark runs.",
    )
    parser.add_argument(
        "--seed-demo-cases",
        action="store_true",
        help="Seed demo eval cases before running the selected dataset.",
    )
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser


def apply_local_baseline_env() -> None:
    os.environ["EMBEDDING_PROVIDER"] = "deterministic"
    os.environ["QUERY_REWRITE_PROVIDER"] = "deterministic"
    os.environ["RERANK_PROVIDER"] = "heuristic"
    os.environ["RETRIEVAL_HEURISTIC_RERANK_ENABLED"] = "false"
    os.environ["RETRIEVAL_VECTOR_ENABLED"] = "false"
    os.environ["RERANK_MAX_CANDIDATES"] = "16"
    os.environ["RETRIEVAL_DOCUMENT_DIVERSITY_MAX_CHUNKS"] = "5"
    os.environ["RETRIEVAL_IN_DOCUMENT_EXPANSION_ENABLED"] = "true"
    os.environ["RETRIEVAL_IN_DOCUMENT_EXPANSION_SEED_COUNT"] = "10"
    os.environ["RETRIEVAL_IN_DOCUMENT_EXPANSION_PER_DOCUMENT"] = "5"
    os.environ["RETRIEVAL_IN_DOCUMENT_EXPANSION_MAX_CANDIDATES"] = "32"
    os.environ["RETRIEVAL_IN_DOCUMENT_EXPANSION_SCORE_WEIGHT"] = "0.42"
    os.environ["ANSWER_PROVIDER"] = "deterministic"
    os.environ["ROUTER_PROVIDER"] = "deterministic"
    from app.core.config import get_settings

    get_settings.cache_clear()


def apply_retrieval_ablation_env(ablation_name: str) -> None:
    if ablation_name == "full_indexed_sparse":
        values = {
            "RETRIEVAL_LEXICAL_ENABLED": "true",
            "RETRIEVAL_VECTOR_ENABLED": "false",
            "RETRIEVAL_INDEXED_SPARSE_ENABLED": "true",
            "RETRIEVAL_STRUCTURAL_ENABLED": "true",
            "RETRIEVAL_IN_DOCUMENT_EXPANSION_ENABLED": "true",
            "RETRIEVAL_DOCUMENT_EVIDENCE_SWEEP_ENABLED": "false",
            "RETRIEVAL_DOCUMENT_FIRST_EVIDENCE_ENABLED": "false",
            "RETRIEVAL_DOCUMENT_NEIGHBOR_CONTEXT_ENABLED": "false",
            "RETRIEVAL_EVIDENCE_PRESERVATION_ENABLED": "false",
            "RETRIEVAL_FINAL_COVERAGE_ENABLED": "true",
            "RETRIEVAL_EVIDENCE_QUERY_BRIDGE_ENABLED": "false",
            "RETRIEVAL_HEURISTIC_RERANK_ENABLED": "false",
            "RERANK_PROVIDER": "heuristic",
        }
    elif ablation_name == "indexed_sparse_only":
        values = {
            "RETRIEVAL_LEXICAL_ENABLED": "false",
            "RETRIEVAL_VECTOR_ENABLED": "false",
            "RETRIEVAL_INDEXED_SPARSE_ENABLED": "true",
            "RETRIEVAL_STRUCTURAL_ENABLED": "false",
            "RETRIEVAL_IN_DOCUMENT_EXPANSION_ENABLED": "false",
            "RETRIEVAL_DOCUMENT_EVIDENCE_SWEEP_ENABLED": "false",
            "RETRIEVAL_DOCUMENT_FIRST_EVIDENCE_ENABLED": "false",
            "RETRIEVAL_DOCUMENT_NEIGHBOR_CONTEXT_ENABLED": "false",
            "RETRIEVAL_EVIDENCE_PRESERVATION_ENABLED": "false",
            "RETRIEVAL_FINAL_COVERAGE_ENABLED": "true",
            "RETRIEVAL_EVIDENCE_QUERY_BRIDGE_ENABLED": "false",
            "RETRIEVAL_HEURISTIC_RERANK_ENABLED": "false",
            "RERANK_PROVIDER": "heuristic",
        }
    else:
        raise SystemExit(f"Unsupported retrieval ablation: {ablation_name}")

    os.environ.update(values)
    from app.core.config import get_settings

    get_settings.cache_clear()


def load_manifest_case_names(manifest_path: str) -> list[str]:
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    case_names = [
        str(case.get("case_name") or "").strip()
        for case in manifest.get("cases", [])
        if str(case.get("case_name") or "").strip()
    ]
    return list(dict.fromkeys(case_names))


def run_benchmark_eval(
    *,
    dataset_name: str,
    top_k: int,
    admin_email: str,
    seed_demo_cases: bool,
    limit: int | None,
    case_name_contains: str | None,
    case_name: list[str] | None,
    manifest_case_names: list[str] | None = None,
    manifest_path: str | None = None,
) -> dict[str, Any]:
    seed_mock_data()
    session = SessionLocal()
    try:
        admin = UserRepository(session).get_by_email(admin_email)
        if admin is None:
            raise SystemExit(f"Admin user not found: {admin_email}")

        service = EvalService(session)
        case_ids = [
            case.id
            for case in select_cases(
                service=service,
                dataset_name=dataset_name,
                limit=limit,
                case_name_contains=case_name_contains,
                case_names=case_name or [],
                manifest_case_names=manifest_case_names or [],
            )
        ]
        run = service.run_eval(
            admin,
            EvalRunRequest(
                dataset_name=dataset_name,
                top_k=top_k,
                seed_demo_cases=seed_demo_cases,
                case_ids=case_ids,
            ),
        )
        report = enrich_report(run.model_dump(mode="json"))
        if manifest_path:
            report["manifest"] = str(Path(manifest_path).resolve())
            report["manifest_case_count"] = len(manifest_case_names or [])
        return report
    finally:
        session.close()


def report_existing_run(*, run_id: str) -> dict[str, Any]:
    session = SessionLocal()
    try:
        run = session.get(EvalRun, run_id)
        if run is None:
            raise SystemExit(f"Eval run not found: {run_id}")
        admin = UserRepository(session).get_by_email("admin@local.test")
        if admin is None:
            raise SystemExit("Admin user not found: admin@local.test")
        service = EvalService(session)
        return enrich_report(service.get_run(admin, run.id).model_dump(mode="json"))
    finally:
        session.close()


def select_cases(
    *,
    service: EvalService,
    dataset_name: str,
    limit: int | None,
    case_name_contains: str | None,
    case_names: list[str],
    manifest_case_names: list[str],
):
    cases = service.eval_repository.list_cases(dataset_name)
    if manifest_case_names:
        wanted_from_manifest = set(manifest_case_names)
        cases = [case for case in cases if case.case_name in wanted_from_manifest]
    if case_names:
        wanted = set(case_names)
        cases = [case for case in cases if case.case_name in wanted]
    if case_name_contains:
        cases = [case for case in cases if case_name_contains in case.case_name]
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise SystemExit("No eval cases matched the requested filters.")
    return cases


def enrich_report(report: dict[str, Any]) -> dict[str, Any]:
    report["generated_at"] = datetime.now(UTC).isoformat()
    report["partial_result_count"] = len(report.get("results") or [])
    report["computed_summary_json"] = build_computed_summary(report)
    report["failure_summary"] = build_failure_summary(report)
    report["failure_cases"] = extract_failure_cases(report)
    return report


def build_computed_summary(report: dict[str, Any]) -> dict[str, Any]:
    results = list(report.get("results") or [])
    summary = build_result_summary(results)
    answer_results = [item for item in results if case_type_key(item) == "answer_expected"]
    refusal_results = [item for item in results if case_type_key(item) == "refusal_expected"]
    summary["case_type_breakdown"] = {
        "answer_expected": build_result_summary(answer_results, profile="answer_expected", label="回答型"),
        "refusal_expected": build_result_summary(refusal_results, profile="refusal_expected", label="拒答/权限型"),
    }
    return summary


def build_result_summary(
    results: list[dict[str, Any]],
    *,
    profile: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
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
        total = len(results)
        pass_count = sum(1 for item in results if item.get("overall_pass"))
        permission_pass_count = sum(1 for item in results if item.get("permission_isolation_correct"))
        overall_scores = [
            float(((item.get("details_json") or {}).get("metric_breakdown") or {}).get("overall", {}).get("score", 0.0))
            for item in results
        ]
        permission_scores = [
            float(((item.get("details_json") or {}).get("metric_breakdown") or {}).get("permission_isolation", {}).get("score", 0.0))
            for item in results
        ]
        summary = {
            "total_cases": total,
            "pass_count": pass_count,
            "pass_rate": round(pass_count / total, 4),
            "retrieval_hit_rate_avg": average_score(item.get("retrieval_hit_rate") for item in results),
            "citation_accuracy_avg": average_score(item.get("citation_accuracy") for item in results),
            "answer_faithfulness_avg": average_score(item.get("answer_faithfulness") for item in results),
            "overall_score_avg": average_score(overall_scores),
            "permission_isolation_pass_rate": round(permission_pass_count / total, 4),
            "permission_isolation_score_avg": average_score(permission_scores),
        }
    if profile:
        summary["profile"] = profile
    if label:
        summary["label"] = label
    return summary


def average_score(values) -> float:
    clean_values = [float(value or 0.0) for value in values]
    if not clean_values:
        return 0.0
    return round(sum(clean_values) / len(clean_values), 4)


def case_type_key(item: dict[str, Any]) -> str:
    details = item.get("details_json") or {}
    annotations = details.get("case_annotations") or {}
    return "refusal_expected" if annotations.get("expected_outcome") == "refuse" else "answer_expected"


def build_failure_summary(report: dict[str, Any]) -> dict[str, Any]:
    results = list(report.get("results") or [])
    case_type_counts: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    failure_modes: Counter[str] = Counter()
    failure_stages: Counter[str] = Counter()
    for item in results:
        details = item.get("details_json") or {}
        case_type_counts[case_type_key(item)] += 1
        if item.get("overall_pass"):
            continue
        reason = details.get("human_review", {}).get("reason") or "unknown"
        failure_reasons[reason] += 1
        diagnosis = details.get("pipeline_diagnosis") or {}
        if isinstance(diagnosis, dict) and diagnosis.get("stage") and diagnosis.get("stage") != "passed":
            failure_stages[str(diagnosis.get("stage"))] += 1
        failure_modes[classify_failure(details, item)] += 1

    return {
        "total_cases": len(results),
        "passed_cases": sum(1 for item in results if item.get("overall_pass")),
        "failed_cases": sum(1 for item in results if not item.get("overall_pass")),
        "case_type_counts": dict(case_type_counts),
        "failure_reason_counts": dict(failure_reasons),
        "failure_mode_counts": dict(failure_modes),
        "failure_stage_counts": dict(failure_stages),
    }


def extract_failure_cases(report: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    for item in report.get("results") or []:
        if item.get("overall_pass"):
            continue
        details = item.get("details_json") or {}
        metric_breakdown = details.get("metric_breakdown") or {}
        failures.append(
            {
                "case_name": details.get("case_name"),
                "dataset_name": details.get("dataset_name"),
                "acting_user_email": item.get("acting_user_email"),
                "expected_outcome": (details.get("case_annotations") or {}).get("expected_outcome"),
                "failure_mode": classify_failure(details, item),
                "human_review_reason": details.get("human_review", {}).get("reason"),
                "retrieval_hit_rate": item.get("retrieval_hit_rate"),
                "citation_accuracy": item.get("citation_accuracy"),
                "answer_faithfulness": item.get("answer_faithfulness"),
                "permission_isolation_correct": item.get("permission_isolation_correct"),
                "retrieved_document_titles": details.get("retrieved_document_titles", []),
                "citation_document_titles": details.get("citation_document_titles", []),
                "matched_expected_titles": details.get("matched_expected_titles", []),
                "missing_expected_titles": details.get("missing_expected_titles", []),
                "matched_citation_titles": details.get("matched_citation_titles", []),
                "missing_citation_titles": details.get("missing_citation_titles", []),
                "missing_answer_keywords": details.get("missing_answer_keywords", []),
                "permission_checks": details.get("permission_checks", {}),
                "metric_breakdown": {
                    "retrieval": metric_breakdown.get("retrieval", {}),
                    "citation": metric_breakdown.get("citation", {}),
                    "faithfulness": metric_breakdown.get("faithfulness", {}),
                    "permission_isolation": metric_breakdown.get("permission_isolation", {}),
                    "overall": metric_breakdown.get("overall", {}),
                },
                "answer_excerpt": details.get("answer_excerpt"),
                "trace_id": details.get("trace_id"),
                "pipeline_diagnosis": details.get("pipeline_diagnosis"),
            }
        )
    return failures


def classify_failure(details: dict[str, Any], result: dict[str, Any]) -> str:
    diagnosis = details.get("pipeline_diagnosis") or {}
    if isinstance(diagnosis, dict) and diagnosis.get("reason_code"):
        return str(diagnosis["reason_code"])
    metric_breakdown = details.get("metric_breakdown") or {}
    permission = metric_breakdown.get("permission_isolation") or {}
    retrieval = metric_breakdown.get("retrieval") or {}
    citation = metric_breakdown.get("citation") or {}
    faithfulness = metric_breakdown.get("faithfulness") or {}
    if result.get("permission_isolation_correct") is False or not permission.get("passed", True):
        return "permission_leak"
    if retrieval.get("score", 1.0) < 0.5:
        return "retrieval_failure"
    if citation.get("score", 1.0) < 0.5:
        return "citation_failure"
    if faithfulness.get("score", 1.0) < 0.5:
        return "answer_faithfulness_failure"
    annotations = details.get("case_annotations") or {}
    if annotations.get("expected_outcome") == "refuse":
        return "refusal_failure"
    return "overall_failure"


def summary_text(report: dict[str, Any]) -> str:
    summary = report.get("computed_summary_json") or report.get("summary_json") or {}
    failure_summary = report.get("failure_summary") or {}
    lines = [
        f"dataset={report.get('dataset_name')} status={report.get('status')} total={summary.get('total_cases', 0)} pass={summary.get('pass_count', 0)}",
        f"retrieval={summary.get('retrieval_hit_rate_avg', 0)} citation={summary.get('citation_accuracy_avg', 0)} faithfulness={summary.get('answer_faithfulness_avg', 0)} permission={summary.get('permission_isolation_pass_rate', 0)}",
        f"failures={failure_summary.get('failed_cases', 0)} modes={failure_summary.get('failure_mode_counts', {})} stages={failure_summary.get('failure_stage_counts', {})}",
    ]
    return "\n".join(lines)


def default_output_path(dataset_name: str) -> Path:
    safe_name = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in dataset_name)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return BACKEND_DIR / "data" / "eval_outputs" / f"{safe_name}-run-{timestamp}.json"


if __name__ == "__main__":
    main()
