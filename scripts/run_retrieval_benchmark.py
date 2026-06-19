from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from sqlalchemy import select, text


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.db.session import SessionLocal
from app.models.document import Document
from app.repositories.eval_repository import EvalRepository
from app.repositories.user_repository import UserRepository
from app.schemas.search import SearchRequest
from app.services.auth.bootstrap import seed_mock_data
from app.services.eval.service import EvalService
from app.services.retrieval.service import RetrievalService


def main() -> None:
    args = build_parser().parse_args()
    if args.local_baseline:
        apply_local_baseline_env(args.domain_profile)
    else:
        apply_domain_profile_env(args.domain_profile)
    apply_retrieval_ablation_overrides(
        cjk_python_fallback_mode=args.cjk_python_fallback_mode,
        cjk_python_scorer=args.cjk_python_scorer,
    )
    report = run_retrieval_benchmark(
        dataset_name=args.dataset,
        top_k=args.top_k,
        limit=args.limit,
        case_names=args.case_name,
        case_name_contains=args.case_name_contains,
        document_title_prefix=args.document_title_prefix,
        manifest_scope=args.manifest_scope,
        case_statement_timeout_ms=args.case_statement_timeout_ms,
        progress=args.progress,
    )
    output_path = Path(args.output) if args.output else default_output_path(args.dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(summary_text(report))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval-only benchmark metrics for imported eval cases.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--case-name",
        action="append",
        default=None,
        help="Exact eval case name to include. Can be repeated.",
    )
    parser.add_argument("--case-name-contains", default=None)
    parser.add_argument(
        "--document-title-prefix",
        default=None,
        help="Restrict retrieval to documents whose title starts with this prefix. Defaults to '<dataset>:'.",
    )
    parser.add_argument(
        "--manifest-scope",
        default=None,
        help="Restrict retrieval to the document titles declared in this benchmark manifest JSON.",
    )
    parser.add_argument(
        "--local-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic embeddings and disable external rewrite/rerank calls for reproducible benchmark speed.",
    )
    parser.add_argument(
        "--domain-profile",
        choices=["enterprise", "legal_benchmark"],
        default="enterprise",
        help="Retrieval domain profile. enterprise is the product default; legal_benchmark enables STARD/legal-specific rules.",
    )
    parser.add_argument(
        "--cjk-python-fallback-mode",
        choices=["auto", "always", "off"],
        default=None,
        help="Override CJK Python lexical fallback mode after local-baseline env is applied.",
    )
    parser.add_argument(
        "--cjk-python-scorer",
        choices=["weighted", "bm25"],
        default=None,
        help="Override the local CJK Python lexical scorer for sparse retrieval ablations.",
    )
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument(
        "--case-statement-timeout-ms",
        type=int,
        default=None,
        help="Optional PostgreSQL statement_timeout per case. Timed-out cases are recorded as retrieval_timeout.",
    )
    parser.add_argument("--progress", action="store_true", help="Print per-case progress and elapsed time to stderr.")
    return parser


def apply_retrieval_ablation_overrides(*, cjk_python_fallback_mode: str | None, cjk_python_scorer: str | None) -> None:
    if cjk_python_fallback_mode:
        os.environ["RETRIEVAL_CJK_PYTHON_FALLBACK_MODE"] = cjk_python_fallback_mode
    if cjk_python_scorer:
        os.environ["RETRIEVAL_CJK_PYTHON_SCORER"] = cjk_python_scorer
    if cjk_python_fallback_mode or cjk_python_scorer:
        from app.core.config import get_settings

        get_settings.cache_clear()


def apply_local_baseline_env(domain_profile: str = "enterprise") -> None:
    os.environ["EMBEDDING_PROVIDER"] = "deterministic"
    os.environ["QUERY_REWRITE_PROVIDER"] = "deterministic"
    os.environ["RERANK_PROVIDER"] = "heuristic"
    os.environ["RETRIEVAL_HEURISTIC_RERANK_ENABLED"] = "false"
    os.environ["RETRIEVAL_QUERY_PLAN_PROBE_ENABLED"] = "false"
    os.environ["RETRIEVAL_VECTOR_ENABLED"] = "false"
    os.environ["RERANK_MAX_CANDIDATES"] = "16"
    os.environ["RETRIEVAL_DOCUMENT_DIVERSITY_MAX_CHUNKS"] = "5"
    os.environ["RETRIEVAL_CJK_PYTHON_FALLBACK_MODE"] = "auto"
    os.environ["RETRIEVAL_IN_DOCUMENT_EXPANSION_ENABLED"] = "true"
    os.environ["RETRIEVAL_IN_DOCUMENT_EXPANSION_SEED_COUNT"] = "10"
    os.environ["RETRIEVAL_IN_DOCUMENT_EXPANSION_PER_DOCUMENT"] = "5"
    os.environ["RETRIEVAL_IN_DOCUMENT_EXPANSION_MAX_CANDIDATES"] = "32"
    os.environ["RETRIEVAL_IN_DOCUMENT_EXPANSION_SCORE_WEIGHT"] = "0.42"
    os.environ["RETRIEVAL_INDEXED_SPARSE_ENABLED"] = "false"
    os.environ["RETRIEVAL_DOCUMENT_EVIDENCE_SWEEP_ENABLED"] = "false"
    os.environ["RETRIEVAL_EVIDENCE_PRESERVATION_ENABLED"] = "false"
    apply_domain_profile_env(domain_profile)


def apply_domain_profile_env(domain_profile: str = "enterprise") -> None:
    os.environ["RETRIEVAL_DOMAIN_PROFILE"] = domain_profile
    from app.core.config import get_settings

    get_settings.cache_clear()


def run_retrieval_benchmark(
    *,
    dataset_name: str,
    top_k: int,
    limit: int | None,
    case_names: list[str] | None,
    case_name_contains: str | None,
    document_title_prefix: str | None,
    manifest_scope: str | None,
    case_statement_timeout_ms: int | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    seed_mock_data()
    session = SessionLocal()
    try:
        eval_repository = EvalRepository(session)
        user_repository = UserRepository(session)
        retrieval_service = RetrievalService(session)
        scoped_document_ids, document_scope = resolve_scoped_document_ids(
            session=session,
            dataset_name=dataset_name,
            document_title_prefix=document_title_prefix,
            manifest_scope=manifest_scope,
        )
        cases = eval_repository.list_cases(dataset_name)
        if case_names:
            selected = set(case_names)
            cases = [case for case in cases if case.case_name in selected]
        if case_name_contains:
            cases = [case for case in cases if case_name_contains in case.case_name]
        if limit is not None:
            cases = cases[:limit]
        if not cases:
            raise SystemExit("No eval cases matched the requested filters.")

        rows = []
        total_cases = len(cases)
        for case_index, case in enumerate(cases, start=1):
            case_started = perf_counter()
            if progress:
                print(f"[case {case_index}/{total_cases}] start {case.case_name}", file=sys.stderr, flush=True)
            actor = user_repository.get_by_email(case.acting_user_email)
            if actor is None:
                row = missing_actor_row(case)
                row["elapsed_seconds"] = round(perf_counter() - case_started, 4)
                rows.append(row)
                if progress:
                    elapsed = row["elapsed_seconds"]
                    print(f"[case {case_index}/{total_cases}] missing_actor {elapsed:.2f}s", file=sys.stderr, flush=True)
                continue
            annotations = EvalService._resolve_case_annotations(case)
            expected_titles = normalize_titles(annotations["expected_retrieval_titles"])
            forbidden_titles = normalize_titles(case.forbidden_document_titles)
            try:
                if case_statement_timeout_ms is not None and session.bind and session.bind.dialect.name == "postgresql":
                    timeout_ms = max(1, int(case_statement_timeout_ms))
                    session.execute(text(f"SET statement_timeout = {timeout_ms}"))
                response = retrieval_service.search(
                    actor,
                    SearchRequest(query=case.question, top_k=top_k),
                    scoped_document_ids=scoped_document_ids,
                )
            except Exception as exc:  # noqa: BLE001 - benchmark should isolate one failed case.
                session.rollback()
                row = retrieval_error_row(
                    case=case,
                    annotations=annotations,
                    expected_titles=expected_titles,
                    forbidden_titles=forbidden_titles,
                    error=exc,
                    statement_timeout_ms=case_statement_timeout_ms,
                )
                row["elapsed_seconds"] = round(perf_counter() - case_started, 4)
                rows.append(row)
                if progress:
                    elapsed = row["elapsed_seconds"]
                    print(f"[case {case_index}/{total_cases}] error {elapsed:.2f}s {type(exc).__name__}", file=sys.stderr, flush=True)
                continue
            ranked_chunks = [
                {
                    "document_title": item.document_title,
                    "chunk_index": item.chunk_index,
                    "section_title": item.section_title,
                    "clause_full_name": item.clause_full_name,
                    "article_number": item.article_number,
                    "chunk_type": item.chunk_type,
                    "content": item.content,
                }
                for item in response.matched_chunks
            ]
            ranked_titles = ordered_titles(item["document_title"] for item in ranked_chunks)
            row = score_case(
                case=case,
                annotations=annotations,
                ranked_titles=ranked_titles,
                ranked_chunks=ranked_chunks,
                expected_titles=expected_titles,
                forbidden_titles=forbidden_titles,
                retrieval_debug=response.debug.model_dump(),
            )
            row["elapsed_seconds"] = round(perf_counter() - case_started, 4)
            rows.append(row)
            if progress:
                elapsed = row["elapsed_seconds"]
                status = "pass" if row.get("passed") else str(row.get("failure_mode") or "failed")
                print(f"[case {case_index}/{total_cases}] done {elapsed:.2f}s {status}", file=sys.stderr, flush=True)

        report = {
            "dataset_name": dataset_name,
            "top_k": top_k,
            "retrieval_domain_profile": retrieval_service.settings.effective_retrieval_domain_profile,
            "document_scope": document_scope,
            "generated_at": datetime.now(UTC).isoformat(),
            "case_count": len(rows),
            "summary": summarize(rows),
            "latency": summarize_latency(rows),
            "slowest_cases": slowest_cases(rows),
            "cases": rows,
            "failure_cases": [row for row in rows if not row["passed"]],
        }
        return report
    finally:
        session.close()


def resolve_scoped_document_ids(
    *,
    session,
    dataset_name: str,
    document_title_prefix: str | None,
    manifest_scope: str | None = None,
) -> tuple[list | None, dict[str, Any]]:
    if manifest_scope:
        manifest_path = Path(manifest_scope)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        titles = [str(document["title"]) for document in manifest.get("documents", [])]
        rows = list(session.execute(select(Document.id, Document.title).where(Document.title.in_(titles))).all())
        found_titles = {str(row.title) for row in rows}
        ids = [row.id for row in rows]
        return ids, {
            "mode": "manifest",
            "manifest": str(manifest_path),
            "manifest_document_count": len(titles),
            "document_count": len(ids),
            "missing_titles": sorted(set(titles) - found_titles)[:20],
        }
    prefix = document_title_prefix if document_title_prefix is not None else f"{dataset_name}:"
    if not prefix:
        return None, {"mode": "unscoped", "document_count": None}
    ids = list(session.scalars(select(Document.id).where(Document.title.like(f"{prefix}%"))).all())
    return ids, {
        "mode": "title_prefix",
        "title_prefix": prefix,
        "document_count": len(ids),
    }


def score_case(
    *,
    case,
    annotations: dict[str, Any],
    ranked_titles: list[str],
    ranked_chunks: list[dict[str, Any]],
    expected_titles: set[str],
    forbidden_titles: set[str],
    retrieval_debug: dict[str, Any],
) -> dict[str, Any]:
    expected_outcome = annotations.get("expected_outcome") or ("refuse" if not expected_titles else "answer")
    evidence_markers = normalize_evidence_markers(annotations.get("expected_evidence_markers"))
    retrieved_set = set(ranked_titles)
    forbidden_hits = sorted(forbidden_titles.intersection(retrieved_set))
    evidence_score = score_evidence_markers(ranked_chunks, evidence_markers)
    if expected_titles:
        matched = sorted(expected_titles.intersection(retrieved_set))
        missing = sorted(expected_titles.difference(retrieved_set))
        recall = len(matched) / len(expected_titles)
        precision = len(matched) / len(retrieved_set) if retrieved_set else 0.0
        average_precision = average_precision_at_k(ranked_titles, expected_titles)
        mrr = reciprocal_rank(ranked_titles, expected_titles)
        ndcg = binary_ndcg(ranked_titles, expected_titles)
        evidence_required = bool(evidence_markers)
        evidence_passed = evidence_score["recall_at_k"] >= 1.0 if evidence_required else True
        passed = recall >= 1.0 and mrr > 0.0 and evidence_passed and not forbidden_hits
        if passed:
            failure_mode = "passed"
        elif missing:
            failure_mode = "expected_title_missing"
        elif not evidence_passed:
            failure_mode = "expected_evidence_missing"
        else:
            failure_mode = "forbidden_title_retrieved"
    else:
        matched = []
        missing = []
        recall = 1.0 if not forbidden_hits else 0.0
        precision = 1.0 if not forbidden_hits else 0.0
        average_precision = recall
        mrr = recall
        ndcg = recall
        passed = not forbidden_hits
        failure_mode = "passed" if passed else "forbidden_title_retrieved"

    return {
        "case_name": case.case_name,
        "acting_user_email": case.acting_user_email,
        "expected_outcome": expected_outcome,
        "question": case.question,
        "passed": passed,
        "failure_mode": failure_mode,
        "recall_at_k": round(recall, 4),
        "precision_at_k": round(precision, 4),
        "average_precision_at_k": round(average_precision, 4),
        "mrr": round(mrr, 4),
        "ndcg_at_k": round(ndcg, 4),
        "evidence_recall_at_k": evidence_score["recall_at_k"],
        "evidence_mrr": evidence_score["mrr"],
        "matched_evidence_markers": evidence_score["matched"],
        "missing_evidence_markers": evidence_score["missing"],
        "permission_isolation_correct": not forbidden_hits,
        "matched_expected_titles": matched,
        "missing_expected_titles": missing,
        "forbidden_hits": forbidden_hits,
        "retrieved_titles": ranked_titles,
        "retrieved_chunks": summarize_retrieved_chunks(ranked_chunks),
        "retrieval_debug": retrieval_debug,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    case_types = Counter(row["expected_outcome"] for row in rows)
    failure_modes = Counter(row["failure_mode"] for row in rows if not row["passed"])
    latency_summary = summarize_latency(rows)
    return {
        "total_cases": len(rows),
        "pass_count": sum(1 for row in rows if row["passed"]),
        "pass_rate": average_bool(row["passed"] for row in rows),
        "recall_at_k_avg": average(row["recall_at_k"] for row in rows),
        "precision_at_k_avg": average(row["precision_at_k"] for row in rows),
        "map_at_k": average(row["average_precision_at_k"] for row in rows),
        "mrr_avg": average(row["mrr"] for row in rows),
        "ndcg_at_k_avg": average(row["ndcg_at_k"] for row in rows),
        "evidence_recall_at_k_avg": average(row.get("evidence_recall_at_k", 1.0) for row in rows),
        "evidence_mrr_avg": average(row.get("evidence_mrr", 1.0) for row in rows),
        "permission_isolation_pass_rate": average_bool(row["permission_isolation_correct"] for row in rows),
        "case_type_counts": dict(case_types),
        "failure_mode_counts": dict(failure_modes),
        "latency_seconds": latency_summary,
    }


def summarize_latency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = sorted(float(row.get("elapsed_seconds") or 0.0) for row in rows if row.get("elapsed_seconds") is not None)
    if not values:
        return {
            "case_count": 0,
            "avg": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "case_count": len(values),
        "avg": round(sum(values) / len(values), 4),
        "p50": round(percentile(values, 0.50), 4),
        "p90": round(percentile(values, 0.90), 4),
        "p95": round(percentile(values, 0.95), 4),
        "max": round(max(values), 4),
    }


def slowest_cases(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    return [
        {
            "case_name": row.get("case_name"),
            "elapsed_seconds": row.get("elapsed_seconds"),
            "passed": row.get("passed"),
            "failure_mode": row.get("failure_mode"),
            "search_total_latency_ms": (row.get("retrieval_debug") or {}).get("search_total_latency_ms"),
        }
        for row in sorted(rows, key=lambda item: float(item.get("elapsed_seconds") or 0.0), reverse=True)[:limit]
    ]


def percentile(sorted_values: list[float], percentile_value: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile_value
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + ((upper_value - lower_value) * (position - lower))


def missing_actor_row(case) -> dict[str, Any]:
    return {
        "case_name": case.case_name,
        "acting_user_email": case.acting_user_email,
        "expected_outcome": "unknown",
        "question": case.question,
        "passed": False,
        "failure_mode": "missing_actor",
        "recall_at_k": 0.0,
        "precision_at_k": 0.0,
        "average_precision_at_k": 0.0,
        "mrr": 0.0,
        "ndcg_at_k": 0.0,
        "evidence_recall_at_k": 0.0,
        "evidence_mrr": 0.0,
        "matched_evidence_markers": [],
        "missing_evidence_markers": [],
        "permission_isolation_correct": False,
        "matched_expected_titles": [],
        "missing_expected_titles": list(case.expected_document_titles or []),
        "forbidden_hits": [],
        "retrieved_titles": [],
        "retrieved_chunks": [],
        "retrieval_debug": {},
    }


def retrieval_error_row(
    *,
    case,
    annotations: dict[str, Any],
    expected_titles: set[str],
    forbidden_titles: set[str],
    error: Exception,
    statement_timeout_ms: int | None,
) -> dict[str, Any]:
    error_text = str(error)
    is_timeout = "statement timeout" in error_text.lower() or "query_canceled" in error_text.lower()
    evidence_markers = normalize_evidence_markers(annotations.get("expected_evidence_markers"))
    return {
        "case_name": case.case_name,
        "acting_user_email": case.acting_user_email,
        "expected_outcome": annotations.get("expected_outcome") or ("refuse" if not expected_titles else "answer"),
        "question": case.question,
        "passed": False,
        "failure_mode": "retrieval_timeout" if is_timeout else "retrieval_error",
        "recall_at_k": 0.0 if expected_titles else 1.0,
        "precision_at_k": 0.0 if expected_titles else 1.0,
        "average_precision_at_k": 0.0 if expected_titles else 1.0,
        "mrr": 0.0 if expected_titles else 1.0,
        "ndcg_at_k": 0.0 if expected_titles else 1.0,
        "evidence_recall_at_k": 0.0 if evidence_markers else 1.0,
        "evidence_mrr": 0.0 if evidence_markers else 1.0,
        "matched_evidence_markers": [],
        "missing_evidence_markers": [marker["label"] for marker in evidence_markers],
        "permission_isolation_correct": True,
        "matched_expected_titles": [],
        "missing_expected_titles": sorted(expected_titles),
        "forbidden_hits": [],
        "retrieved_titles": [],
        "retrieved_chunks": [],
        "retrieval_debug": {
            "error_type": type(error).__name__,
            "error_message": error_text[:1000],
            "statement_timeout_ms": statement_timeout_ms,
        },
    }


def normalize_titles(values) -> set[str]:
    return {str(item).strip().lower() for item in values or [] if str(item).strip()}


def ordered_titles(values) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        title = str(item).strip().lower()
        if not title or title in seen:
            continue
        ordered.append(title)
        seen.add(title)
    return ordered


def normalize_evidence_markers(values) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    for item in values or []:
        if isinstance(item, dict):
            label = str(item.get("label") or item.get("name") or item.get("marker") or "").strip()
            aliases = [str(alias).strip() for alias in item.get("aliases", []) if str(alias).strip()]
            document_title = str(item.get("document_title") or "").strip().lower()
        else:
            label = str(item).strip()
            aliases = []
            document_title = ""
        if not label and not aliases:
            continue
        aliases = dedupe([label, *aliases])
        markers.append({"label": label or aliases[0], "aliases": aliases, "document_title": document_title})
    return markers


def score_evidence_markers(ranked_chunks: list[dict[str, Any]], markers: list[dict[str, Any]]) -> dict[str, Any]:
    if not markers:
        return {"recall_at_k": 1.0, "mrr": 1.0, "matched": [], "missing": []}

    matched: list[str] = []
    missing: list[str] = []
    ranks: list[int] = []
    for marker in markers:
        rank = first_marker_rank(ranked_chunks, marker)
        if rank is None:
            missing.append(marker["label"])
            continue
        matched.append(marker["label"])
        ranks.append(rank)

    recall = len(matched) / len(markers)
    mrr = (1.0 / min(ranks)) if ranks else 0.0
    return {
        "recall_at_k": round(recall, 4),
        "mrr": round(mrr, 4),
        "matched": matched,
        "missing": missing,
    }


ARTICLE_REFERENCE_RE = re.compile(r"第[〇零一二三四五六七八九十百千万两0-9]+条")
STRUCTURAL_ALIAS_RE = re.compile(
    r"^(?:[\u4e00-\u9fffA-Za-z0-9《》（）()·、]+)?第[〇零一二三四五六七八九十百千万两0-9]+条$"
)
LONG_MARKER_MIN_LENGTH = 80
LONG_MARKER_SHINGLE_SIZE = 6
LONG_MARKER_MIN_SHARED_SHINGLES = 8
LONG_MARKER_MIN_COVERAGE = 0.60


def first_marker_rank(ranked_chunks: list[dict[str, Any]], marker: dict[str, Any]) -> int | None:
    expected_title = marker.get("document_title") or ""
    raw_aliases = dedupe([str(marker.get("label") or ""), *[str(alias) for alias in marker.get("aliases", [])]])
    aliases = [
        (str(alias), normalize_marker_text(alias))
        for alias in raw_aliases
        if normalize_marker_text(alias)
    ]
    if not aliases:
        return None
    for index, chunk in enumerate(ranked_chunks, start=1):
        document_title = str(chunk.get("document_title") or "").strip().lower()
        if expected_title and document_title != expected_title:
            continue
        structural_haystack = normalize_marker_text(
            " ".join(
                str(part or "")
                for part in [
                    chunk.get("document_title"),
                    chunk.get("section_title"),
                    chunk.get("clause_full_name"),
                    chunk.get("article_number"),
                    chunk.get("heading_path"),
                ]
            )
        )
        content_haystack = normalize_marker_text(
            " ".join(
                str(part or "")
                for part in [
                    chunk.get("content"),
                ]
            )
        )
        if any(marker_alias_matches(raw_alias, alias, structural_haystack, content_haystack) for raw_alias, alias in aliases):
            return index
    return None


def marker_alias_matches(
    raw_alias: str,
    normalized_alias: str,
    structural_haystack: str,
    content_haystack: str,
) -> bool:
    if normalized_alias in structural_haystack:
        return True
    if is_structural_marker_alias(raw_alias):
        return False
    if normalized_alias in content_haystack:
        return True
    return long_marker_fuzzy_matches(normalized_alias, content_haystack)


def long_marker_fuzzy_matches(normalized_alias: str, content_haystack: str) -> bool:
    if len(normalized_alias) < LONG_MARKER_MIN_LENGTH:
        return False
    marker_shingles = text_shingles(normalized_alias, LONG_MARKER_SHINGLE_SIZE)
    if len(marker_shingles) < LONG_MARKER_MIN_SHARED_SHINGLES:
        return False
    content_shingles = text_shingles(content_haystack, LONG_MARKER_SHINGLE_SIZE)
    shared = marker_shingles.intersection(content_shingles)
    if len(shared) < LONG_MARKER_MIN_SHARED_SHINGLES:
        return False
    return len(shared) / len(marker_shingles) >= LONG_MARKER_MIN_COVERAGE


def text_shingles(value: str, size: int) -> set[str]:
    if len(value) < size:
        return set()
    return {value[index : index + size] for index in range(len(value) - size + 1)}


def is_structural_marker_alias(value: str) -> bool:
    text = "".join(str(value or "").split())
    if not ARTICLE_REFERENCE_RE.search(text):
        return False
    return bool(STRUCTURAL_ALIAS_RE.fullmatch(text))


def summarize_retrieved_chunks(ranked_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": index,
            "document_title": str(chunk.get("document_title") or ""),
            "chunk_index": chunk.get("chunk_index"),
            "section_title": chunk.get("section_title"),
            "clause_full_name": chunk.get("clause_full_name"),
            "article_number": chunk.get("article_number"),
            "chunk_type": chunk.get("chunk_type"),
            "preview": str(chunk.get("content") or "")[:240],
        }
        for index, chunk in enumerate(ranked_chunks, start=1)
    ]


def normalize_marker_text(value: str) -> str:
    return "".join(
        char
        for char in " ".join(str(value).casefold().split())
        if char not in "，。；：、,.!?！？;:()（）[]【】\"' "
    )


def dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip()
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        result.append(cleaned)
        seen.add(key)
    return result


def average_precision_at_k(ranked_titles: list[str], relevant_titles: set[str]) -> float:
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


def reciprocal_rank(ranked_titles: list[str], relevant_titles: set[str]) -> float:
    if not relevant_titles:
        return 1.0
    for index, title in enumerate(ranked_titles, start=1):
        if title in relevant_titles:
            return 1.0 / index
    return 0.0


def binary_ndcg(ranked_titles: list[str], relevant_titles: set[str]) -> float:
    if not relevant_titles:
        return 1.0
    dcg = sum((1.0 if title in relevant_titles else 0.0) / math.log2(index + 2) for index, title in enumerate(ranked_titles))
    ideal_hits = min(len(relevant_titles), len(ranked_titles))
    if ideal_hits == 0:
        return 0.0
    ideal_dcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def average(values) -> float:
    clean = [float(value or 0.0) for value in values]
    if not clean:
        return 0.0
    return round(sum(clean) / len(clean), 4)


def average_bool(values) -> float:
    clean = [bool(value) for value in values]
    if not clean:
        return 0.0
    return round(sum(1 for value in clean if value) / len(clean), 4)


def default_output_path(dataset_name: str) -> Path:
    safe_name = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in dataset_name)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return BACKEND_DIR / "data" / "eval_outputs" / f"{safe_name}-retrieval-{timestamp}.json"


def summary_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    latency = summary.get("latency_seconds") or {}
    return "\n".join(
        [
            f"dataset={report['dataset_name']} cases={summary['total_cases']} pass={summary['pass_count']} pass_rate={summary['pass_rate']}",
            f"recall@k={summary['recall_at_k_avg']} mrr={summary['mrr_avg']} ndcg@k={summary['ndcg_at_k_avg']} map@k={summary['map_at_k']}",
            f"evidence_recall@k={summary['evidence_recall_at_k_avg']} evidence_mrr={summary['evidence_mrr_avg']}",
            f"latency_seconds=p50:{latency.get('p50')} p90:{latency.get('p90')} p95:{latency.get('p95')} max:{latency.get('max')}",
            f"permission={summary['permission_isolation_pass_rate']} failures={summary['failure_mode_counts']}",
        ]
    )


if __name__ == "__main__":
    main()
