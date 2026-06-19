from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from sqlalchemy import select


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
BACKEND_DIR = ROOT_DIR / "backend"
for import_path in (SCRIPTS_DIR, BACKEND_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_retrieval_benchmark as benchmark

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.models.document import Document
from app.repositories.eval_repository import EvalRepository
from app.repositories.retrieval_repository import RetrievalCandidate
from app.repositories.user_repository import UserRepository
from app.schemas.search import SearchRequest
from app.services.auth.bootstrap import seed_mock_data
from app.services.eval.service import EvalService
from app.services.retrieval.reranker import RerankCandidate
from app.services.retrieval.service import RetrievalService


DEFAULT_ABLATIONS = (
    "full_local",
    "full_with_sweep",
    "full_preserve",
    "full_sweep_preserve",
    "no_rerank",
    "no_expansion",
    "lexical_only",
    "lexical_sweep",
    "structural_only",
    "lexical_structural",
)


@dataclass(frozen=True)
class Ablation:
    name: str
    use_lexical: bool = True
    use_indexed_sparse: bool = False
    use_structural: bool = True
    use_vector: bool = False
    use_expansion: bool = True
    use_sweep: bool = False
    use_document_first: bool = False
    use_neighbor: bool = False
    use_preservation: bool = False
    use_coverage: bool = True
    use_evidence_bridge: bool = False
    use_rerank: bool = True
    force_heuristic_rerank: bool = False


ABLATIONS: dict[str, Ablation] = {
    "full_local": Ablation("full_local"),
    "full_no_coverage": Ablation("full_no_coverage", use_coverage=False),
    "full_heuristic_rerank": Ablation("full_heuristic_rerank", force_heuristic_rerank=True),
    "full_evidence_bridge": Ablation("full_evidence_bridge", use_evidence_bridge=True),
    "full_with_sweep": Ablation("full_with_sweep", use_sweep=True),
    "full_document_neighbor": Ablation("full_document_neighbor", use_neighbor=True),
    "full_document_first": Ablation("full_document_first", use_document_first=True),
    "full_document_first_neighbor": Ablation("full_document_first_neighbor", use_document_first=True, use_neighbor=True),
    "full_document_first_evidence_bridge": Ablation(
        "full_document_first_evidence_bridge",
        use_document_first=True,
        use_evidence_bridge=True,
    ),
    "full_document_first_coverage": Ablation("full_document_first_coverage", use_document_first=True, use_coverage=True),
    "full_document_first_preserve": Ablation("full_document_first_preserve", use_document_first=True, use_preservation=True),
    "full_document_first_preserve_coverage": Ablation(
        "full_document_first_preserve_coverage",
        use_document_first=True,
        use_preservation=True,
        use_coverage=True,
    ),
    "full_preserve": Ablation("full_preserve", use_preservation=True),
    "full_sweep_preserve": Ablation("full_sweep_preserve", use_sweep=True, use_preservation=True),
    "full_indexed_sparse": Ablation("full_indexed_sparse", use_indexed_sparse=True),
    "full_indexed_sparse_preserve": Ablation("full_indexed_sparse_preserve", use_indexed_sparse=True, use_preservation=True),
    "no_rerank": Ablation("no_rerank", use_rerank=False),
    "no_expansion": Ablation("no_expansion", use_expansion=False),
    "lexical_only": Ablation("lexical_only", use_structural=False, use_expansion=False, use_rerank=False),
    "lexical_sweep": Ablation("lexical_sweep", use_structural=False, use_expansion=False, use_sweep=True, use_rerank=False),
    "lexical_document_first": Ablation(
        "lexical_document_first",
        use_structural=False,
        use_expansion=False,
        use_document_first=True,
        use_rerank=False,
    ),
    "lexical_preserve": Ablation("lexical_preserve", use_structural=False, use_expansion=False, use_preservation=True, use_rerank=False),
    "lexical_indexed_sparse": Ablation(
        "lexical_indexed_sparse",
        use_structural=False,
        use_indexed_sparse=True,
        use_expansion=False,
        use_rerank=False,
    ),
    "indexed_sparse_only": Ablation(
        "indexed_sparse_only",
        use_lexical=False,
        use_structural=False,
        use_indexed_sparse=True,
        use_expansion=False,
        use_rerank=False,
    ),
    "lexical_sweep_preserve": Ablation(
        "lexical_sweep_preserve",
        use_structural=False,
        use_expansion=False,
        use_sweep=True,
        use_preservation=True,
        use_rerank=False,
    ),
    "structural_only": Ablation("structural_only", use_lexical=False, use_expansion=False, use_rerank=False),
    "vector_only": Ablation("vector_only", use_lexical=False, use_structural=False, use_vector=True, use_expansion=False, use_rerank=False),
    "lexical_structural": Ablation("lexical_structural", use_expansion=False, use_rerank=False),
    "lexical_vector": Ablation("lexical_vector", use_structural=False, use_vector=True, use_expansion=False, use_rerank=False),
    "structural_vector": Ablation("structural_vector", use_lexical=False, use_vector=True, use_expansion=False, use_rerank=False),
    "hybrid_vector": Ablation("hybrid_vector", use_vector=True),
}


def main() -> None:
    args = build_parser().parse_args()
    if args.local_baseline:
        benchmark.apply_local_baseline_env(args.domain_profile)
    else:
        benchmark.apply_domain_profile_env(args.domain_profile)
    benchmark.apply_retrieval_ablation_overrides(
        cjk_python_fallback_mode=args.cjk_python_fallback_mode,
        cjk_python_scorer=args.cjk_python_scorer,
    )
    apply_in_document_expansion_overrides(args)
    selected_ablations = parse_ablations(args.ablations)
    report = run_diagnostics(
        dataset_name=args.dataset,
        top_k=args.top_k,
        limit=args.limit,
        case_names=args.case_name,
        case_name_contains=args.case_name_contains,
        document_title_prefix=args.document_title_prefix,
        manifest_scope=args.manifest_scope,
        ablations=selected_ablations,
        trace_top_n=args.trace_top_n,
        include_case_traces=not args.summary_only,
        rerank_result_limit=args.rerank_result_limit,
    )
    output_path = Path(args.output) if args.output else default_output_path(args.dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(summary_text(report))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run retrieval diagnostics and ablations without changing the ranking implementation. "
            "The report shows where expected documents/evidence enter or leave the pipeline."
        )
    )
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
        "--ablations",
        default=",".join(DEFAULT_ABLATIONS),
        help=f"Comma-separated ablations. Available: {', '.join(sorted(ABLATIONS))}",
    )
    parser.add_argument(
        "--trace-top-n",
        type=int,
        default=8,
        help="Number of ranked chunks to keep per stage in each case trace.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only write aggregate metrics. Useful for large slow ablation runs.",
    )
    parser.add_argument(
        "--local-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic embeddings and local rewrite/rerank defaults before ablation overrides.",
    )
    parser.add_argument(
        "--domain-profile",
        choices=["enterprise", "legal_benchmark"],
        default="enterprise",
    )
    parser.add_argument(
        "--cjk-python-fallback-mode",
        choices=["auto", "always", "off"],
        default=None,
    )
    parser.add_argument(
        "--cjk-python-scorer",
        choices=["weighted", "bm25"],
        default=None,
    )
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--in-document-expansion-seed-count", type=int, default=None)
    parser.add_argument("--in-document-expansion-per-document", type=int, default=None)
    parser.add_argument("--in-document-expansion-max-candidates", type=int, default=None)
    parser.add_argument("--in-document-expansion-adjacent-window", type=int, default=None)
    parser.add_argument("--in-document-expansion-score-weight", type=float, default=None)
    parser.add_argument("--rerank-max-candidates", type=int, default=None)
    parser.add_argument(
        "--rerank-result-limit",
        type=int,
        default=None,
        help="Diagnostic-only override for how many reranked candidates are passed to final selection.",
    )
    return parser


def apply_in_document_expansion_overrides(args: argparse.Namespace) -> None:
    overrides = {
        "RETRIEVAL_IN_DOCUMENT_EXPANSION_SEED_COUNT": args.in_document_expansion_seed_count,
        "RETRIEVAL_IN_DOCUMENT_EXPANSION_PER_DOCUMENT": args.in_document_expansion_per_document,
        "RETRIEVAL_IN_DOCUMENT_EXPANSION_MAX_CANDIDATES": args.in_document_expansion_max_candidates,
        "RETRIEVAL_IN_DOCUMENT_EXPANSION_ADJACENT_WINDOW": args.in_document_expansion_adjacent_window,
    }
    applied = False
    for env_name, value in overrides.items():
        if value is None:
            continue
        os.environ[env_name] = str(max(0, int(value)))
        applied = True
    if applied:
        from app.core.config import get_settings

        get_settings.cache_clear()
    if args.in_document_expansion_score_weight is not None:
        os.environ["RETRIEVAL_IN_DOCUMENT_EXPANSION_SCORE_WEIGHT"] = str(
            max(0.0, float(args.in_document_expansion_score_weight))
        )
        from app.core.config import get_settings

        get_settings.cache_clear()
    if args.rerank_max_candidates is not None:
        os.environ["RERANK_MAX_CANDIDATES"] = str(max(1, int(args.rerank_max_candidates)))
        from app.core.config import get_settings

        get_settings.cache_clear()


def parse_ablations(raw_value: str) -> list[Ablation]:
    names = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not names:
        raise SystemExit("At least one ablation must be selected.")
    unknown = [name for name in names if name not in ABLATIONS]
    if unknown:
        raise SystemExit(f"Unknown ablation(s): {', '.join(unknown)}. Available: {', '.join(sorted(ABLATIONS))}")
    return [ABLATIONS[name] for name in names]


def run_diagnostics(
    *,
    dataset_name: str,
    top_k: int,
    limit: int | None,
    case_names: list[str] | None,
    case_name_contains: str | None,
    document_title_prefix: str | None,
    manifest_scope: str | None,
    ablations: list[Ablation],
    trace_top_n: int,
    include_case_traces: bool,
    rerank_result_limit: int | None = None,
) -> dict[str, Any]:
    seed_mock_data()
    session = SessionLocal()
    try:
        eval_repository = EvalRepository(session)
        user_repository = UserRepository(session)
        scoped_document_ids, document_scope = resolve_scoped_document_ids(
            session=session,
            dataset_name=dataset_name,
            document_title_prefix=document_title_prefix,
            manifest_scope=manifest_scope,
        )
        cases = eval_repository.list_cases(dataset_name)
        cases = filter_cases(
            cases,
            case_names=case_names,
            case_name_contains=case_name_contains,
            limit=limit,
        )
        if not cases:
            raise SystemExit("No eval cases matched the requested filters.")

        ablation_reports = []
        for ablation in ablations:
            apply_ablation_env(ablation)
            service = RetrievalService(session)
            rows = []
            traces = []
            for case in cases:
                actor = user_repository.get_by_email(case.acting_user_email)
                if actor is None:
                    rows.append(benchmark.missing_actor_row(case))
                    continue
                annotations = EvalService._resolve_case_annotations(case)
                trace = trace_case(
                    service=service,
                    case=case,
                    actor=actor,
                    annotations=annotations,
                    top_k=top_k,
                    scoped_document_ids=scoped_document_ids,
                    ablation=ablation,
                    trace_top_n=trace_top_n,
                    rerank_result_limit=rerank_result_limit,
                )
                rows.append(trace["final_score"])
                if include_case_traces:
                    traces.append(trace)

            ablation_reports.append(
                {
                    "ablation": ablation.__dict__,
                    "summary": summarize_diagnostic_rows(rows, traces if include_case_traces else None),
                    "cases": traces if include_case_traces else [],
                    "failure_cases": [trace for trace in traces if not trace["final_score"]["passed"]] if include_case_traces else [],
                }
            )

        return {
            "dataset_name": dataset_name,
            "top_k": top_k,
            "generated_at": datetime.now(UTC).isoformat(),
            "document_scope": document_scope,
            "case_count": len(cases),
            "case_filters": {
                "case_names": case_names or [],
                "case_name_contains": case_name_contains,
                "limit": limit,
                "rerank_result_limit": rerank_result_limit,
            },
            "diagnostic_version": "v2",
            "notes": [
                "This report is diagnostic. It does not add domain keywords or alter production ranking.",
                "document_sweep is a configurable source-pool channel that scans chunks inside top retrieved documents.",
                "Vector ablations can be much slower and are only run when explicitly selected.",
            ],
            "ablations": ablation_reports,
        }
    finally:
        session.close()


def filter_cases(
    cases: Iterable[Any],
    *,
    case_names: list[str] | None,
    case_name_contains: str | None,
    limit: int | None,
) -> list[Any]:
    selected = list(cases)
    if case_names:
        allowed = set(case_names)
        selected = [case for case in selected if case.case_name in allowed]
    if case_name_contains:
        selected = [case for case in selected if case_name_contains in case.case_name]
    if limit is not None:
        selected = selected[:limit]
    return selected


def apply_ablation_env(ablation: Ablation) -> None:
    os.environ["RETRIEVAL_LEXICAL_ENABLED"] = "true" if ablation.use_lexical else "false"
    os.environ["RETRIEVAL_VECTOR_ENABLED"] = "true" if ablation.use_vector else "false"
    os.environ["RETRIEVAL_INDEXED_SPARSE_ENABLED"] = "true" if ablation.use_indexed_sparse else "false"
    os.environ["RETRIEVAL_STRUCTURAL_ENABLED"] = "true" if ablation.use_structural else "false"
    os.environ["RETRIEVAL_IN_DOCUMENT_EXPANSION_ENABLED"] = "true" if ablation.use_expansion else "false"
    os.environ["RETRIEVAL_DOCUMENT_EVIDENCE_SWEEP_ENABLED"] = "true" if ablation.use_sweep else "false"
    os.environ["RETRIEVAL_DOCUMENT_FIRST_EVIDENCE_ENABLED"] = "true" if ablation.use_document_first else "false"
    os.environ["RETRIEVAL_DOCUMENT_NEIGHBOR_CONTEXT_ENABLED"] = "true" if ablation.use_neighbor else "false"
    os.environ["RETRIEVAL_EVIDENCE_PRESERVATION_ENABLED"] = "true" if ablation.use_preservation else "false"
    os.environ["RETRIEVAL_FINAL_COVERAGE_ENABLED"] = "true" if ablation.use_coverage else "false"
    os.environ["RETRIEVAL_EVIDENCE_QUERY_BRIDGE_ENABLED"] = "true" if ablation.use_evidence_bridge else "false"
    os.environ["RETRIEVAL_HEURISTIC_RERANK_ENABLED"] = "true" if ablation.force_heuristic_rerank else "false"
    os.environ["RERANK_PROVIDER"] = "heuristic"
    get_settings.cache_clear()


def trace_case(
    *,
    service: RetrievalService,
    case,
    actor,
    annotations: dict[str, Any],
    top_k: int,
    scoped_document_ids: list | None,
    ablation: Ablation,
    trace_top_n: int,
    rerank_result_limit: int | None,
) -> dict[str, Any]:
    expected_titles = benchmark.normalize_titles(annotations["expected_retrieval_titles"])
    forbidden_titles = benchmark.normalize_titles(case.forbidden_document_titles)
    evidence_markers = benchmark.normalize_evidence_markers(annotations.get("expected_evidence_markers"))
    accessible_document_ids = service.permission_builder.resolve_accessible_document_ids(service.session, actor, require_manage=False)
    if scoped_document_ids is not None:
        scoped_set = set(scoped_document_ids)
        accessible_document_ids = [item for item in accessible_document_ids if item in scoped_set]

    if not accessible_document_ids:
        final_score = benchmark.score_case(
            case=case,
            annotations=annotations,
            ranked_titles=[],
            ranked_chunks=[],
            expected_titles=expected_titles,
            forbidden_titles=forbidden_titles,
            retrieval_debug={"accessible_document_count": 0, "ablation": ablation.name},
        )
        return {
            "case_name": case.case_name,
            "question": case.question,
            "expected_outcome": annotations.get("expected_outcome"),
            "stage_loss": "no_accessible_documents",
            "query_plan": {},
            "stages": {},
            "latency_ms": {},
            "final_score": final_score,
        }

    query_plan_started = perf_counter()
    query_plan = service.query_optimizer.build(case.question)
    probe_applied = False
    if len(query_plan.candidates) > 1:
        probe_applied = service._select_best_query_plan(  # noqa: SLF001 - diagnostic script intentionally replays the service path.
            query_plan,
            accessible_document_ids=accessible_document_ids,
            target_document_title=None,
        )
    query_plan_latency_ms = elapsed_ms(query_plan_started)
    candidate_pool = service._candidate_pool_size(top_k)  # noqa: SLF001

    stage_data: dict[str, list[dict[str, Any]]] = {}
    stage_scores: dict[str, dict[str, Any]] = {}
    latency: dict[str, int] = {"query_plan": query_plan_latency_ms}

    lexical_started = perf_counter()
    lexical_hits = (
        service._collect_lexical_hits(query_plan.lexical_queries, accessible_document_ids, candidate_pool)  # noqa: SLF001
        if ablation.use_lexical
        else []
    )
    latency["lexical"] = elapsed_ms(lexical_started)
    add_candidate_stage(stage_data, stage_scores, "lexical", lexical_hits, expected_titles, evidence_markers, trace_top_n)

    indexed_sparse_started = perf_counter()
    indexed_sparse_hits = (
        service._collect_indexed_sparse_hits(query_plan.lexical_queries, accessible_document_ids, candidate_pool)  # noqa: SLF001
        if ablation.use_indexed_sparse
        else []
    )
    latency["indexed_sparse"] = elapsed_ms(indexed_sparse_started)
    add_candidate_stage(stage_data, stage_scores, "indexed_sparse", indexed_sparse_hits, expected_titles, evidence_markers, trace_top_n)

    structural_started = perf_counter()
    structural_hits = (
        service._collect_structural_hits(query_plan.lexical_queries, accessible_document_ids, candidate_pool)  # noqa: SLF001
        if ablation.use_structural
        else []
    )
    latency["structural"] = elapsed_ms(structural_started)
    add_candidate_stage(stage_data, stage_scores, "structural", structural_hits, expected_titles, evidence_markers, trace_top_n)

    vector_embedding_latency_ms = 0
    vector_retrieval_started = perf_counter()
    vector_hits: list[RetrievalCandidate] = []
    if ablation.use_vector:
        embedding_started = perf_counter()
        query_embedding = service.embedding_provider.embed_texts([query_plan.retrieval_query])[0]
        vector_embedding_latency_ms = elapsed_ms(embedding_started)
        vector_hits = service.retrieval_repository.search_vector(query_embedding, accessible_document_ids, candidate_pool)
    latency["vector_embedding"] = vector_embedding_latency_ms
    latency["vector"] = elapsed_ms(vector_retrieval_started)
    add_candidate_stage(stage_data, stage_scores, "vector", vector_hits, expected_titles, evidence_markers, trace_top_n)

    fusion_started = perf_counter()
    fused = service._fuse_hits(  # noqa: SLF001
        lexical_hits,
        vector_hits,
        structural_hits=structural_hits,
        indexed_sparse_hits=indexed_sparse_hits,
    )
    latency["fusion"] = elapsed_ms(fusion_started)
    fused_candidates = sorted_rerank_candidates(fused.values())
    add_rerank_stage(stage_data, stage_scores, "fused", fused_candidates, expected_titles, evidence_markers, trace_top_n)

    expansion_hits: list[RetrievalCandidate] = []
    if ablation.use_expansion:
        expansion_started = perf_counter()
        expansion_hits = service._collect_in_document_expansion(query_plan.retrieval_query, fused.values())  # noqa: SLF001
        latency["expansion"] = elapsed_ms(expansion_started)
        if expansion_hits:
            refusion_started = perf_counter()
            fused = service._fuse_hits(  # noqa: SLF001
                lexical_hits,
                vector_hits,
                structural_hits=structural_hits,
                indexed_sparse_hits=indexed_sparse_hits,
                expansion_hits=expansion_hits,
            )
            latency["fusion"] += elapsed_ms(refusion_started)
    else:
        latency["expansion"] = 0
    add_candidate_stage(stage_data, stage_scores, "expansion", expansion_hits, expected_titles, evidence_markers, trace_top_n)

    document_sweep_hits: list[RetrievalCandidate] = []
    if ablation.use_sweep:
        sweep_started = perf_counter()
        document_sweep_hits = service._collect_document_evidence_sweep(query_plan.retrieval_query, fused.values())  # noqa: SLF001
        latency["document_sweep"] = elapsed_ms(sweep_started)
        if document_sweep_hits:
            refusion_started = perf_counter()
            fused = service._fuse_hits(  # noqa: SLF001
                lexical_hits,
                vector_hits,
                structural_hits=structural_hits,
                indexed_sparse_hits=indexed_sparse_hits,
                expansion_hits=expansion_hits,
                document_sweep_hits=document_sweep_hits,
            )
            latency["fusion"] += elapsed_ms(refusion_started)
    else:
        latency["document_sweep"] = 0
    add_candidate_stage(stage_data, stage_scores, "document_sweep", document_sweep_hits, expected_titles, evidence_markers, trace_top_n)

    document_first_hits: list[RetrievalCandidate] = []
    if ablation.use_document_first:
        document_first_started = perf_counter()
        document_first_hits = service._collect_document_first_evidence_hits(query_plan.retrieval_query, fused.values())  # noqa: SLF001
        latency["document_first_evidence"] = elapsed_ms(document_first_started)
        if document_first_hits:
            refusion_started = perf_counter()
            fused = service._fuse_hits(  # noqa: SLF001
                lexical_hits,
                vector_hits,
                structural_hits=structural_hits,
                indexed_sparse_hits=indexed_sparse_hits,
                expansion_hits=expansion_hits,
                document_sweep_hits=document_sweep_hits,
                document_first_hits=document_first_hits,
            )
            latency["fusion"] += elapsed_ms(refusion_started)
    else:
        latency["document_first_evidence"] = 0
    add_candidate_stage(stage_data, stage_scores, "document_first_evidence", document_first_hits, expected_titles, evidence_markers, trace_top_n)

    neighbor_hits: list[RetrievalCandidate] = []
    if ablation.use_neighbor:
        neighbor_started = perf_counter()
        neighbor_hits = service._collect_document_neighbor_context_hits(fused.values())  # noqa: SLF001
        latency["document_neighbor_context"] = elapsed_ms(neighbor_started)
        if neighbor_hits:
            refusion_started = perf_counter()
            fused = service._fuse_hits(  # noqa: SLF001
                lexical_hits,
                vector_hits,
                structural_hits=structural_hits,
                indexed_sparse_hits=indexed_sparse_hits,
                expansion_hits=expansion_hits,
                document_sweep_hits=document_sweep_hits,
                document_first_hits=document_first_hits,
                neighbor_hits=neighbor_hits,
            )
            latency["fusion"] += elapsed_ms(refusion_started)
    else:
        latency["document_neighbor_context"] = 0
    add_candidate_stage(stage_data, stage_scores, "document_neighbor_context", neighbor_hits, expected_titles, evidence_markers, trace_top_n)

    pre_rerank_candidates = sorted_rerank_candidates(fused.values())
    add_rerank_stage(stage_data, stage_scores, "pre_rerank", pre_rerank_candidates, expected_titles, evidence_markers, trace_top_n)

    rerank_started = perf_counter()
    rerank_limit = len(pre_rerank_candidates)
    if ablation.use_rerank and service._should_run_reranker():  # noqa: SLF001
        rerank_query = service._build_rerank_query(case.question, query_plan.retrieval_query)  # noqa: SLF001
        rerank_limit = service._rerank_result_limit(top_k, candidate_pool, len(fused))  # noqa: SLF001
        if rerank_result_limit is not None:
            rerank_limit = min(len(fused), max(top_k, int(rerank_result_limit)))
        reranked_result = service.reranker.rerank(
            rerank_query,
            list(fused.values()),
            rerank_limit,
        )
        reranked_candidates = reranked_result.candidates
        rerank_strategy = reranked_result.strategy
    else:
        reranked_candidates = service._rank_without_rerank(fused.values())  # noqa: SLF001
        rerank_strategy = "disabled-by-diagnostic-ablation" if not ablation.use_rerank else "disabled-local-heuristic"
    latency["rerank"] = elapsed_ms(rerank_started)
    add_rerank_stage(stage_data, stage_scores, "reranked", reranked_candidates, expected_titles, evidence_markers, trace_top_n)

    preservation_candidates = service._collect_evidence_preservation_candidates(pre_rerank_candidates)  # noqa: SLF001
    add_rerank_stage(
        stage_data,
        stage_scores,
        "evidence_preservation_pool",
        preservation_candidates,
        expected_titles,
        evidence_markers,
        trace_top_n,
    )

    base_final_candidates = service._select_final_candidates(reranked_candidates, top_k)  # noqa: SLF001
    base_final_ids = {candidate.candidate.chunk_id for candidate in base_final_candidates}
    preservation_ids = {candidate.candidate.chunk_id for candidate in preservation_candidates}
    coverage_candidates = service._collect_final_coverage_candidates(  # noqa: SLF001
        case.question,
        reranked_candidates,
        base_final_candidates,
        top_k,
    )
    coverage_ids = {candidate.candidate.chunk_id for candidate in coverage_candidates}
    final_candidates = service._select_final_candidates(  # noqa: SLF001
        reranked_candidates,
        top_k,
        preservation_candidates=preservation_candidates,
        coverage_candidates=coverage_candidates,
    )
    add_rerank_stage(stage_data, stage_scores, "final", final_candidates, expected_titles, evidence_markers, trace_top_n)

    final_ranked_chunks = ranked_chunks_from_rerank(final_candidates)
    final_score = benchmark.score_case(
        case=case,
        annotations=annotations,
        ranked_titles=benchmark.ordered_titles(item["document_title"] for item in final_ranked_chunks),
        ranked_chunks=final_ranked_chunks,
        expected_titles=expected_titles,
        forbidden_titles=forbidden_titles,
        retrieval_debug={
            "ablation": ablation.name,
            "accessible_document_count": len(accessible_document_ids),
            "candidate_pool": candidate_pool,
            "lexical_candidate_count": len(lexical_hits),
            "indexed_sparse_candidate_count": len(indexed_sparse_hits),
            "structural_candidate_count": len(structural_hits),
            "vector_candidate_count": len(vector_hits),
            "expansion_candidate_count": len(expansion_hits),
            "document_evidence_sweep_candidate_count": len(document_sweep_hits),
            "document_first_evidence_candidate_count": len(document_first_hits),
            "document_neighbor_context_candidate_count": len(neighbor_hits),
            "evidence_preservation_candidate_count": len(preservation_candidates),
            "evidence_preservation_selected_count": sum(
                1
                for item in final_candidates
                if item.candidate.chunk_id in preservation_ids and item.candidate.chunk_id not in base_final_ids
            ),
            "final_coverage_candidate_count": len(coverage_candidates),
            "final_coverage_selected_count": sum(
                1
                for item in final_candidates
                if item.candidate.chunk_id in coverage_ids and item.candidate.chunk_id not in base_final_ids
            ),
            "pre_rerank_count": len(fused),
            "rerank_result_limit": rerank_limit,
            "post_rerank_count": len(final_candidates),
            "rerank_strategy": rerank_strategy,
            "fusion_strategy": service._fusion_strategy_name(),  # noqa: SLF001
            "retrieval_query": query_plan.retrieval_query,
            "lexical_queries": query_plan.lexical_queries,
            "query_rewrite_strategies": query_plan.applied_strategies,
            "query_plan_selected": query_plan.selected_candidate.label,
            "query_plan_selection_reason": query_plan.selected_candidate_reason,
            "query_plan_probe_applied": probe_applied,
            "latency_ms": latency,
        },
    )

    stage_loss = classify_stage_loss(stage_scores, final_score)
    return {
        "case_name": case.case_name,
        "acting_user_email": case.acting_user_email,
        "question": case.question,
        "expected_outcome": annotations.get("expected_outcome"),
        "stage_loss": stage_loss,
        "query_plan": {
            "retrieval_query": query_plan.retrieval_query,
            "lexical_queries": query_plan.lexical_queries,
            "applied_strategies": query_plan.applied_strategies,
            "selected": query_plan.selected_candidate.label,
            "selection_reason": query_plan.selected_candidate_reason,
            "probe_applied": probe_applied,
            "candidate_count": query_plan.candidate_count,
        },
        "stage_scores": stage_scores,
        "stages": stage_data,
        "latency_ms": latency,
        "final_score": final_score,
    }


def add_candidate_stage(
    stage_data: dict[str, list[dict[str, Any]]],
    stage_scores: dict[str, dict[str, Any]],
    name: str,
    candidates: Iterable[RetrievalCandidate],
    expected_titles: set[str],
    evidence_markers: list[dict[str, Any]],
    trace_top_n: int,
) -> None:
    ranked_chunks = ranked_chunks_from_candidates(list(candidates))
    stage_scores[name] = score_stage(ranked_chunks, expected_titles, evidence_markers)
    stage_data[name] = benchmark.summarize_retrieved_chunks(ranked_chunks[:trace_top_n])


def add_rerank_stage(
    stage_data: dict[str, list[dict[str, Any]]],
    stage_scores: dict[str, dict[str, Any]],
    name: str,
    candidates: Iterable[RerankCandidate],
    expected_titles: set[str],
    evidence_markers: list[dict[str, Any]],
    trace_top_n: int,
) -> None:
    ranked_chunks = ranked_chunks_from_rerank(list(candidates))
    stage_scores[name] = score_stage(ranked_chunks, expected_titles, evidence_markers)
    stage_data[name] = summarize_rerank_candidates(list(candidates)[:trace_top_n])


def score_stage(
    ranked_chunks: list[dict[str, Any]],
    expected_titles: set[str],
    evidence_markers: list[dict[str, Any]],
) -> dict[str, Any]:
    ranked_titles = benchmark.ordered_titles(item["document_title"] for item in ranked_chunks)
    if expected_titles:
        matched_titles = sorted(expected_titles.intersection(set(ranked_titles)))
        title_recall = len(matched_titles) / len(expected_titles)
        title_mrr = benchmark.reciprocal_rank(ranked_titles, expected_titles)
    else:
        matched_titles = []
        title_recall = 1.0
        title_mrr = 1.0
    evidence_score = benchmark.score_evidence_markers(ranked_chunks, evidence_markers)
    return {
        "candidate_count": len(ranked_chunks),
        "title_recall": round(title_recall, 4),
        "title_mrr": round(title_mrr, 4),
        "matched_expected_titles": matched_titles,
        "evidence_recall": evidence_score["recall_at_k"],
        "evidence_mrr": evidence_score["mrr"],
        "matched_evidence_markers": evidence_score["matched"],
        "missing_evidence_markers": evidence_score["missing"],
    }


def classify_stage_loss(stage_scores: dict[str, dict[str, Any]], final_score: dict[str, Any]) -> str:
    if final_score["passed"]:
        return "passed"
    if final_score.get("expected_outcome") == "refuse":
        return final_score.get("failure_mode", "permission_failure")

    base_source_pool = merge_stage_scores(stage_scores, ["lexical", "indexed_sparse", "structural", "vector"])
    source_pool = merge_stage_scores(
        stage_scores,
        [
            "lexical",
            "indexed_sparse",
            "structural",
            "vector",
            "document_sweep",
            "document_first_evidence",
            "document_neighbor_context",
        ],
    )
    if source_pool["title_recall"] <= 0:
        return "source_pool_title_missing"
    if source_pool["evidence_recall"] < 1.0:
        return "source_pool_evidence_missing"
    if stage_scores.get("pre_rerank", {}).get("evidence_recall", 0.0) < 1.0:
        if base_source_pool["evidence_recall"] >= 1.0 and stage_scores.get("fused", {}).get("evidence_recall", 0.0) < 1.0:
            return "fusion_evidence_dropped"
        return "expansion_or_refusion_evidence_dropped"
    if base_source_pool["evidence_recall"] >= 1.0 and stage_scores.get("fused", {}).get("evidence_recall", 0.0) < 1.0:
        return "fusion_evidence_dropped"
    if stage_scores.get("reranked", {}).get("evidence_recall", 0.0) < 1.0:
        if stage_scores.get("evidence_preservation_pool", {}).get("evidence_recall", 0.0) >= 1.0:
            return "evidence_preservation_evidence_dropped"
        return "rerank_evidence_dropped"
    if stage_scores.get("final", {}).get("evidence_recall", 0.0) < 1.0:
        return "final_selection_evidence_dropped"
    return final_score.get("failure_mode", "final_scoring_failure")


def merge_stage_scores(stage_scores: dict[str, dict[str, Any]], names: list[str]) -> dict[str, Any]:
    matched_titles = set()
    matched_evidence = set()
    all_missing_evidence = set()
    candidate_count = 0
    title_recall = 0.0
    for name in names:
        score = stage_scores.get(name) or {}
        candidate_count += int(score.get("candidate_count") or 0)
        title_recall = max(title_recall, float(score.get("title_recall") or 0.0))
        matched_titles.update(score.get("matched_expected_titles") or [])
        matched_evidence.update(score.get("matched_evidence_markers") or [])
        all_missing_evidence.update(score.get("missing_evidence_markers") or [])
    missing_evidence = all_missing_evidence.difference(matched_evidence)
    evidence_total = len(matched_evidence) + len(missing_evidence)
    evidence_recall = (len(matched_evidence) / evidence_total) if evidence_total else 1.0
    return {
        "candidate_count": candidate_count,
        "title_recall": round(title_recall, 4),
        "evidence_recall": round(evidence_recall, 4),
        "matched_expected_titles": sorted(matched_titles),
        "matched_evidence_markers": sorted(matched_evidence),
        "missing_evidence_markers": sorted(missing_evidence),
    }


def summarize_diagnostic_rows(rows: list[dict[str, Any]], traces: list[dict[str, Any]] | None) -> dict[str, Any]:
    summary = benchmark.summarize(rows)
    if not traces:
        return summary

    stage_names = [
        "lexical",
        "indexed_sparse",
        "structural",
        "vector",
        "fused",
        "expansion",
        "document_sweep",
        "document_first_evidence",
        "document_neighbor_context",
        "pre_rerank",
        "reranked",
        "evidence_preservation_pool",
        "final",
    ]
    stage_metrics: dict[str, dict[str, Any]] = {}
    for stage_name in stage_names:
        scores = [trace["stage_scores"].get(stage_name, {}) for trace in traces]
        stage_metrics[stage_name] = {
            "avg_candidate_count": benchmark.average(score.get("candidate_count", 0) for score in scores),
            "title_recall_avg": benchmark.average(score.get("title_recall", 0) for score in scores),
            "title_mrr_avg": benchmark.average(score.get("title_mrr", 0) for score in scores),
            "evidence_recall_avg": benchmark.average(score.get("evidence_recall", 0) for score in scores),
            "evidence_mrr_avg": benchmark.average(score.get("evidence_mrr", 0) for score in scores),
        }

    latency_by_name: dict[str, list[int]] = defaultdict(list)
    for trace in traces:
        for key, value in trace.get("latency_ms", {}).items():
            latency_by_name[key].append(int(value or 0))
    latency_summary = {
        key: {
            "avg_ms": benchmark.average(values),
            "max_ms": max(values) if values else 0,
        }
        for key, values in sorted(latency_by_name.items())
    }

    summary.update(
        {
            "stage_metrics": stage_metrics,
            "stage_loss_counts": dict(Counter(trace["stage_loss"] for trace in traces)),
            "latency_summary": latency_summary,
        }
    )
    return summary


def ranked_chunks_from_candidates(candidates: list[RetrievalCandidate]) -> list[dict[str, Any]]:
    return [
        {
            "document_title": item.document_title,
            "chunk_index": item.chunk_index,
            "section_title": item.section_title,
            "clause_full_name": item.clause_full_name,
            "article_number": item.article_number,
            "chunk_type": item.chunk_type,
            "content": item.content,
        }
        for item in candidates
    ]


def ranked_chunks_from_rerank(candidates: list[RerankCandidate]) -> list[dict[str, Any]]:
    return ranked_chunks_from_candidates([item.candidate for item in candidates])


def sorted_rerank_candidates(candidates: Iterable[RerankCandidate]) -> list[RerankCandidate]:
    return sorted(
        candidates,
        key=lambda item: (
            item.fused_score,
            item.lexical_raw,
            item.vector_raw,
            -item.candidate.chunk_index,
        ),
        reverse=True,
    )


def summarize_rerank_candidates(candidates: list[RerankCandidate]) -> list[dict[str, Any]]:
    rows = []
    for rank, item in enumerate(candidates, start=1):
        candidate = item.candidate
        rows.append(
            {
                "rank": rank,
                "document_title": candidate.document_title,
                "chunk_index": candidate.chunk_index,
                "section_title": candidate.section_title,
                "clause_full_name": candidate.clause_full_name,
                "article_number": candidate.article_number,
                "chunk_type": candidate.chunk_type,
                "sources": sorted(item.sources),
                "scores": {
                    "lexical_raw": round(item.lexical_raw, 6),
                    "vector_raw": round(item.vector_raw, 6),
                    "fused": round(item.fused_score, 6),
                    "rerank": round(item.rerank_score, 6),
                },
                "preview": candidate.content[:240],
            }
        )
    return rows


def resolve_scoped_document_ids(
    *,
    session,
    dataset_name: str,
    document_title_prefix: str | None,
    manifest_scope: str | None = None,
) -> tuple[list | None, dict[str, Any]]:
    return benchmark.resolve_scoped_document_ids(
        session=session,
        dataset_name=dataset_name,
        document_title_prefix=document_title_prefix,
        manifest_scope=manifest_scope,
    )


def _legacy_resolve_scoped_document_ids(
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
    return ids, {"mode": "title_prefix", "title_prefix": prefix, "document_count": len(ids)}


def elapsed_ms(started: float) -> int:
    return int((perf_counter() - started) * 1000)


def default_output_path(dataset_name: str) -> Path:
    safe_name = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in dataset_name)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return BACKEND_DIR / "data" / "eval_outputs" / f"{safe_name}-retrieval-diagnostics-{timestamp}.json"


def summary_text(report: dict[str, Any]) -> str:
    lines = [f"dataset={report['dataset_name']} cases={report['case_count']}"]
    for item in report["ablations"]:
        summary = item["summary"]
        lines.append(
            " ".join(
                [
                    f"ablation={item['ablation']['name']}",
                    f"pass_rate={summary['pass_rate']}",
                    f"recall@k={summary['recall_at_k_avg']}",
                    f"evidence_recall@k={summary['evidence_recall_at_k_avg']}",
                    f"losses={summary.get('stage_loss_counts', summary.get('failure_mode_counts', {}))}",
                ]
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
