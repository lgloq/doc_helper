from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import UUID

from sqlalchemy import select


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
BACKEND_DIR = ROOT_DIR / "backend"
for import_path in (SCRIPTS_DIR, BACKEND_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_retrieval_benchmark as benchmark

from app.db.session import SessionLocal
from app.models.chunk import Chunk
from app.models.document import Document
from app.repositories.eval_repository import EvalRepository
from app.repositories.retrieval_repository import RetrievalCandidate
from app.repositories.user_repository import UserRepository
from app.services.ingestion.search_index import tokenize_search_text
from app.services.auth.bootstrap import seed_mock_data
from app.services.eval.service import EvalService
from app.services.retrieval.reranker import RerankCandidate, _select_rerank_candidates
from app.services.retrieval.service import RetrievalService


SOURCE_STAGES = (
    "lexical",
    "structural",
    "indexed_sparse",
    "vector",
    "document_sweep",
    "document_first_evidence",
    "document_neighbor_context",
)
DOCUMENT_LOCAL_DISTANCE_WINDOWS = (2, 5, 10, 20, 50)
LONG_MARKER_MIN_LENGTH = 80
LONG_MARKER_SHINGLE_SIZE = 6
LONG_MARKER_MIN_SHARED_SHINGLES = 8
LONG_MARKER_MIN_COVERAGE = 0.60
PIPELINE_STAGES = (
    "expected_document_chunks",
    "lexical",
    "structural",
    "indexed_sparse",
    "vector",
    "fused",
    "expansion",
    "document_sweep",
    "document_first_evidence",
    "document_neighbor_context",
    "pre_rerank",
    "semantic_rerank_pool",
    "reranked_full",
    "reranked",
    "final",
)


@dataclass(frozen=True)
class StageItem:
    chunk_id: UUID
    document_id: UUID
    document_title: str
    chunk_index: int
    content: str
    section_title: str | None = None
    clause_full_name: str | None = None
    article_number: str | None = None
    chunk_type: str | None = None
    heading_path: str | None = None
    structural_search_text: str | None = None
    lexical_search_text: str | None = None
    sources: tuple[str, ...] = ()
    fused_score: float | None = None
    lexical_raw: float | None = None
    lexical_norm: float | None = None
    vector_raw: float | None = None
    vector_norm: float | None = None
    rerank_score: float | None = None


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
    apply_audit_overrides(args)
    apply_in_document_expansion_overrides(args)

    diagnostics = load_diagnostics(args.diagnostics_report, args.ablation) if args.diagnostics_report else {}
    selected_case_names = select_case_names(diagnostics, args)
    report = audit_cases(args, diagnostics, selected_case_names)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown_output:
        markdown_path = Path(args.markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {output_path}")
    if args.markdown_output:
        print(f"Wrote {args.markdown_output}")
    print(summary_text(report))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay retrieval for benchmark cases and locate expected evidence markers across each pipeline stage. "
            "This is a diagnosis tool: it does not change ranking behavior or tune parameters."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--diagnostics-report", help="Existing run_retrieval_diagnostics.py JSON report.")
    parser.add_argument("--ablation", default="full_local", help="Ablation name inside the diagnostics report.")
    parser.add_argument("--stage-loss", action="append", help="Only audit cases with this diagnostics stage_loss.")
    parser.add_argument("--case-name", action="append", help="Audit one explicit case name. Can be repeated.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many selected cases before applying --limit.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--trace-markers", type=int, default=5)
    parser.add_argument(
        "--document-title-prefix",
        default=None,
        help="Restrict retrieval to documents whose title starts with this prefix. Defaults to '<dataset>:'.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output")
    parser.add_argument("--local-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--domain-profile", choices=["enterprise", "legal_benchmark"], default="enterprise")
    parser.add_argument("--cjk-python-fallback-mode", choices=["auto", "always", "off"], default=None)
    parser.add_argument("--cjk-python-scorer", choices=["weighted", "bm25"], default=None)
    parser.add_argument("--include-vector", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-indexed-sparse", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-document-sweep", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-document-first", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-document-neighbor", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-preservation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-final-coverage", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-evidence-bridge", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force-heuristic-rerank", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument(
        "--include-full-rerank",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also record the rerank order before the service top-k rerank truncation. Diagnostic only.",
    )
    parser.add_argument(
        "--match-mode",
        choices=["strict", "lenient"],
        default="strict",
        help=(
            "strict requires clause/article aliases to match structural fields. "
            "lenient reproduces the benchmark substring behavior."
        ),
    )
    return parser


def apply_audit_overrides(args: argparse.Namespace) -> None:
    os.environ["RETRIEVAL_VECTOR_ENABLED"] = "true" if args.include_vector else "false"
    os.environ["RETRIEVAL_INDEXED_SPARSE_ENABLED"] = "true" if args.include_indexed_sparse else "false"
    os.environ["RETRIEVAL_DOCUMENT_EVIDENCE_SWEEP_ENABLED"] = "true" if args.include_document_sweep else "false"
    os.environ["RETRIEVAL_DOCUMENT_FIRST_EVIDENCE_ENABLED"] = "true" if args.include_document_first else "false"
    os.environ["RETRIEVAL_DOCUMENT_NEIGHBOR_CONTEXT_ENABLED"] = "true" if args.include_document_neighbor else "false"
    os.environ["RETRIEVAL_EVIDENCE_PRESERVATION_ENABLED"] = "true" if args.include_preservation else "false"
    os.environ["RETRIEVAL_FINAL_COVERAGE_ENABLED"] = "true" if args.include_final_coverage else "false"
    os.environ["RETRIEVAL_EVIDENCE_QUERY_BRIDGE_ENABLED"] = "true" if args.include_evidence_bridge else "false"
    os.environ["RETRIEVAL_HEURISTIC_RERANK_ENABLED"] = "true" if args.force_heuristic_rerank else "false"


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


def load_diagnostics(path: str, ablation_name: str) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    ablation = next((item for item in report.get("ablations", []) if item.get("ablation", {}).get("name") == ablation_name), None)
    if ablation is None:
        names = [item.get("ablation", {}).get("name") for item in report.get("ablations", [])]
        raise SystemExit(f"Ablation '{ablation_name}' not found in diagnostics report. Available: {names}")
    return {
        "path": path,
        "dataset_name": report.get("dataset_name"),
        "top_k": report.get("top_k"),
        "ablation": ablation.get("ablation", {}),
        "summary": ablation.get("summary", {}),
        "cases": {case["case_name"]: case for case in ablation.get("cases", [])},
    }


def select_case_names(diagnostics: dict[str, Any], args: argparse.Namespace) -> list[str] | None:
    if args.case_name:
        names = list(dict.fromkeys(args.case_name))
    elif diagnostics:
        allowed_losses = set(args.stage_loss or [])
        cases = list(diagnostics["cases"].values())
        if allowed_losses:
            cases = [case for case in cases if case.get("stage_loss") in allowed_losses]
        names = [case["case_name"] for case in cases]
    else:
        names = None
    if names is not None:
        offset = max(0, int(args.offset or 0))
        if offset:
            names = names[offset:]
        if args.limit is not None:
            names = names[: args.limit]
    return names


def audit_cases(args: argparse.Namespace, diagnostics: dict[str, Any], selected_case_names: list[str] | None) -> dict[str, Any]:
    seed_mock_data()
    session = SessionLocal()
    try:
        eval_repository = EvalRepository(session)
        user_repository = UserRepository(session)
        service = RetrievalService(session)
        scoped_document_ids = resolve_scoped_document_ids(session, args.dataset, args.document_title_prefix)
        cases = eval_repository.list_cases(args.dataset)
        if selected_case_names is not None:
            selected = set(selected_case_names)
            cases = [case for case in cases if case.case_name in selected]
        if selected_case_names is None:
            offset = max(0, int(args.offset or 0))
            if offset:
                cases = cases[offset:]
            if args.limit is not None:
                cases = cases[: args.limit]
        rows = []
        for case in cases:
            actor = user_repository.get_by_email(case.acting_user_email)
            if actor is None:
                continue
            annotations = EvalService._resolve_case_annotations(case)
            rows.append(
                audit_case(
                    service=service,
                    session=session,
                    case=case,
                    actor=actor,
                    annotations=annotations,
                    scoped_document_ids=scoped_document_ids,
                    top_k=args.top_k,
                    diagnostics_case=diagnostics.get("cases", {}).get(case.case_name) if diagnostics else None,
                    trace_markers=args.trace_markers,
                    include_vector=args.include_vector,
                    include_indexed_sparse=args.include_indexed_sparse,
                    include_document_sweep=args.include_document_sweep,
                    include_document_first=args.include_document_first,
                    include_document_neighbor=args.include_document_neighbor,
                    include_preservation=args.include_preservation,
                    include_final_coverage=args.include_final_coverage,
                    force_heuristic_rerank=args.force_heuristic_rerank,
                    include_full_rerank=args.include_full_rerank,
                    rerank_result_limit=args.rerank_result_limit,
                    match_mode=args.match_mode,
                )
            )
        return {
            "dataset_name": args.dataset,
            "diagnostics_report": diagnostics.get("path") if diagnostics else None,
            "diagnostics_ablation": diagnostics.get("ablation") if diagnostics else None,
            "top_k": args.top_k,
            "document_scope": {
                "title_prefix": args.document_title_prefix if args.document_title_prefix is not None else f"{args.dataset}:",
                "document_count": len(scoped_document_ids) if scoped_document_ids is not None else None,
            },
            "audit_options": {
                "include_vector": args.include_vector,
                "include_indexed_sparse": args.include_indexed_sparse,
                "include_document_sweep": args.include_document_sweep,
                "include_document_first": args.include_document_first,
                "include_document_neighbor": args.include_document_neighbor,
                "include_preservation": args.include_preservation,
                "include_final_coverage": args.include_final_coverage,
                "include_evidence_bridge": args.include_evidence_bridge,
                "force_heuristic_rerank": args.force_heuristic_rerank,
                "include_full_rerank": args.include_full_rerank,
                "rerank_result_limit": args.rerank_result_limit,
                "match_mode": args.match_mode,
            },
            "summary": summarize(rows),
            "cases": rows,
        }
    finally:
        session.close()


def resolve_scoped_document_ids(session, dataset_name: str, document_title_prefix: str | None) -> list[UUID] | None:
    prefix = document_title_prefix if document_title_prefix is not None else f"{dataset_name}:"
    if not prefix:
        return None
    return list(session.scalars(select(Document.id).where(Document.title.like(f"{prefix}%"))).all())


def audit_case(
    *,
    service: RetrievalService,
    session,
    case,
    actor,
    annotations: dict[str, Any],
    scoped_document_ids: list[UUID] | None,
    top_k: int,
    diagnostics_case: dict[str, Any] | None,
    trace_markers: int,
    include_vector: bool,
    include_indexed_sparse: bool,
    include_document_sweep: bool,
    include_document_first: bool,
    include_document_neighbor: bool,
    include_preservation: bool,
    include_final_coverage: bool,
    force_heuristic_rerank: bool,
    include_full_rerank: bool,
    rerank_result_limit: int | None,
    match_mode: str,
) -> dict[str, Any]:
    expected_titles = benchmark.normalize_titles(annotations["expected_retrieval_titles"])
    evidence_markers = benchmark.normalize_evidence_markers(annotations.get("expected_evidence_markers"))[:trace_markers]
    accessible_document_ids = service.permission_builder.resolve_accessible_document_ids(service.session, actor, require_manage=False)
    if scoped_document_ids is not None:
        scoped = set(scoped_document_ids)
        accessible_document_ids = [item for item in accessible_document_ids if item in scoped]

    query_plan = service.query_optimizer.build(case.question)
    probe_applied = False
    if len(query_plan.candidates) > 1:
        probe_applied = service._select_best_query_plan(  # noqa: SLF001
            query_plan,
            accessible_document_ids=accessible_document_ids,
            target_document_title=None,
        )
    candidate_pool = service._candidate_pool_size(top_k)  # noqa: SLF001
    lexical_hits = service._collect_lexical_hits(query_plan.lexical_queries, accessible_document_ids, candidate_pool)  # noqa: SLF001
    structural_hits = service._collect_structural_hits(query_plan.lexical_queries, accessible_document_ids, candidate_pool)  # noqa: SLF001
    indexed_sparse_hits = (
        service._collect_indexed_sparse_hits(query_plan.lexical_queries, accessible_document_ids, candidate_pool)  # noqa: SLF001
        if include_indexed_sparse
        else []
    )
    vector_hits: list[RetrievalCandidate] = []
    if include_vector:
        query_embedding = service.embedding_provider.embed_texts([query_plan.retrieval_query])[0]
        vector_hits = service.retrieval_repository.search_vector(query_embedding, accessible_document_ids, candidate_pool)

    fused = service._fuse_hits(  # noqa: SLF001
        lexical_hits,
        vector_hits,
        structural_hits=structural_hits,
        indexed_sparse_hits=indexed_sparse_hits,
    )
    expansion_hits = service._collect_in_document_expansion(query_plan.retrieval_query, fused.values())  # noqa: SLF001
    if expansion_hits:
        fused = service._fuse_hits(  # noqa: SLF001
            lexical_hits,
            vector_hits,
            structural_hits=structural_hits,
            indexed_sparse_hits=indexed_sparse_hits,
            expansion_hits=expansion_hits,
        )
    document_sweep_hits = []
    if include_document_sweep:
        document_sweep_hits = service._collect_document_evidence_sweep(query_plan.retrieval_query, fused.values())  # noqa: SLF001
        if document_sweep_hits:
            fused = service._fuse_hits(  # noqa: SLF001
                lexical_hits,
                vector_hits,
                structural_hits=structural_hits,
                indexed_sparse_hits=indexed_sparse_hits,
                expansion_hits=expansion_hits,
                document_sweep_hits=document_sweep_hits,
            )

    document_first_hits = []
    if include_document_first:
        document_first_hits = service._collect_document_first_evidence_hits(query_plan.retrieval_query, fused.values())  # noqa: SLF001
        if document_first_hits:
            fused = service._fuse_hits(  # noqa: SLF001
                lexical_hits,
                vector_hits,
                structural_hits=structural_hits,
                indexed_sparse_hits=indexed_sparse_hits,
                expansion_hits=expansion_hits,
                document_sweep_hits=document_sweep_hits,
                document_first_hits=document_first_hits,
            )

    neighbor_hits = []
    if include_document_neighbor:
        neighbor_hits = service._collect_document_neighbor_context_hits(fused.values())  # noqa: SLF001
        if neighbor_hits:
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

    pre_rerank = sorted_rerank_candidates(fused.values())
    semantic_rerank_pool = _select_rerank_candidates(
        list(fused.values()),
        max(1, int(getattr(service.settings, "rerank_max_candidates", 16) or 16)),
    )
    rerank_query = service._build_rerank_query(case.question, query_plan.retrieval_query)  # noqa: SLF001
    if service._should_run_reranker():  # noqa: SLF001
        rerank_limit = service._rerank_result_limit(top_k, candidate_pool, len(fused))  # noqa: SLF001
        if rerank_result_limit is not None:
            rerank_limit = min(len(fused), max(top_k, int(rerank_result_limit)))
        reranked_result = service.reranker.rerank(rerank_query, list(fused.values()), rerank_limit)
    else:
        ranked_without_rerank = service._rank_without_rerank(fused.values())  # noqa: SLF001
        from app.services.retrieval.reranker import RerankResult

        reranked_result = RerankResult(
            candidates=ranked_without_rerank,
            strategy="disabled-local-heuristic",
            pre_rerank_count=len(fused),
            post_rerank_count=len(ranked_without_rerank),
        )
    reranked_full_candidates = []
    if include_full_rerank and service._should_run_reranker():  # noqa: SLF001
        reranked_full_result = service.reranker.rerank(rerank_query, list(fused.values()), max(len(fused), top_k))
        reranked_full_candidates = reranked_full_result.candidates
    preservation_candidates = service._collect_evidence_preservation_candidates(pre_rerank) if include_preservation else []  # noqa: SLF001
    base_final_candidates = service._select_final_candidates(reranked_result.candidates, top_k)  # noqa: SLF001
    coverage_candidates = (
        service._collect_final_coverage_candidates(case.question, reranked_result.candidates, base_final_candidates, top_k)  # noqa: SLF001
        if include_final_coverage
        else []
    )
    final_candidates = service._select_final_candidates(  # noqa: SLF001
        reranked_result.candidates,
        top_k,
        preservation_candidates=preservation_candidates,
        coverage_candidates=coverage_candidates,
    )

    expected_document_chunks = load_expected_document_chunks(session, expected_titles)
    stages = {
        "expected_document_chunks": expected_document_chunks,
        "lexical": [from_candidate(item) for item in lexical_hits],
        "structural": [from_candidate(item) for item in structural_hits],
        "indexed_sparse": [from_candidate(item) for item in indexed_sparse_hits],
        "vector": [from_candidate(item) for item in vector_hits],
        "fused": [from_rerank_candidate(item) for item in sorted_rerank_candidates(fused.values())],
        "expansion": [from_candidate(item) for item in expansion_hits],
        "document_sweep": [from_candidate(item) for item in document_sweep_hits],
        "document_first_evidence": [from_candidate(item) for item in document_first_hits],
        "document_neighbor_context": [from_candidate(item) for item in neighbor_hits],
        "pre_rerank": [from_rerank_candidate(item) for item in pre_rerank],
        "semantic_rerank_pool": [from_rerank_candidate(item) for item in semantic_rerank_pool],
        "reranked_full": [from_rerank_candidate(item) for item in reranked_full_candidates],
        "reranked": [from_rerank_candidate(item) for item in reranked_result.candidates],
        "final": [from_rerank_candidate(item) for item in final_candidates],
    }
    marker_rows = [locate_marker(marker, stages, query=case.question, match_mode=match_mode) for marker in evidence_markers]
    marker_root_causes = [marker["root_cause"] for marker in marker_rows]
    return {
        "case_name": case.case_name,
        "question": case.question,
        "diagnostics_stage_loss": diagnostics_case.get("stage_loss") if diagnostics_case else None,
        "diagnostics_failure_mode": diagnostics_case.get("final_score", {}).get("failure_mode") if diagnostics_case else None,
        "case_root_cause": classify_case_root_cause(marker_root_causes),
        "query_plan": {
            "retrieval_query": query_plan.retrieval_query,
            "lexical_queries": query_plan.lexical_queries,
            "selected": query_plan.selected_candidate.label,
            "selection_reason": query_plan.selected_candidate_reason,
            "probe_applied": probe_applied,
            "candidate_count": query_plan.candidate_count,
        },
        "stage_counts": {stage: len(items) for stage, items in stages.items()},
        "stage_document_hits": stage_document_hits(stages, expected_titles),
        "markers": marker_rows,
    }


def load_expected_document_chunks(session, expected_titles: set[str]) -> list[StageItem]:
    if not expected_titles:
        return []
    rows = session.execute(
        select(Chunk, Document)
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.title.in_(expected_titles))
        .where(Chunk.document_version_id == Document.current_version_id)
        .order_by(Document.title, Chunk.chunk_index)
    ).all()
    return [
        StageItem(
            chunk_id=chunk.id,
            document_id=document.id,
            document_title=document.title,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            section_title=chunk.section_title,
            clause_full_name=chunk.clause_full_name,
            article_number=chunk.article_number,
            chunk_type=chunk.chunk_type,
            heading_path=chunk.heading_path,
            structural_search_text=chunk.structural_search_text,
            lexical_search_text=chunk.lexical_search_text,
        )
        for chunk, document in rows
    ]


def from_candidate(candidate: RetrievalCandidate) -> StageItem:
    return StageItem(
        chunk_id=candidate.chunk_id,
        document_id=candidate.document_id,
        document_title=candidate.document_title,
        chunk_index=candidate.chunk_index,
        content=candidate.content,
        section_title=candidate.section_title,
        clause_full_name=candidate.clause_full_name,
        article_number=candidate.article_number,
        chunk_type=candidate.chunk_type,
        heading_path=candidate.heading_path,
        structural_search_text=candidate.structural_search_text,
        lexical_search_text=candidate.lexical_search_text,
        lexical_raw=candidate.lexical_score,
        vector_raw=candidate.vector_score,
    )


def from_rerank_candidate(candidate: RerankCandidate) -> StageItem:
    item = candidate.candidate
    return StageItem(
        chunk_id=item.chunk_id,
        document_id=item.document_id,
        document_title=item.document_title,
        chunk_index=item.chunk_index,
        content=item.content,
        section_title=item.section_title,
        clause_full_name=item.clause_full_name,
        article_number=item.article_number,
        chunk_type=item.chunk_type,
        heading_path=item.heading_path,
        structural_search_text=item.structural_search_text,
        lexical_search_text=item.lexical_search_text,
        sources=tuple(sorted(candidate.sources)),
        fused_score=candidate.fused_score,
        lexical_raw=candidate.lexical_raw,
        lexical_norm=candidate.lexical_norm,
        vector_raw=candidate.vector_raw,
        vector_norm=candidate.vector_norm,
        rerank_score=candidate.rerank_score,
    )


def sorted_rerank_candidates(candidates: Iterable[RerankCandidate]) -> list[RerankCandidate]:
    return sorted(
        candidates,
        key=lambda item: (item.fused_score, item.lexical_raw, -item.candidate.chunk_index),
        reverse=True,
    )


def locate_marker(marker: dict[str, Any], stages: dict[str, list[StageItem]], *, query: str, match_mode: str) -> dict[str, Any]:
    aliases = normalized_marker_aliases(marker)
    expected_title = str(marker.get("document_title") or "")
    stage_hits = {stage: locate_in_stage(items, aliases, expected_title, match_mode=match_mode) for stage, items in stages.items()}
    root_cause = classify_marker_root_cause(stage_hits)
    first_hit = next((stage_hits[stage] for stage in PIPELINE_STAGES if stage_hits.get(stage)), None)
    overlap = query_marker_overlap(query, marker, first_hit)
    proximity = document_local_proximity(stage_hits, stages, expected_title)
    return {
        "label": marker.get("label"),
        "document_title": expected_title,
        "aliases_checked": marker.get("aliases", [])[:5],
        "root_cause": root_cause,
        "query_evidence_overlap": overlap,
        "document_local_proximity": proximity,
        "first_available_location": compact_hit(first_hit),
        "stage_ranks": {stage: hit["rank"] if hit else None for stage, hit in stage_hits.items()},
        "stage_hits": {stage: compact_hit(hit) for stage, hit in stage_hits.items() if hit},
    }


def query_marker_overlap(query: str, marker: dict[str, Any], first_hit: dict[str, Any] | None) -> dict[str, Any]:
    query_tokens = set(tokenize_search_text(query))
    evidence_text = " ".join(
        str(part or "")
        for part in [
            marker.get("label"),
            " ".join(str(alias) for alias in marker.get("aliases", [])[:5]),
            (first_hit or {}).get("preview"),
        ]
    )
    evidence_tokens = set(tokenize_search_text(evidence_text))
    overlap_terms = sorted(query_tokens.intersection(evidence_tokens))
    denominator = max(len(query_tokens), 1)
    return {
        "query_token_count": len(query_tokens),
        "evidence_token_count": len(evidence_tokens),
        "overlap_token_count": len(overlap_terms),
        "overlap_ratio": round(len(overlap_terms) / denominator, 6),
        "overlap_terms": overlap_terms[:24],
    }


def document_local_proximity(
    stage_hits: dict[str, dict[str, Any] | None],
    stages: dict[str, list[StageItem]],
    expected_title: str,
) -> dict[str, Any]:
    expected_hit = stage_hits.get("expected_document_chunks")
    if not expected_hit:
        return {
            "available": False,
            "reason": "marker_not_found_in_expected_document_chunks",
        }
    evidence_chunk_index = int(expected_hit["chunk_index"])
    source_nearest_by_stage = {
        stage: hit
        for stage in SOURCE_STAGES
        if (hit := nearest_document_candidate(stage, stages.get(stage, []), expected_title, evidence_chunk_index))
    }
    source_nearest = nearest_distance_hit(source_nearest_by_stage.values())
    expansion_nearest = nearest_document_candidate("expansion", stages.get("expansion", []), expected_title, evidence_chunk_index)
    pipeline_nearest_by_stage = {
        stage: hit
        for stage in ("fused", "pre_rerank", "semantic_rerank_pool", "reranked_full", "reranked", "final")
        if (hit := nearest_document_candidate(stage, stages.get(stage, []), expected_title, evidence_chunk_index))
    }
    pipeline_nearest = nearest_distance_hit(pipeline_nearest_by_stage.values())
    return {
        "available": True,
        "evidence_chunk_index": evidence_chunk_index,
        "evidence_position_bucket": chunk_position_bucket(evidence_chunk_index),
        "source_nearest": source_nearest,
        "source_within_windows": distance_window_flags(source_nearest["distance"] if source_nearest else None),
        "source_nearest_by_stage": source_nearest_by_stage,
        "expansion_nearest": expansion_nearest,
        "pipeline_nearest": pipeline_nearest,
    }


def nearest_document_candidate(
    stage: str,
    items: list[StageItem],
    expected_title: str,
    evidence_chunk_index: int,
) -> dict[str, Any] | None:
    candidates = []
    for rank, item in enumerate(items, start=1):
        if expected_title and item.document_title != expected_title:
            continue
        distance = abs(item.chunk_index - evidence_chunk_index)
        candidates.append(
            {
                "stage": stage,
                "rank": rank,
                "document_title": item.document_title,
                "chunk_index": item.chunk_index,
                "distance": distance,
                "section_title": item.section_title,
                "clause_full_name": item.clause_full_name,
                "article_number": item.article_number,
                "chunk_type": item.chunk_type,
                "sources": list(item.sources),
                "preview": compact_preview(item.content, limit=140),
            }
        )
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item["distance"], item["rank"]))


def nearest_distance_hit(hits: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    available = [hit for hit in hits if hit]
    if not available:
        return None
    return min(available, key=lambda item: (item["distance"], item["rank"]))


def distance_window_flags(distance: int | None) -> dict[str, bool]:
    return {f"within_{window}": distance is not None and distance <= window for window in DOCUMENT_LOCAL_DISTANCE_WINDOWS}


def chunk_position_bucket(chunk_index: int) -> str:
    if chunk_index <= 20:
        return "<=20"
    if chunk_index <= 80:
        return "21-80"
    if chunk_index <= 120:
        return "81-120"
    if chunk_index <= 300:
        return "121-300"
    return ">300"


def normalized_marker_aliases(marker: dict[str, Any]) -> list[tuple[str, str]]:
    aliases = [str(item) for item in marker.get("aliases", []) if str(item).strip()]
    if marker.get("label"):
        aliases.insert(0, str(marker["label"]))
    seen: set[str] = set()
    normalized = []
    for alias in aliases:
        key = normalize_marker_text(alias)
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append((alias, key))
    return normalized


def locate_in_stage(
    items: list[StageItem],
    aliases: list[tuple[str, str]],
    expected_title: str,
    *,
    match_mode: str,
) -> dict[str, Any] | None:
    for rank, item in enumerate(items, start=1):
        if expected_title and item.document_title != expected_title:
            continue
        structural_haystack = normalize_marker_text(
            " ".join(
                str(part or "")
                for part in [
                    item.document_title,
                    item.section_title,
                    item.clause_full_name,
                    item.article_number,
                    item.heading_path,
                ]
            )
        )
        content_haystack = normalize_marker_text(
            " ".join(
                str(part or "")
                for part in [
                    item.structural_search_text,
                    item.lexical_search_text,
                    item.content,
                ]
            )
        )
        for raw_alias, normalized_alias in aliases:
            match_field = marker_match_field(
                raw_alias=raw_alias,
                normalized_alias=normalized_alias,
                structural_haystack=structural_haystack,
                content_haystack=content_haystack,
                match_mode=match_mode,
            )
            if match_field:
                return {
                    "rank": rank,
                    "matched_alias": raw_alias,
                    "match_field": match_field,
                    "document_title": item.document_title,
                    "chunk_index": item.chunk_index,
                    "section_title": item.section_title,
                    "clause_full_name": item.clause_full_name,
                    "article_number": item.article_number,
                    "chunk_type": item.chunk_type,
                    "sources": list(item.sources),
                    "fused_score": round(item.fused_score, 6) if item.fused_score is not None else None,
                    "lexical_raw": round(item.lexical_raw, 6) if item.lexical_raw is not None else None,
                    "lexical_norm": round(item.lexical_norm, 6) if item.lexical_norm is not None else None,
                    "rerank_score": round(item.rerank_score, 6) if item.rerank_score is not None else None,
                    "preview": compact_preview(item.content),
                }
    return None


def marker_match_field(
    *,
    raw_alias: str,
    normalized_alias: str,
    structural_haystack: str,
    content_haystack: str,
    match_mode: str,
) -> str | None:
    if normalized_alias in structural_haystack:
        return "structural"
    if match_mode == "lenient":
        return "content_or_reference" if normalized_alias in content_haystack else None
    if is_structural_marker_alias(raw_alias):
        return None
    if normalized_alias in content_haystack:
        return "content"
    if long_marker_fuzzy_matches(normalized_alias, content_haystack):
        return "content_fuzzy"
    return None


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
    text = str(value or "").strip()
    if not text:
        return False
    if re.search(r"第[〇零一二三四五六七八九十百千万两0-9]+条", text):
        return True
    return bool(re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9《》（）()]+法[\u4e00-\u9fffA-Za-z0-9《》（）()]*", text))


def classify_marker_root_cause(stage_hits: dict[str, dict[str, Any] | None]) -> str:
    if not stage_hits["expected_document_chunks"]:
        return "marker_not_found_in_expected_document_chunks"
    source_hit = any(stage_hits[stage] for stage in SOURCE_STAGES)
    if not source_hit:
        if stage_hits["expansion"]:
            return "only_recovered_by_in_document_expansion"
        if stage_hits["document_sweep"]:
            return "only_recovered_by_document_sweep"
        return "source_candidate_generation_missed"
    if not stage_hits["fused"]:
        return "fusion_dropped_source_hit"
    if not stage_hits["pre_rerank"]:
        return "pre_rerank_missing_after_fusion"
    if not stage_hits["reranked"]:
        return "rerank_truncated_evidence"
    if not stage_hits["final"]:
        return "final_top_k_truncated_evidence"
    return "final_hit"


def classify_case_root_cause(marker_root_causes: Sequence[str]) -> str:
    if not marker_root_causes:
        return "no_evidence_markers"
    if all(item == "final_hit" for item in marker_root_causes):
        return "passed"
    priority = [
        "marker_not_found_in_expected_document_chunks",
        "source_candidate_generation_missed",
        "only_recovered_by_in_document_expansion",
        "only_recovered_by_document_sweep",
        "fusion_dropped_source_hit",
        "pre_rerank_missing_after_fusion",
        "rerank_truncated_evidence",
        "final_top_k_truncated_evidence",
    ]
    causes = set(marker_root_causes)
    for cause in priority:
        if cause in causes:
            return cause
    return "mixed_unclassified"


def stage_document_hits(stages: dict[str, list[StageItem]], expected_titles: set[str]) -> dict[str, list[str]]:
    result = {}
    for stage, items in stages.items():
        titles = []
        seen = set()
        for item in items:
            if item.document_title in expected_titles and item.document_title not in seen:
                seen.add(item.document_title)
                titles.append(item.document_title)
        result[stage] = titles
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    marker_causes = Counter(
        marker["root_cause"]
        for row in rows
        for marker in row.get("markers", [])
    )
    case_causes = Counter(row["case_root_cause"] for row in rows)
    diagnostics_losses = Counter(row.get("diagnostics_stage_loss") or "unknown" for row in rows)
    markers = [marker for row in rows for marker in row.get("markers", [])]
    source_missed_markers = [marker for marker in markers if marker.get("root_cause") == "source_candidate_generation_missed"]
    return {
        "case_count": len(rows),
        "case_root_cause_counts": dict(case_causes),
        "marker_root_cause_counts": dict(marker_causes),
        "diagnostics_stage_loss_counts": dict(diagnostics_losses),
        "source_candidate_generation_missed_proximity": summarize_document_local_proximity(source_missed_markers),
        "semantic_rerank_pool_visibility": summarize_semantic_rerank_pool_visibility(markers),
        "semantic_rerank_pool_oracle": summarize_semantic_rerank_pool_oracle(rows),
        "document_neighbor_candidate_oracle": summarize_document_neighbor_candidate_oracle(rows),
    }


def summarize_document_local_proximity(markers: list[dict[str, Any]]) -> dict[str, Any]:
    proximities = [marker.get("document_local_proximity") or {} for marker in markers]
    available = [item for item in proximities if item.get("available")]
    with_source_nearest = [item for item in available if item.get("source_nearest")]
    distances = sorted(int(item["source_nearest"]["distance"]) for item in with_source_nearest)
    stage_counts = Counter(item["source_nearest"]["stage"] for item in with_source_nearest)
    bucket_counts = Counter(str(item.get("evidence_position_bucket") or "unknown") for item in available)
    window_counts = {
        f"within_{window}": sum(
            1 for item in with_source_nearest if int(item["source_nearest"]["distance"]) <= window
        )
        for window in DOCUMENT_LOCAL_DISTANCE_WINDOWS
    }
    return {
        "marker_count": len(markers),
        "available_marker_count": len(available),
        "with_source_neighbor_count": len(with_source_nearest),
        "without_source_neighbor_count": len(available) - len(with_source_nearest),
        "distance_min": distances[0] if distances else None,
        "distance_median": percentile_value(distances, 0.5),
        "distance_p90": percentile_value(distances, 0.9),
        "distance_max": distances[-1] if distances else None,
        "within_window_counts": window_counts,
        "nearest_source_stage_counts": dict(stage_counts),
        "evidence_position_bucket_counts": dict(bucket_counts),
    }


def percentile_value(values: Sequence[int], percentile: float) -> int | None:
    if not values:
        return None
    index = round((len(values) - 1) * percentile)
    return values[index]


def summarize_semantic_rerank_pool_visibility(markers: list[dict[str, Any]]) -> dict[str, Any]:
    visible = []
    fused_available = []
    fused_available_visible = []
    expansion_recovered = []
    expansion_recovered_visible = []
    visible_by_root = Counter()
    missing_by_root = Counter()
    for marker in markers:
        ranks = marker.get("stage_ranks") or {}
        root_cause = str(marker.get("root_cause") or "unknown")
        in_semantic_pool = ranks.get("semantic_rerank_pool") is not None
        if in_semantic_pool:
            visible.append(marker)
            visible_by_root[root_cause] += 1
        else:
            missing_by_root[root_cause] += 1
        if ranks.get("fused") is not None:
            fused_available.append(marker)
            if in_semantic_pool:
                fused_available_visible.append(marker)
        if root_cause in {"only_recovered_by_in_document_expansion", "only_recovered_by_document_sweep"}:
            expansion_recovered.append(marker)
            if in_semantic_pool:
                expansion_recovered_visible.append(marker)
    marker_count = len(markers)
    fused_count = len(fused_available)
    expansion_count = len(expansion_recovered)
    return {
        "marker_count": marker_count,
        "visible_marker_count": len(visible),
        "visible_ratio": ratio(len(visible), marker_count),
        "fused_available_marker_count": fused_count,
        "fused_available_visible_count": len(fused_available_visible),
        "fused_available_visible_ratio": ratio(len(fused_available_visible), fused_count),
        "expansion_recovered_marker_count": expansion_count,
        "expansion_recovered_visible_count": len(expansion_recovered_visible),
        "expansion_recovered_visible_ratio": ratio(len(expansion_recovered_visible), expansion_count),
        "visible_by_root_cause": dict(visible_by_root),
        "missing_by_root_cause": dict(missing_by_root),
    }


def summarize_semantic_rerank_pool_oracle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    classifications = Counter(classify_semantic_rerank_pool_oracle(row.get("markers", [])) for row in rows)
    rescuable_cases = [
        row
        for row in rows
        if classify_semantic_rerank_pool_oracle(row.get("markers", [])) == "semantic_pool_oracle_possible"
    ]
    return {
        "case_count": len(rows),
        "classification_counts": dict(classifications),
        "oracle_rescuable_case_count": len(rescuable_cases),
        "oracle_rescuable_case_names": [str(row.get("case_name")) for row in rescuable_cases[:20]],
    }


def classify_semantic_rerank_pool_oracle(markers: list[dict[str, Any]]) -> str:
    if not markers:
        return "no_evidence_markers"
    if all((marker.get("stage_ranks") or {}).get("final") is not None for marker in markers):
        return "already_final"
    if all(marker_final_or_semantic_pool_visible(marker) for marker in markers):
        return "semantic_pool_oracle_possible"
    if any((marker.get("stage_ranks") or {}).get("fused") is None for marker in markers):
        return "blocked_by_candidate_generation"
    return "blocked_by_semantic_pool_visibility"


def marker_final_or_semantic_pool_visible(marker: dict[str, Any]) -> bool:
    ranks = marker.get("stage_ranks") or {}
    return ranks.get("final") is not None or ranks.get("semantic_rerank_pool") is not None


def summarize_document_neighbor_candidate_oracle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    marker_rows = [marker for row in rows for marker in row.get("markers", [])]
    not_fused_markers = [marker for marker in marker_rows if (marker.get("stage_ranks") or {}).get("fused") is None]
    window_summaries = {}
    for window in DOCUMENT_LOCAL_DISTANCE_WINDOWS:
        case_classifications = Counter(classify_document_neighbor_candidate_oracle(row.get("markers", []), window) for row in rows)
        possible_cases = [
            row
            for row in rows
            if classify_document_neighbor_candidate_oracle(row.get("markers", []), window) == "candidate_pool_oracle_possible"
        ]
        recovered_not_fused = [marker for marker in not_fused_markers if marker_document_neighbor_visible(marker, window)]
        source_missed_recovered = [
            marker
            for marker in marker_rows
            if marker.get("root_cause") == "source_candidate_generation_missed"
            and marker_document_neighbor_visible(marker, window)
        ]
        window_summaries[f"within_{window}"] = {
            "window": window,
            "case_classification_counts": dict(case_classifications),
            "candidate_pool_oracle_possible_case_count": len(possible_cases),
            "candidate_pool_oracle_possible_case_names": [str(row.get("case_name")) for row in possible_cases[:20]],
            "not_fused_marker_count": len(not_fused_markers),
            "not_fused_marker_neighbor_visible_count": len(recovered_not_fused),
            "not_fused_marker_neighbor_visible_ratio": ratio(len(recovered_not_fused), len(not_fused_markers)),
            "source_missed_marker_neighbor_visible_count": len(source_missed_recovered),
        }
    return {
        "case_count": len(rows),
        "marker_count": len(marker_rows),
        "not_fused_marker_count": len(not_fused_markers),
        "windows": window_summaries,
    }


def classify_document_neighbor_candidate_oracle(markers: list[dict[str, Any]], window: int) -> str:
    if not markers:
        return "no_evidence_markers"
    if all((marker.get("stage_ranks") or {}).get("final") is not None for marker in markers):
        return "already_final"
    if all(marker_fused_or_neighbor_visible(marker, window) for marker in markers):
        return "candidate_pool_oracle_possible"
    if any(not marker_has_expected_location(marker) for marker in markers):
        return "missing_expected_marker_location"
    return "blocked_by_neighbor_window"


def marker_fused_or_neighbor_visible(marker: dict[str, Any], window: int) -> bool:
    ranks = marker.get("stage_ranks") or {}
    if ranks.get("fused") is not None:
        return True
    return marker_document_neighbor_visible(marker, window)


def marker_document_neighbor_visible(marker: dict[str, Any], window: int) -> bool:
    proximity = marker.get("document_local_proximity") or {}
    source_nearest = proximity.get("source_nearest") or {}
    distance = source_nearest.get("distance")
    return isinstance(distance, int) and distance <= window


def marker_has_expected_location(marker: dict[str, Any]) -> bool:
    return bool((marker.get("document_local_proximity") or {}).get("available"))


def ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def compact_hit(hit: dict[str, Any] | None) -> dict[str, Any] | None:
    if not hit:
        return None
    keys = [
        "rank",
        "matched_alias",
        "match_field",
        "document_title",
        "chunk_index",
        "section_title",
        "clause_full_name",
        "article_number",
        "chunk_type",
        "sources",
        "fused_score",
        "lexical_raw",
        "lexical_norm",
        "rerank_score",
        "preview",
    ]
    return {key: hit[key] for key in keys if key in hit and hit[key] not in (None, [], "")}


def compact_preview(value: str, limit: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def normalize_marker_text(value: str) -> str:
    return "".join(
        char
        for char in " ".join(str(value).casefold().split())
        if char not in "，。；：、,.!?！？;:()（）[]【】\"' \n\r\t"
    )


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Retrieval Root Cause Audit",
        "",
        f"- Dataset: `{report['dataset_name']}`",
        f"- Diagnostics report: `{report.get('diagnostics_report')}`",
        f"- Ablation: `{(report.get('diagnostics_ablation') or {}).get('name')}`",
        f"- Cases audited: `{summary['case_count']}`",
        "",
        "## Case Root Causes",
        "",
        "| Root cause | Count |",
        "| --- | ---: |",
    ]
    for cause, count in sorted(summary["case_root_cause_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{cause}` | {count} |")
    lines.extend(["", "## Marker Root Causes", "", "| Root cause | Count |", "| --- | ---: |"])
    for cause, count in sorted(summary["marker_root_cause_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{cause}` | {count} |")
    lines.extend(["", "## Diagnostics Stage Losses", "", "| Stage loss | Count |", "| --- | ---: |"])
    for cause, count in sorted(summary["diagnostics_stage_loss_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{cause}` | {count} |")
    proximity = summary.get("source_candidate_generation_missed_proximity") or {}
    if proximity:
        lines.extend(
            [
                "",
                "## Source-Missed Document-Local Proximity",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Markers | {proximity.get('marker_count')} |",
                f"| Available expected-marker locations | {proximity.get('available_marker_count')} |",
                f"| With same-document source neighbor | {proximity.get('with_source_neighbor_count')} |",
                f"| Without same-document source neighbor | {proximity.get('without_source_neighbor_count')} |",
                f"| Distance median | {proximity.get('distance_median')} |",
                f"| Distance p90 | {proximity.get('distance_p90')} |",
                f"| Distance max | {proximity.get('distance_max')} |",
            ]
        )
        lines.extend(["", "| Source neighbor window | Count |", "| --- | ---: |"])
        for window, count in (proximity.get("within_window_counts") or {}).items():
            lines.append(f"| `{window}` | {count} |")
        lines.extend(["", "| Nearest source stage | Count |", "| --- | ---: |"])
        for stage, count in sorted((proximity.get("nearest_source_stage_counts") or {}).items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"| `{stage}` | {count} |")
        lines.extend(["", "| Evidence chunk position | Count |", "| --- | ---: |"])
        for bucket, count in sorted((proximity.get("evidence_position_bucket_counts") or {}).items()):
            lines.append(f"| `{bucket}` | {count} |")
    visibility = summary.get("semantic_rerank_pool_visibility") or {}
    if visibility:
        lines.extend(
            [
                "",
                "## Semantic Rerank Pool Visibility",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Markers | {visibility.get('marker_count')} |",
                f"| Visible in semantic pool | {visibility.get('visible_marker_count')} |",
                f"| Visible ratio | {visibility.get('visible_ratio')} |",
                f"| Fused-available markers | {visibility.get('fused_available_marker_count')} |",
                f"| Fused-available visible | {visibility.get('fused_available_visible_count')} |",
                f"| Fused-available visible ratio | {visibility.get('fused_available_visible_ratio')} |",
                f"| Expansion-recovered markers | {visibility.get('expansion_recovered_marker_count')} |",
                f"| Expansion-recovered visible | {visibility.get('expansion_recovered_visible_count')} |",
                f"| Expansion-recovered visible ratio | {visibility.get('expansion_recovered_visible_ratio')} |",
            ]
        )
        lines.extend(["", "| Visible root cause | Count |", "| --- | ---: |"])
        for cause, count in sorted((visibility.get("visible_by_root_cause") or {}).items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"| `{cause}` | {count} |")
        lines.extend(["", "| Missing root cause | Count |", "| --- | ---: |"])
        for cause, count in sorted((visibility.get("missing_by_root_cause") or {}).items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"| `{cause}` | {count} |")
    oracle = summary.get("semantic_rerank_pool_oracle") or {}
    if oracle:
        lines.extend(
            [
                "",
                "## Semantic Rerank Pool Oracle",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Cases | {oracle.get('case_count')} |",
                f"| Oracle-rescuable cases | {oracle.get('oracle_rescuable_case_count')} |",
            ]
        )
        lines.extend(["", "| Classification | Count |", "| --- | ---: |"])
        for label, count in sorted((oracle.get("classification_counts") or {}).items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"| `{label}` | {count} |")
    neighbor_oracle = summary.get("document_neighbor_candidate_oracle") or {}
    neighbor_windows = neighbor_oracle.get("windows") or {}
    if neighbor_windows:
        lines.extend(
            [
                "",
                "## Document Neighbor Candidate Oracle",
                "",
                "| Window | Oracle-possible cases | Not-fused markers visible | Source-missed markers visible |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for name, item in neighbor_windows.items():
            lines.append(
                f"| `{name}` | "
                f"{item.get('candidate_pool_oracle_possible_case_count')} | "
                f"{item.get('not_fused_marker_neighbor_visible_count')}/{item.get('not_fused_marker_count')} "
                f"({item.get('not_fused_marker_neighbor_visible_ratio')}) | "
                f"{item.get('source_missed_marker_neighbor_visible_count')} |"
            )
    lines.extend(["", "## Representative Cases", ""])
    for row in report["cases"][:20]:
        lines.append(f"### {row['case_name']} - `{row['case_root_cause']}`")
        lines.append("")
        lines.append(f"- Diagnostics stage loss: `{row.get('diagnostics_stage_loss')}`")
        lines.append(f"- Question: {row['question']}")
        for marker in row.get("markers", [])[:3]:
            ranks = marker["stage_ranks"]
            overlap_ratio = (marker.get("query_evidence_overlap") or {}).get("overlap_ratio")
            proximity = marker.get("document_local_proximity") or {}
            source_nearest = proximity.get("source_nearest") or {}
            source_distance = source_nearest.get("distance")
            source_stage = source_nearest.get("stage")
            source_chunk = source_nearest.get("chunk_index")
            lines.append(
                "- Marker "
                f"`{marker.get('label')}`: `{marker['root_cause']}`, "
                f"query/evidence overlap={overlap_ratio}, "
                f"nearest source={source_stage}@chunk{source_chunk}/distance{source_distance}, "
                f"source ranks lexical/structural/indexed/vector="
                f"{ranks.get('lexical')}/{ranks.get('structural')}/{ranks.get('indexed_sparse')}/{ranks.get('vector')}, "
                f"pre/semantic-pool/full-rerank/reranked/final="
                f"{ranks.get('pre_rerank')}/{ranks.get('semantic_rerank_pool')}/"
                f"{ranks.get('reranked_full')}/{ranks.get('reranked')}/{ranks.get('final')}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def summary_text(report: dict[str, Any]) -> str:
    return json.dumps(report["summary"], ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
