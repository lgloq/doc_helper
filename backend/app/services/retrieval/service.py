from __future__ import annotations

import re
from dataclasses import dataclass, replace
from statistics import mean
from time import perf_counter
from typing import Callable, Iterable
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.document import Document
from app.models.user import User
from app.repositories.retrieval_repository import RetrievalCandidate, RetrievalRepository
from app.schemas.search import SearchDebugInfo, SearchRequest, SearchResponse, SearchResultChunk, SearchScoreBreakdown
from app.services.ingestion.embeddings import EmbeddingProviderFactory
from app.services.ingestion.search_index import tokenize_search_text
from app.services.permissions.service import PermissionFilterBuilder
from app.services.retrieval.query_optimizer import QueryOptimizer, QueryOptimizationPlan, QueryPlanCandidate, QuerySubquery
from app.services.retrieval.reranker import RerankCandidate, RerankerFactory, RerankResult

LEXICAL_WEIGHT = 0.45
VECTOR_WEIGHT = 0.55
PLAN_PROBE_LIMIT = 6
RRF_STRUCTURAL_WEIGHT = 1.35
RRF_LEXICAL_WEIGHT = 1.0
RRF_INDEXED_SPARSE_WEIGHT = 0.92
RRF_VECTOR_WEIGHT = 0.85
RRF_EXPANSION_WEIGHT = 0.65
RRF_DOCUMENT_SWEEP_WEIGHT = 0.58
DOCUMENT_DIVERSITY_WEAK_FUSED_CUTOFF = 0.08
DOCUMENT_DIVERSITY_STRONG_DEFERRED_FUSED = 0.18
DOCUMENT_DIVERSITY_RERANK_MARGIN = 0.20
FINAL_COVERAGE_QUERY_HINTS = (
    "包括",
    "列出",
    "清单",
    "步骤",
    "材料",
    "范围",
)
FINAL_COVERAGE_LIST_QUERY_HINTS = ("哪些", "有哪些", "有什么")
FINAL_COVERAGE_BROAD_QUERY_HINTS = (
    "要求",
    "义务",
    "责任",
    "职责",
    "条件",
    "流程",
    "措施",
    "依据",
)
FINAL_COVERAGE_MULTI_DELIMITERS = ("和", "及", "与", "以及", "、", "分别", "同时", "各自")
SUBQUERY_DOCUMENT_EVIDENCE_SOURCE = "subquery_document_evidence"
SUBQUERY_NEIGHBOR_CONTEXT_SOURCE = "subquery_neighbor_context"
FINAL_COVERAGE_SOURCE_NAMES = {
    "document_first_evidence",
    "document_expansion",
    "document_sweep",
    "document_neighbor_context",
    SUBQUERY_DOCUMENT_EVIDENCE_SOURCE,
    SUBQUERY_NEIGHBOR_CONTEXT_SOURCE,
}
FINAL_REPLACEMENT_PROTECTED_SOURCE_NAMES = {
    "document_first_evidence",
    "document_sweep",
    SUBQUERY_DOCUMENT_EVIDENCE_SOURCE,
    SUBQUERY_NEIGHBOR_CONTEXT_SOURCE,
}
FINAL_COVERAGE_LOW_SIGNAL_TERMS = {
    "一下",
    "哪些",
    "有哪",
    "什么",
    "怎么",
    "如何",
    "分别",
    "各自",
    "以及",
    "相关",
    "内容",
    "要求",
    "合规",
    "适用",
    "企业",
    "文档",
    "规定",
}
FINAL_COVERAGE_THRESHOLD_QUERY_HINTS = (
    "规模",
    "数量",
    "数额",
    "金额",
    "额度",
    "比例",
    "达到",
    "超过",
    "不满",
    "以上",
    "以下",
    "以内",
    "不少于",
    "不超过",
    "多少",
    "多久",
    "时限",
    "期限",
)
FINAL_COVERAGE_THRESHOLD_VALUE_PATTERN = re.compile(
    r"(?:不少于|不低于|不超过|不满|超过|达到|高于|低于|大于|小于)\s*"
    r"(?:\d+(?:\.\d+)?|[一二三四五六七八九十百千万亿两]+)\s*(?:万|亿)?"
    r"(?:人|户|个|件|次|元|万元|亿元|天|日|月|年|小时|分钟|工作日|%|％)?"
    r"|"
    r"(?:\d+(?:\.\d+)?|[一二三四五六七八九十百千万亿两]+)\s*(?:万|亿)?"
    r"(?:人|户|个|件|次|元|万元|亿元|天|日|月|年|小时|分钟|工作日|%|％)?\s*"
    r"(?:以上|以下|以内|以外|不满|少于|超过)"
)
FINAL_COVERAGE_MEASURE_VALUE_PATTERN = re.compile(
    r"(?:\d+(?:\.\d+)?|[一二三四五六七八九十百千万亿两]+)\s*(?:万|亿)?"
    r"(?:人|户|个|件|次|元|万元|亿元|天|日|月|年|小时|分钟|工作日|%|％)"
)
TABLE_LOOKUP_PAIR_PATTERN = re.compile(
    r"(?P<field>[\u4e00-\u9fffA-Za-z0-9（）() /·\-_%％]{2,32}?)\s*"
    r"(?:=|＝|:|：|为|是)\s*"
    r"[“”\"'‘’「」『』]?"
    r"(?P<value>[\u4e00-\u9fffA-Za-z0-9（）() /·\-_.%％]{2,80})"
)
PERMISSION_PROBE_TARGET_PATTERN = re.compile(
    r"(?:直接查看|查看)(?P<target>[\u4e00-\u9fffA-Za-z0-9（）() /·\-]{4,80}?)"
    r"(?:这份|的)?(?:受限材料|受限文件|受限文档)"
)


@dataclass
class QueryPlanProbeResult:
    candidate: QueryPlanCandidate
    hits: list[RetrievalCandidate]
    hit_count: int
    top_score: float
    average_score: float
    concentration: float
    title_match: float
    final_score: float = 0.0


@dataclass
class DecomposedSourceCollectionResult:
    hits: list[RetrievalCandidate]
    subquery_candidate_counts: list[dict[str, object]]
    timeout_count: int = 0
    timeout_fallback_candidate_count: int = 0


@dataclass(frozen=True)
class PermissionProbeEarlyStopResult:
    target_hint: str
    accessible_target_count: int
    inaccessible_target_count: int


class RetrievalService:
    def __init__(self, session: Session):
        self.session = session
        self.permission_builder = PermissionFilterBuilder()
        self.retrieval_repository = RetrievalRepository(session)
        self.embedding_provider = EmbeddingProviderFactory.create()
        self._settings = get_settings()
        self._last_structural_timeout = False
        self._configure_retrieval_components()

    @property
    def settings(self):
        return self._settings

    @settings.setter
    def settings(self, value):
        self._settings = value
        self._configure_retrieval_components()

    def _configure_retrieval_components(self) -> None:
        self.reranker = RerankerFactory.create(self._settings)
        self.query_optimizer = QueryOptimizer(self._settings)

    def search(
        self,
        actor: User,
        payload: SearchRequest,
        scoped_document_ids: list[UUID] | None = None,
        target_document_title: str | None = None,
    ) -> SearchResponse:
        search_started = perf_counter()
        permission_started = perf_counter()
        accessible_document_ids = self.permission_builder.resolve_accessible_document_ids(self.session, actor, require_manage=False)
        if scoped_document_ids is not None:
            scoped_set = {item for item in scoped_document_ids}
            accessible_document_ids = [item for item in accessible_document_ids if item in scoped_set]
        permission_filter_latency_ms = int((perf_counter() - permission_started) * 1000)
        query_plan = self.query_optimizer.build(payload.query, target_document_title=target_document_title)
        probe_applied = False
        probe_latency_ms = 0
        probe_skip_reason = self._query_plan_probe_skip_reason(
            query_plan,
            accessible_document_ids=accessible_document_ids,
            target_document_title=target_document_title,
        )
        probe_started = perf_counter()
        if probe_skip_reason is None:
            probe_applied = self._select_best_query_plan(
                query_plan,
                accessible_document_ids=accessible_document_ids,
                target_document_title=target_document_title,
            )
        probe_latency_ms = int((perf_counter() - probe_started) * 1000)
        permission_probe_early_stop = self._permission_probe_early_stop(payload.query, accessible_document_ids)
        if permission_probe_early_stop:
            return SearchResponse(
                query=payload.query,
                top_k=payload.top_k,
                matched_chunks=[],
                debug=SearchDebugInfo(
                    accessible_document_count=len(accessible_document_ids),
                    lexical_candidate_count=0,
                    vector_candidate_count=0,
                    structural_candidate_count=0,
                    fusion_strategy="permission-probe early stop",
                    pre_rerank_count=0,
                    post_rerank_count=0,
                    rerank_strategy="none",
                    retrieval_query=query_plan.retrieval_query,
                    lexical_queries=query_plan.lexical_queries,
                    query_rewrite_applied=query_plan.rewrite_applied,
                    query_rewrite_strategies=query_plan.applied_strategies,
                    query_rewrite_provider=query_plan.rewrite_provider,
                    query_rewrite_model=query_plan.rewrite_model,
                    query_rewrite_latency_ms=query_plan.rewrite_latency_ms,
                    llm_rewrite_attempted=query_plan.llm_rewrite_attempted,
                    llm_rewrite_skipped_reason=query_plan.llm_rewrite_skipped_reason,
                    llm_rewrite_latency_ms=query_plan.llm_rewrite_latency_ms,
                    query_decomposition_applied=self._query_decomposition_applies(query_plan),
                    subquery_count=len(query_plan.subqueries),
                    subquery_candidate_counts=[],
                    subquery_timeout_count=0,
                    subquery_timeout_fallback_candidate_count=0,
                    permission_filter_latency_ms=permission_filter_latency_ms,
                    lexical_retrieval_latency_ms=0,
                    indexed_sparse_candidate_count=0,
                    indexed_sparse_retrieval_latency_ms=0,
                    structural_retrieval_latency_ms=0,
                    vector_embedding_latency_ms=0,
                    vector_retrieval_latency_ms=0,
                    expansion_candidate_count=0,
                    in_document_expansion_latency_ms=0,
                    document_evidence_sweep_candidate_count=0,
                    document_evidence_sweep_latency_ms=0,
                    subquery_document_evidence_candidate_count=0,
                    subquery_document_evidence_latency_ms=0,
                    subquery_neighbor_context_candidate_count=0,
                    subquery_neighbor_context_latency_ms=0,
                    document_first_evidence_candidate_count=0,
                    document_first_evidence_latency_ms=0,
                    document_neighbor_context_candidate_count=0,
                    document_neighbor_context_latency_ms=0,
                    evidence_preservation_candidate_count=0,
                    evidence_preservation_selected_count=0,
                    final_coverage_candidate_count=0,
                    final_coverage_selected_count=0,
                    subquery_final_coverage_candidate_count=0,
                    subquery_final_coverage_selected_count=0,
                    fusion_latency_ms=0,
                    rerank_latency_ms=0,
                    search_total_latency_ms=int((perf_counter() - search_started) * 1000),
                    query_plan_candidate_count=query_plan.candidate_count,
                    query_plan_selected=query_plan.selected_candidate.label,
                    query_plan_selection_reason=query_plan.selected_candidate_reason,
                    query_plan_probe_applied=probe_applied,
                    query_plan_probe_latency_ms=probe_latency_ms,
                    query_plan_probe_skipped_reason=probe_skip_reason,
                    permission_probe_early_stop_applied=True,
                    permission_probe_target_hint=permission_probe_early_stop.target_hint,
                    permission_probe_accessible_target_count=permission_probe_early_stop.accessible_target_count,
                    permission_probe_inaccessible_target_count=permission_probe_early_stop.inaccessible_target_count,
                ),
            )
        if not accessible_document_ids:
            return SearchResponse(
                query=payload.query,
                top_k=payload.top_k,
                matched_chunks=[],
                debug=SearchDebugInfo(
                    accessible_document_count=0,
                    lexical_candidate_count=0,
                    vector_candidate_count=0,
                    structural_candidate_count=0,
                    fusion_strategy="min-max weighted sum",
                    pre_rerank_count=0,
                    post_rerank_count=0,
                    rerank_strategy="none",
                    retrieval_query=query_plan.retrieval_query,
                    lexical_queries=query_plan.lexical_queries,
                    query_rewrite_applied=query_plan.rewrite_applied,
                    query_rewrite_strategies=query_plan.applied_strategies,
                    query_rewrite_provider=query_plan.rewrite_provider,
                    query_rewrite_model=query_plan.rewrite_model,
                    query_rewrite_latency_ms=query_plan.rewrite_latency_ms,
                    llm_rewrite_attempted=query_plan.llm_rewrite_attempted,
                    llm_rewrite_skipped_reason=query_plan.llm_rewrite_skipped_reason,
                    llm_rewrite_latency_ms=query_plan.llm_rewrite_latency_ms,
                    query_decomposition_applied=self._query_decomposition_applies(query_plan),
                    subquery_count=len(query_plan.subqueries),
                    subquery_candidate_counts=[],
                    subquery_timeout_count=0,
                    subquery_timeout_fallback_candidate_count=0,
                    permission_filter_latency_ms=permission_filter_latency_ms,
                    search_total_latency_ms=int((perf_counter() - search_started) * 1000),
                    query_plan_candidate_count=query_plan.candidate_count,
                    query_plan_selected=query_plan.selected_candidate.label,
                    query_plan_selection_reason=query_plan.selected_candidate_reason,
                    query_plan_probe_applied=probe_applied,
                    query_plan_probe_latency_ms=probe_latency_ms,
                    query_plan_probe_skipped_reason=probe_skip_reason,
                ),
            )
        candidate_pool = self._candidate_pool_size(payload.top_k)
        lexical_started = perf_counter()
        subquery_candidate_counts: list[dict[str, object]] = []
        subquery_timeout_count = 0
        subquery_timeout_fallback_candidate_count = 0
        if not bool(getattr(self.settings, "retrieval_lexical_enabled", True)) or self._skip_decomposed_lexical_source(query_plan):
            lexical_hits = []
        elif self._subquery_source_retrieval_applies(query_plan):
            lexical_result = self._collect_decomposed_source_hits(
                query_plan.subqueries,
                accessible_document_ids,
                candidate_pool,
                source_name="lexical",
                search_fn=self.retrieval_repository.search_lexical,
            )
            lexical_hits = lexical_result.hits
            subquery_candidate_counts = lexical_result.subquery_candidate_counts
            subquery_timeout_count += lexical_result.timeout_count
            subquery_timeout_fallback_candidate_count += lexical_result.timeout_fallback_candidate_count
        else:
            lexical_hits = self._collect_lexical_hits(query_plan.lexical_queries, accessible_document_ids, candidate_pool)
        lexical_latency_ms = int((perf_counter() - lexical_started) * 1000)
        indexed_sparse_started = perf_counter()
        if self._subquery_source_retrieval_applies(query_plan):
            indexed_sparse_result = self._collect_decomposed_source_hits(
                query_plan.subqueries,
                accessible_document_ids,
                candidate_pool,
                source_name="indexed_sparse",
                search_fn=self.retrieval_repository.search_indexed_sparse,
                source_enabled=bool(self.settings.retrieval_indexed_sparse_enabled),
            )
            indexed_sparse_hits = indexed_sparse_result.hits
            subquery_candidate_counts = self._merge_subquery_candidate_counts(
                subquery_candidate_counts,
                indexed_sparse_result.subquery_candidate_counts,
            )
            subquery_timeout_count += indexed_sparse_result.timeout_count
            subquery_timeout_fallback_candidate_count += indexed_sparse_result.timeout_fallback_candidate_count
        else:
            indexed_sparse_hits = self._collect_indexed_sparse_hits(query_plan.lexical_queries, accessible_document_ids, candidate_pool)
        indexed_sparse_latency_ms = int((perf_counter() - indexed_sparse_started) * 1000)
        structural_started = perf_counter()
        structural_skip_reason = self._structural_retrieval_skip_reason(query_plan, target_document_title=target_document_title)
        structural_timeout = False
        if structural_skip_reason is not None:
            structural_hits = []
        elif self._subquery_source_retrieval_applies(query_plan):
            structural_result = self._collect_decomposed_source_hits(
                query_plan.subqueries,
                accessible_document_ids,
                candidate_pool,
                source_name="structural",
                search_fn=self.retrieval_repository.search_structural,
                source_enabled=bool(self.settings.retrieval_structural_enabled),
            )
            structural_hits = structural_result.hits
            subquery_candidate_counts = self._merge_subquery_candidate_counts(
                subquery_candidate_counts,
                structural_result.subquery_candidate_counts,
            )
            subquery_timeout_count += structural_result.timeout_count
        else:
            structural_hits = self._collect_structural_hits(query_plan.lexical_queries, accessible_document_ids, candidate_pool)
            structural_timeout = self._last_structural_timeout
        structural_latency_ms = int((perf_counter() - structural_started) * 1000)
        vector_enabled = bool(self.settings.retrieval_vector_enabled)
        skip_vector_retrieval = self._skip_vector_when_keyword_hits_exist(
            query_plan,
            [*lexical_hits, *indexed_sparse_hits, *structural_hits],
        )
        if not vector_enabled:
            vector_skip_reason = "disabled"
        elif skip_vector_retrieval:
            vector_skip_reason = "keyword_hits_sufficient"
        else:
            vector_skip_reason = None
        vector_embedding_started = perf_counter()
        query_embedding = (
            self.embedding_provider.embed_texts([query_plan.retrieval_query])[0]
            if vector_enabled and not skip_vector_retrieval
            else []
        )
        vector_embedding_latency_ms = int((perf_counter() - vector_embedding_started) * 1000)
        vector_retrieval_started = perf_counter()
        vector_hits = (
            self.retrieval_repository.search_vector(query_embedding, accessible_document_ids, candidate_pool)
            if vector_enabled and query_embedding and not skip_vector_retrieval
            else []
        )
        vector_retrieval_latency_ms = int((perf_counter() - vector_retrieval_started) * 1000)
        fusion_started = perf_counter()
        fused_candidates = self._fuse_hits(
            lexical_hits,
            vector_hits,
            structural_hits=structural_hits,
            indexed_sparse_hits=indexed_sparse_hits,
        )
        fusion_latency_ms = int((perf_counter() - fusion_started) * 1000)
        expansion_started = perf_counter()
        expansion_hits = self._collect_in_document_expansion(query_plan.retrieval_query, fused_candidates.values())
        in_document_expansion_latency_ms = int((perf_counter() - expansion_started) * 1000)
        if expansion_hits:
            fusion_started = perf_counter()
            fused_candidates = self._fuse_hits(
                lexical_hits,
                vector_hits,
                structural_hits=structural_hits,
                indexed_sparse_hits=indexed_sparse_hits,
                expansion_hits=expansion_hits,
            )
            fusion_latency_ms += int((perf_counter() - fusion_started) * 1000)
        sweep_started = perf_counter()
        fused_candidates_for_sweep = list(fused_candidates.values())
        document_sweep_skip_reason = self._document_evidence_sweep_skip_reason(
            query_plan,
            fused_candidates_for_sweep,
            search_started=search_started,
        )
        document_sweep_hits = (
            []
            if document_sweep_skip_reason is not None
            else self._collect_document_evidence_sweep(query_plan.retrieval_query, fused_candidates_for_sweep)
        )
        document_evidence_sweep_latency_ms = int((perf_counter() - sweep_started) * 1000)
        if document_sweep_hits:
            fusion_started = perf_counter()
            fused_candidates = self._fuse_hits(
                lexical_hits,
                vector_hits,
                structural_hits=structural_hits,
                indexed_sparse_hits=indexed_sparse_hits,
                expansion_hits=expansion_hits,
                document_sweep_hits=document_sweep_hits,
            )
            fusion_latency_ms += int((perf_counter() - fusion_started) * 1000)
        subquery_document_started = perf_counter()
        subquery_document_hits = self._collect_subquery_document_evidence_hits(query_plan, fused_candidates.values())
        subquery_document_evidence_latency_ms = int((perf_counter() - subquery_document_started) * 1000)
        if subquery_document_hits:
            fusion_started = perf_counter()
            fused_candidates = self._fuse_hits(
                lexical_hits,
                vector_hits,
                structural_hits=structural_hits,
                indexed_sparse_hits=indexed_sparse_hits,
                expansion_hits=expansion_hits,
                document_sweep_hits=document_sweep_hits,
                subquery_document_hits=subquery_document_hits,
            )
            fusion_latency_ms += int((perf_counter() - fusion_started) * 1000)
        subquery_neighbor_started = perf_counter()
        subquery_neighbor_hits = self._collect_subquery_neighbor_context_hits(
            query_plan,
            [*indexed_sparse_hits, *subquery_document_hits],
        )
        subquery_neighbor_context_latency_ms = int((perf_counter() - subquery_neighbor_started) * 1000)
        if subquery_neighbor_hits:
            fusion_started = perf_counter()
            fused_candidates = self._fuse_hits(
                lexical_hits,
                vector_hits,
                structural_hits=structural_hits,
                indexed_sparse_hits=indexed_sparse_hits,
                expansion_hits=expansion_hits,
                document_sweep_hits=document_sweep_hits,
                subquery_document_hits=subquery_document_hits,
                subquery_neighbor_hits=subquery_neighbor_hits,
            )
            fusion_latency_ms += int((perf_counter() - fusion_started) * 1000)
        document_first_started = perf_counter()
        document_first_hits = self._collect_document_first_evidence_hits(query_plan.retrieval_query, fused_candidates.values())
        document_first_evidence_latency_ms = int((perf_counter() - document_first_started) * 1000)
        if document_first_hits:
            fusion_started = perf_counter()
            fused_candidates = self._fuse_hits(
                lexical_hits,
                vector_hits,
                structural_hits=structural_hits,
                indexed_sparse_hits=indexed_sparse_hits,
                expansion_hits=expansion_hits,
                document_sweep_hits=document_sweep_hits,
                subquery_document_hits=subquery_document_hits,
                subquery_neighbor_hits=subquery_neighbor_hits,
                document_first_hits=document_first_hits,
            )
            fusion_latency_ms += int((perf_counter() - fusion_started) * 1000)
        neighbor_started = perf_counter()
        neighbor_hits = self._collect_document_neighbor_context_hits(fused_candidates.values())
        document_neighbor_context_latency_ms = int((perf_counter() - neighbor_started) * 1000)
        if neighbor_hits:
            fusion_started = perf_counter()
            fused_candidates = self._fuse_hits(
                lexical_hits,
                vector_hits,
                structural_hits=structural_hits,
                indexed_sparse_hits=indexed_sparse_hits,
                expansion_hits=expansion_hits,
                document_sweep_hits=document_sweep_hits,
                subquery_document_hits=subquery_document_hits,
                subquery_neighbor_hits=subquery_neighbor_hits,
                document_first_hits=document_first_hits,
                neighbor_hits=neighbor_hits,
            )
            fusion_latency_ms += int((perf_counter() - fusion_started) * 1000)
        target_document_id = scoped_document_ids[0] if scoped_document_ids and len(scoped_document_ids) == 1 else None
        rerank_started = perf_counter()
        reranked = self._rerank_or_rank_candidates(
            payload.query,
            query_plan.retrieval_query,
            list(fused_candidates.values()),
            top_k=payload.top_k,
            candidate_pool=candidate_pool,
            target_document_id=target_document_id,
        )
        preservation_candidates = self._collect_evidence_preservation_candidates(fused_candidates.values())
        base_final_candidates = self._select_final_candidates(reranked.candidates, payload.top_k)
        coverage_candidates = self._collect_final_coverage_candidates(
            payload.query,
            reranked.candidates,
            base_final_candidates,
            payload.top_k,
        )
        subquery_coverage_candidates = self._collect_subquery_final_coverage_candidates(
            query_plan,
            reranked.candidates,
            base_final_candidates,
            payload.top_k,
        )
        final_candidates = self._select_final_candidates(
            reranked.candidates,
            payload.top_k,
            query_plan=query_plan,
            preservation_candidates=preservation_candidates,
            coverage_candidates=coverage_candidates,
            subquery_coverage_candidates=subquery_coverage_candidates,
        )
        base_final_ids = {item.candidate.chunk_id for item in base_final_candidates}
        coverage_ids = {item.candidate.chunk_id for item in coverage_candidates}
        subquery_coverage_ids = {item.candidate.chunk_id for item in subquery_coverage_candidates}
        preservation_ids = {item.candidate.chunk_id for item in preservation_candidates}
        selected_preservation_count = sum(
            1
            for item in final_candidates
            if item.candidate.chunk_id in preservation_ids and item.candidate.chunk_id not in base_final_ids
        )
        selected_coverage_count = sum(
            1
            for item in final_candidates
            if item.candidate.chunk_id in coverage_ids and item.candidate.chunk_id not in base_final_ids
        )
        selected_subquery_coverage_count = sum(
            1
            for item in final_candidates
            if item.candidate.chunk_id in subquery_coverage_ids and item.candidate.chunk_id not in base_final_ids
        )
        rerank_latency_ms = int((perf_counter() - rerank_started) * 1000)
        search_total_latency_ms = int((perf_counter() - search_started) * 1000)

        return SearchResponse(
            query=payload.query,
            top_k=payload.top_k,
            matched_chunks=[self._to_schema(candidate) for candidate in final_candidates],
            debug=SearchDebugInfo(
                accessible_document_count=len(accessible_document_ids),
                lexical_candidate_count=len(lexical_hits),
                vector_candidate_count=len(vector_hits),
                structural_candidate_count=len(structural_hits),
                fusion_strategy=self._fusion_strategy_name(),
                pre_rerank_count=reranked.pre_rerank_count,
                post_rerank_count=len(final_candidates),
                rerank_strategy=reranked.strategy,
                retrieval_query=query_plan.retrieval_query,
                lexical_queries=query_plan.lexical_queries,
                query_rewrite_applied=query_plan.rewrite_applied,
                query_rewrite_strategies=query_plan.applied_strategies,
                query_rewrite_provider=query_plan.rewrite_provider,
                query_rewrite_model=query_plan.rewrite_model,
                query_rewrite_latency_ms=query_plan.rewrite_latency_ms,
                llm_rewrite_attempted=query_plan.llm_rewrite_attempted,
                llm_rewrite_skipped_reason=query_plan.llm_rewrite_skipped_reason,
                llm_rewrite_latency_ms=query_plan.llm_rewrite_latency_ms,
                query_decomposition_applied=self._query_decomposition_applies(query_plan),
                subquery_count=len(query_plan.subqueries),
                subquery_candidate_counts=subquery_candidate_counts,
                subquery_timeout_count=subquery_timeout_count,
                subquery_timeout_fallback_candidate_count=subquery_timeout_fallback_candidate_count,
                permission_filter_latency_ms=permission_filter_latency_ms,
                lexical_retrieval_latency_ms=lexical_latency_ms,
                indexed_sparse_candidate_count=len(indexed_sparse_hits),
                indexed_sparse_retrieval_latency_ms=indexed_sparse_latency_ms,
                structural_retrieval_latency_ms=structural_latency_ms,
                structural_retrieval_skipped=structural_skip_reason is not None,
                structural_retrieval_skip_reason=structural_skip_reason,
                structural_retrieval_timeout=structural_timeout,
                vector_embedding_latency_ms=vector_embedding_latency_ms,
                vector_retrieval_latency_ms=vector_retrieval_latency_ms,
                vector_retrieval_skipped=vector_skip_reason is not None,
                vector_retrieval_skip_reason=vector_skip_reason,
                vector_retrieval_timeout=False,
                expansion_candidate_count=len(expansion_hits),
                in_document_expansion_latency_ms=in_document_expansion_latency_ms,
                document_evidence_sweep_candidate_count=len(document_sweep_hits),
                document_evidence_sweep_latency_ms=document_evidence_sweep_latency_ms,
                document_evidence_sweep_skipped=document_sweep_skip_reason is not None,
                document_evidence_sweep_skip_reason=document_sweep_skip_reason,
                subquery_document_evidence_candidate_count=len(subquery_document_hits),
                subquery_document_evidence_latency_ms=subquery_document_evidence_latency_ms,
                subquery_neighbor_context_candidate_count=len(subquery_neighbor_hits),
                subquery_neighbor_context_latency_ms=subquery_neighbor_context_latency_ms,
                document_first_evidence_candidate_count=len(document_first_hits),
                document_first_evidence_latency_ms=document_first_evidence_latency_ms,
                document_neighbor_context_candidate_count=len(neighbor_hits),
                document_neighbor_context_latency_ms=document_neighbor_context_latency_ms,
                evidence_preservation_candidate_count=len(preservation_candidates),
                evidence_preservation_selected_count=selected_preservation_count,
                final_coverage_candidate_count=len(coverage_candidates),
                final_coverage_selected_count=selected_coverage_count,
                subquery_final_coverage_candidate_count=len(subquery_coverage_candidates),
                subquery_final_coverage_selected_count=selected_subquery_coverage_count,
                fusion_latency_ms=fusion_latency_ms,
                rerank_latency_ms=rerank_latency_ms,
                search_total_latency_ms=search_total_latency_ms,
                query_plan_candidate_count=query_plan.candidate_count,
                query_plan_selected=query_plan.selected_candidate.label,
                query_plan_selection_reason=query_plan.selected_candidate_reason,
                query_plan_probe_applied=probe_applied,
                query_plan_probe_latency_ms=probe_latency_ms,
                query_plan_probe_skipped_reason=probe_skip_reason,
            ),
        )

    def _skip_vector_when_keyword_hits_exist(
        self,
        query_plan: QueryOptimizationPlan,
        keyword_hits: list[RetrievalCandidate],
    ) -> bool:
        if not bool(getattr(self.settings, "retrieval_vector_skip_when_keyword_hits_enabled", False)):
            return False
        if self._subquery_source_retrieval_applies(query_plan):
            return False
        min_hits = max(1, int(getattr(self.settings, "retrieval_vector_skip_min_keyword_hits", 1) or 1))
        unique_hit_count = len({hit.chunk_id for hit in keyword_hits})
        return unique_hit_count >= min_hits

    def _query_plan_probe_skip_reason(
        self,
        query_plan: QueryOptimizationPlan,
        *,
        accessible_document_ids: list[UUID],
        target_document_title: str | None,
    ) -> str | None:
        if not bool(getattr(self.settings, "retrieval_query_plan_probe_enabled", True)):
            return "disabled"
        if not bool(getattr(self.settings, "retrieval_lexical_enabled", True)):
            return "lexical_disabled"
        if not accessible_document_ids:
            return "no_accessible_documents"
        if query_plan.candidate_count <= 1:
            return "single_candidate"
        max_candidates = max(0, int(getattr(self.settings, "retrieval_query_plan_probe_max_candidates", 0) or 0))
        if max_candidates > 0 and query_plan.candidate_count > max_candidates:
            return "candidate_limit"
        if (
            bool(getattr(self.settings, "retrieval_query_plan_probe_simple_skip_enabled", True))
            and not target_document_title
            and self._is_simple_retrieval_query(query_plan)
        ):
            return "simple_query"
        return None

    def _structural_retrieval_skip_reason(
        self,
        query_plan: QueryOptimizationPlan,
        *,
        target_document_title: str | None,
    ) -> str | None:
        if not bool(getattr(self.settings, "retrieval_structural_enabled", True)):
            return "disabled"
        if not bool(getattr(self.settings, "retrieval_structural_simple_query_skip_enabled", True)):
            return None
        if target_document_title or self._has_structural_anchor(query_plan):
            return None
        if self._is_simple_retrieval_query(query_plan):
            return "simple_query_without_structural_anchor"
        return None

    def _document_evidence_sweep_skip_reason(
        self,
        query_plan: QueryOptimizationPlan,
        candidates: Iterable[RerankCandidate],
        *,
        search_started: float,
    ) -> str | None:
        if not bool(getattr(self.settings, "retrieval_document_evidence_sweep_enabled", False)):
            return "disabled"
        candidate_list = list(candidates)
        if not candidate_list:
            return "no_seed_candidates"
        if self._is_simple_retrieval_query(query_plan):
            return "simple_query"
        min_remaining = max(
            0,
            int(getattr(self.settings, "retrieval_document_evidence_sweep_min_remaining_budget_ms", 0) or 0),
        )
        if min_remaining > 0 and self._remaining_latency_budget_ms(search_started, query_plan) < min_remaining:
            return "latency_budget_exhausted"
        return None

    def _remaining_latency_budget_ms(self, search_started: float, query_plan: QueryOptimizationPlan) -> int:
        budget_ms = self._latency_budget_ms(query_plan)
        if budget_ms <= 0:
            return 2_147_483_647
        elapsed_ms = int((perf_counter() - search_started) * 1000)
        return max(0, budget_ms - elapsed_ms)

    def _latency_budget_ms(self, query_plan: QueryOptimizationPlan) -> int:
        setting_name = "retrieval_latency_budget_simple_ms" if self._is_simple_retrieval_query(query_plan) else "retrieval_latency_budget_complex_ms"
        return max(0, int(getattr(self.settings, setting_name, 0) or 0))

    def _is_simple_retrieval_query(self, query_plan: QueryOptimizationPlan) -> bool:
        if query_plan.query_decomposition_applied:
            return False
        if query_plan.subqueries:
            return False
        if self._has_structural_anchor(query_plan):
            return False
        if "title_anchor" in query_plan.applied_strategies:
            return False
        text = self._query_plan_text(query_plan)
        original_query = query_plan.original_query
        original_compact = self._compact_match_text(original_query)
        if len(original_compact) > 96:
            return False
        if any(marker in text for marker in ('“', '”', '"', '比较', '分别', '各自', '同时核对', '两个事项')):
            return False
        if re.search(r"(?:和|及|与|以及|、).{0,18}(?:是什么|哪些|要求|结论|清单|材料|流程|条件)", original_query):
            return False
        return True

    def _has_structural_anchor(self, query_plan: QueryOptimizationPlan) -> bool:
        text = self._query_plan_text(query_plan)
        if re.search(r"第[一二三四五六七八九十百千万零〇两0-9]+[章节条款项]", text):
            return True
        structural_markers = ("条款全称", "章节", "小节", "章标题", "节标题", "条款标题", "款项")
        return any(marker in text for marker in structural_markers)

    @staticmethod
    def _query_plan_text(query_plan: QueryOptimizationPlan) -> str:
        return " ".join(
            item
            for item in [query_plan.original_query, query_plan.retrieval_query, *query_plan.lexical_queries]
            if item
        )

    def _set_local_statement_timeout(self, timeout_ms: int) -> bool:
        if timeout_ms <= 0 or not self._is_postgresql_session():
            return False
        self.session.execute(text(f"SET LOCAL statement_timeout = {int(timeout_ms)}"))
        return True

    def _clear_local_statement_timeout(self) -> None:
        if not self._is_postgresql_session():
            return
        try:
            self.session.execute(text("SET LOCAL statement_timeout = 0"))
        except Exception:
            self.session.rollback()

    def _is_postgresql_session(self) -> bool:
        bind = getattr(self.session, "bind", None) or self.session.get_bind()
        dialect = getattr(bind, "dialect", None)
        return getattr(dialect, "name", "") == "postgresql"

    def _permission_probe_early_stop(
        self,
        query: str,
        accessible_document_ids: list[UUID],
    ) -> PermissionProbeEarlyStopResult | None:
        target_hint = self._extract_permission_probe_target_hint(query)
        if not target_hint:
            return None

        accessible_matches = self._document_ids_matching_org_hint(accessible_document_ids, target_hint)
        if accessible_matches:
            return None

        corpus_matches = self._document_ids_matching_org_hint(None, target_hint)
        accessible_set = set(accessible_document_ids)
        inaccessible_count = sum(1 for item in corpus_matches if item not in accessible_set)
        if inaccessible_count <= 0:
            return None

        return PermissionProbeEarlyStopResult(
            target_hint=target_hint,
            accessible_target_count=len(accessible_matches),
            inaccessible_target_count=inaccessible_count,
        )

    @classmethod
    def _extract_permission_probe_target_hint(cls, query: str) -> str | None:
        normalized = " ".join(str(query or "").split())
        if not normalized:
            return None
        if not any(marker in normalized for marker in ("普通查看用户", "普通用户", "查看用户")):
            return None
        if not any(marker in normalized for marker in ("能否", "是否可以", "能不能", "可否")):
            return None
        if not any(marker in normalized for marker in ("受限材料", "受限文件", "受限文档")):
            return None

        match = PERMISSION_PROBE_TARGET_PATTERN.search(normalized)
        if not match:
            return None
        target_hint = match.group("target")
        target_hint = re.sub(r"^(?:作为)?(?:普通查看用户|普通用户|查看用户)[，,。；;：:\s]*", "", target_hint)
        target_hint = target_hint.strip(" ，,。；;：:")
        return target_hint if len(cls._compact_match_text(target_hint)) >= 4 else None

    def _select_best_query_plan(
        self,
        query_plan: QueryOptimizationPlan,
        *,
        accessible_document_ids: list[UUID],
        target_document_title: str | None,
    ) -> bool:
        probes = [
            self._probe_query_plan(
                candidate,
                accessible_document_ids=accessible_document_ids,
                target_document_title=target_document_title,
            )
            for candidate in query_plan.candidates
        ]
        if len(probes) <= 1:
            return False

        hit_counts = [probe.hit_count for probe in probes]
        top_scores = [probe.top_score for probe in probes]
        avg_scores = [probe.average_score for probe in probes]

        for probe in probes:
            probe.final_score = (
                0.35 * self._relative_score(probe.top_score, top_scores)
                + 0.2 * self._relative_score(probe.average_score, avg_scores)
                + 0.2 * self._relative_score(probe.hit_count, hit_counts)
                + 0.15 * probe.concentration
                + 0.1 * probe.title_match
            )
            if "title_anchor" in probe.candidate.applied_strategies and probe.hit_count > 0:
                probe.final_score += 0.08

        selected = max(
            probes,
            key=lambda item: (
                item.final_score,
                item.title_match,
                item.hit_count,
                item.average_score,
                item.top_score,
            ),
        )
        query_plan.select_candidate(
            selected.candidate.key,
            reason=self._build_selection_reason(selected, probes, target_document_title=target_document_title),
        )
        return True

    def _probe_query_plan(
        self,
        candidate: QueryPlanCandidate,
        *,
        accessible_document_ids: list[UUID],
        target_document_title: str | None,
    ) -> QueryPlanProbeResult:
        timeout_ms = max(0, int(getattr(self.settings, "retrieval_query_plan_probe_timeout_ms", 0) or 0))
        timeout_applied = self._set_local_statement_timeout(timeout_ms)
        try:
            hits = self._collect_lexical_hits(candidate.lexical_queries[:3], accessible_document_ids, PLAN_PROBE_LIMIT)
        finally:
            if timeout_applied:
                self._clear_local_statement_timeout()
        top_scores = [hit.lexical_score or 0.0 for hit in hits[:3]]
        top_documents = [hit.document_title for hit in hits[:3]]
        concentration = 0.0
        if top_documents:
            most_common = max(top_documents.count(title) for title in set(top_documents))
            concentration = most_common / len(top_documents)
        title_match = 0.0
        if target_document_title:
            normalized_title = target_document_title.casefold().strip()
            title_match = 1.0 if any(item.casefold().strip() == normalized_title for item in top_documents) else 0.0
        return QueryPlanProbeResult(
            candidate=candidate,
            hits=hits,
            hit_count=len(hits),
            top_score=top_scores[0] if top_scores else 0.0,
            average_score=mean(top_scores) if top_scores else 0.0,
            concentration=concentration,
            title_match=title_match,
        )

    @staticmethod
    def _relative_score(value: float, values: list[float]) -> float:
        if not values:
            return 0.0
        maximum = max(values)
        minimum = min(values)
        if maximum == minimum:
            return 1.0 if value > 0 else 0.0
        return (value - minimum) / (maximum - minimum)

    def _build_selection_reason(
        self,
        selected: QueryPlanProbeResult,
        all_probes: list[QueryPlanProbeResult],
        *,
        target_document_title: str | None,
    ) -> str:
        reasons: list[str] = []
        max_hits = max((probe.hit_count for probe in all_probes), default=0)
        max_top_score = max((probe.top_score for probe in all_probes), default=0.0)
        if target_document_title and selected.title_match > 0:
            reasons.append("标题命中更集中")
        if selected.hit_count == max_hits and selected.hit_count > 0:
            reasons.append(f"试探召回 {selected.hit_count} 个候选")
        if selected.top_score == max_top_score and selected.top_score > 0:
            reasons.append("关键词检索信号更强")
        if not reasons:
            reasons.append("默认采用早期信号更稳的方案")
        return "；".join(reasons[:2])

    @staticmethod
    def _build_rerank_query(original_query: str, retrieval_query: str | None) -> str:
        cleaned_original = " ".join(str(original_query or "").split())
        cleaned_retrieval = " ".join(str(retrieval_query or "").split())
        if not cleaned_retrieval or cleaned_retrieval.casefold() == cleaned_original.casefold():
            return cleaned_original
        return f"{cleaned_original} {cleaned_retrieval}".strip()

    def _query_decomposition_applies(self, query_plan: QueryOptimizationPlan) -> bool:
        if not bool(getattr(self.settings, "retrieval_query_decomposition_enabled", True)):
            return False
        return query_plan.query_decomposition_applied

    def _subquery_source_retrieval_applies(self, query_plan: QueryOptimizationPlan) -> bool:
        if not bool(getattr(self.settings, "retrieval_query_decomposition_enabled", True)):
            return False
        return bool(query_plan.subqueries)

    def _skip_decomposed_lexical_source(self, query_plan: QueryOptimizationPlan) -> bool:
        if not bool(getattr(self.settings, "retrieval_indexed_sparse_enabled", False)):
            return False
        if not self._subquery_source_retrieval_applies(query_plan):
            return False
        case_shapes = {subquery.case_shape for subquery in query_plan.subqueries}
        if case_shapes == {"cross_document_comparison"}:
            return bool(
                getattr(
                    self.settings,
                    "retrieval_query_decomposition_cross_document_skip_lexical_when_indexed_sparse_enabled",
                    True,
                )
            )
        if case_shapes == {"same_document_two_matters"}:
            return bool(
                getattr(
                    self.settings,
                    "retrieval_query_decomposition_same_document_skip_lexical_when_indexed_sparse_enabled",
                    True,
                )
            )
        return False

    def _collect_decomposed_source_hits(
        self,
        subqueries: list[QuerySubquery],
        accessible_document_ids: list[UUID],
        candidate_pool: int,
        *,
        source_name: str,
        search_fn: Callable[[str, list[UUID], int], list[RetrievalCandidate]],
        source_enabled: bool = True,
    ) -> DecomposedSourceCollectionResult:
        if not source_enabled:
            return DecomposedSourceCollectionResult(
                hits=[],
                subquery_candidate_counts=[
                    self._subquery_candidate_count_payload(subquery, index, source_name, 0, timeout=False)
                    for index, subquery in enumerate(subqueries)
                ],
                timeout_count=0,
                timeout_fallback_candidate_count=0,
            )

        total_limit = self._decomposed_source_total_limit(source_name, candidate_pool)
        if total_limit <= 0:
            return DecomposedSourceCollectionResult(
                hits=[],
                subquery_candidate_counts=[],
                timeout_count=0,
                timeout_fallback_candidate_count=0,
            )

        per_subquery_limit = self._decomposed_source_per_subquery_limit(total_limit, len(subqueries))
        subquery_hits: list[list[RetrievalCandidate]] = []
        counts: list[dict[str, object]] = []
        timeout_count = 0
        timeout_fallback_candidate_count = 0
        for index, subquery in enumerate(subqueries):
            timeout = False
            fallback_candidate_count = 0
            source_query_text = self._subquery_source_query_text(subquery, source_name)
            scoped_document_ids = self._source_document_ids_for_subquery(subquery, accessible_document_ids, source_name)
            source_limit = self._decomposed_source_limit_for_subquery(source_name, subquery, per_subquery_limit)
            try:
                table_lookup_hits = self._collect_table_lookup_source_hits(source_name, subquery, scoped_document_ids, source_limit)
                if table_lookup_hits:
                    hits = table_lookup_hits
                elif exact_anchor_hits := self._collect_exact_anchor_source_hits(
                    source_name,
                    subquery,
                    scoped_document_ids,
                    source_limit,
                ):
                    hits = exact_anchor_hits
                elif self._use_cross_document_python_sparse_source(source_name, subquery):
                    hits = self.retrieval_repository.search_python_sparse(source_query_text, scoped_document_ids, source_limit)
                else:
                    hits = search_fn(source_query_text, scoped_document_ids, source_limit)
            except Exception as exc:
                if not self._is_timeout_error(exc):
                    raise
                self.session.rollback()
                timeout = True
                timeout_count += 1
                hits = self._collect_decomposed_source_timeout_fallback(
                    source_name,
                    source_query_text,
                    scoped_document_ids,
                    source_limit,
                )
                fallback_candidate_count = len(hits)
                timeout_fallback_candidate_count += fallback_candidate_count
            subquery_hits.append(hits)
            counts.append(
                self._subquery_candidate_count_payload(
                    subquery,
                    index,
                    source_name,
                    len(hits),
                    source_query_text=source_query_text,
                    timeout=timeout,
                    timeout_fallback_candidate_count=fallback_candidate_count,
                )
            )

        return DecomposedSourceCollectionResult(
            hits=self._merge_decomposed_source_hits(subquery_hits, total_limit),
            subquery_candidate_counts=counts,
            timeout_count=timeout_count,
            timeout_fallback_candidate_count=timeout_fallback_candidate_count,
        )

    def _source_document_ids_for_subquery(
        self,
        subquery: QuerySubquery,
        accessible_document_ids: list[UUID],
        source_name: str,
    ) -> list[UUID]:
        if source_name not in {"indexed_sparse", "lexical", "structural"} or not subquery.org_hint:
            return accessible_document_ids
        matched_ids = self._document_ids_matching_title_hint(accessible_document_ids, subquery.org_hint)
        if not matched_ids:
            matched_ids = self._document_ids_matching_org_hint(accessible_document_ids, subquery.org_hint)
        return matched_ids or accessible_document_ids

    def _decomposed_source_limit_for_subquery(
        self,
        source_name: str,
        subquery: QuerySubquery,
        default_limit: int,
    ) -> int:
        if source_name not in {"indexed_sparse", "lexical"}:
            return default_limit
        if (
            source_name == "lexical"
            and subquery.case_shape == "cross_document_comparison"
            and bool(getattr(self.settings, "retrieval_indexed_sparse_enabled", False))
        ):
            cap = int(getattr(self.settings, "retrieval_query_decomposition_cross_document_lexical_candidate_cap", 0) or 0)
            if cap > 0:
                return max(1, min(default_limit, cap))
        if source_name == "indexed_sparse" and subquery.case_shape == "cross_document_comparison":
            cap = int(getattr(self.settings, "retrieval_query_decomposition_cross_document_indexed_sparse_candidate_cap", 0) or 0)
            if cap > 0:
                return max(1, min(default_limit, cap))
        if source_name == "indexed_sparse" and subquery.case_shape == "table_structured_lookup":
            cap = int(getattr(self.settings, "retrieval_table_lookup_source_candidate_cap", 0) or 0)
            if cap > 0:
                return max(1, min(default_limit, cap))
        if subquery.case_shape not in {"single_evidence_anchor", "table_structured_lookup"}:
            return default_limit
        if not subquery.org_hint or len(self._compact_match_text(subquery.evidence_hint)) < 6:
            return default_limit

        cap = int(getattr(self.settings, "retrieval_query_decomposition_single_anchor_candidate_cap", 0) or 0)
        if cap <= 0:
            return default_limit
        return max(1, min(default_limit, cap))

    def _use_cross_document_python_sparse_source(self, source_name: str, subquery: QuerySubquery) -> bool:
        return bool(
            source_name == "indexed_sparse"
            and subquery.case_shape == "cross_document_comparison"
            and getattr(self.settings, "retrieval_query_decomposition_cross_document_python_sparse_enabled", True)
        )

    def _collect_table_lookup_source_hits(
        self,
        source_name: str,
        subquery: QuerySubquery,
        scoped_document_ids: list[UUID],
        source_limit: int,
    ) -> list[RetrievalCandidate]:
        if source_name != "indexed_sparse" or subquery.case_shape != "table_structured_lookup":
            return []
        if not bool(getattr(self.settings, "retrieval_table_lookup_source_enabled", True)):
            return []
        lookup_pairs = self._subquery_table_lookup_pairs(subquery)
        if not lookup_pairs:
            return []
        return self.retrieval_repository.search_table_lookup_pairs(lookup_pairs, scoped_document_ids, source_limit)

    def _collect_exact_anchor_source_hits(
        self,
        source_name: str,
        subquery: QuerySubquery,
        scoped_document_ids: list[UUID],
        source_limit: int,
    ) -> list[RetrievalCandidate]:
        if source_name not in {"indexed_sparse", "lexical"}:
            return []
        if subquery.case_shape in {"single_category_anchor", "table_structured_lookup"}:
            return []
        if not bool(getattr(self.settings, "retrieval_exact_anchor_source_enabled", True)):
            return []
        query_text = subquery.evidence_hint or subquery.query_text
        return self.retrieval_repository.search_exact_text_in_documents(query_text, scoped_document_ids, source_limit)

    def _document_ids_matching_title_hint(self, accessible_document_ids: list[UUID], title_hint: str) -> list[UUID]:
        cleaned_hint = " ".join(str(title_hint or "").split())
        if not cleaned_hint:
            return []
        rows = (
            self.session.query(Document.id)
            .filter(Document.id.in_(accessible_document_ids))
            .filter(Document.title.ilike(f"%{cleaned_hint}%"))
            .all()
        )
        return [row[0] for row in rows]

    def _document_ids_matching_org_hint(self, document_ids: list[UUID] | None, org_hint: str) -> list[UUID]:
        cleaned_hint = " ".join(str(org_hint or "").split())
        if not cleaned_hint:
            return []

        query = self.session.query(Document.id, Document.title)
        if document_ids is not None:
            if not document_ids:
                return []
            query = query.filter(Document.id.in_(document_ids))

        matches: list[UUID] = []
        for document_id, title in query.all():
            if self._document_matches_org_hint(title, cleaned_hint):
                matches.append(document_id)
        return matches

    def _collect_decomposed_source_timeout_fallback(
        self,
        source_name: str,
        query_text: str,
        accessible_document_ids: list[UUID],
        limit: int,
    ) -> list[RetrievalCandidate]:
        if source_name != "indexed_sparse":
            return []
        try:
            return self.retrieval_repository.search_indexed_sparse_timeout_fallback(
                query_text,
                accessible_document_ids,
                limit,
            )
        except Exception as exc:
            if not self._is_timeout_error(exc):
                raise
            self.session.rollback()
            return []

    def _decomposed_source_total_limit(self, source_name: str, candidate_pool: int) -> int:
        if source_name == "indexed_sparse":
            multiplier = max(1, int(getattr(self.settings, "retrieval_indexed_sparse_candidate_multiplier", 1) or 1))
            return max(candidate_pool, min(candidate_pool * multiplier, self.settings.retrieval_candidate_max))
        return candidate_pool

    def _decomposed_source_per_subquery_limit(self, total_limit: int, subquery_count: int) -> int:
        if subquery_count <= 0:
            return max(0, total_limit)
        minimum = max(1, int(getattr(self.settings, "retrieval_query_decomposition_min_subquery_candidates", 4) or 4))
        balanced = (max(1, total_limit) + subquery_count - 1) // subquery_count
        return min(max(1, total_limit), max(minimum, balanced))

    def _merge_decomposed_source_hits(
        self,
        subquery_hits: list[list[RetrievalCandidate]],
        total_limit: int,
    ) -> list[RetrievalCandidate]:
        if total_limit <= 0:
            return []

        best_by_chunk: dict[UUID, RetrievalCandidate] = {}
        for hits in subquery_hits:
            for candidate in hits:
                existing = best_by_chunk.get(candidate.chunk_id)
                if existing is None or self._candidate_source_score(candidate) > self._candidate_source_score(existing):
                    best_by_chunk[candidate.chunk_id] = candidate

        coverage_floor = max(
            1,
            int(getattr(self.settings, "retrieval_query_decomposition_min_subquery_candidates", 4) or 4),
        )
        selected: list[RetrievalCandidate] = []
        selected_ids: set[UUID] = set()

        def add_candidate(candidate: RetrievalCandidate) -> bool:
            if candidate.chunk_id in selected_ids:
                return False
            selected_ids.add(candidate.chunk_id)
            selected.append(best_by_chunk.get(candidate.chunk_id, candidate))
            return True

        for hits in subquery_hits:
            if len(selected) >= total_limit:
                break
            kept_for_subquery = 0
            seen_documents: set[UUID] = set()
            for candidate in hits:
                if candidate.document_id in seen_documents:
                    continue
                if add_candidate(candidate):
                    kept_for_subquery += 1
                    seen_documents.add(candidate.document_id)
                if kept_for_subquery >= coverage_floor or len(selected) >= total_limit:
                    break
            if kept_for_subquery >= coverage_floor or len(selected) >= total_limit:
                continue
            for candidate in hits:
                if add_candidate(candidate):
                    kept_for_subquery += 1
                if kept_for_subquery >= coverage_floor or len(selected) >= total_limit:
                    break

        globally_ranked = sorted(
            best_by_chunk.values(),
            key=lambda item: (self._candidate_source_score(item), -item.chunk_index),
            reverse=True,
        )
        for candidate in globally_ranked:
            if len(selected) >= total_limit:
                break
            add_candidate(candidate)
        return selected[:total_limit]

    def _subquery_source_query_text(self, subquery: QuerySubquery, source_name: str) -> str:
        if source_name in {"indexed_sparse", "lexical", "structural"}:
            if subquery.case_shape == "single_category_anchor":
                return subquery.query_text
            if source_name in {"indexed_sparse", "lexical"} and subquery.case_shape == "cross_document_comparison":
                return self._cross_document_source_query_text(subquery.evidence_hint or subquery.query_text)
            return subquery.evidence_hint or subquery.query_text
        return subquery.query_text

    def _cross_document_source_query_text(self, query_text: str) -> str:
        max_terms = max(
            0,
            int(getattr(self.settings, "retrieval_query_decomposition_cross_document_source_max_query_terms", 0) or 0),
        )
        if max_terms <= 0:
            return query_text
        terms = RetrievalRepository._select_sql_query_terms(query_text, max_terms=max_terms)  # noqa: SLF001
        return " ".join(terms) if terms else query_text

    @staticmethod
    def _candidate_source_score(candidate: RetrievalCandidate) -> float:
        return float(candidate.lexical_score or candidate.vector_score or 0.0)

    @staticmethod
    def _subquery_candidate_count_payload(
        subquery: QuerySubquery,
        index: int,
        source_name: str,
        candidate_count: int,
        *,
        source_query_text: str | None = None,
        timeout: bool,
        timeout_fallback_candidate_count: int = 0,
    ) -> dict[str, object]:
        return {
            "subquery_id": index,
            "case_shape": subquery.case_shape,
            "org_hint": subquery.org_hint,
            "evidence_hint": subquery.evidence_hint,
            "query_text": subquery.query_text,
            f"{source_name}_query_text": source_query_text or subquery.query_text,
            f"{source_name}_candidate_count": candidate_count,
            f"{source_name}_timeout": timeout,
            f"{source_name}_timeout_fallback_candidate_count": timeout_fallback_candidate_count,
        }

    @staticmethod
    def _merge_subquery_candidate_counts(
        left: list[dict[str, object]],
        right: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        merged: dict[int, dict[str, object]] = {}
        for item in [*left, *right]:
            subquery_id = int(item.get("subquery_id", len(merged)))
            current = merged.setdefault(subquery_id, {})
            current.update(item)
        return [merged[key] for key in sorted(merged)]

    @staticmethod
    def _is_timeout_error(error: Exception) -> bool:
        error_text = str(error).casefold()
        return (
            "statement timeout" in error_text
            or "query_canceled" in error_text
            or "canceling statement due to statement timeout" in error_text
        )

    def _collect_lexical_hits(
        self,
        queries: list[str],
        accessible_document_ids: list[UUID],
        candidate_pool: int,
    ) -> list[RetrievalCandidate]:
        merged: dict[UUID, RetrievalCandidate] = {}
        for query in queries:
            try:
                candidates = self.retrieval_repository.search_lexical(query, accessible_document_ids, candidate_pool)
            except Exception as exc:
                if not self._is_timeout_error(exc):
                    raise
                self.session.rollback()
                continue
            for candidate in candidates:
                existing = merged.get(candidate.chunk_id)
                if existing is None or (candidate.lexical_score or 0.0) > (existing.lexical_score or 0.0):
                    merged[candidate.chunk_id] = candidate
        hits = list(merged.values())
        hits.sort(key=lambda item: ((item.lexical_score or 0.0), -item.chunk_index), reverse=True)
        return hits[:candidate_pool]

    def _collect_structural_hits(
        self,
        queries: list[str],
        accessible_document_ids: list[UUID],
        candidate_pool: int,
    ) -> list[RetrievalCandidate]:
        self._last_structural_timeout = False
        if not self.settings.retrieval_structural_enabled:
            return []
        merged: dict[UUID, RetrievalCandidate] = {}
        structural_limit = max(candidate_pool, min(candidate_pool * 2, self.settings.retrieval_candidate_max))
        timeout_ms = max(0, int(getattr(self.settings, "retrieval_structural_timeout_ms", 0) or 0))
        for query in queries:
            timeout_applied = False
            timeout_error = False
            try:
                timeout_applied = self._set_local_statement_timeout(timeout_ms)
                candidates = self.retrieval_repository.search_structural(query, accessible_document_ids, structural_limit)
            except Exception as exc:
                timeout_error = self._is_timeout_error(exc)
                if not timeout_error:
                    raise
                self.session.rollback()
                self._last_structural_timeout = True
                continue
            finally:
                if timeout_applied and not timeout_error:
                    self._clear_local_statement_timeout()
            for candidate in candidates:
                existing = merged.get(candidate.chunk_id)
                if existing is None or (candidate.lexical_score or 0.0) > (existing.lexical_score or 0.0):
                    merged[candidate.chunk_id] = candidate
        hits = list(merged.values())
        hits.sort(key=lambda item: ((item.lexical_score or 0.0), -item.chunk_index), reverse=True)
        return hits[:candidate_pool]
    def _collect_indexed_sparse_hits(
        self,
        queries: list[str],
        accessible_document_ids: list[UUID],
        candidate_pool: int,
    ) -> list[RetrievalCandidate]:
        if not self.settings.retrieval_indexed_sparse_enabled:
            return []
        multiplier = max(1, int(getattr(self.settings, "retrieval_indexed_sparse_candidate_multiplier", 1) or 1))
        sparse_limit = max(candidate_pool, min(candidate_pool * multiplier, self.settings.retrieval_candidate_max))
        merged: dict[UUID, RetrievalCandidate] = {}
        for query in queries:
            try:
                candidates = self.retrieval_repository.search_indexed_sparse(query, accessible_document_ids, sparse_limit)
            except Exception as exc:
                if not self._is_timeout_error(exc):
                    raise
                self.session.rollback()
                candidates = self._collect_decomposed_source_timeout_fallback(
                    "indexed_sparse",
                    query,
                    accessible_document_ids,
                    sparse_limit,
                )
            for candidate in candidates:
                existing = merged.get(candidate.chunk_id)
                if existing is None or (candidate.lexical_score or 0.0) > (existing.lexical_score or 0.0):
                    merged[candidate.chunk_id] = candidate
        hits = list(merged.values())
        hits.sort(key=lambda item: ((item.lexical_score or 0.0), -item.chunk_index), reverse=True)
        return hits[:sparse_limit]

    def _collect_in_document_expansion(
        self,
        query: str,
        candidates: Iterable[RerankCandidate],
    ) -> list[RetrievalCandidate]:
        if not self.settings.retrieval_in_document_expansion_enabled:
            return []

        seed_count = max(0, int(self.settings.retrieval_in_document_expansion_seed_count or 0))
        per_document_limit = max(0, int(self.settings.retrieval_in_document_expansion_per_document or 0))
        max_candidates = max(0, int(self.settings.retrieval_in_document_expansion_max_candidates or 0))
        if seed_count <= 0 or per_document_limit <= 0 or max_candidates <= 0:
            return []

        seeds = sorted(
            candidates,
            key=lambda item: (
                item.fused_score,
                item.lexical_raw,
                item.vector_raw,
                -item.candidate.chunk_index,
            ),
            reverse=True,
        )[:seed_count]
        return self.retrieval_repository.expand_within_documents(
            query,
            [item.candidate for item in seeds],
            per_document_limit=per_document_limit,
            max_candidates=max_candidates,
            adjacent_window=max(0, int(getattr(self.settings, "retrieval_in_document_expansion_adjacent_window", 0) or 0)),
        )

    def _collect_document_evidence_sweep(
        self,
        query: str,
        candidates: Iterable[RerankCandidate],
    ) -> list[RetrievalCandidate]:
        if not self.settings.retrieval_document_evidence_sweep_enabled:
            return []

        seed_document_limit = max(0, int(self.settings.retrieval_document_evidence_sweep_seed_documents or 0))
        per_document_limit = max(0, int(self.settings.retrieval_document_evidence_sweep_per_document or 0))
        max_candidates = max(0, int(self.settings.retrieval_document_evidence_sweep_max_candidates or 0))
        if seed_document_limit <= 0 or per_document_limit <= 0 or max_candidates <= 0:
            return []

        seed_candidates = sorted(
            candidates,
            key=lambda item: (
                item.fused_score,
                item.lexical_raw,
                item.vector_raw,
                -item.candidate.chunk_index,
            ),
            reverse=True,
        )[: max(seed_document_limit * 4, seed_document_limit)]
        return self.retrieval_repository.sweep_within_documents(
            query,
            [item.candidate for item in seed_candidates],
            seed_document_limit=seed_document_limit,
            per_document_limit=per_document_limit,
            max_candidates=max_candidates,
        )

    def _collect_subquery_document_evidence_hits(
        self,
        query_plan: QueryOptimizationPlan,
        candidates: Iterable[RerankCandidate],
    ) -> list[RetrievalCandidate]:
        if not bool(getattr(self.settings, "retrieval_subquery_document_evidence_enabled", True)):
            return []
        if not query_plan.subqueries:
            return []

        seed_document_limit = max(0, int(getattr(self.settings, "retrieval_subquery_document_evidence_seed_documents", 0) or 0))
        per_subquery_limit = max(0, int(getattr(self.settings, "retrieval_subquery_document_evidence_per_subquery", 0) or 0))
        max_candidates = max(0, int(getattr(self.settings, "retrieval_subquery_document_evidence_max_candidates", 0) or 0))
        if seed_document_limit <= 0 or per_subquery_limit <= 0 or max_candidates <= 0:
            return []

        ranked_candidates = sorted(
            candidates,
            key=lambda item: (
                item.fused_score,
                item.lexical_raw,
                item.vector_raw,
                -item.candidate.chunk_index,
            ),
            reverse=True,
        )
        per_subquery_hits: list[list[RetrievalCandidate]] = []
        for subquery in query_plan.subqueries:
            if subquery.case_shape == "single_category_anchor":
                continue
            seeds = self._subquery_seed_candidates(subquery, ranked_candidates, seed_document_limit)
            if not seeds:
                continue
            query_text = subquery.evidence_hint or subquery.query_text
            exact_hits = self.retrieval_repository.search_exact_text_within_documents(
                query_text,
                [item.candidate for item in seeds],
                seed_document_limit=seed_document_limit,
                per_document_limit=per_subquery_limit,
                max_candidates=per_subquery_limit,
            )
            hits = self.retrieval_repository.sweep_within_documents(
                query_text,
                [item.candidate for item in seeds],
                seed_document_limit=seed_document_limit,
                per_document_limit=per_subquery_limit,
                max_candidates=per_subquery_limit,
            )
            merged_hits = self._merge_decomposed_source_hits([exact_hits, hits], per_subquery_limit)
            if merged_hits:
                per_subquery_hits.append(merged_hits)
        if not per_subquery_hits:
            return []
        return self._merge_decomposed_source_hits(per_subquery_hits, max_candidates)

    def _collect_subquery_neighbor_context_hits(
        self,
        query_plan: QueryOptimizationPlan,
        source_hits: Iterable[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        if not bool(getattr(self.settings, "retrieval_subquery_neighbor_context_enabled", True)):
            return []
        if not query_plan.subqueries:
            return []

        seed_count = max(0, int(getattr(self.settings, "retrieval_subquery_neighbor_context_seed_count", 0) or 0))
        window = max(0, int(getattr(self.settings, "retrieval_subquery_neighbor_context_window", 0) or 0))
        per_subquery_limit = max(0, int(getattr(self.settings, "retrieval_subquery_neighbor_context_per_subquery", 0) or 0))
        max_candidates = max(0, int(getattr(self.settings, "retrieval_subquery_neighbor_context_max_candidates", 0) or 0))
        if seed_count <= 0 or window <= 0 or per_subquery_limit <= 0 or max_candidates <= 0:
            return []

        ranked_source_hits = sorted(
            self._dedupe_source_hits(source_hits),
            key=lambda item: (self._candidate_source_score(item), -item.chunk_index),
            reverse=True,
        )
        if not ranked_source_hits:
            return []

        per_subquery_hits: list[list[RetrievalCandidate]] = []
        for subquery in query_plan.subqueries:
            if subquery.case_shape not in {"same_document_two_matters", "cross_document_comparison"}:
                continue
            seeds = self._subquery_neighbor_seed_hits(subquery, ranked_source_hits, seed_count)
            if not seeds:
                continue
            raw_neighbor_limit = self._subquery_neighbor_raw_candidate_limit(
                seed_count=seed_count,
                window=window,
                per_subquery_limit=per_subquery_limit,
            )
            neighbor_hits = self.retrieval_repository.collect_neighbor_context(
                seeds,
                window=window,
                per_document_limit=raw_neighbor_limit,
                max_candidates=raw_neighbor_limit,
            )
            rescored_hits = self._rescore_subquery_neighbor_hits(subquery, neighbor_hits)
            if rescored_hits:
                per_subquery_hits.append(rescored_hits[:per_subquery_limit])

        if not per_subquery_hits:
            return []
        return self._merge_decomposed_source_hits(per_subquery_hits, max_candidates)

    @staticmethod
    def _subquery_neighbor_raw_candidate_limit(*, seed_count: int, window: int, per_subquery_limit: int) -> int:
        window_candidates = max(1, seed_count) * max(1, window) * 2
        return max(per_subquery_limit, min(window_candidates, max(per_subquery_limit * 4, 16)))

    def _subquery_neighbor_seed_hits(
        self,
        subquery: QuerySubquery,
        source_hits: list[RetrievalCandidate],
        seed_count: int,
    ) -> list[RetrievalCandidate]:
        scored: list[tuple[float, RetrievalCandidate]] = []
        for hit in source_hits:
            if subquery.org_hint and not self._document_matches_org_hint(hit.document_title, subquery.org_hint):
                continue
            item = RerankCandidate(
                candidate=hit,
                lexical_raw=hit.lexical_score or 0.0,
                lexical_norm=hit.lexical_score or 0.0,
                fused_score=hit.lexical_score or hit.vector_score or 0.0,
            )
            score = self._subquery_coverage_score(item, subquery)
            if score <= 0 and subquery.case_shape != "single_category_anchor":
                continue
            scored.append((score, hit))

        scored.sort(
            key=lambda entry: (
                entry[0],
                self._candidate_source_score(entry[1]),
                -entry[1].chunk_index,
            ),
            reverse=True,
        )
        selected: list[RetrievalCandidate] = []
        seen: set[UUID] = set()
        for _, hit in scored:
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            selected.append(hit)
            if len(selected) >= seed_count:
                break
        return selected

    def _rescore_subquery_neighbor_hits(
        self,
        subquery: QuerySubquery,
        hits: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        evidence_terms = self._subquery_evidence_terms(subquery)
        evidence_hint = self._compact_match_text(subquery.evidence_hint or subquery.query_text)
        rescored: list[RetrievalCandidate] = []
        for hit in hits:
            candidate_terms = self._final_coverage_terms(self._retrieval_candidate_text(hit))
            overlap = self._final_coverage_overlap(evidence_terms, candidate_terms)
            exact_hint_bonus = 0.0
            if evidence_hint and evidence_hint in self._compact_match_text(self._retrieval_candidate_text(hit)):
                exact_hint_bonus = 0.55
            elif overlap < 0.45:
                continue
            structural_bonus = 0.08 if hit.section_title or hit.heading_path or hit.clause_full_name else 0.0
            adjacency_score = float(hit.lexical_score or 0.0)
            score = adjacency_score + (overlap * 1.35) + exact_hint_bonus + structural_bonus
            if score <= 0:
                continue
            rescored.append(replace(hit, lexical_score=float(score), vector_score=None))

        rescored.sort(key=lambda item: (self._candidate_source_score(item), -item.chunk_index), reverse=True)
        return rescored

    @staticmethod
    def _dedupe_source_hits(source_hits: Iterable[RetrievalCandidate]) -> list[RetrievalCandidate]:
        best_by_chunk: dict[UUID, RetrievalCandidate] = {}
        for hit in source_hits:
            existing = best_by_chunk.get(hit.chunk_id)
            if existing is None:
                best_by_chunk[hit.chunk_id] = hit
                continue
            existing_score = float(existing.lexical_score or existing.vector_score or 0.0)
            hit_score = float(hit.lexical_score or hit.vector_score or 0.0)
            if hit_score > existing_score:
                best_by_chunk[hit.chunk_id] = hit
        return list(best_by_chunk.values())

    @staticmethod
    def _retrieval_candidate_text(candidate: RetrievalCandidate) -> str:
        return " ".join(
            str(part or "")
            for part in [
                candidate.document_title,
                candidate.section_title,
                candidate.clause_full_name,
                candidate.article_number,
                candidate.heading_path,
                candidate.content,
            ]
        )

    def _subquery_seed_candidates(
        self,
        subquery: QuerySubquery,
        candidates: list[RerankCandidate],
        seed_document_limit: int,
    ) -> list[RerankCandidate]:
        if seed_document_limit <= 0:
            return []
        selected: list[RerankCandidate] = []
        seen_documents: set[UUID] = set()

        for item in candidates:
            document_id = item.candidate.document_id
            if document_id in seen_documents:
                continue
            if subquery.org_hint and not self._document_matches_org_hint(item.candidate.document_title, subquery.org_hint):
                continue
            selected.append(item)
            seen_documents.add(document_id)
            if len(selected) >= seed_document_limit:
                return selected

        if selected or subquery.org_hint:
            return selected

        for item in candidates:
            document_id = item.candidate.document_id
            if document_id in seen_documents:
                continue
            selected.append(item)
            seen_documents.add(document_id)
            if len(selected) >= seed_document_limit:
                break
        return selected

    @classmethod
    def _document_matches_org_hint(cls, document_title: str, org_hint: str) -> bool:
        normalized_title = cls._compact_match_text(document_title)
        normalized_hint = cls._compact_match_text(org_hint)
        if not normalized_title or not normalized_hint:
            return False
        if normalized_hint in normalized_title or normalized_title in normalized_hint:
            return True
        hint_core = re.sub(r"(?:有限公司|股份有限公司|集团有限公司|集团|公司)$", "", normalized_hint)
        return len(hint_core) >= 4 and hint_core in normalized_title

    @staticmethod
    def _compact_match_text(value: str | None) -> str:
        return "".join(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", str(value or "").casefold()))

    def _collect_document_first_evidence_hits(
        self,
        query: str,
        candidates: Iterable[RerankCandidate],
    ) -> list[RetrievalCandidate]:
        if not self.settings.retrieval_document_first_evidence_enabled:
            return []

        seed_document_limit = max(0, int(getattr(self.settings, "retrieval_document_first_evidence_seed_documents", 0) or 0))
        per_document_limit = max(0, int(getattr(self.settings, "retrieval_document_first_evidence_per_document", 0) or 0))
        max_candidates = max(0, int(getattr(self.settings, "retrieval_document_first_evidence_max_candidates", 0) or 0))
        if seed_document_limit <= 0 or per_document_limit <= 0 or max_candidates <= 0:
            return []

        seed_candidates = sorted(
            candidates,
            key=lambda item: (
                item.fused_score,
                item.lexical_raw,
                item.vector_raw,
                -item.candidate.chunk_index,
            ),
            reverse=True,
        )[: max(seed_document_limit * 6, seed_document_limit)]
        return self.retrieval_repository.search_document_first_evidence(
            query,
            [item.candidate for item in seed_candidates],
            seed_document_limit=seed_document_limit,
            per_document_limit=per_document_limit,
            max_candidates=max_candidates,
        )

    def _collect_document_neighbor_context_hits(
        self,
        candidates: Iterable[RerankCandidate],
    ) -> list[RetrievalCandidate]:
        if not self.settings.retrieval_document_neighbor_context_enabled:
            return []

        seed_count = max(0, int(getattr(self.settings, "retrieval_document_neighbor_context_seed_count", 0) or 0))
        window = max(0, int(getattr(self.settings, "retrieval_document_neighbor_context_window", 0) or 0))
        per_document_limit = max(0, int(getattr(self.settings, "retrieval_document_neighbor_context_per_document", 0) or 0))
        max_candidates = max(0, int(getattr(self.settings, "retrieval_document_neighbor_context_max_candidates", 0) or 0))
        if seed_count <= 0 or window <= 0 or per_document_limit <= 0 or max_candidates <= 0:
            return []

        seeds = sorted(
            candidates,
            key=lambda item: (
                item.fused_score,
                item.lexical_raw,
                item.vector_raw,
                -item.candidate.chunk_index,
            ),
            reverse=True,
        )[:seed_count]
        return self.retrieval_repository.collect_neighbor_context(
            [item.candidate for item in seeds],
            window=window,
            per_document_limit=per_document_limit,
            max_candidates=max_candidates,
        )

    def _candidate_pool_size(self, top_k: int) -> int:
        multiplier = max(1, int(self.settings.retrieval_candidate_multiplier or 1))
        minimum = max(top_k, int(self.settings.retrieval_candidate_min or top_k))
        maximum = max(minimum, int(self.settings.retrieval_candidate_max or minimum))
        return min(max(top_k * multiplier, minimum), maximum)

    def _rerank_result_limit(self, top_k: int, candidate_pool: int, candidate_count: int) -> int:
        base_limit = max(top_k, min(candidate_count, candidate_pool))
        if not self.settings.retrieval_final_coverage_enabled:
            return base_limit
        scan_limit = max(base_limit, int(getattr(self.settings, "retrieval_final_coverage_scan_limit", base_limit) or base_limit))
        return min(candidate_count, scan_limit)

    def _rerank_or_rank_candidates(
        self,
        original_query: str,
        retrieval_query: str,
        candidates: list[RerankCandidate],
        *,
        top_k: int,
        candidate_pool: int,
        target_document_id: UUID | None,
    ) -> RerankResult:
        if not candidates:
            return RerankResult(
                candidates=[],
                strategy="disabled-local-heuristic",
                pre_rerank_count=0,
                post_rerank_count=0,
            )
        if not self._should_run_reranker():
            ranked = self._rank_without_rerank(candidates)
            return RerankResult(
                candidates=ranked,
                strategy="disabled-local-heuristic",
                pre_rerank_count=len(candidates),
                post_rerank_count=len(ranked),
            )

        rerank_query = self._build_rerank_query(original_query, retrieval_query)
        rerank_limit = self._rerank_result_limit(top_k, candidate_pool, len(candidates))
        return self.reranker.rerank(
            rerank_query,
            candidates,
            rerank_limit,
            target_document_id=target_document_id,
        )

    def _should_run_reranker(self) -> bool:
        strategy_name = str(getattr(self.reranker, "strategy_name", "") or "")
        if strategy_name != "heuristic-overlap":
            return True
        return bool(getattr(self.settings, "retrieval_heuristic_rerank_enabled", False))

    @staticmethod
    def _rank_without_rerank(candidates: Iterable[RerankCandidate]) -> list[RerankCandidate]:
        ranked = sorted(
            candidates,
            key=lambda item: (
                item.fused_score,
                item.lexical_raw,
                item.vector_raw,
                -item.candidate.chunk_index,
            ),
            reverse=True,
        )
        for item in ranked:
            item.rerank_score = item.fused_score
        return ranked

    def _collect_evidence_preservation_candidates(
        self,
        candidates: Iterable[RerankCandidate],
    ) -> list[RerankCandidate]:
        if not self.settings.retrieval_evidence_preservation_enabled:
            return []

        pool_limit = max(0, int(getattr(self.settings, "retrieval_evidence_preservation_pool_limit", 0) or 0))
        max_slots = max(0, int(getattr(self.settings, "retrieval_evidence_preservation_max_slots", 0) or 0))
        if pool_limit <= 0 or max_slots <= 0:
            return []

        min_lexical_norm = max(0.0, float(getattr(self.settings, "retrieval_evidence_preservation_min_lexical_norm", 0.0) or 0.0))
        min_fused_score = max(0.0, float(getattr(self.settings, "retrieval_evidence_preservation_min_fused_score", 0.0) or 0.0))
        ranked_pool = sorted(
            candidates,
            key=lambda item: (
                self._evidence_preservation_score(item),
                item.fused_score,
                item.lexical_norm,
                item.lexical_raw,
                -item.candidate.chunk_index,
            ),
            reverse=True,
        )

        preservation_candidates: list[RerankCandidate] = []
        seen: set[UUID] = set()
        for item in ranked_pool:
            if len(preservation_candidates) >= pool_limit:
                break
            if item.candidate.chunk_id in seen:
                continue
            seen.add(item.candidate.chunk_id)
            if not self._has_preservable_source(item):
                continue
            if item.lexical_norm < min_lexical_norm and item.fused_score < min_fused_score:
                continue
            preservation_candidates.append(item)
        return preservation_candidates

    def _collect_final_coverage_candidates(
        self,
        query: str,
        candidates: list[RerankCandidate],
        selected: list[RerankCandidate],
        top_k: int,
    ) -> list[RerankCandidate]:
        if not self.settings.retrieval_final_coverage_enabled or top_k <= 1:
            return []
        if not self._is_final_coverage_query(query):
            return []

        max_slots = min(
            max(0, int(getattr(self.settings, "retrieval_final_coverage_max_slots", 0) or 0)),
            max(top_k - 1, 0),
        )
        if max_slots <= 0:
            return []

        selected_ids = {item.candidate.chunk_id for item in selected}
        anchor_document_ids = {item.candidate.document_id for item in selected}
        if not anchor_document_ids:
            return []

        min_rerank = max(0.0, float(getattr(self.settings, "retrieval_final_coverage_min_rerank_score", 0.0) or 0.0))
        min_fused = max(0.0, float(getattr(self.settings, "retrieval_final_coverage_min_fused_score", 0.0) or 0.0))
        selected_terms_by_document = self._final_coverage_selected_terms_by_document(selected)
        selected_aspect_scores_by_document = self._final_coverage_selected_aspect_scores_by_document(query, selected)
        coverage_pool: list[tuple[RerankCandidate, float, int]] = []
        seen: set[UUID] = set()
        for position, item in enumerate(candidates):
            chunk_id = item.candidate.chunk_id
            if chunk_id in selected_ids or chunk_id in seen:
                continue
            if item.candidate.document_id not in anchor_document_ids:
                continue
            if item.rerank_score < min_rerank and item.fused_score < min_fused:
                continue
            if not self._is_final_coverage_candidate(item, min_rerank=min_rerank, min_fused=min_fused):
                continue
            seen.add(chunk_id)
            score = self._final_coverage_candidate_score(
                query,
                item,
                selected_terms_by_document=selected_terms_by_document,
                selected_aspect_scores_by_document=selected_aspect_scores_by_document,
            )
            coverage_pool.append((item, score, position))
        coverage_pool.sort(
            key=lambda entry: (
                entry[1],
                entry[0].rerank_score,
                entry[0].fused_score,
                -entry[2],
            ),
            reverse=True,
        )
        return [item for item, _, _ in coverage_pool[:max_slots]]

    def _collect_subquery_final_coverage_candidates(
        self,
        query_plan: QueryOptimizationPlan,
        candidates: list[RerankCandidate],
        selected: list[RerankCandidate],
        top_k: int,
    ) -> list[RerankCandidate]:
        if not query_plan.subqueries or top_k <= 1:
            return []

        max_slots = min(
            max(0, int(getattr(self.settings, "retrieval_subquery_final_coverage_max_slots", 0) or 0)),
            max(top_k - 1, 0),
        )
        if max_slots <= 0:
            return []

        selected_ids = {item.candidate.chunk_id for item in selected}
        selected_by_subquery = {
            index
            for index, subquery in enumerate(query_plan.subqueries)
            if any(self._selected_candidate_covers_subquery(item, subquery) for item in selected)
        }
        coverage_candidates: list[tuple[RerankCandidate, float, int]] = []
        seen_ids: set[UUID] = set()
        for subquery_index, subquery in enumerate(query_plan.subqueries):
            if subquery_index in selected_by_subquery or subquery.case_shape == "single_category_anchor":
                continue
            best_entry: tuple[RerankCandidate, float, int] | None = None
            for position, item in enumerate(candidates):
                chunk_id = item.candidate.chunk_id
                if chunk_id in selected_ids or chunk_id in seen_ids:
                    continue
                if not self._candidate_covers_subquery(item, subquery):
                    continue
                score = self._subquery_coverage_score(item, subquery)
                if score <= 0:
                    continue
                entry = (item, score, position)
                if best_entry is None or self._subquery_coverage_entry_key(entry) > self._subquery_coverage_entry_key(best_entry):
                    best_entry = entry
            if best_entry is not None:
                coverage_candidates.append(best_entry)
                seen_ids.add(best_entry[0].candidate.chunk_id)

        coverage_candidates.sort(
            key=lambda entry: (
                entry[1],
                entry[0].rerank_score,
                entry[0].fused_score,
                -entry[2],
            ),
            reverse=True,
        )
        return [item for item, _, _ in coverage_candidates[:max_slots]]

    @staticmethod
    def _subquery_coverage_entry_key(entry: tuple[RerankCandidate, float, int]) -> tuple[float, int, float, float, int]:
        item, score, position = entry
        source_priority = 0
        if SUBQUERY_NEIGHBOR_CONTEXT_SOURCE in item.sources:
            source_priority = 2
        elif SUBQUERY_DOCUMENT_EVIDENCE_SOURCE in item.sources:
            source_priority = 1
        return (
            score,
            source_priority,
            max(item.rerank_score, item.fused_score),
            item.lexical_raw,
            -position,
        )

    def _candidate_covers_subquery(self, item: RerankCandidate, subquery: QuerySubquery) -> bool:
        if subquery.org_hint and not self._document_matches_org_hint(item.candidate.document_title, subquery.org_hint):
            return False
        if subquery.case_shape == "table_structured_lookup":
            table_pairs = self._subquery_table_lookup_pairs(subquery)
            if table_pairs:
                return self._candidate_matches_table_lookup_pairs(item, table_pairs)
        if not self._candidate_contains_subquery_discriminators(item, subquery):
            return False
        evidence_terms = self._subquery_evidence_terms(subquery)
        if not evidence_terms:
            return False
        candidate_text = self._final_coverage_candidate_text(item)
        evidence_hint = self._normalized_coverage_text(subquery.evidence_hint or subquery.query_text)
        if len(evidence_hint) >= 10 and evidence_hint in self._normalized_coverage_text(candidate_text):
            return self._candidate_has_sufficient_exact_subquery_context(item, subquery)
        candidate_terms = self._final_coverage_terms(candidate_text)
        matched_terms = len(evidence_terms.intersection(candidate_terms))
        if matched_terms < min(3, len(evidence_terms)):
            return False
        overlap = self._final_coverage_overlap(evidence_terms, candidate_terms)
        return overlap >= 0.45

    def _selected_candidate_covers_subquery(self, item: RerankCandidate, subquery: QuerySubquery) -> bool:
        if subquery.org_hint and not self._document_matches_org_hint(item.candidate.document_title, subquery.org_hint):
            return False
        if subquery.case_shape == "table_structured_lookup":
            table_pairs = self._subquery_table_lookup_pairs(subquery)
            if table_pairs:
                return self._candidate_matches_table_lookup_pairs(item, table_pairs)
        if not self._candidate_contains_subquery_discriminators(item, subquery):
            return False
        evidence_terms = self._subquery_evidence_terms(subquery)
        if not evidence_terms:
            return False
        candidate_text = self._final_coverage_candidate_text(item)
        evidence_hint = self._normalized_coverage_text(subquery.evidence_hint or subquery.query_text)
        if len(evidence_hint) >= 12 and evidence_hint in self._normalized_coverage_text(candidate_text):
            return self._candidate_has_sufficient_exact_subquery_context(item, subquery)
        candidate_terms = self._final_coverage_terms(candidate_text)
        matched_terms = len(evidence_terms.intersection(candidate_terms))
        if len(evidence_hint) >= 20:
            if matched_terms < min(6, len(evidence_terms)):
                return False
            minimum_overlap = 0.98 if self._subquery_discriminator_terms(subquery) else 0.98
            return self._final_coverage_overlap(evidence_terms, candidate_terms) >= minimum_overlap
        if matched_terms < min(4, len(evidence_terms)):
            return False
        return self._final_coverage_overlap(evidence_terms, candidate_terms) >= 0.55

    def _candidate_has_sufficient_exact_subquery_context(
        self,
        item: RerankCandidate,
        subquery: QuerySubquery,
    ) -> bool:
        evidence_hint = self._normalized_coverage_text(subquery.evidence_hint or subquery.query_text)
        if len(evidence_hint) < 20:
            return True
        candidate_text = self._normalized_coverage_text(self._final_coverage_candidate_text(item))
        position = candidate_text.find(evidence_hint)
        if position < 0:
            return True

        trailing_chars = len(candidate_text) - position - len(evidence_hint)
        required_trailing_chars = min(48, max(24, len(evidence_hint) // 4))
        if trailing_chars >= required_trailing_chars:
            return True

        raw_suffix = self._candidate_raw_suffix_after_subquery_hint(item, subquery)
        if raw_suffix and re.search(r"[。；;.!！？?》）)]", raw_suffix[:80]):
            return True
        return False

    def _candidate_raw_suffix_after_subquery_hint(
        self,
        item: RerankCandidate,
        subquery: QuerySubquery,
    ) -> str | None:
        raw_hint = " ".join(str(subquery.evidence_hint or subquery.query_text or "").split())
        if len(raw_hint) < 10:
            return None
        raw_text = self._final_coverage_candidate_text(item)
        position = raw_text.find(raw_hint)
        if position >= 0:
            return raw_text[position + len(raw_hint) :]

        compact_hint = self._compact_match_text(raw_hint)
        if len(compact_hint) < 10:
            return None

        compact_chars: list[str] = []
        compact_to_raw_index: list[int] = []
        for raw_index, char in enumerate(raw_text):
            if re.match(r"[A-Za-z0-9\u4e00-\u9fff]", char.casefold()):
                compact_chars.append(char.casefold())
                compact_to_raw_index.append(raw_index)
        compact_text = "".join(compact_chars)
        compact_position = compact_text.find(compact_hint)
        if compact_position < 0:
            return None

        compact_end_index = compact_position + len(compact_hint) - 1
        if compact_end_index >= len(compact_to_raw_index):
            return None
        raw_end_index = compact_to_raw_index[compact_end_index] + 1
        return raw_text[raw_end_index:]

    def _subquery_coverage_score(self, item: RerankCandidate, subquery: QuerySubquery) -> float:
        evidence_terms = self._subquery_evidence_terms(subquery)
        candidate_terms = self._final_coverage_terms(self._final_coverage_candidate_text(item))
        overlap = self._final_coverage_overlap(evidence_terms, candidate_terms)
        source_bonus = 0.18 if SUBQUERY_DOCUMENT_EVIDENCE_SOURCE in item.sources else 0.0
        if SUBQUERY_NEIGHBOR_CONTEXT_SOURCE in item.sources:
            source_bonus = max(source_bonus, 0.28)
        if subquery.case_shape == "table_structured_lookup":
            table_pairs = self._subquery_table_lookup_pairs(subquery)
            if table_pairs and self._candidate_matches_table_lookup_pairs(item, table_pairs):
                source_bonus = max(source_bonus, 0.72)
        return overlap + source_bonus + min(max(item.rerank_score, item.fused_score), 1.0) * 0.18

    @staticmethod
    def _subquery_evidence_terms(subquery: QuerySubquery) -> set[str]:
        return {
            term
            for term in tokenize_search_text(subquery.evidence_hint or subquery.query_text)
            if len(term) > 1 and term not in FINAL_COVERAGE_LOW_SIGNAL_TERMS
        }

    @classmethod
    def _subquery_table_lookup_pairs(cls, subquery: QuerySubquery) -> set[tuple[str, str]]:
        return cls._extract_table_lookup_pairs(subquery.evidence_hint or subquery.query_text)

    def _candidate_matches_table_lookup_pairs(
        self,
        item: RerankCandidate,
        lookup_pairs: set[tuple[str, str]],
    ) -> bool:
        if not lookup_pairs:
            return False
        candidate_pairs = self._extract_table_lookup_pairs(self._final_coverage_candidate_text(item))
        return bool(lookup_pairs.intersection(candidate_pairs))

    @classmethod
    def _extract_table_lookup_pairs(cls, value: str | None) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        text = str(value or "")
        for start in range(len(text)):
            match = TABLE_LOOKUP_PAIR_PATTERN.match(text, start)
            if not match:
                continue
            field = cls._compact_match_text(match.group("field"))
            cell_value = cls._compact_match_text(match.group("value"))
            if len(field) < 2 or len(cell_value) < 2:
                continue
            pairs.add((field, cell_value))
        return {
            (field, cell_value)
            for field, cell_value in pairs
            if not any(
                other_value == cell_value
                and field != other_field
                and field in other_field
                and len(other_field) > len(field)
                for other_field, other_value in pairs
            )
        }

    @staticmethod
    def _is_final_coverage_query(query: str) -> bool:
        normalized = str(query or "").strip()
        if any(hint in normalized for hint in FINAL_COVERAGE_QUERY_HINTS):
            return True
        has_broad_hint = any(hint in normalized for hint in FINAL_COVERAGE_BROAD_QUERY_HINTS)
        if has_broad_hint and any(hint in normalized for hint in FINAL_COVERAGE_LIST_QUERY_HINTS):
            return True
        if not has_broad_hint:
            return False
        return any(delimiter in normalized for delimiter in FINAL_COVERAGE_MULTI_DELIMITERS)

    @staticmethod
    def _is_final_coverage_candidate(item: RerankCandidate, *, min_rerank: float, min_fused: float) -> bool:
        if item.sources.intersection(FINAL_COVERAGE_SOURCE_NAMES):
            return True
        candidate = item.candidate
        return bool(
            candidate.clause_full_name
            or candidate.article_number
            or candidate.heading_path
            or candidate.section_title
        ) and (item.rerank_score >= min_rerank or item.fused_score >= min_fused)

    def _final_coverage_candidate_score(
        self,
        query: str,
        item: RerankCandidate,
        *,
        selected_terms_by_document: dict[UUID, set[str]],
        selected_aspect_scores_by_document: dict[UUID, dict[int, float]],
    ) -> float:
        query_terms = self._final_coverage_terms(query)
        aspect_terms = self._final_coverage_query_aspects(query)
        candidate_text = self._final_coverage_candidate_text(item)
        candidate_terms = self._final_coverage_terms(candidate_text)
        document_id = item.candidate.document_id
        selected_terms = selected_terms_by_document.get(document_id, set())
        overlap_terms = query_terms.intersection(candidate_terms)

        query_overlap = len(overlap_terms) / max(len(query_terms), 1)
        novel_query_overlap = len(overlap_terms.difference(selected_terms)) / max(len(query_terms), 1)
        selected_aspect_scores = selected_aspect_scores_by_document.get(document_id, {})
        aspect_gain = 0.0
        for index, terms in enumerate(aspect_terms):
            candidate_aspect_score = self._final_coverage_overlap(terms, candidate_terms)
            aspect_gain = max(aspect_gain, candidate_aspect_score - selected_aspect_scores.get(index, 0.0))

        score = 0.0
        score += query_overlap * 0.45
        score += novel_query_overlap * 0.75
        score += max(aspect_gain, 0.0) * 1.25
        score += self._final_coverage_threshold_score(query, candidate_text)
        score += min(max(item.rerank_score, item.fused_score), 1.0) * 0.22
        if item.sources.intersection(FINAL_COVERAGE_SOURCE_NAMES):
            score += 0.08
        if item.candidate.article_number or item.candidate.clause_full_name:
            score += 0.04
        return score

    @staticmethod
    def _final_coverage_candidate_text(item: RerankCandidate) -> str:
        candidate = item.candidate
        return " ".join(
            part
            for part in [
                candidate.document_title,
                candidate.section_title or "",
                candidate.clause_full_name or "",
                candidate.article_number or "",
                candidate.heading_path or "",
                candidate.content[:1800],
            ]
            if part
        )

    def _final_coverage_selected_terms_by_document(self, selected: list[RerankCandidate]) -> dict[UUID, set[str]]:
        terms_by_document: dict[UUID, set[str]] = {}
        for item in selected:
            document_terms = terms_by_document.setdefault(item.candidate.document_id, set())
            document_terms.update(self._final_coverage_terms(self._final_coverage_candidate_text(item)))
        return terms_by_document

    def _final_coverage_selected_aspect_scores_by_document(
        self,
        query: str,
        selected: list[RerankCandidate],
    ) -> dict[UUID, dict[int, float]]:
        aspect_terms = self._final_coverage_query_aspects(query)
        if not aspect_terms:
            return {}
        scores_by_document: dict[UUID, dict[int, float]] = {}
        for item in selected:
            candidate_terms = self._final_coverage_terms(self._final_coverage_candidate_text(item))
            document_scores = scores_by_document.setdefault(item.candidate.document_id, {})
            for index, terms in enumerate(aspect_terms):
                document_scores[index] = max(
                    document_scores.get(index, 0.0),
                    self._final_coverage_overlap(terms, candidate_terms),
                )
        return scores_by_document

    @staticmethod
    def _final_coverage_terms(value: str) -> set[str]:
        return {
            term
            for term in tokenize_search_text(value)
            if len(term) > 1 and term not in FINAL_COVERAGE_LOW_SIGNAL_TERMS
        }

    @classmethod
    def _final_coverage_query_aspects(cls, query: str) -> list[set[str]]:
        parts = [
            part.strip()
            for part in re.split(r"以及|分别|同时|各自|[、与及和]", str(query or ""))
            if len(part.strip()) >= 2
        ]
        aspect_terms = [cls._final_coverage_terms(part) for part in parts]
        return [terms for terms in aspect_terms if terms]

    @staticmethod
    def _final_coverage_overlap(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left.intersection(right)) / max(len(left), 1)

    @staticmethod
    def _normalized_coverage_text(value: str) -> str:
        return "".join(
            char
            for char in " ".join(str(value or "").casefold().split())
            if char not in "，。；：、,.!?！？;:()（）[]【】\"' "
        )

    @staticmethod
    def _final_coverage_threshold_score(query: str, value: str) -> float:
        normalized_query = re.sub(r"\s+", "", str(query or ""))
        if not any(hint in normalized_query for hint in FINAL_COVERAGE_THRESHOLD_QUERY_HINTS):
            return 0.0
        normalized_value = re.sub(r"\s+", "", str(value or ""))
        if FINAL_COVERAGE_THRESHOLD_VALUE_PATTERN.search(normalized_value):
            return 0.55
        if any(hint in normalized_query for hint in ("多少", "多久", "时限", "期限")) and FINAL_COVERAGE_MEASURE_VALUE_PATTERN.search(
            normalized_value
        ):
            return 0.22
        return 0.0

    def _candidate_contains_subquery_discriminators(self, item: RerankCandidate, subquery: QuerySubquery) -> bool:
        required_terms = self._subquery_discriminator_terms(subquery)
        if not required_terms:
            return True
        candidate_text = self._compact_match_text(self._final_coverage_candidate_text(item))
        return all(term in candidate_text for term in required_terms)

    @classmethod
    def _subquery_discriminator_terms(cls, subquery: QuerySubquery) -> set[str]:
        text = str(subquery.evidence_hint or subquery.query_text or "")
        patterns = (
            r"20\d{2}\s*年(?:\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?)?",
            r"\d{1,2}\s*月\s*\d{1,2}\s*日",
            r"第[一二三四五六七八九十百千万两0-9]+(?:期|条|章|节|阶段|次|轮)",
            r"[A-Za-z]*\d[A-Za-z0-9\u4e00-\u9fff_.\-]*",
        )
        terms: set[str] = set()
        for pattern in patterns:
            for match in re.findall(pattern, text):
                compact = cls._compact_match_text(match)
                if len(compact) >= 2:
                    terms.add(compact)
        return terms

    def _select_final_candidates(
        self,
        candidates: list[RerankCandidate],
        top_k: int,
        *,
        query_plan: QueryOptimizationPlan | None = None,
        preservation_candidates: list[RerankCandidate] | None = None,
        coverage_candidates: list[RerankCandidate] | None = None,
        subquery_coverage_candidates: list[RerankCandidate] | None = None,
    ) -> list[RerankCandidate]:
        selected = self._select_diverse_final_candidates(candidates, top_k)
        replacement_protected_chunk_ids = self._subquery_selected_coverage_chunk_ids(query_plan, selected)
        selected = self._inject_final_candidates(
            selected,
            coverage_candidates,
            top_k,
            max_slots=int(getattr(self.settings, "retrieval_final_coverage_max_slots", 0) or 0),
            prefer_same_document_replacement=True,
            replacement_protected_chunk_ids=replacement_protected_chunk_ids,
        )
        selected = self._inject_final_candidates(
            selected,
            subquery_coverage_candidates,
            top_k,
            max_slots=int(getattr(self.settings, "retrieval_subquery_final_coverage_max_slots", 0) or 0),
        )
        if not preservation_candidates or top_k <= 0:
            return selected

        max_slots = min(
            max(0, int(getattr(self.settings, "retrieval_evidence_preservation_max_slots", 0) or 0)),
            top_k,
        )
        return self._inject_final_candidates(selected, preservation_candidates, top_k, max_slots=max_slots)

    def _inject_final_candidates(
        self,
        selected: list[RerankCandidate],
        candidates: list[RerankCandidate] | None,
        top_k: int,
        *,
        max_slots: int,
        prefer_same_document_replacement: bool = False,
        replacement_protected_chunk_ids: set[UUID] | None = None,
    ) -> list[RerankCandidate]:
        if not candidates or top_k <= 0 or max_slots <= 0:
            return selected

        selected_ids = {item.candidate.chunk_id for item in selected}
        injected: list[RerankCandidate] = []
        updated = list(selected)
        for item in candidates:
            if len(injected) >= min(max_slots, top_k):
                break
            if item.candidate.chunk_id in selected_ids:
                continue
            if prefer_same_document_replacement:
                replacement_index = self._same_document_replacement_index(
                    updated,
                    item,
                    replacement_protected_chunk_ids=replacement_protected_chunk_ids,
                )
                if replacement_index is None:
                    continue
                selected_ids.discard(updated[replacement_index].candidate.chunk_id)
                updated[replacement_index] = item
                selected_ids.add(item.candidate.chunk_id)
                injected.append(item)
                continue
            injected.append(item)
            selected_ids.add(item.candidate.chunk_id)
        if not injected:
            return selected
        if prefer_same_document_replacement:
            return updated[:top_k]

        retained_count = max(top_k - len(injected), 0)
        return [*selected[:retained_count], *injected][:top_k]

    def _same_document_replacement_index(
        self,
        selected: list[RerankCandidate],
        candidate: RerankCandidate,
        *,
        replacement_protected_chunk_ids: set[UUID] | None = None,
    ) -> int | None:
        diversity_protected_slots = max(
            0,
            int(getattr(self.settings, "retrieval_document_diversity_protected_top_k_slots", 0) or 0),
        )
        configured_coverage_protected_slots = max(
            0,
            int(getattr(self.settings, "retrieval_final_coverage_protected_top_k_slots", 0) or 0),
        )
        coverage_protected_slots = min(configured_coverage_protected_slots, max(len(selected) - 1, 0))
        protected_slots = min(max(diversity_protected_slots, coverage_protected_slots), len(selected))
        replacement_indexes = [
                index
                for index, item in enumerate(selected[protected_slots:], start=protected_slots)
                if item.candidate.document_id == candidate.candidate.document_id
                and item.candidate.chunk_id != candidate.candidate.chunk_id
                and item.candidate.chunk_id not in (replacement_protected_chunk_ids or set())
                and not self._is_final_replacement_protected(item)
            ]
        if not replacement_indexes and configured_coverage_protected_slots >= len(selected):
            replacement_indexes = [
                index
                for index, item in enumerate(selected[diversity_protected_slots:], start=diversity_protected_slots)
                if item.candidate.document_id == candidate.candidate.document_id
                and item.candidate.chunk_id != candidate.candidate.chunk_id
                and item.candidate.chunk_id not in (replacement_protected_chunk_ids or set())
                and not self._is_final_replacement_protected(item)
            ]
        if not replacement_indexes:
            return None
        return min(
            replacement_indexes,
            key=lambda index: (
                self._final_quality_score(selected[index]),
                selected[index].fused_score,
                selected[index].lexical_raw,
            ),
        )

    def _subquery_selected_coverage_chunk_ids(
        self,
        query_plan: QueryOptimizationPlan | None,
        selected: list[RerankCandidate],
    ) -> set[UUID]:
        if query_plan is None or not query_plan.subqueries:
            return set()
        protected_ids: set[UUID] = set()
        for item in selected:
            if any(self._selected_candidate_covers_subquery(item, subquery) for subquery in query_plan.subqueries):
                protected_ids.add(item.candidate.chunk_id)
        return protected_ids

    def _select_diverse_final_candidates(self, candidates: list[RerankCandidate], top_k: int) -> list[RerankCandidate]:
        if not self.settings.retrieval_document_diversity_enabled or top_k <= 1:
            return candidates[:top_k]

        configured_limit = max(1, int(self.settings.retrieval_document_diversity_max_chunks or 1))
        dynamic_limit = max(1, min(configured_limit, max(2, top_k // 2)))
        per_document_limit = dynamic_limit
        selected: list[RerankCandidate] = []
        per_document_counts: dict[UUID, int] = {}
        deferred: list[RerankCandidate] = []
        weak_diversity_deferred: list[RerankCandidate] = []
        for item in candidates:
            count = per_document_counts.get(item.candidate.document_id, 0)
            if count < per_document_limit and self._should_defer_weak_diversity_candidate(item, deferred):
                weak_diversity_deferred.append(item)
            elif count < per_document_limit:
                selected.append(item)
                per_document_counts[item.candidate.document_id] = count + 1
            else:
                deferred.append(item)
            if len(selected) >= top_k:
                return self._protect_reranked_top_k_candidates(selected, candidates, top_k)

        for item in [*deferred, *weak_diversity_deferred]:
            if len(selected) >= top_k:
                break
            selected.append(item)
        return self._protect_reranked_top_k_candidates(selected, candidates, top_k)

    @staticmethod
    def _final_quality_score(item: RerankCandidate) -> float:
        return item.rerank_score if item.rerank_score > 0 else item.fused_score

    @staticmethod
    def _is_final_replacement_protected(item: RerankCandidate) -> bool:
        return bool(item.sources.intersection(FINAL_REPLACEMENT_PROTECTED_SOURCE_NAMES))

    def _should_defer_weak_diversity_candidate(
        self,
        item: RerankCandidate,
        deferred: list[RerankCandidate],
    ) -> bool:
        if not deferred or item.fused_score > DOCUMENT_DIVERSITY_WEAK_FUSED_CUTOFF:
            return False
        strongest_deferred = max(deferred, key=lambda candidate: (self._final_quality_score(candidate), candidate.fused_score))
        if strongest_deferred.fused_score < DOCUMENT_DIVERSITY_STRONG_DEFERRED_FUSED:
            return False
        return self._final_quality_score(strongest_deferred) >= (
            self._final_quality_score(item) + DOCUMENT_DIVERSITY_RERANK_MARGIN
        )

    def _protect_reranked_top_k_candidates(
        self,
        selected: list[RerankCandidate],
        candidates: list[RerankCandidate],
        top_k: int,
    ) -> list[RerankCandidate]:
        if len(selected) < top_k:
            return selected
        protected_slots = min(
            max(0, int(getattr(self.settings, "retrieval_document_diversity_protected_top_k_slots", 0) or 0)),
            top_k,
        )
        if protected_slots <= 0:
            return selected

        original_rank = {item.candidate.chunk_id: index for index, item in enumerate(candidates)}
        selected_ids = {item.candidate.chunk_id for item in selected}
        protected_candidates = [
            item
            for item in candidates[:top_k]
            if item.candidate.chunk_id not in selected_ids
        ][:protected_slots]
        if not protected_candidates:
            return selected

        protected_ids = {item.candidate.chunk_id for item in protected_candidates}
        protected_rank_cutoff = max(original_rank[item.candidate.chunk_id] for item in protected_candidates)
        replacement_indexes = [
            index
            for index, item in enumerate(selected)
            if original_rank.get(item.candidate.chunk_id, top_k + index) > protected_rank_cutoff
            and item.candidate.chunk_id not in protected_ids
        ]
        if len(replacement_indexes) < len(protected_candidates):
            replacement_indexes = list(range(len(selected) - 1, -1, -1))

        updated = list(selected)
        for replacement_index, candidate in zip(replacement_indexes[-len(protected_candidates) :], protected_candidates, strict=False):
            updated[replacement_index] = candidate

        seen: set[UUID] = set()
        deduped: list[RerankCandidate] = []
        for item in sorted(updated, key=lambda item: original_rank.get(item.candidate.chunk_id, len(candidates))):
            if item.candidate.chunk_id in seen:
                continue
            seen.add(item.candidate.chunk_id)
            deduped.append(item)
        return deduped[:top_k]

    @staticmethod
    def _has_preservable_source(item: RerankCandidate) -> bool:
        return bool(
            item.sources.intersection(
                {
                    "lexical",
                    "indexed_sparse",
                    "structural",
                    "document_expansion",
                    "document_sweep",
                    SUBQUERY_DOCUMENT_EVIDENCE_SOURCE,
                    SUBQUERY_NEIGHBOR_CONTEXT_SOURCE,
                    "document_first_evidence",
                    "document_neighbor_context",
                }
            )
        )

    @staticmethod
    def _evidence_preservation_score(item: RerankCandidate) -> float:
        score = (item.fused_score * 0.46) + (item.lexical_norm * 0.42) + (min(item.lexical_raw, 1.0) * 0.12)
        if "structural" in item.sources:
            score += 0.06
        if "document_sweep" in item.sources:
            score += 0.04
        if SUBQUERY_DOCUMENT_EVIDENCE_SOURCE in item.sources:
            score += 0.05
        if SUBQUERY_NEIGHBOR_CONTEXT_SOURCE in item.sources:
            score += 0.04
        if "document_first_evidence" in item.sources:
            score += 0.04
        if "document_neighbor_context" in item.sources:
            score += 0.02
        if "document_expansion" in item.sources:
            score += 0.02
        return score

    def _fuse_hits(
        self,
        lexical_hits: list[RetrievalCandidate],
        vector_hits: list[RetrievalCandidate],
        *,
        structural_hits: list[RetrievalCandidate] | None = None,
        indexed_sparse_hits: list[RetrievalCandidate] | None = None,
        expansion_hits: list[RetrievalCandidate] | None = None,
        document_sweep_hits: list[RetrievalCandidate] | None = None,
        subquery_document_hits: list[RetrievalCandidate] | None = None,
        subquery_neighbor_hits: list[RetrievalCandidate] | None = None,
        document_first_hits: list[RetrievalCandidate] | None = None,
        neighbor_hits: list[RetrievalCandidate] | None = None,
    ) -> dict:
        combined: dict = {}

        for hit in lexical_hits:
            current = combined.setdefault(hit.chunk_id, RerankCandidate(candidate=hit))
            current.lexical_raw = hit.lexical_score or 0.0
            current.sources.add("lexical")
        for hit in structural_hits or []:
            current = combined.setdefault(hit.chunk_id, RerankCandidate(candidate=hit))
            if current.candidate.content != hit.content:
                current.candidate = hit
            current.lexical_raw = max(current.lexical_raw, hit.lexical_score or 0.0)
            current.sources.add("structural")
        for hit in indexed_sparse_hits or []:
            current = combined.setdefault(hit.chunk_id, RerankCandidate(candidate=hit))
            if current.candidate.content != hit.content:
                current.candidate = hit
            sparse_weight = max(0.0, min(float(getattr(self.settings, "retrieval_indexed_sparse_score_weight", 0.45)), 1.0))
            current.lexical_raw = max(current.lexical_raw, (hit.lexical_score or 0.0) * sparse_weight)
            current.sources.add("indexed_sparse")
        for hit in expansion_hits or []:
            current = combined.setdefault(hit.chunk_id, RerankCandidate(candidate=hit))
            if current.candidate.content != hit.content:
                current.candidate = hit
            expansion_weight = max(0.0, min(float(getattr(self.settings, "retrieval_in_document_expansion_score_weight", 0.35)), 1.0))
            current.lexical_raw = max(current.lexical_raw, (hit.lexical_score or 0.0) * expansion_weight)
            current.sources.add("document_expansion")
        for hit in document_sweep_hits or []:
            current = combined.setdefault(hit.chunk_id, RerankCandidate(candidate=hit))
            if current.candidate.content != hit.content:
                current.candidate = hit
            sweep_weight = max(0.0, min(float(getattr(self.settings, "retrieval_document_evidence_sweep_score_weight", 0.35)), 1.0))
            current.lexical_raw = max(current.lexical_raw, (hit.lexical_score or 0.0) * sweep_weight)
            current.sources.add("document_sweep")
        for hit in subquery_document_hits or []:
            current = combined.setdefault(hit.chunk_id, RerankCandidate(candidate=hit))
            if current.candidate.content != hit.content:
                current.candidate = hit
            subquery_weight = max(
                0.0,
                min(float(getattr(self.settings, "retrieval_subquery_document_evidence_score_weight", 0.42)), 1.0),
            )
            current.lexical_raw = max(current.lexical_raw, (hit.lexical_score or 0.0) * subquery_weight)
            current.sources.add(SUBQUERY_DOCUMENT_EVIDENCE_SOURCE)
        for hit in subquery_neighbor_hits or []:
            current = combined.setdefault(hit.chunk_id, RerankCandidate(candidate=hit))
            if current.candidate.content != hit.content:
                current.candidate = hit
            subquery_neighbor_weight = max(
                0.0,
                min(float(getattr(self.settings, "retrieval_subquery_neighbor_context_score_weight", 0.55)), 1.0),
            )
            current.lexical_raw = max(current.lexical_raw, (hit.lexical_score or 0.0) * subquery_neighbor_weight)
            current.sources.add(SUBQUERY_NEIGHBOR_CONTEXT_SOURCE)
        for hit in document_first_hits or []:
            current = combined.setdefault(hit.chunk_id, RerankCandidate(candidate=hit))
            if current.candidate.content != hit.content:
                current.candidate = hit
            document_first_weight = max(0.0, min(float(getattr(self.settings, "retrieval_document_first_evidence_score_weight", 0.34)), 1.0))
            current.lexical_raw = max(current.lexical_raw, (hit.lexical_score or 0.0) * document_first_weight)
            current.sources.add("document_first_evidence")
        for hit in neighbor_hits or []:
            current = combined.setdefault(hit.chunk_id, RerankCandidate(candidate=hit))
            if current.candidate.content != hit.content:
                current.candidate = hit
            neighbor_weight = max(0.0, min(float(getattr(self.settings, "retrieval_document_neighbor_context_score_weight", 0.28)), 1.0))
            current.lexical_raw = max(current.lexical_raw, (hit.lexical_score or 0.0) * neighbor_weight)
            current.sources.add("document_neighbor_context")
        for hit in vector_hits:
            current = combined.setdefault(hit.chunk_id, RerankCandidate(candidate=hit))
            if current.candidate.content != hit.content:
                current.candidate = hit
            current.vector_raw = hit.vector_score or 0.0
            current.sources.add("vector")

        self._normalize_scores(combined.values(), score_field="lexical_raw", normalized_field="lexical_norm")
        self._normalize_scores(combined.values(), score_field="vector_raw", normalized_field="vector_norm")

        if self._use_rrf_fusion():
            self._apply_rrf_scores(
                combined,
                lexical_hits=lexical_hits,
                vector_hits=vector_hits,
                structural_hits=structural_hits or [],
                indexed_sparse_hits=indexed_sparse_hits or [],
                expansion_hits=expansion_hits or [],
                document_sweep_hits=document_sweep_hits or [],
                subquery_document_hits=subquery_document_hits or [],
                subquery_neighbor_hits=subquery_neighbor_hits or [],
                document_first_hits=document_first_hits or [],
                neighbor_hits=neighbor_hits or [],
            )
        else:
            self._apply_weighted_scores(combined)
        return combined

    def _apply_weighted_scores(self, combined: dict) -> None:
        for item in combined.values():
            item.fused_score = (LEXICAL_WEIGHT * item.lexical_norm) + (VECTOR_WEIGHT * item.vector_norm)
            if "structural" in item.sources:
                item.fused_score = max(item.fused_score, min(1.0, item.lexical_norm * 0.72))
            if item.sources == {"lexical"}:
                item.fused_score = max(item.fused_score, item.lexical_norm * LEXICAL_WEIGHT)
            if item.sources == {"vector"}:
                item.fused_score = max(item.fused_score, item.vector_norm * VECTOR_WEIGHT)

    def _apply_rrf_scores(
        self,
        combined: dict,
        *,
        lexical_hits: list[RetrievalCandidate],
        vector_hits: list[RetrievalCandidate],
        structural_hits: list[RetrievalCandidate],
        indexed_sparse_hits: list[RetrievalCandidate],
        expansion_hits: list[RetrievalCandidate],
        document_sweep_hits: list[RetrievalCandidate],
        subquery_document_hits: list[RetrievalCandidate],
        subquery_neighbor_hits: list[RetrievalCandidate],
        document_first_hits: list[RetrievalCandidate],
        neighbor_hits: list[RetrievalCandidate],
    ) -> None:
        rrf_scores = {chunk_id: 0.0 for chunk_id in combined}
        rrf_k = self._rrf_k()
        for hits, weight in [
            (structural_hits, RRF_STRUCTURAL_WEIGHT),
            (lexical_hits, RRF_LEXICAL_WEIGHT),
            (indexed_sparse_hits, RRF_INDEXED_SPARSE_WEIGHT),
            (vector_hits, RRF_VECTOR_WEIGHT),
            (expansion_hits, RRF_EXPANSION_WEIGHT),
            (document_sweep_hits, RRF_DOCUMENT_SWEEP_WEIGHT),
            (subquery_document_hits, RRF_DOCUMENT_SWEEP_WEIGHT),
            (subquery_neighbor_hits, RRF_EXPANSION_WEIGHT),
            (document_first_hits, RRF_DOCUMENT_SWEEP_WEIGHT),
            (neighbor_hits, RRF_EXPANSION_WEIGHT),
        ]:
            for rank, hit in enumerate(hits, start=1):
                if hit.chunk_id not in rrf_scores:
                    continue
                rrf_scores[hit.chunk_id] += weight / (rrf_k + rank)
        max_score = max(rrf_scores.values(), default=0.0)
        for item in combined.values():
            normalized_rrf = (rrf_scores.get(item.candidate.chunk_id, 0.0) / max_score) if max_score > 0 else 0.0
            score_floor = 0.0
            if "structural" in item.sources:
                score_floor = max(score_floor, min(1.0, item.lexical_norm * 0.9))
            if "lexical" in item.sources:
                score_floor = max(score_floor, item.lexical_norm * LEXICAL_WEIGHT)
            if "vector" in item.sources:
                score_floor = max(score_floor, item.vector_norm * VECTOR_WEIGHT)
            item.fused_score = max(normalized_rrf, score_floor)

    def _rrf_k(self) -> int:
        return max(1, int(getattr(self.settings, "retrieval_rrf_k", 60) or 60))

    def _use_rrf_fusion(self) -> bool:
        return str(getattr(self.settings, "retrieval_fusion_strategy", "weighted") or "weighted").strip().lower() == "rrf"

    def _fusion_strategy_name(self) -> str:
        if self._use_rrf_fusion():
            return f"rrf(k={self._rrf_k()}) + multi-query lexical + structural"
        return "min-max weighted sum + multi-query lexical + structural"

    @staticmethod
    def _normalize_scores(items: Iterable[RerankCandidate], score_field: str, normalized_field: str) -> None:
        values = [getattr(item, score_field) for item in items]
        if not values:
            return
        maximum = max(values)
        minimum = min(values)
        if maximum == minimum:
            for item in items:
                raw_value = getattr(item, score_field)
                setattr(item, normalized_field, 1.0 if raw_value > 0 else 0.0)
            return
        for item in items:
            raw_value = getattr(item, score_field)
            normalized = (raw_value - minimum) / (maximum - minimum) if raw_value > 0 else 0.0
            setattr(item, normalized_field, normalized)

    @staticmethod
    def _to_schema(item: RerankCandidate) -> SearchResultChunk:
        candidate = item.candidate
        return SearchResultChunk(
            chunk_id=candidate.chunk_id,
            document_id=candidate.document_id,
            document_title=candidate.document_title,
            document_version_id=candidate.document_version_id,
            version_number=candidate.version_number,
            chunk_index=candidate.chunk_index,
            content=candidate.content,
            preview=candidate.content[:240],
            section_title=candidate.section_title,
            page_number_start=candidate.page_number_start,
            page_number_end=candidate.page_number_end,
            paragraph_start=candidate.paragraph_start,
            paragraph_end=candidate.paragraph_end,
            char_start=candidate.char_start,
            char_end=candidate.char_end,
            clause_full_name=candidate.clause_full_name,
            article_number=candidate.article_number,
            chunk_type=candidate.chunk_type,
            heading_path=candidate.heading_path,
            citation_metadata=candidate.citation_metadata,
            score=SearchScoreBreakdown(
                lexical_raw=item.lexical_raw,
                lexical_normalized=item.lexical_norm,
                vector_raw=item.vector_raw,
                vector_normalized=item.vector_norm,
                fused=item.fused_score,
                rerank=item.rerank_score,
            ),
            citation_preview={
                "document_title": candidate.document_title,
                "version_number": candidate.version_number,
                "chunk_id": str(candidate.chunk_id),
                "section_title": candidate.section_title,
                "page_number_start": candidate.page_number_start,
                "page_number_end": candidate.page_number_end,
                "paragraph_start": candidate.paragraph_start,
                "paragraph_end": candidate.paragraph_end,
                "clause_full_name": candidate.clause_full_name,
                "article_number": candidate.article_number,
                "chunk_type": candidate.chunk_type,
                "heading_path": candidate.heading_path,
                "preview": candidate.content[:240],
            },
        )
