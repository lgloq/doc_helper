from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from time import perf_counter
from typing import Iterable
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.user import User
from app.repositories.retrieval_repository import RetrievalCandidate, RetrievalRepository
from app.schemas.search import SearchDebugInfo, SearchRequest, SearchResponse, SearchResultChunk, SearchScoreBreakdown
from app.services.ingestion.embeddings import EmbeddingProviderFactory
from app.services.permissions.service import PermissionFilterBuilder
from app.services.retrieval.query_optimizer import QueryOptimizer, QueryOptimizationPlan, QueryPlanCandidate
from app.services.retrieval.reranker import RerankCandidate, RerankerFactory

LEXICAL_WEIGHT = 0.45
VECTOR_WEIGHT = 0.55
PLAN_PROBE_LIMIT = 6


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


class RetrievalService:
    def __init__(self, session: Session):
        self.session = session
        self.permission_builder = PermissionFilterBuilder()
        self.retrieval_repository = RetrievalRepository(session)
        self.embedding_provider = EmbeddingProviderFactory.create()
        self.reranker = RerankerFactory.create()
        self.query_optimizer = QueryOptimizer()

    def search(
        self,
        actor: User,
        payload: SearchRequest,
        scoped_document_ids: list[UUID] | None = None,
        target_document_title: str | None = None,
    ) -> SearchResponse:
        search_started = perf_counter()
        accessible_document_ids = self.permission_builder.resolve_accessible_document_ids(self.session, actor, require_manage=False)
        if scoped_document_ids is not None:
            scoped_set = {item for item in scoped_document_ids}
            accessible_document_ids = [item for item in accessible_document_ids if item in scoped_set]
        query_plan = self.query_optimizer.build(payload.query, target_document_title=target_document_title)
        probe_applied = False
        if len(query_plan.candidates) > 1 and accessible_document_ids:
            probe_applied = self._select_best_query_plan(
                query_plan,
                accessible_document_ids=accessible_document_ids,
                target_document_title=target_document_title,
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
                    search_total_latency_ms=int((perf_counter() - search_started) * 1000),
                    query_plan_candidate_count=query_plan.candidate_count,
                    query_plan_selected=query_plan.selected_candidate.label,
                    query_plan_selection_reason=query_plan.selected_candidate_reason,
                    query_plan_probe_applied=probe_applied,
                ),
            )

        candidate_pool = max(payload.top_k * 5, 20)
        lexical_started = perf_counter()
        lexical_hits = self._collect_lexical_hits(query_plan.lexical_queries, accessible_document_ids, candidate_pool)
        lexical_latency_ms = int((perf_counter() - lexical_started) * 1000)
        vector_embedding_started = perf_counter()
        query_embedding = self.embedding_provider.embed_texts([query_plan.retrieval_query])[0]
        vector_embedding_latency_ms = int((perf_counter() - vector_embedding_started) * 1000)
        vector_retrieval_started = perf_counter()
        vector_hits = self.retrieval_repository.search_vector(query_embedding, accessible_document_ids, candidate_pool)
        vector_retrieval_latency_ms = int((perf_counter() - vector_retrieval_started) * 1000)
        fusion_started = perf_counter()
        fused_candidates = self._fuse_hits(lexical_hits, vector_hits)
        fusion_latency_ms = int((perf_counter() - fusion_started) * 1000)
        target_document_id = scoped_document_ids[0] if scoped_document_ids and len(scoped_document_ids) == 1 else None
        rerank_started = perf_counter()
        reranked = self.reranker.rerank(
            payload.query,
            list(fused_candidates.values()),
            payload.top_k,
            target_document_id=target_document_id,
        )
        rerank_latency_ms = int((perf_counter() - rerank_started) * 1000)
        search_total_latency_ms = int((perf_counter() - search_started) * 1000)

        return SearchResponse(
            query=payload.query,
            top_k=payload.top_k,
            matched_chunks=[self._to_schema(candidate) for candidate in reranked.candidates],
            debug=SearchDebugInfo(
                accessible_document_count=len(accessible_document_ids),
                lexical_candidate_count=len(lexical_hits),
                vector_candidate_count=len(vector_hits),
                fusion_strategy="min-max weighted sum + multi-query lexical",
                pre_rerank_count=reranked.pre_rerank_count,
                post_rerank_count=reranked.post_rerank_count,
                rerank_strategy=reranked.strategy,
                retrieval_query=query_plan.retrieval_query,
                lexical_queries=query_plan.lexical_queries,
                query_rewrite_applied=query_plan.rewrite_applied,
                query_rewrite_strategies=query_plan.applied_strategies,
                query_rewrite_provider=query_plan.rewrite_provider,
                query_rewrite_model=query_plan.rewrite_model,
                query_rewrite_latency_ms=query_plan.rewrite_latency_ms,
                lexical_retrieval_latency_ms=lexical_latency_ms,
                vector_embedding_latency_ms=vector_embedding_latency_ms,
                vector_retrieval_latency_ms=vector_retrieval_latency_ms,
                fusion_latency_ms=fusion_latency_ms,
                rerank_latency_ms=rerank_latency_ms,
                search_total_latency_ms=search_total_latency_ms,
                query_plan_candidate_count=query_plan.candidate_count,
                query_plan_selected=query_plan.selected_candidate.label,
                query_plan_selection_reason=query_plan.selected_candidate_reason,
                query_plan_probe_applied=probe_applied,
            ),
        )

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
        hits = self._collect_lexical_hits(candidate.lexical_queries[:3], accessible_document_ids, PLAN_PROBE_LIMIT)
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

    def _collect_lexical_hits(
        self,
        queries: list[str],
        accessible_document_ids: list[UUID],
        candidate_pool: int,
    ) -> list[RetrievalCandidate]:
        merged: dict[UUID, RetrievalCandidate] = {}
        for query in queries:
            for candidate in self.retrieval_repository.search_lexical(query, accessible_document_ids, candidate_pool):
                existing = merged.get(candidate.chunk_id)
                if existing is None or (candidate.lexical_score or 0.0) > (existing.lexical_score or 0.0):
                    merged[candidate.chunk_id] = candidate
        hits = list(merged.values())
        hits.sort(key=lambda item: ((item.lexical_score or 0.0), -item.chunk_index), reverse=True)
        return hits[:candidate_pool]

    def _fuse_hits(
        self,
        lexical_hits: list[RetrievalCandidate],
        vector_hits: list[RetrievalCandidate],
    ) -> dict:
        combined: dict = {}

        for hit in lexical_hits:
            current = combined.setdefault(hit.chunk_id, RerankCandidate(candidate=hit))
            current.lexical_raw = hit.lexical_score or 0.0
            current.sources.add("lexical")
        for hit in vector_hits:
            current = combined.setdefault(hit.chunk_id, RerankCandidate(candidate=hit))
            if current.candidate.content != hit.content:
                current.candidate = hit
            current.vector_raw = hit.vector_score or 0.0
            current.sources.add("vector")

        self._normalize_scores(combined.values(), score_field="lexical_raw", normalized_field="lexical_norm")
        self._normalize_scores(combined.values(), score_field="vector_raw", normalized_field="vector_norm")

        for item in combined.values():
            item.fused_score = (LEXICAL_WEIGHT * item.lexical_norm) + (VECTOR_WEIGHT * item.vector_norm)
            if item.sources == {"lexical"}:
                item.fused_score = max(item.fused_score, item.lexical_norm * LEXICAL_WEIGHT)
            if item.sources == {"vector"}:
                item.fused_score = max(item.fused_score, item.vector_norm * VECTOR_WEIGHT)
        return combined

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
                "preview": candidate.content[:240],
            },
        )
