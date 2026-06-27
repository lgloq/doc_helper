from __future__ import annotations

from pydantic import BaseModel, Field
from uuid import UUID


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchScoreBreakdown(BaseModel):
    lexical_raw: float
    lexical_normalized: float
    vector_raw: float
    vector_normalized: float
    fused: float
    rerank: float | None = None


class SearchResultChunk(BaseModel):
    chunk_id: UUID
    document_id: UUID
    document_title: str
    document_version_id: UUID
    version_number: int
    chunk_index: int
    content: str
    preview: str
    section_title: str | None = None
    page_number_start: int | None = None
    page_number_end: int | None = None
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    clause_full_name: str | None = None
    article_number: str | None = None
    chunk_type: str | None = None
    heading_path: str | None = None
    citation_metadata: dict | None = None
    citation_preview: dict
    score: SearchScoreBreakdown


class SearchDebugInfo(BaseModel):
    accessible_document_count: int
    lexical_candidate_count: int
    vector_candidate_count: int
    structural_candidate_count: int = 0
    fusion_strategy: str
    pre_rerank_count: int = 0
    post_rerank_count: int = 0
    rerank_strategy: str = "none"
    retrieval_query: str | None = None
    lexical_queries: list[str] = Field(default_factory=list)
    query_rewrite_applied: bool = False
    query_rewrite_strategies: list[str] = Field(default_factory=list)
    query_rewrite_provider: str | None = None
    query_rewrite_model: str | None = None
    query_rewrite_latency_ms: int | None = None
    llm_rewrite_attempted: bool = False
    llm_rewrite_skipped_reason: str | None = None
    llm_rewrite_latency_ms: int | None = None
    query_decomposition_applied: bool = False
    subquery_count: int = 0
    subquery_candidate_counts: list[dict[str, object]] = Field(default_factory=list)
    subquery_timeout_count: int = 0
    subquery_timeout_fallback_candidate_count: int = 0
    permission_filter_latency_ms: int | None = None
    lexical_retrieval_latency_ms: int | None = None
    indexed_sparse_candidate_count: int = 0
    indexed_sparse_retrieval_latency_ms: int | None = None
    structural_retrieval_latency_ms: int | None = None
    structural_retrieval_skipped: bool = False
    structural_retrieval_skip_reason: str | None = None
    structural_retrieval_timeout: bool = False
    vector_embedding_latency_ms: int | None = None
    vector_retrieval_latency_ms: int | None = None
    vector_retrieval_skipped: bool = False
    vector_retrieval_skip_reason: str | None = None
    vector_retrieval_timeout: bool = False
    expansion_candidate_count: int = 0
    in_document_expansion_latency_ms: int | None = None
    document_evidence_sweep_candidate_count: int = 0
    document_evidence_sweep_latency_ms: int | None = None
    document_evidence_sweep_skipped: bool = False
    document_evidence_sweep_skip_reason: str | None = None
    subquery_document_evidence_candidate_count: int = 0
    subquery_document_evidence_latency_ms: int | None = None
    subquery_neighbor_context_candidate_count: int = 0
    subquery_neighbor_context_latency_ms: int | None = None
    document_first_evidence_candidate_count: int = 0
    document_first_evidence_latency_ms: int | None = None
    document_neighbor_context_candidate_count: int = 0
    document_neighbor_context_latency_ms: int | None = None
    evidence_preservation_candidate_count: int = 0
    evidence_preservation_selected_count: int = 0
    final_coverage_candidate_count: int = 0
    final_coverage_selected_count: int = 0
    subquery_final_coverage_candidate_count: int = 0
    subquery_final_coverage_selected_count: int = 0
    fusion_latency_ms: int | None = None
    rerank_latency_ms: int | None = None
    search_total_latency_ms: int | None = None
    query_plan_candidate_count: int = 1
    query_plan_selected: str | None = None
    query_plan_selection_reason: str | None = None
    query_plan_probe_applied: bool = False
    query_plan_probe_latency_ms: int | None = None
    query_plan_probe_skipped_reason: str | None = None
    permission_probe_early_stop_applied: bool = False
    permission_probe_target_hint: str | None = None
    permission_probe_accessible_target_count: int = 0
    permission_probe_inaccessible_target_count: int = 0
    permission_refusal_reason_code: str | None = None
    permission_refusal_reason: str | None = None


class SearchResponse(BaseModel):
    query: str
    top_k: int
    matched_chunks: list[SearchResultChunk]
    debug: SearchDebugInfo
