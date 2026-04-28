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
    citation_metadata: dict | None = None
    citation_preview: dict
    score: SearchScoreBreakdown


class SearchDebugInfo(BaseModel):
    accessible_document_count: int
    lexical_candidate_count: int
    vector_candidate_count: int
    fusion_strategy: str
    pre_rerank_count: int = 0
    post_rerank_count: int = 0
    rerank_strategy: str = "none"


class SearchResponse(BaseModel):
    query: str
    top_k: int
    matched_chunks: list[SearchResultChunk]
    debug: SearchDebugInfo
