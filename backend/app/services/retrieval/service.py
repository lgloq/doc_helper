from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from sqlalchemy.orm import Session

from app.models.user import User
from app.repositories.retrieval_repository import RetrievalCandidate, RetrievalRepository
from app.schemas.search import SearchDebugInfo, SearchRequest, SearchResponse, SearchResultChunk, SearchScoreBreakdown
from app.services.ingestion.embeddings import EmbeddingProviderFactory
from app.services.permissions.service import PermissionFilterBuilder

LEXICAL_WEIGHT = 0.45
VECTOR_WEIGHT = 0.55


@dataclass
class CombinedCandidate:
    candidate: RetrievalCandidate
    lexical_raw: float = 0.0
    vector_raw: float = 0.0
    lexical_norm: float = 0.0
    vector_norm: float = 0.0
    fused_score: float = 0.0
    sources: set[str] = field(default_factory=set)


class RetrievalService:
    def __init__(self, session: Session):
        self.session = session
        self.permission_builder = PermissionFilterBuilder()
        self.retrieval_repository = RetrievalRepository(session)
        self.embedding_provider = EmbeddingProviderFactory.create()

    def search(self, actor: User, payload: SearchRequest) -> SearchResponse:
        accessible_document_ids = self.permission_builder.resolve_accessible_document_ids(self.session, actor, require_manage=False)
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
                ),
            )

        candidate_pool = max(payload.top_k * 5, 20)
        lexical_hits = self.retrieval_repository.search_lexical(payload.query, accessible_document_ids, candidate_pool)
        query_embedding = self.embedding_provider.embed_texts([payload.query])[0]
        vector_hits = self.retrieval_repository.search_vector(query_embedding, accessible_document_ids, candidate_pool)
        fused_candidates = self._fuse_hits(lexical_hits, vector_hits)
        top_candidates = sorted(fused_candidates.values(), key=lambda item: item.fused_score, reverse=True)[: payload.top_k]

        return SearchResponse(
            query=payload.query,
            top_k=payload.top_k,
            matched_chunks=[self._to_schema(candidate) for candidate in top_candidates],
            debug=SearchDebugInfo(
                accessible_document_count=len(accessible_document_ids),
                lexical_candidate_count=len(lexical_hits),
                vector_candidate_count=len(vector_hits),
                fusion_strategy="min-max weighted sum",
            ),
        )

    def _fuse_hits(
        self,
        lexical_hits: list[RetrievalCandidate],
        vector_hits: list[RetrievalCandidate],
    ) -> dict:
        combined: dict = {}

        for hit in lexical_hits:
            current = combined.setdefault(hit.chunk_id, CombinedCandidate(candidate=hit))
            current.lexical_raw = hit.lexical_score or 0.0
            current.sources.add("lexical")
        for hit in vector_hits:
            current = combined.setdefault(hit.chunk_id, CombinedCandidate(candidate=hit))
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
    def _normalize_scores(items: Iterable[CombinedCandidate], score_field: str, normalized_field: str) -> None:
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
    def _to_schema(item: CombinedCandidate) -> SearchResultChunk:
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
