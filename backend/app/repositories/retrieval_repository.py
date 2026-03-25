from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence
from uuid import UUID

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.models.chunk import Chunk
from app.models.document import Document, DocumentVersion


@dataclass
class RetrievalCandidate:
    chunk_id: UUID
    document_id: UUID
    document_title: str
    document_version_id: UUID
    version_number: int
    chunk_index: int
    content: str
    token_count: int
    section_title: str | None
    page_number_start: int | None
    page_number_end: int | None
    paragraph_start: int | None
    paragraph_end: int | None
    char_start: int | None
    char_end: int | None
    citation_metadata: dict | None
    lexical_score: float | None = None
    vector_score: float | None = None


class RetrievalRepository:
    def __init__(self, session: Session):
        self.session = session

    def search_lexical(self, query_text: str, accessible_document_ids: Sequence[UUID], limit: int) -> list[RetrievalCandidate]:
        if not accessible_document_ids:
            return []
        if self.session.bind and self.session.bind.dialect.name == "postgresql":
            return self._search_lexical_postgres(query_text, accessible_document_ids, limit)
        return self._search_lexical_python(query_text, accessible_document_ids, limit)

    def search_vector(self, query_embedding: list[float], accessible_document_ids: Sequence[UUID], limit: int) -> list[RetrievalCandidate]:
        if not accessible_document_ids:
            return []
        if self.session.bind and self.session.bind.dialect.name == "postgresql":
            try:
                return self._search_vector_postgres(query_embedding, accessible_document_ids, limit)
            except AttributeError:
                pass
        return self._search_vector_python(query_embedding, accessible_document_ids, limit)

    def _base_current_version_statement(self, accessible_document_ids: Sequence[UUID]):
        return (
            select(
                Chunk,
                Document.title.label("document_title"),
                DocumentVersion.version_number.label("version_number"),
            )
            .join(Document, Document.id == Chunk.document_id)
            .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
            .where(Chunk.document_id.in_(accessible_document_ids))
            .where(Document.current_version_id == Chunk.document_version_id)
        )

    def _search_lexical_postgres(self, query_text: str, accessible_document_ids: Sequence[UUID], limit: int) -> list[RetrievalCandidate]:
        ts_vector = func.to_tsvector("simple", Chunk.content)
        ts_query = func.plainto_tsquery("simple", query_text)
        rank_expr = func.ts_rank_cd(ts_vector, ts_query).label("lexical_score")
        statement = (
            self._base_current_version_statement(accessible_document_ids)
            .add_columns(rank_expr)
            .where(ts_vector.op("@@")(ts_query))
            .order_by(desc(rank_expr), Chunk.chunk_index.asc())
            .limit(limit)
        )
        rows = self.session.execute(statement).all()
        return [self._row_to_candidate(row, lexical_score=float(row.lexical_score or 0.0)) for row in rows]

    def _search_vector_postgres(self, query_embedding: list[float], accessible_document_ids: Sequence[UUID], limit: int) -> list[RetrievalCandidate]:
        distance_expr = Chunk.embedding.cosine_distance(query_embedding)
        score_expr = (1 - distance_expr).label("vector_score")
        statement = (
            self._base_current_version_statement(accessible_document_ids)
            .add_columns(score_expr)
            .where(Chunk.embedding.is_not(None))
            .order_by(distance_expr.asc(), Chunk.chunk_index.asc())
            .limit(limit)
        )
        rows = self.session.execute(statement).all()
        return [self._row_to_candidate(row, vector_score=float(row.vector_score or 0.0)) for row in rows]

    def _search_lexical_python(self, query_text: str, accessible_document_ids: Sequence[UUID], limit: int) -> list[RetrievalCandidate]:
        query_terms = self._tokenize(query_text)
        if not query_terms:
            return []
        candidates = []
        for row in self.session.execute(self._base_current_version_statement(accessible_document_ids)).all():
            content_terms = self._tokenize(row.Chunk.content)
            if not content_terms:
                continue
            match_count = sum(content_terms.count(term) for term in query_terms)
            if match_count <= 0:
                continue
            lexical_score = match_count / max(len(query_terms), 1)
            candidates.append(self._row_to_candidate(row, lexical_score=float(lexical_score)))
        candidates.sort(key=lambda item: (item.lexical_score or 0.0, -item.chunk_index), reverse=True)
        return candidates[:limit]

    def _search_vector_python(self, query_embedding: list[float], accessible_document_ids: Sequence[UUID], limit: int) -> list[RetrievalCandidate]:
        candidates = []
        for row in self.session.execute(self._base_current_version_statement(accessible_document_ids)).all():
            embedding = row.Chunk.embedding
            if not embedding:
                continue
            vector_score = self._cosine_similarity(query_embedding, embedding)
            candidates.append(self._row_to_candidate(row, vector_score=float(vector_score)))
        candidates.sort(key=lambda item: (item.vector_score or 0.0, -item.chunk_index), reverse=True)
        return candidates[:limit]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token for token in re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", text.lower()) if token]

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
        right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
        return numerator / (left_norm * right_norm)

    @staticmethod
    def _row_to_candidate(row, lexical_score: float | None = None, vector_score: float | None = None) -> RetrievalCandidate:
        chunk = row.Chunk
        return RetrievalCandidate(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_title=row.document_title,
            document_version_id=chunk.document_version_id,
            version_number=row.version_number,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            token_count=chunk.token_count,
            section_title=chunk.section_title,
            page_number_start=chunk.page_number_start,
            page_number_end=chunk.page_number_end,
            paragraph_start=chunk.paragraph_start,
            paragraph_end=chunk.paragraph_end,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            citation_metadata=chunk.citation_metadata,
            lexical_score=lexical_score,
            vector_score=vector_score,
        )
