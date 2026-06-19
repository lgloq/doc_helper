from __future__ import annotations

from collections import Counter
import math
import re
from dataclasses import dataclass, replace
from typing import Sequence
from uuid import UUID

from sqlalchemy import desc, func, or_, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.chunk import Chunk
from app.models.document import Document, DocumentVersion
from app.services.ingestion.search_index import build_weighted_lexical_terms, parse_lexical_search_text, tokenize_search_text
from app.services.ingestion.structure import article_numbers_in_text, normalize_structural_key


CJK_SQL_LOW_INFORMATION_SUBSTRINGS = (
    "有限公司",
    "集团有限",
    "股份有限",
    "有限责任",
    "责任公司",
    "公司的",
    "限公司",
    "公司和",
    "公司两",
    "集团有",
    "材料中",
    "材料在",
    "文件中",
    "制度中",
    "办法中",
    "财务披露",
    "资与财务",
    "融资与财",
    "披露材料",
    "露材料",
    "露上的披",
    "露材料中",
    "露或规定",
    "或规定的",
    "具体是",
    "是什么",
    "怎么",
    "如何",
)


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
    clause_full_name: str | None = None
    article_number: str | None = None
    chunk_type: str | None = None
    heading_path: str | None = None
    structural_search_text: str | None = None
    lexical_search_text: str | None = None
    lexical_score: float | None = None
    vector_score: float | None = None


@dataclass(frozen=True)
class _LexicalIndexEntry:
    candidate: RetrievalCandidate
    terms: Counter[str]


@dataclass(frozen=True)
class _StructuralIndexEntry:
    candidate: RetrievalCandidate
    normalized_fields: dict[str, str]
    terms: Counter[str]


class RetrievalRepository:
    def __init__(self, session: Session):
        self.session = session
        self.settings = get_settings()
        self._lexical_cache: dict[tuple[UUID, UUID], Counter[str]] = {}
        self._lexical_index_cache: dict[tuple[UUID, ...], list[_LexicalIndexEntry]] = {}
        self._structural_index_cache: dict[tuple[UUID, ...], list[_StructuralIndexEntry]] = {}
        self._lexical_df_cache: dict[tuple[tuple[UUID, ...], str], int] = {}
        self._lexical_scope_size_cache: dict[tuple[UUID, ...], int] = {}
        self._lexical_avg_doc_len_cache: dict[tuple[UUID, ...], float] = {}

    def search_lexical(self, query_text: str, accessible_document_ids: Sequence[UUID], limit: int) -> list[RetrievalCandidate]:
        if not accessible_document_ids:
            return []
        if self.session.bind and self.session.bind.dialect.name == "postgresql":
            postgres_hits = self._search_lexical_postgres(query_text, accessible_document_ids, limit)
            if self._should_use_cjk_python_fallback(query_text, postgres_hits, limit):
                cjk_hits = self._search_lexical_python(query_text, accessible_document_ids, limit)
                return self._merge_candidates(postgres_hits, cjk_hits, limit)
            return postgres_hits
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

    def search_indexed_sparse(self, query_text: str, accessible_document_ids: Sequence[UUID], limit: int) -> list[RetrievalCandidate]:
        if not accessible_document_ids or limit <= 0:
            return []
        if self.session.bind and self.session.bind.dialect.name == "postgresql":
            if self._contains_cjk(query_text):
                return self._search_indexed_sparse_postgres_cjk(query_text, accessible_document_ids, limit)
            return self._search_lexical_postgres_plain(query_text, accessible_document_ids, limit)
        return self._search_lexical_python(query_text, accessible_document_ids, limit)

    def search_indexed_sparse_timeout_fallback(
        self,
        query_text: str,
        accessible_document_ids: Sequence[UUID],
        limit: int,
    ) -> list[RetrievalCandidate]:
        if not accessible_document_ids or limit <= 0:
            return []
        if not bool(getattr(self.settings, "retrieval_indexed_sparse_timeout_fallback_enabled", True)):
            return []

        if self.session.bind and self.session.bind.dialect.name == "postgresql" and self._contains_cjk(query_text):
            fallback_max_terms = max(
                1,
                int(getattr(self.settings, "retrieval_indexed_sparse_timeout_fallback_max_query_terms", 4) or 4),
            )
            primary_max_terms = max(1, int(getattr(self.settings, "retrieval_indexed_sparse_max_query_terms", 10) or 10))
            narrowed_max_terms = min(fallback_max_terms, primary_max_terms)
            try:
                narrowed_hits = self._search_indexed_sparse_postgres_cjk(
                    query_text,
                    accessible_document_ids,
                    limit,
                    max_query_terms=narrowed_max_terms,
                )
            except Exception as exc:
                if not self._is_timeout_error(exc):
                    raise
                self.session.rollback()
                narrowed_hits = []
            if narrowed_hits:
                return narrowed_hits

        if not bool(getattr(self.settings, "retrieval_indexed_sparse_timeout_python_fallback_enabled", False)):
            return []
        return self._search_lexical_python(query_text, accessible_document_ids, limit)

    def search_python_sparse(self, query_text: str, accessible_document_ids: Sequence[UUID], limit: int) -> list[RetrievalCandidate]:
        return self._search_lexical_python(query_text, accessible_document_ids, limit)

    def search_table_lookup_pairs(
        self,
        lookup_pairs: set[tuple[str, str]],
        accessible_document_ids: Sequence[UUID],
        limit: int,
    ) -> list[RetrievalCandidate]:
        if not lookup_pairs or not accessible_document_ids or limit <= 0:
            return []

        normalized_pairs = {
            (self._compact_exact_text(field), self._compact_exact_text(value))
            for field, value in lookup_pairs
            if len(self._compact_exact_text(field)) >= 2 and len(self._compact_exact_text(value)) >= 2
        }
        if not normalized_pairs:
            return []

        scored: list[tuple[float, int, RetrievalCandidate]] = []
        for entry in self._get_lexical_index(accessible_document_ids):
            candidate = entry.candidate
            candidate_text = self._compact_exact_text(
                " ".join(
                    str(part or "")
                    for part in [
                        candidate.document_title,
                        candidate.section_title,
                        candidate.heading_path,
                        candidate.content,
                    ]
                )
            )
            if not candidate_text:
                continue

            best_score = 0.0
            for field, value in normalized_pairs:
                field_position = candidate_text.find(field)
                value_position = candidate_text.find(value)
                if field_position < 0 or value_position < 0:
                    continue
                distance = abs(value_position - field_position)
                score = 96.0 - min(distance / 8.0, 24.0)
                if candidate.chunk_type == "table" or "tablerow" in candidate_text:
                    score += 12.0
                best_score = max(best_score, score)
            if best_score <= 0:
                continue
            scored.append((best_score, -candidate.chunk_index, replace(candidate, lexical_score=best_score, vector_score=None)))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [candidate for _, _, candidate in scored[:limit]]

    def search_exact_text_in_documents(
        self,
        query_text: str,
        accessible_document_ids: Sequence[UUID],
        limit: int,
    ) -> list[RetrievalCandidate]:
        if not accessible_document_ids or limit <= 0:
            return []

        compact_query = self._compact_exact_text(query_text)
        if len(compact_query) < 10 or self._is_low_information_exact_query(compact_query):
            return []

        scored_by_document: dict[UUID, list[tuple[float, int, RetrievalCandidate]]] = {}
        for entry in self._get_lexical_index(accessible_document_ids):
            candidate = entry.candidate
            haystack = self._compact_exact_text(
                " ".join(
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
            )
            position = haystack.find(compact_query)
            if position < 0:
                continue
            trailing_chars = max(len(haystack) - position - len(compact_query), 0)
            continuation_bonus = min(trailing_chars / 40.0, 4.0)
            score = 72.0 + continuation_bonus
            scored_by_document.setdefault(candidate.document_id, []).append(
                (score, -candidate.chunk_index, replace(candidate, lexical_score=score, vector_score=None))
            )

        results: list[RetrievalCandidate] = []
        for document_rows in scored_by_document.values():
            document_rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
            results.extend(candidate for _, _, candidate in document_rows[:limit])

        results.sort(key=lambda item: ((item.lexical_score or 0.0), -item.chunk_index), reverse=True)
        return results[:limit]

    def search_structural(self, query_text: str, accessible_document_ids: Sequence[UUID], limit: int) -> list[RetrievalCandidate]:
        if not accessible_document_ids:
            return []
        query_key = normalize_structural_key(query_text)
        query_articles = article_numbers_in_text(query_text)
        query_terms = self._tokenize(query_text)
        if not query_key and not query_articles and not query_terms:
            return []
        has_explicit_structural_anchor = bool(query_articles) or any(
            marker in query_text for marker in ("条款全称", "第", "章节", "小节", "标题", "制度", "办法", "手册", "指南", "规范")
        )
        if not has_explicit_structural_anchor:
            return []
        query_counts = Counter(query_terms)
        query_law_hints = self._extract_query_law_hints(query_text)
        query_law_article_pairs = self._extract_query_law_article_pairs(query_text)
        if self.session.bind and self.session.bind.dialect.name == "postgresql":
            return self._search_structural_postgres(
                query_text=query_text,
                accessible_document_ids=accessible_document_ids,
                limit=limit,
                query_key=query_key,
                query_articles=query_articles,
                query_counts=query_counts,
                query_law_hints=query_law_hints,
                query_law_article_pairs=query_law_article_pairs,
            )

        candidates: list[RetrievalCandidate] = []
        seen: set[UUID] = set()
        for entry in self._get_structural_index(accessible_document_ids):
            score = self._score_structural_entry(
                entry,
                query_key=query_key,
                query_articles=query_articles,
                query_counts=query_counts,
                query_law_hints=query_law_hints,
                query_law_article_pairs=query_law_article_pairs,
            )
            if score <= 0 or entry.candidate.chunk_id in seen:
                continue
            seen.add(entry.candidate.chunk_id)
            candidates.append(replace(entry.candidate, lexical_score=float(score), vector_score=None))
        candidates.sort(key=lambda item: ((item.lexical_score or 0.0), -item.chunk_index), reverse=True)
        return candidates[:limit]

    def expand_within_documents(
        self,
        query_text: str,
        seeds: Sequence[RetrievalCandidate],
        *,
        per_document_limit: int,
        max_candidates: int,
        adjacent_window: int = 0,
    ) -> list[RetrievalCandidate]:
        if not seeds or per_document_limit <= 0 or max_candidates <= 0:
            return []

        query_terms = self._tokenize(query_text)
        if not query_terms:
            return []
        query_counts = Counter(query_terms)
        existing_chunk_ids = {seed.chunk_id for seed in seeds}
        document_ids = list(dict.fromkeys(seed.document_id for seed in seeds))
        index_entries = self._get_lexical_index(document_ids)
        if not index_entries:
            return []

        seed_indexes_by_document: dict[UUID, set[int]] = {}
        for seed in seeds:
            seed_indexes_by_document.setdefault(seed.document_id, set()).add(seed.chunk_index)

        scored_by_document: dict[UUID, list[tuple[float, int, RetrievalCandidate]]] = {}
        for entry in index_entries:
            candidate = entry.candidate
            if candidate.chunk_id in existing_chunk_ids:
                continue

            seed_indexes = seed_indexes_by_document.get(candidate.document_id, set())
            nearest_distance = min((abs(candidate.chunk_index - index) for index in seed_indexes), default=999)
            adjacency_bonus = 0.0
            if nearest_distance == 1:
                adjacency_bonus = 0.22
            elif nearest_distance == 2:
                adjacency_bonus = 0.12
            elif nearest_distance <= 4:
                adjacency_bonus = 0.05
            elif adjacent_window > 4 and nearest_distance <= adjacent_window:
                remaining_window = max(adjacent_window - 4, 1)
                distance_decay = 1.0 - ((nearest_distance - 4) / remaining_window)
                adjacency_bonus = max(0.01, 0.04 * distance_decay)

            content_terms = entry.terms
            lexical_score = self._score_weighted_terms(query_counts, content_terms) if content_terms else 0.0
            is_adjacent_context = adjacent_window > 0 and nearest_distance <= adjacent_window
            if lexical_score <= 0 and not is_adjacent_context:
                continue
            score = lexical_score + adjacency_bonus
            if score <= 0:
                continue
            scored_by_document.setdefault(candidate.document_id, []).append((score, -candidate.chunk_index, candidate))

        expanded: list[RetrievalCandidate] = []
        for document_rows in scored_by_document.values():
            document_rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
            for score, _, candidate in document_rows[:per_document_limit]:
                expanded.append(replace(candidate, lexical_score=float(score), vector_score=None))

        expanded.sort(key=lambda item: ((item.lexical_score or 0.0), -item.chunk_index), reverse=True)
        return expanded[:max_candidates]

    def collect_neighbor_context(
        self,
        seeds: Sequence[RetrievalCandidate],
        *,
        window: int,
        per_document_limit: int,
        max_candidates: int,
    ) -> list[RetrievalCandidate]:
        if not seeds or window <= 0 or per_document_limit <= 0 or max_candidates <= 0:
            return []

        document_ids = list(dict.fromkeys(seed.document_id for seed in seeds))
        index_entries = self._get_lexical_index(document_ids)
        if not index_entries:
            return []

        seed_chunk_ids = {seed.chunk_id for seed in seeds}
        seed_indexes_by_document: dict[UUID, set[int]] = {}
        for seed in seeds:
            seed_indexes_by_document.setdefault(seed.document_id, set()).add(seed.chunk_index)

        scored_by_document: dict[UUID, list[tuple[float, int, RetrievalCandidate]]] = {}
        for entry in index_entries:
            candidate = entry.candidate
            if candidate.chunk_id in seed_chunk_ids:
                continue
            seed_indexes = seed_indexes_by_document.get(candidate.document_id, set())
            nearest_distance = min((abs(candidate.chunk_index - index) for index in seed_indexes), default=window + 1)
            if nearest_distance <= 0 or nearest_distance > window:
                continue
            adjacency_score = 1.0 / (nearest_distance + 1)
            scored_by_document.setdefault(candidate.document_id, []).append((adjacency_score, -candidate.chunk_index, candidate))

        results: list[RetrievalCandidate] = []
        for document_rows in scored_by_document.values():
            document_rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
            for score, _, candidate in document_rows[:per_document_limit]:
                results.append(replace(candidate, lexical_score=float(score), vector_score=None))

        results.sort(key=lambda item: ((item.lexical_score or 0.0), -item.chunk_index), reverse=True)
        return results[:max_candidates]

    def search_exact_text_within_documents(
        self,
        query_text: str,
        seeds: Sequence[RetrievalCandidate],
        *,
        seed_document_limit: int,
        per_document_limit: int,
        max_candidates: int,
    ) -> list[RetrievalCandidate]:
        if not seeds or seed_document_limit <= 0 or per_document_limit <= 0 or max_candidates <= 0:
            return []

        compact_query = self._compact_exact_text(query_text)
        if len(compact_query) < 10 or self._is_low_information_exact_query(compact_query):
            return []

        document_ids: list[UUID] = []
        for seed in seeds:
            if seed.document_id in document_ids:
                continue
            document_ids.append(seed.document_id)
            if len(document_ids) >= seed_document_limit:
                break
        if not document_ids:
            return []

        scored_by_document: dict[UUID, list[tuple[float, int, RetrievalCandidate]]] = {}
        for entry in self._get_lexical_index(document_ids):
            candidate = entry.candidate
            haystack = self._compact_exact_text(
                " ".join(
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
            )
            position = haystack.find(compact_query)
            if position < 0:
                continue
            trailing_chars = max(len(haystack) - position - len(compact_query), 0)
            continuation_bonus = min(trailing_chars / 40.0, 4.0)
            score = 64.0 + continuation_bonus
            scored_by_document.setdefault(candidate.document_id, []).append((score, -candidate.chunk_index, candidate))

        results: list[RetrievalCandidate] = []
        for document_rows in scored_by_document.values():
            document_rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
            for score, _, candidate in document_rows[:per_document_limit]:
                results.append(replace(candidate, lexical_score=float(score), vector_score=None))

        results.sort(key=lambda item: ((item.lexical_score or 0.0), -item.chunk_index), reverse=True)
        return results[:max_candidates]

    @staticmethod
    def _compact_exact_text(value: str | None) -> str:
        return "".join(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", str(value or "").casefold()))

    @staticmethod
    def _is_low_information_exact_query(compact_query: str) -> bool:
        if re.fullmatch(r"(?:截至)?20\d{2}年(?:末|底|[0-9]{1,2}月)?", compact_query):
            return True
        return compact_query in {"截至2025年末", "截至2024年末", "截至2023年末"}

    def sweep_within_documents(
        self,
        query_text: str,
        seeds: Sequence[RetrievalCandidate],
        *,
        seed_document_limit: int,
        per_document_limit: int,
        max_candidates: int,
    ) -> list[RetrievalCandidate]:
        if not seeds or seed_document_limit <= 0 or per_document_limit <= 0 or max_candidates <= 0:
            return []

        use_bm25 = self._uses_python_bm25_scorer()
        query_terms = self._select_sql_query_terms(query_text, max_terms=64) if use_bm25 else self._tokenize(query_text)
        if not query_terms:
            return []
        query_counts = Counter(query_terms)

        document_rank: dict[UUID, int] = {}
        existing_chunk_ids = {seed.chunk_id for seed in seeds}
        for seed in seeds:
            if seed.document_id in document_rank:
                continue
            if len(document_rank) >= seed_document_limit:
                break
            document_rank[seed.document_id] = len(document_rank)
        document_ids = list(document_rank)
        index_entries = self._get_lexical_index(document_ids)
        if not index_entries:
            return []

        scope_key = tuple(sorted(document_ids, key=str))
        idf_by_term: dict[str, float] = {}
        avg_doc_len = 0.0
        if use_bm25:
            idf_by_term = self._corpus_idf_by_term(scope_key, index_entries, query_counts)
            avg_doc_len = self._lexical_avg_doc_len(scope_key, index_entries)

        query_articles = article_numbers_in_text(query_text)
        query_law_hints = self._extract_query_law_hints(query_text)
        query_law_article_pairs = self._extract_query_law_article_pairs(query_text)
        scored_by_document: dict[UUID, list[tuple[float, int, RetrievalCandidate]]] = {}
        document_count = max(len(document_rank), 1)
        for entry in index_entries:
            candidate = entry.candidate
            if candidate.chunk_id in existing_chunk_ids:
                continue

            lexical_score = (
                self._score_bm25_terms(
                    query_counts=query_counts,
                    content_terms=entry.terms,
                    idf_by_term=idf_by_term,
                    avg_doc_len=avg_doc_len,
                )
                if use_bm25
                else self._score_weighted_terms(query_counts, entry.terms)
            )
            lexical_score = self._adjust_lexical_score_for_structural_anchor(
                candidate,
                lexical_score,
                query_articles=query_articles,
                query_law_hints=query_law_hints,
                query_law_article_pairs=query_law_article_pairs,
            )
            lexical_score = self._adjust_lexical_score_for_answer_intent(candidate, lexical_score, query_text)
            if lexical_score <= 0:
                continue

            rank = document_rank.get(candidate.document_id, document_count)
            document_prior = 0.08 * (1.0 - (rank / document_count))
            score = lexical_score + document_prior
            scored_by_document.setdefault(candidate.document_id, []).append((score, -candidate.chunk_index, candidate))

        swept: list[RetrievalCandidate] = []
        for document_rows in scored_by_document.values():
            document_rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
            for score, _, candidate in document_rows[:per_document_limit]:
                swept.append(replace(candidate, lexical_score=float(score), vector_score=None))

        swept.sort(key=lambda item: ((item.lexical_score or 0.0), -item.chunk_index), reverse=True)
        return swept[:max_candidates]

    def search_document_first_evidence(
        self,
        query_text: str,
        seeds: Sequence[RetrievalCandidate],
        *,
        seed_document_limit: int,
        per_document_limit: int,
        max_candidates: int,
    ) -> list[RetrievalCandidate]:
        if not seeds or seed_document_limit <= 0 or per_document_limit <= 0 or max_candidates <= 0:
            return []

        use_bm25 = self._uses_python_bm25_scorer()
        query_terms = self._select_sql_query_terms(query_text, max_terms=64) if use_bm25 else self._tokenize(query_text)
        if not query_terms:
            return []
        query_counts = Counter(query_terms)

        document_rank: dict[UUID, int] = {}
        existing_chunk_ids = {seed.chunk_id for seed in seeds}
        seed_terms_by_document: dict[UUID, Counter[str]] = {}
        for seed in seeds:
            if len(document_rank) < seed_document_limit and seed.document_id not in document_rank:
                document_rank[seed.document_id] = len(document_rank)
            if seed.document_id in document_rank:
                seed_terms_by_document.setdefault(seed.document_id, Counter()).update(self._candidate_weighted_terms(seed))

        document_ids = list(document_rank)
        index_entries = self._get_lexical_index(document_ids)
        if not index_entries:
            return []

        scope_key = tuple(sorted(document_ids, key=str))
        idf_by_term: dict[str, float] = {}
        avg_doc_len = 0.0
        if use_bm25:
            idf_by_term = self._corpus_idf_by_term(scope_key, index_entries, query_counts)
            avg_doc_len = self._lexical_avg_doc_len(scope_key, index_entries)

        query_articles = article_numbers_in_text(query_text)
        query_law_hints = self._extract_query_law_hints(query_text)
        query_law_article_pairs = self._extract_query_law_article_pairs(query_text)
        document_count = max(len(document_rank), 1)
        scored_by_document: dict[UUID, list[tuple[float, int, RetrievalCandidate]]] = {}
        for entry in index_entries:
            candidate = entry.candidate
            if candidate.chunk_id in existing_chunk_ids:
                continue

            lexical_score = (
                self._score_bm25_terms(
                    query_counts=query_counts,
                    content_terms=entry.terms,
                    idf_by_term=idf_by_term,
                    avg_doc_len=avg_doc_len,
                )
                if use_bm25
                else self._score_weighted_terms(query_counts, entry.terms)
            )
            lexical_score = self._adjust_lexical_score_for_structural_anchor(
                candidate,
                lexical_score,
                query_articles=query_articles,
                query_law_hints=query_law_hints,
                query_law_article_pairs=query_law_article_pairs,
            )
            lexical_score = self._adjust_lexical_score_for_answer_intent(candidate, lexical_score, query_text)

            structural_score = self._score_structural_entry(
                _StructuralIndexEntry(
                    candidate=candidate,
                    normalized_fields=self._normalized_candidate_fields(candidate),
                    terms=self._candidate_structural_terms(candidate),
                ),
                query_key=normalize_structural_key(query_text),
                query_articles=query_articles,
                query_counts=query_counts,
                query_law_hints=query_law_hints,
                query_law_article_pairs=query_law_article_pairs,
            )
            focus_score = 0.0
            seed_terms = seed_terms_by_document.get(candidate.document_id)
            if seed_terms:
                focus_score = min(self._score_weighted_terms(seed_terms, entry.terms), 3.0) * 0.08
            if lexical_score <= 0 and structural_score <= 0 and focus_score <= 0:
                continue

            rank = document_rank.get(candidate.document_id, document_count)
            document_prior = 0.12 * (1.0 - (rank / document_count))
            score = lexical_score + (structural_score * 0.28) + focus_score + document_prior
            if score <= 0:
                continue
            scored_by_document.setdefault(candidate.document_id, []).append((score, -candidate.chunk_index, candidate))

        results: list[RetrievalCandidate] = []
        for document_rows in scored_by_document.values():
            document_rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
            for score, _, candidate in document_rows[:per_document_limit]:
                results.append(replace(candidate, lexical_score=float(score), vector_score=None))

        results.sort(key=lambda item: ((item.lexical_score or 0.0), -item.chunk_index), reverse=True)
        return results[:max_candidates]

    def _base_current_version_statement(self, accessible_document_ids: Sequence[UUID], *, include_embedding: bool = False):
        columns = [
            Chunk.id.label("chunk_id"),
            Chunk.document_id.label("document_id"),
            Chunk.document_version_id.label("document_version_id"),
            Chunk.chunk_index.label("chunk_index"),
            Chunk.content.label("content"),
            Chunk.token_count.label("token_count"),
            Chunk.section_title.label("section_title"),
            Chunk.page_number_start.label("page_number_start"),
            Chunk.page_number_end.label("page_number_end"),
            Chunk.paragraph_start.label("paragraph_start"),
            Chunk.paragraph_end.label("paragraph_end"),
            Chunk.char_start.label("char_start"),
            Chunk.char_end.label("char_end"),
            Chunk.clause_full_name.label("clause_full_name"),
            Chunk.article_number.label("article_number"),
            Chunk.chunk_type.label("chunk_type"),
            Chunk.heading_path.label("heading_path"),
            Chunk.structural_search_text.label("structural_search_text"),
            Chunk.lexical_search_text.label("lexical_search_text"),
            Chunk.citation_metadata.label("citation_metadata"),
            Document.title.label("document_title"),
            DocumentVersion.version_number.label("version_number"),
        ]
        if include_embedding:
            columns.append(Chunk.embedding.label("embedding"))
        return (
            select(*columns)
            .join(Document, Document.id == Chunk.document_id)
            .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
            .where(Chunk.document_id.in_(accessible_document_ids))
            .where(Document.current_version_id == Chunk.document_version_id)
        )

    def _search_lexical_postgres(self, query_text: str, accessible_document_ids: Sequence[UUID], limit: int) -> list[RetrievalCandidate]:
        if self._contains_cjk(query_text):
            plain_hits = self._search_lexical_postgres_plain(query_text, accessible_document_ids, limit)
            if not self.settings.retrieval_cjk_sql_sparse_enabled:
                return plain_hits
            cjk_hits = (
                self._search_lexical_postgres_cjk(query_text, accessible_document_ids, limit)
                if limit > 8
                else []
            )
            return self._rank_fuse_candidates(
                [(plain_hits, 1.0), (cjk_hits, 0.72)],
                limit=limit,
            )
        else:
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

    def _search_lexical_postgres_plain(
        self,
        query_text: str,
        accessible_document_ids: Sequence[UUID],
        limit: int,
    ) -> list[RetrievalCandidate]:
        ts_vector = func.to_tsvector("simple", func.coalesce(Chunk.lexical_search_text, ""))
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

    def _search_lexical_postgres_cjk(
        self,
        query_text: str,
        accessible_document_ids: Sequence[UUID],
        limit: int,
    ) -> list[RetrievalCandidate]:
        query_terms = self._select_sql_query_terms(query_text, max_terms=48)
        if not query_terms:
            return []

        ts_query = func.to_tsquery("simple", self._build_or_tsquery(query_terms))
        ts_vector = func.to_tsvector("simple", func.coalesce(Chunk.lexical_search_text, ""))
        rank_expr = func.ts_rank_cd(ts_vector, ts_query).label("lexical_score")
        sql_limit = self._sql_candidate_limit(limit)
        statement = (
            self._base_current_version_statement(accessible_document_ids)
            .add_columns(rank_expr)
            .where(ts_vector.op("@@")(ts_query))
            .order_by(desc(rank_expr), Chunk.chunk_index.asc())
            .limit(sql_limit)
        )
        rows = self.session.execute(statement).all()
        if not rows:
            return []

        query_counts = Counter(query_terms)
        row_terms = [(row, self._weighted_lexical_terms(row)) for row in rows]
        idf_by_term = self._candidate_local_idf_by_term(row_terms, query_terms)
        avg_doc_len = sum(sum(terms.values()) for _, terms in row_terms) / max(len(row_terms), 1)
        query_articles = article_numbers_in_text(query_text)
        query_law_hints = self._extract_query_law_hints(query_text)
        query_law_article_pairs = self._extract_query_law_article_pairs(query_text)

        candidates: list[RetrievalCandidate] = []
        for row, terms in row_terms:
            score = self._score_bm25_terms(
                query_counts=query_counts,
                content_terms=terms,
                idf_by_term=idf_by_term,
                avg_doc_len=avg_doc_len,
            )
            score += float(getattr(row, "lexical_score", 0.0) or 0.0) * 0.04
            score = self._adjust_lexical_score_for_structural_anchor(
                self._row_to_candidate(row),
                score,
                query_articles=query_articles,
                query_law_hints=query_law_hints,
                query_law_article_pairs=query_law_article_pairs,
            )
            if score <= 0:
                continue
            candidates.append(self._row_to_candidate(row, lexical_score=float(score)))
        candidates.sort(key=lambda item: ((item.lexical_score or 0.0), -item.chunk_index), reverse=True)
        return candidates[:limit]

    def _search_indexed_sparse_postgres_cjk(
        self,
        query_text: str,
        accessible_document_ids: Sequence[UUID],
        limit: int,
        *,
        max_query_terms: int | None = None,
    ) -> list[RetrievalCandidate]:
        configured_max_terms = max_query_terms
        if configured_max_terms is None:
            configured_max_terms = int(getattr(self.settings, "retrieval_indexed_sparse_max_query_terms", 18) or 18)
        max_terms = max(1, int(configured_max_terms))
        query_terms = self._select_sql_query_terms(query_text, max_terms=max_terms)
        if not query_terms:
            return []

        ts_query = func.to_tsquery("simple", self._build_or_tsquery(query_terms))
        ts_vector = func.to_tsvector("simple", func.coalesce(Chunk.lexical_search_text, ""))
        rank_expr = func.ts_rank_cd(ts_vector, ts_query).label("lexical_score")
        row_multiplier = max(1, int(getattr(self.settings, "retrieval_indexed_sparse_sql_row_multiplier", 1) or 1))
        configured_max = max(int(getattr(self.settings, "retrieval_candidate_max", limit) or limit), int(limit or 1))
        sql_limit = min(max(limit * row_multiplier, limit), configured_max * row_multiplier)
        statement = (
            self._base_current_version_statement(accessible_document_ids)
            .add_columns(rank_expr)
            .where(ts_vector.op("@@")(ts_query))
            .order_by(desc(rank_expr), Chunk.chunk_index.asc())
            .limit(sql_limit)
        )
        rows = self.session.execute(statement).all()
        if not rows:
            return []

        query_counts = Counter(query_terms)
        row_terms = [(row, self._weighted_lexical_terms(row)) for row in rows]
        idf_by_term = self._candidate_local_idf_by_term(row_terms, query_terms)
        avg_doc_len = sum(sum(terms.values()) for _, terms in row_terms) / max(len(row_terms), 1)
        query_articles = article_numbers_in_text(query_text)
        query_law_hints = self._extract_query_law_hints(query_text)
        query_law_article_pairs = self._extract_query_law_article_pairs(query_text)

        candidates: list[RetrievalCandidate] = []
        for row, terms in row_terms:
            score = self._score_bm25_terms(
                query_counts=query_counts,
                content_terms=terms,
                idf_by_term=idf_by_term,
                avg_doc_len=avg_doc_len,
            )
            score += float(getattr(row, "lexical_score", 0.0) or 0.0) * 0.04
            score = self._adjust_lexical_score_for_structural_anchor(
                self._row_to_candidate(row),
                score,
                query_articles=query_articles,
                query_law_hints=query_law_hints,
                query_law_article_pairs=query_law_article_pairs,
            )
            score = self._adjust_lexical_score_for_answer_intent(self._row_to_candidate(row), score, query_text)
            if score <= 0:
                continue
            candidates.append(self._row_to_candidate(row, lexical_score=float(score)))
        candidates.sort(key=lambda item: ((item.lexical_score or 0.0), -item.chunk_index), reverse=True)
        return candidates[:limit]

    def _search_structural_postgres(
        self,
        *,
        query_text: str,
        accessible_document_ids: Sequence[UUID],
        limit: int,
        query_key: str,
        query_articles: list[str],
        query_counts: Counter[str],
        query_law_hints: set[str],
        query_law_article_pairs: set[tuple[str, str]],
    ) -> list[RetrievalCandidate]:
        query_terms = self._select_sql_query_terms(query_text, max_terms=36)
        predicates = []
        rank_expr = None
        if query_terms:
            ts_query = func.to_tsquery("simple", self._build_or_tsquery(query_terms))
            ts_vector = func.to_tsvector("simple", func.coalesce(Chunk.structural_search_text, ""))
            rank_expr = func.ts_rank_cd(ts_vector, ts_query).label("structural_rank")
            predicates.append(ts_vector.op("@@")(ts_query))
        if query_articles:
            predicates.append(Chunk.article_number.in_(query_articles))
        if query_key and len(query_key) >= 4:
            like_value = f"%{query_key}%"
            predicates.extend(
                [
                    func.regexp_replace(func.lower(func.coalesce(Document.title, "")), r"[\s[:punct:]]+", "", "g").like(like_value),
                    func.regexp_replace(func.lower(func.coalesce(Chunk.clause_full_name, "")), r"[\s[:punct:]]+", "", "g").like(like_value),
                    func.regexp_replace(func.lower(func.coalesce(Chunk.section_title, "")), r"[\s[:punct:]]+", "", "g").like(like_value),
                    func.regexp_replace(func.lower(func.coalesce(Chunk.heading_path, "")), r"[\s[:punct:]]+", "", "g").like(like_value),
                ]
            )
        if not predicates:
            return []

        sql_limit = self._sql_candidate_limit(limit)
        statement = self._base_current_version_statement(accessible_document_ids).where(or_(*predicates))
        if rank_expr is not None:
            statement = statement.add_columns(rank_expr).order_by(desc(rank_expr), Chunk.chunk_index.asc())
        else:
            statement = statement.order_by(Chunk.chunk_index.asc())
        rows = self.session.execute(statement.limit(sql_limit)).all()

        candidates: list[RetrievalCandidate] = []
        seen: set[UUID] = set()
        for row in rows:
            entry = _StructuralIndexEntry(
                candidate=self._row_to_candidate(row),
                terms=self._structural_terms(row),
                normalized_fields=self._normalized_structural_fields(row),
            )
            score = self._score_structural_entry(
                entry,
                query_key=query_key,
                query_articles=query_articles,
                query_counts=query_counts,
                query_law_hints=query_law_hints,
                query_law_article_pairs=query_law_article_pairs,
            )
            if score <= 0 or entry.candidate.chunk_id in seen:
                continue
            seen.add(entry.candidate.chunk_id)
            candidates.append(replace(entry.candidate, lexical_score=float(score), vector_score=None))
        candidates.sort(key=lambda item: ((item.lexical_score or 0.0), -item.chunk_index), reverse=True)
        return candidates[:limit]

    def _should_use_cjk_python_fallback(
        self,
        query_text: str,
        postgres_hits: Sequence[RetrievalCandidate],
        limit: int,
    ) -> bool:
        if not self._contains_cjk(query_text):
            return False
        mode = str(getattr(self.settings, "retrieval_cjk_python_fallback_mode", "auto") or "auto").strip().lower()
        if mode in {"off", "false", "disabled", "never"}:
            return False
        if mode in {"on", "true", "enabled", "always"}:
            return True
        return len(postgres_hits) < limit

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
        use_bm25 = self._uses_python_bm25_scorer()
        query_terms = self._select_sql_query_terms(query_text, max_terms=64) if use_bm25 else self._tokenize(query_text)
        if not query_terms:
            return []
        query_counts = Counter(query_terms)
        query_articles = article_numbers_in_text(query_text)
        query_law_hints = self._extract_query_law_hints(query_text)
        query_law_article_pairs = self._extract_query_law_article_pairs(query_text)
        index_entries = self._get_lexical_index(accessible_document_ids)
        scope_key = tuple(sorted(accessible_document_ids, key=str))
        idf_by_term: dict[str, float] = {}
        avg_doc_len = 0.0
        if use_bm25:
            idf_by_term = self._corpus_idf_by_term(scope_key, index_entries, query_counts)
            avg_doc_len = self._lexical_avg_doc_len(scope_key, index_entries)
        candidates = []
        for entry in index_entries:
            content_terms = entry.terms
            if not content_terms:
                continue
            lexical_score = (
                self._score_bm25_terms(
                    query_counts=query_counts,
                    content_terms=content_terms,
                    idf_by_term=idf_by_term,
                    avg_doc_len=avg_doc_len,
                )
                if use_bm25
                else self._score_weighted_terms(query_counts, content_terms)
            )
            lexical_score = self._adjust_lexical_score_for_structural_anchor(
                entry.candidate,
                lexical_score,
                query_articles=query_articles,
                query_law_hints=query_law_hints,
                query_law_article_pairs=query_law_article_pairs,
            )
            lexical_score = self._adjust_lexical_score_for_answer_intent(entry.candidate, lexical_score, query_text)
            if lexical_score <= 0:
                continue
            candidates.append(replace(entry.candidate, lexical_score=float(lexical_score), vector_score=None))
        candidates.sort(key=lambda item: (item.lexical_score or 0.0, -item.chunk_index), reverse=True)
        return candidates[:limit]

    def _uses_python_bm25_scorer(self) -> bool:
        scorer = str(getattr(self.settings, "retrieval_cjk_python_scorer", "weighted") or "weighted").strip().lower()
        return scorer in {"bm25", "okapi", "sparse"}

    def _get_lexical_index(self, accessible_document_ids: Sequence[UUID]) -> list[_LexicalIndexEntry]:
        if not accessible_document_ids:
            return []
        cache_key = tuple(sorted(accessible_document_ids, key=str))
        if self.settings.retrieval_cjk_lexical_cache_enabled and cache_key in self._lexical_index_cache:
            return self._lexical_index_cache[cache_key]

        entries: list[_LexicalIndexEntry] = []
        for row in self.session.execute(self._base_current_version_statement(accessible_document_ids)).all():
            terms = self._weighted_lexical_terms(row)
            if terms:
                entries.append(_LexicalIndexEntry(candidate=self._row_to_candidate(row), terms=terms))
        if self.settings.retrieval_cjk_lexical_cache_enabled:
            self._lexical_index_cache[cache_key] = entries
        return entries

    def _get_structural_index(self, accessible_document_ids: Sequence[UUID]) -> list[_StructuralIndexEntry]:
        if not accessible_document_ids:
            return []
        cache_key = tuple(sorted(accessible_document_ids, key=str))
        if self.settings.retrieval_cjk_lexical_cache_enabled and cache_key in self._structural_index_cache:
            return self._structural_index_cache[cache_key]

        entries: list[_StructuralIndexEntry] = []
        for row in self.session.execute(self._base_current_version_statement(accessible_document_ids)).all():
            terms = self._structural_terms(row)
            normalized_fields = self._normalized_structural_fields(row)
            if terms or any(normalized_fields.values()):
                entries.append(_StructuralIndexEntry(candidate=self._row_to_candidate(row), terms=terms, normalized_fields=normalized_fields))
        if self.settings.retrieval_cjk_lexical_cache_enabled:
            self._structural_index_cache[cache_key] = entries
        return entries

    def _weighted_lexical_terms(self, row) -> Counter[str]:
        cache_key = (row.document_version_id, row.chunk_id)
        if self.settings.retrieval_cjk_lexical_cache_enabled and cache_key in self._lexical_cache:
            return self._lexical_cache[cache_key]

        terms = parse_lexical_search_text(getattr(row, "lexical_search_text", None))
        if not terms:
            terms = build_weighted_lexical_terms(
                document_title=row.document_title,
                section_title=row.section_title,
                clause_full_name=getattr(row, "clause_full_name", None),
                article_number=getattr(row, "article_number", None),
                heading_path=getattr(row, "heading_path", None),
                structural_search_text=getattr(row, "structural_search_text", None),
                content=row.content,
            )
        if self.settings.retrieval_cjk_lexical_cache_enabled:
            self._lexical_cache[cache_key] = terms
        return terms

    @staticmethod
    def _candidate_weighted_terms(candidate: RetrievalCandidate) -> Counter[str]:
        terms = parse_lexical_search_text(candidate.lexical_search_text)
        if terms:
            return terms
        return build_weighted_lexical_terms(
            document_title=candidate.document_title,
            section_title=candidate.section_title,
            clause_full_name=candidate.clause_full_name,
            article_number=candidate.article_number,
            heading_path=candidate.heading_path,
            structural_search_text=candidate.structural_search_text,
            content=candidate.content,
        )

    @staticmethod
    def _candidate_structural_terms(candidate: RetrievalCandidate) -> Counter[str]:
        terms: Counter[str] = Counter()
        terms.update(RetrievalRepository._tokenize(candidate.document_title) * 5)
        terms.update(RetrievalRepository._tokenize(candidate.section_title) * 6)
        terms.update(RetrievalRepository._tokenize(candidate.clause_full_name) * 8)
        terms.update(RetrievalRepository._tokenize(candidate.article_number) * 8)
        terms.update(RetrievalRepository._tokenize(candidate.heading_path) * 5)
        return terms

    @staticmethod
    def _normalized_candidate_fields(candidate: RetrievalCandidate) -> dict[str, str]:
        return {
            "document_title": normalize_structural_key(candidate.document_title),
            "section_title": normalize_structural_key(candidate.section_title),
            "clause_full_name": normalize_structural_key(candidate.clause_full_name),
            "article_number": normalize_structural_key(candidate.article_number),
            "heading_path": normalize_structural_key(candidate.heading_path),
        }

    @staticmethod
    def _score_weighted_terms(query_counts: Counter[str], content_terms: Counter[str]) -> float:
        match_count = sum(min(count, content_terms.get(term, 0)) for term, count in query_counts.items())
        if match_count <= 0:
            return 0.0
        coverage = sum(1 for term in query_counts if content_terms.get(term, 0) > 0) / max(len(query_counts), 1)
        return (match_count / max(sum(query_counts.values()), 1)) + (coverage * 0.35)

    @staticmethod
    def _score_bm25_terms(
        *,
        query_counts: Counter[str],
        content_terms: Counter[str],
        idf_by_term: dict[str, float],
        avg_doc_len: float,
    ) -> float:
        if not query_counts or not content_terms:
            return 0.0
        k1 = 1.2
        b = 0.75
        doc_len = max(float(sum(content_terms.values())), 1.0)
        avg_len = max(float(avg_doc_len or 0.0), 1.0)
        score = 0.0
        matched_terms = 0
        for term, query_tf in query_counts.items():
            term_tf = float(content_terms.get(term, 0))
            if term_tf <= 0:
                continue
            matched_terms += 1
            idf = idf_by_term.get(term, 0.0)
            denominator = term_tf + k1 * (1.0 - b + b * (doc_len / avg_len))
            score += idf * ((term_tf * (k1 + 1.0)) / denominator) * min(float(query_tf), 2.0)
        if matched_terms <= 0:
            return 0.0
        coverage = matched_terms / max(len(query_counts), 1)
        return score + (coverage * 0.18)

    def _search_vector_python(self, query_embedding: list[float], accessible_document_ids: Sequence[UUID], limit: int) -> list[RetrievalCandidate]:
        candidates = []
        for row in self.session.execute(self._base_current_version_statement(accessible_document_ids, include_embedding=True)).all():
            embedding = row.embedding
            if not embedding:
                continue
            vector_score = self._cosine_similarity(query_embedding, embedding)
            candidates.append(self._row_to_candidate(row, vector_score=float(vector_score)))
        candidates.sort(key=lambda item: (item.vector_score or 0.0, -item.chunk_index), reverse=True)
        return candidates[:limit]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return tokenize_search_text(text)

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    @staticmethod
    def _merge_candidates(
        primary: list[RetrievalCandidate],
        fallback: list[RetrievalCandidate],
        limit: int,
    ) -> list[RetrievalCandidate]:
        merged: dict[UUID, RetrievalCandidate] = {}
        for candidate in [*primary, *fallback]:
            existing = merged.get(candidate.chunk_id)
            if existing is None or (candidate.lexical_score or 0.0) > (existing.lexical_score or 0.0):
                merged[candidate.chunk_id] = candidate
        candidates = list(merged.values())
        candidates.sort(key=lambda item: ((item.lexical_score or 0.0), -item.chunk_index), reverse=True)
        return candidates[:limit]

    @staticmethod
    def _rank_fuse_candidates(
        ranked_sources: Sequence[tuple[Sequence[RetrievalCandidate], float]],
        *,
        limit: int,
        rrf_k: int = 60,
    ) -> list[RetrievalCandidate]:
        fused_scores: dict[UUID, float] = {}
        candidates_by_id: dict[UUID, RetrievalCandidate] = {}
        for source_candidates, weight in ranked_sources:
            for rank, candidate in enumerate(source_candidates, start=1):
                candidates_by_id.setdefault(candidate.chunk_id, candidate)
                fused_scores[candidate.chunk_id] = fused_scores.get(candidate.chunk_id, 0.0) + (weight / (rrf_k + rank))
        fused = [
            replace(candidate, lexical_score=float(fused_scores.get(chunk_id, 0.0)), vector_score=None)
            for chunk_id, candidate in candidates_by_id.items()
        ]
        fused.sort(key=lambda item: ((item.lexical_score or 0.0), -item.chunk_index), reverse=True)
        return fused[:limit]

    @staticmethod
    def _candidate_local_idf_by_term(row_terms: Sequence[tuple[object, Counter[str]]], query_terms: Sequence[str]) -> dict[str, float]:
        total_chunks = len(row_terms)
        if total_chunks <= 0:
            return {term: 0.0 for term in query_terms}
        idf_by_term: dict[str, float] = {}
        for term in query_terms:
            df = sum(1 for _, terms in row_terms if terms.get(term, 0) > 0)
            idf_by_term[term] = math.log(1.0 + ((total_chunks - df + 0.5) / (df + 0.5)))
        return idf_by_term

    def _corpus_idf_by_term(
        self,
        scope_key: tuple[UUID, ...],
        entries: Sequence[_LexicalIndexEntry],
        query_counts: Counter[str],
    ) -> dict[str, float]:
        total_chunks = len(entries)
        if total_chunks <= 0:
            return {term: 0.0 for term in query_counts}
        idf_by_term: dict[str, float] = {}
        for term in query_counts:
            cache_key = (scope_key, term)
            if self.settings.retrieval_cjk_lexical_cache_enabled and cache_key in self._lexical_df_cache:
                df = self._lexical_df_cache[cache_key]
            else:
                df = sum(1 for entry in entries if entry.terms.get(term, 0) > 0)
                if self.settings.retrieval_cjk_lexical_cache_enabled:
                    self._lexical_df_cache[cache_key] = df
            idf_by_term[term] = math.log(1.0 + ((total_chunks - df + 0.5) / (df + 0.5)))
        return idf_by_term

    def _lexical_avg_doc_len(self, scope_key: tuple[UUID, ...], entries: Sequence[_LexicalIndexEntry]) -> float:
        if self.settings.retrieval_cjk_lexical_cache_enabled and scope_key in self._lexical_avg_doc_len_cache:
            return self._lexical_avg_doc_len_cache[scope_key]
        avg_doc_len = sum(sum(entry.terms.values()) for entry in entries) / max(len(entries), 1)
        if self.settings.retrieval_cjk_lexical_cache_enabled:
            self._lexical_avg_doc_len_cache[scope_key] = avg_doc_len
        return avg_doc_len

    def _lexical_scope_size(self, scope_key: tuple[UUID, ...], accessible_document_ids: Sequence[UUID]) -> int:
        if scope_key not in self._lexical_scope_size_cache:
            statement = (
                select(func.count())
                .select_from(Chunk)
                .join(Document, Document.id == Chunk.document_id)
                .where(Chunk.document_id.in_(accessible_document_ids))
                .where(Document.current_version_id == Chunk.document_version_id)
            )
            self._lexical_scope_size_cache[scope_key] = int(self.session.execute(statement).scalar_one() or 0)
        return self._lexical_scope_size_cache[scope_key]

    @staticmethod
    def _select_sql_query_terms(query_text: str, *, max_terms: int) -> list[str]:
        terms = []
        seen: set[str] = set()
        for term in tokenize_search_text(query_text):
            if not term or term in seen:
                continue
            if re.fullmatch(r"\d{1,2}", term):
                continue
            if re.fullmatch(r"[\u4e00-\u9fff]", term):
                continue
            if RetrievalRepository._is_low_information_cjk_sql_term(term):
                continue
            seen.add(term)
            terms.append(term)
        terms.sort(key=RetrievalRepository._sql_query_term_sort_key, reverse=True)
        return terms[:max_terms]

    @staticmethod
    def _is_low_information_cjk_sql_term(term: str) -> bool:
        if not re.search(r"[\u4e00-\u9fff]", term):
            return False
        return any(fragment in term for fragment in CJK_SQL_LOW_INFORMATION_SUBSTRINGS)

    @staticmethod
    def _sql_query_term_sort_key(term: str) -> tuple[int, int, int, str]:
        has_digit = 1 if re.search(r"\d", term) else 0
        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", term))
        return (has_digit, len(term), cjk_count, term)

    @staticmethod
    def _build_or_tsquery(query_terms: Sequence[str]) -> str:
        cleaned = [term for term in query_terms if re.fullmatch(r"[a-z0-9\u4e00-\u9fff]+", term)]
        return " | ".join(cleaned)

    @staticmethod
    def _is_timeout_error(error: Exception) -> bool:
        error_text = str(error).casefold()
        return (
            "statement timeout" in error_text
            or "query_canceled" in error_text
            or "canceling statement due to statement timeout" in error_text
        )

    def _sql_candidate_limit(self, limit: int) -> int:
        requested = max(int(limit or 1), 1)
        configured_max = max(requested, int(getattr(self.settings, "retrieval_candidate_max", requested) or requested))
        return min(max(requested * 8, requested), max(configured_max * 8, requested))

    @staticmethod
    def _structural_terms(row) -> Counter[str]:
        terms: Counter[str] = Counter()
        terms.update(RetrievalRepository._tokenize(row.document_title) * 5)
        if row.section_title:
            terms.update(RetrievalRepository._tokenize(row.section_title) * 6)
        if getattr(row, "clause_full_name", None):
            terms.update(RetrievalRepository._tokenize(row.clause_full_name) * 8)
        if getattr(row, "article_number", None):
            terms.update(RetrievalRepository._tokenize(row.article_number) * 8)
        if getattr(row, "heading_path", None):
            terms.update(RetrievalRepository._tokenize(row.heading_path) * 5)
        return terms

    @staticmethod
    def _normalized_structural_fields(row) -> dict[str, str]:
        return {
            "document_title": normalize_structural_key(row.document_title),
            "section_title": normalize_structural_key(row.section_title),
            "clause_full_name": normalize_structural_key(getattr(row, "clause_full_name", None)),
            "article_number": normalize_structural_key(getattr(row, "article_number", None)),
            "heading_path": normalize_structural_key(getattr(row, "heading_path", None)),
        }

    @staticmethod
    def _score_structural_entry(
        entry: _StructuralIndexEntry,
        *,
        query_key: str,
        query_articles: list[str],
        query_counts: Counter[str],
        query_law_hints: set[str],
        query_law_article_pairs: set[tuple[str, str]],
    ) -> float:
        normalized = entry.normalized_fields
        score = 0.0
        normalized_articles = [normalize_structural_key(item) for item in query_articles]
        normalized_document = normalized.get("document_title", "")
        law_hint_match = not query_law_hints or any(hint and hint in normalized_document for hint in query_law_hints)
        paired_article_match = any(
            RetrievalRepository._article_anchor_matches_candidate(
                article_key,
                normalized_document=normalized_document,
                query_law_hints=query_law_hints,
                query_law_article_pairs=query_law_article_pairs,
            )
            for article_key in normalized_articles
        )
        term_score = RetrievalRepository._score_weighted_terms(query_counts, entry.terms)
        if normalized_articles:
            if query_law_article_pairs:
                score += term_score * (1.2 if paired_article_match else 0.12)
            else:
                score += term_score * (1.2 if law_hint_match else 0.18)
        else:
            score += term_score * 0.55

        if query_key:
            for name, field_key in normalized.items():
                if not field_key:
                    continue
                if query_key == field_key:
                    score += 4.0 if name in {"clause_full_name", "article_number"} else 2.0
                elif query_key in field_key:
                    score += {
                        "document_title": 0.35,
                        "section_title": 0.7,
                        "clause_full_name": 1.8,
                        "article_number": 2.4,
                        "heading_path": 0.8,
                    }.get(name, 0.1)
                elif field_key in query_key and len(field_key) >= 4:
                    score += {
                        "document_title": 0.45,
                        "section_title": 0.9,
                        "clause_full_name": 1.9,
                        "article_number": 2.4,
                        "heading_path": 0.7,
                    }.get(name, 0.1)

        for article_key in normalized_articles:
            if not article_key:
                continue
            if not RetrievalRepository._article_anchor_matches_candidate(
                article_key,
                normalized_document=normalized_document,
                query_law_hints=query_law_hints,
                query_law_article_pairs=query_law_article_pairs,
            ):
                continue
            if normalized.get("article_number") == article_key:
                score += 4.2
            if article_key and article_key in normalized.get("clause_full_name", ""):
                score += 2.5
            if article_key and article_key in normalized.get("section_title", ""):
                score += 1.8

        return score

    @staticmethod
    def _adjust_lexical_score_for_structural_anchor(
        candidate: RetrievalCandidate,
        score: float,
        *,
        query_articles: list[str],
        query_law_hints: set[str],
        query_law_article_pairs: set[tuple[str, str]],
    ) -> float:
        if score <= 0 or not query_articles:
            return score
        candidate_article = normalize_structural_key(candidate.article_number)
        normalized_articles = {normalize_structural_key(item) for item in query_articles}
        if not candidate_article or candidate_article not in normalized_articles:
            return score
        if not query_law_hints:
            return score

        candidate_scope = normalize_structural_key(
            " ".join(
                item
                for item in [
                    candidate.document_title,
                    candidate.clause_full_name or "",
                    candidate.heading_path or "",
                ]
                if item
            )
        )
        if query_law_article_pairs:
            if any((hint, candidate_article) in query_law_article_pairs and hint in candidate_scope for hint in query_law_hints):
                return score + 0.18
            return score * 0.16

        if any(hint and hint in candidate_scope for hint in query_law_hints):
            return score + 0.18
        return score * 0.16

    @staticmethod
    def _adjust_lexical_score_for_answer_intent(candidate: RetrievalCandidate, score: float, query_text: str) -> float:
        if score <= 0:
            return score
        normalized_query = re.sub(r"\s+", "", str(query_text or "").casefold())
        normalized_value = re.sub(
            r"\s+",
            "",
            " ".join(
                item
                for item in [
                    candidate.section_title or "",
                    candidate.clause_full_name or "",
                    candidate.content[:1800],
                ]
                if item
            ).casefold(),
        )
        if "审批" in normalized_query and any(token in normalized_query for token in ("谁", "审批人", "由谁", "负责人")):
            if (
                "审批人=" in normalized_value
                or "共同审批" in normalized_value
                or re.search(r"由[\u4e00-\u9fffa-z0-9=；;、,，和及与]{1,48}审批", normalized_value)
            ):
                score += 0.65
            elif "审批记录" in normalized_value and "归档" in normalized_value:
                score *= 0.72
        return score

    @staticmethod
    def _article_anchor_matches_candidate(
        article_key: str,
        *,
        normalized_document: str,
        query_law_hints: set[str],
        query_law_article_pairs: set[tuple[str, str]],
    ) -> bool:
        if not query_law_hints:
            return True
        if query_law_article_pairs:
            return any((hint, article_key) in query_law_article_pairs and hint in normalized_document for hint in query_law_hints)
        return any(hint and hint in normalized_document for hint in query_law_hints)

    @staticmethod
    def _extract_query_law_hints(query_text: str) -> set[str]:
        normalized = normalize_structural_key(query_text)
        hints: set[str] = set()
        for marker, law_name in RetrievalRepository._law_hint_aliases().items():
            if normalize_structural_key(marker) in normalized:
                hints.add(normalize_structural_key(law_name))
        return hints

    @staticmethod
    def _extract_query_law_article_pairs(query_text: str) -> set[tuple[str, str]]:
        normalized = normalize_structural_key(query_text)
        if not normalized:
            return set()

        occurrences: list[tuple[int, int, str, int]] = []
        for marker, law_name in RetrievalRepository._law_hint_aliases().items():
            marker_key = normalize_structural_key(marker)
            if not marker_key:
                continue
            for match in re.finditer(re.escape(marker_key), normalized):
                occurrences.append((match.start(), match.end(), normalize_structural_key(law_name), len(marker_key)))
        if not occurrences:
            return set()

        occurrences.sort(key=lambda item: (item[0], -item[3], item[1]))
        deduped: list[tuple[int, int, str, int]] = []
        for occurrence in occurrences:
            start, end, _, _ = occurrence
            if deduped and start < deduped[-1][1]:
                continue
            deduped.append(occurrence)

        pairs: set[tuple[str, str]] = set()
        for index, (_, end, law_name, _) in enumerate(deduped):
            next_start = deduped[index + 1][0] if index + 1 < len(deduped) else min(len(normalized), end + 80)
            window = normalized[end:next_start]
            for article in article_numbers_in_text(window):
                pairs.add((law_name, normalize_structural_key(article)))
        return pairs

    @staticmethod
    def _law_hint_aliases() -> dict[str, str]:
        return {
            "民法典": "中华人民共和国民法典",
            "中华人民共和国民法典": "中华人民共和国民法典",
            "农村土地承包法": "农村土地承包法",
            "土地承包法": "农村土地承包法",
            "农村土地承包": "农村土地承包法",
            "产品质量法": "产品质量法",
            "食品安全法": "食品安全法",
            "药品管理法": "药品管理法",
            "商标法": "商标法",
            "仲裁法": "仲裁法",
            "土地管理法": "土地管理法",
            "民事诉讼法": "民事诉讼法",
            "个体工商户条例": "个体工商户条例",
            "促进个体工商户发展条例": "促进个体工商户发展条例",
            "土地承包纠纷司法解释": "最高人民法院关于审理涉及农村土地承包纠纷案件适用法律问题的解释",
            "农村土地承包纠纷案件适用法律问题": "最高人民法院关于审理涉及农村土地承包纠纷案件适用法律问题的解释",
            "最高人民法院关于审理涉及农村土地承包纠纷案件": "最高人民法院关于审理涉及农村土地承包纠纷案件适用法律问题的解释",
            "民诉法解释": "最高人民法院关于适用中华人民共和国民事诉讼法的解释",
            "最高人民法院关于适用民事诉讼法": "最高人民法院关于适用中华人民共和国民事诉讼法的解释",
        }

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
        right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
        return numerator / (left_norm * right_norm)

    @staticmethod
    def _row_to_candidate(row, lexical_score: float | None = None, vector_score: float | None = None) -> RetrievalCandidate:
        return RetrievalCandidate(
            chunk_id=row.chunk_id,
            document_id=row.document_id,
            document_title=row.document_title,
            document_version_id=row.document_version_id,
            version_number=row.version_number,
            chunk_index=row.chunk_index,
            content=row.content,
            token_count=row.token_count,
            section_title=row.section_title,
            page_number_start=row.page_number_start,
            page_number_end=row.page_number_end,
            paragraph_start=row.paragraph_start,
            paragraph_end=row.paragraph_end,
            char_start=row.char_start,
            char_end=row.char_end,
            clause_full_name=getattr(row, "clause_full_name", None),
            article_number=getattr(row, "article_number", None),
            chunk_type=getattr(row, "chunk_type", None),
            heading_path=getattr(row, "heading_path", None),
            structural_search_text=getattr(row, "structural_search_text", None),
            lexical_search_text=getattr(row, "lexical_search_text", None),
            citation_metadata=row.citation_metadata,
            lexical_score=lexical_score,
            vector_score=vector_score,
        )
