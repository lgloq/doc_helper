from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence
from uuid import UUID

from app.repositories.retrieval_repository import RetrievalCandidate

ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "which",
}

CHINESE_STOPWORDS = {
    "什么",
    "怎么",
    "如何",
    "哪些",
    "一下",
    "关于",
    "里面",
    "要求",
    "内容",
    "需要",
    "规定",
    "说明",
    "相关",
    "里的",
    "文档",
    "手册",
    "指南",
    "登记",
}


@dataclass
class RerankCandidate:
    candidate: RetrievalCandidate
    lexical_raw: float = 0.0
    vector_raw: float = 0.0
    lexical_norm: float = 0.0
    vector_norm: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0
    sources: set[str] = field(default_factory=set)


@dataclass
class RerankResult:
    candidates: list[RerankCandidate]
    strategy: str
    pre_rerank_count: int
    post_rerank_count: int


class HeuristicReranker:
    strategy_name = "heuristic-overlap"

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        top_k: int,
        *,
        target_document_id: UUID | None = None,
    ) -> RerankResult:
        if not candidates:
            return RerankResult(
                candidates=[],
                strategy=self.strategy_name,
                pre_rerank_count=0,
                post_rerank_count=0,
            )

        query_features = _feature_tokens(query)
        reranked: list[RerankCandidate] = []
        for item in candidates:
            candidate = item.candidate
            content_features = _feature_tokens(
                " ".join(
                    part
                    for part in [candidate.document_title, candidate.section_title or "", candidate.content[:420]]
                    if part
                )
            )
            title_features = _feature_tokens(candidate.document_title)
            section_features = _feature_tokens(candidate.section_title or "")

            overlap = _feature_overlap(query_features, content_features)
            title_overlap = _feature_overlap(query_features, title_features)
            section_overlap = _feature_overlap(query_features, section_features)
            lexical_support = item.lexical_raw > 0

            score = item.fused_score
            score += overlap * 0.22
            score += title_overlap * 0.12
            score += section_overlap * 0.06
            if lexical_support:
                score += 0.04
            if target_document_id is not None and candidate.document_id == target_document_id:
                score += 0.08
            if overlap < 0.08 and not lexical_support:
                score -= 0.09
            if overlap < 0.12 and title_overlap == 0 and section_overlap == 0 and item.fused_score < 0.35:
                score -= 0.06

            item.rerank_score = score
            reranked.append(item)

        reranked.sort(
            key=lambda item: (
                item.rerank_score,
                item.fused_score,
                item.lexical_raw,
                -item.candidate.chunk_index,
            ),
            reverse=True,
        )

        top_candidates = reranked[:top_k]
        return RerankResult(
            candidates=top_candidates,
            strategy=self.strategy_name,
            pre_rerank_count=len(candidates),
            post_rerank_count=len(top_candidates),
        )


def _feature_tokens(value: str) -> set[str]:
    normalized = value.casefold()
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return set()

    features: set[str] = set()
    for ascii_token in re.findall(r"[a-z0-9]+", normalized):
        if ascii_token and ascii_token not in ENGLISH_STOPWORDS:
            features.add(ascii_token)

    for chinese_run in re.findall(r"[\u4e00-\u9fff]+", normalized):
        if chinese_run in CHINESE_STOPWORDS:
            continue
        if len(chinese_run) <= 4:
            features.add(chinese_run)
        for size in (2, 3):
            if len(chinese_run) < size:
                continue
            for index in range(len(chinese_run) - size + 1):
                token = chinese_run[index : index + size]
                if token in CHINESE_STOPWORDS:
                    continue
                features.add(token)

    return features


def _feature_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = left & right
    if not overlap:
        return 0.0
    return len(overlap) / max(min(len(left), len(right)), 1)
