from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings, get_settings
from app.services.llm.openai_compatible import (
    create_openai_compatible_client,
    has_openai_compatible_credentials,
    request_chat_completion,
)

CHINESE_FILLER_PHRASES = (
    "请问",
    "想问一下",
    "想问下",
    "麻烦问一下",
    "麻烦问下",
    "帮我看下",
    "帮我看一下",
    "能不能告诉我",
    "可以告诉我",
    "可以帮我",
    "里面提到",
    "里提到",
    "里面说",
    "里说",
    "是怎么说的",
    "怎么说",
)

CHINESE_STOP_TERMS = {
    "一下",
    "一下子",
    "吗",
    "么",
    "呢",
    "呀",
    "啊",
    "吧",
    "请",
    "问",
    "一下呢",
    "里面",
    "里",
    "提到",
    "提及",
    "关于",
}

ENGLISH_FILLER_PHRASES = (
    "please tell me",
    "can you tell me",
    "help me find",
    "what does",
    "what is",
)


@dataclass
class QueryPlanCandidate:
    key: str
    label: str
    retrieval_query: str
    lexical_queries: list[str]
    applied_strategies: list[str]
    rewrite_provider: str | None = None
    rewrite_model: str | None = None
    rewrite_latency_ms: int | None = None

    @property
    def rewrite_applied(self) -> bool:
        return bool(self.applied_strategies)


@dataclass
class QueryOptimizationPlan:
    original_query: str
    candidates: list[QueryPlanCandidate]
    selected_candidate_key: str | None = None
    selected_candidate_reason: str | None = None

    @property
    def selected_candidate(self) -> QueryPlanCandidate:
        if self.selected_candidate_key:
            for candidate in self.candidates:
                if candidate.key == self.selected_candidate_key:
                    return candidate
        return self.candidates[-1]

    @property
    def rewrite_applied(self) -> bool:
        return self.selected_candidate.rewrite_applied

    @property
    def retrieval_query(self) -> str:
        return self.selected_candidate.retrieval_query

    @property
    def lexical_queries(self) -> list[str]:
        return self.selected_candidate.lexical_queries

    @property
    def applied_strategies(self) -> list[str]:
        return self.selected_candidate.applied_strategies

    @property
    def rewrite_provider(self) -> str | None:
        return self.selected_candidate.rewrite_provider

    @property
    def rewrite_model(self) -> str | None:
        return self.selected_candidate.rewrite_model

    @property
    def rewrite_latency_ms(self) -> int | None:
        return self.selected_candidate.rewrite_latency_ms

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    def select_candidate(self, candidate_key: str, *, reason: str | None = None) -> None:
        if any(candidate.key == candidate_key for candidate in self.candidates):
            self.selected_candidate_key = candidate_key
            self.selected_candidate_reason = reason


@dataclass
class QueryRewriteSuggestion:
    retrieval_query: str
    lexical_queries: list[str]
    provider: str
    model: str | None
    latency_ms: int | None


class QueryOptimizer:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def build(self, query: str, *, target_document_title: str | None = None) -> QueryOptimizationPlan:
        original_query = query.strip()
        normalized_query = _normalize_query(original_query)
        focused_query = _focus_query(normalized_query)
        anchored_title = target_document_title or _extract_quoted_document_title(original_query)
        candidates: list[QueryPlanCandidate] = []

        baseline_queries = [original_query]
        baseline_strategies: list[str] = []
        baseline_retrieval_query = normalized_query or original_query
        if normalized_query and normalized_query != original_query:
            baseline_queries.append(normalized_query)
            baseline_strategies.append("normalize")
        self._append_candidate(
            candidates,
            key="baseline",
            label="原始表达",
            retrieval_query=baseline_retrieval_query,
            lexical_queries=baseline_queries,
            applied_strategies=baseline_strategies,
        )

        focused_retrieval_query = focused_query or baseline_retrieval_query
        if _normalize_for_compare(focused_retrieval_query) != _normalize_for_compare(baseline_retrieval_query):
            self._append_candidate(
                candidates,
                key="focused",
                label="关键词聚焦",
                retrieval_query=focused_retrieval_query,
                lexical_queries=[focused_retrieval_query, *baseline_queries],
                applied_strategies=_unique_strings([*baseline_strategies, "focus_keywords"]),
            )

        anchored_query: str | None = None
        if anchored_title:
            anchored_query = _anchor_query_to_document(anchored_title, focused_retrieval_query)
            self._append_candidate(
                candidates,
                key="title_anchor",
                label="标题锚定",
                retrieval_query=anchored_query,
                lexical_queries=[anchored_query, focused_retrieval_query, *baseline_queries],
                applied_strategies=_unique_strings(
                    [*baseline_strategies, *(["focus_keywords"] if focused_query else []), "title_anchor"]
                ),
            )

        llm_suggestion = self._llm_rewrite(
            original_query=original_query,
            retrieval_query=anchored_query or focused_retrieval_query,
            lexical_queries=_unique_nonempty_queries(
                [*(anchored_query and [anchored_query] or []), focused_retrieval_query, *baseline_queries]
            ),
            target_document_title=anchored_title,
        )
        if llm_suggestion is not None:
            llm_retrieval_query = llm_suggestion.retrieval_query or anchored_query or focused_retrieval_query
            if anchored_title:
                llm_retrieval_query = _anchor_query_to_document(anchored_title, llm_retrieval_query)
            llm_queries = _unique_nonempty_queries(
                [
                    llm_retrieval_query,
                    *llm_suggestion.lexical_queries,
                    *(anchored_query and [anchored_query] or []),
                    focused_retrieval_query,
                    *baseline_queries,
                ]
            )
            self._append_candidate(
                candidates,
                key="llm_rewrite",
                label="LLM 改写",
                retrieval_query=llm_retrieval_query,
                lexical_queries=llm_queries,
                applied_strategies=_unique_strings(
                    [*baseline_strategies, *(["focus_keywords"] if focused_query else []), *(["title_anchor"] if anchored_title else []), "llm_rewrite"]
                ),
                rewrite_provider=llm_suggestion.provider,
                rewrite_model=llm_suggestion.model,
                rewrite_latency_ms=llm_suggestion.latency_ms,
            )

        return QueryOptimizationPlan(
            original_query=original_query,
            candidates=candidates,
            selected_candidate_key=candidates[-1].key,
        )

    def _append_candidate(
        self,
        candidates: list[QueryPlanCandidate],
        *,
        key: str,
        label: str,
        retrieval_query: str,
        lexical_queries: list[str],
        applied_strategies: list[str],
        rewrite_provider: str | None = None,
        rewrite_model: str | None = None,
        rewrite_latency_ms: int | None = None,
    ) -> None:
        cleaned_retrieval_query = _normalize_query(retrieval_query)
        cleaned_lexical_queries = _unique_nonempty_queries([cleaned_retrieval_query, *lexical_queries])
        if not cleaned_retrieval_query or not cleaned_lexical_queries:
            return
        signature = (
            _normalize_for_compare(cleaned_retrieval_query),
            tuple(_normalize_for_compare(item) for item in cleaned_lexical_queries),
        )
        for existing in candidates:
            existing_signature = (
                _normalize_for_compare(existing.retrieval_query),
                tuple(_normalize_for_compare(item) for item in existing.lexical_queries),
            )
            if existing_signature == signature:
                existing.applied_strategies = _unique_strings([*existing.applied_strategies, *applied_strategies])
                if "title_anchor" in applied_strategies and existing.key == "focused":
                    existing.label = label
                if rewrite_provider:
                    existing.rewrite_provider = rewrite_provider
                if rewrite_model:
                    existing.rewrite_model = rewrite_model
                if rewrite_latency_ms is not None:
                    existing.rewrite_latency_ms = rewrite_latency_ms
                return
        candidates.append(
            QueryPlanCandidate(
                key=key,
                label=label,
                retrieval_query=cleaned_retrieval_query,
                lexical_queries=cleaned_lexical_queries,
                applied_strategies=_unique_strings(applied_strategies),
                rewrite_provider=rewrite_provider,
                rewrite_model=rewrite_model,
                rewrite_latency_ms=rewrite_latency_ms,
            )
        )

    def _llm_rewrite(
        self,
        *,
        original_query: str,
        retrieval_query: str,
        lexical_queries: list[str],
        target_document_title: str | None,
    ) -> QueryRewriteSuggestion | None:
        provider = (self.settings.query_rewrite_provider or "auto").strip().lower()
        if provider == "deterministic":
            return None
        if not self._should_use_llm(original_query, target_document_title):
            return None
        if provider not in {"auto", "openai_compatible", "openai"}:
            return None
        if not has_openai_compatible_credentials(self.settings):
            return None

        client = create_openai_compatible_client(self.settings)
        started = time.perf_counter()
        try:
            response = request_chat_completion(
                client,
                max_attempts=1,
                model=self.settings.effective_query_rewrite_model,
                messages=_build_query_rewrite_messages(
                    original_query=original_query,
                    retrieval_query=retrieval_query,
                    lexical_queries=lexical_queries,
                    target_document_title=target_document_title,
                    max_variants=self.settings.query_rewrite_max_variants,
                ),
                temperature=0.0,
                response_format={"type": "json_object"},
                timeout=8.0,
            )
        except Exception:
            return None

        latency_ms = int((time.perf_counter() - started) * 1000)
        content = response.choices[0].message.content or "{}"
        payload = _parse_json_payload(content)
        candidate_retrieval_query = _normalize_query(str(payload.get("retrieval_query") or retrieval_query))
        candidate_lexical_queries = [
            _normalize_query(str(item))
            for item in payload.get("lexical_queries", [])
            if isinstance(item, str) and item.strip()
        ]
        candidate_lexical_queries = _unique_nonempty_queries(candidate_lexical_queries)[: self.settings.query_rewrite_max_variants]
        if not candidate_retrieval_query and not candidate_lexical_queries:
            return None
        return QueryRewriteSuggestion(
            retrieval_query=candidate_retrieval_query or retrieval_query,
            lexical_queries=candidate_lexical_queries,
            provider="openai_compatible",
            model=self.settings.effective_query_rewrite_model,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _should_use_llm(query: str, target_document_title: str | None) -> bool:
        normalized = _normalize_query(query)
        has_filler = any(phrase in normalized for phrase in CHINESE_FILLER_PHRASES)
        if target_document_title:
            if len(normalized) >= 14:
                return True
            return has_filler and len(normalized) >= 10
        if len(normalized) >= 24:
            return True
        if len(normalized) >= 18 and has_filler:
            return True
        if re.search(r"[，；;。！？?]", normalized) and len(normalized) >= 20:
            return True
        return False


def _normalize_query(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    normalized = re.sub(r"[“”\"'`]+", " ", normalized)
    normalized = re.sub(r"[《》〈〉【】\[\]（）()]+", " ", normalized)
    normalized = re.sub(r"[\t\r\n]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _focus_query(value: str) -> str:
    focused = value.casefold()
    for phrase in ENGLISH_FILLER_PHRASES:
        focused = focused.replace(phrase, " ")
    focused = unicodedata.normalize("NFKC", focused)
    for phrase in CHINESE_FILLER_PHRASES:
        focused = focused.replace(phrase, " ")
    focused = re.sub(r"[，。！？、,:;；]+", " ", focused)
    focused = re.sub(r"\s+", " ", focused).strip()
    for term in sorted(CHINESE_STOP_TERMS, key=len, reverse=True):
        focused = re.sub(re.escape(term), " ", focused)
    focused = re.sub(r"\s+", " ", focused).strip()
    return focused


def _extract_quoted_document_title(value: str) -> str | None:
    match = re.search(r"《([^》]{1,80})》", value)
    if match:
        return match.group(1).strip()
    return None


def _anchor_query_to_document(document_title: str, query: str) -> str:
    document_title = _normalize_query(document_title)
    query = _normalize_query(query)
    if not document_title:
        return query
    if document_title in query:
        return query
    return f"{document_title} {query}".strip()


def _unique_nonempty_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in queries:
        cleaned = item.strip()
        if not cleaned:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
    return unique


def _normalize_for_compare(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _unique_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _build_query_rewrite_messages(
    *,
    original_query: str,
    retrieval_query: str,
    lexical_queries: list[str],
    target_document_title: str | None,
    max_variants: int,
) -> list[dict[str, str]]:
    target_line = f"目标文档：{target_document_title}" if target_document_title else "目标文档：无"
    current_variants = " | ".join(item for item in lexical_queries if item.strip())
    return [
        {
            "role": "system",
            "content": (
                "你是企业文档检索系统中的 Query Rewrite 模块。"
                "请把用户问题改写成更适合检索的表达，保持原意，不要虚构事实。"
                "输出 JSON，字段只有 retrieval_query 和 lexical_queries。"
                f"lexical_queries 最多 {max_variants} 条。"
                "如果用户问题已经明确指向某份制度/手册，请保留文档名。"
                "优先生成适合企业制度、流程、字段、审批、时限类知识库检索的表达。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"原始问题：{original_query}\n"
                f"当前规则改写：{retrieval_query}\n"
                f"当前检索变体：{current_variants}\n"
                f"{target_line}\n"
                "请输出更适合检索的 retrieval_query，并给出若干 lexical_queries 变体。"
            ),
        },
    ]


def _parse_json_payload(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        loaded = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}
