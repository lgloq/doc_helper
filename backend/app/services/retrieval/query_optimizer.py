from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import dataclass, field
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

EVIDENCE_BRIDGE_QUESTION_HINTS = (
    "是否",
    "能否",
    "可否",
    "可以",
    "哪些",
    "有什么",
    "如何",
    "怎么",
    "谁",
    "哪个",
    "找谁",
    "多少",
    "多久",
    "多长",
)


@dataclass
class QuerySubquery:
    query_text: str
    org_hint: str | None
    evidence_hint: str
    case_shape: str


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
    subqueries: list[QuerySubquery] = field(default_factory=list)

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

    @property
    def query_decomposition_applied(self) -> bool:
        return len(self.subqueries) >= 2

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

        domain_queries = _build_domain_lexical_queries(
            original_query,
            focused_retrieval_query,
            domain_profile=self.settings.effective_retrieval_domain_profile,
        )
        if domain_queries:
            self._append_candidate(
                candidates,
                key="domain_expansion",
                label="领域词扩展",
                retrieval_query=domain_queries[0],
                lexical_queries=[*domain_queries, focused_retrieval_query, *baseline_queries],
                applied_strategies=_unique_strings(
                    [*baseline_strategies, *(["focus_keywords"] if focused_query else []), "domain_expansion"]
                ),
            )

        if self.settings.retrieval_evidence_query_bridge_enabled:
            evidence_bridge_queries = _build_evidence_bridge_lexical_queries(
                original_query,
                focused_retrieval_query,
                max_queries=self.settings.retrieval_evidence_query_bridge_max_queries,
            )
            if evidence_bridge_queries:
                self._append_candidate(
                    candidates,
                    key="evidence_bridge",
                    label="证据意图扩展",
                    retrieval_query=evidence_bridge_queries[0],
                    lexical_queries=[*evidence_bridge_queries, focused_retrieval_query, *baseline_queries],
                    applied_strategies=_unique_strings(
                        [*baseline_strategies, *(["focus_keywords"] if focused_query else []), "evidence_bridge"]
                    ),
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
            subqueries=_build_deterministic_subqueries(original_query),
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


def _build_deterministic_subqueries(query: str) -> list[QuerySubquery]:
    return (
        _build_cross_document_subqueries(query)
        or _build_same_document_subqueries(query)
        or _build_table_lookup_subqueries(query)
        or _build_single_evidence_anchor_subqueries(query)
    )


def _build_cross_document_subqueries(query: str) -> list[QuerySubquery]:
    if "比较" not in query or "两份" not in query or "分别关注" not in query:
        return []
    quoted_segments = _extract_quoted_segments(query)
    if len(quoted_segments) < 2:
        return []

    context_match = re.search(
        r"比较(?P<left>.+?)和(?P<right>.+?)两份(?P<document_hint>[^，,。；;]{0,48})",
        query,
    )
    if not context_match:
        return []

    org_hints = [
        _clean_org_hint(context_match.group("left")),
        _clean_org_hint(context_match.group("right")),
    ]
    document_hint = _clean_document_hint(context_match.group("document_hint"))
    if not all(org_hints):
        return []

    return _subqueries_from_hints(
        org_hints=org_hints,
        evidence_hints=quoted_segments[:2],
        document_hint=document_hint,
        case_shape="cross_document_comparison",
    )


def _build_same_document_subqueries(query: str) -> list[QuerySubquery]:
    if "同时核对" not in query or "两个事项" not in query:
        return []
    quoted_segments = _extract_quoted_segments(query)
    if len(quoted_segments) < 2:
        return []

    context_match = re.search(
        r"同时核对(?P<org>.+?)这份(?P<document_hint>[^，,。；;]{0,48}?)中的两个事项",
        query,
    )
    if not context_match:
        return []

    org_hint = _clean_org_hint(context_match.group("org"))
    document_hint = _clean_document_hint(context_match.group("document_hint"))
    if not org_hint:
        return []

    return _subqueries_from_hints(
        org_hints=[org_hint, org_hint],
        evidence_hints=quoted_segments[:2],
        document_hint=document_hint,
        case_shape="same_document_two_matters",
    )


def _build_table_lookup_subqueries(query: str) -> list[QuerySubquery]:
    if "表格或清单信息" not in query:
        return []
    quoted_segments = _extract_quoted_segments(query)
    if not quoted_segments:
        return []

    context_match = re.search(r"核对(?P<org>.+?)文件中的表格或清单信息", query)
    if not context_match:
        return []
    org_hint = _clean_org_hint(context_match.group("org"))
    if not org_hint:
        return []

    return _subqueries_from_hints(
        org_hints=[org_hint],
        evidence_hints=quoted_segments[:1],
        document_hint="表格 清单",
        case_shape="table_structured_lookup",
    )


def _build_single_evidence_anchor_subqueries(query: str) -> list[QuerySubquery]:
    if any(marker in query for marker in ("普通查看用户", "受限材料", "能否直接查看")):
        return []
    if "分别关注" in query or "两个事项" in query:
        return []

    quoted_segments = _extract_quoted_segments(query)
    if len(quoted_segments) != 1:
        return []

    context_match = re.search(
        r"(?P<org>[\u4e00-\u9fffA-Za-z0-9（）() /·\-]+?)(?:这份|的)(?P<document_hint>[^，,。；;“”]{0,48}?)(?:中|里)",
        query,
    )
    if not context_match:
        return []

    org_hint = _clean_org_hint(context_match.group("org"))
    document_hint = _clean_document_hint(context_match.group("document_hint"))
    if not org_hint:
        return []

    case_shape = "single_category_anchor" if ("相关事项" in query or "处理口径" in query) else "single_evidence_anchor"
    return _subqueries_from_hints(
        org_hints=[org_hint],
        evidence_hints=quoted_segments,
        document_hint=document_hint,
        case_shape=case_shape,
    )


def _subqueries_from_hints(
    *,
    org_hints: list[str],
    evidence_hints: list[str],
    document_hint: str,
    case_shape: str,
) -> list[QuerySubquery]:
    subqueries: list[QuerySubquery] = []
    for index, evidence_hint in enumerate(evidence_hints):
        cleaned_evidence = _clean_evidence_hint(evidence_hint)
        if not cleaned_evidence:
            continue
        org_hint = org_hints[min(index, len(org_hints) - 1)] if org_hints else ""
        query_text = _normalize_query(" ".join(item for item in [org_hint, document_hint, cleaned_evidence] if item))
        if not query_text:
            continue
        subqueries.append(
            QuerySubquery(
                query_text=query_text,
                org_hint=org_hint or None,
                evidence_hint=cleaned_evidence,
                case_shape=case_shape,
            )
        )
    return subqueries


def _extract_quoted_segments(query: str) -> list[str]:
    segments: list[str] = []
    for pattern in (r"“(?P<value>[^”]{2,240})”", r'"(?P<value>[^"]{2,240})"'):
        for match in re.finditer(pattern, query):
            cleaned = _clean_evidence_hint(match.group("value"))
            if cleaned:
                segments.append(cleaned)
    return _unique_strings(segments)


def _clean_org_hint(value: str) -> str:
    cleaned = unicodedata.normalize("NFKC", value).strip()
    cleaned = re.sub(r"[“”\"'`]+", " ", cleaned)
    cleaned = re.sub(r"[《》〈〉【】\[\]]+", " ", cleaned)
    cleaned = re.sub(r"[\t\r\n]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"^(?:请|请你|麻烦|帮我|需要在|在)?(?:比较|核对)?", "", cleaned)
    cleaned = re.sub(r"的$", "", cleaned)
    return cleaned.strip(" ，,。；;：:")


def _clean_document_hint(value: str) -> str:
    cleaned = _normalize_query(value)
    return cleaned.strip(" ，,。；;：:")


def _clean_evidence_hint(value: str) -> str:
    cleaned = _normalize_query(value)
    cleaned = cleaned.strip(" ，,。；;：:、")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


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


def _build_domain_lexical_queries(original_query: str, focused_query: str, *, domain_profile: str = "enterprise") -> list[str]:
    normalized = re.sub(r"\s+", "", f"{original_query} {focused_query}".casefold())
    expansions: list[str] = _build_definition_lexical_queries(original_query, focused_query)
    enterprise_expansion_rules: list[tuple[tuple[str, ...], str]] = [
        (("客户手机号",), "客户手机号 数据导出 审批人 处理时限 脱敏要求 敏感字段 加密交付"),
        (("客户数据导出",), "客户数据导出 数据范围 审批人 处理时限 脱敏要求 证据要求 交付渠道"),
        (("临时高权限访问",), "临时高权限访问 生产环境 审批人 有效期 回收责任人 日志要求 证据要求"),
        (("生产环境", "高权限"), "生产环境 临时高权限访问 允许方式 有效期 回收责任人 日志要求 操作审计"),
        (("l4", "供应商"), "L4 高风险供应商 准入等级 审批链路 复核周期 退出要求 账号回收"),
        (("高风险供应商",), "L4 高风险供应商 准入等级 审批链路 复核周期 退出要求 账号回收"),
        (("数据处理服务",), "数据处理服务 交付类型 验收材料 验收人 保留期限 归档位置"),
        (("验收材料",), "验收材料 字段说明 脱敏方式 抽样检查结果 验收人 保留期限"),
        (("制度版本", "检查"), "制度版本 版本更新 检查清单 检查项 是否必须 知识库维护动作"),
        (("扫描", "动作"), "扫描附件 OCR 表格 编号 动作 负责人 时限 核验日志"),
    ]
    for required_terms, expansion in sorted(enterprise_expansion_rules, key=lambda item: len(item[0]), reverse=True):
        if all(term.casefold() in normalized for term in required_terms):
            expansions.append(expansion)
    if domain_profile != "legal_benchmark":
        return _unique_nonempty_queries(expansions)

    expansion_rules: list[tuple[tuple[str, ...], str]] = [
        (("谁可以成为", "个体工商户"), "中华人民共和国民法典 第五十四条 个体工商户条例 第二条 自然人 公民 依法登记 从事工商业经营 有经营能力 字号"),
        (("什么人", "个体工商户"), "中华人民共和国民法典 第五十四条 个体工商户条例 第二条 自然人 公民 依法登记 从事工商业经营 有经营能力 字号"),
        (("平台经济", "劳动关系"), "促进个体工商户发展条例 第三十条 平台经济 从业者 个体工商户 劳动关系 经营者 权益保障"),
        (("特许经营",), "商业特许经营管理条例 第三条 特许人 被特许人 经营资源 经营模式 特许经营合同"),
        (("个体工商户", "债务"), "个体工商户 债务 个人经营 家庭经营 财产承担 无法区分"),
        (("个人财产", "个体工商户"), "中华人民共和国民法典 第五十六条 个体工商户 债务 个人经营 家庭经营 财产承担"),
        (("营业执照", "实际经营者"), "中华人民共和国民法典 第五十六条 个体工商户 登记 经营者 实际经营 债务 财产承担 共同责任"),
        (("诉讼主体",), "中华人民共和国民法典 第五十四条 中华人民共和国民法典 第五十六条 自然人 从事工商业经营 依法登记 个体工商户 字号 债务承担"),
        (("注销", "个体工商户"), "个体工商户 债务 个人经营 家庭经营 财产承担 注销"),
        (("名为", "合伙"), "合伙合同 共享利益 共担风险 民事法律行为无效 恶意串通 公序良俗"),
        (("内部发生纠纷", "合伙"), "合伙合同 共享利益 共担风险 民事法律行为无效"),
        (("内部发生纠纷", "合伙"), "无民事行为能力 虚假意思表示 强制性规定 公序良俗 恶意串通"),
        (("正当使用", "商标"), "注册商标 通用名称 图形 型号 正当使用 地名 三维标志"),
        (("先用权",), "商标注册人 先于商标注册人使用 原使用范围 区别标识"),
        (("抗辩", "商标"), "先于商标注册人使用 原使用范围 继续使用 区别标识"),
        (("家庭承包经营户", "死亡"), "家庭承包 农户 家庭成员 平等享有 承包收益 继承 林地 继续承包"),
        (("土地承包份额", "继承"), "家庭承包 承包收益 继承 林地 承包期内 继续承包"),
        (("农户成员", "去世"), "农村土地承包法 第十六条 农村土地承包法 第二十一条 农村土地承包法 第二十七条 家庭承包 农户 家庭成员 平等享有 承包期 承包合同 承包地 承包收益 继承"),
        (("村集体", "收回"), "农村土地承包法 第二十一条 农村土地承包法 第二十七条 家庭承包 农户 承包期内 不得收回 承包地"),
        (("承包经营权", "转让"), "农村土地承包 经营权 流转 民法典 第三百三十三条 民法典 第三百三十四条 民法典 第三百三十五条 农村土地承包法 第三条 农村土地承包法 第十六条 农村土地承包法 第三十四条 农村土地承包法 第三十五条 家庭承包 招标 拍卖 公开协商"),
        (("土地承包经营权", "区别"), "土地承包经营权 土地经营权 民法典 第三百三十三条 民法典 第三百三十九条 民法典 第三百四十二条 农村土地承包法 第二十三条"),
        (("土地经营权", "区别"), "土地承包经营权 土地经营权 民法典 第三百三十三条 民法典 第三百三十九条 民法典 第三百四十二条 农村土地承包法 第二十三条"),
        (("享有土地承包经营权",), "土地承包经营权 民法典 第三百三十三条 承包合同 登记 承包方"),
        (("外嫁",), "妇女 土地承包经营权 农村土地承包法 第三十一条 夫妻共同财产 承包地权益"),
        (("夫妻共同财产", "土地承包经营权"), "土地承包经营权 农村土地承包法 第三十一条 民法典 第一千零八十七条 承包地权益"),
        (("共同被告", "土地承包经营权"), "集体经济组织 土地承包经营权 农村土地承包法 第三十一条 农村土地承包法 第六条 民法典 第一千零八十七条"),
        (("互换", "承包地"), "土地承包经营权 互换 民法典 第三百三十三条 民法典 第三百三十四条"),
        (("征地补偿",), "土地承包经营权 转让 登记 民法典 第三百三十三条 民法典 第三百三十四条 民法典 第三百三十五条 农村土地承包法 第三十五条"),
        (("本村以外", "合同"), "农村土地承包法 第三条 农村土地承包法 第十六条 农村土地承包法 第三十四条 农村土地承包法 第五十二条 家庭承包 承包方 权益 土地承包经营权 转让 本集体经济组织"),
        (("本集体经济组织", "其他农户"), "农村土地承包法 第三十四条 承包方 转让 土地承包经营权 本集体经济组织 其他农户 发包方同意"),
        (("转让", "其他农户"), "农村土地承包法 第三十四条 承包方 转让 土地承包经营权 本集体经济组织 其他农户 发包方同意"),
        (("本村村民",), "其他方式承包 农村土地承包法 第四十八条 农村土地承包法 第五十二条 本集体经济组织成员 村民会议"),
        (("民主议定程序", "家庭承包"), "农村土地承包法 第十九条 农村土地承包法 第二十八条 家庭承包 民主协商 村民会议 三分之二 承包合同"),
        (("没有经过民主议定程序", "家庭承包"), "农村土地承包法 第十九条 农村土地承包法 第二十八条 家庭承包 民主协商 村民会议 三分之二 承包合同"),
        (("民主议定程序", "其他承包"), "农村土地承包法 第四十八条 农村土地承包法 第五十二条 其他方式承包 村民会议 三分之二 村民代表 书面合同"),
        (("民主议定程序",), "集体经营性建设用地 土地管理法 第十三条 土地管理法 第六十三条 农村土地承包法 第五十二条 村民会议 三分之二 村民代表 书面合同"),
        (("违反民主议定程序",), "集体经营性建设用地 土地管理法 第十三条 土地管理法 第六十三条 农村土地承包法 第五十二条 村民会议 三分之二 村民代表 书面合同"),
        (("弃耕",), "最高人民法院关于审理涉及农村土地承包纠纷案件适用法律问题的解释 第六条 农村土地承包法 第二十七条 弃耕 抛荒 重新发包 收回承包地"),
        (("非家庭承包", "继承"), "招标 拍卖 公开协商 土地经营权 承包收益 继承 继续承包"),
        (("公开协商", "继承"), "招标 拍卖 公开协商 承包收益 继承 继续承包"),
        (("未按时缴纳费用",), "土地经营权 流转合同 单方解除 严重违约 解除合同"),
        (("解除合同", "发包方"), "约定解除 解除合同 土地经营权 流转合同 终止土地经营权流转合同"),
        (("解除合同", "发包方"), "当事人协商一致 可以解除合同 约定解除 解除权人"),
        (("改为鱼塘",), "耕地 永久基本农田 挖塘养鱼 禁止 土地用途 农业用途"),
        (("鱼塘", "违约"), "耕地 永久基本农田 挖塘养鱼 禁止 擅自改变农业用途"),
        (("鱼塘", "违约"), "基本农田保护区 建窑 建房 建坟 挖砂 采石 采矿 取土 挖塘养鱼"),
        (("自杀",), "生命权 身体权 健康权 人格权 民法典 第九百九十条 民法典 第一千零二条 民法典 第一千一百六十五条 侵权责任 损害赔偿 精神损害赔偿 安全保障义务"),
        (("溺水",), "生命权 身体权 健康权 民法典 第九百九十条 民法典 第一千零二条 侵权责任 安全保障义务 损害赔偿"),
        (("坠楼",), "生命权 身体权 健康权 民法典 第九百九十条 民法典 第一千零二条 侵权责任 安全保障义务 损害赔偿"),
        (("尸体",), "遗体 人格利益 死者 民法典 第九百九十四条 民法典 第一千一百八十三条 姓名 肖像 名誉 荣誉 隐私 精神损害赔偿"),
        (("火化",), "遗体 人格利益 死者 民法典 第九百九十四条 民法典 第一千一百八十三条 近亲属 精神损害赔偿"),
        (("搜身",), "身体权 人身自由 人格尊严 民法典 第一千零三条 民法典 第一千零一十一条 民法典 第一千一百八十三条 侵权责任 精神损害赔偿"),
        (("惊吓",), "身体权 健康权 民法典 第一千零四条 民法典 第一千一百八十三条 侵权责任 损害赔偿 精神损害赔偿"),
        (("惹怒",), "身体权 健康权 民法典 第一千零四条 民法典 第一千一百八十三条 侵权责任 损害赔偿 精神损害赔偿"),
        (("精神损害",), "精神损害赔偿 民法典 第一千一百八十三条 人格权 民法典 第九百九十条 生命权 身体权 健康权 侵权责任"),
        (("医院", "自杀"), "医疗机构 患者 生命权 民法典 第九百九十条 民法典 第一千二百二十一条 安全保障义务 医疗损害责任 过错 损害赔偿"),
        (("医院", "赔偿"), "医疗机构 医务人员 诊疗活动 民法典 第一千二百一十八条 民法典 第一千二百二十一条 过错 医疗损害责任 损害赔偿"),
        (("孕检",), "医疗机构 医务人员 诊疗活动 民法典 第一千二百一十八条 民法典 第一千二百二十一条 过错 医疗损害责任 损害赔偿"),
        (("食品", "赔"), "食品安全 生产者 销售者 网络交易平台 民法典 第一千二百零三条 民法典 第一千零四条 损害赔偿 连带责任"),
        (("食品", "腹泻"), "食品安全 生产者 销售者 产品责任 民法典 第一千二百零三条 民法典 第一千零四条 损害赔偿"),
        (("假药",), "药品 生产者 销售者 医疗机构 产品责任 民法典 第一千二百零三条 民法典 第一千二百二十三条 损害赔偿 连带责任"),
        (("药", "肝损伤"), "药品 生产者 销售者 医疗机构 产品责任 民法典 第一千二百零三条 民法典 第一千二百二十三条 损害赔偿"),
        (("购物平台",), "网络交易平台 食品安全 产品责任 损害赔偿 连带责任"),
        (("代言人",), "广告 代言 推荐 食品药品 损害赔偿 连带责任"),
        (("产品侵权",), "产品缺陷 产品质量法 第四十一条 中华人民共和国民法典 第一千二百零三条 中华人民共和国民法典 第一千二百零七条 生产者 销售者 举证责任 侵权责任"),
        (("产品侵权", "惩罚性赔偿"), "中华人民共和国民法典 第一千二百零七条 产品缺陷 明知 仍然生产销售 赔偿 惩罚性赔偿"),
        (("产品侵权", "证据"), "民事诉讼法 第六十七条 产品质量法 第四十一条 产品缺陷 举证责任 证据 侵权责任"),
        (("产品侵权", "法院"), "最高人民法院关于适用中华人民共和国民事诉讼法的解释 第二十六条 侵权行为地 被告住所地 产品质量纠纷 管辖"),
        (("产品生产者",), "产品缺陷 产品质量法 第二十六条 产品质量法 第二十七条 产品质量法 第二十九条 产品质量法 第四十一条 生产者 质量义务 侵权责任 损害赔偿"),
        (("生产者", "义务"), "产品质量法 第二十六条 产品质量法 第二十七条 产品质量法 第二十九条 生产者 产品质量 标识 警示 不得生产"),
        (("销售者", "义务"), "产品质量法 第三十三条 产品质量法 第三十四条 产品质量法 第三十五条 销售者 进货检查验收 保持产品质量 不得销售"),
        (("销售者", "承担侵权责任"), "产品质量法 第四十条 产品质量法 第四十二条 销售者 赔偿责任 过错 不能指明生产者 供货者"),
        (("生产者承担",), "产品缺陷 产品质量法 第四十一条 生产者 侵权责任 损害赔偿"),
        (("姓名",), "姓名权 名称权 人格权 民法典 第一千零一十二条 民法典 第一千零一十五条 变更姓名 不得违背公序良俗"),
        (("笔名",), "姓名权 名称权 人格权 民法典 第一千零一十二条 民法典 第一千零一十七条 艺名 网名 参照适用"),
        (("艺名",), "姓名权 名称权 人格权 民法典 第一千零一十二条 民法典 第一千零一十七条 笔名 网名 参照适用"),
        (("网名",), "姓名权 名称权 人格权 民法典 第一千零一十二条 民法典 第一千零一十七条 笔名 艺名 参照适用"),
        (("名人", "商标"), "姓名权 名称权 肖像权 民法典 第一千零一十四条 民法典 第一千零一十七条 商标注册 损害社会公共利益 不得申请注册"),
        (("终止妊娠",), "生育权 夫妻 女方 终止妊娠 损害赔偿"),
    ]
    for required_terms, expansion in sorted(expansion_rules, key=lambda item: len(item[0]), reverse=True):
        if all(term in normalized for term in required_terms):
            expansions.append(expansion)

    return _unique_nonempty_queries(expansions)


def _build_evidence_bridge_lexical_queries(original_query: str, focused_query: str, *, max_queries: int) -> list[str]:
    max_queries = max(0, int(max_queries or 0))
    if max_queries <= 0:
        return []
    normalized = re.sub(r"\s+", "", f"{original_query} {focused_query}".casefold())
    if not normalized or not any(hint in normalized for hint in EVIDENCE_BRIDGE_QUESTION_HINTS):
        return []

    base = _normalize_query(focused_query or original_query)
    query_groups: list[str] = []
    generic_terms: list[str] = []

    if any(hint in normalized for hint in ("哪些", "有什么", "包括", "列出", "分别", "范围")):
        generic_terms.extend(["条件", "情形", "范围", "要求", "义务", "责任", "标准", "处理"])
    if any(hint in normalized for hint in ("是否", "能否", "可否", "可以", "会不会", "需不需要", "要不要")):
        generic_terms.extend(["条件", "要求", "规定", "允许", "禁止", "可以", "不得", "有效", "无效", "责任", "义务"])
    if any(hint in normalized for hint in ("责任", "赔偿", "损害", "损失", "侵害", "受伤", "承担")):
        generic_terms.extend(["责任", "义务", "赔偿", "损害", "过错", "风险", "处理"])
    if any(hint in normalized for hint in ("合同", "协议", "承诺", "约定")):
        generic_terms.extend(["合同", "协议", "效力", "有效", "无效", "条件", "程序", "同意", "审批"])
    if any(hint in normalized for hint in ("谁", "哪个", "哪一个", "找谁", "由谁")):
        generic_terms.extend(["主体", "负责人", "责任人", "部门", "机构", "范围"])
    if any(hint in normalized for hint in ("多少", "多久", "多长", "时限", "期限", "时间")):
        generic_terms.extend(["时限", "期限", "时间", "要求", "工作日", "完成", "响应"])

    generic_terms = _unique_strings(generic_terms)
    if generic_terms:
        query_groups.append(_normalize_query(f"{base} {' '.join(generic_terms[:12])}"))

    focused_terms = _important_query_terms(base)
    if focused_terms and generic_terms:
        query_groups.append(_normalize_query(f"{' '.join(focused_terms[:10])} {' '.join(generic_terms[:10])}"))
    if focused_terms:
        query_groups.append(_normalize_query(" ".join(focused_terms[:14])))

    return _unique_nonempty_queries(query_groups)[:max_queries]


def _important_query_terms(value: str) -> list[str]:
    normalized = _normalize_query(value)
    tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", normalized)
    ignored = {
        "是否",
        "能否",
        "可否",
        "可以",
        "哪些",
        "什么",
        "如何",
        "怎么",
        "多少",
        "多久",
        "需要",
        "要求",
    }
    result: list[str] = []
    for token in tokens:
        if token in ignored:
            continue
        if len(token) > 24 and re.fullmatch(r"[\u4e00-\u9fff]+", token):
            result.extend(token[index : index + 6] for index in range(0, len(token), 6))
        else:
            result.append(token)
    return _unique_strings(result)


def _build_definition_lexical_queries(original_query: str, focused_query: str) -> list[str]:
    text = _normalize_query(f"{original_query} {focused_query}")
    compact = re.sub(r"\s+", "", text.casefold())
    if _looks_like_table_lookup_question(compact):
        return []
    if not any(token in compact for token in ("定义", "是什么", "什么是", "是指什么", "含义", "概念")):
        return []

    patterns = (
        r"(?:对|关于)(?P<term>[\u4e00-\u9fffA-Za-z0-9《》“”\"_\-]{2,40}?)(?:本身)?(?:是怎么定义的|怎么定义|如何定义|的定义|定义)",
        r"(?P<term>[\u4e00-\u9fffA-Za-z0-9《》“”\"_\-]{2,40}?)(?:本身)?(?:是指什么|是什么|的含义|的概念|的定义)",
        r"什么是(?P<term>[\u4e00-\u9fffA-Za-z0-9《》“”\"_\-]{2,40})",
    )
    for pattern in patterns:
        match = re.search(pattern, compact)
        if not match:
            continue
        term = _clean_definition_term(match.group("term"))
        if term:
            return [
                f"{term} 是指 本办法所称 {term} 定义 含义 概念",
                f"本办法所称 {term} 是指",
            ]
    return []


def _looks_like_table_lookup_question(compact_query: str) -> bool:
    return (
        "表格或清单信息" in compact_query
        or "对应的数值" in compact_query
        or "对象或判断" in compact_query
    )


def _clean_definition_term(value: str) -> str:
    term = value.strip("《》“”\"' ")
    term = re.sub(r"^(?:办法|条例|规定|制度|文档|文件|手册)?(?:中|里)?", "", term)
    term = re.sub(r"(?:本身|相关|这个|该)$", "", term)
    term = term.strip("的 ")
    attribute_suffixes = (
        "要求",
        "时限",
        "材料",
        "流程",
        "步骤",
        "方式",
        "条件",
        "责任",
        "期限",
        "审批人",
        "字段",
        "内容",
        "事项",
    )
    if term.endswith(attribute_suffixes):
        return ""
    if len(term) < 2 or len(term) > 30:
        return ""
    return term


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
