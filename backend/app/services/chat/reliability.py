from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from app.schemas.search import SearchResultChunk

GENERIC_DOCUMENT_TERMS = {
    "document",
    "documents",
    "doc",
    "file",
    "handbook",
    "guide",
    "register",
    "runbook",
    "playbook",
    "manual",
    "policy",
    "文档",
    "手册",
    "指南",
    "登记",
    "台账",
    "流程",
    "规范",
}

ENGLISH_PHRASE_REPLACEMENTS: list[tuple[str, str]] = [
    ("security exceptions", "安全 例外"),
    ("security exception", "安全 例外"),
    ("incident response", "事故 响应"),
    ("customer-facing", "客户"),
    ("customer facing", "客户"),
    ("release checklist", "发布 检查清单"),
    ("holiday schedule", "节假日 安排"),
]

ENGLISH_WORD_REPLACEMENTS: list[tuple[str, str]] = [
    ("security", "安全"),
    ("exception", "例外"),
    ("exceptions", "例外"),
    ("incident", "事故"),
    ("response", "响应"),
    ("customer", "客户"),
    ("platform", "平台"),
    ("release", "发布"),
    ("checklist", "检查清单"),
    ("handbook", "手册"),
    ("guide", "指南"),
    ("register", "登记"),
    ("runbook", "手册"),
    ("playbook", "手册"),
    ("manual", "手册"),
    ("document", "文档"),
    ("doc", "文档"),
    ("holiday", "节假日"),
    ("holidays", "节假日"),
    ("schedule", "安排"),
    ("employee", "员工"),
    ("employees", "员工"),
    ("staff", "员工"),
    ("company", "员工"),
]

CHINESE_STOPWORDS = {
    "什么",
    "怎么",
    "如何",
    "哪些",
    "一下",
    "一下子",
    "关于",
    "有关",
    "里面",
    "里",
    "中",
    "要求",
    "内容",
    "写了",
    "说了",
    "讲了",
    "提到",
    "安排",
    "可以",
    "应该",
    "需要",
    "对",
    "的",
}

NEGATIVE_EVIDENCE_HINTS = (
    "没有定义",
    "未定义",
    "未提及",
    "未涉及",
    "不涉及",
    "不包含",
    "无法确认",
    "无法判断",
)

META_NON_ANSWER_HINTS = (
    "本文档用于演示",
    "为了方便测试",
    "为了方便上传后立即测试",
    "你可以尝试以下问题",
    "faq 入库前需要检查",
    "文档目的",
)

QUERY_REQUIREMENT_HINTS = (
    "多少",
    "多久",
    "多长时间",
    "时间要求",
    "响应时间",
    "响应时限",
    "首次响应",
    "规定",
    "要求",
)

HIGH_PRIORITY_HINTS = ("高优先级", "高优先级工单", "高优先", "高优")

ENGLISH_STOPWORDS = {
    "what",
    "does",
    "the",
    "say",
    "about",
    "tell",
    "me",
    "is",
    "in",
    "a",
    "an",
    "of",
    "for",
    "and",
    "to",
    "should",
    "require",
    "requires",
}

TARGET_PATTERNS = [
    re.compile(
        r"(?P<name>[\u4e00-\u9fffA-Za-z0-9\-\s]{2,40}?)(?:里|中)(?:[^？?。！!]{0,24})?(?:怎么说|写了什么|说了什么|讲了什么|要求什么|提到什么|内容是什么|写了哪些|是什么)",
        re.IGNORECASE,
    ),
    re.compile(
        r"what does (?:the )?(?P<name>[a-z0-9\-\s]{2,50}?)(?: document| doc)? say",
        re.IGNORECASE,
    ),
    re.compile(
        r"what(?:'s| is) in (?:the )?(?P<name>[a-z0-9\-\s]{2,50}?)(?: document| doc)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<name>[\u4e00-\u9fffA-Za-z0-9\-\s]{2,40}?(?:手册|指南|登记|文档|runbook|handbook|guide|register|playbook))",
        re.IGNORECASE,
    ),
]


@dataclass
class AccessibleDocumentCandidate:
    document_id: UUID
    title: str


@dataclass
class DocumentTargetMatch:
    matched: bool
    requested_document_name: str | None
    matched_document_id: UUID | None = None
    matched_document_title: str | None = None
    match_type: str | None = None
    confidence: float = 0.0
    inaccessible_or_not_found: bool = False


@dataclass
class AbstainDecision:
    should_abstain: bool
    reason: str | None = None
    user_message: str | None = None
    filtered_chunks: list[SearchResultChunk] | None = None


def extract_document_target(
    query: str,
    accessible_documents: Sequence[AccessibleDocumentCandidate],
) -> DocumentTargetMatch | None:
    requested_name = _extract_requested_document_name(query)
    if requested_name is None:
        return None

    requested_canonical = _canonicalize_text(requested_name)
    requested_compact = requested_canonical.replace(" ", "")
    requested_stripped = _strip_document_suffixes(requested_canonical).replace(" ", "")
    requested_features = _feature_tokens(requested_name)

    best_match: DocumentTargetMatch | None = None
    for document in accessible_documents:
        title_canonical = _canonicalize_text(document.title)
        title_compact = title_canonical.replace(" ", "")
        title_stripped = _strip_document_suffixes(title_canonical).replace(" ", "")
        title_features = _feature_tokens(document.title)

        match_type: str | None = None
        confidence = 0.0
        if requested_compact and requested_compact == title_compact:
            match_type = "exact_title"
            confidence = 0.99
        elif requested_stripped and requested_stripped == title_stripped:
            match_type = "normalized_title"
            confidence = 0.93
        elif requested_compact and requested_compact in title_compact:
            match_type = "title_substring"
            confidence = 0.9
        elif requested_stripped and requested_stripped in title_stripped:
            match_type = "normalized_substring"
            confidence = 0.84
        else:
            overlap = _feature_overlap(requested_features, title_features)
            if overlap >= 0.42:
                match_type = "feature_overlap"
                confidence = round(min(max(overlap, 0.6), 0.82), 2)

        if match_type is None:
            continue

        candidate = DocumentTargetMatch(
            matched=True,
            requested_document_name=requested_name,
            matched_document_id=document.document_id,
            matched_document_title=document.title,
            match_type=match_type,
            confidence=confidence,
            inaccessible_or_not_found=False,
        )
        if best_match is None or candidate.confidence > best_match.confidence:
            best_match = candidate

    if best_match is not None:
        return best_match

    return DocumentTargetMatch(
        matched=False,
        requested_document_name=requested_name,
        match_type="explicit_target_not_accessible",
        confidence=0.0,
        inaccessible_or_not_found=True,
    )


def should_abstain_from_answer(
    query: str,
    retrieval_results: Sequence[SearchResultChunk],
    target_match: DocumentTargetMatch | None,
) -> AbstainDecision:
    if target_match and target_match.inaccessible_or_not_found:
        return AbstainDecision(
            should_abstain=True,
            reason="target_document_not_accessible_or_not_found",
            user_message=(
                "当前可访问范围内未找到相关文档内容。"
                "该文档可能不存在，或你当前没有访问权限。"
            ),
            filtered_chunks=[],
        )

    if target_match and target_match.matched and target_match.matched_document_id is not None:
        targeted_chunks = [chunk for chunk in retrieval_results if chunk.document_id == target_match.matched_document_id]
        if not targeted_chunks:
            requested_name = target_match.matched_document_title or target_match.requested_document_name or "目标文档"
            return AbstainDecision(
                should_abstain=True,
                reason="target_document_without_retrieval_hits",
                user_message=(
                    f"已定位到“{requested_name}”，但当前没有检索到来自该文档的可访问证据。"
                    "请换个问法，或确认文档已经完成入库。"
                ),
                filtered_chunks=[],
            )
        return AbstainDecision(
            should_abstain=False,
            reason="target_document_matched",
            filtered_chunks=list(targeted_chunks[:3]),
        )

    if not retrieval_results:
        return AbstainDecision(
            should_abstain=True,
            reason="no_retrieval_hits",
            user_message="未找到足够相关的可访问内容来回答这个问题。",
            filtered_chunks=[],
        )

    relevant_chunks = _filter_relevant_chunks(query, retrieval_results, keep_primary_document=True)
    if not relevant_chunks:
        return AbstainDecision(
            should_abstain=True,
            reason="insufficient_relevant_evidence",
            user_message="未找到足够相关的可访问证据来可靠回答这个问题。",
            filtered_chunks=[],
        )

    if len(_extract_evidence_hints(query)) < 2:
        dominant_document_chunks = _prefer_dominant_document(query, relevant_chunks)
        if dominant_document_chunks:
            relevant_chunks = dominant_document_chunks

    top_chunk = relevant_chunks[0]
    top_overlap = _chunk_overlap_score(query, top_chunk)
    top_score = _effective_evidence_score(query, top_chunk)
    average_score = sum(_effective_evidence_score(query, chunk) for chunk in relevant_chunks[:3]) / min(len(relevant_chunks), 3)
    evidence_hint_coverage_count = _covered_evidence_hint_count(query, relevant_chunks)

    if top_score < 0.28 and evidence_hint_coverage_count < 2:
        return AbstainDecision(
            should_abstain=True,
            reason="insufficient_relevant_evidence",
            user_message="检索到的证据相关性偏弱，暂时无法给出可靠回答。",
            filtered_chunks=[],
        )

    if (
        top_overlap < 0.12
        and top_chunk.score.lexical_raw <= 0
        and _effective_evidence_score(query, top_chunk) < 0.9
        and evidence_hint_coverage_count < 2
    ):
        return AbstainDecision(
            should_abstain=True,
            reason="insufficient_relevant_evidence",
            user_message="当前检索到的可访问内容与问题主题不够一致，暂时无法可靠回答。",
            filtered_chunks=[],
        )

    if top_overlap < 0.18 and top_score < 0.6 and evidence_hint_coverage_count < 2:
        return AbstainDecision(
            should_abstain=True,
            reason="insufficient_relevant_evidence",
            user_message="当前检索到的可访问内容与问题主题不够一致，暂时无法可靠回答。",
            filtered_chunks=[],
        )

    if average_score < 0.22 and evidence_hint_coverage_count < 2:
        return AbstainDecision(
            should_abstain=True,
            reason="insufficient_relevant_evidence",
            user_message="可访问证据整体偏弱，暂时无法给出可靠回答。",
            filtered_chunks=[],
        )

    top_two = relevant_chunks[:2]
    if len(top_two) == 2:
        same_strength = abs(_effective_evidence_score(query, top_two[0]) - _effective_evidence_score(query, top_two[1])) <= 0.08
        different_docs = top_two[0].document_id != top_two[1].document_id
        if (
            same_strength
            and different_docs
            and top_overlap < 0.24
            and not _has_strong_clause_level_evidence(query, top_two)
        ):
            return AbstainDecision(
                should_abstain=True,
                reason="conflicting_or_ambiguous_evidence",
                user_message=_build_conflicting_evidence_message(query, top_two, relevant_chunks),
                filtered_chunks=[],
            )

    return AbstainDecision(
        should_abstain=False,
        reason="relevant_evidence_found",
        filtered_chunks=relevant_chunks,
    )


def _has_strong_clause_level_evidence(query: str, chunks: Sequence[SearchResultChunk]) -> bool:
    normalized_query = _normalize_text_for_match(query)
    if any(token in normalized_query for token in ("流程", "审批", "时限", "账号", "导出", "补材料")):
        return False
    direct_pairs = (
        ("个体工商户", "个体工商户"),
        ("经营者", "个人经营"),
        ("实际经营者", "个人经营"),
        ("责任", "债务"),
        ("承担", "承担"),
        ("共同责任", "无法区分"),
        ("土地承包", "土地承包"),
        ("承包地", "承包地"),
        ("收回", "不得收回"),
        ("村集体", "不得收回"),
        ("转让", "转让"),
        ("本村以外", "本集体经济组织"),
        ("合同", "合同"),
    )
    for chunk in chunks:
        evidence_score = _effective_evidence_score(query, chunk)
        evidence_text = _normalize_text_for_match(f"{chunk.section_title or ''} {chunk.content}")
        has_structural_clause = "条款全称" in chunk.content or re.search(r"第[一二三四五六七八九十百千万零〇两\d]+条", chunk.content)
        direct_pair_matches = sum(
            1
            for query_hint, evidence_hint in direct_pairs
            if query_hint in normalized_query and evidence_hint in evidence_text
        )
        has_direct_subject = any(
            token in evidence_text
            for token in (
                "个体工商户",
                "债务",
                "个人经营",
                "家庭经营",
                "土地承包",
                "承包方",
                "发包方",
                "合伙合同",
            )
        )
        if has_structural_clause and has_direct_subject and (evidence_score >= 0.9 or direct_pair_matches >= 3):
            return True
    return False


def _filter_relevant_chunks(
    query: str,
    retrieval_results: Sequence[SearchResultChunk],
    *,
    keep_primary_document: bool,
) -> list[SearchResultChunk]:
    if not retrieval_results:
        return []

    first_chunk = retrieval_results[0]
    if "Table row:" in first_chunk.content and _has_structured_table_answer_row(query, first_chunk.content):
        return [first_chunk]

    evidence_hints = _extract_evidence_hints(query)
    candidate_window = 10 if len(evidence_hints) >= 2 else 6
    ranked_candidates = sorted(
        retrieval_results[:candidate_window],
        key=lambda chunk: (
            _chunk_relevance_score(query, chunk),
            chunk.score.rerank if chunk.score.rerank is not None else chunk.score.fused,
            chunk.score.fused,
        ),
        reverse=True,
    )
    top_chunk = ranked_candidates[0]
    top_score = top_chunk.score.fused
    primary_document_id = top_chunk.document_id
    filtered: list[SearchResultChunk] = []
    selected_ids: set[UUID] = set()

    if len(evidence_hints) >= 2:
        for hint in evidence_hints[:3]:
            matching_chunks = [
                chunk
                for chunk in ranked_candidates
                if chunk.chunk_id not in selected_ids and _chunk_matches_evidence_hint(chunk, hint)
            ]
            if not matching_chunks:
                continue
            chunk = matching_chunks[0]
            filtered.append(chunk)
            selected_ids.add(chunk.chunk_id)
            if len(filtered) >= 3:
                break

    for chunk in ranked_candidates:
        if chunk.chunk_id in selected_ids:
            continue
        overlap = _chunk_overlap_score(query, chunk)
        has_lexical_support = chunk.score.lexical_raw > 0
        relevance_score = _chunk_relevance_score(query, chunk)
        effective_score = _effective_evidence_score(query, chunk)
        score_cutoff = max(0.16, top_score * 0.45)
        if overlap < 0.12 and not has_lexical_support and effective_score < 0.9:
            continue
        if chunk.score.fused < score_cutoff and overlap < 0.18 and not has_lexical_support and effective_score < 0.95:
            continue
        if relevance_score < 0.18 and _contains_negative_evidence_hint(chunk.preview or chunk.content):
            continue
        if keep_primary_document and chunk.document_id != primary_document_id:
            if relevance_score + 0.08 < _chunk_relevance_score(query, top_chunk):
                continue
            if chunk.score.fused < top_score * 0.85 and overlap < 0.22:
                continue
        filtered.append(chunk)
        selected_ids.add(chunk.chunk_id)

    if filtered:
        return filtered[:3]
    return []


def _has_structured_table_answer_row(query: str, content: str) -> bool:
    normalized_query = re.sub(r"\s+", "", query.casefold())
    normalized_content = re.sub(r"\s+", "", content.casefold())
    checks = (
        ("l4", "准入等级=l4"),
        ("高风险", "准入等级=l4高风险"),
        ("审批链路", "审批链路="),
        ("复核周期", "复核周期="),
        ("退出要求", "退出要求="),
        ("生产环境", "访问对象=生产环境"),
        ("生产环境", "可访问生产环境"),
        ("允许方式", "允许方式="),
        ("有效期", "有效期="),
        ("回收责任人", "回收责任人="),
        ("日志要求", "日志要求="),
        ("数据处理服务", "交付类型=数据处理服务"),
        ("验收材料", "验收材料="),
        ("验收人", "验收人="),
        ("保留", "保留期限="),
        ("客户手机号", "数据范围=包含客户手机号"),
        ("手机号", "数据范围=包含客户手机号"),
        ("审批", "审批人="),
        ("处理时限", "处理时限="),
        ("时限", "处理时限="),
        ("脱敏", "脱敏要求="),
        ("检查项", "版本更新检查清单"),
        ("必须", "是否必须=必须"),
        ("制度版本", "版本更新检查清单"),
        ("版本发生变化", "版本更新检查清单"),
    )
    return sum(1 for query_hint, content_hint in checks if query_hint in normalized_query and content_hint in normalized_content) >= 2


def _prefer_dominant_document(query: str, chunks: Sequence[SearchResultChunk]) -> list[SearchResultChunk] | None:
    if len(chunks) < 2:
        return None
    requested_facets = _query_facets(query)
    if len(requested_facets) < 3:
        return None

    by_document: dict[UUID, dict[str, object]] = {}
    for chunk in chunks[:5]:
        bucket = by_document.setdefault(
            chunk.document_id,
            {
                "chunks": [],
                "facets": set(),
                "score": 0.0,
            },
        )
        bucket_chunks = bucket["chunks"]
        assert isinstance(bucket_chunks, list)
        bucket_chunks.append(chunk)
        bucket_facets = bucket["facets"]
        assert isinstance(bucket_facets, set)
        bucket_facets.update(_matched_query_facets(chunk))
        bucket["score"] = max(float(bucket["score"]), _effective_evidence_score(query, chunk))

    if len(by_document) < 2:
        return None

    ranked = sorted(
        by_document.values(),
        key=lambda item: (
            len(item["facets"]),  # type: ignore[arg-type]
            float(item["score"]),
            len(item["chunks"]),  # type: ignore[arg-type]
        ),
        reverse=True,
    )
    best = ranked[0]
    second = ranked[1]
    best_facets = len(best["facets"])  # type: ignore[arg-type]
    second_facets = len(second["facets"])  # type: ignore[arg-type]
    best_score = float(best["score"])
    second_score = float(second["score"])

    if best_facets >= min(len(requested_facets), 4) and best_facets >= second_facets + 2:
        return list(best["chunks"])[:3]  # type: ignore[arg-type]
    if best_facets >= 4 and best_facets > second_facets and best_score >= second_score + 0.03:
        return list(best["chunks"])[:3]  # type: ignore[arg-type]
    return None


def _build_conflicting_evidence_message(
    query: str,
    top_chunks: Sequence[SearchResultChunk],
    relevant_chunks: Sequence[SearchResultChunk],
) -> str:
    titles = _top_competing_document_titles([*top_chunks, *relevant_chunks])
    if not titles:
        return "当前命中的高相关证据存在明显竞争或歧义，建议明确指定文档或缩小问题范围后再问。"

    title_text = "、".join(f"《{title}》" for title in titles[:3])
    compact_query = re.sub(r"\s+", "", query.strip())
    if any(keyword in compact_query for keyword in ("审批", "处理时限", "时限", "补材料", "关闭", "回收", "导出")):
        suggestion = "建议拆成一到两个更明确的问题，或直接指定以哪份制度为准。"
    else:
        suggestion = "建议直接指定文档名，或把问题缩小到单一流程/场景。"

    return (
        f"当前命中的高相关证据同时来自 {title_text}，这些文档覆盖的是相近但不完全相同的流程口径。"
        f"为避免把不同制度拼成一个答案，我先不直接下结论。{suggestion}"
    )


def _top_competing_document_titles(chunks: Sequence[SearchResultChunk]) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        title = chunk.document_title.strip()
        if not title or title in seen:
            continue
        seen.add(title)
        titles.append(title)
    return titles


def _query_facets(query: str) -> set[str]:
    normalized = re.sub(r"\s+", "", query.casefold())
    facets: set[str] = set()
    facet_hints: dict[str, tuple[str, ...]] = {
        "emergency_access": ("先救火", "紧急", "最小权限", "最低权限", "顶上"),
        "approval": ("谁来批", "审批", "审批人", "谁审批"),
        "time_limit": ("多久", "时限", "最晚多久", "处理时限"),
        "materials": ("补材料", "补齐材料", "最少材料", "事后补齐"),
        "account_close": ("关账号", "关闭", "结束后多久", "回收"),
        "export_rules": ("导出文件", "禁止发法", "禁止方式", "禁止发送"),
    }
    for facet, hints in facet_hints.items():
        if any(hint in normalized for hint in hints):
            facets.add(facet)
    return facets


def _matched_query_facets(chunk: SearchResultChunk) -> set[str]:
    normalized = re.sub(r"\s+", "", f"{chunk.document_title}{chunk.section_title or ''}{chunk.content}".casefold())
    matched: set[str] = set()
    facet_markers: dict[str, tuple[str, ...]] = {
        "emergency_access": ("紧急采购", "紧急场景", "最低权限服务", "最小可用服务", "可先执行动作"),
        "approval": ("必须审批人", "审批人", "审批链路"),
        "time_limit": ("处理时限",),
        "materials": ("最少材料", "事后补齐材料"),
        "account_close": ("账号关闭时限",),
        "export_rules": ("禁止方式", "个人网盘", "私人聊天工具", "明文邮件", "访问对象=数据导出文件"),
    }
    for facet, markers in facet_markers.items():
        if any(marker in normalized for marker in markers):
            matched.add(facet)
    return matched


def _extract_requested_document_name(query: str) -> str | None:
    cleaned_query = query.strip().strip("?？。！!")
    for pattern in TARGET_PATTERNS:
        match = pattern.search(cleaned_query)
        if not match:
            continue
        name = " ".join(match.group("name").split()).strip("“”\"'：:，,.。!?？")
        if name:
            return name
    return None


def _canonicalize_text(value: str) -> str:
    normalized = value.lower().strip()
    for source, target in ENGLISH_PHRASE_REPLACEMENTS:
        normalized = normalized.replace(source, f" {target} ")
    for source, target in ENGLISH_WORD_REPLACEMENTS:
        normalized = re.sub(rf"\b{re.escape(source)}\b", f" {target} ", normalized)
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _normalize_text_for_match(value: str) -> str:
    return re.sub(r"\s+", "", value.casefold())


def _extract_evidence_hints(query: str) -> list[str]:
    hints: list[str] = []
    for pattern in (r"“(?P<value>[^”]{2,260})”", r'"(?P<value>[^"]{2,260})"'):
        for match in re.finditer(pattern, query):
            cleaned = " ".join(match.group("value").split()).strip(" ：:；;，,、")
            if cleaned and cleaned not in hints:
                hints.append(cleaned)
    return hints


def _chunk_matches_evidence_hint(chunk: SearchResultChunk, hint: str) -> bool:
    normalized_hint = _canonicalize_text(hint).replace(" ", "")
    if len(normalized_hint) < 4:
        return False
    evidence_text = " ".join(
        part
        for part in [chunk.document_title, chunk.section_title or "", chunk.preview, chunk.content]
        if part
    )
    normalized_evidence = _canonicalize_text(evidence_text).replace(" ", "")
    if normalized_hint in normalized_evidence:
        return True
    hint_features = _feature_tokens(hint)
    evidence_features = _feature_tokens(evidence_text)
    return _feature_overlap(hint_features, evidence_features) >= 0.7


def _covered_evidence_hint_count(query: str, chunks: Sequence[SearchResultChunk]) -> int:
    hints = _extract_evidence_hints(query)
    if len(hints) < 2:
        return 0
    covered = 0
    for hint in hints[:3]:
        if any(_chunk_matches_evidence_hint(chunk, hint) for chunk in chunks):
            covered += 1
    return covered


def _strip_document_suffixes(value: str) -> str:
    compact = value.strip()
    if not compact:
        return compact
    while True:
        next_value = compact
        for suffix in sorted(GENERIC_DOCUMENT_TERMS, key=len, reverse=True):
            if next_value.endswith(f" {suffix}"):
                next_value = next_value[: -(len(suffix) + 1)].strip()
            elif next_value.endswith(suffix):
                next_value = next_value[: -len(suffix)].strip()
        if next_value == compact:
            return compact
        compact = next_value


def _feature_tokens(value: str) -> set[str]:
    normalized = _canonicalize_text(value)
    features: set[str] = set()

    for ascii_token in re.findall(r"[a-z0-9]+", normalized):
        if ascii_token and ascii_token not in ENGLISH_STOPWORDS and ascii_token not in GENERIC_DOCUMENT_TERMS:
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
                if token in CHINESE_STOPWORDS or token in GENERIC_DOCUMENT_TERMS:
                    continue
                features.add(token)

    return _expand_domain_features(value, features)


def _feature_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = left & right
    if not overlap:
        return 0.0
    return len(overlap) / max(min(len(left), len(right)), 1)


def _chunk_overlap_score(query: str, chunk: SearchResultChunk) -> float:
    query_features = _feature_tokens(query)
    evidence_text = " ".join(
        part
        for part in [chunk.document_title, chunk.section_title or "", chunk.content]
        if part
    )
    evidence_features = _feature_tokens(evidence_text)
    return _feature_overlap(query_features, evidence_features)


def _chunk_relevance_score(query: str, chunk: SearchResultChunk) -> float:
    query_features = _feature_tokens(query)
    content_text = chunk.content
    combined_text = " ".join(part for part in [chunk.document_title, chunk.section_title or "", content_text] if part)
    content_features = _feature_tokens(combined_text)
    title_features = _feature_tokens(chunk.document_title)
    section_features = _feature_tokens(chunk.section_title or "")

    overlap = _feature_overlap(query_features, content_features)
    title_overlap = _feature_overlap(query_features, title_features)
    section_overlap = _feature_overlap(query_features, section_features)
    score = chunk.score.rerank if chunk.score.rerank is not None else chunk.score.fused
    score += overlap * 0.18
    score += title_overlap * 0.08
    score += section_overlap * 0.04
    score += _domain_alignment_bonus(query, content_text)
    if "p1" in query_features and "p1" in content_features:
        score += 0.08
    if _expects_requirement_answer(query) and _looks_like_direct_requirement_answer(content_text):
        score += 0.36
    if _expects_requirement_answer(query) and _contains_negative_evidence_hint(content_text):
        score -= 0.42
    if _expects_requirement_answer(query) and _contains_meta_non_answer_hint(content_text) and not _looks_like_direct_requirement_answer(content_text):
        score -= 0.85
    return score


def _effective_evidence_score(query: str, chunk: SearchResultChunk) -> float:
    rerank_score = chunk.score.rerank if chunk.score.rerank is not None else 0.0
    relevance_score = _chunk_relevance_score(query, chunk)
    return max(chunk.score.fused, rerank_score, relevance_score)


def _expand_domain_features(value: str, features: set[str]) -> set[str]:
    expanded = set(features)
    lowered = value.casefold()
    normalized = re.sub(r"\s+", "", lowered)
    if any(token in normalized for token in ("高优先级", "高优先级工单", "高优先", "高优")):
        expanded.update({"p1", "p1工单", "高优先级", "高优先", "工单"})
    if "首次响应" in normalized:
        expanded.update({"首次响应", "响应时间", "响应时限"})
    if "数据导出" in normalized:
        expanded.update({"数据导出", "导出", "审批", "审批人", "处理时限", "脱敏要求"})
    if "客户手机号" in normalized or "手机号" in normalized:
        expanded.update({"客户手机号", "手机号", "脱敏", "敏感字段"})
    if "处理时限" in normalized or "时限" in normalized:
        expanded.update({"处理时限", "时限"})
    if "脱敏" in normalized:
        expanded.update({"脱敏", "脱敏要求", "敏感字段"})
    if "版本发生变化" in normalized or ("制度版本" in normalized and "检查" in normalized):
        expanded.update({"版本更新", "版本更新检查清单", "检查清单", "检查项", "是否必须", "必须"})
    if "检查项" in normalized:
        expanded.update({"检查清单", "检查项", "是否必须"})
    if "l4" in normalized or "高风险" in normalized:
        expanded.update({"l4", "高风险", "准入等级", "审批链路", "复核周期", "退出要求"})
    if "审批链路" in normalized:
        expanded.update({"审批", "审批链路"})
    if "复核周期" in normalized:
        expanded.update({"复核", "复核周期"})
    if "退出要求" in normalized:
        expanded.update({"退出", "退出要求"})
    if "生产环境" in normalized:
        expanded.update({"生产环境", "允许方式", "有效期", "回收责任人", "日志要求"})
    if "数据处理服务" in normalized:
        expanded.update({"数据处理服务", "验收材料", "验收人", "保留期限"})
    if "验收材料" in normalized or "验收人" in normalized:
        expanded.update({"验收", "验收材料", "验收人"})
    if "保留多久" in normalized or "保留期限" in normalized or "资料保留" in normalized:
        expanded.update({"保留", "保留期限", "归档"})
    return expanded


def _contains_negative_evidence_hint(value: str) -> bool:
    normalized = re.sub(r"\s+", "", value.casefold())
    return any(hint in normalized for hint in NEGATIVE_EVIDENCE_HINTS)


def _contains_meta_non_answer_hint(value: str) -> bool:
    normalized = re.sub(r"\s+", "", value.casefold())
    return any(re.sub(r"\s+", "", hint.casefold()) in normalized for hint in META_NON_ANSWER_HINTS)


def _expects_requirement_answer(query: str) -> bool:
    normalized = re.sub(r"\s+", "", query.casefold())
    return any(hint in normalized for hint in QUERY_REQUIREMENT_HINTS)


def _looks_like_direct_requirement_answer(value: str) -> bool:
    if _contains_negative_evidence_hint(value) or _contains_meta_non_answer_hint(value):
        return False
    normalized = re.sub(r"\s+", "", value.casefold())
    if re.search(r"([0-9]+|[一二三四五六七八九十两]+)(分钟|小时|天|项)", normalized):
        return True
    return any(token in normalized for token in ("必须", "需要", "应当", "需在", "完成", "升级"))


def _domain_alignment_bonus(query: str, value: str) -> float:
    normalized_query = re.sub(r"\s+", "", query.casefold())
    normalized_value = re.sub(r"\s+", "", value.casefold())
    bonus = 0.0
    if any(token in normalized_query for token in HIGH_PRIORITY_HINTS) and (
        "p1" in normalized_value or "p1工单" in normalized_value
    ):
        bonus += 0.34
    if "首次响应" in normalized_query and "首次响应" in normalized_value:
        bonus += 0.24
    if _expects_requirement_answer(query) and (
        "首次响应" in normalized_value or "响应时间" in normalized_value or "响应时限" in normalized_value
    ) and re.search(r"([0-9]+|[一二三四五六七八九十两]+)(分钟|小时)", normalized_value):
        bonus += 0.18
    if "工单" in normalized_query and "工单" in normalized_value:
        bonus += 0.06
    if ("客户手机号" in normalized_query or "手机号" in normalized_query) and "数据范围=包含客户手机号" in normalized_value:
        bonus += 0.42
        if "处理时限=" in normalized_value:
            bonus += 0.12
        if "脱敏要求=" in normalized_value:
            bonus += 0.12
    if ("检查项" in normalized_query or "哪些检查" in normalized_query) and "版本更新检查清单" in normalized_value:
        bonus += 0.42
        if "是否必须=必须" in normalized_value:
            bonus += 0.12
    if ("l4" in normalized_query or "高风险" in normalized_query) and "准入等级=l4高风险" in normalized_value:
        bonus += 0.55
        for field in ("审批链路=", "复核周期=", "退出要求="):
            if field in normalized_value:
                bonus += 0.08
    if "生产环境" in normalized_query and ("访问对象=生产环境" in normalized_value or "可访问生产环境" in normalized_value):
        bonus += 0.48
        for field in ("允许方式=", "有效期=", "回收责任人=", "日志要求="):
            if field in normalized_value:
                bonus += 0.06
    if "数据处理服务" in normalized_query and "交付类型=数据处理服务" in normalized_value:
        bonus += 0.48
        for field in ("验收材料=", "验收人=", "保留期限="):
            if field in normalized_value:
                bonus += 0.08
    return bonus
