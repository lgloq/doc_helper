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
        requested_name = target_match.requested_document_name or "目标文档"
        return AbstainDecision(
            should_abstain=True,
            reason="target_document_not_accessible_or_not_found",
            user_message=(
                f"当前可访问文档中未找到“{requested_name}”的相关内容。"
                "你可能没有权限访问该文档，或该文档不存在。"
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

    top_chunk = relevant_chunks[0]
    top_overlap = _chunk_overlap_score(query, top_chunk)
    top_score = top_chunk.score.fused
    average_score = sum(chunk.score.fused for chunk in relevant_chunks[:3]) / min(len(relevant_chunks), 3)

    if top_score < 0.28:
        return AbstainDecision(
            should_abstain=True,
            reason="insufficient_relevant_evidence",
            user_message="检索到的证据相关性偏弱，暂时无法给出可靠回答。",
            filtered_chunks=[],
        )

    if top_overlap < 0.12 and top_chunk.score.lexical_raw <= 0:
        return AbstainDecision(
            should_abstain=True,
            reason="insufficient_relevant_evidence",
            user_message="当前检索到的可访问内容与问题主题不够一致，暂时无法可靠回答。",
            filtered_chunks=[],
        )

    if top_overlap < 0.18 and top_score < 0.6:
        return AbstainDecision(
            should_abstain=True,
            reason="insufficient_relevant_evidence",
            user_message="当前检索到的可访问内容与问题主题不够一致，暂时无法可靠回答。",
            filtered_chunks=[],
        )

    if average_score < 0.22:
        return AbstainDecision(
            should_abstain=True,
            reason="insufficient_relevant_evidence",
            user_message="可访问证据整体偏弱，暂时无法给出可靠回答。",
            filtered_chunks=[],
        )

    top_two = relevant_chunks[:2]
    if len(top_two) == 2:
        same_strength = abs(top_two[0].score.fused - top_two[1].score.fused) <= 0.08
        different_docs = top_two[0].document_id != top_two[1].document_id
        if same_strength and different_docs and top_overlap < 0.24:
            return AbstainDecision(
                should_abstain=True,
                reason="conflicting_or_ambiguous_evidence",
                user_message="当前可访问证据存在明显竞争或歧义，暂时不适合直接下结论。",
                filtered_chunks=[],
            )

    return AbstainDecision(
        should_abstain=False,
        reason="relevant_evidence_found",
        filtered_chunks=relevant_chunks,
    )


def _filter_relevant_chunks(
    query: str,
    retrieval_results: Sequence[SearchResultChunk],
    *,
    keep_primary_document: bool,
) -> list[SearchResultChunk]:
    if not retrieval_results:
        return []

    top_chunk = retrieval_results[0]
    top_score = top_chunk.score.fused
    primary_document_id = top_chunk.document_id
    filtered: list[SearchResultChunk] = []

    for chunk in retrieval_results[:6]:
        overlap = _chunk_overlap_score(query, chunk)
        has_lexical_support = chunk.score.lexical_raw > 0
        score_cutoff = max(0.16, top_score * 0.45)
        if overlap < 0.12 and not has_lexical_support:
            continue
        if chunk.score.fused < score_cutoff and overlap < 0.18 and not has_lexical_support:
            continue
        if keep_primary_document and chunk.document_id != primary_document_id:
            if chunk.score.fused < top_score * 0.9 or overlap < 0.24:
                continue
        filtered.append(chunk)

    if filtered:
        return filtered[:3]
    return []


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

    return features


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
        for part in [chunk.document_title, chunk.section_title or "", chunk.preview or chunk.content]
        if part
    )
    evidence_features = _feature_tokens(evidence_text)
    return _feature_overlap(query_features, evidence_features)
