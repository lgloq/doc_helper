from __future__ import annotations

from difflib import SequenceMatcher
import re
import unicodedata
from typing import Any


SUPPORTED_THRESHOLD = 0.75
PARTIAL_THRESHOLD = 0.5


def build_evidence_audit(answer_text: str, selected_chunks: list[Any]) -> dict:
    claims = _extract_answer_claims(answer_text)
    evidence_payloads = _evidence_payloads(selected_chunks)
    if not claims:
        return {
            "status": "not_applicable",
            "score": None,
            "claim_count": 0,
            "supported_count": 0,
            "partial_count": 0,
            "unsupported_count": 0,
            "claims": [],
            "extraction_method": "deterministic_sentence_clause_split",
        }

    scored_claims: list[dict] = []
    for index, claim in enumerate(claims, start=1):
        best = {
            "score": 0.0,
            "reasons": ["no selected citation evidence"],
            "citations": [],
        }
        for payload in evidence_payloads:
            candidate = _score_claim_against_evidence(claim["text"], payload["text"])
            if candidate["score"] > best["score"]:
                best = {
                    "score": candidate["score"],
                    "reasons": candidate["reasons"],
                    "citations": [
                        {
                            "rank": payload["rank"],
                            "chunk_id": payload["chunk_id"],
                            "document_id": payload["document_id"],
                            "document_title": payload["document_title"],
                            "version_number": payload["version_number"],
                            "location": payload["location"],
                        }
                    ],
                }

        support_score = _round_score(best["score"])
        scored_claims.append(
            {
                "index": index,
                "text": claim["text"],
                "normalized": claim["normalized"],
                "length": claim["length"],
                "support_status": _support_status(support_score),
                "support_score": support_score,
                "support_citations": best["citations"],
                "support_reasons": best["reasons"],
            }
        )

    supported_count = sum(1 for item in scored_claims if item["support_status"] == "supported")
    partial_count = sum(1 for item in scored_claims if item["support_status"] == "partial")
    unsupported_count = sum(1 for item in scored_claims if item["support_status"] == "unsupported")
    score = _round_score(sum(item["support_score"] for item in scored_claims) / len(scored_claims))
    return {
        "status": _overall_status(supported_count, partial_count, unsupported_count),
        "score": score,
        "claim_count": len(scored_claims),
        "supported_count": supported_count,
        "partial_count": partial_count,
        "unsupported_count": unsupported_count,
        "claims": scored_claims,
        "extraction_method": "deterministic_sentence_clause_split",
    }


def _support_status(score: float) -> str:
    if score >= SUPPORTED_THRESHOLD:
        return "supported"
    if score >= PARTIAL_THRESHOLD:
        return "partial"
    return "unsupported"


def _overall_status(supported_count: int, partial_count: int, unsupported_count: int) -> str:
    if unsupported_count:
        return "needs_review"
    if partial_count:
        return "partial"
    if supported_count:
        return "supported"
    return "not_applicable"


def _extract_answer_claims(answer_text: str) -> list[dict]:
    cleaned = _strip_citation_markers(answer_text or "")
    cleaned = re.sub(r"\s+", " ", cleaned.replace("\r", "\n")).strip()
    if not cleaned:
        return []

    cleaned = re.sub(
        r"(?:^|[，,；;。]\s*)(第一|第二|第三|第四|第五|第六|第七|第八|其一|其二|其三|一是|二是|三是)\s*[，,:：]",
        "；",
        cleaned,
    )
    parts = [
        part.strip()
        for part in re.split(r"[。！？!?；;\n]+", cleaned)
        if part.strip()
    ]

    claims: list[dict] = []
    seen: set[str] = set()
    for raw_part in parts:
        claim_text = _clean_claim_candidate(raw_part)
        if not _looks_like_factual_claim(claim_text):
            continue
        normalized = _normalize_match_text(claim_text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        claims.append(
            {
                "text": claim_text,
                "normalized": normalized,
                "length": len(normalized),
            }
        )
    return claims


def _strip_citation_markers(text: str) -> str:
    cleaned = re.sub(r"\[[0-9,\s]+\]", "", text)
    cleaned = re.sub(r"【[^】]{1,80}】", "", cleaned)
    cleaned = re.sub(r"\(来源[:：][^)）]{1,120}[)）]", "", cleaned)
    return cleaned


def _clean_claim_candidate(text: str) -> str:
    cleaned = text.strip(" \t\r\n，,。；;：:")
    cleaned = re.sub(r"^根据当前可访问文档中的证据[，,]?", "", cleaned)
    cleaned = re.sub(r"^当前可访问文档中的证据[，,]?", "", cleaned)
    cleaned = re.sub(r"^《[^》]{1,120}》(?:里|中)?", "", cleaned)
    cleaned = re.sub(r"^[^：:]{0,160}（[^）]{1,120}）提到[:：]", "", cleaned)
    cleaned = re.sub(r"^[^：:]{0,160}(?:提到|显示|说明|指出)[:：]", "", cleaned)
    cleaned = re.sub(
        r"^[^：:]{0,160}(?:主要有[一二三四五六七八九十两0-9]+点|主要说明|直接相关的要求是|要求是|要求为)[:：]",
        "",
        cleaned,
    )
    cleaned = re.sub(r"^(?:第一|第二|第三|第四|第五|第六|第七|第八|其一|其二|其三|一是|二是|三是)[，,:：]?", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ，,。；;：:")
    return cleaned


def _looks_like_factual_claim(text: str) -> bool:
    if not text:
        return False
    normalized = _normalize_match_text(text)
    if len(normalized) < 6 and not _extract_numeric_constraints(text):
        return False
    low_information_markers = (
        "未找到足够相关",
        "证据不足",
        "暂时无法",
        "请换个问法",
        "建议结合引用片段",
        "建议你继续",
        "建议你把问题",
        "我已经检索到",
        "进一步确认",
        "当前可访问范围内未找到",
        "文档可能不存在",
        "没有访问权限",
    )
    if any(marker in text for marker in low_information_markers):
        return False
    factual_markers = (
        "为",
        "是",
        "包括",
        "包含",
        "需要",
        "必须",
        "应",
        "不得",
        "禁止",
        "负责",
        "审批",
        "处理",
        "时限",
        "要求",
        "同步",
        "补齐",
        "关闭",
        "回收",
        "导出",
        "升级",
        "建立",
        "明确",
        "可以",
        "先",
    )
    return len(normalized) >= 14 or any(marker in text for marker in factual_markers)


def _evidence_payloads(chunks: list[Any]) -> list[dict]:
    payloads: list[dict] = []
    for index, chunk in enumerate(chunks, start=1):
        parts: list[str] = []
        for key in (
            "document_title",
            "section_title",
            "heading_path",
            "clause_full_name",
            "article_number",
            "preview",
            "content",
        ):
            value = _read_attr(chunk, key)
            if isinstance(value, str) and value.strip():
                parts.append(value)

        chunk_id = _read_attr(chunk, "chunk_id")
        document_id = _read_attr(chunk, "document_id")
        document_title = _read_attr(chunk, "document_title")
        version_number = _read_attr(chunk, "version_number")
        payloads.append(
            {
                "rank": index,
                "chunk_id": str(chunk_id) if chunk_id is not None else None,
                "document_id": str(document_id) if document_id is not None else None,
                "document_title": document_title,
                "version_number": version_number,
                "location": _location_label(chunk),
                "text": " ".join(parts),
            }
        )
    return payloads


def _read_attr(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _location_label(chunk: Any) -> str | None:
    page = _read_attr(chunk, "page_number_start")
    paragraph = _read_attr(chunk, "paragraph_start")
    if page:
        return f"第 {page} 页"
    if paragraph:
        return f"第 {paragraph} 段"
    chunk_index = _read_attr(chunk, "chunk_index")
    if chunk_index is not None:
        return f"分块 {chunk_index}"
    return None


def _score_claim_against_evidence(claim_text: str, evidence_text: str) -> dict:
    claim_norm = _normalize_match_text(claim_text)
    evidence_norm = _normalize_match_text(evidence_text)
    if not claim_norm or not evidence_norm:
        return {"score": 0.0, "reasons": ["empty claim or evidence"]}
    if claim_norm in evidence_norm:
        return {"score": 1.0, "reasons": ["normalized claim is contained in citation evidence"]}

    lookup_score, lookup_reasons = _score_precise_table_lookup_support(claim_text, evidence_text)
    structured_score, structured_reasons = _score_structured_claim_support(claim_text, evidence_text)
    claim_constraints = _extract_numeric_constraints(claim_text)
    missing_constraints = [
        item
        for item in claim_constraints
        if _normalize_match_text(item) not in evidence_norm
    ]

    token_recall = _token_recall(
        _extract_support_tokens(claim_text),
        _extract_support_tokens(evidence_text),
    )
    ordered_ratio = _ordered_part_support_ratio(claim_text, evidence_norm)
    longest_ratio = _longest_common_match_ratio(claim_norm, evidence_norm)
    score = max(
        lookup_score,
        structured_score,
        (0.55 * token_recall) + (0.25 * ordered_ratio) + (0.20 * longest_ratio),
    )

    reasons = [
        f"token_recall={_round_score(token_recall)}",
        f"ordered_part_ratio={_round_score(ordered_ratio)}",
        f"longest_common_ratio={_round_score(longest_ratio)}",
    ]
    if lookup_reasons:
        reasons.extend(lookup_reasons)
    if structured_reasons:
        reasons.extend(structured_reasons)
    if missing_constraints:
        cap = 0.35 if len(missing_constraints) == len(claim_constraints) else 0.65
        score = min(score, cap)
        reasons.append(f"missing_numeric_or_date_constraints={missing_constraints}")

    relation_tokens = _extract_relation_tokens(claim_text)
    if relation_tokens:
        evidence_relation_tokens = _extract_relation_tokens(evidence_text)
        if not set(relation_tokens).intersection(evidence_relation_tokens):
            score = min(score, 0.65)
            reasons.append("claim relation/action tokens not found in evidence")

    if max(lookup_score, structured_score) <= 0 and token_recall < 0.28 and ordered_ratio < 0.34:
        score = min(score, 0.35)
        reasons.append("only weak topical overlap")

    return {
        "score": _clamp_score(score),
        "reasons": reasons,
    }


def _score_precise_table_lookup_support(claim_text: str, evidence_text: str) -> tuple[float, list[str]]:
    claim_norm = _normalize_match_text(claim_text)
    if not claim_norm:
        return 0.0, []

    claim_constraints = [_normalize_match_text(item) for item in _extract_numeric_constraints(claim_text)]
    response_markers = ("首次响应", "响应时间", "响应要求", "多久响应")
    if not any(marker in claim_norm for marker in response_markers):
        return 0.0, []

    evidence_units = _extract_table_row_evidence_units(evidence_text)
    if not evidence_units:
        evidence_units = [evidence_text]

    asks_p1 = any(marker in claim_norm for marker in ("p1", "高优先级", "高优"))
    for unit in evidence_units:
        unit_norm = _normalize_match_text(unit)
        if not unit_norm:
            continue
        if claim_constraints and not all(item in unit_norm for item in claim_constraints):
            continue
        field_supported = "首次响应" in unit_norm or "历史响应时间" in unit_norm
        p1_supported = (
            "工单等级p1" in unit_norm
            or "问题类型高优先级工单" in unit_norm
            or not asks_p1
        )
        if field_supported and p1_supported and (claim_constraints or "5分钟内" in unit_norm):
            return 1.0, ["precise_table_lookup_support=1.0"]

    return 0.0, []


def _extract_table_row_evidence_units(evidence_text: str) -> list[str]:
    rows = [
        match.group(0).strip()
        for match in re.finditer(r"Table row:.*?(?=(?:\s+Table row:)|$)", evidence_text, flags=re.S)
    ]
    return [row for row in rows if row]


def _score_structured_claim_support(claim_text: str, evidence_text: str) -> tuple[float, list[str]]:
    claim_pairs = _extract_key_value_pairs(claim_text)
    if not claim_pairs:
        return 0.0, []

    evidence_pairs = _extract_key_value_pairs(evidence_text)
    if not evidence_pairs:
        return 0.0, ["structured_claim_without_key_value_evidence"]

    pair_scores: list[float] = []
    for claim_key, claim_value in claim_pairs:
        best_pair_score = 0.0
        claim_key_tokens = _extract_support_tokens(claim_key)
        claim_value_tokens = _extract_support_tokens(claim_value)
        for evidence_key, evidence_value in evidence_pairs:
            key_score = _token_recall(claim_key_tokens, _extract_support_tokens(evidence_key))
            value_norm = _normalize_match_text(claim_value)
            evidence_value_norm = _normalize_match_text(evidence_value)
            if value_norm and value_norm in evidence_value_norm:
                value_score = 1.0
            else:
                value_score = _token_recall(
                    claim_value_tokens,
                    _extract_support_tokens(evidence_value),
                )
            best_pair_score = max(best_pair_score, (0.35 * key_score) + (0.65 * value_score))
        pair_scores.append(best_pair_score)

    score = sum(pair_scores) / len(pair_scores)
    return _clamp_score(score), [f"structured_pair_support={_round_score(score)}"]


def _extract_key_value_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    normalized = text.replace("：", "=").replace(":", "=")
    matches = list(re.finditer(r"(?P<key>[^=;；。,\n]{2,32}?)=", normalized))
    for index, match in enumerate(matches):
        key = match.group("key").strip(" ，,；;。")
        value_start = match.end()
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        value = normalized[value_start:value_end].strip(" ，,；;。")
        if key and value and len(_normalize_match_text(value)) >= 2:
            pairs.append((key, value))

    natural_patterns = (
        r"(?P<key>[\u4e00-\u9fffA-Za-z0-9]{2,18}?)(?:为|是|包括|包含)(?P<value>[^；;。,.，]{2,60})",
        r"(?P<key>账号)(?:应在|需要在)(?P<value>[^；;。,.，]{2,60})",
        r"(?P<key>导出文件|文件)(?:禁止通过|不得通过)(?P<value>[^；;。,.，]{2,60})",
        r"(?P<key>紧急场景|临时场景)(?:下)?可以先(?P<value>[^；;。,.，]{2,60})",
        r"(?P<key>先同步|同步)(?P<value>[^；;。,.，]{2,60})",
    )
    for pattern in natural_patterns:
        for match in re.finditer(pattern, text):
            key = match.group("key").strip()
            value = match.group("value").strip(" ，,；;。")
            if key and value:
                pairs.append((key, value))

    unique: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for key, value in pairs:
        identity = (_normalize_match_text(key), _normalize_match_text(value))
        if identity in seen:
            continue
        seen.add(identity)
        unique.append((key, value))
    return unique


def _extract_numeric_constraints(text: str) -> list[str]:
    patterns = (
        r"\d{4}[-/年]\d{1,2}(?:[-/月]\d{1,2}日?)?",
        r"\d+(?:\.\d+)?\s*(?:%|％|个工作日|工作日|分钟内|小时内|日内|天内|分钟|小时|个月|月|天|日|年|万元|亿元|元|人|次|条|份|级|类)",
        r"[一二两三四五六七八九十百千]+(?:个)?(?:工作日|分钟内|小时内|日内|天内|分钟|小时|个月|月|天|日|年)",
    )
    found: list[str] = []
    for pattern in patterns:
        found.extend(match.group(0).strip() for match in re.finditer(pattern, text))
    return list(dict.fromkeys(item for item in found if item))


def _extract_support_tokens(text: str) -> list[str]:
    normalized = text.casefold()
    tokens: list[str] = []
    tokens.extend(_normalize_match_text(item) for item in _extract_numeric_constraints(text))
    tokens.extend(re.findall(r"[a-z0-9]{2,}", normalized))
    stop_terms = {
        "根据",
        "当前",
        "可访问",
        "文档",
        "证据",
        "主要",
        "这个",
        "场景",
        "相关",
        "要求",
        "说明",
        "包括",
        "包含",
        "以及",
        "同时",
        "其中",
        "应当",
        "需要",
        "必须",
        "可以",
        "进行",
        "通过",
        "如果",
        "对于",
        "里面",
        "条款",
    }
    for segment in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
        parts = [
            part
            for part in re.split(r"的|了|在|和|与|及|或|并|对|中|里|为|是|由|将|把|向|于", segment)
            if len(part) >= 2 and part not in stop_terms
        ]
        for part in parts:
            tokens.append(part)
            if len(part) > 4:
                for size in (2, 3, 4):
                    tokens.extend(part[index : index + size] for index in range(0, len(part) - size + 1))
    return [token for token in dict.fromkeys(tokens) if token and token not in stop_terms]


def _extract_relation_tokens(text: str) -> list[str]:
    relation_markers = (
        "需要",
        "必须",
        "应",
        "不得",
        "禁止",
        "负责",
        "审批",
        "提交",
        "保留",
        "关闭",
        "回收",
        "升级",
        "建立",
        "明确",
        "同步",
        "补齐",
        "脱敏",
        "导出",
        "执行",
        "处理",
        "验收",
        "复核",
        "通知",
    )
    return [marker for marker in relation_markers if marker in text]


def _token_recall(claim_tokens: list[str], evidence_tokens: list[str]) -> float:
    if not claim_tokens:
        return 0.0
    evidence_set = set(evidence_tokens)
    if not evidence_set:
        return 0.0
    matched = sum(1 for token in claim_tokens if token in evidence_set)
    return matched / len(claim_tokens)


def _ordered_part_support_ratio(claim_text: str, evidence_norm: str) -> float:
    parts = _ordered_alias_parts(claim_text)
    if not parts:
        return 0.0
    matched = 0
    cursor = 0
    for part in parts:
        found_at = evidence_norm.find(part, cursor)
        if found_at < 0:
            continue
        matched += 1
        cursor = found_at + len(part)
    return matched / len(parts)


def _longest_common_match_ratio(claim_norm: str, evidence_norm: str) -> float:
    if not claim_norm or not evidence_norm:
        return 0.0
    evidence_window = evidence_norm[:5000]
    match = SequenceMatcher(None, claim_norm, evidence_window, autojunk=False).find_longest_match(
        0,
        len(claim_norm),
        0,
        len(evidence_window),
    )
    return match.size / len(claim_norm) if claim_norm else 0.0


def _normalize_match_text(text: str) -> str:
    return "".join(
        char
        for char in text.casefold()
        if not char.isspace() and unicodedata.category(char)[0] not in {"P", "S"}
    )


def _ordered_alias_parts(alias: str) -> list[str]:
    parts = [
        _normalize_match_text(part)
        for part in re.split(r"(?:并且|并|且|同时|然后|以及|需要|必须|应当|应|要|需|，|。|；|;|：|:|、|\(|\)|（|）)", alias.casefold())
    ]
    return [part for part in parts if len(part) >= 4]


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


def _round_score(value: float) -> float:
    return round(value, 4)
