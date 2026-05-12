from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from app.core.config import get_settings
from app.schemas.search import SearchResultChunk
from app.services.chat.prompts import build_grounded_messages
from app.services.llm.openai_compatible import (
    create_openai_compatible_client,
    has_openai_compatible_credentials,
    uses_openai_compatible_provider,
)


@dataclass
class AnswerGenerationResult:
    answer: str
    insufficient_evidence: bool
    evidence_conflict: bool
    used_chunk_ids: list[str]
    answer_basis: str | None
    provider_name: str
    model_name: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int | None
    raw_payload: dict[str, Any] | None = None


class AnswerGenerator(Protocol):
    def generate(
        self,
        *,
        question: str,
        retrieved_chunks: list[SearchResultChunk],
        history_lines: list[str],
        conversation_context: str | None = None,
        allow_low_score: bool = False,
    ) -> AnswerGenerationResult: ...


class DeterministicAnswerGenerator:
    provider_name = "deterministic"
    model_name = "grounded-fallback-v2"

    def generate(
        self,
        *,
        question: str,
        retrieved_chunks: list[SearchResultChunk],
        history_lines: list[str],
        conversation_context: str | None = None,
        allow_low_score: bool = False,
    ) -> AnswerGenerationResult:
        started = time.perf_counter()
        if not retrieved_chunks:
            answer = "未找到足够相关的可访问证据来支持可靠回答。请换个问法，或确认相关文档已上传并已授权。"
            return AnswerGenerationResult(
                answer=answer,
                insufficient_evidence=True,
                evidence_conflict=False,
                used_chunk_ids=[],
                answer_basis="no_retrieval_hits",
                provider_name=self.provider_name,
                model_name=self.model_name,
                prompt_tokens=0,
                completion_tokens=max(len(answer) // 4, 1),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        selected_chunks = _select_grounded_chunks(retrieved_chunks, allow_low_score=allow_low_score)
        if not selected_chunks:
            answer = "当前可访问证据不足，暂时无法给出可靠回答。请换个问法，或直接查看引用片段。"
            return AnswerGenerationResult(
                answer=answer,
                insufficient_evidence=True,
                evidence_conflict=False,
                used_chunk_ids=[],
                answer_basis="weak_evidence",
                provider_name=self.provider_name,
                model_name=self.model_name,
                prompt_tokens=0,
                completion_tokens=max(len(answer) // 4, 1),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        top_score = _candidate_signal_score(selected_chunks[0]) if selected_chunks else 0.0
        top_lexical = selected_chunks[0].score.lexical_raw if selected_chunks else 0.0
        evidence_conflict = _detect_evidence_conflict(retrieved_chunks)
        if not allow_low_score and top_score < 0.08 and top_lexical <= 0:
            answer = "当前证据强度不足，暂时无法给出可靠回答。请缩小问题范围，或直接查看相关引用片段。"
            return AnswerGenerationResult(
                answer=answer,
                insufficient_evidence=True,
                evidence_conflict=evidence_conflict,
                used_chunk_ids=[],
                answer_basis="weak_evidence",
                provider_name=self.provider_name,
                model_name=self.model_name,
                prompt_tokens=0,
                completion_tokens=max(len(answer) // 4, 1),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        summaries = [
            summary
            for summary in (_build_chunk_summary(chunk, question) for chunk in selected_chunks[:2])
            if summary
        ]
        answer = _compose_grounded_answer(question, selected_chunks[:2], summaries)
        if evidence_conflict and not allow_low_score:
            answer += " 但前两条高分证据来自不同文档，建议结合引用来源进一步确认。"

        return AnswerGenerationResult(
            answer=answer,
            insufficient_evidence=False,
            evidence_conflict=evidence_conflict,
            used_chunk_ids=[str(chunk.chunk_id) for chunk in selected_chunks[:2]],
            answer_basis="grounded_summary",
            provider_name=self.provider_name,
            model_name=self.model_name,
            prompt_tokens=max((len(question) + sum(len(chunk.content) for chunk in selected_chunks)) // 4, 1),
            completion_tokens=max(len(answer) // 4, 1),
            latency_ms=int((time.perf_counter() - started) * 1000),
            raw_payload={
                "history_lines": history_lines,
                "conversation_context": conversation_context,
                "allow_low_score": allow_low_score,
            },
        )


class OpenAIAnswerGenerator:
    provider_name = "openai-compatible"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.model_name = self.settings.effective_llm_chat_model

    def generate(
        self,
        *,
        question: str,
        retrieved_chunks: list[SearchResultChunk],
        history_lines: list[str],
        conversation_context: str | None = None,
        allow_low_score: bool = False,
    ) -> AnswerGenerationResult:
        started = time.perf_counter()
        client = create_openai_compatible_client(self.settings)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=build_grounded_messages(question, retrieved_chunks, history_lines, context_summary=conversation_context),
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        payload = _parse_json_payload(content)
        latency_ms = int((time.perf_counter() - started) * 1000)
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        return AnswerGenerationResult(
            answer=str(payload.get("answer") or "").strip(),
            insufficient_evidence=bool(payload.get("insufficient_evidence", False)),
            evidence_conflict=bool(payload.get("evidence_conflict", False)),
            used_chunk_ids=[str(item) for item in payload.get("used_chunk_ids", []) if str(item).strip()],
            answer_basis=str(payload.get("answer_basis") or "llm"),
            provider_name=self.provider_name,
            model_name=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            raw_payload=payload,
        )


class AnswerGeneratorFactory:
    @staticmethod
    def create() -> AnswerGenerator:
        settings = get_settings()
        if uses_openai_compatible_provider(settings.answer_provider) and has_openai_compatible_credentials(settings):
            return OpenAIAnswerGenerator()
        return DeterministicAnswerGenerator()



def _parse_json_payload(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        loaded = json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "answer": "模型返回了无效的结构化结果，暂时无法生成可靠回答。",
            "insufficient_evidence": True,
            "evidence_conflict": False,
            "used_chunk_ids": [],
            "answer_basis": "invalid_model_payload",
        }
    if not isinstance(loaded, dict):
        return {
            "answer": "模型返回的数据结构不符合预期，暂时无法生成可靠回答。",
            "insufficient_evidence": True,
            "evidence_conflict": False,
            "used_chunk_ids": [],
            "answer_basis": "invalid_model_payload",
        }
    return loaded



def _select_grounded_chunks(retrieved_chunks: list[SearchResultChunk], *, allow_low_score: bool) -> list[SearchResultChunk]:
    if not retrieved_chunks:
        return []
    top_document_id = retrieved_chunks[0].document_id
    same_document_chunks = [
        chunk
        for chunk in retrieved_chunks
        if chunk.document_id == top_document_id
        and (_candidate_signal_score(chunk) > 0.0 or chunk.score.lexical_raw > 0.0)
    ]
    if same_document_chunks:
        return same_document_chunks[:2]
    if allow_low_score:
        fallback_chunks = [chunk for chunk in retrieved_chunks if chunk.document_id == top_document_id]
        if fallback_chunks:
            return fallback_chunks[:2]
    return [
        chunk
        for chunk in retrieved_chunks[:2]
        if _candidate_signal_score(chunk) > 0.0 or chunk.score.lexical_raw > 0.0
    ]


def _candidate_signal_score(chunk: SearchResultChunk) -> float:
    return max(chunk.score.fused, chunk.score.rerank or 0.0)



def _build_chunk_summary(chunk: SearchResultChunk, question: str) -> str:
    table_summary = _build_table_row_summary(chunk.content, question)
    if table_summary:
        return table_summary

    source_text = " ".join(chunk.content.split())
    if not source_text:
        return ""
    sentences = _split_sentences(source_text)
    if not sentences:
        return ""

    query_terms = _extract_query_terms(question)
    scored_sentences = []
    for index, sentence in enumerate(sentences):
        score = _sentence_match_score(sentence, query_terms)
        scored_sentences.append((score, -index, sentence))
    scored_sentences.sort(reverse=True)

    selected: list[str] = []
    total_length = 0
    for score, _, sentence in scored_sentences:
        cleaned = _clean_summary_sentence(sentence)
        if not cleaned or cleaned in selected:
            continue
        if score <= 0 and selected:
            continue
        next_length = total_length + len(cleaned)
        if selected and next_length > 150:
            continue
        selected.append(cleaned)
        total_length = next_length
        if len(selected) >= 2:
            break

    if not selected:
        fallback_sentence = max(sentences, key=lambda item: len(item), default="")
        fallback_cleaned = _clean_summary_sentence(fallback_sentence)
        if fallback_cleaned:
            selected = [fallback_cleaned]

    return " ".join(item for item in selected if item)


def _build_table_row_summary(content: str, question: str) -> str:
    rows = _extract_table_rows(content)
    if not rows:
        return ""

    query_terms = _extract_query_terms(question)
    scored_rows: list[tuple[int, int, str]] = []
    for index, row in enumerate(rows):
        score = _sentence_match_score(row, query_terms)
        score += _table_field_match_bonus(question, row)
        scored_rows.append((score, -index, row))
    scored_rows.sort(reverse=True)

    selected: list[str] = []
    total_length = 0
    for score, _, row in scored_rows:
        if score <= 0 and selected:
            continue
        cleaned = _clean_table_row(row)
        if not cleaned or cleaned in selected:
            continue
        next_length = total_length + len(cleaned)
        if selected and next_length > 260:
            continue
        selected.append(cleaned)
        total_length = next_length
        if len(selected) >= 3:
            break

    if not selected and scored_rows:
        selected = [_clean_table_row(scored_rows[0][2])]

    return " ".join(item for item in selected if item)


def _extract_table_rows(content: str) -> list[str]:
    rows: list[str] = []
    for line in content.splitlines():
        cleaned = line.strip()
        if cleaned.startswith("Table row:"):
            rows.append(cleaned)
    return rows


def _clean_table_row(row: str) -> str:
    cleaned = row.removeprefix("Table row:").strip()
    cleaned = cleaned.replace(". ", "：", 1)
    cleaned = cleaned.replace("; ", "；")
    return _ensure_sentence_ending(_compact_text(cleaned, 180))


def _table_field_match_bonus(question: str, row: str) -> int:
    normalized_question = re.sub(r"\s+", "", question.casefold())
    normalized_row = re.sub(r"\s+", "", row.casefold())
    bonus = 0
    field_pairs = (
        ("l4", "准入等级=l4"),
        ("高风险", "准入等级=l4高风险"),
        ("审批链路", "审批链路"),
        ("复核周期", "复核周期"),
        ("退出要求", "退出要求"),
        ("生产环境", "访问对象=生产环境"),
        ("生产环境", "可访问生产环境"),
        ("允许方式", "允许方式"),
        ("有效期", "有效期"),
        ("回收责任人", "回收责任人"),
        ("日志要求", "日志要求"),
        ("数据处理服务", "交付类型=数据处理服务"),
        ("验收材料", "验收材料"),
        ("验收人", "验收人"),
        ("保留多久", "保留期限"),
        ("保留期限", "保留期限"),
        ("客户手机号", "客户手机号"),
        ("手机号", "手机号"),
        ("审批", "审批人"),
        ("处理时限", "处理时限"),
        ("时限", "处理时限"),
        ("脱敏", "脱敏要求"),
        ("检查项", "检查项="),
        ("必须", "是否必须=必须"),
        ("制度版本", "版本更新检查清单"),
        ("版本发生变化", "版本更新检查清单"),
    )
    for query_hint, row_hint in field_pairs:
        if query_hint in normalized_question and row_hint in normalized_row:
            bonus += 3
    return bonus



def _compose_grounded_answer(question: str, chunks: list[SearchResultChunk], summaries: list[str]) -> str:
    if not chunks:
        return "当前可访问证据不足，暂时无法给出可靠回答。"

    if not summaries:
        fallback = _compact_text(" ".join((chunks[0].preview or chunks[0].content).split()), 180)
        summaries = [_ensure_sentence_ending(fallback)] if fallback else []

    primary_title = chunks[0].document_title
    unique_titles = {chunk.document_title for chunk in chunks}
    clauses = [_to_clause(summary) for summary in summaries if _to_clause(summary)]

    if len(unique_titles) == 1:
        if len(clauses) >= 2:
            return f"根据当前可访问文档中的证据，{primary_title}主要有两点：第一，{clauses[0]}；第二，{clauses[1]}。"
        if len(clauses) == 1:
            return f"根据当前可访问文档中的证据，{primary_title}主要说明：{clauses[0]}。"
        return f"根据当前可访问文档中的证据，{primary_title}包含与该问题相关的说明，建议结合引用片段进一步确认。"

    paired = []
    for chunk, summary in zip(chunks, summaries, strict=False):
        clause = _to_clause(summary)
        if not clause:
            continue
        location = _format_location(chunk)
        paired.append(f"{chunk.document_title}（{location}）提到：{clause}")
    return _ensure_sentence_ending("根据当前可访问文档中的证据，" + "；".join(paired))



def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    ends_cleanly = bool(re.search(r"[。！？!?；;]$", normalized))
    parts = [part.strip(" ，；;。") for part in re.split(r"(?<=[。！？!?；;])\s*", normalized) if part.strip(" ，；;。")]
    if len(parts) > 1 and not ends_cleanly:
        parts = parts[:-1]
    return [part for part in parts if _looks_like_complete_sentence(part)]



def _extract_query_terms(question: str) -> list[str]:
    normalized = question.casefold()
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", normalized)
    english_terms = re.findall(r"[a-z0-9]{3,}", normalized)
    stop_terms = {
        "什么", "怎么", "如何", "多少", "哪些", "这个", "那个", "一下", "关于", "里面", "里", "中", "写了", "说了", "要求", "应该", "发生",
        "什么样", "问题", "内容", "文档", "指南", "手册", "登记",
        "what", "does", "document", "guide", "handbook", "runbook", "about", "from", "with", "should",
    }
    split_tokens = r"的|了|在|时|对|和|与|及|里|中|关于|什么|怎么|应该|是否"

    terms: list[str] = []

    for term in chinese_terms:
        for part in re.split(split_tokens, term):
            cleaned = part.strip()
            if len(cleaned) < 2 or cleaned in stop_terms:
                continue
            if len(cleaned) > 6:
                reduced = [cleaned[i:j] for i in range(len(cleaned)) for j in range(i + 2, min(len(cleaned), i + 5) + 1)]
                for item in reduced:
                    if item not in stop_terms and item not in terms:
                        terms.append(item)
            elif cleaned not in terms:
                terms.append(cleaned)

    for term in english_terms:
        if term in stop_terms:
            continue
        if term not in terms:
            terms.append(term)
    return terms



def _sentence_match_score(sentence: str, query_terms: list[str]) -> int:
    normalized = sentence.casefold()
    score = 0
    for term in query_terms:
        if term in normalized:
            score += 2 if len(term) >= 4 else 1
    if any(token in normalized for token in ("需要", "应", "必须", "负责", "安排", "流程", "步骤", "升级", "回滚", "沟通", "复盘")):
        score += 1
    return score



def _clean_summary_sentence(sentence: str) -> str:
    cleaned = re.sub(r"\s+", " ", sentence).strip(" ：:；;，,")
    return _ensure_sentence_ending(_compact_text(cleaned, 120))


def _looks_like_complete_sentence(sentence: str) -> bool:
    cleaned = sentence.strip()
    if len(cleaned) < 12:
        return False
    if cleaned.endswith(("例如", "比如", "另外", "还有", "只有这", "以及", "并", "和")):
        return False
    if re.search(r"[A-Za-z0-9\u4e00-\u9fff]$", cleaned) and len(cleaned) <= 16:
        return False
    return True



def _compact_text(text: str, limit: int) -> str:
    compact = " ".join(text.split()).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _ensure_sentence_ending(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    if cleaned.endswith(("。", "！", "？", "…")):
        return cleaned
    if cleaned.endswith(("；", ";", "：", ":", "，", ",")):
        return cleaned[:-1].rstrip() + "。"
    return cleaned + "。"


def _to_clause(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"[。！？!?；;，,：:]$", "", cleaned)
    return cleaned.strip()



def _format_location(chunk: SearchResultChunk) -> str:
    if chunk.page_number_start is not None:
        return f"第 {chunk.page_number_start} 页"
    if chunk.paragraph_start is not None:
        return f"第 {chunk.paragraph_start} 段"
    return f"分块 {chunk.chunk_index}"



def _detect_evidence_conflict(chunks: list[SearchResultChunk]) -> bool:
    if len(chunks) < 2:
        return False
    first = chunks[0]
    second = chunks[1]
    return first.document_id != second.document_id and abs(first.score.fused - second.score.fused) <= 0.1
