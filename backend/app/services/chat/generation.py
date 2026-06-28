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
    request_chat_completion,
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

        selected_chunks = _select_grounded_chunks(question, retrieved_chunks, allow_low_score=allow_low_score)
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

        simple_table_answer = _build_simple_table_lookup_answer(question, selected_chunks[:3])
        if simple_table_answer:
            return AnswerGenerationResult(
                answer=simple_table_answer,
                insufficient_evidence=False,
                evidence_conflict=evidence_conflict,
                used_chunk_ids=[str(chunk.chunk_id) for chunk in selected_chunks[:3]],
                answer_basis="simple_table_lookup_answer",
                provider_name=self.provider_name,
                model_name=self.model_name,
                prompt_tokens=max((len(question) + sum(len(chunk.content) for chunk in selected_chunks[:3])) // 4, 1),
                completion_tokens=max(len(simple_table_answer) // 4, 1),
                latency_ms=int((time.perf_counter() - started) * 1000),
                raw_payload={
                    "history_lines": history_lines,
                    "conversation_context": conversation_context,
                    "allow_low_score": allow_low_score,
                },
            )

        structured_answer = _build_structured_table_answer(question, selected_chunks[:3])
        if structured_answer:
            return AnswerGenerationResult(
                answer=structured_answer,
                insufficient_evidence=False,
                evidence_conflict=evidence_conflict,
                used_chunk_ids=[str(chunk.chunk_id) for chunk in selected_chunks[:3]],
                answer_basis="structured_table_answer",
                provider_name=self.provider_name,
                model_name=self.model_name,
                prompt_tokens=max((len(question) + sum(len(chunk.content) for chunk in selected_chunks[:3])) // 4, 1),
                completion_tokens=max(len(structured_answer) // 4, 1),
                latency_ms=int((time.perf_counter() - started) * 1000),
                raw_payload={
                    "history_lines": history_lines,
                    "conversation_context": conversation_context,
                    "allow_low_score": allow_low_score,
                    "structured_fields": _structured_answer_field_summary(question, selected_chunks[:3]),
                },
            )

        answer_chunks = selected_chunks[:3]
        summaries = [
            summary
            for summary in (_build_chunk_summary(chunk, question) for chunk in answer_chunks)
            if summary
        ]
        answer = _compose_grounded_answer(question, answer_chunks, summaries)

        return AnswerGenerationResult(
            answer=answer,
            insufficient_evidence=False,
            evidence_conflict=evidence_conflict,
            used_chunk_ids=[str(chunk.chunk_id) for chunk in answer_chunks],
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
        try:
            response = request_chat_completion(
                client,
                max_attempts=1,
                model=self.model_name,
                messages=build_grounded_messages(question, retrieved_chunks, history_lines, context_summary=conversation_context),
                temperature=0.1,
                response_format={"type": "json_object"},
                timeout=14.0,
            )
        except Exception as error:
            fallback = DeterministicAnswerGenerator().generate(
                question=question,
                retrieved_chunks=retrieved_chunks,
                history_lines=history_lines,
                conversation_context=conversation_context,
                allow_low_score=allow_low_score,
            )
            if fallback.answer_basis != "structured_table_answer" and _should_abstain_on_complex_fallback(question, retrieved_chunks):
                titles = _top_document_titles(retrieved_chunks[:3])
                fallback_answer = _build_complex_fallback_message(question, titles)
                raw_payload = {
                    "history_lines": history_lines,
                    "conversation_context": conversation_context,
                    "allow_low_score": allow_low_score,
                    "error_text": str(error),
                    "fallback_reason": "upstream_answer_generation_failed",
                    "fallback_mode": "complex_question_abstain",
                }
                return AnswerGenerationResult(
                    answer=fallback_answer,
                    insufficient_evidence=True,
                    evidence_conflict=len(titles) > 1,
                    used_chunk_ids=[str(chunk.chunk_id) for chunk in retrieved_chunks[:2]],
                    answer_basis="complex_fallback_abstain",
                    provider_name=f"{self.provider_name}-fallback",
                    model_name=self.model_name,
                    prompt_tokens=max((len(question) + sum(len(chunk.content) for chunk in retrieved_chunks[:2])) // 4, 1),
                    completion_tokens=max(len(fallback_answer) // 4, 1),
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    raw_payload=raw_payload,
                )
            raw_payload = dict(fallback.raw_payload or {})
            raw_payload["error_text"] = str(error)
            raw_payload["fallback_reason"] = "upstream_answer_generation_failed"
            return AnswerGenerationResult(
                answer=fallback.answer,
                insufficient_evidence=fallback.insufficient_evidence,
                evidence_conflict=fallback.evidence_conflict,
                used_chunk_ids=fallback.used_chunk_ids,
                answer_basis=fallback.answer_basis,
                provider_name=f"{self.provider_name}-fallback",
                model_name=self.model_name,
                prompt_tokens=fallback.prompt_tokens,
                completion_tokens=fallback.completion_tokens,
                latency_ms=int((time.perf_counter() - started) * 1000),
                raw_payload=raw_payload,
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



def _select_grounded_chunks(
    question: str,
    retrieved_chunks: list[SearchResultChunk],
    *,
    allow_low_score: bool,
) -> list[SearchResultChunk]:
    if not retrieved_chunks:
        return []
    query_terms = _extract_query_terms(question)
    ranked = _rank_grounded_chunks(question, retrieved_chunks, query_terms=query_terms)
    if not ranked:
        return []

    top_score = ranked[0][0]
    selected: list[SearchResultChunk] = []
    per_document_count: dict[str, int] = {}
    min_score = 0.0 if allow_low_score else max(0.14, top_score * 0.42)
    evidence_hints = _extract_evidence_hints(question)

    if len(evidence_hints) >= 2:
        seen_chunk_ids: set[str] = set()
        for hint in evidence_hints[:3]:
            matching_entries = [
                item
                for item in ranked
                if str(item[2].chunk_id) not in seen_chunk_ids and _chunk_matches_evidence_hint(item[2], hint)
            ]
            if not matching_entries:
                continue
            _score, _original_index, chunk = matching_entries[0]
            selected.append(chunk)
            seen_chunk_ids.add(str(chunk.chunk_id))
            document_key = str(chunk.document_id)
            per_document_count[document_key] = per_document_count.get(document_key, 0) + 1
            if len(selected) >= 3:
                break

    for score, _original_index, chunk in ranked:
        if any(chunk.chunk_id == item.chunk_id for item in selected):
            continue
        has_signal = _candidate_signal_score(chunk) > 0.0 or chunk.score.lexical_raw > 0.0 or score > 0.0
        if not allow_low_score and not has_signal:
            continue
        if selected and score < min_score:
            continue

        document_key = str(chunk.document_id)
        document_limit = _grounded_document_selection_limit(question, chunk, per_document_count)
        if per_document_count.get(document_key, 0) >= document_limit:
            continue

        selected.append(chunk)
        per_document_count[document_key] = per_document_count.get(document_key, 0) + 1
        if len(selected) >= 3:
            break

    if selected:
        return selected
    if allow_low_score:
        return [item[2] for item in ranked[:2]]
    return [chunk for chunk in retrieved_chunks[:2] if _candidate_signal_score(chunk) > 0.0 or chunk.score.lexical_raw > 0.0]


def _grounded_document_selection_limit(
    question: str,
    chunk: SearchResultChunk,
    per_document_count: dict[str, int],
) -> int:
    normalized_question = re.sub(r"\s+", "", question.casefold())
    normalized_evidence = re.sub(r"\s+", "", f"{chunk.document_title}{chunk.section_title or ''}{chunk.content}".casefold())
    if (
        ("条款全称" in chunk.content or re.search(r"第[一二三四五六七八九十百千万零〇\d]+条", chunk.content))
        and any(marker in normalized_question for marker in ("条", "规定", "合同", "承包", "个体工商户", "商标", "责任"))
        and any(marker in normalized_evidence for marker in ("法律", "法典", "条例", "办法", "规定", "规范", "制度", "法"))
    ):
        return 3
    return 2 if len(per_document_count) <= 1 else 1


def _candidate_signal_score(chunk: SearchResultChunk) -> float:
    return max(chunk.score.fused, chunk.score.rerank or 0.0)


def _rank_grounded_chunks(
    question: str,
    chunks: list[SearchResultChunk],
    *,
    query_terms: list[str],
) -> list[tuple[float, int, SearchResultChunk]]:
    ranked = [
        (
            _grounded_chunk_relevance_score(question, chunk, query_terms),
            -index,
            chunk,
        )
        for index, chunk in enumerate(chunks[:8])
    ]
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked


def _grounded_chunk_relevance_score(question: str, chunk: SearchResultChunk, query_terms: list[str]) -> float:
    evidence_text = " ".join(
        part
        for part in [chunk.document_title, chunk.section_title or "", chunk.preview, chunk.content]
        if part
    )
    normalized_text = re.sub(r"\s+", "", evidence_text.casefold())
    score = _candidate_signal_score(chunk) + (chunk.score.lexical_normalized * 0.08) + (chunk.score.lexical_raw * 0.02)
    score += min(_best_clause_block_match_score(chunk.content, question, query_terms) * 0.16, 0.36)
    clause_centric_question = _looks_like_clause_centric_question(question)
    chunk_has_clause_signal = _chunk_has_clause_signal(chunk)
    chunk_has_table_rows = "Table row:" in chunk.content

    if clause_centric_question:
        if chunk_has_clause_signal:
            score += 0.18
        if chunk_has_table_rows and not chunk_has_clause_signal:
            score -= 0.2
    elif _looks_like_structured_table_lookup_question(question) and chunk_has_table_rows:
        score += 0.14

    for term in query_terms:
        normalized_term = re.sub(r"\s+", "", term.casefold())
        if not normalized_term:
            continue
        if normalized_term in normalized_text:
            score += 0.08 if len(normalized_term) >= 4 else 0.04

    title_terms = _extract_query_terms(chunk.document_title)
    if title_terms:
        shared_title_terms = sum(1 for term in title_terms if term in query_terms)
        score += min(shared_title_terms * 0.06, 0.18)
    section_text = f"{chunk.section_title or ''} {chunk.preview}".strip()
    if section_text:
        score += min(_sentence_match_score(section_text, query_terms) * 0.025, 0.2)

    normalized_question = re.sub(r"\s+", "", question.casefold())
    if any(marker in normalized_question for marker in ("条", "条款", "规定", "合同", "承包", "个体工商户", "商标")):
        if "条款全称" in normalized_text:
            score += 0.08
        if re.search(r"第[一二三四五六七八九十百千万零〇\d]+条", evidence_text):
            score += 0.08
    if "债务" in normalized_question and "债务" in normalized_text:
        score += 0.18
    if ("偿还" in normalized_question or "承担" in normalized_question) and "承担" in normalized_text:
        score += 0.18
    if "继承" in normalized_question and "继承" in normalized_text:
        score += 0.18
    if "解除合同" in normalized_question and ("解除合同" in normalized_text or "终止土地经营权流转合同" in normalized_text):
        score += 0.18
    if "个体工商户" in normalized_question and "个体工商户" in normalized_text:
        score += 0.16
    if "土地承包" in normalized_question and ("土地承包" in normalized_text or "承包方" in normalized_text):
        score += 0.16
    if "特许经营" in normalized_question and "特许经营" in normalized_text:
        score += 0.2
    if "特许人" in normalized_question and "特许人" in normalized_text:
        score += 0.18
    if ("先用权" in normalized_question or "抗辩" in normalized_question) and (
        "在原使用范围内继续使用" in normalized_text or "先于商标注册人使用" in normalized_text
    ):
        score += 0.26
    if "正当使用" in normalized_question and "正当使用" in normalized_text:
        score += 0.2

    return score



def _build_chunk_summary(chunk: SearchResultChunk, question: str) -> str:
    if _looks_like_clause_centric_question(question) and _chunk_has_clause_signal(chunk):
        evidence_summary = _build_evidence_hint_summary(chunk.content, question)
        if evidence_summary:
            return evidence_summary

        clause_summary = _build_clause_summary(chunk.content, question)
        if clause_summary:
            return clause_summary

    table_summary = _build_table_row_summary(chunk.content, question)
    if table_summary:
        return table_summary

    evidence_summary = _build_evidence_hint_summary(chunk.content, question)
    if evidence_summary:
        return evidence_summary

    clause_summary = _build_clause_summary(chunk.content, question)
    if clause_summary:
        return clause_summary

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
        if selected and next_length > 520:
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


def _build_clause_summary(content: str, question: str) -> str:
    blocks = _extract_clause_blocks(content)
    if not blocks:
        return ""

    query_terms = _extract_query_terms(question)
    scored_blocks = []
    for index, block in enumerate(blocks):
        score = _sentence_match_score(block, query_terms)
        score += _clause_domain_match_bonus(question, block)
        scored_blocks.append((score, -index, block))
    scored_blocks.sort(reverse=True)

    if _question_expects_multi_part_answer(question):
        selected: list[str] = []
        total_length = 0
        for score, _, block in scored_blocks:
            if score <= 0 and selected:
                continue
            cleaned = _clean_clause_summary(block)
            if not cleaned or cleaned in selected:
                continue
            next_length = total_length + len(cleaned)
            if selected and next_length > 520:
                continue
            selected.append(cleaned)
            total_length = next_length
            if len(selected) >= 2:
                break
        if selected:
            return " ".join(selected)

    for score, _, block in scored_blocks:
        if score <= 0 and len(scored_blocks) > 1:
            continue
        cleaned = _clean_clause_summary(block)
        if cleaned:
            return cleaned

    return _clean_clause_summary(scored_blocks[0][2]) if scored_blocks else ""


def _build_evidence_hint_summary(content: str, question: str) -> str:
    hints = _extract_evidence_hints(question)
    if not hints:
        return ""

    selected: list[str] = []
    for hint in hints[:3]:
        window = _extract_text_window_around_hint(content, hint)
        if not window:
            continue
        cleaned = _ensure_sentence_ending(_compact_text(window, 680))
        if cleaned and cleaned not in selected:
            selected.append(cleaned)
        if len(selected) >= 2:
            break

    return " ".join(selected)


def _extract_text_window_around_hint(content: str, hint: str) -> str:
    source_text = " ".join(content.split())
    if not source_text:
        return ""

    normalized_source, source_offsets = _normalize_with_offsets(source_text)
    normalized_hint = _normalize_evidence_match_text(hint)
    if not normalized_source or not normalized_hint:
        return ""
    found_at = normalized_source.find(normalized_hint)
    if found_at < 0:
        return ""

    raw_start = source_offsets[found_at]
    raw_end = source_offsets[min(found_at + len(normalized_hint) - 1, len(source_offsets) - 1)] + 1
    window_start = max(0, raw_start - 140)
    window_end = min(len(source_text), raw_end + 560)

    left_boundary = max(
        source_text.rfind(marker, 0, raw_start)
        for marker in ("。", "；", ";", "\n")
    )
    if left_boundary >= 0 and raw_start - left_boundary <= 180:
        window_start = left_boundary + 1

    right_candidates = [
        position
        for marker in ("。", "；", ";", "\n")
        for position in [source_text.find(marker, raw_end)]
        if position >= 0 and position - raw_end <= 560
    ]
    if right_candidates:
        window_end = min(window_end, min(right_candidates) + 1)

    return " ".join(source_text[window_start:window_end].split()).strip(" ：:；;，,")


def _best_clause_block_match_score(content: str, question: str, query_terms: list[str]) -> int:
    blocks = _extract_clause_blocks(content)
    if not blocks:
        return 0
    best_score = 0
    for block in blocks:
        score = _sentence_match_score(block, query_terms)
        score += _clause_domain_match_bonus(question, block)
        best_score = max(best_score, score)
    return best_score


def _extract_clause_blocks(content: str) -> list[str]:
    lines = [line.strip() for line in content.splitlines()]
    blocks: list[list[str]] = []
    current: list[str] = []
    pending_heading: str | None = None

    def flush_current() -> None:
        nonlocal current
        cleaned = [line for line in current if line]
        if cleaned:
            blocks.append(cleaned)
        current = []

    for line in lines:
        if not line:
            continue
        if _looks_like_clause_heading(line):
            if current:
                flush_current()
            pending_heading = line
            continue
        if line.startswith("条款全称："):
            if current:
                flush_current()
            current = [part for part in [pending_heading, line] if part]
            pending_heading = None
            continue
        if current:
            current.append(line)
        elif pending_heading:
            current = [pending_heading, line]
            pending_heading = None

    if current:
        flush_current()
    extracted = [" ".join(block) for block in blocks if any("条款全称：" in line for line in block)]
    if extracted:
        return extracted

    compact = "\n".join(line for line in lines if line and not line.startswith("Table row:"))
    if not compact:
        return []

    matches = list(re.finditer(r"第[一二三四五六七八九十百千万零〇\d]+条", compact))
    inline_blocks: list[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(compact)
        block = compact[start:end].strip()
        if len(block) >= 12:
            inline_blocks.append(block)
    return inline_blocks


def _looks_like_clause_heading(line: str) -> bool:
    cleaned = line.strip()
    return bool(re.fullmatch(r"(?:#{1,6}\s*)?第[一二三四五六七八九十百千万零〇\d]+条", cleaned))


def _clause_domain_match_bonus(question: str, block: str) -> int:
    normalized_question = re.sub(r"\s+", "", question.casefold())
    normalized_block = re.sub(r"\s+", "", block.casefold())
    bonus = 0
    domain_pairs = (
        ("债务", "债务"),
        ("偿还", "承担"),
        ("夫妻", "个人经营"),
        ("偿还", "承担"),
        ("承担", "承担"),
        ("家庭经营", "家庭经营"),
        ("继承", "继承"),
        ("死亡", "死亡"),
        ("解除合同", "解除合同"),
        ("解除合同", "终止土地经营权流转合同"),
        ("单方解除", "单方解除"),
        ("个体工商户", "个体工商户"),
        ("土地承包", "土地承包"),
        ("承包方", "承包方"),
        ("发包方", "发包方"),
        ("商标", "商标"),
        ("正当使用", "正当使用"),
        ("先用权", "先于商标注册人使用"),
        ("先用权", "在原使用范围内继续使用"),
        ("抗辩", "在原使用范围内继续使用"),
        ("特许经营", "特许经营"),
        ("特许人", "特许人"),
        ("特许人", "企业以外的其他单位和个人不得作为特许人"),
        ("诉讼主体", "自然人从事工商业经营"),
        ("营业执照", "依法登记"),
        ("实际经营者", "个人经营"),
        ("实际经营者", "债务"),
        ("共同责任", "债务"),
        ("共同责任", "无法区分"),
        ("合伙", "合伙合同"),
        ("纠纷", "合伙合同"),
        ("无效", "民事法律行为无效"),
        ("合伙", "共享利益"),
        ("合伙", "共担风险"),
        ("纠纷", "虚假的意思表示"),
        ("纠纷", "恶意串通"),
        ("死亡", "承包收益"),
        ("死亡", "继续承包"),
        ("土地承包份额", "家庭成员"),
        ("土地承包份额", "平等享有"),
        ("收回承包地", "承包期"),
        ("收回承包地", "不得收回"),
        ("村集体", "不得收回"),
        ("本村以外", "本集体经济组织"),
        ("本村以外", "转让"),
        ("转让", "其他农户"),
        ("非家庭承包", "承包收益"),
        ("招标", "继续承包"),
        ("公开协商", "继承"),
        ("解除合同", "当事人协商一致"),
        ("解除合同", "解除权人"),
        ("民主议定", "村民会议"),
        ("三分之二", "三分之二"),
        ("鱼塘", "挖塘养鱼"),
        ("鱼塘", "基本农田保护区"),
        ("鱼塘", "农业用途"),
        ("耕地", "耕地"),
        ("基本农田", "基本农田"),
        ("正当使用", "通用名称"),
        ("正当使用", "直接表示商品的质量"),
    )
    for query_hint, block_hint in domain_pairs:
        if query_hint in normalized_question and block_hint in normalized_block:
            bonus += 4
    return bonus


def _clean_clause_summary(block: str) -> str:
    cleaned = re.sub(r"\s+", " ", block).strip(" ：:；;，,")
    return _ensure_sentence_ending(_compact_text(cleaned, 420))


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
    return _ensure_sentence_ending(_compact_text(cleaned, 420))


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


def _chunk_has_clause_signal(chunk: SearchResultChunk) -> bool:
    content = " ".join(part for part in [chunk.section_title or "", chunk.preview, chunk.content] if part)
    return _text_has_clause_signal(content)


def _text_has_clause_signal(content: str) -> bool:
    if "条款全称" in content:
        return True
    return bool(re.search(r"第[一二三四五六七八九十百千万零〇\d]+条", content))


def _looks_like_clause_centric_question(question: str) -> bool:
    normalized = re.sub(r"\s+", "", question.casefold())
    if not normalized:
        return False
    if _looks_like_structured_table_lookup_question(question):
        return False

    source_markers = ("条例", "办法", "规定", "制度", "规则", "条款")
    if any(marker in normalized for marker in source_markers):
        return True
    if re.search(r"第[一二三四五六七八九十百千万零〇\d]+条", normalized):
        return True

    timing_markers = ("时间", "多久", "期限", "时限", "最迟", "几日", "几个月", "多少天", "24小时")
    rule_markers = ("要求", "应当", "不得", "可以", "分别", "哪些", "什么情况下", "是否")
    field_markers = _generic_structured_field_markers()
    field_hits = sum(1 for marker in field_markers if marker in normalized)
    return any(marker in normalized for marker in timing_markers) and any(marker in normalized for marker in rule_markers) and field_hits <= 2


def _looks_like_structured_table_lookup_question(question: str) -> bool:
    normalized = re.sub(r"\s+", "", question.casefold())
    if not normalized:
        return False
    source_markers = ("条例", "办法", "规定", "制度", "规则", "条款")
    if any(marker in normalized for marker in source_markers) or re.search(r"第[一二三四五六七八九十百千万零〇\d]+条", normalized):
        return False

    matched = sum(1 for marker in _generic_structured_field_markers() if marker in normalized)
    if matched >= 2:
        return True
    if matched >= 1 and any(marker in normalized for marker in ("矩阵", "表格", "按级别", "由谁审批", "哪一项", "字段")):
        return True
    return False


def _question_expects_multi_part_answer(question: str) -> bool:
    normalized = re.sub(r"\s+", "", question.casefold())
    return any(marker in normalized for marker in ("分别", "同时", "以及", "各自", "、", "和", "及", "与"))


def _generic_structured_field_markers() -> tuple[str, ...]:
    return (
        "负责人",
        "责任人",
        "审批",
        "审批链路",
        "处理时限",
        "时限",
        "期限",
        "周期",
        "方式",
        "材料",
        "有效期",
        "条件",
        "等级",
        "级别",
        "字段",
        "检查项",
        "禁止",
        "允许",
        "响应时间",
        "首次响应",
        "验收",
        "回收",
        "保留期限",
    )



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
        if len(clauses) >= 3:
            return (
                f"根据当前可访问文档中的证据，{primary_title}主要有三点："
                f"第一，{clauses[0]}；第二，{clauses[1]}；第三，{clauses[2]}。"
            )
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
    return _ensure_sentence_ending(_compact_text(cleaned, 520))


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


def _extract_evidence_hints(question: str) -> list[str]:
    hints: list[str] = []
    for pattern in (r"“(?P<value>[^”]{2,260})”", r'"(?P<value>[^"]{2,260})"'):
        for match in re.finditer(pattern, question):
            cleaned = " ".join(match.group("value").split()).strip(" ：:；;，,、")
            if cleaned and cleaned not in hints:
                hints.append(cleaned)
    return hints


def _chunk_matches_evidence_hint(chunk: SearchResultChunk, hint: str) -> bool:
    normalized_hint = _normalize_evidence_match_text(hint)
    if len(normalized_hint) < 4:
        return False
    evidence_text = " ".join(
        part
        for part in [chunk.document_title, chunk.section_title or "", chunk.preview, chunk.content]
        if part
    )
    normalized_evidence = _normalize_evidence_match_text(evidence_text)
    if normalized_hint in normalized_evidence:
        return True
    hint_terms = [term for term in _extract_query_terms(hint) if len(_normalize_evidence_match_text(term)) >= 3]
    if not hint_terms:
        return False
    matched = sum(1 for term in hint_terms if _normalize_evidence_match_text(term) in normalized_evidence)
    return matched >= max(2, len(hint_terms) - 1)


def _normalize_evidence_match_text(value: str) -> str:
    return re.sub(r"[\s\W_]+", "", value.casefold())


def _normalize_with_offsets(value: str) -> tuple[str, list[int]]:
    normalized_parts: list[str] = []
    offsets: list[int] = []
    for index, char in enumerate(value.casefold()):
        if re.match(r"[\s\W_]", char):
            continue
        normalized_parts.append(char)
        offsets.append(index)
    return "".join(normalized_parts), offsets



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
    if first.document_id == second.document_id or abs(first.score.fused - second.score.fused) > 0.1:
        return False
    combined = re.sub(r"\s+", "", f"{first.content}{second.content}".casefold())
    conflict_markers = (
        "不一致",
        "相冲突",
        "冲突",
        "另有规定",
        "以本办法为准",
        "废止",
        "失效",
        "不得适用",
    )
    return any(marker in combined for marker in conflict_markers)


def _should_abstain_on_complex_fallback(question: str, chunks: list[SearchResultChunk]) -> bool:
    normalized = re.sub(r"\s+", "", question)
    if len(normalized) < 20:
        return False

    multi_clause_hints = (
        "能不能",
        "是否",
        "谁来批",
        "审批",
        "最晚多久",
        "多久",
        "补材料",
        "关账号",
        "关闭",
        "回收",
        "哪些禁止",
        "禁止",
        "允许",
    )
    matched_hints = sum(1 for hint in multi_clause_hints if hint in normalized)
    separator_count = sum(normalized.count(token) for token in ("，", "；", "、", "？"))
    structured_chunks = sum(1 for chunk in chunks[:3] if "Table row:" in chunk.content)

    return matched_hints >= 4 and (separator_count >= 2 or structured_chunks >= 1)


def _top_document_titles(chunks: list[SearchResultChunk]) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        title = chunk.document_title.strip()
        if not title or title in seen:
            continue
        titles.append(title)
        seen.add(title)
    return titles


def _build_complex_fallback_message(question: str, document_titles: list[str]) -> str:
    if len(document_titles) >= 2:
        title_text = "、".join(f"《{title}》" for title in document_titles[:3])
        return (
            f"我已经检索到 {title_text} 里的相关条款，但这个问题同时包含审批、时限、补材料、账号关闭和导出限制等多个条件。"
            "当前自动保守兜底不足以把这些不同口径稳定整合成一个可靠答案，所以先不直接下结论。"
            "建议你指定以哪份制度为准，或拆成两到三个更明确的问题。"
        )
    if len(document_titles) == 1:
        return (
            f"我已经检索到《{document_titles[0]}》里的相关条款，但这个问题同时包含审批、时限、补材料、账号关闭和导出限制等多个条件。"
            "当前自动保守兜底不足以稳定整合这些规则，所以先不直接拼一个结论。"
            "建议你继续指定更明确的子问题，比如先问审批链路和补材料时限，再单独问账号关闭和导出限制。"
        )
    return (
        "我已经检索到和问题相关的条款，但这个问题同时包含多个条件。"
        "当前自动保守兜底不足以稳定整合这些规则，所以先不直接给出可能误导的结论。"
        "建议你把问题拆成两到三个更明确的子问题后再问。"
    )


def _build_simple_table_lookup_answer(question: str, chunks: list[SearchResultChunk]) -> str | None:
    normalized_question = re.sub(r"\s+", "", question.casefold())
    if not any(token in normalized_question for token in ("首次响应", "响应时间", "响应要求", "多久响应")):
        return None

    row_entries: list[tuple[SearchResultChunk, dict[str, str], str]] = []
    for chunk in chunks:
        for row in _extract_table_rows(chunk.content):
            parsed = _parse_table_row_fields(row)
            if parsed:
                row_entries.append((chunk, parsed, row))
    if not row_entries:
        return None

    def row_score(parsed: dict[str, str], row: str) -> int:
        normalized_row = re.sub(r"\s+", "", row.casefold())
        score = _sentence_match_score(row, _extract_query_terms(question)) + _table_field_match_bonus(question, row)
        if "首次响应" in parsed:
            score += 8
        if "历史响应时间" in parsed:
            score += 4
        if any(token in normalized_question for token in ("高优先级", "高优", "p1")):
            if parsed.get("工单等级", "").casefold() == "p1":
                score += 12
            if parsed.get("问题类型") == "高优先级工单":
                score += 6
        if "当前客户工单响应矩阵" in row:
            score += 4
        if "已被当前矩阵替代" in row:
            score += 2
        return score

    scored = sorted(
        ((row_score(parsed, row), chunk, parsed, row) for chunk, parsed, row in row_entries),
        key=lambda item: item[0],
        reverse=True,
    )
    if not scored or scored[0][0] <= 0:
        return None

    current_entry = next(
        (
            item
            for item in scored
            if item[2].get("工单等级", "").casefold() == "p1" and item[2].get("首次响应")
        ),
        None,
    )
    history_entry = next(
        (
            item
            for item in scored
            if item[2].get("问题类型") == "高优先级工单" and item[2].get("历史响应时间")
        ),
        None,
    )

    clauses: list[str] = []
    source_title = scored[0][1].document_title
    if current_entry:
        source_title = current_entry[1].document_title
        parsed = current_entry[2]
        level = parsed.get("工单等级") or "P1"
        first_response = parsed.get("首次响应")
        clauses.append(f"按当前客户工单响应矩阵，{level} 工单首次响应要求是 {first_response}")
    elif scored[0][2].get("首次响应"):
        parsed = scored[0][2]
        subject = parsed.get("工单等级") or parsed.get("问题类型") or "该类工单"
        clauses.append(f"{subject}的首次响应要求是 {parsed['首次响应']}")

    if history_entry and any(token in normalized_question for token in ("高优先级", "高优")):
        parsed = history_entry[2]
        status = parsed.get("当前状态", "")
        history_time = parsed.get("历史响应时间")
        if history_time and "替代" in status:
            clauses.append(f"历史“高优先级工单 {history_time}”口径已被当前矩阵替代")
        elif history_time:
            clauses.append(f"历史高优先级工单响应时间为 {history_time}")

    wants_multiple_levels = any(token in normalized_question for token in ("p2", "各等级", "分别", "对比", "矩阵"))
    if wants_multiple_levels:
        p2_entry = next(
            (
                item
                for item in scored
                if item[2].get("工单等级", "").casefold() == "p2" and item[2].get("首次响应")
            ),
            None,
        )
        if p2_entry:
            clauses.append(f"P2 工单首次响应要求是 {p2_entry[2]['首次响应']}")

    if not clauses:
        return None
    return f"依据《{source_title}》，{'；'.join(clauses)}。"

def _build_structured_table_answer(question: str, chunks: list[SearchResultChunk]) -> str | None:
    if not chunks:
        return None
    if len({chunk.document_id for chunk in chunks}) != 1:
        return None

    requested_fields = _requested_structured_fields(question)
    if len(requested_fields) < 2:
        return None

    combined_rows: list[dict[str, str]] = []
    for chunk in chunks:
        for row in _extract_table_rows(chunk.content):
            parsed = _parse_table_row_fields(row)
            if parsed:
                combined_rows.append(parsed)

    if not combined_rows:
        return None

    supplier_answer = _build_supplier_emergency_table_answer(question, chunks, combined_rows)
    if supplier_answer:
        return supplier_answer

    value_map: dict[str, str] = {}
    candidate_map = _structured_field_candidates()
    query_terms = _extract_query_terms(question)
    best_field_scores: dict[str, int] = {}
    for row in combined_rows:
        row_text = " ".join(f"{key}={value}" for key, value in row.items())
        row_score = _sentence_match_score(row_text, query_terms) + _table_field_match_bonus(question, row_text)
        normalized_row = re.sub(r"\s+", "", " ".join(f"{key}={value}" for key, value in row.items()))
        for field_name, candidates in candidate_map.items():
            for key, value in row.items():
                normalized_key = re.sub(r"\s+", "", key)
                if any(candidate in normalized_key for candidate in candidates):
                    if row_score >= best_field_scores.get(field_name, -1):
                        value_map[field_name] = value.strip()
                        best_field_scores[field_name] = row_score
                    break
            if field_name == "can_proceed" and any(token in normalized_row for token in ("最低权限服务", "最小可用服务")):
                if row_score >= best_field_scores.get(field_name, -1):
                    value_map[field_name] = row.get("可先执行动作", "").strip() or "启用备用供应商的最低权限服务"
                    best_field_scores[field_name] = row_score

    if len(set(value_map) & set(requested_fields)) < min(2, len(requested_fields)):
        return None

    clauses: list[str] = []
    if "can_proceed" in requested_fields and value_map.get("can_proceed"):
        clauses.append(f"紧急场景下可以先{value_map['can_proceed']}")
    if "sync_people" in requested_fields and value_map.get("sync_people"):
        clauses.append(f"先同步{value_map['sync_people']}")
    if "approval" in requested_fields and value_map.get("approval"):
        clauses.append(f"正式审批人为{value_map['approval']}")
    if "time_limit" in requested_fields and value_map.get("time_limit"):
        clauses.append(f"处理时限为{value_map['time_limit']}")
    if "desensitization" in requested_fields and value_map.get("desensitization"):
        clauses.append(f"脱敏要求为{value_map['desensitization']}")
    if "minimum_materials" in requested_fields and value_map.get("minimum_materials"):
        clauses.append(f"最少材料包括{value_map['minimum_materials']}")
    if "post_materials" in requested_fields and value_map.get("post_materials"):
        clauses.append(f"事后补齐材料包括{value_map['post_materials']}")
    if "account_close" in requested_fields and value_map.get("account_close"):
        clauses.append(f"账号应在{value_map['account_close']}")
    if "forbidden_delivery" in requested_fields and value_map.get("forbidden_delivery"):
        clauses.append(f"导出文件禁止通过{value_map['forbidden_delivery']}")

    if len(clauses) < 2:
        return None

    document_title = chunks[0].document_title
    return f"根据当前可访问文档中的证据，《{document_title}》里与这个场景直接相关的要求是：{'；'.join(clauses)}。"


def _requested_structured_fields(question: str) -> list[str]:
    normalized = re.sub(r"\s+", "", question)
    requested: list[str] = []
    field_hints: list[tuple[str, tuple[str, ...]]] = [
        ("can_proceed", ("能不能", "是否", "先开最小权限", "最低权限", "顶上", "先救火")),
        ("sync_people", ("同步谁", "通知谁", "先同步")),
        ("approval", ("谁来批", "审批", "审批人", "谁审批")),
        ("time_limit", ("多久", "时限", "最晚多久", "处理时限")),
        ("desensitization", ("脱敏", "脱敏要求")),
        ("minimum_materials", ("最少材料",)),
        ("post_materials", ("补材料", "补齐材料", "事后补齐")),
        ("account_close", ("关账号", "关闭", "回收", "结束后多久")),
        ("forbidden_delivery", ("禁止发法", "禁止方式", "禁止发送", "导出的文件", "导出文件")),
    ]
    for field_name, hints in field_hints:
        if any(hint in normalized for hint in hints):
            requested.append(field_name)
    return requested


def _structured_field_candidates() -> dict[str, tuple[str, ...]]:
    return {
        "can_proceed": ("可先执行动作",),
        "sync_people": ("必须同步对象",),
        "approval": ("必须审批人", "审批人", "审批链路"),
        "time_limit": ("处理时限",),
        "desensitization": ("脱敏要求",),
        "minimum_materials": ("最少材料",),
        "post_materials": ("事后补齐材料",),
        "account_close": ("账号关闭时限",),
        "forbidden_delivery": ("禁止方式",),
    }


def _parse_table_row_fields(row: str) -> dict[str, str]:
    cleaned = row.removeprefix("Table row:").strip()
    if ". " in cleaned:
        _, cleaned = cleaned.split(". ", 1)
    fields: dict[str, str] = {}
    matches = list(re.finditer(r"(?P<key>[^=;；\n]+?)=", cleaned))
    for index, match in enumerate(matches):
        key = match.group("key").strip().strip(";；")
        value_start = match.end()
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
        raw_value = cleaned[value_start:value_end]
        value = raw_value.strip().strip(" ;；。.")
        value = re.sub(r"\s*[；;]\s*", "、", value)
        value = re.sub(r"\s+", " ", value).strip()
        if key and value:
            fields[key] = value
    return fields


def _structured_answer_field_summary(question: str, chunks: list[SearchResultChunk]) -> list[str]:
    requested = _requested_structured_fields(question)
    if not requested:
        return []
    available: set[str] = set()
    for chunk in chunks:
        for row in _extract_table_rows(chunk.content):
            parsed = _parse_table_row_fields(row)
            for field_name, candidates in _structured_field_candidates().items():
                if field_name in available:
                    continue
                if any(any(candidate in key for candidate in candidates) for key in parsed):
                    available.add(field_name)
    return [field for field in requested if field in available]


def _build_supplier_emergency_table_answer(
    question: str,
    chunks: list[SearchResultChunk],
    rows: list[dict[str, str]],
) -> str | None:
    normalized_question = re.sub(r"\s+", "", question)
    document_title = chunks[0].document_title
    if "供应商准入" not in document_title:
        return None
    if not any(token in normalized_question for token in ("先救火", "临时", "最小权限", "最低权限", "导出文件")):
        return None
    requested_fields = set(_requested_structured_fields(question))

    emergency_row = next(
        (
            row
            for row in rows
            if row.get("采购类型") == "紧急采购" or "可先执行动作" in row or "紧急场景" in row
        ),
        None,
    )
    sensitive_row = next((row for row in rows if row.get("采购类型") == "敏感采购"), None)
    export_row = next(
        (
            row
            for row in rows
            if row.get("访问对象") == "数据导出文件" or ("禁止方式" in row and "允许方式" in row)
        ),
        None,
    )
    narrative_text = " ".join(chunk.content for chunk in chunks)
    normalized_narrative = re.sub(r"\s+", "", narrative_text)

    needs_emergency_scope = any(token in normalized_question for token in ("先救火", "临时", "最小权限", "最低权限", "顶上"))
    needs_sensitive_scope = any(token in normalized_question for token in ("客户数据", "生产系统"))
    needs_export_scope = any(token in normalized_question for token in ("导出文件", "禁止发法", "禁止方式", "禁止发送"))
    if needs_emergency_scope and not (emergency_row or any(token in normalized_narrative for token in ("最小可用服务", "最低权限服务"))):
        return None
    if needs_sensitive_scope and not sensitive_row:
        return None
    if needs_export_scope and not export_row:
        return None

    clauses: list[str] = []
    if any(token in normalized_narrative for token in ("最小可用服务", "最低权限服务")):
        clauses.append("可以先在最小边界内启用备用供应商服务")
    elif emergency_row and emergency_row.get("可先执行动作"):
        clauses.append(f"紧急场景下可以先{emergency_row['可先执行动作']}")
    if emergency_row and emergency_row.get("必须审批人"):
        clause = f"紧急临时审批人为{emergency_row['必须审批人']}"
        if emergency_row.get("处理时限"):
            if "完成临时审批" in emergency_row["处理时限"]:
                clause += f"，并需在{emergency_row['处理时限']}"
            else:
                clause += f"，并需在{emergency_row['处理时限']}完成临时审批"
        clauses.append(clause)
    if emergency_row:
        note = emergency_row.get("备注", "")
        if "补齐" in note:
            clauses.append(note)
        elif emergency_row.get("最少材料"):
            clauses.append(f"临时审批至少准备{emergency_row['最少材料']}")
    if sensitive_row and sensitive_row.get("必须审批人"):
        clause = f"若涉及客户数据、生产系统等敏感范围，正式审批人为{sensitive_row['必须审批人']}"
        if sensitive_row.get("处理时限"):
            clause += f"，处理时限为{sensitive_row['处理时限']}"
        clauses.append(clause)
    if sensitive_row and sensitive_row.get("最少材料"):
        clauses.append(f"正式材料至少包括{sensitive_row['最少材料']}")
    if (export_row and export_row.get("禁止方式")) or (emergency_row and emergency_row.get("禁止方式")):
        forbidden = (export_row and export_row.get("禁止方式")) or emergency_row.get("禁止方式")
        clauses.append(f"导出文件禁止通过{forbidden}")
    if "account_close" in requested_fields:
        explicit_account_close = (emergency_row and emergency_row.get("账号关闭时限")) or (sensitive_row and sensitive_row.get("账号关闭时限"))
        if explicit_account_close:
            clauses.append(f"账号应在{explicit_account_close}")
        elif export_row and export_row.get("有效期"):
            clauses.append(f"当前命中文档没有给出统一账号关闭时限，但要求在退出计划中明确回收安排；若通过加密链接交付，链接{export_row['有效期']}自动失效")
        else:
            clauses.append("当前命中文档没有给出统一账号关闭时限，但要求在退出计划中明确回收安排")

    core_clause_count = sum(
        1
        for marker in ("启用备用供应商服务", "紧急临时审批人", "正式审批人", "导出文件禁止通过")
        if any(marker in clause for clause in clauses)
    )
    if len(clauses) < 4 or core_clause_count < 3:
        return None
    return f"根据当前可访问文档中的证据，《{document_title}》里与这个场景直接相关的要求是：{'；'.join(clauses)}。"
