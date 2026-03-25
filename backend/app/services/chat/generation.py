from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from app.core.config import get_settings
from app.schemas.search import SearchResultChunk
from app.services.chat.prompts import build_grounded_messages


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
        allow_low_score: bool = False,
    ) -> AnswerGenerationResult: ...


class DeterministicAnswerGenerator:
    provider_name = "deterministic"
    model_name = "grounded-fallback-v1"

    def generate(
        self,
        *,
        question: str,
        retrieved_chunks: list[SearchResultChunk],
        history_lines: list[str],
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

        top_score = selected_chunks[0].score.fused if selected_chunks else 0.0
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

        synthesized_points = []
        for chunk in selected_chunks[:2]:
            location = _format_location(chunk)
            preview = " ".join((chunk.preview or chunk.content).split())
            synthesized_points.append(f"{chunk.document_title}（v{chunk.version_number}，{location}）提到：{preview}")

        answer = "根据当前可访问文档中的证据，" + "；".join(synthesized_points)
        if evidence_conflict and not allow_low_score:
            answer += " 不过，前两条高分证据来自不同文档，请结合引用来源谨慎判断。"

        return AnswerGenerationResult(
            answer=answer,
            insufficient_evidence=False,
            evidence_conflict=evidence_conflict,
            used_chunk_ids=[str(chunk.chunk_id) for chunk in selected_chunks[:2]],
            answer_basis="extractive_summary",
            provider_name=self.provider_name,
            model_name=self.model_name,
            prompt_tokens=max((len(question) + sum(len(chunk.content) for chunk in selected_chunks)) // 4, 1),
            completion_tokens=max(len(answer) // 4, 1),
            latency_ms=int((time.perf_counter() - started) * 1000),
            raw_payload={"history_lines": history_lines, "allow_low_score": allow_low_score},
        )


class OpenAIAnswerGenerator:
    provider_name = "openai"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.model_name = self.settings.openai_chat_model

    def generate(
        self,
        *,
        question: str,
        retrieved_chunks: list[SearchResultChunk],
        history_lines: list[str],
        allow_low_score: bool = False,
    ) -> AnswerGenerationResult:
        from openai import OpenAI

        started = time.perf_counter()
        client = OpenAI(api_key=self.settings.openai_api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=build_grounded_messages(question, retrieved_chunks, history_lines),
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
        if settings.answer_provider.lower() == "openai" and settings.openai_api_key:
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
        chunk for chunk in retrieved_chunks if chunk.document_id == top_document_id and (chunk.score.fused > 0.0 or chunk.score.lexical_raw > 0.0)
    ]
    if same_document_chunks:
        return same_document_chunks[:2]
    if allow_low_score:
        fallback_chunks = [chunk for chunk in retrieved_chunks if chunk.document_id == top_document_id]
        if fallback_chunks:
            return fallback_chunks[:2]
    return [chunk for chunk in retrieved_chunks[:2] if chunk.score.fused > 0.0 or chunk.score.lexical_raw > 0.0]



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
