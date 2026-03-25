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
    ) -> AnswerGenerationResult:
        started = time.perf_counter()
        if not retrieved_chunks:
            answer = (
                "I could not find enough evidence in the documents you can access to support a reliable answer. "
                "Please refine the question or confirm that the relevant document has been uploaded and shared."
            )
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

        selected_chunks = [chunk for chunk in retrieved_chunks[:3] if chunk.score.fused > 0.0]
        top_score = selected_chunks[0].score.fused if selected_chunks else 0.0
        evidence_conflict = _detect_evidence_conflict(selected_chunks)
        if top_score < 0.2:
            answer = (
                "The retrieved evidence is too weak to support a confident answer. "
                "Please narrow the question or inspect the cited chunks directly."
            )
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
            preview = " ".join(chunk.preview.split())
            synthesized_points.append(f"{chunk.document_title} (v{chunk.version_number}, {location}) states: {preview}")

        answer = "Based on the retrieved evidence, " + " ".join(synthesized_points)
        if evidence_conflict:
            answer += " The top evidence comes from multiple competing sources, so treat this as provisional."

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
            raw_payload={"history_lines": history_lines},
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
            "answer": "I could not produce a grounded answer because the model response was not valid JSON.",
            "insufficient_evidence": True,
            "evidence_conflict": False,
            "used_chunk_ids": [],
            "answer_basis": "invalid_model_payload",
        }
    if not isinstance(loaded, dict):
        return {
            "answer": "I could not produce a grounded answer because the model response shape was invalid.",
            "insufficient_evidence": True,
            "evidence_conflict": False,
            "used_chunk_ids": [],
            "answer_basis": "invalid_model_payload",
        }
    return loaded



def _format_location(chunk: SearchResultChunk) -> str:
    if chunk.page_number_start is not None:
        return f"page {chunk.page_number_start}"
    if chunk.paragraph_start is not None:
        return f"paragraph {chunk.paragraph_start}"
    return f"chunk {chunk.chunk_index}"



def _detect_evidence_conflict(chunks: list[SearchResultChunk]) -> bool:
    if len(chunks) < 2:
        return False
    first = chunks[0]
    second = chunks[1]
    return first.document_id != second.document_id and abs(first.score.fused - second.score.fused) <= 0.1
