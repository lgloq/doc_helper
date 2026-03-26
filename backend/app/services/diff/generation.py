from __future__ import annotations

import json
import re
from dataclasses import dataclass
import logging
from typing import Any, Protocol

from app.core.config import get_settings
from app.services.llm.openai_compatible import (
    create_openai_compatible_client,
    has_openai_compatible_credentials,
    uses_openai_compatible_provider,
)

logger = logging.getLogger(__name__)


@dataclass
class DiffSummaryResult:
    summary: str
    additions: list[str]
    deletions: list[str]
    modifications: list[str]
    provider_name: str
    model_name: str | None


class DiffSummaryGenerator(Protocol):
    def generate(
        self,
        *,
        from_version_number: int,
        to_version_number: int,
        diff_changes: list[dict[str, Any]],
        unified_diff: str,
    ) -> DiffSummaryResult: ...


class DeterministicDiffSummaryGenerator:
    provider_name = "deterministic"
    model_name = "diff-summary-fallback-v1"

    def generate(
        self,
        *,
        from_version_number: int,
        to_version_number: int,
        diff_changes: list[dict[str, Any]],
        unified_diff: str,
    ) -> DiffSummaryResult:
        additions = [_compact(change.get("new_text")) for change in diff_changes if change.get("change_type") == "insert"]
        deletions = [_compact(change.get("old_text")) for change in diff_changes if change.get("change_type") == "delete"]
        modifications = [
            _compact(f"From: {change.get('old_text') or ''} To: {change.get('new_text') or ''}")
            for change in diff_changes
            if change.get("change_type") == "replace"
        ]
        additions = [item for item in additions if item][:3]
        deletions = [item for item in deletions if item][:3]
        modifications = [item for item in modifications if item][:3]
        summary_parts = [
            f"Compared version {from_version_number} to version {to_version_number}.",
            f"Detected {len(additions)} highlighted additions, {len(deletions)} highlighted deletions, and {len(modifications)} highlighted modifications.",
        ]
        if additions:
            summary_parts.append(f"Key additions include: {additions[0]}")
        if deletions:
            summary_parts.append(f"Key deletions include: {deletions[0]}")
        if modifications:
            summary_parts.append(f"Key modifications include: {modifications[0]}")
        return DiffSummaryResult(
            summary=" ".join(summary_parts),
            additions=additions,
            deletions=deletions,
            modifications=modifications,
            provider_name=self.provider_name,
            model_name=self.model_name,
        )


class OpenAIDiffSummaryGenerator:
    provider_name = "openai-compatible"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.model_name = self.settings.effective_llm_reasoning_model
        self.client = create_openai_compatible_client(self.settings)
        self.fallback = DeterministicDiffSummaryGenerator()

    def generate(
        self,
        *,
        from_version_number: int,
        to_version_number: int,
        diff_changes: list[dict[str, Any]],
        unified_diff: str,
    ) -> DiffSummaryResult:
        change_lines = []
        for index, change in enumerate(diff_changes[:30], start=1):
            change_lines.append(
                json.dumps(
                    {
                        "index": index,
                        "change_type": change.get("change_type"),
                        "old_text": _compact(change.get("old_text"), 300),
                        "new_text": _compact(change.get("new_text"), 300),
                        "from_paragraph_start": change.get("from_paragraph_start"),
                        "to_paragraph_start": change.get("to_paragraph_start"),
                    },
                    ensure_ascii=False,
                )
            )
        joined_change_lines = "\n".join(change_lines)
        prompt = (
            "You summarize version diffs for enterprise documents.\n"
            "Use only the diff data below. Do not infer from the full document.\n"
            f"Comparing version {from_version_number} to version {to_version_number}.\n"
            "Return JSON only with keys: summary, additions, deletions, modifications.\n"
            "Each additions/deletions/modifications field must be an array of short strings.\n\n"
            "Structured changes:\n"
            f"{joined_change_lines}\n\n"
            "Unified diff excerpt:\n"
            f"{unified_diff[:6000]}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Return valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
                timeout=20.0,
            )
            content = response.choices[0].message.content or "{}"
            payload = _parse_json_payload(content)
            summary = str(payload.get("summary") or "").strip()
            additions = [str(item).strip() for item in payload.get("additions", []) if str(item).strip()][:5]
            deletions = [str(item).strip() for item in payload.get("deletions", []) if str(item).strip()][:5]
            modifications = [str(item).strip() for item in payload.get("modifications", []) if str(item).strip()][:5]
            if not summary and not additions and not deletions and not modifications:
                raise ValueError("Empty diff summary payload from OpenAI-compatible provider.")
            return DiffSummaryResult(
                summary=summary or f"Compared version {from_version_number} to version {to_version_number}.",
                additions=additions,
                deletions=deletions,
                modifications=modifications,
                provider_name=self.provider_name,
                model_name=self.model_name,
            )
        except Exception as error:
            logger.warning("OpenAI-compatible diff summary failed; falling back to deterministic summary.", exc_info=error)
            fallback_result = self.fallback.generate(
                from_version_number=from_version_number,
                to_version_number=to_version_number,
                diff_changes=diff_changes,
                unified_diff=unified_diff,
            )
            return DiffSummaryResult(
                summary=fallback_result.summary,
                additions=fallback_result.additions,
                deletions=fallback_result.deletions,
                modifications=fallback_result.modifications,
                provider_name="deterministic_fallback",
                model_name=fallback_result.model_name,
            )


class DiffSummaryGeneratorFactory:
    @staticmethod
    def create() -> DiffSummaryGenerator:
        settings = get_settings()
        if uses_openai_compatible_provider(settings.diff_summary_provider) and has_openai_compatible_credentials(settings):
            return OpenAIDiffSummaryGenerator()
        return DeterministicDiffSummaryGenerator()



def _parse_json_payload(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}



def _compact(value: str | None, limit: int = 220) -> str:
    if not value:
        return ""
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."

