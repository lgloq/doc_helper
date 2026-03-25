from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from app.core.config import get_settings


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
    provider_name = "openai"

    def __init__(self) -> None:
        from openai import OpenAI

        self.settings = get_settings()
        self.model_name = self.settings.openai_diff_model
        self.client = OpenAI(api_key=self.settings.openai_api_key)

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
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        payload = _parse_json_payload(content)
        return DiffSummaryResult(
            summary=str(payload.get("summary") or "").strip() or f"Compared version {from_version_number} to version {to_version_number}.",
            additions=[str(item).strip() for item in payload.get("additions", []) if str(item).strip()][:5],
            deletions=[str(item).strip() for item in payload.get("deletions", []) if str(item).strip()][:5],
            modifications=[str(item).strip() for item in payload.get("modifications", []) if str(item).strip()][:5],
            provider_name=self.provider_name,
            model_name=self.model_name,
        )


class DiffSummaryGeneratorFactory:
    @staticmethod
    def create() -> DiffSummaryGenerator:
        settings = get_settings()
        if settings.diff_summary_provider.lower() == "openai" and settings.openai_api_key:
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
