from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
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
    model_name = "diff-summary-fallback-v2"

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
            _compact(_format_modification(change.get("old_text"), change.get("new_text")))
            for change in diff_changes
            if change.get("change_type") == "replace"
        ]
        additions = [item for item in additions if item][:3]
        deletions = [item for item in deletions if item][:3]
        modifications = [item for item in modifications if item][:3]

        summary_parts = [
            f"对比 v{from_version_number} 与 v{to_version_number} 后，系统共识别出 {len(additions)} 处新增、{len(deletions)} 处删除、{len(modifications)} 处修改。",
        ]
        if modifications:
            summary_parts.append(f"重点变化主要体现在：{modifications[0]}。")
        elif additions:
            summary_parts.append(f"最明显的新增内容包括：{additions[0]}。")
        elif deletions:
            summary_parts.append(f"最明显的删除内容包括：{deletions[0]}。")
        else:
            summary_parts.append("当前差异中没有提取到足够明显的变更重点，建议结合原始 diff 查看具体段落。")

        return DiffSummaryResult(
            summary="".join(summary_parts),
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
        # 版本差异摘要更偏向结构化整理，优先使用 chat model 能显著降低 OpenAI-compatible
        # 提供方（尤其是 DeepSeek）在 JSON 模式和高延迟推理模型上的失败率。
        self.model_name = self.settings.effective_llm_chat_model
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
        for index, change in enumerate(diff_changes[:14], start=1):
            change_lines.append(
                json.dumps(
                    {
                        "index": index,
                        "change_type": change.get("change_type"),
                        "old_text": _compact(change.get("old_text"), 220),
                        "new_text": _compact(change.get("new_text"), 220),
                        "from_paragraph_start": change.get("from_paragraph_start"),
                        "to_paragraph_start": change.get("to_paragraph_start"),
                    },
                    ensure_ascii=False,
                )
            )
        joined_change_lines = "\n".join(change_lines)
        prompt = (
            "你是企业知识库的版本差异摘要助手。\n"
            "请只基于下面提供的结构化 diff 信息进行总结，不要假设完整文档内容。\n"
            f"当前正在比较 v{from_version_number} 和 v{to_version_number}。\n"
            "请直接返回 JSON，不要使用 Markdown 代码块，也不要添加额外解释。\n"
            "JSON 结构必须包含：summary, additions, deletions, modifications。\n"
            "要求：summary 用简洁中文概括 2 到 3 句；其余三个字段都是字符串数组，每项尽量短，最多 5 条。\n\n"
            "结构化变更：\n"
            f"{joined_change_lines}\n\n"
            "原始 diff 节选：\n"
            f"{unified_diff[:2800]}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是结构化 JSON 输出助手。只返回 JSON 对象本身。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=700,
                timeout=12.0,
            )
            content = response.choices[0].message.content or "{}"
            payload = _parse_json_payload(content)
            summary = str(payload.get("summary") or "").strip()
            additions = [_normalize_list_item(item) for item in payload.get("additions", [])][:5]
            deletions = [_normalize_list_item(item) for item in payload.get("deletions", [])][:5]
            modifications = [_normalize_list_item(item) for item in payload.get("modifications", [])][:5]
            additions = [item for item in additions if item]
            deletions = [item for item in deletions if item]
            modifications = [item for item in modifications if item]
            if not summary and not additions and not deletions and not modifications:
                raise ValueError("Empty diff summary payload from OpenAI-compatible provider.")
            return DiffSummaryResult(
                summary=_ensure_sentence(summary or f"已完成 v{from_version_number} 与 v{to_version_number} 的差异摘要。"),
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
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
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


def _format_modification(old_text: str | None, new_text: str | None) -> str:
    old_part = _compact(old_text, 90)
    new_part = _compact(new_text, 90)
    if old_part and new_part:
        return f"由“{old_part}”调整为“{new_part}”"
    return new_part or old_part


def _normalize_list_item(value: Any) -> str:
    return _ensure_sentence(str(value).strip()) if str(value).strip() else ""


def _ensure_sentence(value: str) -> str:
    if not value:
        return ""
    cleaned = value.strip().rstrip("；;，,")
    if cleaned.endswith(("。", "！", "？")):
        return cleaned
    return cleaned + "。"
