from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.models.chat import ChatMessage
from app.models.enums import MessageRole
from app.services.chat.prompts import format_history_line

RECENT_ROUNDS = 2
MAX_SUMMARY_LINES = 6


@dataclass
class ConversationMemory:
    history_lines: list[str]
    older_summary: str | None
    previous_target_document: str | None
    previous_tool_name: str | None
    previous_artifact_type: str | None
    previous_intent: str | None
    previous_refusal_reason: str | None
    previous_insufficient_evidence: bool
    recent_message_count: int
    older_message_count: int

    def to_router_context(self) -> str | None:
        parts: list[str] = []
        if self.older_summary:
            parts.append(f"Earlier conversation summary:\n{self.older_summary}")
        if self.history_lines:
            parts.append("Recent chat history:\n" + "\n".join(self.history_lines))
        hints = self._hint_lines()
        if hints:
            parts.append("Previous conversation hints:\n" + "\n".join(hints))
        if not parts:
            return None
        return "\n\n".join(parts)

    def to_answer_context(self) -> str | None:
        parts: list[str] = []
        if self.older_summary:
            parts.append(f"Earlier conversation summary:\n{self.older_summary}")
        if self.history_lines:
            parts.append("Recent chat history:\n" + "\n".join(self.history_lines))
        hints = self._hint_lines()
        if hints:
            parts.append("Carry-over hints:\n" + "\n".join(hints))
        if not parts:
            return None
        return "\n\n".join(parts)

    def _hint_lines(self) -> list[str]:
        hints: list[str] = []
        if self.previous_target_document:
            hints.append(f"- previous_target_document: {self.previous_target_document}")
        if self.previous_tool_name:
            hints.append(f"- previous_tool_name: {self.previous_tool_name}")
        if self.previous_artifact_type:
            hints.append(f"- previous_artifact_type: {self.previous_artifact_type}")
        if self.previous_intent:
            hints.append(f"- previous_intent: {self.previous_intent}")
        if self.previous_refusal_reason:
            hints.append(f"- previous_refusal_reason: {self.previous_refusal_reason}")
        return hints


def build_conversation_memory(
    existing_messages: Sequence[ChatMessage],
    *,
    recent_rounds: int = RECENT_ROUNDS,
) -> ConversationMemory:
    messages = list(existing_messages)
    recent_messages = _select_recent_round_messages(messages, recent_rounds)
    older_count = max(len(messages) - len(recent_messages), 0)
    older_messages = messages[:older_count] if older_count else []

    previous_target_document = None
    previous_tool_name = None
    previous_artifact_type = None
    previous_intent = None
    previous_refusal_reason = None
    previous_insufficient_evidence = False

    for message in reversed(messages):
        if message.role != MessageRole.ASSISTANT:
            continue
        metadata = message.message_metadata if isinstance(message.message_metadata, dict) else {}
        router_decision = metadata.get("router_decision") if isinstance(metadata.get("router_decision"), dict) else {}
        tool_execution = metadata.get("tool_execution") if isinstance(metadata.get("tool_execution"), dict) else {}
        structured_result = metadata.get("structured_result") if isinstance(metadata.get("structured_result"), dict) else {}

        previous_target_document = _first_string(
            structured_result.get("target_document"),
            router_decision.get("target_document_title"),
            router_decision.get("requested_document_name"),
        )
        previous_tool_name = _as_string(tool_execution.get("tool_name"))
        previous_artifact_type = _as_string(structured_result.get("artifact_type"))
        previous_intent = _as_string(router_decision.get("intent"))
        previous_refusal_reason = _as_string(structured_result.get("refusal_reason"))
        previous_insufficient_evidence = bool(message.insufficient_evidence)
        break

    history_lines = [format_history_line(message.role.value, message.content) for message in recent_messages]
    older_summary = _summarize_older_messages(older_messages)
    return ConversationMemory(
        history_lines=history_lines,
        older_summary=older_summary,
        previous_target_document=previous_target_document,
        previous_tool_name=previous_tool_name,
        previous_artifact_type=previous_artifact_type,
        previous_intent=previous_intent,
        previous_refusal_reason=previous_refusal_reason,
        previous_insufficient_evidence=previous_insufficient_evidence,
        recent_message_count=len(recent_messages),
        older_message_count=older_count,
    )


def has_usable_workflow_context(existing_messages: Sequence[ChatMessage]) -> bool:
    for message in reversed(existing_messages):
        if message.role != MessageRole.ASSISTANT:
            continue
        if message.insufficient_evidence:
            continue
        if message.content.strip():
            return True
    return False


def _select_recent_round_messages(messages: Sequence[ChatMessage], recent_rounds: int) -> list[ChatMessage]:
    if not messages or recent_rounds <= 0:
        return []

    collected: list[ChatMessage] = []
    user_turns = 0
    for message in reversed(messages):
        collected.append(message)
        if message.role == MessageRole.USER:
            user_turns += 1
            if user_turns >= recent_rounds:
                break
    return list(reversed(collected))


def _summarize_older_messages(messages: Sequence[ChatMessage]) -> str | None:
    if not messages:
        return None
    lines = [format_history_line(message.role.value, message.content) for message in messages]
    if len(lines) > MAX_SUMMARY_LINES:
        omitted = len(lines) - MAX_SUMMARY_LINES
        lines = lines[-MAX_SUMMARY_LINES:]
        lines.insert(0, f"... {omitted} earlier message(s) omitted ...")
    return "\n".join(lines)


def _as_string(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _first_string(*values: object) -> str | None:
    for value in values:
        cleaned = _as_string(value)
        if cleaned:
            return cleaned
    return None
