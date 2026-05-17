from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from app.schemas.search import SearchResultChunk

SYSTEM_PROMPT = """You are an enterprise knowledge assistant.
Answer only from the provided evidence snippets.
Do not invent documents, versions, policies, dates, owners, or citations.
If the evidence is missing, weak, or conflicting, say so plainly and set insufficient_evidence to true.
When evidence contains "Table row:" lines, treat each row as structured evidence and preserve relevant fields such as 审批人, 处理时限, 脱敏要求, 检查项, 是否必须, 负责人, and 完成时限.
Prefer concise answers in the same language as the user's question.
Use only chunk ids that appear in the evidence list.
Return valid JSON only with keys: answer, insufficient_evidence, evidence_conflict, used_chunk_ids, answer_basis.
"""


def build_grounded_messages(
    question: str,
    retrieved_chunks: Sequence[SearchResultChunk],
    history_lines: Sequence[str],
    context_summary: str | None = None,
) -> list[dict[str, str]]:
    history_block = "\n".join(history_lines) if history_lines else "No prior chat history."
    context_block = context_summary or "No earlier conversation context."
    evidence_block = "\n\n".join(_format_chunk(chunk) for chunk in retrieved_chunks)
    user_prompt = (
        "User question:\n"
        f"{question}\n\n"
        "Conversation context:\n"
        f"{context_block}\n\n"
        "Recent chat history:\n"
        f"{history_block}\n\n"
        "Evidence snippets:\n"
        f"{evidence_block}\n\n"
        "Remember: answer only from the evidence above. If the evidence does not support a conclusion,"
        " return insufficient_evidence=true and explain the gap briefly."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]



def _format_chunk(chunk: SearchResultChunk) -> str:
    location_parts: list[str] = [f"chunk_index={chunk.chunk_index}"]
    if chunk.page_number_start is not None:
        if chunk.page_number_end and chunk.page_number_end != chunk.page_number_start:
            location_parts.append(f"pages={chunk.page_number_start}-{chunk.page_number_end}")
        else:
            location_parts.append(f"page={chunk.page_number_start}")
    if chunk.paragraph_start is not None:
        if chunk.paragraph_end and chunk.paragraph_end != chunk.paragraph_start:
            location_parts.append(f"paragraphs={chunk.paragraph_start}-{chunk.paragraph_end}")
        else:
            location_parts.append(f"paragraph={chunk.paragraph_start}")
    location = ", ".join(location_parts)
    section = chunk.section_title or "n/a"
    return (
        f"Chunk ID: {chunk.chunk_id}\n"
        f"Document: {chunk.document_title}\n"
        f"Version: {chunk.version_number}\n"
        f"Section: {section}\n"
        f"Location: {location}\n"
        f"Excerpt:\n{chunk.content}"
    )



def truncate_session_title(question: str, limit: int = 80) -> str:
    compact = " ".join(question.strip().split())
    if len(compact) <= limit:
        return compact or "新会话"
    return compact[: limit - 3].rstrip() + "..."



def format_history_line(role: str, content: str) -> str:
    compact = " ".join(content.strip().split())
    return f"{role}: {compact[:400]}"



def validate_used_chunk_ids(used_chunk_ids: Sequence[str], available_chunk_ids: set[str]) -> list[UUID]:
    validated: list[UUID] = []
    seen: set[str] = set()
    for chunk_id in used_chunk_ids:
        if chunk_id in seen or chunk_id not in available_chunk_ids:
            continue
        try:
            validated_uuid = UUID(chunk_id)
        except ValueError:
            continue
        seen.add(chunk_id)
        validated.append(validated_uuid)
    return validated
