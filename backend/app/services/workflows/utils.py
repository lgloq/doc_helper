from __future__ import annotations

import re
from collections.abc import Iterable

from app.models.chat import ChatMessage
from app.models.enums import MessageRole

ACTION_PATTERNS = [
    r"\bnotify\b",
    r"\bdocument\b",
    r"\breview\b",
    r"\bprepare\b",
    r"\bupdate\b",
    r"\bcheck\b",
    r"\bverify\b",
    r"\bsend\b",
    r"\balign\b",
    r"\bfix\b",
    r"\bmonitor\b",
    r"\bdraft\b",
    r"\bcomplete\b",
    r"\btriage\b",
    r"\bfollow up\b",
    r"\bcommunicate\b",
    r"\bconfirm\b",
    r"\bmust\b",
    r"\bshould\b",
    r"\bneed to\b",
]

HIGH_PRIORITY_PATTERNS = [r"urgent", r"critical", r"security", r"immediately", r"deadline", r"blocker", r"sla"]
MEDIUM_PRIORITY_PATTERNS = [r"must", r"should", r"review", r"notify", r"prepare", r"update", r"verify"]
RISK_PATTERNS = [r"risk", r"blocker", r"delay", r"issue", r"failure", r"missing", r"conflict", r"uncertain"]



def split_into_sentences(text: str) -> list[str]:
    fragments = re.split(r"(?<=[\.!?。；;])\s+|\n+", text)
    return [fragment.strip(" -\t") for fragment in fragments if fragment and fragment.strip(" -\t")]



def is_actionable_sentence(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in ACTION_PATTERNS)



def infer_priority(text: str) -> str:
    lowered = text.lower()
    if any(re.search(pattern, lowered) for pattern in HIGH_PRIORITY_PATTERNS):
        return "high"
    if any(re.search(pattern, lowered) for pattern in MEDIUM_PRIORITY_PATTERNS):
        return "medium"
    return "low"



def looks_like_risk(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in RISK_PATTERNS)



def compact_text(text: str, limit: int = 220) -> str:
    compact = " ".join(text.strip().split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."



def normalize_title(text: str, limit: int = 100) -> str:
    compact = compact_text(text, limit=limit)
    compact = re.sub(r"^(must|should|need to)\s+", "", compact, flags=re.IGNORECASE)
    return compact[:1].upper() + compact[1:] if compact else "Follow up"



def iter_question_answer_pairs(messages: list[ChatMessage]) -> list[tuple[ChatMessage, ChatMessage]]:
    pairs: list[tuple[ChatMessage, ChatMessage]] = []
    pending_user: ChatMessage | None = None
    for message in messages:
        if message.role == MessageRole.USER:
            pending_user = message
            continue
        if message.role == MessageRole.ASSISTANT and pending_user is not None:
            pairs.append((pending_user, message))
            pending_user = None
    return pairs



def unique_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered
