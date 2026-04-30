from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from app.core.config import get_settings
from app.schemas.llm import RouterAccessibleDocument, RouterDecision, RouterDecisionResult
from app.services.chat.memory import ConversationMemory
from app.services.llm.openai_compatible import (
    create_openai_compatible_client,
    has_openai_compatible_credentials,
    uses_openai_compatible_provider,
)

ROUTER_SYSTEM_PROMPT = """You are the routing layer for an enterprise knowledge copilot.
Your job is to classify the user's request and select the best tool path.
You must not answer the user question itself.
Return valid JSON only.

Rules:
- intent must be one of: document_qa, topic_qa, version_compare, workflow_generation, unsupported_or_unclear.
- If the user clearly asks about a specific document, use document_qa.
- When a specific document is requested, choose target_document_title and target_document_id only from the accessible document list provided below.
- If the user appears to reference a specific document that is not in the accessible list, keep requested_document_name, leave target_document_title and target_document_id null, and set should_refuse_if_inaccessible=true.
- If the user asks to compare versions or asks what changed between versions, use version_compare.
- Extract version refs such as v1, v2, latest, newest, current, previous, 最新, 上一版, 前一版 when present.
- artifact_type may be set together with document_qa, topic_qa, or version_compare when the user wants to search/compare first and then generate tasks, a weekly report, or FAQ.
- Use workflow_generation only when the user is mainly asking to generate tasks, a weekly report, or FAQ from existing conversation context.
- If the request is vague or unsupported, use unsupported_or_unclear.
- needs_citations should usually be true for document_qa and topic_qa, false otherwise.
"""

DOCUMENT_HINT_TOKENS = ("手册", "指南", "登记", "文档", "document", "doc", "guide", "handbook", "runbook", "policy")
UNSUPPORTED_PATTERNS = re.compile(r"^[\s\W_。？?！!…]+$")
VERSION_COMPARE_PATTERNS = (
    "比较",
    "对比",
    "差异",
    "改了什么",
    "compare",
    "diff",
    "changed",
)
WORKFLOW_PATTERNS = {
    "tasks": ("待办", "任务", "待办事项", "整理成待办", "todo", "action item"),
    "weekly_report": ("周报", "weekly report", "status report"),
    "faq": ("faq", "常见问题"),
}


@dataclass
class _ResolvedAccessibleDocument:
    document_id: UUID
    title: str


class RouterProvider(Protocol):
    def route(
        self,
        *,
        question: str,
        accessible_documents: list[RouterAccessibleDocument],
        conversation_context: ConversationMemory | None = None,
    ) -> RouterDecisionResult: ...


class DeterministicRouterProvider:
    provider_name = "deterministic-router"
    model_name = "router-fallback-v1"

    def route(
        self,
        *,
        question: str,
        accessible_documents: list[RouterAccessibleDocument],
        conversation_context: ConversationMemory | None = None,
    ) -> RouterDecisionResult:
        started = time.perf_counter()
        lowered = question.strip().lower()
        decision = _stabilize_router_decision(
            question=question,
            accessible_documents=accessible_documents,
            conversation_context=conversation_context,
            decision=self._route(question, accessible_documents, conversation_context),
        )
        return RouterDecisionResult(
            decision=decision,
            provider_name=self.provider_name,
            model_name=self.model_name,
            prompt_tokens=max(len(question) // 4, 1),
            completion_tokens=max(len(decision.reasoning_brief) // 4, 1),
            latency_ms=int((time.perf_counter() - started) * 1000),
            raw_payload={
                "lowered_question": lowered,
                "conversation_context": conversation_context.to_router_context() if conversation_context else None,
            },
        )

    def _route(
        self,
        question: str,
        accessible_documents: list[RouterAccessibleDocument],
        conversation_context: ConversationMemory | None,
    ) -> RouterDecision:
        lowered = question.strip().lower()
        stripped = question.strip()
        if not lowered or UNSUPPORTED_PATTERNS.match(stripped) or len(re.sub(r"[\\s？?！!。,.…]", "", stripped)) <= 2:
            return RouterDecision(
                intent="unsupported_or_unclear",
                needs_citations=False,
                reasoning_brief="问题过短或缺少明确上下文，无法可靠归类。",
            )

        artifact_type = _infer_artifact_type(lowered)

        if any(pattern in lowered for pattern in VERSION_COMPARE_PATTERNS):
            matched_document = _match_accessible_document(question, accessible_documents) or _match_context_document(
                question,
                accessible_documents,
                conversation_context,
            )
            return RouterDecision(
                intent="version_compare",
                target_document_id=matched_document.document_id if matched_document else None,
                target_document_title=matched_document.title if matched_document else None,
                requested_document_name=_resolve_requested_document_name(question, conversation_context),
                artifact_type=artifact_type,  # type: ignore[arg-type]
                from_version_ref=_extract_version_ref(question, prefer="from"),
                to_version_ref=_extract_version_ref(question, prefer="to"),
                needs_citations=False,
                should_refuse_if_inaccessible=matched_document is None and _looks_like_document_request(question, conversation_context),
                reasoning_brief="问题询问版本变化或差异，适合进入版本对比工具。",
            )

        if artifact_type and _looks_like_context_workflow_request(question, conversation_context):
            return RouterDecision(
                intent="workflow_generation",
                artifact_type=artifact_type,  # type: ignore[arg-type]
                needs_citations=False,
                reasoning_brief="用户希望基于已有会话上下文生成结构化产物。",
            )

        matched_document = _match_accessible_document(question, accessible_documents) or _match_context_document(
            question,
            accessible_documents,
            conversation_context,
        )
        if matched_document:
            return RouterDecision(
                intent="document_qa",
                target_document_id=matched_document.document_id,
                target_document_title=matched_document.title,
                requested_document_name=matched_document.title,
                artifact_type=artifact_type,  # type: ignore[arg-type]
                needs_citations=True,
                should_refuse_if_inaccessible=True,
                reasoning_brief="问题明确指向一个当前可访问的具体文档。",
            )

        requested_document_name = _resolve_requested_document_name(question, conversation_context)
        if requested_document_name:
            return RouterDecision(
                intent="document_qa",
                requested_document_name=requested_document_name,
                artifact_type=artifact_type,  # type: ignore[arg-type]
                needs_citations=True,
                should_refuse_if_inaccessible=True,
                reasoning_brief="问题看起来在询问特定文档，但该文档不在当前可访问范围内。",
            )

        return RouterDecision(
            intent="topic_qa",
            topic=question.strip(),
            artifact_type=artifact_type,  # type: ignore[arg-type]
            needs_citations=True,
            reasoning_brief="问题围绕一个主题展开，必要时可先检索再生成结构化结果。",
        )


class OpenAIRouterProvider:
    provider_name = "openai-compatible-router"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.model_name = self.settings.effective_llm_router_model

    def route(
        self,
        *,
        question: str,
        accessible_documents: list[RouterAccessibleDocument],
        conversation_context: ConversationMemory | None = None,
    ) -> RouterDecisionResult:
        started = time.perf_counter()
        client = create_openai_compatible_client(self.settings)
        response = client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": self._build_user_prompt(question, accessible_documents, conversation_context),
                },
            ],
        )
        content = response.choices[0].message.content or "{}"
        raw_payload = _parse_json_payload(content)
        decision = _stabilize_router_decision(
            question=question,
            accessible_documents=accessible_documents,
            conversation_context=conversation_context,
            decision=_coerce_router_decision(raw_payload, accessible_documents),
        )
        usage = getattr(response, "usage", None)
        return RouterDecisionResult(
            decision=decision,
            provider_name=self.provider_name,
            model_name=self.model_name,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            latency_ms=int((time.perf_counter() - started) * 1000),
            raw_payload=raw_payload,
        )

    @staticmethod
    def _build_user_prompt(
        question: str,
        accessible_documents: list[RouterAccessibleDocument],
        conversation_context: ConversationMemory | None,
    ) -> str:
        accessible_json = json.dumps(
            [
                {"document_id": str(item.document_id), "title": item.title}
                for item in accessible_documents
            ],
            ensure_ascii=False,
        )
        context_block = conversation_context.to_router_context() if conversation_context else "No prior conversation context."
        return (
            "User question:\n"
            f"{question}\n\n"
            "Conversation context:\n"
            f"{context_block}\n\n"
            "Currently accessible documents (choose target_document_id/title only from this list when applicable):\n"
            f"{accessible_json}\n\n"
            "Return JSON with keys: intent, target_document_id, target_document_title, requested_document_name, topic, artifact_type, from_version_ref, to_version_ref, needs_citations, should_refuse_if_inaccessible, reasoning_brief."
        )


class LLMRouterService:
    def __init__(self) -> None:
        self.provider = RouterProviderFactory.create()

    def route(
        self,
        *,
        question: str,
        accessible_documents: list[RouterAccessibleDocument],
        conversation_context: ConversationMemory | None = None,
    ) -> RouterDecisionResult:
        return self.provider.route(
            question=question,
            accessible_documents=accessible_documents,
            conversation_context=conversation_context,
        )


class RouterProviderFactory:
    @staticmethod
    def create() -> RouterProvider:
        settings = get_settings()
        if uses_openai_compatible_provider(settings.router_provider) and has_openai_compatible_credentials(settings):
            return OpenAIRouterProvider()
        return DeterministicRouterProvider()



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



def _coerce_router_decision(payload: dict[str, Any], accessible_documents: list[RouterAccessibleDocument]) -> RouterDecision:
    accessible_by_id = {str(item.document_id): item for item in accessible_documents}
    accessible_by_title = {item.title.casefold(): item for item in accessible_documents}

    target_document_id = payload.get("target_document_id")
    target_document_title = payload.get("target_document_title")
    resolved_document = None
    if target_document_id and str(target_document_id) in accessible_by_id:
        resolved_document = accessible_by_id[str(target_document_id)]
    elif target_document_title and str(target_document_title).casefold() in accessible_by_title:
        resolved_document = accessible_by_title[str(target_document_title).casefold()]

    try:
        intent = str(payload.get("intent") or "unsupported_or_unclear")
    except Exception:
        intent = "unsupported_or_unclear"
    if intent not in {"document_qa", "topic_qa", "version_compare", "workflow_generation", "unsupported_or_unclear"}:
        intent = "unsupported_or_unclear"

    artifact_type = payload.get("artifact_type")
    if artifact_type not in {"tasks", "weekly_report", "faq", None}:
        artifact_type = None

    return RouterDecision(
        intent=intent,  # type: ignore[arg-type]
        target_document_id=resolved_document.document_id if resolved_document else None,
        target_document_title=resolved_document.title if resolved_document else None,
        requested_document_name=_string_or_none(payload.get("requested_document_name")),
        topic=_string_or_none(payload.get("topic")),
        artifact_type=artifact_type,
        from_version_ref=_string_or_none(payload.get("from_version_ref")),
        to_version_ref=_string_or_none(payload.get("to_version_ref")),
        needs_citations=bool(payload.get("needs_citations", intent in {"document_qa", "topic_qa"})),
        should_refuse_if_inaccessible=bool(payload.get("should_refuse_if_inaccessible", False)),
        reasoning_brief=_string_or_none(payload.get("reasoning_brief")) or "Router completed.",
    )


def _stabilize_router_decision(
    *,
    question: str,
    accessible_documents: list[RouterAccessibleDocument],
    conversation_context: ConversationMemory | None,
    decision: RouterDecision,
) -> RouterDecision:
    matched_document = _match_accessible_document(question, accessible_documents) or _match_context_document(
        question,
        accessible_documents,
        conversation_context,
    )
    requested_document_name = _resolve_requested_document_name(question, conversation_context)
    has_followup_reference = _looks_like_followup_document_reference(question)
    has_explicit_document_anchor = matched_document is not None or requested_document_name is not None or has_followup_reference

    if decision.intent == "topic_qa" and matched_document and _looks_like_document_request(question, conversation_context):
        return decision.model_copy(
            update={
                "intent": "document_qa",
                "target_document_id": matched_document.document_id,
                "target_document_title": matched_document.title,
                "requested_document_name": requested_document_name or matched_document.title,
                "topic": None,
                "needs_citations": True,
                "should_refuse_if_inaccessible": True,
                "reasoning_brief": "问题明确指向当前可访问的具体文档，按文档问答处理。",
            }
        )

    if decision.intent == "document_qa":
        if matched_document:
            return decision.model_copy(
                update={
                    "target_document_id": matched_document.document_id,
                    "target_document_title": matched_document.title,
                    "requested_document_name": requested_document_name or matched_document.title,
                    "needs_citations": True,
                    "should_refuse_if_inaccessible": True,
                }
            )
        if requested_document_name and not decision.target_document_title:
            return decision.model_copy(
                update={
                    "requested_document_name": requested_document_name,
                    "needs_citations": True,
                    "should_refuse_if_inaccessible": True,
                }
            )
        if decision.target_document_title and not has_explicit_document_anchor:
            return decision.model_copy(
                update={
                    "intent": "topic_qa",
                    "target_document_id": None,
                    "target_document_title": None,
                    "requested_document_name": None,
                    "topic": question.strip(),
                    "needs_citations": True,
                    "should_refuse_if_inaccessible": False,
                    "reasoning_brief": "问题未明确指定文档，按主题问答处理，避免提前锁定单一文档。",
                }
            )

    if decision.intent == "version_compare" and matched_document and not decision.target_document_title:
        return decision.model_copy(
            update={
                "target_document_id": matched_document.document_id,
                "target_document_title": matched_document.title,
                "requested_document_name": requested_document_name or matched_document.title,
            }
        )

    return decision



def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None



def _infer_artifact_type(lowered_question: str) -> str | None:
    task_patterns = WORKFLOW_PATTERNS["tasks"]
    if any(pattern in lowered_question for pattern in task_patterns):
        return "tasks"
    if "整理" in lowered_question and "事项" in lowered_question:
        return "tasks"
    for artifact_type in ("weekly_report", "faq"):
        patterns = WORKFLOW_PATTERNS[artifact_type]
        if any(pattern in lowered_question for pattern in patterns):
            return artifact_type
    return None



def _match_accessible_document(question: str, accessible_documents: list[RouterAccessibleDocument]) -> _ResolvedAccessibleDocument | None:
    lowered_question = question.casefold()
    normalized_question = _normalize_text(question)
    scored_matches: list[tuple[float, RouterAccessibleDocument]] = []
    for document in accessible_documents:
        title_cf = document.title.casefold()
        normalized_title = _normalize_text(document.title)
        score = 0.0
        if title_cf in lowered_question:
            score = 1.0
        elif normalized_title and normalized_title in normalized_question:
            score = 0.92
        else:
            stripped_title = _strip_document_suffixes(normalized_title)
            if stripped_title and stripped_title in normalized_question:
                score = 0.8
        if score > 0:
            scored_matches.append((score, document))
    if not scored_matches:
        return None
    scored_matches.sort(key=lambda item: item[0], reverse=True)
    best = scored_matches[0][1]
    return _ResolvedAccessibleDocument(document_id=best.document_id, title=best.title)


def _match_context_document(
    question: str,
    accessible_documents: list[RouterAccessibleDocument],
    conversation_context: ConversationMemory | None,
) -> _ResolvedAccessibleDocument | None:
    if conversation_context is None or not conversation_context.previous_target_document:
        return None
    if not _looks_like_followup_document_reference(question):
        return None
    return _resolve_document_title(conversation_context.previous_target_document, accessible_documents)



def _extract_requested_document_name(question: str) -> str | None:
    patterns = [
        r"《(?P<name>[^》]{1,60})》",
        r"(?P<name>[\u4e00-\u9fa5A-Za-z0-9《》\-\s]{2,40}?)(?:里|中)关于",
        r"(?P<name>[\u4e00-\u9fa5A-Za-z0-9《》\-\s]{2,40}?)(?:里|中)(?:[^？?。！!]{0,24})?(?:怎么说|写了什么|说了什么|讲了什么|要求什么|提到什么|内容是什么|写了哪些|是什么|是谁)",
        r"(?:what does|compare|show|summarize)\s+(?P<name>[A-Za-z0-9\-\s]{2,60}?)\s+(?:document|doc|guide|handbook|runbook|say|show)",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            candidate = match.group("name").strip(" 《》")
            if candidate:
                return candidate
    return None


def _resolve_requested_document_name(question: str, conversation_context: ConversationMemory | None) -> str | None:
    extracted = _extract_requested_document_name(question)
    if extracted:
        return extracted
    if conversation_context and conversation_context.previous_target_document and _looks_like_followup_document_reference(question):
        return conversation_context.previous_target_document
    return None



def _extract_version_ref(question: str, *, prefer: str) -> str | None:
    matches = re.findall(r"v\s*(\d+)", question, flags=re.IGNORECASE)
    if prefer == "from":
        if len(matches) >= 1:
            return f"v{matches[0]}"
        if re.search(r"上一版|前一版|previous|prior", question, flags=re.IGNORECASE):
            return "previous"
        return None
    if len(matches) >= 2:
        return f"v{matches[1]}"
    if len(matches) == 1:
        return "latest"
    if re.search(r"最新|latest|newest|current", question, flags=re.IGNORECASE):
        return "latest"
    return None



def _looks_like_document_request(question: str, conversation_context: ConversationMemory | None = None) -> bool:
    lowered = question.casefold()
    if any(token in lowered for token in DOCUMENT_HINT_TOKENS):
        return True
    if conversation_context and conversation_context.previous_target_document and _looks_like_followup_document_reference(question):
        return True
    return bool(re.search(r"里.*(怎么说|写了什么|要求什么|改了什么)", question))


def _looks_like_followup_document_reference(question: str) -> bool:
    lowered = question.casefold()
    markers = (
        "这个文档",
        "这份文档",
        "刚才那个文档",
        "刚才那份文档",
        "上一份文档",
        "上一个文档",
        "那个文档",
        "那份文档",
        "这个手册",
        "这份手册",
        "这个指南",
        "这份指南",
        "那个手册",
        "那份手册",
    )
    return any(marker in lowered for marker in markers)


def _looks_like_context_workflow_request(question: str, conversation_context: ConversationMemory | None) -> bool:
    lowered = question.casefold()
    direct_context_markers = (
        "刚才",
        "上面",
        "上一轮",
        "刚刚",
        "当前会话",
        "这次会话",
        "根据刚才",
        "把刚才",
        "根据上文",
        "把上文",
    )
    if any(marker in lowered for marker in direct_context_markers):
        return True
    if conversation_context and conversation_context.previous_tool_name and any(
        marker in lowered for marker in ("整理一份", "生成", "提取", "沉淀成")
    ):
        return True
    return False



def _normalize_text(text: str) -> str:
    lowered = text.casefold()
    lowered = re.sub(r"[《》\-_/,.?？!！:：;；'\"\(\)\[\]]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered



def _strip_document_suffixes(text: str) -> str:
    stripped = text
    for suffix in (" document", " doc", " handbook", " guide", " runbook", " policy", " 文档", " 手册", " 指南", " 登记"):
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)].strip()
    return stripped


def _resolve_document_title(title: str, accessible_documents: list[RouterAccessibleDocument]) -> _ResolvedAccessibleDocument | None:
    lowered = title.casefold()
    normalized = _normalize_text(title)
    stripped = _strip_document_suffixes(normalized)
    for document in accessible_documents:
        document_title = document.title.casefold()
        document_normalized = _normalize_text(document.title)
        document_stripped = _strip_document_suffixes(document_normalized)
        if lowered == document_title or normalized == document_normalized or (stripped and stripped == document_stripped):
            return _ResolvedAccessibleDocument(document_id=document.document_id, title=document.title)
    return None



