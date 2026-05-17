from __future__ import annotations

from uuid import uuid4

from app.schemas.llm import RouterAccessibleDocument, RouterDecision
from app.services.llm.router import OpenAIRouterProvider, _stabilize_router_decision


def test_unsupported_or_unclear_is_upgraded_to_topic_qa_for_general_policy_question() -> None:
    decision = RouterDecision(
        intent="unsupported_or_unclear",
        needs_citations=False,
        reasoning_brief="model unsure",
    )

    stabilized = _stabilize_router_decision(
        question="节假日安排是什么样的？",
        accessible_documents=[
            RouterAccessibleDocument(document_id=uuid4(), title="员工手册"),
            RouterAccessibleDocument(document_id=uuid4(), title="平台发布手册"),
        ],
        conversation_context=None,
        decision=decision,
    )

    assert stabilized.intent == "topic_qa"
    assert stabilized.topic == "节假日安排是什么样的？"
    assert stabilized.needs_citations is True


def test_openai_router_uses_fast_deterministic_path_for_obvious_topic_question(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.llm.router.create_openai_compatible_client",
        lambda settings: (_ for _ in ()).throw(AssertionError("llm router should not be called")),
    )

    provider = OpenAIRouterProvider()
    result = provider.route(
        question="节假日安排是什么样的？",
        accessible_documents=[RouterAccessibleDocument(document_id=uuid4(), title="员工手册")],
        conversation_context=None,
    )

    assert result.decision.intent == "topic_qa"
    assert result.provider_name == "deterministic-router"
    assert result.raw_payload
    assert result.raw_payload["fast_path_reason"] == "obvious_document_or_topic_query"
