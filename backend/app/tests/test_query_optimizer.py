from __future__ import annotations

from app.core.config import Settings
from app.services.retrieval.query_optimizer import QueryOptimizer
from app.services.retrieval.query_optimizer import QueryRewriteSuggestion


def test_query_optimizer_builds_title_anchored_retrieval_query() -> None:
    optimizer = QueryOptimizer()

    plan = optimizer.build("《平台发布手册》里面提到，发布工单至少写明哪些信息？")

    assert plan.rewrite_applied is True
    assert "title_anchor" in plan.applied_strategies
    assert "focus_keywords" in plan.applied_strategies
    assert plan.retrieval_query.startswith("平台发布手册")
    assert len(plan.lexical_queries) >= 2
    assert plan.candidate_count >= 2
    assert any("title_anchor" in candidate.applied_strategies for candidate in plan.candidates)
    assert any(item.startswith("平台发布手册") for item in plan.lexical_queries)


def test_query_optimizer_uses_explicit_target_document_title() -> None:
    optimizer = QueryOptimizer()

    plan = optimizer.build(
        "高权限令牌传播和放行要注意什么？",
        target_document_title="安全例外登记",
    )

    assert plan.retrieval_query.startswith("安全例外登记")
    assert "title_anchor" in plan.applied_strategies
    assert any(item.startswith("安全例外登记") for item in plan.lexical_queries)


def test_query_optimizer_merges_llm_rewrite_suggestion() -> None:
    class StubQueryOptimizer(QueryOptimizer):
        def _llm_rewrite(self, **kwargs):  # type: ignore[override]
            return QueryRewriteSuggestion(
                retrieval_query="平台发布手册 发布工单 字段 要求",
                lexical_queries=["发布工单 字段 要求", "平台发布 工单 字段"],
                provider="openai_compatible",
                model="rewrite-model",
                latency_ms=128,
            )

    optimizer = StubQueryOptimizer(Settings(query_rewrite_provider="auto"))
    plan = optimizer.build("《平台发布手册》里面提到，发布工单至少写明哪些信息？")

    assert plan.retrieval_query == "平台发布手册 发布工单 字段 要求"
    assert "llm_rewrite" in plan.applied_strategies
    assert plan.rewrite_provider == "openai_compatible"
    assert plan.rewrite_model == "rewrite-model"
    assert plan.rewrite_latency_ms == 128
    assert "发布工单 字段 要求" in plan.lexical_queries
    assert any(candidate.label == "LLM 改写" for candidate in plan.candidates)


def test_query_optimizer_skips_llm_rewrite_for_short_direct_topic_question() -> None:
    optimizer = QueryOptimizer(Settings(query_rewrite_provider="auto"))

    assert optimizer._should_use_llm("节假日安排是什么样的？", None) is False
