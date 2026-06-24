from __future__ import annotations

from app.core.config import Settings
from app.services.retrieval.query_optimizer import QueryOptimizer
from app.services.retrieval.query_optimizer import QueryRewriteSuggestion


def test_query_optimizer_builds_title_anchored_retrieval_query() -> None:
    optimizer = QueryOptimizer(Settings(query_rewrite_provider="deterministic"))

    plan = optimizer.build("《平台发布手册》里面提到，发布工单至少写明哪些信息？")

    assert plan.rewrite_applied is True
    assert "title_anchor" in plan.applied_strategies
    assert "focus_keywords" in plan.applied_strategies
    assert plan.retrieval_query.startswith("平台发布手册")
    assert len(plan.lexical_queries) >= 2
    assert plan.candidate_count >= 2
    assert any("title_anchor" in candidate.applied_strategies for candidate in plan.candidates)
    assert any(item.startswith("平台发布手册") for item in plan.lexical_queries)


def test_query_optimizer_decomposes_cross_document_comparison() -> None:
    optimizer = QueryOptimizer(Settings(query_rewrite_provider="deterministic"))

    plan = optimizer.build(
        "比较山东钢铁集团有限公司和深圳市环境水务集团有限公司两份融资与财务披露材料在融资安排与偿债披露上的披露，"
        "分别关注“战略重组情况 1、关于涉及战略重组的提示性公告发行人于 2021年”和"
        "“出资人机构深圳市人民政府国有资产监督管理委员会作为履行出资人职责的机”，各引用一处原文依据。"
    )

    assert plan.query_decomposition_applied is True
    assert len(plan.subqueries) == 2
    assert plan.subqueries[0].case_shape == "cross_document_comparison"
    assert plan.subqueries[0].org_hint == "山东钢铁集团有限公司"
    assert plan.subqueries[1].org_hint == "深圳市环境水务集团有限公司"
    assert "战略重组情况" in plan.subqueries[0].query_text
    assert "出资人机构深圳市人民政府" in plan.subqueries[1].query_text


def test_query_optimizer_decomposes_same_document_two_matters() -> None:
    optimizer = QueryOptimizer(Settings(query_rewrite_provider="deterministic"))

    plan = optimizer.build(
        "请同时核对青岛世园(集团)有限公司这份融资与财务披露材料中的两个事项："
        "“股权结构截至本募集说明书签署日”和“通过向上游供应商采购相关贸易商品后”，分别引用依据。"
    )

    assert plan.query_decomposition_applied is True
    assert len(plan.subqueries) == 2
    assert {item.case_shape for item in plan.subqueries} == {"same_document_two_matters"}
    assert {item.org_hint for item in plan.subqueries} == {"青岛世园(集团)有限公司"}
    assert "股权结构截至本募集说明书签署日" in plan.subqueries[0].query_text
    assert "通过向上游供应商采购相关贸易商品后" in plan.subqueries[1].query_text


def test_query_optimizer_does_not_decompose_single_fact_question() -> None:
    optimizer = QueryOptimizer(Settings(query_rewrite_provider="deterministic"))

    plan = optimizer.build("西部矿业集团有限公司的融资与财务披露材料中，“但报告期内持续改善”具体是怎么披露或规定的？")

    assert plan.query_decomposition_applied is False
    assert len(plan.subqueries) == 1
    assert plan.subqueries[0].case_shape == "single_evidence_anchor"
    assert plan.subqueries[0].org_hint == "西部矿业集团有限公司"
    assert "但报告期内持续改善" in plan.subqueries[0].query_text


def test_query_optimizer_marks_low_overlap_category_anchor() -> None:
    optimizer = QueryOptimizer(Settings(query_rewrite_provider="deterministic"))

    plan = optimizer.build(
        "投研或财务团队准备底稿时，需要在广东恒健投资控股有限公司的融资与财务披露材料里确认"
        "“融资安排与偿债披露”相关事项的处理口径。请指出相关原文依据。"
    )

    assert plan.query_decomposition_applied is False
    assert len(plan.subqueries) == 1
    assert plan.subqueries[0].case_shape == "single_category_anchor"
    assert plan.subqueries[0].org_hint == "广东恒健投资控股有限公司"
    assert "广东恒健投资控股有限公司" in plan.subqueries[0].query_text


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


def test_query_optimizer_skips_llm_rewrite_for_simple_target_document_lookup(monkeypatch) -> None:
    def fail_rewrite(*args, **kwargs):
        raise AssertionError("LLM query rewrite should not run for simple target-document lookup")

    monkeypatch.setattr("app.services.retrieval.query_optimizer.request_chat_completion", fail_rewrite)
    optimizer = QueryOptimizer(
        Settings(
            query_rewrite_provider="openai_compatible",
            llm_api_key="test-key",
            llm_base_url="https://example.invalid",
            llm_chat_model="test-model",
        )
    )

    plan = optimizer.build("总体目标是什么？", target_document_title="工业领域数据安全能力提升实施方案")

    assert "llm_rewrite" not in plan.applied_strategies
    assert plan.llm_rewrite_attempted is False
    assert plan.llm_rewrite_skipped_reason == "simple_or_precise_query"


def test_query_optimizer_uses_configured_llm_rewrite_timeout(monkeypatch) -> None:
    captured: dict[str, float] = {}

    class Message:
        content = '{"retrieval_query":"供应商准入 风险 赔偿 责任","lexical_queries":["供应商准入 风险 赔偿"]}'

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    def fake_request_chat_completion(*args, **kwargs):
        captured["timeout"] = kwargs["timeout"]
        return Response()

    monkeypatch.setattr("app.services.retrieval.query_optimizer.has_openai_compatible_credentials", lambda settings: True)
    monkeypatch.setattr("app.services.retrieval.query_optimizer.create_openai_compatible_client", lambda settings: object())
    monkeypatch.setattr("app.services.retrieval.query_optimizer.request_chat_completion", fake_request_chat_completion)
    optimizer = QueryOptimizer(
        Settings(
            query_rewrite_provider="openai_compatible",
            query_rewrite_timeout_seconds=0.25,
            llm_api_key="test-key",
            llm_base_url="https://example.invalid",
            llm_chat_model="test-model",
        )
    )

    plan = optimizer.build("请帮我梳理供应商准入风险以及赔偿责任相关的制度要求")

    assert captured["timeout"] == 0.25
    assert plan.llm_rewrite_attempted is True
    assert plan.llm_rewrite_latency_ms is not None
    assert "llm_rewrite" in plan.applied_strategies


def test_query_optimizer_enterprise_profile_does_not_apply_legal_benchmark_expansion() -> None:
    optimizer = QueryOptimizer(Settings(query_rewrite_provider="deterministic"))

    labor_plan = optimizer.build("平台经济从业者注册为个体工商户，那还存在劳动关系吗？")
    franchise_plan = optimizer.build("个体工商户作为特许人所签订的特许经营合同是否有效？")

    assert "促进个体工商户发展条例 第三十条" not in labor_plan.retrieval_query
    assert "中华人民共和国民法典 第五十四条" not in labor_plan.retrieval_query
    assert "商业特许经营管理条例 第三条" not in franchise_plan.retrieval_query
    assert "中华人民共和国民法典 第五十四条" not in franchise_plan.retrieval_query


def test_query_optimizer_legal_profile_keeps_specific_business_intent_over_generic_entity_expansion() -> None:
    optimizer = QueryOptimizer(Settings(query_rewrite_provider="deterministic", retrieval_domain_profile="legal_benchmark"))

    labor_plan = optimizer.build("平台经济从业者注册为个体工商户，那还存在劳动关系吗？")
    franchise_plan = optimizer.build("个体工商户作为特许人所签订的特许经营合同是否有效？")

    assert "促进个体工商户发展条例 第三十条" in labor_plan.retrieval_query
    assert "中华人民共和国民法典 第五十四条" not in labor_plan.retrieval_query
    assert "商业特许经营管理条例 第三条" in franchise_plan.retrieval_query
    assert "中华人民共和国民法典 第五十四条" not in franchise_plan.retrieval_query


def test_query_optimizer_enterprise_profile_keeps_generic_enterprise_expansions() -> None:
    optimizer = QueryOptimizer(Settings(query_rewrite_provider="deterministic"))

    plan = optimizer.build("客户手机号的数据导出审批人、处理时限和脱敏要求是什么？")

    assert "客户手机号" in plan.retrieval_query
    assert "处理时限" in plan.retrieval_query
    assert "脱敏要求" in plan.retrieval_query
    assert "民法典" not in plan.retrieval_query


def test_query_optimizer_evidence_bridge_is_default_off() -> None:
    optimizer = QueryOptimizer(Settings(query_rewrite_provider="deterministic"))

    plan = optimizer.build("供应商合同是否需要承担赔偿责任？")

    assert "evidence_bridge" not in plan.applied_strategies
    assert all("证据意图扩展" != candidate.label for candidate in plan.candidates)


def test_query_optimizer_evidence_bridge_adds_generic_intent_terms_without_domain_law_terms() -> None:
    optimizer = QueryOptimizer(
        Settings(
            query_rewrite_provider="deterministic",
            retrieval_evidence_query_bridge_enabled=True,
            retrieval_evidence_query_bridge_max_queries=3,
        )
    )

    plan = optimizer.build("供应商合同是否需要承担赔偿责任？")

    assert "evidence_bridge" in plan.applied_strategies
    assert "责任" in plan.retrieval_query
    assert "赔偿" in plan.retrieval_query
    assert "条件" in plan.retrieval_query
    assert "民法典" not in plan.retrieval_query
    assert "合同法" not in plan.retrieval_query


def test_query_optimizer_expands_chinese_definition_questions() -> None:
    optimizer = QueryOptimizer(Settings(query_rewrite_provider="deterministic"))

    plan = optimizer.build("办法中对绿色工厂本身是怎么定义的？")

    assert "domain_expansion" in plan.applied_strategies
    assert "绿色工厂 是指" in plan.retrieval_query
    assert any("本办法所称 绿色工厂 是指" in item for item in plan.lexical_queries)


def test_query_optimizer_does_not_treat_table_lookup_as_definition_question() -> None:
    optimizer = QueryOptimizer(Settings(query_rewrite_provider="deterministic"))

    plan = optimizer.build("请核对淄博市城市资产运营集团有限公司文件中的表格或清单信息，“2025年1-3月为7.19”对应的数值、对象或判断是什么？")

    assert "domain_expansion" not in plan.applied_strategies
    assert "本办法所称" not in plan.retrieval_query
    assert "2025年1-3月为7.19" in plan.retrieval_query


def test_query_optimizer_legal_profile_prioritizes_family_contract_democratic_procedure_rules() -> None:
    optimizer = QueryOptimizer(Settings(query_rewrite_provider="deterministic", retrieval_domain_profile="legal_benchmark"))

    plan = optimizer.build("如果没有经过民主议定程序，家庭承包的土地承包经营合同还有效吗？")

    assert plan.retrieval_query.startswith("农村土地承包法 第十九条")
    assert "农村土地承包法 第二十八条" in plan.retrieval_query
