from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from uuid import uuid4

from app.core.config import Settings
from app.repositories.retrieval_repository import RetrievalCandidate, RetrievalRepository
from app.services.retrieval.reranker import (
    HeuristicReranker,
    LLMReranker,
    QwenReranker,
    RerankCandidate,
    RerankerFactory,
    _build_rerank_payload,
    _best_structural_block_score,
    _resolve_qwen_rerank_url,
    _select_rerank_candidates,
)


def _candidate(
    *,
    document_title: str,
    content: str,
    section_title: str | None = None,
    fused_score: float,
    lexical_raw: float = 0.0,
    vector_raw: float = 0.0,
    document_id=None,
    chunk_index: int = 0,
) -> RerankCandidate:
    doc_id = document_id or uuid4()
    return RerankCandidate(
        candidate=RetrievalCandidate(
            chunk_id=uuid4(),
            document_id=doc_id,
            document_title=document_title,
            document_version_id=uuid4(),
            version_number=1,
            chunk_index=chunk_index,
            content=content,
            token_count=len(content),
            section_title=section_title,
            page_number_start=None,
            page_number_end=None,
            paragraph_start=None,
            paragraph_end=None,
            char_start=None,
            char_end=None,
            citation_metadata=None,
            lexical_score=lexical_raw,
            vector_score=vector_raw,
        ),
        lexical_raw=lexical_raw,
        vector_raw=vector_raw,
        fused_score=fused_score,
    )


def _completion_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _settings(**overrides) -> Settings:
    defaults = {
        "rerank_provider": "llm",
        "llm_api_key": "test-key",
        "llm_router_model": "deepseek-v4-flash",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def test_reranker_prefers_candidate_with_stronger_title_and_content_overlap() -> None:
    reranker = HeuristicReranker()
    query = "员工手册 请假规定"

    platform_candidate = _candidate(
        document_title="平台发布手册",
        section_title="发布检查",
        content="平台发布窗口、回滚联系人和值班安排说明。",
        fused_score=0.61,
        vector_raw=0.61,
    )
    handbook_candidate = _candidate(
        document_title="员工手册",
        section_title="请假管理",
        content="员工申请年假前需要提交请假审批并同步直属负责人。",
        fused_score=0.56,
        lexical_raw=0.12,
        vector_raw=0.55,
    )

    result = reranker.rerank(query, [platform_candidate, handbook_candidate], top_k=2)

    assert result.candidates[0].candidate.document_title == "员工手册"
    assert result.candidates[0].rerank_score > result.candidates[1].rerank_score


def test_reranker_applies_target_document_bonus_after_acl_safe_retrieval() -> None:
    reranker = HeuristicReranker()
    target_document_id = uuid4()
    query = "请假规定"

    other_candidate = _candidate(
        document_title="员工手册",
        content="员工请假前需提交审批。",
        fused_score=0.52,
        lexical_raw=0.08,
        document_id=uuid4(),
    )
    target_candidate = _candidate(
        document_title="员工手册",
        content="请假规定要求提前同步负责人并完成审批。",
        fused_score=0.5,
        lexical_raw=0.08,
        document_id=target_document_id,
    )

    result = reranker.rerank(query, [other_candidate, target_candidate], top_k=2, target_document_id=target_document_id)

    assert result.candidates[0].candidate.document_id == target_document_id


def test_reranker_limits_results_to_top_k() -> None:
    reranker = HeuristicReranker()
    query = "发布检查清单"
    candidates = [
        _candidate(document_title="平台发布手册", content=f"发布检查项 {index}", fused_score=0.4 + (index * 0.01), lexical_raw=0.1)
        for index in range(5)
    ]

    result = reranker.rerank(query, candidates, top_k=3)

    assert len(result.candidates) == 3
    assert result.pre_rerank_count == 5
    assert result.post_rerank_count == 3
    assert result.strategy == "heuristic-overlap"


def test_reranker_reorders_business_candidates_after_hybrid_retrieval() -> None:
    reranker = HeuristicReranker()
    query = "平台发布检查清单里回滚联系人和验收检查项要求什么"

    semantically_close_but_generic = _candidate(
        document_title="事故响应指南",
        section_title="值班安排",
        content="出现异常后需要同步负责人、确认联系人，并在处置结束后补充记录。",
        fused_score=0.57,
        vector_raw=0.57,
        chunk_index=1,
    )
    policy_chunk = _candidate(
        document_title="平台发布手册",
        section_title="回滚与验收检查项",
        content="发布工单中需要记录回滚联系人名单和验收检查项，并在发布结束后补全时间线。",
        fused_score=0.52,
        lexical_raw=0.11,
        vector_raw=0.5,
        chunk_index=0,
    )

    result = reranker.rerank(query, [semantically_close_but_generic, policy_chunk], top_k=2)

    assert result.candidates[0].candidate.document_title == "平台发布手册"
    assert result.candidates[0].fused_score < result.candidates[1].fused_score
    assert result.candidates[0].rerank_score > result.candidates[1].rerank_score


def test_reranker_penalizes_negative_evidence_even_when_terms_overlap() -> None:
    reranker = HeuristicReranker()
    query = "客服接到高优先级工单后，首次响应时间要求是多少？"

    negative_candidate = _candidate(
        document_title="客户事故响应指南",
        content="这里没有定义客服对工单的首次响应时间，只要求经理在五分钟内建立事故沟通渠道。",
        fused_score=0.58,
        vector_raw=0.58,
        chunk_index=0,
    )
    grounded_candidate = _candidate(
        document_title="客户支持、数据导出与知识库维护协作规范",
        content="P1 工单：五分钟内完成首次响应，十分钟内完成内部升级。首次响应至少要包含三项信息。",
        fused_score=0.49,
        lexical_raw=0.05,
        vector_raw=0.49,
        chunk_index=1,
    )

    result = reranker.rerank(query, [negative_candidate, grounded_candidate], top_k=2)

    assert result.candidates[0].candidate.document_title == "客户支持、数据导出与知识库维护协作规范"
    assert result.candidates[0].rerank_score > result.candidates[1].rerank_score


def test_reranker_can_find_table_row_later_in_chunk_content() -> None:
    reranker = HeuristicReranker()
    query = "包含客户手机号的数据导出由谁审批，处理时限和脱敏要求是什么？"

    generic_policy_chunk = _candidate(
        document_title="客户支持、数据导出与知识库维护协作规范",
        section_title="客户沟通要求",
        content="不应直接对外同步未验证的根因猜测、内部员工姓名、内部系统名称或审计信息。涉及客户数据时应谨慎处理。",
        fused_score=0.62,
        vector_raw=0.62,
        chunk_index=2,
    )
    table_chunk = _candidate(
        document_title="客户支持、数据导出与知识库维护协作规范",
        section_title="数据导出审批与脱敏矩阵",
        content=(
            "数据导出申请需要根据字段敏感程度、客户范围和交付渠道判断审批人。"
            "普通运营报表由部门负责人审批。包含客户邮箱时由管理员审批。"
            "其他说明用于拉长 chunk，避免答案总在开头。"
            "交付记录需要包含导出编号、审批人、生成时间、字段说明、脱敏方式和接收确认状态。"
            "Table row: 数据导出审批与脱敏矩阵. 数据范围=包含客户手机号; 审批人=管理员; "
            "处理时限=2 个工作日; 脱敏要求=保留前三位和后四位; 允许交付渠道=加密邮件."
        ),
        fused_score=0.51,
        lexical_raw=0.03,
        vector_raw=0.51,
        chunk_index=8,
    )

    result = reranker.rerank(query, [generic_policy_chunk, table_chunk], top_k=2)

    assert result.candidates[0].candidate.section_title == "数据导出审批与脱敏矩阵"
    assert "包含客户手机号" in result.candidates[0].candidate.content


def test_reranker_prefers_pdf_table_row_for_l4_supplier_requirements() -> None:
    reranker = HeuristicReranker()
    query = "L4 高风险供应商的审批链路、复核周期和退出要求是什么？"

    generic_role_chunk = _candidate(
        document_title="供应商准入、合同变更与临时采购协作规范",
        section_title="PDF page 2 table 1",
        content=(
            "Table row: PDF page 2 table 1. 角色=采购专员; 主要职责=检查报价、比价、供应商资质和流程完整性; "
            "不应承担的职责=不判断系统安全风险; 必须输出=比价记录；供应商准入记录."
        ),
        fused_score=0.62,
        vector_raw=0.62,
        chunk_index=2,
    )
    l4_table_chunk = _candidate(
        document_title="供应商准入、合同变更与临时采购协作规范",
        section_title="PDF page 3 table 1",
        content=(
            "Table row: PDF page 3 table 1. 准入等级=L4 高风险; "
            "触发条件=可访问生产环境、核心数据库、客户敏感字段或长期驻场; "
            "审批链路=部门负责人；法务负责人；财务负责人；信息安全负责人; "
            "复核周期=每月复核一次; 允许接触数据=原则上不允许直接接触原始敏感数据; "
            "退出要求=必须有退出清单、账号回收证明和复盘记录."
        ),
        fused_score=0.48,
        lexical_raw=0.02,
        vector_raw=0.48,
        chunk_index=3,
    )

    result = reranker.rerank(query, [generic_role_chunk, l4_table_chunk], top_k=2)

    assert "准入等级=L4 高风险" in result.candidates[0].candidate.content
    assert result.candidates[0].rerank_score > result.candidates[1].rerank_score


def test_reranker_prefers_pdf_table_row_for_data_processing_acceptance() -> None:
    reranker = HeuristicReranker()
    query = "数据处理服务验收时需要哪些材料，验收人是谁，资料保留多久？"

    generic_chunk = _candidate(
        document_title="供应商准入、合同变更与临时采购协作规范",
        section_title="验收、付款与资料归档",
        content="验收不是确认供应商已经做了事，而是确认交付物满足合同和业务目标。若验收材料不完整，财务不得安排付款。",
        fused_score=0.61,
        vector_raw=0.61,
        chunk_index=6,
    )
    table_chunk = _candidate(
        document_title="供应商准入、合同变更与临时采购协作规范",
        section_title="PDF page 5 table 2",
        content=(
            "Table row: PDF page 5 table 2. 交付类型=数据处理服务; "
            "验收材料=字段说明；脱敏方式；抽样检查结果; 验收人=数据 owner；信息安全负责人; "
            "付款前置条件=字段范围一致；无未授权字段; 归档位置=数据治理平台; 保留期限=5 年."
        ),
        fused_score=0.47,
        lexical_raw=0.03,
        vector_raw=0.47,
        chunk_index=7,
    )

    result = reranker.rerank(query, [generic_chunk, table_chunk], top_k=2)

    assert "交付类型=数据处理服务" in result.candidates[0].candidate.content
    assert result.candidates[0].rerank_score > result.candidates[1].rerank_score


def test_retrieval_repository_tokenizes_chinese_query_ngrams() -> None:
    tokens = set(RetrievalRepository._tokenize("扫描附件里扫描A的处理动作是什么？临时高权限访问由谁审批？"))

    assert "扫描" in tokens
    assert "附件" in tokens
    assert "处理" in tokens
    assert "动作" in tokens
    assert "临时" in tokens
    assert "权限" in tokens
    assert "访问" in tokens
    assert "审批" in tokens
    assert "a" in tokens


def test_retrieval_repository_builds_sql_terms_for_chinese_ngram_search() -> None:
    terms = RetrievalRepository._select_sql_query_terms("客户手机号数据导出处理时限", max_terms=12)
    tsquery = RetrievalRepository._build_or_tsquery(terms)

    assert "客户手机" in terms
    assert "数据导出" in terms
    assert "处理时限" in terms
    assert "|" in tsquery


def test_retrieval_repository_bm25_score_prefers_rare_query_terms() -> None:
    query_counts = Counter({"商标": 1, "先用权": 1, "个体工商户": 1})
    idf_by_term = {"商标": 2.4, "先用权": 3.1, "个体工商户": 0.1}

    rare_term_score = RetrievalRepository._score_bm25_terms(
        query_counts=query_counts,
        content_terms=Counter({"商标": 2, "先用权": 1}),
        idf_by_term=idf_by_term,
        avg_doc_len=20,
    )
    frequent_term_score = RetrievalRepository._score_bm25_terms(
        query_counts=query_counts,
        content_terms=Counter({"个体工商户": 8}),
        idf_by_term=idf_by_term,
        avg_doc_len=20,
    )

    assert rare_term_score > frequent_term_score


def test_reranker_prefers_chinese_ocr_table_row_for_scanned_case_action() -> None:
    reranker = HeuristicReranker()
    query = "扫描附件里扫描A的处理动作是什么？"

    generic_chunk = _candidate(
        document_title="客户数据导出与临时权限管理办法",
        section_title="总则",
        content="客户数据导出、临时高权限访问和供应商例外处理必须在工单系统中留痕，并按最小必要原则执行。",
        fused_score=0.62,
        lexical_raw=0.2,
        chunk_index=0,
    )
    table_chunk = _candidate(
        document_title="客户数据导出与临时权限管理办法",
        section_title="PDF page 4 OCR table 1",
        content="Table row: PDF page 4 OCR table 1. 编号=扫描 A; 负责 人=张 三; 时 限=6 小 时; 动作=核验 日 志.",
        fused_score=0.48,
        lexical_raw=0.18,
        chunk_index=1,
    )

    result = reranker.rerank(query, [generic_chunk, table_chunk], top_k=2)

    assert result.candidates[0].candidate.section_title == "PDF page 4 OCR table 1"
    assert "核验 日 志" in result.candidates[0].candidate.content


def test_reranker_enterprise_profile_does_not_apply_legal_benchmark_bonus() -> None:
    reranker = HeuristicReranker()
    query = "谁可以成为个体工商户？"

    exact_legal_clause = _candidate(
        document_title="中华人民共和国民法典",
        section_title="第五十四条",
        content="条款全称：中华人民共和国民法典第五十四条\n\n自然人从事工商业经营，经依法登记，为个体工商户。",
        fused_score=0.3,
        lexical_raw=0.0,
        chunk_index=1,
    )
    generic_business_chunk = _candidate(
        document_title="个体经营服务说明",
        section_title="登记服务",
        content="个体工商户登记服务说明，市场主体可以查看办理流程、材料清单和经营者信息。",
        fused_score=0.61,
        lexical_raw=0.2,
        chunk_index=0,
    )

    result = reranker.rerank(query, [generic_business_chunk, exact_legal_clause], top_k=2)

    assert result.candidates[0].candidate.document_title == "个体经营服务说明"


def test_reranker_prefers_direct_enterprise_approval_owner_answer() -> None:
    reranker = HeuristicReranker()
    query = "客户手机号导出要谁审批？"
    archive_chunk = _candidate(
        document_title="客户数据导出与临时权限管理办法",
        section_title="归档要求",
        content="导出申请、审批记录和脱敏说明至少保留五年。",
        fused_score=0.72,
        lexical_raw=0.2,
        chunk_index=2,
    )
    approval_chunk = _candidate(
        document_title="客户数据导出与临时权限管理办法",
        section_title="审批条件",
        content="包含客户手机号的数据导出必须由数据 owner 和信息安全负责人共同审批。",
        fused_score=0.58,
        lexical_raw=0.18,
        chunk_index=1,
    )

    result = reranker.rerank(query, [archive_chunk, approval_chunk], top_k=2)

    assert result.candidates[0].candidate.section_title == "审批条件"


def test_reranker_prefers_important_data_export_assessment_over_generic_risk_assessment() -> None:
    reranker = HeuristicReranker()
    query = "重要数据出境有哪些安全评估要求？"
    generic_risk_assessment = _candidate(
        document_title="网络数据安全管理条例",
        section_title="第三十三条",
        content="重要数据的处理者应当每年度对其网络数据处理活动开展风险评估，并报送风险评估报告。",
        fused_score=0.64,
        lexical_raw=0.24,
        chunk_index=33,
    )
    export_assessment = _candidate(
        document_title="网络数据安全管理条例",
        section_title="第三十七条",
        content="网络数据处理者在境内运营中收集和产生的重要数据确需向境外提供的，应当通过国家网信部门组织的数据出境安全评估。",
        fused_score=0.43,
        lexical_raw=0.12,
        chunk_index=37,
    )

    result = reranker.rerank(query, [generic_risk_assessment, export_assessment], top_k=2)

    assert result.candidates[0].candidate.section_title == "第三十七条"


def test_reranker_prefers_direct_list_answer_for_kind_questions() -> None:
    reranker = HeuristicReranker()
    query = "国有企业管理人员处分的种类包括哪些？"
    generic_chunk = _candidate(
        document_title="国有企业管理人员处分条例",
        section_title="第六条",
        content="给予国有企业管理人员处分，应当事实清楚、证据确凿、定性准确、处理恰当、程序合法、手续完备。第二章 处分的种类和适用",
        fused_score=0.72,
        lexical_raw=0.4,
        chunk_index=6,
    )
    list_chunk = _candidate(
        document_title="国有企业管理人员处分条例",
        section_title="第七条",
        content="第七条 处分的种类为：（一）警告；（二）记过；（三）记大过；（四）降级；（五）撤职；（六）开除。",
        fused_score=0.45,
        lexical_raw=0.3,
        chunk_index=7,
    )

    result = reranker.rerank(query, [generic_chunk, list_chunk], top_k=2)

    assert result.candidates[0].candidate.section_title == "第七条"


def test_reranker_legal_profile_scores_relevant_structural_clause_block_inside_long_chunk() -> None:
    reranker = HeuristicReranker(Settings(retrieval_domain_profile="legal_benchmark"))
    query = "个体工商户的诉讼主体是什么，债务由谁承担？"

    generic_business_chunk = _candidate(
        document_title="促进个体工商户发展条例",
        section_title="扶持措施",
        content="个体工商户可以依法享受登记、税费、金融和创业扶持政策，市场监督管理部门应当优化服务。",
        fused_score=0.62,
        lexical_raw=0.2,
        chunk_index=2,
    )
    civil_code_chunk = _candidate(
        document_title="中华人民共和国民法典",
        section_title="第五十六条",
        content=(
            "## 第五十四条\n\n"
            "条款全称：中华人民共和国民法典第五十四条\n\n"
            "自然人从事工商业经营，经依法登记，为个体工商户。个体工商户可以起字号。\n\n"
            "## 第五十六条\n\n"
            "条款全称：中华人民共和国民法典第五十六条\n\n"
            "个体工商户的债务，个人经营的，以个人财产承担；家庭经营的，以家庭财产承担；无法区分的，以家庭财产承担。"
        ),
        fused_score=0.48,
        lexical_raw=0.06,
        chunk_index=5,
    )

    result = reranker.rerank(query, [generic_business_chunk, civil_code_chunk], top_k=2)

    assert result.candidates[0].candidate.document_title == "中华人民共和国民法典"
    assert result.candidates[0].rerank_score > result.candidates[1].rerank_score


def test_structural_block_score_focuses_best_clause_not_whole_chunk() -> None:
    content = (
        "## 第六十二条\n\n"
        "条款全称：农村土地承包法第六十二条\n\n"
        "发包方应当维护承包方的土地承包经营权。\n\n"
        "## 第六十四条\n\n"
        "条款全称：农村土地承包法第六十四条\n\n"
        "承包方不得单方解除土地经营权流转合同，但受让方擅自改变土地的农业用途、弃耕抛荒连续两年以上或者其他严重违约行为的除外。"
    )
    query_features = {"发包方", "解除合同", "未按时缴纳费用", "严重违约"}

    score = _best_structural_block_score("农村土地承包方未按时缴纳费用，发包方可以解除合同吗？", content, query_features)

    assert score > 0.2


def test_reranker_factory_defaults_to_heuristic() -> None:
    reranker = RerankerFactory.create(Settings(rerank_provider="heuristic"))

    assert isinstance(reranker, HeuristicReranker)


def test_reranker_factory_auto_prefers_qwen_when_qwen_credentials_exist() -> None:
    reranker = RerankerFactory.create(
        Settings(rerank_provider="auto", qwen_api_key="qwen-key", llm_api_key="llm-key"),
    )

    assert isinstance(reranker, QwenReranker)


def test_rerank_settings_defaults_match_latency_budget(monkeypatch) -> None:
    monkeypatch.delenv("RETRIEVAL_IN_DOCUMENT_EXPANSION_ADJACENT_WINDOW", raising=False)
    monkeypatch.delenv("RETRIEVAL_INDEXED_SPARSE_ENABLED", raising=False)
    settings = Settings(
        _env_file=None,
        rerank_max_candidates=16,
        retrieval_in_document_expansion_enabled=True,
    )

    assert settings.rerank_max_candidates == 16
    assert settings.rerank_timeout_seconds == 8.0
    assert settings.retrieval_cjk_sql_sparse_enabled is False
    assert settings.retrieval_cjk_python_scorer == "bm25"
    assert settings.retrieval_in_document_expansion_enabled is True
    assert settings.retrieval_in_document_expansion_adjacent_window == 2
    assert settings.retrieval_indexed_sparse_enabled is False
    assert settings.retrieval_indexed_sparse_candidate_multiplier == 1
    assert settings.retrieval_indexed_sparse_sql_row_multiplier == 1
    assert settings.retrieval_indexed_sparse_max_query_terms == 10
    assert settings.retrieval_indexed_sparse_timeout_fallback_enabled is True
    assert settings.retrieval_indexed_sparse_timeout_fallback_max_query_terms == 4
    assert settings.retrieval_indexed_sparse_timeout_python_fallback_enabled is False
    assert settings.retrieval_query_decomposition_enabled is True
    assert settings.retrieval_query_decomposition_min_subquery_candidates == 4
    assert settings.retrieval_subquery_document_evidence_enabled is True
    assert settings.retrieval_subquery_document_evidence_seed_documents == 3
    assert settings.retrieval_subquery_document_evidence_per_subquery == 4
    assert settings.retrieval_subquery_document_evidence_max_candidates == 32
    assert settings.retrieval_subquery_document_evidence_score_weight == 0.42
    assert settings.retrieval_subquery_neighbor_context_enabled is True
    assert settings.retrieval_subquery_neighbor_context_seed_count == 4
    assert settings.retrieval_subquery_neighbor_context_window == 5
    assert settings.retrieval_subquery_neighbor_context_per_subquery == 8
    assert settings.retrieval_subquery_neighbor_context_max_candidates == 16
    assert settings.retrieval_subquery_neighbor_context_score_weight == 0.55
    assert settings.retrieval_subquery_final_coverage_max_slots == 2
    assert settings.retrieval_document_evidence_sweep_enabled is False
    assert settings.retrieval_document_evidence_sweep_seed_documents == 6
    assert settings.retrieval_document_neighbor_context_enabled is False
    assert settings.retrieval_document_neighbor_context_window == 2
    assert settings.retrieval_evidence_preservation_enabled is False
    assert settings.retrieval_evidence_preservation_max_slots == 1
    assert settings.retrieval_final_coverage_protected_top_k_slots == 3
    assert settings.retrieval_document_diversity_protected_top_k_slots == 1
    assert settings.retrieval_heuristic_rerank_enabled is False


def test_semantic_rerank_candidate_selection_reserves_evidence_source_candidates() -> None:
    regular_candidates = [
        _candidate(
            document_title=f"普通制度 {index}",
            content=f"普通候选 {index}",
            fused_score=1.0 - (index * 0.05),
            lexical_raw=0.2,
            chunk_index=index,
        )
        for index in range(6)
    ]
    expansion_candidate = _candidate(
        document_title="客户数据导出审批办法",
        content="负责人=信息安全负责人；处理时限=2 个工作日。",
        fused_score=0.01,
        lexical_raw=0.9,
        chunk_index=30,
    )
    expansion_candidate.sources.add("document_expansion")

    selected = _select_rerank_candidates([*regular_candidates, expansion_candidate], limit=4)

    assert regular_candidates[0] in selected
    assert regular_candidates[1] in selected
    assert regular_candidates[2] in selected
    assert expansion_candidate in selected
    assert regular_candidates[3] not in selected


def test_semantic_rerank_candidate_selection_balances_evidence_sources() -> None:
    regular_candidates = [
        _candidate(
            document_title=f"普通制度 {index}",
            content=f"普通候选 {index}",
            fused_score=1.0 - (index * 0.02),
            lexical_raw=0.2,
            chunk_index=index,
        )
        for index in range(20)
    ]
    document_first_candidates = [
        _candidate(
            document_title=f"高分文档优先候选 {index}",
            content=f"高分文档优先候选 {index}",
            fused_score=0.05,
            lexical_raw=0.95 - (index * 0.01),
            chunk_index=100 + index,
        )
        for index in range(8)
    ]
    for candidate in document_first_candidates:
        candidate.sources.add("document_first_evidence")
    expansion_candidates = [
        _candidate(
            document_title=f"扩展候选 {index}",
            content=f"扩展候选 {index}",
            fused_score=0.01,
            lexical_raw=0.5 - (index * 0.01),
            chunk_index=200 + index,
        )
        for index in range(3)
    ]
    for candidate in expansion_candidates:
        candidate.sources.add("document_expansion")

    selected = _select_rerank_candidates(
        [*regular_candidates, *document_first_candidates, *expansion_candidates],
        limit=16,
    )

    assert expansion_candidates[2] in selected
    assert any(item in selected for item in document_first_candidates)
    assert len(selected) == 16


def test_qwen_rerank_url_appends_reranks_for_compatible_base() -> None:
    settings = Settings(qwen_base_url="https://dashscope.aliyuncs.com/compatible-api/v1")

    assert _resolve_qwen_rerank_url(settings) == "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"


def test_llm_rerank_payload_keeps_relevant_table_row_even_when_late_in_chunk() -> None:
    table_candidate = _candidate(
        document_title="客户支持、数据导出与知识库维护协作规范",
        section_title="数据导出审批与脱敏矩阵",
        content=(
            "数据导出申请需要根据字段敏感程度、客户范围和交付渠道判断审批人。"
            "普通运营报表由部门负责人审批。包含客户邮箱时由管理员审批。"
            "其他说明用于拉长 chunk，避免答案总在开头。"
            "交付记录需要包含导出编号、审批人、生成时间、字段说明、脱敏方式和接收确认状态。"
            "Table row: 数据导出审批与脱敏矩阵. 数据范围=普通运营数据; 审批人=部门负责人; 处理时限=1 个工作日."
            "Table row: 数据导出审批与脱敏矩阵. 数据范围=包含客户邮箱; 审批人=管理员; 处理时限=1 个工作日."
            "Table row: 数据导出审批与脱敏矩阵. 数据范围=包含客户手机号; 审批人=管理员; "
            "处理时限=2 个工作日; 脱敏要求=保留前三位和后四位; 允许交付渠道=加密邮件."
        ),
        fused_score=0.51,
        lexical_raw=0.03,
        vector_raw=0.51,
        chunk_index=8,
    )

    payload = _build_rerank_payload(
        "包含客户手机号的数据导出由谁审批，处理时限和脱敏要求是什么？",
        [table_candidate],
    )
    preview = payload["candidates"][0]["content_preview"]

    assert "数据范围=包含客户手机号" in preview
    assert "脱敏要求=保留前三位和后四位" in preview
    assert len(preview) <= 800


def test_llm_rerank_payload_keeps_normal_text_preview_short() -> None:
    content = ("发布工单需要记录回滚联系人和值班信息。" * 80).strip()
    text_candidate = _candidate(
        document_title="平台发布手册",
        section_title="回滚安排",
        content=content,
        fused_score=0.71,
        lexical_raw=0.14,
        vector_raw=0.7,
        chunk_index=0,
    )

    payload = _build_rerank_payload("回滚联系人和值班信息要求什么", [text_candidate])
    preview = payload["candidates"][0]["content_preview"]

    assert preview == content[:600]
    assert len(preview) == 600


def test_llm_reranker_uses_chat_completion_and_applies_llm_order() -> None:
    captured: dict[str, object] = {}

    first_candidate = _candidate(
        document_title="平台发布手册",
        section_title="回滚安排",
        content="发布前确认回滚联系人和值班信息。",
        fused_score=0.71,
        lexical_raw=0.14,
        vector_raw=0.7,
        chunk_index=0,
    )
    second_candidate = _candidate(
        document_title="平台发布手册",
        section_title="验收检查项",
        content="发布工单需要记录验收检查项和结束时间。",
        fused_score=0.66,
        lexical_raw=0.09,
        vector_raw=0.64,
        chunk_index=1,
    )

    def completion_request(_client, **kwargs):
        captured.update(kwargs)
        return _completion_response(
            '{"ranked":['
            f'{{"chunk_id":"{second_candidate.candidate.chunk_id}","score":0.97,"reason":"direct answer"}},'
            f'{{"chunk_id":"{first_candidate.candidate.chunk_id}","score":0.81,"reason":"supporting detail"}}'
            "]}"
        )

    reranker = LLMReranker(
        settings=_settings(),
        client_factory=lambda _settings: object(),
        completion_request=completion_request,
    )

    result = reranker.rerank(
        "平台发布检查清单里回滚联系人和验收检查项要求什么",
        [first_candidate, second_candidate],
        top_k=2,
    )

    assert result.strategy == "llm-json"
    assert result.candidates[0].candidate.chunk_id == second_candidate.candidate.chunk_id
    assert result.candidates[0].rerank_score == 0.97
    assert captured["model"] == "deepseek-v4-flash"
    assert captured["temperature"] == 0.0
    assert "top_p" not in captured
    assert captured["response_format"] == {"type": "json_object"}


def test_llm_reranker_ignores_unknown_and_duplicate_chunk_ids() -> None:
    first_candidate = _candidate(
        document_title="事故响应指南",
        section_title="联系人同步",
        content="需要同步负责人和相关联系人。",
        fused_score=0.74,
        vector_raw=0.74,
        chunk_index=0,
    )
    second_candidate = _candidate(
        document_title="平台发布手册",
        section_title="回滚与验收检查项",
        content="发布工单中需要记录回滚联系人名单和验收检查项。",
        fused_score=0.63,
        lexical_raw=0.12,
        vector_raw=0.62,
        chunk_index=1,
    )
    third_candidate = _candidate(
        document_title="平台发布手册",
        section_title="发布时间线",
        content="发布结束后补全时间线和验收结果。",
        fused_score=0.58,
        lexical_raw=0.05,
        vector_raw=0.57,
        chunk_index=2,
    )

    reranker = LLMReranker(
        settings=_settings(),
        client_factory=lambda _settings: object(),
        completion_request=lambda _client, **kwargs: _completion_response(
            '{"ranked":['
            '{"chunk_id":"00000000-0000-0000-0000-000000000000","score":1.0,"reason":"unknown"},'
            f'{{"chunk_id":"{second_candidate.candidate.chunk_id}","score":0.93,"reason":"best match"}},'
            f'{{"chunk_id":"{second_candidate.candidate.chunk_id}","score":0.61,"reason":"duplicate"}},'
            f'{{"chunk_id":"{third_candidate.candidate.chunk_id}","score":0.55,"reason":"secondary"}}'
            "]}"
        ),
    )

    result = reranker.rerank(
        "平台发布检查清单里回滚联系人和验收检查项要求什么",
        [first_candidate, second_candidate, third_candidate],
        top_k=3,
    )

    ordered_chunk_ids = [item.candidate.chunk_id for item in result.candidates]

    assert result.strategy == "llm-json"
    assert ordered_chunk_ids[0] == second_candidate.candidate.chunk_id
    assert ordered_chunk_ids[1] == third_candidate.candidate.chunk_id
    assert ordered_chunk_ids[2] == first_candidate.candidate.chunk_id


def test_llm_reranker_falls_back_to_heuristic_on_llm_failure() -> None:
    first_candidate = _candidate(
        document_title="事故响应指南",
        section_title="值班安排",
        content="出现异常后需要同步负责人、确认联系人，并在处置结束后补充记录。",
        fused_score=0.57,
        vector_raw=0.57,
        chunk_index=1,
    )
    second_candidate = _candidate(
        document_title="平台发布手册",
        section_title="回滚与验收检查项",
        content="发布工单中需要记录回滚联系人名单和验收检查项，并在发布结束后补全时间线。",
        fused_score=0.52,
        lexical_raw=0.11,
        vector_raw=0.5,
        chunk_index=0,
    )

    heuristic_result = HeuristicReranker().rerank(
        "平台发布检查清单里回滚联系人和验收检查项要求什么",
        [first_candidate, second_candidate],
        top_k=2,
    )
    reranker = LLMReranker(
        settings=_settings(),
        client_factory=lambda _settings: object(),
        completion_request=lambda _client, **kwargs: _completion_response('{"ranked":[]}'),
    )

    result = reranker.rerank(
        "平台发布检查清单里回滚联系人和验收检查项要求什么",
        [first_candidate, second_candidate],
        top_k=2,
    )

    assert result.strategy == "llm-json-fallback-heuristic"
    assert [item.candidate.chunk_id for item in result.candidates] == [
        item.candidate.chunk_id for item in heuristic_result.candidates
    ]


def test_qwen_reranker_orders_candidates_by_score() -> None:
    first_candidate = _candidate(
        document_title="平台发布手册",
        section_title="回滚安排",
        content="发布前确认回滚联系人和值班信息。",
        fused_score=0.71,
        lexical_raw=0.14,
        vector_raw=0.7,
        chunk_index=0,
    )
    second_candidate = _candidate(
        document_title="平台发布手册",
        section_title="验收检查项",
        content="发布工单需要记录验收检查项和结束时间。",
        fused_score=0.66,
        lexical_raw=0.09,
        vector_raw=0.64,
        chunk_index=1,
    )

    reranker = QwenReranker(
        settings=_settings(rerank_provider="qwen", qwen_api_key="qwen-key"),
        request_rerank=lambda settings, *, query, documents, timeout: {
            "results": [
                {"index": 1, "relevance_score": 0.98},
                {"index": 0, "relevance_score": 0.77},
            ]
        },
    )

    result = reranker.rerank(
        "平台发布检查清单里回滚联系人和验收检查项要求什么",
        [first_candidate, second_candidate],
        top_k=2,
    )

    assert result.strategy == "qwen-rerank"
    assert result.candidates[0].candidate.chunk_id == second_candidate.candidate.chunk_id
    assert result.candidates[0].rerank_score == 0.98


def test_qwen_reranker_maps_scores_by_sorted_candidate_index() -> None:
    highest_fused = _candidate(
        document_title="事故响应指南",
        section_title="联系人同步",
        content="需要同步负责人和相关联系人。",
        fused_score=0.9,
        vector_raw=0.9,
        chunk_index=0,
    )
    lowest_fused = _candidate(
        document_title="平台发布手册",
        section_title="验收检查项",
        content="发布工单需要记录验收检查项和结束时间。",
        fused_score=0.4,
        lexical_raw=0.2,
        vector_raw=0.35,
        chunk_index=1,
    )

    reranker = QwenReranker(
        settings=_settings(rerank_provider="qwen", qwen_api_key="qwen-key"),
        request_rerank=lambda settings, *, query, documents, timeout: {
            "results": [{"index": 1, "relevance_score": 0.95}]
        },
    )

    result = reranker.rerank(
        "平台发布检查清单里回滚联系人和验收检查项要求什么",
        [lowest_fused, highest_fused],
        top_k=2,
    )

    assert result.candidates[0].candidate.chunk_id == lowest_fused.candidate.chunk_id
    assert result.candidates[0].rerank_score == 0.95


def test_qwen_reranker_ignores_duplicate_missing_and_out_of_range_indexes() -> None:
    first_candidate = _candidate(
        document_title="事故响应指南",
        section_title="联系人同步",
        content="需要同步负责人和相关联系人。",
        fused_score=0.74,
        vector_raw=0.74,
        chunk_index=0,
    )
    second_candidate = _candidate(
        document_title="平台发布手册",
        section_title="回滚与验收检查项",
        content="发布工单中需要记录回滚联系人名单和验收检查项。",
        fused_score=0.63,
        lexical_raw=0.12,
        vector_raw=0.62,
        chunk_index=1,
    )
    third_candidate = _candidate(
        document_title="平台发布手册",
        section_title="发布时间线",
        content="发布结束后补全时间线和验收结果。",
        fused_score=0.58,
        lexical_raw=0.05,
        vector_raw=0.57,
        chunk_index=2,
    )

    reranker = QwenReranker(
        settings=_settings(rerank_provider="qwen", qwen_api_key="qwen-key"),
        request_rerank=lambda settings, *, query, documents, timeout: {
            "results": [
                {"index": 99, "relevance_score": 1.0},
                {"index": 1, "relevance_score": 0.91},
                {"index": 1, "relevance_score": 0.3},
            ]
        },
    )

    heuristic_result = HeuristicReranker().rerank(
        "平台发布检查清单里回滚联系人和验收检查项要求什么",
        [first_candidate, second_candidate, third_candidate],
        top_k=3,
    )
    result = reranker.rerank(
        "平台发布检查清单里回滚联系人和验收检查项要求什么",
        [first_candidate, second_candidate, third_candidate],
        top_k=3,
    )

    assert result.strategy == "qwen-rerank"
    assert result.candidates[0].candidate.chunk_id == second_candidate.candidate.chunk_id
    assert [item.candidate.chunk_id for item in result.candidates[1:]] == [
        item.candidate.chunk_id
        for item in heuristic_result.candidates
        if item.candidate.chunk_id != second_candidate.candidate.chunk_id
    ]


def test_qwen_reranker_falls_back_to_heuristic_on_empty_or_invalid_results() -> None:
    first_candidate = _candidate(
        document_title="事故响应指南",
        section_title="值班安排",
        content="出现异常后需要同步负责人、确认联系人，并在处置结束后补充记录。",
        fused_score=0.57,
        vector_raw=0.57,
        chunk_index=1,
    )
    second_candidate = _candidate(
        document_title="平台发布手册",
        section_title="回滚与验收检查项",
        content="发布工单中需要记录回滚联系人名单和验收检查项，并在发布结束后补全时间线。",
        fused_score=0.52,
        lexical_raw=0.11,
        vector_raw=0.5,
        chunk_index=0,
    )

    heuristic_result = HeuristicReranker().rerank(
        "平台发布检查清单里回滚联系人和验收检查项要求什么",
        [first_candidate, second_candidate],
        top_k=2,
    )
    reranker = QwenReranker(
        settings=_settings(rerank_provider="qwen", qwen_api_key="qwen-key"),
        request_rerank=lambda settings, *, query, documents, timeout: {"results": []},
    )

    result = reranker.rerank(
        "平台发布检查清单里回滚联系人和验收检查项要求什么",
        [first_candidate, second_candidate],
        top_k=2,
    )

    assert result.strategy == "qwen-rerank-fallback-heuristic"
    assert [item.candidate.chunk_id for item in result.candidates] == [
        item.candidate.chunk_id for item in heuristic_result.candidates
    ]


def test_qwen_reranker_falls_back_to_heuristic_on_request_error() -> None:
    first_candidate = _candidate(
        document_title="事故响应指南",
        section_title="值班安排",
        content="出现异常后需要同步负责人、确认联系人，并在处置结束后补充记录。",
        fused_score=0.57,
        vector_raw=0.57,
        chunk_index=1,
    )
    second_candidate = _candidate(
        document_title="平台发布手册",
        section_title="回滚与验收检查项",
        content="发布工单中需要记录回滚联系人名单和验收检查项，并在发布结束后补全时间线。",
        fused_score=0.52,
        lexical_raw=0.11,
        vector_raw=0.5,
        chunk_index=0,
    )

    heuristic_result = HeuristicReranker().rerank(
        "平台发布检查清单里回滚联系人和验收检查项要求什么",
        [first_candidate, second_candidate],
        top_k=2,
    )

    def raising_request(*args, **kwargs):
        raise RuntimeError("boom")

    reranker = QwenReranker(
        settings=_settings(rerank_provider="qwen", qwen_api_key="qwen-key"),
        request_rerank=raising_request,
    )

    result = reranker.rerank(
        "平台发布检查清单里回滚联系人和验收检查项要求什么",
        [first_candidate, second_candidate],
        top_k=2,
    )

    assert result.strategy == "qwen-rerank-fallback-heuristic"
    assert [item.candidate.chunk_id for item in result.candidates] == [
        item.candidate.chunk_id for item in heuristic_result.candidates
    ]
