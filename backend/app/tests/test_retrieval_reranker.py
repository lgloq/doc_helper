from __future__ import annotations

from uuid import uuid4

from app.repositories.retrieval_repository import RetrievalCandidate
from app.services.retrieval.reranker import HeuristicReranker, RerankCandidate


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
