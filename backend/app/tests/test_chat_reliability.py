from __future__ import annotations

from uuid import UUID, uuid4

from app.schemas.search import SearchResultChunk, SearchScoreBreakdown
from app.services.chat.evidence_audit import build_evidence_audit
from app.services.chat.reliability import should_abstain_from_answer


def _chunk(
    *,
    content: str,
    fused: float = 0.4,
    rerank: float | None = None,
    chunk_index: int = 1,
    document_title: str = "供应商准入、合同变更与临时采购协作规范",
    document_id: UUID | None = None,
) -> SearchResultChunk:
    return SearchResultChunk(
        chunk_id=uuid4(),
        document_id=document_id or uuid4(),
        document_title=document_title,
        document_version_id=uuid4(),
        version_number=1,
        chunk_index=chunk_index,
        content=content,
        preview=content[:500],
        section_title="PDF table",
        page_number_start=3,
        page_number_end=3,
        paragraph_start=20,
        paragraph_end=20,
        citation_preview={
            "document_title": document_title,
            "version_number": 1,
            "chunk_id": str(uuid4()),
            "section_title": "PDF table",
            "preview": content[:500],
        },
        score=SearchScoreBreakdown(
            lexical_raw=0.0,
            lexical_normalized=0.0,
            vector_raw=fused,
            vector_normalized=fused,
            fused=fused,
            rerank=rerank,
        ),
    )


def test_reliability_prefers_response_time_table_rows_over_meta_chunks() -> None:
    meta_chunk = _chunk(
        content=(
            "本文档用于演示企业知识库系统在客户支持、权限控制、引用式问答等场景下的处理能力。"
            "为了方便上传后立即测试，你可以尝试以下问题：P1 工单的首次响应时间是多少？"
        ),
        fused=0.45,
        document_title="客户支持、数据导出与知识库维护协作规范",
    )
    faq_chunk = _chunk(
        content=(
            "Table row: FAQ 沉淀候选. 问题=临时高权限访问多久回收; "
            "推荐回答要点=最长 7 天，到期自动回收权限."
        ),
        fused=0.18,
        document_title="运营审批与客户响应规范",
    )
    table_chunk = _chunk(
        content=(
            "Table row: 当前客户工单响应矩阵. 工单等级=P1; 首次响应=5 分钟内; "
            "升级条件=核心业务中断、无法登录、付费能力不可用或数据损坏风险; "
            "负责人=一线客服立即通知值班经理."
        ),
        fused=0.08,
        document_title="运营审批与客户响应规范",
    )

    decision = should_abstain_from_answer(
        "客服接到高优先级工单后，首次响应时间要求是多少？",
        [meta_chunk, faq_chunk, table_chunk],
        None,
    )

    assert decision.should_abstain is False
    assert decision.filtered_chunks
    assert decision.filtered_chunks[0].chunk_id == table_chunk.chunk_id
    assert "首次响应=5 分钟内" in decision.filtered_chunks[0].content


def test_evidence_audit_supports_precise_table_lookup_answer() -> None:
    table_chunk = _chunk(
        content=(
            "Table row: 当前客户工单响应矩阵. 工单等级=P1; 首次响应=5 分钟内; "
            "升级条件=核心业务中断、无法登录、付费能力不可用或数据损坏风险; "
            "负责人=一线客服立即通知值班经理."
        ),
        document_title="运营审批与客户响应规范",
    )

    audit = build_evidence_audit(
        "依据《运营审批与客户响应规范》，按当前客户工单响应矩阵，P1 工单首次响应要求是 5 分钟内。",
        [table_chunk],
    )

    assert audit["status"] == "supported"
    assert audit["score"] == 1.0
    assert audit["supported_count"] == 1
    assert audit["claims"][0]["support_status"] == "supported"

def test_evidence_audit_does_not_join_values_across_table_rows() -> None:
    table_chunk = _chunk(
        content=(
            "Table row: 当前客户工单响应矩阵. 工单等级=P1; 首次响应=5 分钟内.\n"
            "Table row: 当前客户工单响应矩阵. 工单等级=P2; 首次响应=30 分钟内."
        ),
        document_title="运营审批与客户响应规范",
    )

    audit = build_evidence_audit(
        "依据《运营审批与客户响应规范》，P1 工单首次响应要求是 30 分钟内。",
        [table_chunk],
    )

    assert audit["claims"][0]["support_status"] != "supported"

def test_reliability_keeps_relevant_l4_pdf_table_row() -> None:
    l4_chunk = _chunk(
        content=(
            "Table row: PDF page 3 table 1. 准入等级=L4 高风险; "
            "触发条件=可访问生产环境、核心数据库、客户敏感字段或长期驻场; "
            "审批链路=部门负责人；法务负责人；财务负责人；信息安全负责人; "
            "复核周期=每月复核一次; 退出要求=必须有退出清单、账号回收证明和复盘记录."
        ),
        fused=0.35,
        rerank=1.4,
        chunk_index=3,
    )
    role_chunk = _chunk(
        content=(
            "Table row: PDF page 2 table 1. 角色=采购专员; "
            "主要职责=检查报价、比价、供应商资质和流程完整性."
        ),
        fused=0.55,
        rerank=1.1,
        chunk_index=2,
    )

    decision = should_abstain_from_answer(
        "L4 高风险供应商的审批链路、复核周期和退出要求是什么？",
        [l4_chunk, role_chunk],
        None,
    )

    assert decision.should_abstain is False
    assert decision.filtered_chunks
    assert "准入等级=L4 高风险" in decision.filtered_chunks[0].content


def test_reliability_keeps_relevant_data_processing_pdf_table_row() -> None:
    table_chunk = _chunk(
        content=(
            "Table row: PDF page 5 table 2. 交付类型=数据处理服务; "
            "验收材料=字段说明；脱敏方式；抽样检查结果; 验收人=数据 owner；信息安全负责人; "
            "付款前置条件=字段范围一致；无未授权字段; 归档位置=数据治理平台; 保留期限=5 年."
        ),
        fused=0.36,
        rerank=1.3,
        chunk_index=7,
    )

    decision = should_abstain_from_answer(
        "数据处理服务验收时需要哪些材料，验收人是谁，资料保留多久？",
        [table_chunk],
        None,
    )

    assert decision.should_abstain is False
    assert decision.filtered_chunks
    assert "交付类型=数据处理服务" in decision.filtered_chunks[0].content


def test_reliability_explains_conflicting_documents_in_user_message() -> None:
    supplier_doc_id = uuid4()
    export_doc_id = uuid4()
    chunk_a = _chunk(
        content=(
            "紧急采购可以先启用备用供应商的最低权限服务，必须同步业务负责人、采购经理和值班法务，"
            "并在事后补齐正式审批材料。"
        ),
        fused=0.82,
        rerank=1.0,
        chunk_index=1,
        document_title="供应商准入、合同变更与临时采购协作规范",
        document_id=supplier_doc_id,
    )
    chunk_b = _chunk(
        content=(
            "客户数据导出和临时高权限访问必须分别按各自事项审批，执行完成后统一归档并回收权限。"
        ),
        fused=0.79,
        rerank=0.98,
        chunk_index=2,
        document_title="客户数据导出与临时权限管理办法",
        document_id=export_doc_id,
    )

    decision = should_abstain_from_answer(
        "先救火时供应商接触生产系统和客户数据，谁审批、多久补材料、多久关账号？",
        [chunk_a, chunk_b],
        None,
    )

    assert decision.should_abstain is True
    assert decision.reason == "conflicting_or_ambiguous_evidence"
    assert decision.user_message
    assert "《供应商准入、合同变更与临时采购协作规范》" in decision.user_message
    assert "《客户数据导出与临时权限管理办法》" in decision.user_message
    assert "指定" in decision.user_message or "拆成" in decision.user_message


def test_reliability_does_not_abstain_when_strong_clause_evidence_is_available() -> None:
    civil_doc_id = uuid4()
    registry_doc_id = uuid4()
    civil_chunk = _chunk(
        content=(
            "条款全称：中华人民共和国民法典第五十六条\n\n"
            "个体工商户的债务，个人经营的，以个人财产承担；家庭经营的，以家庭财产承担；无法区分的，以家庭财产承担。"
        ),
        fused=0.76,
        rerank=0.82,
        document_title="中华人民共和国民法典",
        document_id=civil_doc_id,
    )
    registry_chunk = _chunk(
        content=(
            "条款全称：个体工商户条例第二条\n\n"
            "有经营能力的公民，经登记从事工商业经营的，为个体工商户。"
        ),
        fused=0.74,
        rerank=0.81,
        document_title="个体工商户条例",
        document_id=registry_doc_id,
    )

    decision = should_abstain_from_answer(
        "个体工商户营业执照上登记的经营者与实际经营者不一致的，是否承担共同责任？",
        [civil_chunk, registry_chunk],
        None,
    )

    assert decision.should_abstain is False
    assert decision.filtered_chunks


def test_reliability_preserves_each_quoted_cross_document_evidence_hint() -> None:
    left_doc_id = uuid4()
    right_doc_id = uuid4()
    left_chunk = _chunk(
        content=(
            "4、第三期股权激励平台基本情况 2023 年 10 月 13 日，发行人召开第一届董事会第二次会议，"
            "审议通过了第三期股权激励计划相关议案。"
        ),
        fused=0.42,
        rerank=0.42,
        chunk_index=12,
        document_title="苏州汇川联合动力系统股份有限公司",
        document_id=left_doc_id,
    )
    right_chunk = _chunk(
        content=(
            "九、本次发行后公司股利分配政策，公司可以采取现金、股票或者现金加股票相结合的方式分配利润，"
            "具备现金分红条件的，应当优先采用现金分红。"
        ),
        fused=0.92,
        rerank=0.92,
        chunk_index=3,
        document_title="深圳市东方嘉盛供应链股份有限公司首次公开发行 A 股股票招股说明书",
        document_id=right_doc_id,
    )
    duplicate_right_chunk = _chunk(
        content="报告期内实际股利分配情况显示，公司曾向全体股东按各自持股比例分配利润。",
        fused=0.88,
        rerank=0.88,
        chunk_index=4,
        document_title="深圳市东方嘉盛供应链股份有限公司首次公开发行 A 股股票招股说明书",
        document_id=right_doc_id,
    )

    decision = should_abstain_from_answer(
        "比较苏州汇川联合动力系统股份有限公司和深圳市东方嘉盛供应链股份有限公司两份招股与上市申报材料，"
        "分别关注“第三期股权激励平台基本情况 2023 年 10 月 13 日”和"
        "“公司可以采取现金、股票或者现金加股票相结合的方式分配利润”，各引用一处原文依据。",
        [right_chunk, duplicate_right_chunk, left_chunk],
        None,
    )

    assert decision.should_abstain is False
    assert decision.filtered_chunks
    filtered_ids = {chunk.document_id for chunk in decision.filtered_chunks}
    assert left_doc_id in filtered_ids
    assert right_doc_id in filtered_ids


def test_reliability_allows_low_score_chunks_when_two_quoted_hints_are_covered() -> None:
    left_doc_id = uuid4()
    right_doc_id = uuid4()
    left_chunk = _chunk(
        content="05 倍（发行市净率按照每股发行价格除以发行后每股净资产计算）。",
        fused=0.12,
        rerank=0.12,
        document_title="河北科力汽车装备股份有限公司上市公告书",
        document_id=left_doc_id,
    )
    right_chunk = _chunk(
        content="若上述情形发生于公司本次公开发行的新股已完成上市交易之后，公司及控股股东将按承诺回购。",
        fused=0.11,
        rerank=0.11,
        document_title="中仑新材料股份有限公司上市公告书",
        document_id=right_doc_id,
    )

    decision = should_abstain_from_answer(
        "比较河北科力汽车装备股份有限公司和中仑新材料股份有限公司两份招股与上市申报材料，"
        "分别关注“05 倍（发行市净率按照每股发行价格除以发行后每股净资产计算”和"
        "“若上述情形发生于公司本次公开发行的新股已完成上市交易之后”，各引用一处原文依据。",
        [left_chunk, right_chunk],
        None,
    )

    assert decision.should_abstain is False
    assert decision.filtered_chunks
    assert {chunk.document_id for chunk in decision.filtered_chunks} == {left_doc_id, right_doc_id}


def test_reliability_prefers_dominant_supplier_document_for_multi_clause_question() -> None:
    supplier_doc_id = uuid4()
    export_doc_id = uuid4()
    supplier_chunk_a = _chunk(
        content=(
            "Table row: PDF page 1 table 1. 采购类型=敏感采购; 典型场景=接触客户数据、生产系统、日志平台或内部工单; "
            "必须审批人=直属主管；采购经理；信息安全负责人; 处理时限=5 个工作日; "
            "最少材料=数据范围说明；安全评估表；供应商资质；访问权限清单."
        ),
        fused=0.74,
        rerank=1.42,
        chunk_index=1,
        document_title="供应商准入、合同变更与临时采购协作规范",
        document_id=supplier_doc_id,
    )
    supplier_chunk_b = _chunk(
        content=(
            "Table row: PDF page 4 table 1. 紧急场景=现有供应商系统故障导致客户服务中断; "
            "可先执行动作=启用备用供应商的最低权限服务; 必须同步对象=业务负责人；采购经理；值班法务; "
            "事后补齐材料=影响说明、审批单、访问记录和恢复计划; 账号关闭时限=服务结束后 24 小时内; "
            "禁止方式=个人网盘；私人聊天工具；明文邮件."
        ),
        fused=0.88,
        rerank=1.56,
        chunk_index=5,
        document_title="供应商准入、合同变更与临时采购协作规范",
        document_id=supplier_doc_id,
    )
    export_chunk = _chunk(
        content=(
            "所有涉及客户数据、生产环境、审计日志和高权限账号的处理，必须在工单系统中留痕，并在审批通过后按照最小必要原则执行。"
        ),
        fused=0.98,
        rerank=1.46,
        chunk_index=1,
        document_title="客户数据导出与临时权限管理办法",
        document_id=export_doc_id,
    )

    decision = should_abstain_from_answer(
        "如果为了先救火，需要临时让供应商接触生产系统、客户数据和导出文件，能不能先开最小权限顶上？谁来批、最晚多久补材料、结束后多久关账号，导出的文件又有哪些禁止发法？",
        [export_chunk, supplier_chunk_b, supplier_chunk_a],
        None,
    )

    assert decision.should_abstain is False
    assert decision.filtered_chunks
    assert all(chunk.document_id == supplier_doc_id for chunk in decision.filtered_chunks)


def test_evidence_audit_links_answer_claims_to_selected_citations() -> None:
    chunk = _chunk(
        content=(
            "客户事故响应流程：经理需要在五分钟内建立事故沟通渠道，"
            "并同步客户影响范围与恢复进展。"
        ),
        document_title="客户事故响应指南",
    )

    audit = build_evidence_audit(
        "经理需要在五分钟内建立事故沟通渠道。账号可以两小时后回收。",
        [chunk],
    )

    assert audit["claim_count"] == 2
    assert audit["supported_count"] == 1
    assert audit["unsupported_count"] == 1
    assert audit["status"] == "needs_review"
    assert audit["claims"][0]["support_status"] == "supported"
    assert audit["claims"][0]["support_score"] == 1.0
    assert audit["claims"][0]["support_citations"][0]["rank"] == 1
    assert audit["claims"][0]["support_citations"][0]["document_title"] == "客户事故响应指南"
    assert audit["claims"][1]["support_status"] == "unsupported"
