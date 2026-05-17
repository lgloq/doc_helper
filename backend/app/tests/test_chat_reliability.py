from __future__ import annotations

from uuid import UUID, uuid4

from app.schemas.search import SearchResultChunk, SearchScoreBreakdown
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
