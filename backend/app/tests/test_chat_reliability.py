from __future__ import annotations

from uuid import uuid4

from app.schemas.search import SearchResultChunk, SearchScoreBreakdown
from app.services.chat.reliability import should_abstain_from_answer


def _chunk(*, content: str, fused: float = 0.4, rerank: float | None = None, chunk_index: int = 1) -> SearchResultChunk:
    document_title = "供应商准入、合同变更与临时采购协作规范"
    return SearchResultChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
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
