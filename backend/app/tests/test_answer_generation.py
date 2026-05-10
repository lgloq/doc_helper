from __future__ import annotations

from uuid import uuid4

from app.schemas.search import SearchResultChunk, SearchScoreBreakdown
from app.services.chat.generation import DeterministicAnswerGenerator


def _chunk(*, content: str, preview: str, paragraph_start: int = 1) -> SearchResultChunk:
    return SearchResultChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_title="运营审批与客户响应规范",
        document_version_id=uuid4(),
        version_number=1,
        chunk_index=4,
        content=content,
        preview=preview,
        section_title="数据导出审批与脱敏矩阵",
        paragraph_start=paragraph_start,
        paragraph_end=paragraph_start,
        citation_preview={
            "document_title": "运营审批与客户响应规范",
            "version_number": 1,
            "chunk_id": str(uuid4()),
            "section_title": "数据导出审批与脱敏矩阵",
            "paragraph_start": paragraph_start,
            "paragraph_end": paragraph_start,
            "preview": preview,
        },
        score=SearchScoreBreakdown(
            lexical_raw=0.1,
            lexical_normalized=1.0,
            vector_raw=0.5,
            vector_normalized=1.0,
            fused=0.6,
            rerank=1.0,
        ),
    )


def test_deterministic_answer_uses_table_row_beyond_preview() -> None:
    generator = DeterministicAnswerGenerator()
    content = (
        "数据导出申请需要根据字段敏感程度、客户范围和交付渠道判断审批人。"
        "任何导出都必须记录导出编号、审批人、生成时间、字段说明、脱敏方式、交付渠道和接收确认状态。\n\n"
        "Table row: 数据导出审批与脱敏矩阵. 数据范围=普通运营报表; 审批人=部门负责人; 处理时限=1 个工作日; "
        "脱敏要求=不包含客户身份字段时可不脱敏; 允许交付渠道=客户工单系统.\n\n"
        "Table row: 数据导出审批与脱敏矩阵. 数据范围=包含客户手机号; 审批人=管理员; 处理时限=2 个工作日; "
        "脱敏要求=保留前三位和后四位; 允许交付渠道=加密邮件."
    )

    result = generator.generate(
        question="包含客户手机号的数据导出由谁审批，处理时限和脱敏要求是什么？",
        retrieved_chunks=[
            _chunk(
                content=content,
                preview="数据导出申请需要根据字段敏感程度、客户范围和交付渠道判断审批人。",
            )
        ],
        history_lines=[],
    )

    assert result.insufficient_evidence is False
    assert "管理员" in result.answer
    assert "2 个工作日" in result.answer
    assert "保留前三位和后四位" in result.answer


def test_deterministic_answer_keeps_multiple_required_checklist_rows() -> None:
    generator = DeterministicAnswerGenerator()
    content = (
        "当本规范发生版本更新时，文档 owner 需要检查关联知识库条目、FAQ、客服模板和审批流程是否仍然有效。\n\n"
        "Table row: 版本更新检查清单. 检查项=检查 FAQ 是否引用旧口径; 是否必须=必须; 负责人=客服组长; 说明=重点检查响应时限和审批人.\n\n"
        "Table row: 版本更新检查清单. 检查项=同步数据导出审批变化; 是否必须=必须; 负责人=管理员; 说明=涉及敏感字段时必须同步.\n\n"
        "Table row: 版本更新检查清单. 检查项=更新安全例外到期动作; 是否必须=必须; 负责人=安全管理员; 说明=防止例外长期有效.\n\n"
        "Table row: 版本更新检查清单. 检查项=通知所有客户; 是否必须=非必须; 负责人=业务负责人; 说明=仅客户可见流程变化时需要."
    )

    result = generator.generate(
        question="如果制度版本发生变化，哪些检查项是必须完成的？",
        retrieved_chunks=[
            _chunk(
                content=content,
                preview="当本规范发生版本更新时，文档 owner 需要检查关联知识库条目。",
                paragraph_start=67,
            )
        ],
        history_lines=[],
    )

    assert result.insufficient_evidence is False
    assert "检查 FAQ 是否引用旧口径" in result.answer
    assert "同步数据导出审批变化" in result.answer
    assert "更新安全例外到期动作" in result.answer
    assert "通知所有客户" not in result.answer
