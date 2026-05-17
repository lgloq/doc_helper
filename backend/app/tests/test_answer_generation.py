from __future__ import annotations

from uuid import UUID, uuid4

from app.schemas.search import SearchResultChunk, SearchScoreBreakdown
from app.schemas.llm import RouterDecision, RouterDecisionResult
from app.services.chat.memory import ConversationMemory
from app.services.chat.generation import DeterministicAnswerGenerator, OpenAIAnswerGenerator
from app.services.llm.orchestrator import CopilotOrchestrator


def _chunk(
    *,
    content: str,
    preview: str,
    paragraph_start: int = 1,
    document_id: UUID | None = None,
    document_title: str = "运营审批与客户响应规范",
) -> SearchResultChunk:
    return SearchResultChunk(
        chunk_id=uuid4(),
        document_id=document_id or uuid4(),
        document_title=document_title,
        document_version_id=uuid4(),
        version_number=1,
        chunk_index=4,
        content=content,
        preview=preview,
        section_title="数据导出审批与脱敏矩阵",
        paragraph_start=paragraph_start,
        paragraph_end=paragraph_start,
        citation_preview={
            "document_title": document_title,
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


def test_generation_focuses_relevant_pdf_table_row_before_answering() -> None:
    content = (
        "供应商准入需要根据访问范围、数据敏感度和驻场周期判断风险等级。\n"
        "Table row: PDF page 3 table 1. 准入等级=L1 低风险; 触发条件=仅访问公开资料; "
        "审批链路=直属主管; 复核周期=半年复核一次; 退出要求=关闭临时账号.\n"
        "Table row: PDF page 3 table 1. 准入等级=L2 中风险; 触发条件=访问测试环境或低敏数据; "
        "审批链路=部门负责人；采购经理; 复核周期=季度复核一次; 退出要求=提交交接记录.\n"
        "Table row: PDF page 3 table 1. 准入等级=L4 高风险; "
        "触发条件=可访问生产环境、核心数据库、客户敏感字段或长期驻场; "
        "审批链路=部门负责人；法务负责人；财务负责人；信息安全负责人; "
        "复核周期=每月复核一次; 退出要求=必须有退出清单、账号回收证明和复盘记录."
    )

    focused = CopilotOrchestrator._focus_generation_chunks(
        "L4 高风险供应商的审批链路、复核周期和退出要求是什么？",
        [
            _chunk(
                content=content,
                preview="供应商准入需要根据访问范围、数据敏感度和驻场周期判断风险等级。",
                paragraph_start=3,
            )
        ],
    )

    assert len(focused) == 1
    assert "准入等级=L4 高风险" in focused[0].content
    assert "审批链路=部门负责人；法务负责人；财务负责人；信息安全负责人" in focused[0].content
    assert "准入等级=L1" not in focused[0].content


def test_generation_focus_keeps_supporting_narrative_for_supplier_emergency_question() -> None:
    content = (
        "临时采购可以先行建立最小可用服务，但必须限定边界、保留记录，并在事后补齐正式审批。\n"
        "Table row: PDF page 4 table 1. 访问对象=数据导出文件; 允许方式=加密链接；到期自动失效; "
        "禁止方式=个人网盘；私人聊天工具；明文邮件; 有效期=最长 3 天; 回收责任人=数据 owner.\n"
        "Table row: PDF page 1 table 1. 采购类型=紧急采购; 典型场景=生产故障导致服务中断，需要临时替换供应商; "
        "必须审批人=业务负责人；采购经理；值班法务; 处理时限=4 小时内完成临时审批; "
        "最少材料=故障说明；替换原因；临时服务边界；退出计划; 备注=事后 2 个工作日内补齐正式材料."
    )

    focused = CopilotOrchestrator._focus_generation_chunks(
        "如果为了先救火，需要临时让供应商接触生产系统、客户数据和导出文件，能不能先开最小权限顶上？谁来批、最晚多久补材料、结束后多久关账号，导出的文件又有哪些禁止发法？",
        [
            _chunk(
                content=content,
                preview="紧急供应商替换与导出文件控制。",
                paragraph_start=4,
                document_title="供应商准入、合同变更与临时采购协作规范",
            )
        ],
    )

    assert len(focused) == 1
    assert "最小可用服务" in focused[0].content
    assert "禁止方式=个人网盘" in focused[0].content


def test_deterministic_answer_prefers_l4_pdf_table_row() -> None:
    generator = DeterministicAnswerGenerator()
    content = (
        "Table row: PDF page 3 table 1. 准入等级=L2 中风险; 触发条件=访问测试环境或低敏数据; "
        "审批链路=部门负责人；采购经理; 复核周期=季度复核一次; 退出要求=提交交接记录.\n"
        "Table row: PDF page 3 table 1. 准入等级=L4 高风险; "
        "触发条件=可访问生产环境、核心数据库、客户敏感字段或长期驻场; "
        "审批链路=部门负责人；法务负责人；财务负责人；信息安全负责人; "
        "复核周期=每月复核一次; 退出要求=必须有退出清单、账号回收证明和复盘记录."
    )

    result = generator.generate(
        question="L4 高风险供应商的审批链路、复核周期和退出要求是什么？",
        retrieved_chunks=[
            _chunk(
                content=content,
                preview="供应商准入风险等级表。",
                paragraph_start=3,
            )
        ],
        history_lines=[],
    )

    assert result.insufficient_evidence is False
    assert "法务负责人" in result.answer
    assert "每月复核一次" in result.answer
    assert "账号回收证明" in result.answer


def test_openai_answer_generator_falls_back_to_deterministic_on_connection_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.chat.generation.create_openai_compatible_client",
        lambda settings: object(),
    )

    def _raise(*args, **kwargs):
        raise RuntimeError("upstream unavailable")

    monkeypatch.setattr("app.services.chat.generation.request_chat_completion", _raise)

    generator = OpenAIAnswerGenerator()
    content = (
        "Table row: 数据导出审批与脱敏矩阵. 数据范围=包含客户手机号; 审批人=管理员; "
        "处理时限=2 个工作日; 脱敏要求=保留前三位和后四位."
    )

    result = generator.generate(
        question="包含客户手机号的数据导出由谁审批，处理时限和脱敏要求是什么？",
        retrieved_chunks=[_chunk(content=content, preview="数据导出审批与脱敏矩阵。")],
        history_lines=[],
    )

    assert result.provider_name == "openai-compatible-fallback"
    assert result.insufficient_evidence is False
    assert "管理员" in result.answer
    assert result.raw_payload
    assert result.raw_payload["fallback_reason"] == "upstream_answer_generation_failed"
    assert "upstream unavailable" in result.raw_payload["error_text"]


def test_openai_answer_generator_uses_structured_table_fallback_for_complex_supplier_question(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.chat.generation.create_openai_compatible_client",
        lambda settings: object(),
    )

    def _raise(*args, **kwargs):
        raise RuntimeError("upstream unavailable")

    monkeypatch.setattr("app.services.chat.generation.request_chat_completion", _raise)

    generator = OpenAIAnswerGenerator()
    supplier_doc_id = uuid4()
    retrieved_chunks = [
        _chunk(
            content=(
                "Table row: PDF page 1 table 1. 采购类型=敏感采购; 典型场景=接触客户数据、生产系统、日志平台或内部工单; "
                "必须审批人=直属主管；采购经理；信息安全负责人; 处理时限=5 个工作日; "
                "最少材料=数据范围说明；安全评估表；供应商资质；访问权限清单."
            ),
            preview="供应商敏感采购审批表。",
            document_id=supplier_doc_id,
            document_title="供应商准入、合同变更与临时采购协作规范",
        ),
            _chunk(
                content=(
                    "临时采购可以先行建立最小可用服务，但必须限定边界、保留记录，并在事后补齐正式审批。\n"
                    "Table row: PDF page 1 table 1. 采购类型=紧急采购; 典型场景=生产故障导致服务中断，需要临时替换供应商; "
                    "必须审批人=业务负责人；采购经理；值班法务; 处理时限=4 小时内完成临时审批; "
                    "最少材料=故障说明；替换原因；临时服务边界；退出计划; 备注=事后 2 个工作日内补齐正式材料.\n"
                    "Table row: PDF page 4 table 1. 访问对象=数据导出文件; 允许方式=加密链接；到期自动失效; "
                    "禁止方式=个人网盘；私人聊天工具；明文邮件; 有效期=最长 3 天; 回收责任人=数据 owner."
                ),
                preview="紧急供应商替换与导出文件控制。",
                paragraph_start=4,
                document_id=supplier_doc_id,
                document_title="供应商准入、合同变更与临时采购协作规范",
        ),
    ]

    result = generator.generate(
        question="如果为了先救火，需要临时让供应商接触生产系统、客户数据和导出文件，能不能先开最小权限顶上？谁来批、最晚多久补材料、结束后多久关账号，导出的文件又有哪些禁止发法？",
        retrieved_chunks=retrieved_chunks,
        history_lines=[],
    )

    assert result.provider_name == "openai-compatible-fallback"
    assert result.insufficient_evidence is False
    assert result.answer_basis == "structured_table_answer"
    assert "《供应商准入、合同变更与临时采购协作规范》" in result.answer
    assert "备用供应商服务" in result.answer
    assert "业务负责人" in result.answer
    assert "采购经理" in result.answer
    assert "信息安全负责人" in result.answer
    assert "4 小时内完成临时审批" in result.answer
    assert "2 个工作日内补齐正式材料" in result.answer
    assert "直属主管" in result.answer
    assert "5 个工作日" in result.answer
    assert "安全评估表" in result.answer
    assert "个人网盘" in result.answer
    assert "没有给出统一账号关闭时限" in result.answer
    assert "账号应在最长 3 天" not in result.answer
    assert result.raw_payload
    assert result.raw_payload["fallback_reason"] == "upstream_answer_generation_failed"


def test_orchestrator_prefers_structured_table_fastpath_for_complex_supplier_question() -> None:
    supplier_doc_id = uuid4()
    candidate_chunks = [
        _chunk(
            content=(
                "临时采购可以先行建立最小可用服务，但必须限定边界、保留记录，并在事后补齐正式审批。\n"
                "Table row: PDF page 4 table 1. 访问对象=数据导出文件; 允许方式=加密链接；到期自动失效; "
                "禁止方式=个人网盘；私人聊天工具；明文邮件; 有效期=最长 3 天; 回收责任人=数据 owner."
            ),
            preview="紧急供应商替换与导出文件控制。",
            paragraph_start=4,
            document_id=supplier_doc_id,
            document_title="供应商准入、合同变更与临时采购协作规范",
        ),
        _chunk(
            content=(
                "Table row: PDF page 1 table 1. 采购类型=敏感采购; 典型场景=接触客户数据、生产系统、日志平台或内部工单; "
                "必须审批人=直属主管；采购经理；信息安全负责人; 处理时限=5 个工作日; "
                "最少材料=数据范围说明；安全评估表；供应商资质；访问权限清单.\n"
                "Table row: PDF page 1 table 1. 采购类型=紧急采购; 典型场景=生产故障导致服务中断，需要临时替换供应商; "
                "必须审批人=业务负责人；采购经理；值班法务; 处理时限=4 小时内完成临时审批; "
                "最少材料=故障说明；替换原因；临时服务边界；退出计划; 备注=事后 2 个工作日内补齐正式材料."
            ),
            preview="供应商敏感采购审批表。",
            document_id=supplier_doc_id,
            document_title="供应商准入、合同变更与临时采购协作规范",
        ),
    ]

    generation = CopilotOrchestrator._try_structured_table_fastpath(
        question="如果为了先救火，需要临时让供应商接触生产系统、客户数据和导出文件，能不能先开最小权限顶上？谁来批、最晚多久补材料、结束后多久关账号，导出的文件又有哪些禁止发法？",
        candidate_chunks=candidate_chunks,
        history_lines=[],
        conversation_context=None,
        allow_low_score=False,
    )

    assert generation is not None
    assert generation.answer_basis == "structured_table_answer"
    assert "备用供应商服务" in generation.answer
    assert "采购经理" in generation.answer
    assert "安全评估表" in generation.answer
    assert "没有给出统一账号关闭时限" in generation.answer


def test_fast_grounded_summary_prefers_simple_topic_question_with_concentrated_evidence() -> None:
    shared_document_id = uuid4()
    chunks = [
        _chunk(
            content="员工手册说明节假日安排和值班协同要求。",
            preview="员工手册说明节假日安排和值班协同要求。",
            document_id=shared_document_id,
        ),
        _chunk(
            content="节假日期间如需安排值班，组长应提前同步值班安排和值班联系人。",
            preview="节假日期间如需安排值班，组长应提前同步值班安排和值班联系人。",
            paragraph_start=2,
            document_id=shared_document_id,
        ),
    ]
    router_result = RouterDecisionResult(
        decision=RouterDecision(intent="topic_qa", needs_citations=True, topic="节假日安排是什么样的？"),
        provider_name="test-router",
        model_name="test-router-model",
        prompt_tokens=None,
        completion_tokens=None,
        latency_ms=0,
        raw_payload={},
    )
    conversation_memory = ConversationMemory(
        history_lines=[],
        older_summary=None,
        previous_target_document=None,
        previous_tool_name=None,
        previous_observation_summary=None,
        previous_artifact_type=None,
        previous_intent=None,
        previous_refusal_reason=None,
        previous_insufficient_evidence=False,
        recent_message_count=0,
        older_message_count=0,
    )

    assert CopilotOrchestrator._should_prefer_fast_grounded_summary(
        question="节假日安排是什么样的？",
        router_result=router_result,
        candidate_chunks=chunks,
        conversation_memory=conversation_memory,
    )
