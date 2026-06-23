from __future__ import annotations

from uuid import UUID, uuid4

from app.schemas.search import SearchResultChunk, SearchScoreBreakdown
from app.schemas.llm import RouterDecision, RouterDecisionResult
from app.services.chat.memory import ConversationMemory
from app.services.chat.generation import AnswerGenerationResult, DeterministicAnswerGenerator, OpenAIAnswerGenerator
from app.services.llm.orchestrator import CopilotOrchestrator


def _chunk(
    *,
    content: str,
    preview: str,
    paragraph_start: int = 1,
    document_id: UUID | None = None,
    document_title: str = "运营审批与客户响应规范",
    section_title: str = "数据导出审批与脱敏矩阵",
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
        section_title=section_title,
        paragraph_start=paragraph_start,
        paragraph_end=paragraph_start,
        citation_preview={
            "document_title": document_title,
            "version_number": 1,
            "chunk_id": str(uuid4()),
            "section_title": section_title,
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


def _with_score(
    chunk: SearchResultChunk,
    *,
    fused: float,
    rerank: float,
    lexical_raw: float = 0.1,
    lexical_normalized: float = 1.0,
) -> SearchResultChunk:
    return chunk.model_copy(
        update={
            "score": SearchScoreBreakdown(
                lexical_raw=lexical_raw,
                lexical_normalized=lexical_normalized,
                vector_raw=fused,
                vector_normalized=fused,
                fused=fused,
                rerank=rerank,
            )
        }
    )


def test_deterministic_answer_keeps_simple_first_response_lookup_concise() -> None:
    generator = DeterministicAnswerGenerator()
    content = (
        "Table row: 当前客户工单响应矩阵. 工单等级=P1; 首次响应=5 分钟内; "
        "升级条件=核心业务中断、无法登录、付费能力不可用或数据损坏风险; "
        "负责人=一线客服立即通知值班经理; 必须记录的信息=影响范围、事故 owner、下一次同步时间.\n"
        "Table row: 当前客户工单响应矩阵. 工单等级=P2; 首次响应=30 分钟内; "
        "升级条件=关键流程受限，但仍有临时绕过方式; 负责人=客服组长跟进.\n"
        "Table row: 历史响应口径. 问题类型=高优先级工单; 历史响应时间=10 分钟内; "
        "历史处理方式=先建沟通群，再通知值班经理; 当前状态=已被当前矩阵替代.\n"
        "Table row: 角色职责矩阵. 角色=客服组长; 主要职责=判断问题等级、协调跨团队处理、复核客户回复."
    )

    result = generator.generate(
        question="客服接到高优先级工单后，首次响应时间要求是多少？",
        retrieved_chunks=[
            _chunk(
                content=content,
                preview="当前客户工单响应矩阵。",
                section_title="当前客户工单响应矩阵",
            )
        ],
        history_lines=[],
    )

    assert result.insufficient_evidence is False
    assert result.answer_basis == "simple_table_lookup_answer"
    assert "P1 工单首次响应要求是 5 分钟内" in result.answer
    assert "10 分钟内" in result.answer
    assert "已被当前矩阵替代" in result.answer
    assert "P2 工单首次响应" not in result.answer
    assert "角色职责" not in result.answer

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


def test_deterministic_answer_can_use_relevant_chunks_from_multiple_documents() -> None:
    generator = DeterministicAnswerGenerator()
    policy_doc_id = uuid4()
    handbook_doc_id = uuid4()
    policy_chunk = _chunk(
        content=(
            "第二章 供应商临时访问。\n\n"
            "紧急故障期间，供应商可以先在最小边界内启用备用服务，但必须限定生产系统访问范围。"
        ),
        preview="供应商临时访问规则。",
        document_id=policy_doc_id,
        document_title="供应商临时访问管理办法",
    )
    handbook_chunk = _chunk(
        content=(
            "第五章 客户数据导出。\n\n"
            "涉及客户数据导出文件时，禁止通过个人网盘、私人聊天工具或明文邮件交付。"
        ),
        preview="客户数据导出交付限制。",
        document_id=handbook_doc_id,
        document_title="客户数据导出与交付手册",
    )
    weak_same_doc_chunk = _chunk(
        content="供应商年度评估应记录合作质量、合同金额和联系人变更。",
        preview="供应商年度评估。",
        document_id=policy_doc_id,
        document_title="供应商临时访问管理办法",
    )

    result = generator.generate(
        question="供应商先救火接触生产系统时能否先开最小权限，客户数据导出文件又禁止哪些发法？",
        retrieved_chunks=[policy_chunk, weak_same_doc_chunk, handbook_chunk],
        history_lines=[],
    )

    assert result.insufficient_evidence is False
    assert str(policy_chunk.chunk_id) in result.used_chunk_ids
    assert str(handbook_chunk.chunk_id) in result.used_chunk_ids
    assert "最小边界" in result.answer
    assert "个人网盘" in result.answer


def test_deterministic_answer_preserves_quoted_enterprise_evidence_coverage() -> None:
    generator = DeterministicAnswerGenerator()
    doc_id = uuid4()
    document_title = "济南水务集团有限公司2026年度第一期中期票据募集说明书"
    governance_fact = (
        "六、发行人治理结构、组织结构及内控制度 （一）公司治理结构为规范公司的组织和行为，"
        "保护公司、股东和债权人的合法权益，依法履行公司权利，承担公司义务，"
        "根据《中华人民共和国公司法》《中国共产党章程》及其有关法律、法规的规定，"
        "结合公司实际情况，制定了《济南水务集团有限公司章程》。"
    )
    executive_fact = (
        "济南水务集团有限公司2026年度第一期中期票据募集说明书计算机中心高级工程师，"
        "济南供水集团有限责任公司设备管理部部长助理，济南水业集团有限责任公司生产运营部部长助理、"
        "调度质检部副经理、副总工程师、设计院院长，济南水务集团有限公司副总工程师、设计院院长、"
        "党委委员、总工程师、董事，现任济南水务集团有限公司党委委员、副总经理。"
    )
    governance_chunk = _with_score(
        _chunk(
            content=f"{governance_fact}\n\n后续段落介绍组织架构图和内控制度执行情况。",
            preview="发行人治理结构、组织结构及内控制度。",
            document_id=doc_id,
            document_title=document_title,
            paragraph_start=39,
        ),
        fused=0.35,
        rerank=0.35,
    )
    executive_chunk = _with_score(
        _chunk(
            content=f"{executive_fact}\n\n陈峰，男，1975年7月出生，山东济南人，硕士研究生学历。",
            preview="高级管理人员履历。",
            document_id=doc_id,
            document_title=document_title,
            paragraph_start=56,
        ),
        fused=0.9,
        rerank=0.9,
    )
    duplicate_executive_chunk = _with_score(
        _chunk(
            content=f"{executive_fact}\n\n该段继续列示任职资格和职责分工。",
            preview="高级管理人员任职经历。",
            document_id=doc_id,
            document_title=document_title,
            paragraph_start=57,
        ),
        fused=0.85,
        rerank=0.85,
    )

    result = generator.generate(
        question=(
            "请同时核对济南水务集团有限公司这份融资与财务披露材料中的两个事项："
            "“公司治理结构为规范公司的组织和行为”和“济南供水集团有限责任公司设备管理部部长助理”，"
            "分别引用依据。"
        ),
        retrieved_chunks=[executive_chunk, duplicate_executive_chunk, governance_chunk],
        history_lines=[],
    )

    assert result.insufficient_evidence is False
    assert str(governance_chunk.chunk_id) in result.used_chunk_ids
    assert str(executive_chunk.chunk_id) in result.used_chunk_ids
    assert "发行人治理结构、组织结构及内控制度" in result.answer
    assert "保护公司、股东和债权人的合法权益" in result.answer
    assert "济南供水集团有限责任公司设备管理部部长助理" in result.answer
    assert "党委委员、副总经理" in result.answer
    assert "副总工程师…" not in result.answer


def test_deterministic_answer_preserves_complete_clause_facts() -> None:
    generator = DeterministicAnswerGenerator()
    content = (
        "第四十一条\n\n"
        "条款全称：供应商临时访问管理办法第四十一条\n\n"
        "供应商不得单方延长临时访问期限，但业务负责人和安全管理员共同确认生产故障尚未恢复的除外；"
        "延长期限不得超过二十四小时，且必须补充访问记录。\n\n"
        "第四十二条\n\n"
        "条款全称：供应商临时访问管理办法第四十二条\n\n"
        "供应商擅自扩大生产系统访问范围、连续两次未提交访问记录或者造成客户数据泄露风险的，"
        "系统 owner 应当要求终止临时访问并回收账号。"
    )

    result = generator.generate(
        question="供应商什么情况下不得单方延长临时访问，例外条件和延长期限是什么？",
        retrieved_chunks=[
            _chunk(
                content=content,
                preview="供应商临时访问期限和终止规则。",
                document_title="供应商临时访问管理办法",
            )
        ],
        history_lines=[],
    )

    assert result.insufficient_evidence is False
    assert "不得单方延长临时访问期限" in result.answer
    assert "生产故障尚未恢复" in result.answer
    assert "不得超过二十四小时" in result.answer


def test_deterministic_answer_prefers_direct_land_recovery_clause_over_adjacent_inheritance_clause() -> None:
    generator = DeterministicAnswerGenerator()
    recovery_chunk = _chunk(
        document_title="农村土地承包法",
        section_title="第二十七条",
        content=(
            "条款全称：农村土地承包法第二十七条\n\n"
            "承包期内，发包方不得收回承包地。国家保护进城农户的土地承包经营权。"
        ),
        preview="承包期内，发包方不得收回承包地。",
    )
    inheritance_chunk = _chunk(
        document_title="农村土地承包法",
        section_title="第三十二条",
        content=(
            "条款全称：农村土地承包法第三十二条\n\n"
            "承包人应得的承包收益，依照继承法的规定继承。"
        ),
        preview="承包收益可以继承。",
    )

    result = generator.generate(
        question="农户成员去世，村集体能否收回承包地？",
        retrieved_chunks=[inheritance_chunk, recovery_chunk],
        history_lines=[],
    )

    assert str(recovery_chunk.chunk_id) in result.used_chunk_ids
    assert "不得收回承包地" in result.answer


def test_deterministic_answer_keeps_three_direct_clauses_from_same_long_policy() -> None:
    generator = DeterministicAnswerGenerator()
    doc_id = uuid4()
    family_chunk = _chunk(
        document_id=doc_id,
        document_title="农村土地承包法",
        section_title="第十六条",
        content=(
            "条款全称：农村土地承包法第十六条\n\n"
            "家庭承包的承包方是本集体经济组织的农户。农户内家庭成员依法平等享有承包土地的各项权益。"
        ),
        preview="家庭承包的承包方是本集体经济组织的农户。",
    )
    duration_chunk = _chunk(
        document_id=doc_id,
        document_title="农村土地承包法",
        section_title="第二十一条",
        content=(
            "条款全称：农村土地承包法第二十一条\n\n"
            "耕地的承包期为三十年。草地的承包期为三十年至五十年。林地的承包期为三十年至七十年。"
        ),
        preview="耕地的承包期为三十年。",
    )
    recovery_chunk = _chunk(
        document_id=doc_id,
        document_title="农村土地承包法",
        section_title="第二十七条",
        content=(
            "条款全称：农村土地承包法第二十七条\n\n"
            "承包期内，发包方不得收回承包地。国家保护进城农户的土地承包经营权。"
        ),
        preview="承包期内，发包方不得收回承包地。",
    )

    result = generator.generate(
        question="农户成员去世，村集体能否收回承包地？",
        retrieved_chunks=[family_chunk, duration_chunk, recovery_chunk],
        history_lines=[],
    )

    assert set(result.used_chunk_ids) == {
        str(family_chunk.chunk_id),
        str(duration_chunk.chunk_id),
        str(recovery_chunk.chunk_id),
    }
    assert "本集体经济组织的农户" in result.answer
    assert "不得收回承包地" in result.answer
    assert "三十年" in result.answer


def test_generation_focus_preserves_each_quoted_enterprise_matter() -> None:
    doc_id = uuid4()
    governance_chunk = _with_score(
        _chunk(
            content="六、发行人治理结构、组织结构及内控制度 （一）公司治理结构为规范公司的组织和行为，保护公司、股东和债权人的合法权益。",
            preview="公司治理结构。",
            document_id=doc_id,
        ),
        fused=0.25,
        rerank=0.25,
    )
    executive_chunk = _with_score(
        _chunk(
            content="计算机中心高级工程师，济南供水集团有限责任公司设备管理部部长助理，现任济南水务集团有限公司党委委员、副总经理。",
            preview="管理人员履历。",
            document_id=doc_id,
        ),
        fused=0.9,
        rerank=0.9,
    )
    duplicate_executive_chunk = _with_score(
        _chunk(
            content="济南供水集团有限责任公司设备管理部部长助理，济南水业集团有限责任公司生产运营部部长助理。",
            preview="管理人员任职经历。",
            document_id=doc_id,
        ),
        fused=0.86,
        rerank=0.86,
    )

    focused = CopilotOrchestrator._focus_generation_chunks(
        "请同时核对济南水务集团有限公司这份融资与财务披露材料中的两个事项："
        "“公司治理结构为规范公司的组织和行为”和“济南供水集团有限责任公司设备管理部部长助理”，分别引用依据。",
        [executive_chunk, duplicate_executive_chunk, governance_chunk],
    )

    assert governance_chunk.chunk_id in {chunk.chunk_id for chunk in focused}
    assert executive_chunk.chunk_id in {chunk.chunk_id for chunk in focused}


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


def test_orchestrator_prefers_simple_table_lookup_fastpath_for_first_response_question() -> None:
    document_id = uuid4()
    candidate_chunks = [
        _chunk(
            content=(
                "Table row: 当前客户工单响应矩阵. 工单等级=P1; 首次响应=5 分钟内; "
                "升级条件=核心业务中断、无法登录、付费能力不可用或数据损坏风险; 负责人=一线客服立即通知值班经理.\n"
                "Table row: 当前客户工单响应矩阵. 工单等级=P2; 首次响应=30 分钟内; "
                "升级条件=关键流程受限，但仍有临时绕过方式; 负责人=客服组长跟进.\n"
                "Table row: 历史响应口径. 问题类型=高优先级工单; 历史响应时间=10 分钟内; "
                "历史处理方式=客服直接记录; 当前状态=已被当前矩阵替代."
            ),
            preview="客户工单响应矩阵。",
            document_id=document_id,
            document_title="运营审批与客户响应规范",
        )
    ]

    question = "客服接到高优先级工单后，首次响应时间要求是多少？"
    focused_chunks = CopilotOrchestrator._focus_generation_chunks(question, candidate_chunks)

    assert focused_chunks
    assert "工单等级=P1" in focused_chunks[0].content

    generation = CopilotOrchestrator._try_structured_table_fastpath(
        question=question,
        candidate_chunks=focused_chunks,
        history_lines=[],
        conversation_context=None,
        allow_low_score=False,
    )

    assert generation is not None
    assert generation.answer_basis == "simple_table_lookup_answer"
    assert "P1 工单首次响应要求是 5 分钟内" in generation.answer
    assert "10 分钟内" in generation.answer
    assert "已被当前矩阵替代" in generation.answer
    assert "P2 工单首次响应" not in generation.answer


def test_orchestrator_uses_focused_table_preview_for_citation() -> None:
    candidate_chunk = _chunk(
        content=(
            "文档说明：这里有一段很长的说明文字，用于模拟普通预览被开头内容占满的情况。"
            "系统在展示引用时应当展示真正命中答案的表格行，而不是只展示这段说明文字。\n"
            "Table row: 当前客户工单响应矩阵. 工单等级=P1; 首次响应=5 分钟内; "
            "升级条件=核心业务中断、无法登录、付费能力不可用或数据损坏风险; 负责人=一线客服立即通知值班经理.\n"
            "Table row: 当前客户工单响应矩阵. 工单等级=P2; 首次响应=30 分钟内; "
            "升级条件=关键流程受限，但仍有临时绕过方式; 负责人=客服组长跟进."
        ),
        preview="文档说明：这里有一段很长的说明文字，用于模拟普通预览被开头内容占满的情况。",
        document_title="运营审批与客户响应规范",
    )

    question = "客服接到高优先级工单后，首次响应时间要求是多少？"
    focused_chunks = CopilotOrchestrator._focus_generation_chunks(question, [candidate_chunk])
    selected = CopilotOrchestrator._select_citation_chunks(focused_chunks, [candidate_chunk.chunk_id])

    assert selected
    assert selected[0].chunk_id == candidate_chunk.chunk_id
    assert "工单等级=P1" in selected[0].preview
    assert "首次响应=5 分钟内" in selected[0].preview


def test_orchestrator_marks_table_fastpath_confidence_high() -> None:
    chunk = _with_score(
        _chunk(
            content="Table row: 当前客户工单响应矩阵. 工单等级=P1; 首次响应=5 分钟内.",
            preview="Table row: 当前客户工单响应矩阵. 工单等级=P1; 首次响应=5 分钟内.",
        ),
        fused=0.06,
        rerank=0.06,
    )
    result = AnswerGenerationResult(
        answer="依据《运营审批与客户响应规范》，P1 工单首次响应要求是 5 分钟内。",
        insufficient_evidence=False,
        evidence_conflict=False,
        used_chunk_ids=[str(chunk.chunk_id)],
        answer_basis="simple_table_lookup_answer",
        provider_name="deterministic",
        model_name="grounded-fallback-v2",
        prompt_tokens=10,
        completion_tokens=10,
        latency_ms=1,
    )

    assert CopilotOrchestrator._compute_confidence([chunk], result) == "high"

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
