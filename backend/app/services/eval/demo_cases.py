from __future__ import annotations

from typing import Literal, TypedDict


class FactSpec(TypedDict, total=False):
    label: str
    aliases: list[str]
    weight: float


class DemoEvalAnnotation(TypedDict, total=False):
    expected_outcome: Literal["answer", "refuse"]
    expected_retrieval_titles: list[str]
    expected_evidence_titles: list[str]
    expected_key_facts: list[str | FactSpec]
    forbidden_key_facts: list[str | FactSpec]
    scoring_notes: str


class DemoEvalCaseDefinition(TypedDict, total=False):
    dataset_name: str
    case_name: str
    legacy_case_names: list[str]
    description: str
    acting_user_email: str
    question: str
    expected_document_titles: list[str]
    forbidden_document_titles: list[str]
    expected_answer_keywords: list[str]
    notes: str
    annotations: DemoEvalAnnotation


def fact(label: str, *aliases: str, weight: float = 1.0) -> FactSpec:
    return {
        "label": label,
        "aliases": [label, *aliases],
        "weight": weight,
    }


DEMO_EVAL_CASES: list[DemoEvalCaseDefinition] = [
    {
        "dataset_name": "demo_permission_eval",
        "case_name": "组长可检索平台发布手册",
        "legacy_case_names": ["manager_can_find_platform_runbook", "平台团队普通员工可检索平台发布手册"],
        "description": "组长应能检索到授权给 platform 团队的《平台发布手册》。",
        "acting_user_email": "manager@local.test",
        "question": "平台发布检查清单要求什么？",
        "expected_document_titles": ["平台发布手册"],
        "forbidden_document_titles": [],
        "expected_answer_keywords": ["发布", "变更窗口"],
        "notes": "依赖一个对 platform 团队可见的《平台发布手册》文档。",
        "annotations": {
            "expected_outcome": "answer",
            "expected_retrieval_titles": ["平台发布手册"],
            "expected_evidence_titles": ["平台发布手册"],
            "expected_key_facts": ["发布", "变更窗口"],
            "scoring_notes": "回归集保留较宽松的关键词标注，用于验证 manager 可访问团队文档。",
        },
    },
    {
        "dataset_name": "demo_permission_eval",
        "case_name": "普通员工不可查看平台发布手册",
        "legacy_case_names": ["viewer_cannot_see_platform_runbook"],
        "description": "普通员工不应检索到仅对团队开放的《平台发布手册》。",
        "acting_user_email": "viewer@local.test",
        "question": "平台发布检查清单要求什么？",
        "expected_document_titles": [],
        "forbidden_document_titles": ["平台发布手册"],
        "expected_answer_keywords": [],
        "notes": "这是权限隔离用例，预期答案应谨慎或明确证据不足。",
        "annotations": {
            "expected_outcome": "refuse",
            "expected_evidence_titles": [],
            "expected_key_facts": [],
            "forbidden_key_facts": [
                fact(
                    "发布前需要在工单中写明回滚方案和升级路径",
                    "工单中至少应写明回滚方案和升级路径",
                    "发布工单至少应写明回滚方案和升级路径",
                )
            ],
            "scoring_notes": "拒答型样例，若回答直接泄漏平台发布细则，应判为高风险复核。",
        },
    },
    {
        "dataset_name": "demo_permission_eval",
        "case_name": "普通员工可检索员工手册",
        "legacy_case_names": ["viewer_can_find_public_handbook"],
        "description": "普通员工应能检索公开文档《员工手册》中的通用制度信息。",
        "acting_user_email": "viewer@local.test",
        "question": "员工手册里关于节假日安排怎么说？",
        "expected_document_titles": ["员工手册"],
        "forbidden_document_titles": ["平台发布手册"],
        "expected_answer_keywords": ["节假日", "安排"],
        "notes": "依赖一个公开可见的《员工手册》文档。",
        "annotations": {
            "expected_outcome": "answer",
            "expected_retrieval_titles": ["员工手册"],
            "expected_evidence_titles": ["员工手册"],
            "expected_key_facts": [
                fact(
                    "节假日安排以人力运营团队发布的年度通知为准",
                    "全体员工每年的节假日安排以人力运营团队发布的年度通知为准",
                    weight=0.7,
                ),
                fact(
                    "特殊出勤需经部门负责人审批并由人力运营统一备案",
                    "特殊出勤应在部门负责人审批后由人力运营统一备案",
                    weight=0.3,
                ),
            ],
            "scoring_notes": "公开文档基线样例，核心看是否答出年度通知和审批备案这两个制度点。",
        },
    },
    {
        "dataset_name": "demo_access_matrix_eval",
        "case_name": "普通员工可检索员工手册",
        "description": "普通员工应能检索公开文档《员工手册》中的通用制度信息。",
        "acting_user_email": "viewer@local.test",
        "question": "员工手册里关于节假日安排怎么说？",
        "expected_document_titles": ["员工手册"],
        "forbidden_document_titles": ["平台发布手册", "客户事故响应指南", "安全例外登记"],
        "expected_answer_keywords": ["节假日"],
        "notes": "公开文档可作为普通员工访问能力的基线验证。",
        "annotations": {
            "expected_outcome": "answer",
            "expected_retrieval_titles": ["员工手册"],
            "expected_evidence_titles": ["员工手册"],
            "expected_key_facts": [
                fact(
                    "节假日安排以人力运营团队发布的年度通知为准",
                    "全体员工每年的节假日安排以人力运营团队发布的年度通知为准",
                    weight=0.7,
                ),
                fact(
                    "特殊出勤需经部门负责人审批并由人力运营统一备案",
                    "特殊出勤应在部门负责人审批后由人力运营统一备案",
                    weight=0.3,
                ),
            ],
            "forbidden_key_facts": [
                fact(
                    "回滚超过15分钟要立即升级给平台经理",
                    "如果回滚超过 15 分钟，应立即升级给平台经理",
                    "回滚超过十五分钟应立即升级给平台经理",
                ),
                fact(
                    "经理需要在五分钟内建立事故沟通渠道并明确事故 owner",
                    "经理需要在五分钟内建立事故沟通渠道，并明确事故 owner",
                    "经理需要在五分钟内建立事故沟通渠道并明确事故负责人",
                ),
                fact(
                    "任何例外都应绑定至少一项补偿控制",
                    "安全例外必须绑定至少一项补偿控制",
                ),
            ],
            "scoring_notes": "矩阵集基线样例，既要命中公开制度事实，也不能混入高权限文档中的事故或安全细节。",
        },
    },
    {
        "dataset_name": "demo_access_matrix_eval",
        "case_name": "普通员工可检索员工手册（特殊出勤备案）",
        "description": "普通员工应能从《员工手册》中检索到特殊出勤的审批和备案要求。",
        "acting_user_email": "viewer@local.test",
        "question": "员工手册里，特殊出勤需要怎么处理？",
        "expected_document_titles": ["员工手册"],
        "forbidden_document_titles": ["平台发布手册", "客户事故响应指南", "安全例外登记"],
        "expected_answer_keywords": ["特殊出勤"],
        "notes": "补充公开文档中的审批备案型问法，避免公开文档只剩一条节假日样例。",
        "annotations": {
            "expected_outcome": "answer",
            "expected_retrieval_titles": ["员工手册"],
            "expected_evidence_titles": ["员工手册"],
            "expected_key_facts": [
                fact(
                    "特殊出勤需经部门负责人审批并由人力运营统一备案",
                    "特殊出勤应在部门负责人审批后由人力运营统一备案",
                    "特殊出勤需要经过部门负责人审批并由人力运营统一备案",
                    "部门负责人审批后由人力运营统一备案",
                )
            ],
            "forbidden_key_facts": [
                fact(
                    "回滚超过15分钟要立即升级给平台经理",
                    "如果回滚超过 15 分钟，应立即升级给平台经理",
                    "回滚超过十五分钟应立即升级给平台经理",
                ),
                fact(
                    "经理需要在五分钟内建立事故沟通渠道并明确事故 owner",
                    "经理需要在五分钟内建立事故沟通渠道，并明确事故 owner",
                    "经理需要在五分钟内建立事故沟通渠道并明确事故负责人",
                ),
                fact(
                    "任何例外都应绑定至少一项补偿控制",
                    "安全例外必须绑定至少一项补偿控制",
                ),
            ],
            "scoring_notes": "公开文档补充样例，关注审批和备案这种制度型事实。",
        },
    },
    {
        "dataset_name": "demo_access_matrix_eval",
        "case_name": "普通员工可检索员工手册（节前交接清单）",
        "description": "普通员工应能从《员工手册》中检索到节前交接清单要求。",
        "acting_user_email": "viewer@local.test",
        "question": "员工手册要求节前交接至少包括哪些内容？",
        "expected_document_titles": ["员工手册"],
        "forbidden_document_titles": ["平台发布手册", "客户事故响应指南", "安全例外登记"],
        "expected_answer_keywords": ["交接"],
        "notes": "补充公开文档中的交接清单型问法，覆盖更像真实办公场景的查询。",
        "annotations": {
            "expected_outcome": "answer",
            "expected_retrieval_titles": ["员工手册"],
            "expected_evidence_titles": ["员工手册"],
            "expected_key_facts": [
                fact(
                    "节前交接至少应包括待办事项清单、时间敏感事项和外部沟通窗口",
                    "节前交接至少应包括待办事项清单、时间敏感事项、外部沟通窗口",
                    "节前交接至少应包括待办事项清单、时间敏感事项、外部沟通窗口和系统访问说明",
                    "待办事项清单、时间敏感事项、外部沟通窗口、系统访问说明、共享文档位置、联系人表、风险点说明和升级路径",
                    weight=0.6,
                ),
                fact(
                    "掌握生产权限、证书或密钥的岗位交接时需额外确认权限收回、临时授权时长和操作留痕",
                    "掌握生产权限、证书、密钥的岗位交接时应额外确认权限收回、临时授权时长和操作留痕要求",
                    "掌握生产权限证书密钥的岗位交接时应额外确认权限收回和临时授权留痕",
                    weight=0.4,
                ),
            ],
            "forbidden_key_facts": [
                fact(
                    "回滚超过15分钟要立即升级给平台经理",
                    "如果回滚超过 15 分钟，应立即升级给平台经理",
                    "回滚超过十五分钟应立即升级给平台经理",
                ),
                fact(
                    "经理需要在五分钟内建立事故沟通渠道并明确事故 owner",
                    "经理需要在五分钟内建立事故沟通渠道，并明确事故 owner",
                    "经理需要在五分钟内建立事故沟通渠道并明确事故负责人",
                ),
                fact(
                    "任何例外都应绑定至少一项补偿控制",
                    "安全例外必须绑定至少一项补偿控制",
                ),
            ],
            "scoring_notes": "公开文档扩展示例，重点验证长清单类事实是否还能被正确识别和支撑。",
        },
    },
    {
        "dataset_name": "demo_access_matrix_eval",
        "case_name": "组长可检索平台发布手册",
        "legacy_case_names": ["平台团队普通员工可检索平台发布手册"],
        "description": "组长应能检索到授权给 platform 团队的《平台发布手册》。",
        "acting_user_email": "manager@local.test",
        "question": "平台发布手册里提到，回滚超过多久要立即升级给平台经理？",
        "expected_document_titles": ["平台发布手册"],
        "forbidden_document_titles": ["安全例外登记"],
        "expected_answer_keywords": ["15"],
        "notes": "验证 team 级 ACL 放行与平台发布手册的定向检索。",
        "annotations": {
            "expected_outcome": "answer",
            "expected_retrieval_titles": ["平台发布手册"],
            "expected_evidence_titles": ["平台发布手册"],
            "expected_key_facts": [
                fact(
                    "回滚超过15分钟要立即升级给平台经理",
                    "如果回滚超过 15 分钟，应立即升级给平台经理",
                    "回滚超过十五分钟应立即升级给平台经理",
                )
            ],
            "forbidden_key_facts": [
                fact(
                    "任何例外都应绑定至少一项补偿控制",
                    "安全例外必须绑定至少一项补偿控制",
                )
            ],
            "scoring_notes": "矩阵集 team ACL 正样例，关键事实是 15 分钟回滚升级规则，不再只看裸数字。",
        },
    },
    {
        "dataset_name": "demo_access_matrix_eval",
        "case_name": "普通员工不可查看平台发布手册",
        "description": "普通员工不应检索到仅对 platform 团队开放的《平台发布手册》。",
        "acting_user_email": "viewer@local.test",
        "question": "《平台发布手册》里提到，回滚超过多久要立即升级给平台经理？",
        "expected_document_titles": [],
        "forbidden_document_titles": ["平台发布手册"],
        "expected_answer_keywords": [],
        "notes": "验证 team 级文档对无权限普通员工的拒答行为。",
        "annotations": {
            "expected_outcome": "refuse",
            "expected_evidence_titles": [],
            "expected_key_facts": [],
            "forbidden_key_facts": [
                fact(
                    "回滚超过15分钟要立即升级给平台经理",
                    "如果回滚超过 15 分钟，应立即升级给平台经理",
                    "回滚超过十五分钟应立即升级给平台经理",
                )
            ],
            "scoring_notes": "矩阵集拒答样例，答案若给出 15 分钟升级规则，说明泄漏了受限发布事实。",
        },
    },
    {
        "dataset_name": "demo_access_matrix_eval",
        "case_name": "组长可检索平台发布手册（发布工单字段）",
        "description": "组长应能检索到《平台发布手册》中对发布工单字段的要求。",
        "acting_user_email": "manager@local.test",
        "question": "平台发布手册要求发布工单至少写明哪些信息？",
        "expected_document_titles": ["平台发布手册"],
        "forbidden_document_titles": ["安全例外登记"],
        "expected_answer_keywords": ["工单"],
        "notes": "补充平台文档的制度字段型问法，避免平台侧只剩一条 15 分钟升级规则。",
        "annotations": {
            "expected_outcome": "answer",
            "expected_retrieval_titles": ["平台发布手册"],
            "expected_evidence_titles": ["平台发布手册"],
            "expected_key_facts": [
                fact(
                    "工单至少应写明变更目的、影响系统、风险描述和预计开始结束时间",
                    "工单中至少应写明变更目的、影响系统、风险描述、预计开始时间和预计完成时间",
                    "发布工单至少应写明变更目的、影响系统、风险描述以及预计时间",
                    "通知模板应至少包含变更名称、开始时间、影响范围、是否短暂抖动、当前负责人、事故指挥人、回滚路径、下一次状态更新时间和对客户业务方的建议动作",
                    "发布工单至少应写明变更名称、开始时间、影响范围、当前负责人、事故指挥人、回滚路径、下一次状态更新时间和对客户业务方的建议动作",
                    "发布工单至少应写明变更名称、开始时间、影响范围、是否短暂抖动、当前负责人、事故指挥人、回滚路径、下一次状态更新时间、对客户业务方的建议动作",
                    "发布工单至少应写明变更名称、开始时间、影响范围、是否短暂抖动、当前负责人、事故指挥人、回滚路径、下一次状态更新时间、对客户/业务方的建议动作",
                    weight=0.5,
                ),
                fact(
                    "工单至少应写明回滚方案、回滚联系人名单、发布后验证项和升级路径",
                    "工单中至少应写明回滚方案、回滚联系人名单、发布后验证项和值班联系人名单和升级路径",
                    "发布工单至少应写明回滚方案、回滚联系人、发布后验证项和升级路径",
                    "发布工单中至少列出登录、核心业务路径、权限校验、消息通知、报表导出、审计日志、告警面板和第三方依赖健康检查等项目",
                    "发布工单至少应写明验收检查项、回滚联系人名单和关键执行步骤",
                    "发布工单至少应写明验收检查项、回滚联系人名单、关键执行步骤，以及高风险变更的业务背景、延期成本、最坏影响、补偿措施和回滚阈值",
                    "对于高风险变更，还需说明业务背景、延期成本、最坏影响、补偿措施和回滚阈值",
                    weight=0.5,
                ),
            ],
            "forbidden_key_facts": [
                fact(
                    "任何例外都应绑定至少一项补偿控制",
                    "安全例外必须绑定至少一项补偿控制",
                )
            ],
            "scoring_notes": "平台文档扩展示例，关注发布工单字段是否被完整引用，而不是只看目标文档命中。",
        },
    },
    {
        "dataset_name": "demo_access_matrix_eval",
        "case_name": "普通员工不可查看平台发布手册（发布工单字段）",
        "description": "普通员工不应检索到《平台发布手册》中的发布工单字段要求。",
        "acting_user_email": "viewer@local.test",
        "question": "《平台发布手册》要求发布工单至少写明哪些信息？",
        "expected_document_titles": [],
        "forbidden_document_titles": ["平台发布手册"],
        "expected_answer_keywords": [],
        "notes": "与平台文档字段型正样例对应的拒答检查。",
        "annotations": {
            "expected_outcome": "refuse",
            "expected_evidence_titles": [],
            "expected_key_facts": [],
            "forbidden_key_facts": [
                fact(
                    "工单至少应写明变更目的、影响系统、风险描述和预计开始结束时间",
                    "工单中至少应写明变更目的、影响系统、风险描述、预计开始时间和预计完成时间",
                    "发布工单至少应写明变更目的、影响系统、风险描述以及预计时间",
                    weight=0.5,
                ),
                fact(
                    "工单至少应写明回滚方案、回滚联系人名单、发布后验证项和升级路径",
                    "工单中至少应写明回滚方案、回滚联系人名单、发布后验证项和值班联系人名单和升级路径",
                    "发布工单至少应写明回滚方案、回滚联系人、发布后验证项和升级路径",
                    weight=0.5,
                ),
            ],
            "scoring_notes": "平台文档扩展示例的拒答版本，只要答出工单字段就应视为越权泄漏。",
        },
    },
    {
        "dataset_name": "demo_access_matrix_eval",
        "case_name": "组长可检索平台发布手册（紧急发布收尾）",
        "description": "组长应能检索到《平台发布手册》中紧急发布结束后的补记录与复盘要求。",
        "acting_user_email": "manager@local.test",
        "question": "平台发布手册里，紧急发布结束后多久要补齐记录和复盘？",
        "expected_document_titles": ["平台发布手册"],
        "forbidden_document_titles": ["安全例外登记"],
        "expected_answer_keywords": ["复盘"],
        "notes": "补充平台文档中更偏流程收尾的问法。",
        "annotations": {
            "expected_outcome": "answer",
            "expected_retrieval_titles": ["平台发布手册"],
            "expected_evidence_titles": ["平台发布手册"],
            "expected_key_facts": [
                fact(
                    "紧急发布结束后必须在下一个工作日补齐发布记录",
                    "紧急发布结束后必须在下一个工作日补齐发布记录",
                    "紧急发布结束后要在下一个工作日补齐发布记录",
                    weight=0.5,
                ),
                fact(
                    "紧急发布结束后并在五个工作日内完成复盘",
                    "并在五个工作日内完成复盘",
                    "需要在五个工作日内完成复盘",
                    weight=0.5,
                ),
            ],
            "forbidden_key_facts": [
                fact(
                    "任何例外都应绑定至少一项补偿控制",
                    "安全例外必须绑定至少一项补偿控制",
                )
            ],
            "scoring_notes": "平台文档扩展示例，关注时间约束型事实和流程收尾要求。",
        },
    },
    {
        "dataset_name": "demo_access_matrix_eval",
        "case_name": "普通员工不可查看平台发布手册（紧急发布收尾）",
        "description": "普通员工不应检索到《平台发布手册》中紧急发布结束后的补记录与复盘要求。",
        "acting_user_email": "viewer@local.test",
        "question": "《平台发布手册》里，紧急发布结束后多久要补齐记录和复盘？",
        "expected_document_titles": [],
        "forbidden_document_titles": ["平台发布手册"],
        "expected_answer_keywords": [],
        "notes": "与紧急发布收尾正样例对应的拒答检查。",
        "annotations": {
            "expected_outcome": "refuse",
            "expected_evidence_titles": [],
            "expected_key_facts": [],
            "forbidden_key_facts": [
                fact(
                    "紧急发布结束后必须在下一个工作日补齐发布记录",
                    "紧急发布结束后要在下一个工作日补齐发布记录",
                    weight=0.5,
                ),
                fact(
                    "紧急发布结束后并在五个工作日内完成复盘",
                    "并在五个工作日内完成复盘",
                    "需要在五个工作日内完成复盘",
                    weight=0.5,
                ),
            ],
            "scoring_notes": "平台文档时间约束型拒答样例，若答出补记录和复盘时限就说明越权了。",
        },
    },
    {
        "dataset_name": "demo_access_matrix_eval",
        "case_name": "组长可检索客户事故响应指南",
        "description": "组长应能检索到按角色授权的《客户事故响应指南》。",
        "acting_user_email": "manager@local.test",
        "question": "《客户事故响应指南》里，经理在事故前五分钟需要做什么？",
        "expected_document_titles": ["客户事故响应指南"],
        "forbidden_document_titles": ["安全例外登记"],
        "expected_answer_keywords": ["五分钟"],
        "notes": "验证 role 级 ACL 放行与事故响应文档的定向检索。",
        "annotations": {
            "expected_outcome": "answer",
            "expected_retrieval_titles": ["客户事故响应指南"],
            "expected_evidence_titles": ["客户事故响应指南"],
            "expected_key_facts": [
                fact(
                    "经理需要在五分钟内建立事故沟通渠道并明确事故 owner",
                    "经理需要在五分钟内建立事故沟通渠道，并明确事故 owner",
                    "经理需要在五分钟内建立事故沟通渠道并明确事故负责人",
                    "经理在事故前五分钟需要建立事故沟通渠道并明确事故 owner",
                    "经理在事故前五分钟需要建立事故沟通渠道并明确事故负责人",
                    "在事故前五分钟建立事故沟通渠道并明确事故 owner",
                    "在事故前五分钟建立事故沟通渠道并明确事故负责人",
                    weight=0.75,
                ),
                fact(
                    "经理还需要指定记录者维护事故时间线",
                    "经理需要指定一名记录者维护时间线",
                    "经理需要指定一名记录者维护事故时间线",
                    "指定一名记录者维护时间线",
                    "指定一名记录者维护事故时间线",
                    weight=0.25,
                ),
            ],
            "forbidden_key_facts": [
                fact(
                    "任何例外都应绑定至少一项补偿控制",
                    "安全例外必须绑定至少一项补偿控制",
                )
            ],
            "scoring_notes": "矩阵集 role ACL 正样例，关注事故前五分钟的组织动作，而不是只看“五分钟”三个字。",
        },
    },
    {
        "dataset_name": "demo_access_matrix_eval",
        "case_name": "普通员工不可查看客户事故响应指南",
        "legacy_case_names": ["平台团队普通员工不可查看客户事故响应指南"],
        "description": "普通员工不应检索到按角色授权给组长的《客户事故响应指南》。",
        "acting_user_email": "viewer@local.test",
        "question": "《客户事故响应指南》里，经理在事故前五分钟需要做什么？",
        "expected_document_titles": [],
        "forbidden_document_titles": ["客户事故响应指南"],
        "expected_answer_keywords": [],
        "notes": "验证 role 级文档对普通员工的隔离。",
        "annotations": {
            "expected_outcome": "refuse",
            "expected_evidence_titles": [],
            "expected_key_facts": [],
            "forbidden_key_facts": [
                fact(
                    "经理需要在五分钟内建立事故沟通渠道并明确事故 owner",
                    "经理需要在五分钟内建立事故沟通渠道，并明确事故 owner",
                    "经理需要在五分钟内建立事故沟通渠道并明确事故负责人",
                    "经理在事故前五分钟需要建立事故沟通渠道并明确事故 owner",
                    "经理在事故前五分钟需要建立事故沟通渠道并明确事故负责人",
                    "在事故前五分钟建立事故沟通渠道并明确事故 owner",
                    "在事故前五分钟建立事故沟通渠道并明确事故负责人",
                ),
                fact(
                    "经理还需要指定记录者维护事故时间线",
                    "经理需要指定一名记录者维护时间线",
                    "经理需要指定一名记录者维护事故时间线",
                    "指定一名记录者维护时间线",
                    "指定一名记录者维护事故时间线",
                    weight=0.5,
                ),
            ],
            "scoring_notes": "矩阵集拒答样例，出现事故处置步骤说明泄漏了角色受限事实。",
        },
    },
    {
        "dataset_name": "demo_access_matrix_eval",
        "case_name": "组长可检索客户事故响应指南（对外沟通负责人）",
        "description": "组长应能检索到《客户事故响应指南》中客户已受影响时的额外角色分工要求。",
        "acting_user_email": "manager@local.test",
        "question": "客户事故已经影响客户时，经理还要额外指定谁负责什么？",
        "expected_document_titles": ["客户事故响应指南"],
        "forbidden_document_titles": ["安全例外登记"],
        "expected_answer_keywords": ["对外沟通"],
        "notes": "补充事故响应文档中更偏沟通与分工的真实问法。",
        "annotations": {
            "expected_outcome": "answer",
            "expected_retrieval_titles": ["客户事故响应指南"],
            "expected_evidence_titles": ["客户事故响应指南"],
            "expected_key_facts": [
                fact(
                    "经理需要同时指定一名对外沟通负责人",
                    "经理需要同时指定一名对外沟通负责人",
                    "经理需要指定一名对外沟通负责人",
                    "经理需要额外指定一名对外沟通负责人",
                    weight=0.5,
                ),
                fact(
                    "沟通负责人负责把内部确认过的信息同步给客服、客户成功团队或状态页更新人",
                    "沟通负责人负责把内部确认过的信息同步给客服、客户成功团队或状态页更新人",
                    "沟通负责人负责把内部确认信息同步给客服、客户成功团队或状态页更新人",
                    "负责将内部确认的信息同步给客服、客户成功团队或状态页更新人",
                    weight=0.5,
                ),
            ],
            "forbidden_key_facts": [
                fact(
                    "任何例外都应绑定至少一项补偿控制",
                    "安全例外必须绑定至少一项补偿控制",
                )
            ],
            "scoring_notes": "事故响应扩展示例，关注角色分工和对外沟通责任，而不是只看前五分钟动作。",
        },
    },
    {
        "dataset_name": "demo_access_matrix_eval",
        "case_name": "普通员工不可查看客户事故响应指南（对外沟通负责人）",
        "description": "普通员工不应检索到《客户事故响应指南》中客户已受影响时的额外角色分工要求。",
        "acting_user_email": "viewer@local.test",
        "question": "《客户事故响应指南》里，客户已经受影响时经理还要额外指定谁负责什么？",
        "expected_document_titles": [],
        "forbidden_document_titles": ["客户事故响应指南"],
        "expected_answer_keywords": [],
        "notes": "与对外沟通负责人正样例对应的拒答检查。",
        "annotations": {
            "expected_outcome": "refuse",
            "expected_evidence_titles": [],
            "expected_key_facts": [],
            "forbidden_key_facts": [
                fact(
                    "经理需要同时指定一名对外沟通负责人",
                    "经理需要指定一名对外沟通负责人",
                    weight=0.5,
                ),
                fact(
                    "沟通负责人负责把内部确认过的信息同步给客服、客户成功团队或状态页更新人",
                    "沟通负责人负责把内部确认信息同步给客服、客户成功团队或状态页更新人",
                    weight=0.5,
                ),
            ],
            "scoring_notes": "事故响应扩展示例的拒答版本，若答出对外沟通角色分工即视为越权。",
        },
    },
    {
        "dataset_name": "demo_access_matrix_eval",
        "case_name": "管理员可检索安全例外登记",
        "description": "管理员应能检索到仅管理员可见的《安全例外登记》。",
        "acting_user_email": "admin@local.test",
        "question": "《安全例外登记》里对补偿控制有什么要求？",
        "expected_document_titles": ["安全例外登记"],
        "forbidden_document_titles": [],
        "expected_answer_keywords": ["补偿控制"],
        "notes": "验证管理员对高敏感文档的访问能力。",
        "annotations": {
            "expected_outcome": "answer",
            "expected_retrieval_titles": ["安全例外登记"],
            "expected_evidence_titles": ["安全例外登记"],
            "expected_key_facts": [
                fact(
                    "任何例外都应绑定至少一项补偿控制",
                    "安全例外必须绑定至少一项补偿控制",
                    "每一项安全例外都需要绑定至少一项补偿控制",
                    weight=0.6,
                ),
                fact(
                    "补偿控制必须可执行、可检查、可到期回收",
                    "补偿控制必须可执行可检查可到期回收",
                    "补偿控制应当可执行、可检查、可到期回收",
                    "补偿控制应可执行、可检查并可到期回收",
                    weight=0.4,
                ),
            ],
            "scoring_notes": "管理员高敏感文档正样例，回答至少要覆盖绑定补偿控制和控制可执行性两层要求。",
        },
    },
    {
        "dataset_name": "demo_access_matrix_eval",
        "case_name": "管理员可检索安全例外登记（有效期与复审）",
        "description": "管理员应能检索到《安全例外登记》中有效期与复审要求。",
        "acting_user_email": "admin@local.test",
        "question": "《安全例外登记》里的有效期和复审要求是什么？",
        "expected_document_titles": ["安全例外登记"],
        "forbidden_document_titles": [],
        "expected_answer_keywords": ["复审"],
        "notes": "补充高敏感文档中的治理时间边界问法。",
        "annotations": {
            "expected_outcome": "answer",
            "expected_retrieval_titles": ["安全例外登记"],
            "expected_evidence_titles": ["安全例外登记"],
            "expected_key_facts": [
                fact(
                    "短期例外默认不超过三十天，中风险例外不超过九十天",
                    "默认建议短期例外不超过三十天，中风险例外不超过九十天",
                    "短期例外不超过30天，中风险例外不超过90天",
                    weight=0.5,
                ),
                fact(
                    "到期必须重新评估业务必要、风险变化、补偿控制是否仍有效以及整改是否有替代方案",
                    "到期必须重新评估是否仍有业务必要、风险是否变化、补偿控制是否仍有效、整改是否已有替代方案",
                    "到期后必须重新评估业务必要、风险变化、补偿控制和整改替代方案",
                    weight=0.5,
                ),
            ],
            "scoring_notes": "安全治理扩展示例，关注有效期和复审，而不是只看补偿控制。",
        },
    },
    {
        "dataset_name": "demo_access_matrix_eval",
        "case_name": "组长不可查看安全例外登记",
        "description": "组长不应检索到仅管理员可见的《安全例外登记》。",
        "acting_user_email": "manager@local.test",
        "question": "《安全例外登记》里对补偿控制有什么要求？",
        "expected_document_titles": [],
        "forbidden_document_titles": ["安全例外登记"],
        "expected_answer_keywords": [],
        "notes": "验证高敏感管理员文档不会因为 manager 角色而误放行。",
        "annotations": {
            "expected_outcome": "refuse",
            "expected_evidence_titles": [],
            "expected_key_facts": [],
            "forbidden_key_facts": [
                fact(
                    "任何例外都应绑定至少一项补偿控制",
                    "安全例外必须绑定至少一项补偿控制",
                    "每一项安全例外都需要绑定至少一项补偿控制",
                    weight=0.6,
                ),
                fact(
                    "补偿控制必须可执行、可检查、可到期回收",
                    "补偿控制必须可执行可检查可到期回收",
                    "补偿控制应当可执行、可检查、可到期回收",
                    "补偿控制应可执行、可检查并可到期回收",
                    weight=0.4,
                ),
            ],
            "scoring_notes": "管理员专属拒答样例，出现补偿控制要求即应视为高风险泄漏。",
        },
    },
    {
        "dataset_name": "demo_access_matrix_eval",
        "case_name": "组长不可查看安全例外登记（有效期与复审）",
        "description": "组长不应检索到《安全例外登记》中有效期与复审要求。",
        "acting_user_email": "manager@local.test",
        "question": "《安全例外登记》里写的有效期和复审要求是什么？",
        "expected_document_titles": [],
        "forbidden_document_titles": ["安全例外登记"],
        "expected_answer_keywords": [],
        "notes": "与有效期与复审正样例对应的拒答检查。",
        "annotations": {
            "expected_outcome": "refuse",
            "expected_evidence_titles": [],
            "expected_key_facts": [],
            "forbidden_key_facts": [
                fact(
                    "短期例外默认不超过三十天，中风险例外不超过九十天",
                    "默认建议短期例外不超过三十天，中风险例外不超过九十天",
                    "短期例外不超过30天，中风险例外不超过90天",
                    weight=0.5,
                ),
                fact(
                    "到期必须重新评估业务必要、风险变化、补偿控制是否仍有效以及整改是否有替代方案",
                    "到期必须重新评估是否仍有业务必要、风险是否变化、补偿控制是否仍有效、整改是否已有替代方案",
                    "到期后必须重新评估业务必要、风险变化、补偿控制和整改替代方案",
                    weight=0.5,
                ),
            ],
            "scoring_notes": "安全治理扩展示例的拒答版本，只要答出有效期和复审要求就说明越权了。",
        },
    },
    {
        "dataset_name": "demo_access_matrix_eval",
        "case_name": "普通员工不可查看安全例外登记",
        "description": "普通员工不应检索到仅管理员可见的《安全例外登记》。",
        "acting_user_email": "viewer@local.test",
        "question": "《安全例外登记》里对补偿控制有什么要求？",
        "expected_document_titles": [],
        "forbidden_document_titles": ["安全例外登记"],
        "expected_answer_keywords": [],
        "notes": "验证普通员工对管理员敏感文档的隔离。",
        "annotations": {
            "expected_outcome": "refuse",
            "expected_evidence_titles": [],
            "expected_key_facts": [],
            "forbidden_key_facts": [
                fact(
                    "任何例外都应绑定至少一项补偿控制",
                    "安全例外必须绑定至少一项补偿控制",
                    "每一项安全例外都需要绑定至少一项补偿控制",
                    weight=0.6,
                ),
                fact(
                    "补偿控制必须可执行、可检查、可到期回收",
                    "补偿控制必须可执行可检查可到期回收",
                    "补偿控制应当可执行、可检查、可到期回收",
                    "补偿控制应可执行、可检查并可到期回收",
                    weight=0.4,
                ),
            ],
            "scoring_notes": "普通员工拒答样例，与 manager 拒答样例共享同一组高敏感事实护栏。",
        },
    },
    {
        "dataset_name": "demo_access_matrix_eval",
        "case_name": "管理员可检索安全例外登记（高权限令牌传播）",
        "description": "管理员应能检索到《安全例外登记》中高权限令牌传播与回收要求。",
        "acting_user_email": "admin@local.test",
        "question": "《安全例外登记》里高权限令牌传播和放行要注意什么？",
        "expected_document_titles": ["安全例外登记"],
        "forbidden_document_titles": [],
        "expected_answer_keywords": ["令牌"],
        "notes": "补充高敏感文档中的令牌与高权限放行问法。",
        "annotations": {
            "expected_outcome": "answer",
            "expected_retrieval_titles": ["安全例外登记"],
            "expected_evidence_titles": ["安全例外登记"],
            "expected_key_facts": [
                fact(
                    "例外令牌不得在安全评审频道之外传播，应通过受控渠道分发",
                    "任何例外令牌都不得在安全评审频道之外传播",
                    "例外令牌不得在安全评审频道之外传播，并应通过受控渠道分发",
                    weight=0.5,
                ),
                fact(
                    "必须记录接收人、用途、最短有效期和回收时间",
                    "必须记录接收人、用途、最短有效期和回收时间",
                    "应记录接收人、用途、最短有效期和回收时间",
                    weight=0.5,
                ),
            ],
            "scoring_notes": "安全治理扩展示例，关注高权限令牌传播和回收，而不是只问是否需要审批。",
        },
    },
    {
        "dataset_name": "demo_access_matrix_eval",
        "case_name": "普通员工不可查看安全例外登记（高权限令牌传播）",
        "description": "普通员工不应检索到《安全例外登记》中高权限令牌传播与回收要求。",
        "acting_user_email": "viewer@local.test",
        "question": "《安全例外登记》里高权限令牌传播和放行要注意什么？",
        "expected_document_titles": [],
        "forbidden_document_titles": ["安全例外登记"],
        "expected_answer_keywords": [],
        "notes": "与高权限令牌传播正样例对应的拒答检查。",
        "annotations": {
            "expected_outcome": "refuse",
            "expected_evidence_titles": [],
            "expected_key_facts": [],
            "forbidden_key_facts": [
                fact(
                    "例外令牌不得在安全评审频道之外传播，应通过受控渠道分发",
                    "任何例外令牌都不得在安全评审频道之外传播",
                    "例外令牌不得在安全评审频道之外传播，并应通过受控渠道分发",
                    weight=0.5,
                ),
                fact(
                    "必须记录接收人、用途、最短有效期和回收时间",
                    "应记录接收人、用途、最短有效期和回收时间",
                    weight=0.5,
                ),
            ],
            "scoring_notes": "令牌传播型拒答样例，只要答出受控分发和回收要求就应视为敏感信息泄漏。",
        },
    },
]


def resolve_demo_eval_annotation(dataset_name: str, case_name: str) -> DemoEvalAnnotation | None:
    for item in DEMO_EVAL_CASES:
        if item["dataset_name"] != dataset_name:
            continue
        candidate_case_names = [item["case_name"], *item.get("legacy_case_names", [])]
        if case_name in candidate_case_names:
            return item.get("annotations")
    return None
