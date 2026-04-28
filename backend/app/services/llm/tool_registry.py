from __future__ import annotations

from dataclasses import dataclass

from app.schemas.llm import PublicToolName


@dataclass(frozen=True)
class ToolDefinition:
    name: PublicToolName
    description: str
    args_schema: dict[str, object]
    safety_constraints: list[str]
    requires_evidence: bool
    output_type: str


class ToolRegistry:
    def __init__(self, definitions: list[ToolDefinition]) -> None:
        self._definitions = {definition.name: definition for definition in definitions}

    def get(self, name: str | None) -> ToolDefinition | None:
        if not name:
            return None
        return self._definitions.get(name)

    def names(self) -> list[PublicToolName]:
        return list(self._definitions.keys())

    def list_definitions(self, names: list[str] | None = None) -> list[ToolDefinition]:
        if not names:
            return list(self._definitions.values())
        return [definition for name in names if (definition := self._definitions.get(name))]

    def sanitize_args(self, tool_name: str, raw_args: dict[str, object] | None) -> dict[str, object]:
        definition = self.get(tool_name)
        if definition is None or not raw_args:
            return {}
        allowed_keys = set(definition.args_schema.keys())
        sanitized: dict[str, object] = {}
        for key, value in raw_args.items():
            if key not in allowed_keys or value in (None, ""):
                continue
            sanitized[key] = value
        return sanitized


DEFAULT_TOOL_REGISTRY = ToolRegistry(
    [
        ToolDefinition(
            name="search_docs",
            description="检索当前用户有权限访问的文档内容，用于文档问答、主题问答或为后续结构化提取准备证据。",
            args_schema={
                "query": "string, required, 当前用户问题或需要检索的主题",
                "target_document": "string, optional, 指定要优先检索的文档标题",
            },
            safety_constraints=[
                "必须保留现有 ACL 过滤，只能返回当前用户有权限访问的文档内容。",
                "当 target_document 不可访问时，不得绕过权限限制。",
            ],
            requires_evidence=False,
            output_type="retrieval_result",
        ),
        ToolDefinition(
            name="compare_versions",
            description="比较某份文档的两个版本，返回差异摘要、增删改内容与影响提示。",
            args_schema={
                "target_document": "string, optional, 需要比较的文档标题",
                "from_version_ref": "string, optional, 起始版本引用，例如 v1、上一版",
                "to_version_ref": "string, optional, 目标版本引用，例如 v2、最新版",
            },
            safety_constraints=[
                "必须保留现有 ACL 过滤，只能比较当前用户可访问文档的版本。",
                "当版本不足或版本对无法解析时，应返回结构化失败 observation。",
            ],
            requires_evidence=False,
            output_type="version_compare_result",
        ),
        ToolDefinition(
            name="extract_todos",
            description="基于当前轮或上一轮已经得到的检索/版本对比证据，提取用户需要处理的待办事项。",
            args_schema={},
            safety_constraints=[
                "没有足够 grounded context 时不能强行生成待办。",
                "只能基于当前会话中已有的可追溯证据生成结构化结果。",
            ],
            requires_evidence=True,
            output_type="workflow_tasks",
        ),
        ToolDefinition(
            name="generate_weekly_report",
            description="基于当前会话中已有的 grounded 问答或对比结果生成周报草稿。",
            args_schema={},
            safety_constraints=[
                "没有足够 grounded context 时不能强行生成周报。",
                "只能基于当前会话中已有的可追溯证据生成结构化结果。",
            ],
            requires_evidence=True,
            output_type="workflow_weekly_report",
        ),
        ToolDefinition(
            name="generate_faq",
            description="基于当前会话中已有的 grounded 问答或对比结果整理 FAQ 草稿。",
            args_schema={},
            safety_constraints=[
                "没有足够 grounded context 时不能强行生成 FAQ。",
                "只能基于当前会话中已有的可追溯证据生成结构化结果。",
            ],
            requires_evidence=True,
            output_type="workflow_faq",
        ),
    ]
)
