# P5 权限审计与企业后端能力增强

## 目标

P5 面向“权限感知的企业知识库问答”，补齐普通 RAG 原型通常缺失的后台能力：

- 管理员能解释某个用户为什么能看或不能看某份文档。
- 管理员能查看某个用户当前可见的企业文档范围。
- 管理员在保存 ACL 前能分析权限变更会影响哪些启用用户。
- 用户尝试探测疑似受限文档时，系统保留独立审计 trace。

## 实现范围

### 用户可见范围

新增接口：

- `GET /api/v1/permissions/users/{user_id}/scope?limit=50`

该接口仅管理员可用，复用 `PermissionFilterBuilder._evaluate_document_access`，返回：

- 被评估用户、部门路径和祖先部门上下文。
- 可见文档总数、可管理文档总数。
- 权限来源统计：管理员、所有者、public/user/role/department ACL，并兼容旧版 team ACL。
- 前 `limit` 条可见文档及命中原因。

前端在用户管理页的编辑面板中展示“可见文档范围”，用于管理员审计某个账号当前能访问哪些企业文档。

### ACL 影响分析

新增接口：

- `POST /api/v1/permissions/documents/{document_id}/acl/impact?preview_limit=30`

请求体复用现有 `DocumentACLCreate`。接口不会写入 ACL，只模拟现有 upsert/revoke 语义，比较变更前后的权限判断：

- `affected_user_count`
- `newly_visible_user_count`
- `no_longer_visible_user_count`
- `newly_manageable_user_count`
- `no_longer_manageable_user_count`
- `users_preview`

前端在文档 ACL 表单中增加“分析影响”按钮，管理员可以在保存前看到新增、更新或撤销授权会影响多少用户。

### 越权检索审计

当检索 debug 显示 `permission_probe_early_stop_applied=true` 时，系统额外记录一条：

- `trace_type = permission_denied_retrieval`

审计 trace 包含：

- 查询文本。
- 当前用户。
- `permission_refusal_reason_code`
- `permission_refusal_reason`
- `permission_probe_target_hint`
- 可访问/不可访问目标数量。

新增接口：

- `GET /api/v1/permissions/audit/traces`

管理员可按用户查看最近的权限拒绝检索审计记录。普通用户仍只能通过 observability 接口查看自己的 trace。

## 权限拒答原因

`SearchDebugInfo` 增加：

- `permission_refusal_reason_code`
- `permission_refusal_reason`

当前覆盖两类场景：

- `permission_probe_blocked_target`：用户询问疑似受限文档，目标文档不在可访问范围内。
- `no_accessible_documents`：用户没有任何可访问文档。

这些字段进入 chat message metadata、trace metadata 和前端执行追踪，不改变 FTS + pgvector 检索架构，也不放宽任何 ACL。

## 验证重点

新增测试覆盖：

- 管理员可查看某用户可见范围。
- 非管理员不能查看权限范围。
- ACL 影响分析能识别新增可见用户。
- 权限探测早停会写入 `permission_denied_retrieval` 审计 trace。

## 残留风险

- ACL 影响分析当前按启用用户逐个模拟，适合当前演示数据和 benchmark 规模；真实大规模企业租户需要分页、后台 job 或增量索引。
- 审计 trace 当前记录在既有 `trace_logs` 表，后续如需合规留存周期、不可篡改或导出，需要单独审计表和保留策略。
- 权限拒答原因目前覆盖显式权限探测和无可访问文档；更细的“候选召回为空但可能由权限过滤导致”可在 P2 诊断体系后续深化。
