# P1 长任务后台化与恢复

更新日期：`2026-06-27`

## 范围

本阶段把明显耗时较长、需要刷新/切页后恢复的请求统一收敛到 `operation_jobs`：

- chat message submission: `POST /api/v1/chat/sessions/{session_id}/messages/async`
- eval runs: `POST /api/v1/eval/run/async`
- document diff summary generation: `POST /api/v1/documents/{document_id}/diff/summary/async`
- document ingestion: `POST /api/v1/documents/{document_id}/ingest/async`

统一查询端点：

```text
GET /api/v1/jobs/{job_id}
```

## Job 模型

`operation_jobs` 为每个后台操作持久化一行：

- `job_type`: `chat_message`, `eval_run`, `document_diff_summary`, `document_ingest`
- `status`: `queued`, `running`, `completed`, `failed`
- `client_request_id`: 调用方提供的幂等键
- `resource_type` / `resource_id`: 指向正在处理的主资源
- `request_payload`: 用于恢复和重放的规范化请求快照
- `result_payload`: 完成后的结果快照
- `error_text`: 前端恢复和调试使用的失败原因
- `arq_job_id`: ARQ 入队成功后的 worker job id

## 幂等与并发处理

各异步路由按以下元组复用同一条 `operation_jobs`：

- `job_type`
- `user_id`
- `client_request_id`

同一 `client_request_id` 只能用于同一份请求 payload。payload 或资源不一致时返回冲突错误，避免刷新、切页、重试或并发提交造成重复任务。

后续 code review 中补强了两个边界：

- `OperationJobService._create_job()` 返回 `(job, created)`，输掉唯一键竞争的一方只复用已有 job，不会再次 enqueue。
- eval async 在创建 `EvalRun` 后若发现已有 operation job，会删除本次未启动、无结果的 duplicate queued run，避免列表里留下永远不会执行的孤儿 run。

## 队列隔离

ARQ 队列已按任务类型拆分：

- `arq:queue:chat`
- `arq:queue:eval`
- `arq:queue:ingest`
- `arq:queue:diff`

docker compose 中对应独立 worker，避免大型 eval 或入库任务长期挤占 chat 问答。

## 前端恢复

`frontend/src/lib/pendingOperations.ts` 保存统一 pending operation 记录：

- `id` 作为稳定的 `client_request_id`
- 后端返回 operation job 后写入 `jobId`
- `jobId` 缺失时保留重放异步接口所需的请求字段

恢复流程：

1. pending operation 已有 `jobId` 时，轮询 `GET /api/v1/jobs/{job_id}`。
2. `jobId` 缺失时，用同一个 `client_request_id` 重放对应异步路由。
3. `queued` / `running` 任务继续留在 localStorage。
4. `completed` / `failed` 任务在刷新 UI 状态后移除。

以下页面支持刷新/切页后的任务恢复：

- `ChatPage.tsx`
- `InsightsPage.tsx`
- `VersionsPage.tsx`
- `DocumentsPage.tsx`

页面级缓存仅用于减少同一登录态内的切页闪烁。登录、退出、token 失效或无 token 状态都会清空页面缓存和选中会话，避免跨账号残留旧用户数据。

## 验证

- 受影响测试：`app/tests/test_eval_async_api.py app/tests/test_operation_jobs_api.py app/tests/test_async_ingest.py`
- 后端全量测试：`426 passed, 2 warnings`
- 前端 build：通过

## 说明

- chat worker 执行时允许同一 `client_request_id` 的 in-flight replay 路径。
- eval 正常路径仍先创建 `eval_runs`，再绑定 operation job；并发 duplicate run 已有清理保护。
- document ingestion 在 enqueue 失败或并发冲突时保留版本状态回滚。
- 本阶段不改变既有 FTS + pgvector 检索架构。
