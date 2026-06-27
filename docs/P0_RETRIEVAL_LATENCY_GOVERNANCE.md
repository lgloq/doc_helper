# P0 检索时延治理

日期：2026-06-24

目标：解决简单企业文档问答也走重检索路径导致响应慢的问题，同时保留权限隔离、FTS + pgvector hybrid retrieval、引用准确性和 answer faithfulness。

## 复现与瓶颈

本轮先通过 `/api/v1/search` 对简单问题和复杂跨文档问题做了 trace 采样，没有调用 chat 生成链路，避免把本地文档证据发送到外部 LLM。

| 查询类型 | 样例 | 现象 |
| --- | --- | --- |
| 简单制度问答 | `数据出境安全评估办法适用哪些情形？` | HTTP 约 6.0s；`search_total_latency_ms` 约 5.9s；主要耗时落在 structural retrieval，约 5.4s。 |
| 简单目标问答 | `工业领域数据安全能力提升实施方案的目标是什么？` | HTTP 约 3.3s；query plan probe 约 286ms；indexed sparse 约 584ms。 |
| 复杂跨文档对比 | 多文档比较并要求分别引用依据 | HTTP 约 15.2s；vector embedding、vector retrieval、subquery document evidence、neighbor context 叠加。 |

根因：

- 简单中文问题包含 `办法`、`制度`、`目标`、`适用情形` 等泛结构词时，会进入宽 structural retrieval；这类查询没有明确条款号或章节锚点，结构化检索收益低但代价高。
- query plan probe 对多候选改写一律做低成本试探，简单问题也会额外触发 FTS probe。
- LLM query rewrite 对明确目标文档的短问题没有必要；超时时间原先偏长。
- evidence sweep 属于证据补强重路径，适合多事实/跨文档/证据覆盖不足场景，不应在简单单事实问题或预算不足时继续扩大扫描。
- 前端 trace 只能看到候选数量，缺少可定位阶段耗时的字段。

## 本轮策略

后端新增可配置预算与跳过原因：

- `query_rewrite_timeout_seconds`：LLM rewrite 单次超时预算，默认 2s。
- `retrieval_query_plan_probe_simple_skip_enabled`：简单问题跳过 query plan probe。
- `retrieval_query_plan_probe_max_candidates`：候选改写超过上限时不做 probe。
- `retrieval_query_plan_probe_timeout_ms`：PostgreSQL `statement_timeout` 保护 probe。
- `retrieval_structural_simple_query_skip_enabled`：没有条款/章节锚点的简单问题跳过 structural retrieval。
- `retrieval_structural_timeout_ms`：PostgreSQL `statement_timeout` 保护 structural retrieval，超时回滚并降级为空结构化候选。
- `retrieval_document_evidence_sweep_min_remaining_budget_ms`：剩余预算不足时跳过 evidence sweep。
- `retrieval_latency_budget_simple_ms` / `retrieval_latency_budget_complex_ms`：区分简单和复杂查询的检索预算。

简单问题判定保持保守：短查询、无 query decomposition、无条款号/章节/条款全称锚点、无显式引文、无比较/分别/多事项信号才视为简单。包含 `第九条` 这类条款定位、显式文档标题锚定、多事项 `材料清单和复核结论`、跨文档对比的查询仍走相应重路径。

## Trace 字段

`SearchDebugInfo` 现在暴露：

- rewrite：`llm_rewrite_attempted`、`llm_rewrite_skipped_reason`、`llm_rewrite_latency_ms`
- permission：`permission_filter_latency_ms`
- probe：`query_plan_probe_skipped_reason`
- structural：`structural_retrieval_skipped`、`structural_retrieval_skip_reason`、`structural_retrieval_timeout`
- vector：`vector_retrieval_skip_reason`、`vector_retrieval_timeout`
- sweep：`document_evidence_sweep_skipped`、`document_evidence_sweep_skip_reason`

Chat assistant metadata 新增 `latency_breakdown`，agent steps 中补充 `router_latency_ms`、`retrieval_latency_ms`、`answer_generation_latency_ms`。前端 `ExecutionTrace` 的“检索细节”展示阶段耗时列表，包含 Query Rewrite、Query Plan Probe、Lexical Retrieval、Vector Retrieval、Evidence Sweep、Rerank 和 Search Total 等阶段。

## 保障

- 权限过滤仍在检索前执行，所有 retrieval source 仍只接收可访问 document ids。
- 没有替换 FTS + pgvector 架构；只是在现有 source 之前加预算、跳过和 PostgreSQL statement timeout。
- 条款号、章节锚点、标题锚定、多事项和跨文档问题不会被简单问题规则跳过。
- structural timeout 只对 PostgreSQL 设置 `SET LOCAL statement_timeout`，非 PostgreSQL 测试环境不执行。
- evidence sweep 默认仍关闭；显式开启时，多事项查询可继续扩展证据池。

## 验证

已运行：

```bash
docker compose exec -T backend python -m pytest app/tests/test_query_optimizer.py
docker compose exec -T backend python -m pytest app/tests/test_search_api.py
docker compose exec -T backend python -m pytest app/tests/test_chat_api.py
```

结果：

- `test_query_optimizer.py`: 18 passed, 1 warning
- `test_search_api.py`: 73 passed, 1 warning
- `test_chat_api.py`: 33 passed, 1 warning

后续建议：用 P2 检索诊断体系把 skip reason、候选召回、候选选择、引用覆盖和答案生成失败原因统一进 benchmark failure report。
