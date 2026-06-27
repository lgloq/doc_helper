# P3 事实级证据体验

更新日期：`2026-06-27`

## 目标

让企业文档 RAG 的答案侧清楚展示：

- 从答案中抽取了哪些关键事实
- 每个事实是完全支撑、部分支撑，还是待核实
- 支撑该事实的引用片段来自哪里

## 当前实现

### 运行时 evidence audit

Chat answer 已携带 `message_metadata.evidence_audit`。本阶段扩展支撑引用 payload，使每条支撑引用可以包含：

- `document_title`
- `location`
- `rank`
- `evidence_excerpt`

证据评分阈值没有因为前端展示而放宽。展示层只把现有 claim-level score 所依据的最佳支撑片段显式呈现出来。

### 前端展示

前端复用 `FactEvidencePanel`：

1. Chat 页面 assistant message 下方
2. Insights trace detail 中的追踪详情

面板展示：

- 事实覆盖率
- 完全支撑 / 部分支撑 / 待核实数量
- 按答案顺序排列的事实列表
- 每个事实关联的证据片段

在 Chat 页面中，用户可以从事实卡片定位到对应引用。定位动作会滚动到来源片段并短暂高亮，帮助核对事实与引用的对应关系。

## 为什么需要

这补齐了以下两者之间的可见链路：

- 后端 claim-level evidence audit
- 前端用户对“答案中每个关键事实来自哪里”的核对

该能力强调企业文档问答中的事实支撑关系，而不是只展示泛化 citation 列表。
