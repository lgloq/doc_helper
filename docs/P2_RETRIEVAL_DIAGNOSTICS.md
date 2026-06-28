# P2 检索诊断体系

更新日期：`2026-06-28`

## 目标

让企业文档 RAG 的失败 case 能归因到具体链路阶段，而不是只看粗粒度分数后再人工推断。

统一阶段：

- `permission_filter`：权限过滤
- `candidate_recall`：候选召回
- `candidate_selection`：候选选择
- `citation_coverage`：引用覆盖
- `answer_generation`：答案生成

## 当前记录内容

### Eval case 明细

每个 eval case 会在 `details_json.pipeline_diagnosis` 中记录：

- `status`
- `stage` / `stage_label`
- `reason_code` / `reason_label`
- `summary`
- `signals`

诊断使用现有运行时信号，不引入新的外部裁判依赖：

- expected document 与 accessible document 的对比
- retrieved title 与 cited title 的对比
- retrieval / citation / faithfulness 分项结果
- permission leak 检查结果
- retrieval debug 计数和 permission-probe early stop 信号
- unsupported answer facts / unsupported claims

### Trace metadata

`trace_metadata.pipeline_diagnosis` 会写入：

- chat trace
- eval case trace

chat trace 使用运行时启发式信号进行归因，例如：

- 当前用户无可访问文档
- 没有召回候选
- 没有选中引用
- 答案存在未支撑事实
- 生成阶段失败

eval trace 直接使用 eval case 中更完整的诊断结果。

### 失败聚合

Eval 前端失败行会优先展示 `pipeline_diagnosis.reason_code`，并同时暴露：

- `stage`
- `stage_label`

Benchmark eval report 会在每个 failure case 中保留 diagnosis，并汇总：

- `failure_mode_counts`：按 `reason_code` 聚合
- `failure_stage_counts`：按失败阶段聚合

## 建议排查流程

1. 先打开失败 case。
2. 先看 `pipeline_diagnosis`，确认失败阶段。
3. 再结合 retrieval debug、citation 和 evidence audit 判断具体原因：
   - ACL 可见范围是否正确
   - 候选召回是否漏掉目标文档或目标 chunk
   - rerank / final top-k 是否选错候选
   - citation 是否覆盖答案事实
   - answer generation 是否输出未支撑结论

该体系的目的不是调整分数口径，而是把后续优化建立在可解释的链路证据上。

## 近期补充：条文题与表格题分流

在 `zh_enterprise_real` 的剩余失败分析里，出现了一类稳定模式：

- 文档级召回已经命中正确制度或法规
- 但长 PDF / 长办法中的表格行把条文段落挤掉
- 最终表现为 `candidate_selection`、`citation_coverage` 或 `answer_generation` 失败

这类 case 不能简单归结为“没检到文档”，而要继续看文档内候选和生成摘要是否跑偏。

当前链路已经新增一条更细的经验规则：

- 明确表结构问题继续走 table fastpath
- 法规 / 办法 / 条文型问题优先条文块和内联 `第X条`
- 对只有表格行、没有条文信号的 chunk，在条文型问题里降权

这里优先依赖的是结构信号，而不是 benchmark case 的专有词：

- 问题里是否出现法规/条文来源信号，如 `条例 / 办法 / 规定 / 第X条`
- 问题是否更像字段型问法，如 `负责人 / 时限 / 周期 / 方式 / 等级`
- chunk 是否包含明显条文结构
- chunk 是否主要由 `Table row:` 组成

后续排查这类 case 时，应优先确认：

1. 问题是否其实是条文题，却被表格 fastpath 抢走。
2. 正确条文是否已在 retrieved chunks 中，但被 chunk ranking 或 summary 阶段压下去。
3. 一个 chunk 内是否同时包含多条条文，但摘要只保留了第一条，导致“分别/同时”类问题丢事实。
