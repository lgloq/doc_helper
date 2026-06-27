# P2 检索诊断体系

更新日期：`2026-06-27`

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
