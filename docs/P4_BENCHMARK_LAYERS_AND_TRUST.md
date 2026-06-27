# P4 Benchmark 分层与可信度说明

更新日期：`2026-06-27`

本文档把当前中文企业文档 RAG benchmark 正式整理为 `smoke / full / hard / latency` 四层，并把来源边界、清洗规则、题目构造规则和人工校验规则收拢到一个地方。

这套 benchmark 的定位保持不变：

- 面向**中文企业文档 RAG**
- 验证**权限感知检索、引用支撑、版本/时间推理、表格问答、多证据综合**
- 使用**公开中文企业 / 类企业材料**
- **不是**真实企业内部私有文档数据集

机器可读分层清单：

```text
backend/data/benchmark_raw/zh_enterprise/v1_benchmark_layers.json
```

生成命令：

```powershell
python scripts/build_benchmark_layer_manifest.py
```

如需按层生成独立 case subset manifest：

```powershell
python scripts/build_benchmark_layer_manifest.py --emit-layer-manifests-dir
```

## 1. 正式分层

所有层都基于同一个**评测语料池**运行：

- 评测语料文档：`102`
- 评测语料 chunks：`17557`
- 其中格式覆盖专用文档：`11`

需要区分两个口径：

- **评测语料文档 / chunks**：真实运行检索时所在的完整语料池，四层共用。
- **layer 覆盖文档 / chunks**：该层 case 实际引用到的唯一目标文档及其 chunk 总量，用来描述该层覆盖面和压力分布。

| 层级 | 用途 | cases | 回答 / 拒答 | layer 覆盖文档 | layer 覆盖 chunks | 关注指标 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `smoke` | 小规模冒烟回归，快速覆盖 6 类企业 RAG 行为 | `24` | `20 / 4` | `19` | `1358` | `pass_rate`、`permission_isolation_correct`、`answer_faithfulness`、`p95/max latency` |
| `full` | 主发布门禁，检查完整产品质量、引用和权限隔离 | `234` | `212 / 22` | `85` | `15706` | `retrieval_hit_rate`、`citation_accuracy`、`answer_faithfulness`、`permission_isolation_correct`、`overall_score` |
| `hard` | 边界难例回归，聚焦跨文档、多表格、时序/版本、拒答 | `116` | `94 / 22` | `50` | `13278` | `pass_rate`、`citation_accuracy`、`answer_faithfulness`、`failure_stage_breakdown`、`permission_isolation_correct` |
| `latency` | 稳定时延回归，集中覆盖 chunk-heavy 和多阶段检索路径 | `24` | `20 / 4` | `12` | `6794` | `p50/p95/max latency`、`stage_latency_breakdown`、`pass_rate_floor` |

### `smoke`

- 规则：每个 `case_type` 固定取 `4` 个 case。
- 选择顺序：优先低难度、低覆盖 chunk 数，再按 `case_name` 稳定排序。
- 目的：做 PR 级快速回归，不替代 `full`。

`smoke` 的六类分布固定为：

```text
single_fact / multi_evidence_same_document / multi_evidence_cross_document / table_structured / version_temporal / permission = 各 4
```

### `full`

- 规则：直接使用固定 strict verified manifest，不删难例，不改 gold，不改题目口径。
- manifest：

```text
backend/data/benchmark_raw/zh_enterprise/v1_case_manifest_strict_evidence_verified.json
```

- 这是当前对外汇报效果、产品回归和版本比较的主 benchmark。

### `hard`

- 规则：纳入全部复杂企业 RAG 行为：
  - `multi_evidence_cross_document`
  - `table_structured`
  - `version_temporal`
  - `permission`
- 不按 `difficulty == hard` 简单切，因为当前 strict manifest 里很多真正重要的企业难点属于 `medium`。

### `latency`

- 规则：每个 `case_type` 固定取 `4` 个 case。
- 选择顺序：优先高覆盖 chunk 数、更多目标文档、更多 evidence marker，再按 `case_name` 排序。
- 目的：把 latency 层和 smoke 层区分开。`smoke` 偏功能完整性，`latency` 偏性能压力和阶段耗时稳定性。

## 2. 当前 full 基线

当前 full 层的固定 strict benchmark 规模：

| 项目 | 值 |
| --- | ---: |
| 语料文档 | `102` |
| cases | `234` |
| layer 覆盖文档 | `85` |
| layer 覆盖 chunks | `15706` |
| `single_fact` | `54` |
| `multi_evidence_same_document` | `64` |
| `multi_evidence_cross_document` | `33` |
| `table_structured` | `37` |
| `version_temporal` | `24` |
| `permission` | `22` |

当前 full 层已有主结果：

- retrieval report：

```text
backend/data/eval_outputs/zh-enterprise-v1-verified234-final-rerun-local.json
```

- product report：

```text
backend/data/eval_outputs/zh-enterprise-v1-verified234-product-metrics-claim-faithfulness-local.json
```

- manifest validation：

```text
backend/data/eval_outputs/zh-enterprise-v1-verified234-final-manifest-validation-local.json
```

- ingestion quality：

```text
backend/data/eval_outputs/zh-enterprise-v1-verified234-final-ingestion-quality-local.json
```

- latency outlier audit：

```text
backend/data/eval_outputs/zh-enterprise-v1-verified234-latency-outliers-local.json
```

## 3. 数据来源边界

该 benchmark 使用的是**公开中文企业 / 类企业材料**，包括但不限于：

- 债券和中票募集说明书、发行文件、补充披露
- 上市公司治理制度、内控制度、采购制度
- IPO 招股书、申报稿、保荐或审核材料
- ESG / 可持续发展报告
- 采购平台、供应商平台规则与公告

边界要求：

- 不包装成真实企业内部知识库数据
- 不混入私有内部制度文本
- 一个源文件保留为一个 benchmark document
- 不把法条、段落、条款拆成“伪文档”刷召回

## 4. 清洗规则

清洗和入池规则来自当前脚本与质量门禁，而不是手工挑题：

1. 来源门槛
   - manifest 校验要求官方或明确 allowlist 的公开来源域名。
   - 每篇文档保留 `source_url`、`source_org`、`file_sha256`、`retrieved_at`、`doc_type`、`domain`。

2. 文档级约束
   - 一个源文件对应一个 document。
   - 文本型 effect 文档要求最小中文文本量；低文本量或 parser-only 文档不进入主效果集，而是留在 `format_coverage_only`。
   - 摄取后必须 READY、有 chunks、checksum 匹配、embedding 完整。

3. 噪声控制
   - 严格过滤登录、分享、回到顶部、ICP备案等 UI 噪声。
   - ingestion quality gate 会单独统计 strict noise、noisy chunk rate、中文密度和表格信号。

4. 口径分离
   - 主效果 benchmark 只统计企业文档 RAG 效果。
   - 格式覆盖回归单独统计，不混入主效果分数。

## 5. 题目构造规则

题目不是为了优化指标而手工拼装的简化样例题，而是从已摄取文档和 evidence marker 反推出来的。

固定规则：

1. 题目生成来源
   - 从已入库 chunk 中提取证据候选。
   - 每个 case 保留 `expected_document_ids`、`expected_evidence_markers`、`source_chunk_index`、`section/page/paragraph/table` locator。

2. case 类型
   - `single_fact`
   - `multi_evidence_same_document`
   - `multi_evidence_cross_document`
   - `table_structured`
   - `version_temporal`
   - `permission`

3. query style
   - `direct_business_question`
   - `same_document_synthesis`
   - `cross_document_comparison`
   - `table_lookup`
   - `temporal_lookup`
   - `denied_access_request`

4. 权限 case
   - 保留 `acting_user_email`
   - 校验 forbidden / inaccessible document 不进入检索、引用和答案

5. 禁止事项
   - 不为了优化指标把问题改成低歧义模板题
   - 不删除 hard case 伪造稳定性
   - 不通过改 gold、缩题、改问法掩盖真实链路问题

## 6. 人工校验与可信度规则

当前可信度来自四层校验，而不是单次跑分。

### 6.1 Manifest 校验

主报告：

```text
backend/data/eval_outputs/zh-enterprise-v1-verified234-final-manifest-validation-local.json
```

当前 strict full 结果：

- documents：`102`
- cases：`234`
- official sources：`102`
- checksum verified：`102`
- low-overlap rate：`0.278302`
- multi-evidence rate：`0.414530`
- cross-document rate：`0.141026`
- table/structured rate：`0.174528`
- version/temporal rate：`0.113208`
- evidence locator rate：`1.000000`

### 6.2 摄取质量

主报告：

```text
backend/data/eval_outputs/zh-enterprise-v1-verified234-final-ingestion-quality-local.json
```

当前 strict full 结果：

- found documents：`102 / 102`
- ready documents：`102`
- chunks：`17557`
- strict noise documents：`0`
- checksum mismatch：`0`
- embedding gap documents：`0`
- table-signal documents：`71`

### 6.3 Anchor / evidence 质量审计

主报告：

```text
backend/data/eval_outputs/zh-enterprise-v1-verified234-anchor-specificity-audit-local.json
```

当前 strict full 审计：

- `strict_exact_evidence`：`174`
- `table_key_value`：`37`
- `permission`：`22`
- `anchor_quality_review`：`1`

这里的意义是：

- broad-document-discovery 类型问题不混进 strict full gate
- 锚点不够可靠的 case 单独打标
- 后续优化时可以明确区分“检索没打到”还是“题目锚点本身不够好”

### 6.4 Source promotion / fixed manifest

当前 full 层使用固定 strict verified manifest，意味着：

- 文档来源先经过 source candidate / review / promotion 流程
- 固定 manifest 后，后续回归只在同一批文档和 case 上比较
- 新增样本或修订标注应通过新的 manifest 版本，而不是静默覆盖旧版本

## 7. 推荐使用方式

| 场景 | 推荐层 |
| --- | --- |
| 小改动、日常 PR、自检 | `smoke` |
| 发布前、重要链路改动、效果汇报 | `full` |
| 检索选择、引用、时序/表格/权限专项优化 | `hard` |
| latency budget、rewrite/probe/rerank 性能优化 | `latency` |

建议顺序：

```text
smoke -> hard or latency -> full
```

## 8. 相关文件

- layer catalog：

```text
backend/data/benchmark_raw/zh_enterprise/v1_benchmark_layers.json
```

- full manifest：

```text
backend/data/benchmark_raw/zh_enterprise/v1_case_manifest_strict_evidence_verified.json
```

- 主要脚本：

```text
scripts/build_benchmark_layer_manifest.py
scripts/build_zh_enterprise_case_manifest.py
scripts/validate_benchmark_manifest.py
scripts/report_zh_ingestion_quality.py
scripts/audit_benchmark_case_quality.py
scripts/run_benchmark_eval.py
```
