# RAG 评测指标摘要

更新日期：`2026-06-17`

本文档记录当前可复现的评测结论。主评测集用于衡量中文企业文档 RAG 效果；格式覆盖集用于验证上传、解析、切块、ACL 和检索链路是否覆盖已声明的文件后缀。两类结果分开统计。

## 主评测集

| 项目 | 值 |
| --- | ---: |
| 数据集 | `zh_enterprise_v1_seed` |
| 固定 manifest | `backend/data/benchmark_raw/zh_enterprise/v1_case_manifest_strict_evidence_verified.json` |
| 文档数量 | `102` |
| 测试问题 / cases | `234` |
| 切块数量 | `17557` |
| 平均每篇文档切块 | `172.13` |
| READY 文档 | `102 / 102` |
| embedding 缺口 | `0` |
| checksum mismatch | `0` |
| 严格噪声文档 | `0` |
| 有表格信号的文档 | `71` |

主评测集的来源格式：

| 来源格式 | 文档数量 |
| --- | ---: |
| `pdf` | `71` |
| `html_with_pdf_attachment` | `22` |
| `html` | `9` |

问题类型分布：

| case type | cases |
| --- | ---: |
| `single_fact` | `54` |
| `multi_evidence_same_document` | `64` |
| `multi_evidence_cross_document` | `33` |
| `table_structured` | `37` |
| `version_temporal` | `24` |
| `permission` | `22` |

## 检索结果

完整 `234` cases 检索评测结果：

```text
backend/data/eval_outputs/zh-enterprise-v1-verified234-final-rerun-local.json
```

| Ablation | Hit / Pass | Recall@10 | MRR | MAP@10 | NDCG@10 | Evidence Recall@10 | Evidence MRR | Permission | P95 latency | Max latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_indexed_sparse` | `233/234 = 0.9957` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `0.9979` | `0.9459` | `1.0000` | `2.2495s` | `5.8011s` |
| `indexed_sparse_only` | `233/234 = 0.9957` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `0.9979` | `0.9108` | `1.0000` | `0.8731s` | `3.7429s` |

这组指标衡量目标文档和证据进入 top-k 候选的情况。最终回答质量由端到端产品指标单独衡量。

## 四个产品指标

当前保留两份报告：

- 检索 / 证据报告：检查目标文档和证据 marker 是否进入 top-10。
- 端到端产品报告：经过 routing、检索、引用选择、答案生成和 claim-level faithfulness 评分。

端到端产品报告：

```text
backend/data/eval_outputs/zh-enterprise-v1-verified234-product-metrics-claim-faithfulness-local.json
```

复现命令：

```powershell
docker compose exec -T backend python /app/scripts/run_benchmark_eval.py --dataset zh_enterprise_v1_seed --top-k 10 --local-baseline --retrieval-ablation full_indexed_sparse --manifest /app/backend/data/benchmark_raw/zh_enterprise/v1_case_manifest_strict_evidence_verified.json --output /app/backend/data/eval_outputs/zh-enterprise-v1-verified234-product-metrics-claim-faithfulness-local.json
```

固定 `234` cases 上的产品指标：

| 原始指标名 | 当前值 | 说明 |
| --- | ---: | --- |
| `retrieval_hit_rate` | `0.9229` | 产品指标结合 title recall、precision、ranking 和 retrieved fact recall，比检索-only 的 `Recall@10=1.0000` 更严格。 |
| `citation_accuracy` | `0.8518` | 衡量最终选中的 citation title 和 evidence fact 覆盖，同时区分证据进入 top-10 与最终引用选择。 |
| `answer_faithfulness` | `0.8986` | 衡量答案里说出的 claim 是否被最终选中的引用证据支撑：`mean(max_claim_support_by_selected_evidence) - forbidden_fact_leak_rate`。 |
| `permission_isolation_correct` | `1.0000` | 受限文档和受限事实没有进入检索、引用或答案。 |

产品评测汇总：

| cases | pass | overall score | failures |
| ---: | ---: | ---: | ---: |
| `234` | `215/234 = 0.9188` | `0.9183` | `19` |

回答型和拒答型拆分：

| profile | cases | pass | retrieval | citation | faithfulness | permission |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| answer expected | `212` | `193/212 = 0.9104` | `0.9149` | `0.8364` | `0.8881` | `1.0000` |
| refusal / permission | `22` | `22/22 = 1.0000` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |

不同问题类型的产品通过率：

| case type | pass | faithfulness |
| --- | ---: | ---: |
| `multi_evidence_cross_document` | `29/33 = 0.8788` | `0.8604` |
| `multi_evidence_same_document` | `61/64 = 0.9531` | `0.9072` |
| `permission` | `22/22 = 1.0000` | `1.0000` |
| `single_fact` | `54/54 = 1.0000` | `0.9471` |
| `table_structured` | `29/37 = 0.7838` | `0.8309` |
| `version_temporal` | `20/24 = 0.8333` | `0.8306` |

未通过类型：

| failure mode | cases |
| --- | ---: |
| `retrieval_failure` | `15` |
| `citation_failure` | `3` |
| `overall_failure` | `1` |

`answer_faithfulness` 使用连续分数。当前 claim-level scorer 会抽取答案中的 claim，并逐条检查是否被最终 citation chunk 支撑。

| 分数区间 | cases |
| --- | ---: |
| exact `1.0` | `54` |
| intermediate `(0, 1)` | `165` |
| zero | `15` |
| unique score values | `135` |

指标摘要：

> 固定中文企业文档评测集包含 `102` 篇文档、`17,557` 个 chunk、`234` 个 case。检索-only 结果为 `Recall@10=1.0000`、`Evidence Recall@10=0.9979`、`Permission=1.0000`；端到端产品链路结果为 `retrieval_hit_rate=0.9229`、`citation_accuracy=0.8518`、`answer_faithfulness=0.8986`、`permission_isolation=1.0000`，通过 `215/234` 个 case。检索指标和最终回答指标分开报告。

检索指标待复核记录：

```text
expected_evidence_missing = 1
```

对应 case：

```text
zh_enterprise_v1:multi_evidence_same_document:bond_shclearing_zh_6ac991e7_t20260604_1800531:42
```

anchor 审计结果：

| metric group | cases |
| --- | ---: |
| `strict_exact_evidence` | `174` |
| `table_key_value` | `37` |
| `permission` | `22` |
| `anchor_quality_review` | `1` |

## 指标定义

- `Hit / Pass`：严格 case 级通过。目标文档要被召回，预期 evidence marker 要被覆盖，受限文档保持缺席。权限 case 的核心检查是 forbidden document 不进入结果。
- `Recall@10`：top-10 中目标文档 title 的平均召回率。
- `MRR`：第一个目标文档命中的 reciprocal rank 均值。
- `MAP@10` / `NDCG@10`：目标文档 title 的排序质量。
- `Evidence Recall@10`：top-10 chunk 中固定 evidence marker 的平均覆盖率。
- `Evidence MRR`：第一个 evidence marker 命中的 reciprocal rank 质量。
- `Permission`：权限隔离通过率。
- `Latency`：逐 case 的检索耗时。

## 格式覆盖回归

格式覆盖集用于验证上传和 parser 链路，与主 RAG 效果分数分开统计。

| 项目 | 值 |
| --- | ---: |
| 数据集 | `format_coverage_zh_parser` |
| manifest | `backend/data/benchmark_raw/format_coverage/zh_enterprise_parser_regression_manifest.json` |
| 文档数量 | `11` |
| 测试问题 / cases | `11` |
| 切块数量 | `95` |
| READY 文档 | `11 / 11` |
| image OCR 文档 | `3` |
| import errors | `0` |

已覆盖后缀：

```text
.txt / .md / .markdown / .html / .htm / .pdf / .docx / .csv / .png / .jpg / .jpeg
```

该历史格式回归未覆盖的后缀：

```text
.doc / .xls / .xlsx / .pptx
```

注：以上是 MarkItDown Office 适配器接入前的历史格式覆盖快照，只描述当次格式回归，不代表当前代码能力边界。当前代码已新增 `.xls / .xlsx / .pptx` 上传与解析支持，但单独格式覆盖 manifest 和对应指标尚未在本摘要中刷新。

格式覆盖检索回归结果：

```text
backend/data/eval_outputs/format-coverage-zh-parser-final-retrieval-smoke-local.json
```

| Ablation | Hit / Pass | Recall@10 | MRR | Evidence Recall@10 | Permission | P95 latency | Max latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_indexed_sparse` | `11/11 = 1.0000` | `1.0000` | `0.9545` | `1.0000` | `1.0000` | `0.8124s` | `0.8755s` |
| `indexed_sparse_only` | `11/11 = 1.0000` | `1.0000` | `0.6470` | `1.0000` | `1.0000` | `0.4967s` | `0.5016s` |

结论：已声明支持的后缀可以走完上传、解析、切块、ACL 可见性和检索链路；主效果 benchmark 仍以 `102` 篇文档和 `234` 个 cases 为准。

## 质量控制说明

当前公开指标由自动化评测脚本生成。端到端产品分数来自确定性的 claim-support 评分；检索、citation 和完整性诊断使用固定目标 title 与 evidence marker。

质量控制主要体现在：

- 公开中文企业 / 类企业文档来源筛选与清洗
- manifest 校验：官方来源、checksum、source metadata、ACL scope
- ingestion quality check：chunk、噪声、checksum、embedding 检查
- anchor specificity audit：`1 / 234` 个 case 标记为 `anchor_quality_review`
