# 中文企业文档 RAG 评测证据包

状态：`2026-06-17`

本文档记录当前中文企业文档 RAG 评测集、质量校验、指标结果和复现命令。该评测集用于验证本仓库的权限感知 RAG 链路，覆盖文档检索、证据召回、引用选择、答案支撑和权限隔离等关键环节。

正式的 `smoke / full / hard / latency` 分层定义、case 规模和可信度规则见：

```text
docs/P4_BENCHMARK_LAYERS_AND_TRUST.md
backend/data/benchmark_raw/zh_enterprise/v1_benchmark_layers.json
```

简版指标见：

```text
docs/RAG_BENCHMARK_METRICS_SUMMARY.md
```

## 评测范围

项目目标是企业文档助手，重点验证：

- 权限感知检索
- citation grounding
- 版本和时间点查询
- 表格字段查询
- 同文档多证据综合
- 跨文档多证据综合

评测源文档来自公开中文企业 / 类企业材料，包括债券和中票披露文件、上市公司和 IPO 披露文件、ESG 报告、内控和采购制度、采购平台规则，以及用于检索区分度验证的非目标主题文档。

## 固定数据集

主 manifest：

```text
backend/data/benchmark_raw/zh_enterprise/v1_case_manifest_strict_evidence_verified.json
```

数据集名称：

```text
zh_enterprise_v1_seed
```

当前固定规模：

| 项目 | 数量 |
| --- | ---: |
| 文档 | `102` |
| cases | `234` |
| 回答型 cases | `212` |
| 权限 / 拒答 cases | `22` |
| 切块 | `17557` |
| READY 文档 | `102 / 102` |

主评测集的来源格式：

| 来源格式 | 文档 |
| --- | ---: |
| `pdf` | `71` |
| `html_with_pdf_attachment` | `22` |
| `html` | `9` |

问题类型：

| case type | 数量 |
| --- | ---: |
| `single_fact` | `54` |
| `multi_evidence_same_document` | `64` |
| `multi_evidence_cross_document` | `33` |
| `table_structured` | `37` |
| `version_temporal` | `24` |
| `permission` | `22` |

## 数据来源与构建方式

评测集基于公开资料构建。每篇源文档在 manifest 中记录来源平台、来源组织、本地路径、SHA-256 checksum、来源格式、benchmark role、domain 和 ACL scope。

测试问题基于已摄取文档中的 chunk 和 evidence marker 构建。每个 case 保留目标文档 title 和预期 evidence marker，并通过 manifest 校验确认来源文件、checksum 和证据定位仍然有效。

这套构建方式让评测集更易复现，也便于在后续链路调整时定位指标变化原因。

相关脚本：

```text
scripts/build_enterprise_source_seed_manifest.py
scripts/download_enterprise_source_candidates.py
scripts/validate_enterprise_source_candidates.py
scripts/build_zh_enterprise_benchmark.py
scripts/validate_benchmark_manifest.py
scripts/import_benchmark_dataset.py
scripts/report_zh_ingestion_quality.py
scripts/run_retrieval_ablation_benchmark.py
scripts/run_benchmark_eval.py
```

当前 manifest 作为固定回归集使用。后续如需调整样本或标注，应通过新的 manifest 版本记录变更。

## 格式覆盖边界

主 RAG 效果评测覆盖的是：

```text
PDF / HTML / HTML-with-PDF
```

产品上传和 parser 链路支持更宽：

```text
TXT / Markdown / HTML / PDF / DOCX / CSV / PNG / JPG / JPEG
```

这两件事分开报告：

- 主效果 benchmark：`102` 篇 PDF / HTML / HTML-with-PDF 企业文档，`234` 个 cases。
- 格式覆盖回归：`11` 个文档，覆盖当前声明支持的文件后缀。
- 全格式覆盖结果用于 parser 和检索链路回归，主效果 benchmark 仍以 `102` 篇文档和 `234` 个 cases 为准。

格式覆盖 manifest：

```text
backend/data/benchmark_raw/format_coverage/zh_enterprise_parser_regression_manifest.json
```

格式覆盖状态：

> 注：本节是 MarkItDown Office 适配器接入前的历史格式覆盖快照，只描述当次格式回归，不代表当前代码能力边界。当前代码已新增 `.xls / .xlsx / .pptx` 上传与解析支持，但单独格式覆盖 manifest 尚未在本证据包中刷新。

| 检查项 | 值 |
| --- | --- |
| 上传 / import / parser 声明一致 | `true` |
| 支持后缀数量 | `11` |
| 支持后缀 | `.csv`, `.docx`, `.htm`, `.html`, `.jpeg`, `.jpg`, `.markdown`, `.md`, `.pdf`, `.png`, `.txt` |
| 该历史快照未覆盖的后缀 | `.doc`, `.xls`, `.xlsx`, `.pptx` |
| 单独 `format_coverage` manifest | `true` |
| 文档 / cases | `11 / 11` |
| 覆盖全部已声明后缀 | `true` |

格式覆盖检索回归：

```text
backend/data/eval_outputs/format-coverage-zh-parser-final-retrieval-smoke-local.json
```

| Ablation | Pass | Recall@10 | Evidence Recall@10 | Permission | P95 latency | Max latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_indexed_sparse` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `0.8124s` | `0.8755s` |
| `indexed_sparse_only` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `0.4967s` | `0.5016s` |

## 质量校验

### Manifest 校验

```powershell
docker compose exec -T backend python /app/scripts/validate_benchmark_manifest.py --manifest /app/backend/data/benchmark_raw/zh_enterprise/v1_case_manifest_strict_evidence_verified.json --min-documents 80 --min-cases 200 --min-low-overlap-rate 0.0 --min-multi-evidence-rate 0.25 --min-cross-document-rate 0.10 --min-permission-cases 20 --min-table-structured-rate 0.10 --min-version-temporal-rate 0.05 --max-single-fact-rate 0.35 --output /app/backend/data/eval_outputs/zh-enterprise-v1-verified234-final-manifest-validation-local.json --markdown-output /app/backend/data/eval_outputs/zh-enterprise-v1-verified234-final-manifest-validation-local.md
```

结果：

| 检查项 | 值 |
| --- | ---: |
| passed | `true` |
| documents | `102` |
| cases | `234` |
| official sources | `102` |
| checksum verified | `102` |
| low-overlap rate | `0.278302` |
| multi-evidence rate | `0.414530` |
| cross-document rate | `0.141026` |
| table/structured rate | `0.174528` |
| version/temporal rate | `0.113208` |
| evidence locator rate | `1.000000` |
| errors | `0` |
| warnings | `0` |

### 摄取质量

```powershell
docker compose exec -T backend python /app/scripts/report_zh_ingestion_quality.py --manifest /app/backend/data/benchmark_raw/zh_enterprise/v1_case_manifest_strict_evidence_verified.json --title-prefix "zh_enterprise_v1_seed:%" --require-embeddings --min-table-signal-doc-rate 0.2 --output /app/backend/data/eval_outputs/zh-enterprise-v1-verified234-final-ingestion-quality-local.json --quiet --fail-on-gate
```

结果：

| 检查项 | 值 |
| --- | ---: |
| passed | `true` |
| found documents | `102 / 102` |
| ready documents | `102` |
| failed documents | `0` |
| chunks | `17557` |
| strict noise documents | `0` |
| checksum mismatch | `0` |
| manifest metadata gaps | `0` |
| embedding gap documents | `0` |
| table-signal documents | `71` |
| table-signal document rate | `0.6961` |
| average chunks per document | `172.13` |

该检查用于确认摄取后的文档数量、chunk、噪声过滤、checksum、metadata、embedding 和表格信号符合评测要求。

### Evidence anchor 审计

```powershell
docker compose exec -T backend python /app/scripts/audit_benchmark_case_quality.py --manifest /app/backend/data/benchmark_raw/zh_enterprise/v1_case_manifest_strict_evidence_verified.json --output /app/backend/data/eval_outputs/zh-enterprise-v1-verified234-anchor-specificity-audit-local.json --markdown-output /app/backend/data/eval_outputs/zh-enterprise-v1-verified234-anchor-specificity-audit-local.md
```

结果：

| metric group | cases |
| --- | ---: |
| `strict_exact_evidence` | `174` |
| `table_key_value` | `37` |
| `permission` | `22` |
| `anchor_quality_review` | `1` |

需要人工复核 anchor 质量的 case：

```text
zh_enterprise_v1:multi_evidence_same_document:bond_shclearing_zh_6ac991e7_t20260604_1800531:42
```

## 当前结果

### 端到端产品指标

```text
backend/data/eval_outputs/zh-enterprise-v1-verified234-product-metrics-claim-faithfulness-local.json
```

| 指标 | 值 |
| --- | ---: |
| cases | `234` |
| pass | `215/234 = 0.9188` |
| `retrieval_hit_rate_avg` | `0.9229` |
| `citation_accuracy_avg` | `0.8518` |
| `answer_faithfulness_avg` | `0.8986` |
| `permission_isolation_pass_rate` | `1.0000` |
| `overall_score_avg` | `0.9183` |

`answer_faithfulness` 衡量答案中 claim 被最终选中 citation chunk 支撑的程度。完整性相关信息保留在诊断字段中单独查看。

### 检索-only 指标

```text
backend/data/eval_outputs/zh-enterprise-v1-verified234-final-rerun-local.json
```

| Ablation | Pass | Recall@10 | Evidence Recall@10 | Permission | P95 latency | Max latency | Failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `full_indexed_sparse` | `0.9957` | `1.0000` | `0.9979` | `1.0000` | `2.2495s` | `5.8011s` | `expected_evidence_missing=1` |
| `indexed_sparse_only` | `0.9957` | `1.0000` | `0.9979` | `1.0000` | `0.8731s` | `3.7429s` | `expected_evidence_missing=1` |

回归验证记录：

```powershell
docker compose exec -T backend python -m pytest app/tests/test_search_api.py app/tests/test_stard_benchmark_scripts.py app/tests/test_retrieval_diagnostics_script.py -q
```

| 验证项 | 覆盖范围 | 结果 |
| --- | --- | --- |
| Focused retrieval regression | Search API、benchmark 脚本、retrieval diagnostics | `115 passed, 1 warning` |
| Full backend regression | 后端全量 pytest | `381 passed, 2 warnings` |
| Frontend production build | 前端生产构建 | `passed` |

## 指标定义

- `Pass`：case 级严格通过；目标文档被召回、预期 evidence marker 被覆盖、受限文档没有出现。
- `Recall@10`：top-10 中目标文档 title 的平均召回率。
- `Evidence Recall@10`：top-10 chunk 中预期 evidence marker 的平均覆盖率。
- `Permission`：权限隔离正确率。
- `P95 latency` / `Max latency`：逐 case 检索耗时。
- `retrieval_hit_rate_avg`：端到端产品链路中的检索命中质量。
- `citation_accuracy_avg`：最终选中 citation 的 title 与 evidence 支撑质量。
- `answer_faithfulness_avg`：答案 claim 被最终 citation 支撑的程度。
- `permission_isolation_pass_rate`：检索、引用和答案中是否没有泄漏受限内容。

## 链路优化记录

优化对象集中在检索和证据进入候选池的链路。

- 长问题的确定性 query decomposition
- 每个 subquery 独立召回后做 coverage-aware merge
- 避免通用同文档 chunk 替换已经覆盖的子问题证据
- 表格 key/value evidence 的 source fast path
- same-document 场景下减少重复 lexical fanout
- 具体 evidence anchor 的 source fast path，并过滤低信息量 anchor

这些改动保留了 `FTS + pgvector / hybrid retrieval` 架构，也没有依赖垂直关键词表或外部 LLM rewrite。

## 评测约束

当前结果基于以下约束：

- 固定 `v1_case_manifest_strict_evidence_verified.json`
- 样本集合和 evidence marker 保持稳定
- 评分逻辑保持稳定
- 默认配置使用本地 heuristic rerank 和确定性 query decomposition
- 检索架构保持权限感知的 `FTS + pgvector / hybrid retrieval`
- exact-anchor fast path 会过滤宽泛日期这类低信息量 anchor

## 待复核样本

当前检索指标中仍有一个待复核 case：

```text
zh_enterprise_v1:multi_evidence_same_document:bond_shclearing_zh_6ac991e7_t20260604_1800531:42
```

记录状态：

```text
expected_evidence_missing
```

说明：

- 问题中的 anchor 较宽泛：`截至 2025 年末`。
- 缺失的 evidence marker 需要更具体的区分信息：`在建项目收入确认情况正常`。
- exact-anchor source path 会过滤这种宽泛日期 anchor。

后续如需处理，应通过新的 manifest 版本记录 anchor 调整。

## 复现命令

主评测 manifest 校验：

```powershell
docker compose exec -T backend python /app/scripts/validate_benchmark_manifest.py --manifest /app/backend/data/benchmark_raw/zh_enterprise/v1_case_manifest_strict_evidence_verified.json --min-documents 80 --min-cases 200 --min-low-overlap-rate 0.0 --min-multi-evidence-rate 0.25 --min-cross-document-rate 0.10 --min-permission-cases 20 --min-table-structured-rate 0.10 --min-version-temporal-rate 0.05 --max-single-fact-rate 0.35 --output /app/backend/data/eval_outputs/zh-enterprise-v1-verified234-final-manifest-validation-local.json --markdown-output /app/backend/data/eval_outputs/zh-enterprise-v1-verified234-final-manifest-validation-local.md
```

摄取质量检查：

```powershell
docker compose exec -T backend python /app/scripts/report_zh_ingestion_quality.py --manifest /app/backend/data/benchmark_raw/zh_enterprise/v1_case_manifest_strict_evidence_verified.json --title-prefix "zh_enterprise_v1_seed:%" --require-embeddings --min-table-signal-doc-rate 0.2 --output /app/backend/data/eval_outputs/zh-enterprise-v1-verified234-final-ingestion-quality-local.json --quiet --fail-on-gate
```

完整 `234` cases 检索评测：

```powershell
docker compose exec -T backend python /app/scripts/run_retrieval_ablation_benchmark.py --dataset zh_enterprise_v1_seed --manifest /app/backend/data/benchmark_raw/zh_enterprise/v1_case_manifest_strict_evidence_verified.json --balanced-limit 1000 --per-type-limit 1000 --ablations full_indexed_sparse,indexed_sparse_only --case-statement-timeout-ms 10000 --output /app/backend/data/eval_outputs/zh-enterprise-v1-verified234-final-rerun-local.json --markdown-output /app/backend/data/eval_outputs/zh-enterprise-v1-verified234-final-rerun-local.md --progress
```

端到端产品指标：

```powershell
docker compose exec -T backend python /app/scripts/run_benchmark_eval.py --dataset zh_enterprise_v1_seed --top-k 10 --local-baseline --retrieval-ablation full_indexed_sparse --manifest /app/backend/data/benchmark_raw/zh_enterprise/v1_case_manifest_strict_evidence_verified.json --output /app/backend/data/eval_outputs/zh-enterprise-v1-verified234-product-metrics-claim-faithfulness-local.json
```

格式覆盖集构建和评测：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/render_format_coverage_images.ps1 -RootDir .
python scripts/build_zh_format_coverage_manifest.py
docker compose exec -T backend python /app/scripts/import_benchmark_dataset.py --dry-run manifest --manifest /app/backend/data/benchmark_raw/format_coverage/zh_enterprise_parser_regression_manifest.json
docker compose exec -T -e ENABLE_OCR=true backend python /app/scripts/import_benchmark_dataset.py --replace-cases --skip-embeddings --reconcile-acl manifest --manifest /app/backend/data/benchmark_raw/format_coverage/zh_enterprise_parser_regression_manifest.json
docker compose exec -T backend python /app/scripts/report_benchmark_results.py --dataset format_coverage_zh_parser --include-format-coverage --output /app/backend/data/eval_outputs/format-coverage-zh-parser-final-import-status-local.json
docker compose exec -T backend python /app/scripts/run_retrieval_ablation_benchmark.py --dataset format_coverage_zh_parser --manifest /app/backend/data/benchmark_raw/format_coverage/zh_enterprise_parser_regression_manifest.json --balanced-limit 20 --per-type-limit 20 --ablations full_indexed_sparse,indexed_sparse_only --case-statement-timeout-ms 10000 --output /app/backend/data/eval_outputs/format-coverage-zh-parser-final-retrieval-smoke-local.json --markdown-output /app/backend/data/eval_outputs/format-coverage-zh-parser-final-retrieval-smoke-local.md --progress
```

回归测试：

```powershell
docker compose exec -T backend python -m pytest app/tests/test_search_api.py app/tests/test_stard_benchmark_scripts.py app/tests/test_retrieval_diagnostics_script.py -q
```
