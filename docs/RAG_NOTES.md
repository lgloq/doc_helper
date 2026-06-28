# 权限感知的 RAG 企业文档知识助手：技术链路说明

本文档说明当前项目中 RAG 链路的实现方式，以及权限过滤、文档解析、切块、向量化、混合检索、citation、多步骤处理、后台任务恢复、诊断和拒答策略在链路中的位置。

## 概述
该 RAG 流程会先将文档整理为可检索、可引用、可追溯的片段，再基于当前用户有权限访问的证据生成回答。

## 完整链路
整条链路可分为 6 个阶段：

1. 文档解析
   - 支持 `TXT / Markdown / HTML / PDF / DOCX / XLSX / XLS / PPTX / CSV / PNG / JPG / JPEG`
   - 尽量保留标题、页码、段落等结构信息
   - Markdown、HTML、DOCX、CSV 和文本型 PDF 表格会转成可检索文本
   - 图片文件和扫描版 PDF 页面在开启 OCR 后会转成普通文本 segment；规整图片表格会 best-effort 转成 `Table row:` 文本
   - 输出统一的 `ParsedDocument`

2. 结构化切块
   - 优先按文档结构切分，不直接做固定长度截断
   - 每个 chunk 保留文档、版本、页码、段落、section 等元数据
   - 这些元数据后续会用于 citation 和 diff 展示

3. 索引构建
   - 原始文本进入 PostgreSQL，用于全文检索
   - embedding 写入 pgvector，用于语义检索
   - 摄取阶段一次完成解析、切块、向量化和入库

4. 意图路由与轻量上下文复用
   - 先读取最近多轮消息
   - 更早历史压缩为摘要
   - 复用上一轮目标文档、上一轮工具、上一轮结果类型和上一轮 observation 摘要
   - 问题类型只做粗分类，后续是继续检索、版本对比还是生成结构化结果，会结合上下文继续决定

5. 权限感知检索
   - 先根据当前用户的 `user / role / department / ACL` 算出可访问文档集合；旧版 `team_name` 仅作为兼容字段参与回退判断
   - lexical retrieval 和 vector retrieval 都只在可访问范围内执行
   - 两路结果先做 score fusion，再进入可配置 rerank provider，得到最终候选 chunk
   - 简单问题会按保守规则跳过不必要的 query plan probe、LLM rewrite 或 evidence sweep
   - debug / trace 会记录 query rewrite、lexical retrieval、vector retrieval、evidence sweep、rerank 等阶段耗时和跳过原因

6. 引用式问答 / 结构化结果生成
   - 回答基于检索结果生成
   - 回答和 citation 分开返回
   - 如果证据不足，系统会拒答，不输出无依据结论
   - evidence audit 会抽取答案中的关键事实，标记支撑状态并关联最相关证据片段
   - 如果是待办、周报、FAQ 请求，则改走对应工作流链路

## 为什么权限过滤必须前置
当前项目将权限过滤纳入检索链路本身，在候选召回阶段就限定可访问范围。

这样做的意义是：
- 无权限文档不会进入候选集
- 无权限 chunk 不会进入 citation
- 无权限内容不会进入 prompt
- 无权限结果不会出现在 trace 和调试信息里

在企业知识库场景中，候选集、prompt、citation 和 trace 都属于权限隔离范围。

## 文档解析与切块策略
### 解析策略
- Markdown / HTML：尽量保留标题层级，并提取文本型表格
- PDF：尽量保留页码；可复制文本型表格会尝试按行转成 key-value 文本
- 扫描版 PDF：不整篇无脑 OCR；先做页级文本解析，页面文本不足或没有有效 segment 时才渲染该页并 OCR
- 图片：开启 OCR 后提取文字；低信息量图号、页码、短噪声会尽量过滤；规整图片表格会按 OCR 坐标 best-effort 聚合为表格行
- DOCX：保留段落、heading 风格和文本型表格；正文和表格中的内嵌图片会做 OCR
- XLSX / XLS / PPTX：当前 v1 仅这三类走 MarkItDown 适配器；转换结果会映射回统一的 `ParsedSegment` 和表格行文本。Excel 会额外规整 `Unnamed:` 伪表头，chunk citation metadata 会保留 `sheet / slide / table` 信息
- TXT：按自然段拆分
- CSV：第一行作为表头，后续行转成 key-value 文本

当前 OCR 是 parser 层的轻量增强，复用现有 `ParsedSegment -> chunk -> embedding -> PostgreSQL FTS + pgvector -> citation` 链路和 ACL 判断。文本 PDF 中的嵌入图片、本地相对路径或 base64 的 Markdown / HTML 图片、以及 DOCX 正文和表格中的内嵌图片，会复用同一套图片 OCR 链路。MarkItDown 当前只接受受控本地文件，并带文件大小、转换超时、输出字符数和表格行数保护。当前能力边界包括：低清扫描、旋转拍照、复杂合并单元格、复杂跨页表格和图片型复杂版面；柱状图、饼图、流程图、组织图等图片当前提取可见标签和文字；HTML 远程图片、Markdown 外链图片、DOCX 页眉页脚或水印类图片、Office 版面级高保真还原属于后续扩展范围。

### 切块策略
- 优先保留语义边界
- 只在段落过长时继续细分
- 每个 chunk 都带定位信息，方便回溯来源

该策略带来的主要收益包括：
- chunk 更接近真实语义单元
- citation 更自然
- 版本 diff 和上下文展示更稳定

## 为什么使用混合检索
项目使用 hybrid retrieval：
- lexical retrieval：PostgreSQL Full Text Search
- dense retrieval：pgvector 相似度
- fusion：对两路分数归一化后做加权融合

采用该方案的主要原因是：
- 文档标题、制度名、缩写、固定术语更适合 lexical
- 自然语言表达、模糊提问更适合 dense
- 两路结合通常比单一路线更稳

## 为什么召回后还要 rerank
混合召回解决的是“把相关 chunk 先找出来”，但不代表最终排序一定最适合回答问题。

当前版本会在可访问候选集上增加一层 rerank provider。默认 heuristic 模式主要考虑：
- query 和 chunk 的 token overlap
- 标题和 section 的命中情况
- lexical 支撑是否存在
- 指定文档场景下的目标文档加分

如果配置外部能力，也可以切换到 LLM JSON rerank 或 Qwen rerank provider；外部 provider 失败时会回退到本地 heuristic，避免检索链路整体不可用。

这样做的目标是提升进入回答阶段的候选排序质量。

## Citation 的作用
citation 会随问答结果一并返回，不依赖前端额外拼接说明。

citation 会记录：
- 文档标题
- 版本号
- chunk id / chunk index
- 页码或段落位置
- 预览片段
- lexical / vector / fused score

这些信息的用途包括：
- 前端独立展示来源片段
- 用户点击回看上下文
- 后续 FAQ 沉淀
- 调试与评测
- 审计与可追踪性

## 事实级证据审计
citation 说明“答案引用了哪些来源”，事实级 evidence audit 进一步说明“答案中的关键事实分别由哪些证据支撑”。

当前 evidence audit 会记录：
- 关键事实文本
- 支撑状态：完全支撑、部分支撑或待核实
- 支撑分数和支撑引用
- 支撑片段的文档标题、位置、rank 和 excerpt

前端会在 Chat 和 trace detail 中展示事实覆盖率、各支撑状态数量和事实到引用的定位关系。展示层只呈现后端审计结果，不放宽事实支撑阈值。

## 多步骤处理与处理轨迹
当前系统采用受控的多步骤处理流程来组织检索、版本对比和结构化结果生成。

主路径包括：
- 系统先判断问题类型
- 再结合问题、上下文、已有 observation 和工具描述输出下一步 action
- 处理流程按 `observe -> decide -> act` 循环执行，最多 3 步
- 执行阶段只允许调用白名单工具
- 超过 `max_steps` 或证据不足时，会收束为最终回答、拒答或补充说明

兼容层仍保留以下五步摘要，便于前端和 trace 对齐展示：
- `query_analysis`
- `tool_selection`
- `tool_execution`
- `evidence_review`
- `answer_generation`

这些信息会同时写入：
- assistant 消息 metadata
- observability trace extra metadata
- 前端“处理轨迹”面板

当前对外工具名统一为：
- `search_docs`
- `compare_versions`
- `extract_todos`
- `generate_weekly_report`
- `generate_faq`

这套处理流程把中间步骤、执行结果和最终回答串成一条可追踪链路，便于前端展示、问题排查和回归验证。

## 后台任务与恢复
明显耗时较长或需要跨页面恢复的操作统一记录到 `operation_jobs`：

- chat message submission
- eval run
- document diff summary generation
- document ingestion

每个 job 记录任务类型、状态、`client_request_id`、资源指向、请求快照、结果快照、失败原因和 ARQ job id。同一用户、同一任务类型、同一 `client_request_id` 会复用同一条 job，避免刷新、切页或并发重试造成重复提交。

ARQ 队列按任务类型拆分为 chat、eval、ingest 和 diff，避免大型评测或入库任务长期挤占问答。前端通过 pending operation 记录和 `GET /api/v1/jobs/{job_id}` 恢复 queued / running / completed / failed 状态。

## 拒答与可靠性策略
证据不足时，系统会通过多层可靠性约束控制回答范围：
- 证据不足拒答：没有足够相关 chunk 时拒答
- 点名文档约束：用户明确点名时优先在目标文档范围内回答
- 权限安全拒答：目标文档不可访问时明确提示
- 弱相关拒答：问题和证据明显不一致时不输出伪造回答

目标是尽量减少“问 A 答 B”和“没有依据也给答案”的情况。

## 权限诊断与审计
权限过滤仍然前置在检索阶段。除此之外，当前链路会把权限相关失败原因显式记录下来：

- `permission_refusal_reason_code`
- `permission_refusal_reason`
- `permission_probe_target_hint`
- 可访问 / 不可访问目标数量

当系统识别到疑似受限文档探测并触发早停时，会额外写入 `permission_denied_retrieval` 审计 trace。管理员也可以查看指定用户的可见文档范围，并在保存 ACL 前预估变更影响。

## Eval 与 Trace 的作用
### Eval
内置评测关注以下指标：
- retrieval hit rate
- citation accuracy
- answer faithfulness
- permission isolation correctness

Benchmark 当前分为四层：
- `smoke`：小规模冒烟回归
- `full`：主发布门禁和完整质量回归
- `hard`：跨文档、表格、时序、权限等难例回归
- `latency`：性能预算和阶段耗时回归

### Trace
每次问答链路会记录：
- query_text
- retrieved chunks
- selected citations
- model_name
- token
- latency
- error
- 阶段耗时和 skip reason
- pipeline diagnosis
- permission refusal reason
- evidence audit 摘要

因此系统不仅输出结果，也支持回看以下信息：
- 为什么答对
- 为什么答错
- 为什么拒答
- 是否存在权限隔离问题
- 失败发生在权限过滤、候选召回、候选选择、引用覆盖还是答案生成

## 总结
这条 RAG 链路的关键点如下：

1. 结构感知的文档解析与切块，不直接做简单固定长度截断。
2. 每个 chunk 都带定位元数据，方便 citation 和回溯。
3. PostgreSQL FTS + pgvector 的 hybrid retrieval，不依赖单一路径检索。
4. 在安全候选集上增加可配置 rerank provider，提升最终进入回答阶段的排序稳定性。
5. 权限过滤前置到检索链路中，避免无权限内容进入候选集和 prompt。
6. 对简单问题加入 latency budget、跳过条件和阶段耗时记录，避免不必要重路径拖慢问答。
7. 增加多步骤处理与轻量上下文复用，便于前端展示、链路追踪和回归验证。
8. 长任务通过 `operation_jobs` 和拆分队列支持刷新、切页和并发重试后的恢复。
9. 回答结果带 citation，并配套事实级 evidence audit、eval、trace 和 pipeline diagnosis，方便分析链路质量。
