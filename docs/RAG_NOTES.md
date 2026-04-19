# 权限感知的 RAG 企业文档知识助手：技术链路说明

本文档说明当前项目中 RAG 链路的实现方式，以及权限过滤、文档解析、切块、向量化、混合检索、citation 和拒答策略在链路中的位置。

## 概述
该 RAG 流程不会将整篇文档直接提交给模型，而是先将文档整理为可检索、可引用、可追溯的片段，再基于当前用户有权限访问的证据生成回答。

## 完整链路
整条链路可分为 5 个阶段：

1. 文档解析
   - 支持 `TXT / Markdown / HTML / PDF / DOCX`
   - 尽量保留标题、页码、段落等结构信息
   - 输出统一的 `ParsedDocument`

2. 结构化切块
   - 优先按文档结构切分，不直接做固定长度截断
   - 每个 chunk 保留文档、版本、页码、段落、section 等元数据
   - 这些元数据后续会用于 citation 和 diff 展示

3. 索引构建
   - 原始文本进入 PostgreSQL，用于全文检索
   - embedding 写入 pgvector，用于语义检索
   - 摄取阶段一次完成解析、切块、向量化和入库

4. 权限感知检索
   - 先根据当前用户的 `user / role / team / ACL` 算出可访问文档集合
   - lexical retrieval 和 vector retrieval 都只在可访问范围内执行
   - 两路结果做 score fusion，得到最终候选 chunk

5. 引用式问答
   - 回答基于检索结果生成
   - 回答和 citation 分开返回
   - 如果证据不足，系统会拒答，不继续拼凑答案

## 为什么权限过滤必须前置
当前项目将权限过滤纳入检索链路本身，而不是在最终展示阶段再做结果过滤。

这样做的意义是：
- 无权限文档不会进入候选集
- 无权限 chunk 不会进入 citation
- 无权限内容不会进入 prompt
- 无权限结果不会出现在 trace 和调试信息里

在企业知识库场景中，风险并不只出现在最终答案展示阶段。一旦无权限内容进入候选集、prompt 或日志，权限边界实际上已经被突破。

## 文档解析与切块策略
### 解析策略
- Markdown / HTML：尽量保留标题层级
- PDF：尽量保留页码
- DOCX：保留段落和 heading 风格
- TXT：按自然段拆分

### 切块策略
- 优先保留语义边界
- 只在段落过长时继续细分
- 每个 chunk 都带定位信息，方便回溯来源

该策略带来的主要收益包括：
- chunk 更接近真实语义单元
- citation 更自然
- 版本 diff 和上下文展示更稳定

## 检索为什么用 Hybrid Retrieval
项目没有只做 dense retrieval，而是用了 hybrid retrieval：
- lexical retrieval：PostgreSQL Full Text Search
- dense retrieval：pgvector 相似度
- fusion：对两路分数归一化后加权融合

采用该方案的主要原因是：
- 文档标题、制度名、缩写、固定术语更适合 lexical
- 自然语言表达、模糊提问更适合 dense
- 两路结合通常比单一路线更稳

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

## 拒答与可靠性策略
项目不会在证据不足时强行生成回答，而是增加了多层可靠性约束：
- 证据不足拒答：没有足够相关 chunk 时拒答
- 点名文档约束：用户明确点名时优先在目标文档范围内回答
- 权限安全拒答：目标文档不可访问时明确提示
- 弱相关拒答：问题和证据明显不一致时不输出伪造回答

目标是尽量减少“问 A 答 B”和“没有依据也给答案”的情况。

## Eval 与 Trace 的作用
### Eval
内置评测关注以下指标：
- retrieval hit rate
- citation accuracy
- answer faithfulness
- permission isolation correctness

### Trace
每次问答链路会记录：
- query_text
- retrieved chunks
- selected citations
- model_name
- token
- latency
- error

因此系统不仅输出结果，也支持回看以下信息：
- 为什么答对
- 为什么答错
- 为什么拒答
- 是否存在权限隔离问题

## 总结
这条 RAG 链路的关键点如下：

1. 结构感知的文档解析与切块，不直接做简单固定长度截断。
2. 每个 chunk 都带定位元数据，方便 citation 和回溯。
3. PostgreSQL FTS + pgvector 的 hybrid retrieval，不依赖单一路径检索。
4. 权限过滤前置到检索链路中，避免无权限内容进入候选集和 prompt。
5. 回答结果带 citation，并配套 eval 和 trace，方便分析链路质量。
