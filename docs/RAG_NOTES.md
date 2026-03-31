# 权限感知的 RAG 企业文档知识助手：RAG 技术链路说明

这份文档专门解释这个项目里的 RAG 是如何工作的，以及权限过滤、文档解析、切块、向量化、混合检索、citation 和拒答策略分别落在什么位置。

## 一句话理解
这个项目里的 RAG 不是“把整篇文档丢给模型”，而是先把文档整理成可检索、可引用、可追溯的知识片段，再只从当前用户有权限访问的证据里生成回答。

## 完整链路
可以把整条链路理解成 5 步：

1. 文档解析
   - 支持 `TXT / Markdown / HTML / PDF / DOCX`
   - 尽量保留标题、页码、段落等结构信息
   - 输出统一的 `ParsedDocument`

2. 结构化切块
   - 优先按文档结构切分，而不是纯固定长度截断
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

5. Grounded QA
   - 回答基于检索结果生成
   - 回答和 citation 分开返回
   - 如果证据不足，系统会拒答而不是硬编

## 为什么权限过滤必须前置
这个项目最重要的设计点之一，是把权限过滤放进检索链路本身，而不是最后才做展示过滤。

这样做的意义是：
- 无权限文档不会进入候选集
- 无权限 chunk 不会进入 citation
- 无权限内容不会进入 prompt
- 无权限结果不会出现在 trace 和调试信息里

对企业知识库场景来说，真正的风险往往不是“最后展示给错用户”，而是“无权限内容在召回、排序、提示词或日志里已经泄漏过一次”。

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

这样做的收益是：
- chunk 更接近真实语义单元
- citation 更自然
- 版本 diff 和上下文展示更稳定

## 检索为什么用 Hybrid Retrieval
项目没有只做 dense retrieval，而是使用 hybrid retrieval：
- lexical retrieval：PostgreSQL Full Text Search
- dense retrieval：pgvector 相似度
- fusion：对两路分数归一化后加权融合

这样做的原因是：
- 文档标题、制度名、缩写、固定术语更适合 lexical
- 自然语言表达、模糊提问更适合 dense
- 两路结合通常比单一路线更稳

## Citation 在系统里的作用
这个项目里 citation 不是附属能力，而是主链路的一部分。

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
项目不是只要召回到点内容就强行回答，还做了几层可靠性约束：
- 证据不足拒答：没有足够相关 chunk 时拒答
- 点名文档约束：用户明确点名时优先在目标文档范围内回答
- 权限安全拒答：目标文档不可访问时明确提示
- 弱相关拒答：问题和证据明显不一致时不输出伪造回答

目标是尽量减少“问 A 答 B”和“没有依据也给答案”的情况。

## Eval 与 Trace 如何服务 RAG
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

这意味着系统不仅能展示结果，还能解释：
- 为什么答对
- 为什么答错
- 为什么拒答
- 是否存在权限隔离问题

## 技术亮点总结
这条 RAG 链路的关键点，可以概括为：

1. 结构感知的文档解析与切块，而不是简单固定长度截断。
2. 每个 chunk 都带定位元数据，方便 citation 和回溯。
3. PostgreSQL FTS + pgvector 的 hybrid retrieval，而不是单一路径检索。
4. 权限过滤前置到检索链路中，避免无权限内容进入候选集和 prompt。
5. 回答结果带 citation，并配套 eval 和 trace，便于分析链路质量。
