# 权限感知的 RAG 企业文档知识助手：项目说明

## 项目一句话
一个面向企业知识库的权限感知 RAG 文档知识助手，支持权限感知检索、引用溯源、版本对比、结构化工作流生成，以及基础评测与追踪。

## 当前完成度
### 已实现
- mock 登录与三类角色：`viewer / manager / admin`
- 支持 `TXT / Markdown / HTML / PDF / DOCX` 文档上传与摄取
- 本地文件存储与文档版本保留
- 文档级 ACL：支持 `public / user / role / team`
- 基于 PostgreSQL FTS + pgvector 的权限感知混合检索
- 带 citation、confidence 和拒答策略的 grounded chat
- 会话与消息历史持久化
- 工作流产物：待办提取、周报草稿、FAQ 草稿
- 文档版本 diff：原始 diff、摘要、影响提示
- Eval 服务与权限隔离测试
- Trace 存储与简单 observability API
- 最小但完整的 React 前端演示链路

### 暂未实现
- 企业级 SSO / LDAP / OAuth
- 多租户组织树与复杂权限继承
- 生产级异步任务队列与 worker 编排
- 扫描版 PDF 的深度 OCR 优化
- Slack / 飞书等外部协作集成
- cross-encoder rerank 与更高级 faithfulness judge

## 这个项目解决什么问题
- 用户只能检索和引用自己有权限访问的文档
- 回答不只给文本，还要尽量给出 citation
- 会话可以继续沉淀为待办、周报草稿和 FAQ 草稿
- 文档版本变化可以被对比、总结和解释
- 整个链路可以被评测和追踪，而不是只能手工演示

## 项目的关键差异点
- 权限过滤发生在检索前或检索中，而不是最后才做展示过滤
- citation 作为结构化结果返回，而不是附带在回答文本里
- 问答只是入口，后面还能继续生成结构化工作流结果
- 不只支持当前文档问答，也支持版本 diff 与变更摘要
- 自带基础 eval 和 trace，便于验证效果和定位问题

## 核心设计
### 权限感知检索
- 用户登录后先解析角色、team 和 ACL
- 后端先计算当前用户的可访问文档集合
- lexical retrieval 和 vector retrieval 只在这个集合内运行
- 无权限 chunk 不会进入候选集、citation 和 prompt

### Grounded QA
- 回答基于检索证据生成
- 返回 answer、citation、confidence
- 证据不足时会明确拒答，避免伪造结果

### 版本对比
- 支持两个版本之间的 raw diff
- 支持差异摘要与 impact hints
- 检索默认只查当前版本，避免历史版本污染问答

### 结构化工作流生成
- 支持从问答会话提取待办
- 支持生成周报草稿
- 支持生成 FAQ 草稿

### 评测与追踪
- Eval 支持 retrieval、citation、faithfulness、permission isolation 等指标
- Trace 记录 query、retrieved chunks、selected citations、token、latency、error

## 真实业务价值
- 降低企业知识问答中的信息泄漏风险
- 提高流程、制度、运维文档问答的可信度和可追溯性
- 把问答直接转成待办、周报和 FAQ，缩短从信息获取到执行的路径
- 帮助团队追踪文档版本变化，理解变更影响
- 为后续模型、检索和提示词优化提供数据基础

## 使用建议
这个仓库当前是一个 MVP / 参考实现，适合作为课程项目、实习项目、内部工具原型或继续产品化的起点。

## 后续可以怎么继续做
- 接入正式 OpenAI structured output
- 在 permission-safe 候选集之后加入 cross-encoder rerank
- 接入 Langfuse / OpenTelemetry
- 将本地文件存储替换为 S3 / MinIO
- 增加 FAQ 审核流
- 增加更完整的离线评测数据集和标注工具
