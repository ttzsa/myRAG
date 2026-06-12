# 评测与观测层设计

## 1. 目标

评测与观测层负责回答两个问题：

1. 当前 RAG 系统到底好不好用
2. 当它不好用时，问题出在什么环节

如果没有这一层，项目就很容易停留在“能跑通”的阶段，无法形成真正可优化、可展示的工程闭环。

## 2. 总体思路

评测与观测层建议同时覆盖两类能力：

- `Evaluation`：离线评估与效果对比
- `Observability`：线上链路日志、性能、成本和 case 回放

目标是将以下链路完整记录下来：

```text
用户Query
  -> Query理解
  -> Query改写
  -> 多路召回结果
  -> RRF融合结果
  -> 精排结果
  -> 上下文构造
  -> 最终回答
  -> 引用输出
  -> 延迟与成本统计
```

## 3. 离线评测集设计

### 3.1 为什么必须自己构建评测集

个人知识库 RAG 的核心数据是你自己的 PDF，通用公开 benchmark 很难真实反映效果。因此必须构建自己的评测集。

### 3.2 评测集规模建议

第一版建议人工整理 `50 ~ 100` 个问题，后续逐步扩展。

### 3.3 问题覆盖类型

评测问题建议覆盖：

- 事实查找
- 定义解释
- 单文档总结
- 多文档对比
- 表格问题
- 图片/图表问题
- 引用定位问题
- 写作辅助问题

### 3.4 每条样本建议字段

建议每个评测样本包含：

- `sample_id`
- `question`
- `question_type`
- `target_doc_ids`
- `gold_chunks`
- `reference_answer`
- `must_have_citations`
- `difficulty`

其中 `gold_chunks` 很重要，它用于检索评测而不只是生成评测。

## 4. 检索评测设计

检索评测的核心问题是：

“正确证据是否被召回到了候选集中？”

### 4.1 推荐指标

- `Recall@k`
- `Hit Rate@k`
- `MRR`

### 4.2 建议评测层级

分别评估：

- dense only
- sparse only
- dense + sparse
- dense + sparse + RRF
- dense + sparse + RRF + rerank

这样你可以明确说明每个模块的收益。

### 4.3 Chunk 粒度对比

建议对不同 chunk 策略也单独评测：

- small chunk
- large chunk
- 双粒度混合

## 5. 生成评测设计

生成评测关注的是：

- 答案是否正确
- 是否忠实于证据
- 是否给出有效引用

### 5.1 推荐维度

- `correctness`
- `faithfulness`
- `citation_consistency`
- `completeness`
- `helpfulness`

### 5.2 评测方式

第一版建议以人工评测为主。

原因：

- 你的知识库是私有 PDF
- 数据规模初期不大
- 人工评估更容易发现实际问题

后续可加入：

- LLM-as-a-judge
- 自动引用一致性检查

## 6. 消融实验设计

这是面试展示中非常有价值的一部分。

### 6.1 推荐消融项

至少做以下几组实验：

1. `Dense Only`
2. `Dense + Sparse`
3. `Dense + Sparse + RRF`
4. `Dense + Sparse + RRF + Cross-Encoder`

如果后续加入图片理解，还可以比较：

5. `Without Image Understanding`
6. `With Image Understanding`

### 6.2 输出形式

建议把实验结果整理成：

- 指标表
- 典型 case 对比
- 失败案例分析

## 7. 线上观测设计

线上观测的重点是“可回放”。

### 7.1 每次 Query 建议记录

- `query_id`
- `raw_query`
- `rewritten_queries`
- `intent_type`
- `dense_results`
- `sparse_results`
- `fused_results`
- `reranked_results`
- `final_context_chunks`
- `final_answer`
- `citations`
- `latency_ms`
- `embedding_cost`
- `rerank_cost`
- `generation_cost`
- `total_token_usage`

### 7.2 为什么要记录这些

当结果不好时，你需要区分：

- 是 query 改写出了问题
- 还是召回丢了正确证据
- 还是 RRF 排名不合理
- 还是精排压掉了正确 chunk
- 还是生成模型没有忠实使用证据

## 8. 延迟与成本监控

由于你计划使用多个 API，这一层必须显式设计。

### 8.1 延迟分解

建议拆分记录：

- query 理解延迟
- 改写延迟
- dense 检索延迟
- sparse 检索延迟
- rerank 延迟
- 生成延迟
- 总延迟

### 8.2 成本分解

建议拆分记录：

- PDF 解析成本
- 图片理解成本
- embedding 成本
- rerank 成本
- LLM 生成成本

### 8.3 价值

这样后续你可以回答非常工程化的问题，例如：

- 哪个阶段最耗时
- 哪个阶段最耗钱
- 哪个优化项收益高但成本低

## 9. Case 回放设计

这是非常适合做展示的功能。

### 9.1 回放目标

给定一个历史 query，可以查看完整处理链路：

- 原 query
- 改写结果
- 各路召回 topk
- 融合结果
- 精排结果
- 最终上下文
- 最终答案

### 9.2 展示价值

case 回放有两个直接价值：

- 方便你自己调试系统
- 面试时能直观展示你不是在黑盒调 API

## 10. 失败案例分析方法

每次 bad case 至少归因到以下某一层：

- `Parsing Failure`
- `Chunking Failure`
- `Dense Retrieval Failure`
- `Sparse Retrieval Failure`
- `Fusion Failure`
- `Rerank Failure`
- `Context Construction Failure`
- `Generation Failure`

如果你能持续积累 bad case 并分类，后续优化会非常高效。

## 11. 推荐第一版评测与观测实现

建议第一版先做这些：

1. 构建 50 条人工评测集
2. 记录 Recall@5、Recall@10、MRR
3. 人工评估答案正确性与引用有效性
4. 保存每次 query 的召回与重排结果
5. 记录总延迟与总 token 消耗
6. 提供简单 case 回放页面或日志导出

## 12. 面试展示重点

评测与观测层是项目说服力最强的部分之一。你可以强调：

1. 我不仅实现了 RAG，还设计了自己的评测集
2. 我能定量比较 dense、sparse、RRF、rerank 的收益
3. 我能定位 bad case 属于解析、召回、重排还是生成问题
4. 我能统计 API 成本与延迟，而不是只追求效果

## 13. 后续可扩展方向

- 自动化评测 pipeline
- LLM judge 引入
- 引用一致性自动校验
- 多版本索引对比评测
- Prompt 版本对比评测
- 模型切换 AB test
- 检索链路 dashboard
