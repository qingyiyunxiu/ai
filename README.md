# VectorSearch-Pro

<div align="center">

**企业级向量检索与RAG引擎**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Chroma](https://img.shields.io/badge/VectorDB-Chroma/FAISS-green.svg)](https://www.trychroma.com/)
[![LangChain](https://img.shields.io/badge/RAG-LangChain-orange.svg)](https://python.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

混合检索 · 重排序 · GraphRAG · 工业级评估

</div>

## 📋 项目概述

VectorSearch-Pro 是一个面向**搜索/向量存储工程师**岗位的专业级开源项目，完整实现了现代向量检索系统的核心技术栈。本项目展示了从底层向量存储、混合检索算法，到上层RAG应用的全链路技术能力。

### 🎯 核心技术亮点

| 技术模块 | 实现能力 |
|---------|---------|
| **向量数据库** | Chroma + FAISS 双后端，HNSW索引参数调优 |
| **混合检索** | BM25稀疏检索 + 稠密向量检索 + RRF无参数融合 |
| **重排序** | Cross-Encoder 精细化重排序，提升检索相关性 |
| **RAG引擎** | 基础RAG + GraphRAG（实体提取 + 知识图谱） |
| **查询优化** | 查询重写、多查询扩展、检索路由 |
| **评估指标** | nDCG / MRR / Precision@k / MAP 工业标准 |

---

## 🏗️ 系统架构

```
<img width="2580" height="1720" alt="rag架构图" src="https://github.com/user-attachments/assets/cd0d2e32-fa08-4e8c-aa5b-d6eddac4d729" />

```

---

## 🚀 快速开始

### 环境安装

```bash
# 克隆项目
git clone https://github.com/yourusername/VectorSearch-Pro.git
cd VectorSearch-Pro

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

### 5分钟快速上手

```python
from vectorsearch import (
    VectorStoreFactory,
    VectorStoreType,
    EmbeddingFactory,
    HybridSearchEngine,
    Document
)

# 1. 初始化核心组件
embedding = EmbeddingFactory.create("sentence-transformer")
vector_store = VectorStoreFactory.create(VectorStoreType.CHROMA, "demo")

# 2. 创建混合检索引擎
engine = HybridSearchEngine(vector_store, embedding)

# 3. 索引文档
docs = [
    Document(id="1", content="向量数据库支持高效的语义检索"),
    Document(id="2", content="BM25是经典的关键词匹配算法"),
    Document(id="3", content="RRF融合多种检索结果提升效果")
]
engine.index_documents(docs)

# 4. 混合检索
results = engine.search("检索技术", top_k=3)
for r in results:
    print(f"[{r.rank}] {r.content}")
```

运行完整演示:
```bash
cd examples
python quick_start.py
```

---

## 📚 核心功能详解

### 1. 向量存储引擎

**支持双后端:**
- **Chroma**: 持久化向量数据库，支持元数据过滤
- **FAISS**: Facebook高性能检索库，HNSW索引优化

**HNSW参数调优:**
```python
from vectorsearch import HNSWConfig

hnsw_config = HNSWConfig(
    M=16,                 # 每个节点邻居数
    ef_construction=200,  # 构建时探索邻居数
    ef_search=50,         # 搜索时探索邻居数
    distance_metric="cosine"
)
```

### 2. 混合检索 (Hybrid Search)

**三阶段检索流水线:**
1. **BM25稀疏检索**: 精准关键词匹配，TF-IDF加权
2. **稠密向量检索**: 语义相似度匹配，余弦距离
3. **RRF融合**: Reciprocal Rank Fusion 无参数结果融合

```python
results = hybrid_engine.search(
    query="查询文本",
    top_k=10,
    hybrid_weights=(0.5, 0.5),  # BM25权重:向量权重
    enable_query_rewrite=True   # 启用查询扩展
)
```

### 3. Cross-Encoder重排序

两阶段检索架构:
- **召回阶段**: 混合检索快速返回Top-100
- **精排阶段**: Cross-Encoder精细化重排序Top-10

```python
from vectorsearch import RerankerFactory

reranker = RerankerFactory.create("cross-encoder")
reranked = reranker.rerank(query, candidate_docs, top_k=5)
```

### 4. RAG & GraphRAG

**基础RAG流程:**
- 文档分块 → 向量化 → 混合检索 → 上下文注入 → LLM生成

**GraphRAG增强:**
- 实体提取 → 知识图谱构建 → 实体链接 → 图遍历扩展检索

### 5. 检索评估指标

工业界标准评估指标完整实现:

| 指标 | 说明 | 用途 |
|------|------|------|
| **nDCG@k** | 归一化折损累计增益 | 衡量排序质量 |
| **MRR** | 平均倒数排名 | 第一个相关文档位置 |
| **Precision@k** | 前k准确率 | 结果精确率 |
| **Recall@k** | 前k召回率 | 覆盖率 |
| **MAP** | 平均准确率均值 | 整体检索质量 |

```python
from vectorsearch import RetrievalEvaluator

evaluator = RetrievalEvaluator()
result = evaluator.evaluate(queries_results, k_values=[1, 3, 5, 10])
evaluator.print_report(result)
```

---

## 📁 项目结构

```
VectorSearch-Pro/
├── src/
│   └── vectorsearch/
│       ├── core/              # 核心模块
│       │   ├── vector_store.py    # 向量数据库 (Chroma/FAISS)
│       │   └── embeddings.py      # 嵌入模型
│       ├── retrieval/         # 检索模块
│       │   └── hybrid_search.py   # 混合检索引擎
│       ├── rerank/            # 重排序模块
│       │   └── cross_encoder.py   # Cross-Encoder
│       ├── rag/               # RAG模块
│       │   └── rag_engine.py      # RAG & GraphRAG
│       ├── evaluation/        # 评估模块
│       │   └── metrics.py         # nDCG/MRR/MAP
│       └── utils/
├── examples/                  # 示例代码
│   ├── 01_basic_vector_search.py
│   ├── 02_hybrid_search.py
│   ├── 03_rag_demo.py
│   ├── 04_evaluation_metrics.py
│   └── quick_start.py
├── tests/                     # 单元测试
├── docs/                      # 文档
├── requirements.txt
├── setup.py
└── README.md
```

---

## 💡 技术深度说明

### 匹配岗位JD的核心能力

本项目完全覆盖**向量数据库/搜索工程师**岗位要求:

✅ **向量数据库生态**: Chroma/FAISS双实现，可扩展Milvus/Qdrant/PGVector  
✅ **混合检索**: BM25 + 稠密向量 + RRF融合策略  
✅ **重排序**: Cross-Encoder二阶段精排  
✅ **GraphRAG**: 实体提取 + 知识图谱 + 图遍历  
✅ **查询优化**: 查询重写、多查询扩展  
✅ **ANN索引调优**: HNSW参数(M/ef_construction/ef_search)  
✅ **检索评估**: nDCG/MRR/Precision完整评估体系  

### 设计亮点

1. **工厂模式**: VectorStoreFactory, EmbeddingFactory, RerankerFactory
2. **策略模式**: 可插拔的检索、重排序、融合策略
3. **配置驱动**: HNSW索引参数可配置，支持性能调优
4. **模块化设计**: 各组件低耦合高内聚，易于扩展

---

## 🧪 运行示例

```bash
# 基础向量搜索
python examples/01_basic_vector_search.py

# 混合检索演示
python examples/02_hybrid_search.py

# RAG问答系统
python examples/03_rag_demo.py

# 检索评估指标
python examples/04_evaluation_metrics.py

# 完整演示
python examples/quick_start.py
```

---

## 🔧 扩展开发

### 添加新的向量数据库

```python
from vectorsearch import BaseVectorStore

class MilvusVectorStore(BaseVectorStore):
    def add_documents(self, documents):
        # 实现Milvus接入
        pass
```

### 自定义融合策略

```python
class WeightedFuser:
    def fuse(self, ranked_lists, weights):
        # 自定义加权融合逻辑
        pass
```

---

## 📊 性能优化建议

1. **HNSW调优**: 大规模数据集增大 `M` 和 `ef_construction`
2. **批量索引**: 文档批量插入减少IO开销
3. **缓存策略**: 热门查询结果缓存
4. **硬件加速**: GPU加速嵌入计算和FAISS检索

---

## 📄 License

MIT License - 可自由用于学习和求职展示

---

## 🙋‍♂️ 关于作者

这是一个面向**向量检索/搜索工程师**岗位的专业展示项目。

**技术栈覆盖:**
- Python后端开发
- 向量数据库 (Chroma, FAISS)
- 信息检索理论 (BM25, TF-IDF, HNSW)
- 混合检索与RRF融合
- Cross-Encoder重排序
- RAG & GraphRAG
- 检索评估指标

---

<div align="center">

**如果这个项目对你有帮助，请给个 ⭐ Star**

</div>
