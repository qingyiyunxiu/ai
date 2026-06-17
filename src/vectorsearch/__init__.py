"""
VectorSearch-Pro - 企业级向量检索与RAG引擎

核心功能：
- 多向量数据库支持 (Chroma, FAISS)
- 混合检索引擎 (BM25 + 稠密向量 + RRF融合)
- Cross-Encoder重排序
- RAG & GraphRAG检索系统
- 检索评估指标 (nDCG, MRR, MAP)
- HNSW索引优化配置
"""

__version__ = "1.0.0"
__author__ = "VectorSearch Team"

from .core import (
    VectorStoreType,
    HNSWConfig,
    Document,
    SearchResult,
    VectorStoreFactory,
    EmbeddingFactory
)

from .retrieval import (
    HybridSearchEngine,
    BM25Retriever,
    RRFFuser
)

from .rerank import (
    RerankerFactory,
    CrossEncoderReranker
)

from .rag import (
    RAGEngine,
    GraphRAGEngine,
    TextSplitter
)

from .evaluation import (
    RetrievalEvaluator,
    RetrievalMetrics
)

from .utils import load_sample_documents

__all__ = [
    # Core
    "VectorStoreType",
    "HNSWConfig",
    "Document",
    "SearchResult",
    "VectorStoreFactory",
    "EmbeddingFactory",
    
    # Retrieval
    "HybridSearchEngine",
    "BM25Retriever",
    "RRFFuser",
    
    # Rerank
    "RerankerFactory",
    "CrossEncoderReranker",
    
    # RAG
    "RAGEngine",
    "GraphRAGEngine",
    "TextSplitter",
    
    # Evaluation
    "RetrievalEvaluator",
    "RetrievalMetrics",
    
    # Utils
    "load_sample_documents"
]
