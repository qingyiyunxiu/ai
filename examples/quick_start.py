"""
快速开始 - VectorSearch-Pro 完整演示
"""

import sys
sys.path.insert(0, '../src')

from vectorsearch import (
    VectorStoreFactory,
    VectorStoreType,
    EmbeddingFactory,
    HybridSearchEngine,
    RAGEngine,
    RetrievalEvaluator,
    load_sample_documents
)

def main():
    print("=" * 70)
    print("VectorSearch-Pro - 企业级向量检索与RAG引擎")
    print("=" * 70)
    
    # 1. 初始化核心组件
    print("\n📦 步骤1: 初始化核心组件")
    print("-" * 50)
    
    embedding_model = EmbeddingFactory.create("sentence-transformer")
    vector_store = VectorStoreFactory.create(
        store_type=VectorStoreType.CHROMA,
        collection_name="quickstart_demo"
    )
    
    print(f"   ✓ 嵌入模型: all-MiniLM-L6-v2 (维度: {embedding_model.dimension})")
    print("   ✓ 向量数据库: Chroma (HNSW索引)")
    
    # 2. 混合检索演示
    print("\n🔍 步骤2: 混合检索演示 (BM25 + 向量 + RRF)")
    print("-" * 50)
    
    hybrid_engine = HybridSearchEngine(
        vector_store=vector_store,
        embedding_model=embedding_model
    )
    
    # 加载示例文档
    sample_docs = load_sample_documents()
    from vectorsearch import Document
    
    documents = [
        Document(id=f"doc_{i}", content=d["content"], metadata=d["metadata"])
        for i, d in enumerate(sample_docs)
    ]
    
    hybrid_engine.index_documents(documents)
    print(f"   ✓ 已索引 {len(documents)} 个技术文档")
    
    # 搜索测试
    query = "混合检索算法"
    results = hybrid_engine.search(query, top_k=3)
    
    print(f"\n   查询: '{query}'")
    for r in results:
        print(f"   [{r.rank}] 分数={r.fused_score:.4f}")
        print(f"        {r.content[:60]}...")
    
    # 3. RAG演示
    print("\n🤖 步骤3: RAG检索增强生成")
    print("-" * 50)
    
    rag_engine = RAGEngine(
        vector_store=vector_store,
        embedding_model=embedding_model
    )
    
    question = "什么是GraphRAG？"
    response = rag_engine.query(question, top_k=2)
    
    print(f"   问题: {question}")
    print(f"   检索到 {len(response.retrieved_contexts)} 个相关上下文")
    for i, ctx in enumerate(response.retrieved_contexts, 1):
        print(f"   [{i}] 相关度: {ctx['score']:.3f}")
    
    # 4. 技术栈总结
    print("\n🏆 核心技术栈展示")
    print("-" * 50)
    print("   ✅ 向量数据库: Chroma / FAISS")
    print("   ✅ 嵌入模型: Sentence-Transformers / OpenAI")
    print("   ✅ 混合检索: BM25 + 稠密向量 + RRF融合")
    print("   ✅ 重排序: Cross-Encoder")
    print("   ✅ RAG: 基础RAG + GraphRAG")
    print("   ✅ 评估指标: nDCG / MRR / Precision@k / MAP")
    print("   ✅ 索引优化: HNSW参数调优")
    
    print("\n" + "=" * 70)
    print("项目已准备就绪，可直接上传GitHub!")
    print("=" * 70)

if __name__ == "__main__":
    main()
