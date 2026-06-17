"""
示例2: 混合检索引擎
演示 BM25 + 稠密向量 + RRF融合 的混合检索
"""

import sys
sys.path.insert(0, '../src')

from vectorsearch import (
    VectorStoreFactory,
    VectorStoreType,
    EmbeddingFactory,
    Document,
    HybridSearchEngine
)

def main():
    print("=" * 60)
    print("示例2: 混合检索引擎 (BM25 + 向量 + RRF)")
    print("=" * 60)
    
    # 1. 初始化组件
    print("\n[1] 初始化组件...")
    embedding_model = EmbeddingFactory.create("sentence-transformer")
    vector_store = VectorStoreFactory.create(
        store_type=VectorStoreType.CHROMA,
        collection_name="hybrid_demo"
    )
    
    # 2. 创建混合检索引擎
    print("\n[2] 创建混合检索引擎 (启用BM25 + RRF融合)...")
    hybrid_engine = HybridSearchEngine(
        vector_store=vector_store,
        embedding_model=embedding_model,
        enable_bm25=True,
        enable_rrf=True,
        rrf_k=60
    )
    
    # 3. 准备文档
    print("\n[3] 索引文档...")
    sample_docs = [
        Document(id="doc1", content="Python是一种广泛使用的高级编程语言"),
        Document(id="doc2", content="Java是一种面向对象的编程语言"),
        Document(id="doc3", content="向量数据库用于AI应用中的语义检索"),
        Document(id="doc4", content="机器学习算法需要大量训练数据"),
        Document(id="doc5", content="深度学习使用神经网络进行特征学习"),
        Document(id="doc6", content="数据库索引优化查询性能"),
        Document(id="doc7", content="自然语言处理NLP是AI的重要分支"),
        Document(id="doc8", content="BM25算法用于信息检索中的关键词匹配")
    ]
    
    hybrid_engine.index_documents(sample_docs)
    print(f"   已索引 {len(sample_docs)} 个文档")
    
    # 4. 执行混合检索
    print("\n[4] 执行混合检索...")
    queries = [
        "编程语言",
        "向量检索",
        "机器学习算法"
    ]
    
    for query in queries:
        print(f"\n   查询: '{query}'")
        results = hybrid_engine.search(
            query=query,
            top_k=3,
            hybrid_weights=(0.5, 0.5)
        )
        
        for r in results:
            bm25_str = f"BM25={r.bm25_score:.3f}" if r.bm25_score else "N/A"
            vec_str = f"Vec={r.vector_score:.3f}" if r.vector_score else "N/A"
            print(f"     [{r.rank}] 融合分={r.fused_score:.4f} | {bm25_str} | {vec_str}")
            print(f"          {r.content}")
    
    print("\n" + "=" * 60)
    print("混合检索演示完成!")
    print("技术点: BM25稀疏检索 + 稠密向量检索 + RRF无参数融合")
    print("=" * 60)

if __name__ == "__main__":
    main()
