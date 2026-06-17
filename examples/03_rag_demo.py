"""
示例3: RAG检索增强生成
演示基础RAG和GraphRAG
"""

import sys
sys.path.insert(0, '../src')

from vectorsearch import (
    VectorStoreFactory,
    VectorStoreType,
    EmbeddingFactory,
    RAGEngine,
    GraphRAGEngine,
    RerankerFactory
)

def main():
    print("=" * 60)
    print("示例3: RAG检索增强生成")
    print("=" * 60)
    
    # 1. 初始化组件
    print("\n[1] 初始化组件...")
    embedding_model = EmbeddingFactory.create("sentence-transformer")
    vector_store = VectorStoreFactory.create(
        store_type=VectorStoreType.CHROMA,
        collection_name="rag_demo"
    )
    
    # 2. 创建RAG引擎
    print("\n[2] 创建RAG引擎...")
    rag_engine = RAGEngine(
        vector_store=vector_store,
        embedding_model=embedding_model,
        use_hybrid_search=True
    )
    
    # 3. 添加知识库文档
    print("\n[3] 添加知识库文档...")
    knowledge_docs = [
        """
        Chroma是一个开源的向量数据库，专为AI应用设计。
        它支持持久化存储、元数据过滤、HNSW索引等功能。
        Chroma提供Python和JavaScript API，易于集成。
        """,
        """
        FAISS是Facebook AI Research开发的相似度搜索库。
        它支持多种索引类型包括Flat、IVF、HNSW等。
        FAISS针对CPU和GPU都有优化，速度极快。
        """,
        """
        混合检索结合了关键词检索和语义检索的优点。
        BM25擅长精确关键词匹配，向量检索擅长语义理解。
        使用RRF算法可以无参数地融合两种排序结果。
        """,
        """
        GraphRAG是微软提出的检索增强生成技术。
        它通过提取文档中的实体和关系构建知识图谱。
        查询时先定位相关实体，再通过图遍历扩展检索范围。
        """
    ]
    
    rag_engine.add_documents(knowledge_docs)
    print(f"   已添加 {len(knowledge_docs)} 篇知识库文档")
    
    # 4. 执行RAG查询
    print("\n[4] RAG问答演示...")
    questions = [
        "Chroma有什么特点？",
        "FAISS是谁开发的？支持什么索引？",
        "混合检索有什么优势？",
        "GraphRAG的工作原理是什么？"
    ]
    
    for question in questions:
        print(f"\n   问题: {question}")
        response = rag_engine.query(question, top_k=2)
        
        print(f"   检索到 {len(response.retrieved_contexts)} 个相关片段:")
        for i, ctx in enumerate(response.retrieved_contexts, 1):
            score = ctx.get('score', 0)
            preview = ctx['content'].strip()[:80]
            print(f"     [{i}] 分数={score:.3f} | {preview}...")
    
    # 5. GraphRAG演示
    print("\n[5] GraphRAG演示...")
    graph_vector_store = VectorStoreFactory.create(
        store_type=VectorStoreType.CHROMA,
        collection_name="graphrag_demo"
    )
    
    graph_rag = GraphRAGEngine(
        vector_store=graph_vector_store,
        embedding_model=embedding_model
    )
    
    graph_rag.add_documents(knowledge_docs)
    print(f"   GraphRAG已提取 {len(graph_rag.entities)} 个实体")
    print(f"   实体列表: {list(graph_rag.entities.keys())[:10]}")
    
    print("\n" + "=" * 60)
    print("RAG演示完成!")
    print("技术点: 文本分块 + 混合检索 + 上下文构建 + GraphRAG实体图谱")
    print("=" * 60)

if __name__ == "__main__":
    main()
