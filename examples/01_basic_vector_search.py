"""
示例1: 基础向量搜索
演示Chroma/FAISS向量数据库的基本使用
"""

import sys
sys.path.insert(0, '../src')

from vectorsearch import (
    VectorStoreFactory,
    VectorStoreType,
    EmbeddingFactory,
    Document,
    HNSWConfig
)

def main():
    print("=" * 60)
    print("示例1: 基础向量搜索")
    print("=" * 60)
    
    # 1. 初始化嵌入模型
    print("\n[1] 初始化嵌入模型...")
    embedding_model = EmbeddingFactory.create("sentence-transformer")
    print(f"   嵌入维度: {embedding_model.dimension}")
    
    # 2. 配置HNSW索引参数
    hnsw_config = HNSWConfig(
        M=16,
        ef_construction=200,
        ef_search=50,
        distance_metric="cosine"
    )
    print(f"\n[2] HNSW配置: {hnsw_config}")
    
    # 3. 创建Chroma向量存储
    print("\n[3] 创建Chroma向量数据库...")
    vector_store = VectorStoreFactory.create(
        store_type=VectorStoreType.CHROMA,
        collection_name="demo_collection",
        hnsw_config=hnsw_config
    )
    
    # 4. 准备示例文档
    print("\n[4] 准备示例文档...")
    sample_texts = [
        "向量数据库用于存储和检索高维向量嵌入",
        "Chroma是一个开源的向量数据库解决方案",
        "FAISS提供高效的近似最近邻搜索算法",
        "HNSW是一种层次化的图索引结构",
        "语义搜索基于向量相似度匹配",
        "BM25是传统的关键词检索算法",
        "RAG结合检索和生成增强AI回答质量"
    ]
    
    documents = []
    for i, text in enumerate(sample_texts):
        embedding = embedding_model.encode(text)
        doc = Document(
            id=f"doc_{i}",
            content=text,
            metadata={"source": "demo", "index": i},
            embedding=embedding
        )
        documents.append(doc)
    
    # 5. 添加文档
    vector_store.add_documents(documents)
    print(f"   已索引 {vector_store.get_document_count()} 个文档")
    
    # 6. 执行搜索
    print("\n[5] 执行向量搜索...")
    query = "什么是向量数据库？"
    query_embedding = embedding_model.encode(query)
    print(f"   查询: {query}")
    
    results = vector_store.search(query_embedding, top_k=3)
    
    print("\n   搜索结果:")
    for r in results:
        print(f"   [{r.rank}] 分数={r.score:.4f} | {r.document.content}")
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
