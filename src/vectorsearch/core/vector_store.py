"""
向量数据库核心模块 - 支持Chroma和FAISS双后端
支持HNSW索引配置、持久化、批量插入等企业级特性
"""

import os
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass
import numpy as np

from loguru import logger


class VectorStoreType(Enum):
    """向量数据库类型"""
    CHROMA = "chroma"
    FAISS = "faiss"


@dataclass
class HNSWConfig:
    """HNSW索引配置 - ANN搜索优化参数"""
    M: int = 16  # 每个节点的邻居数
    ef_construction: int = 200  # 构建时的探索邻居数
    ef_search: int = 50  # 搜索时的探索邻居数
    distance_metric: str = "cosine"  # 距离度量: cosine, l2, ip


@dataclass
class Document:
    """文档数据结构"""
    id: str
    content: str
    metadata: Dict[str, Any] = None
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchResult:
    """搜索结果数据结构"""
    document: Document
    score: float
    rank: int


class BaseVectorStore:
    """向量存储基类"""
    
    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        embedding_dimension: int = 384,
        hnsw_config: Optional[HNSWConfig] = None
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_dimension = embedding_dimension
        self.hnsw_config = hnsw_config or HNSWConfig()
        self._client = None
        self._collection = None
        
    def add_documents(self, documents: List[Document]) -> int:
        """批量添加文档"""
        raise NotImplementedError
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """向量相似度搜索"""
        raise NotImplementedError
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """删除文档"""
        raise NotImplementedError
    
    def get_document_count(self) -> int:
        """获取文档总数"""
        raise NotImplementedError
    
    def persist(self) -> None:
        """持久化到磁盘"""
        raise NotImplementedError


class ChromaVectorStore(BaseVectorStore):
    """Chroma向量数据库实现"""
    
    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "./chroma_db",
        embedding_dimension: int = 384,
        hnsw_config: Optional[HNSWConfig] = None
    ):
        super().__init__(collection_name, persist_directory, embedding_dimension, hnsw_config)
        self._init_client()
    
    def _init_client(self):
        """初始化Chroma客户端"""
        import chromadb
        from chromadb.config import Settings
        
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # HNSW配置
        hnsw_space = {
            "cosine": "cosine",
            "l2": "l2",
            "ip": "ip"
        }.get(self.hnsw_config.distance_metric, "cosine")
        
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:M": self.hnsw_config.M,
                "hnsw:ef_construction": self.hnsw_config.ef_construction,
                "hnsw:ef_search": self.hnsw_config.ef_search,
                "hnsw:space": hnsw_space
            }
        )
        logger.info(f"Chroma collection '{self.collection_name}' initialized with HNSW config")
    
    def add_documents(self, documents: List[Document]) -> int:
        """批量添加文档到Chroma"""
        if not documents:
            return 0
        
        ids = [doc.id for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        embeddings = [doc.embedding for doc in documents if doc.embedding is not None]
        
        if len(embeddings) == len(documents):
            self._collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas,
                embeddings=embeddings
            )
        else:
            self._collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas
            )
        
        logger.info(f"Added {len(documents)} documents to Chroma")
        return len(documents)
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Chroma向量搜索"""
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )
        
        search_results = []
        for idx, (doc_id, content, metadata, distance) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            # Chroma返回的是距离，转换为相似度分数 (cosine: 0~2 -> 1~-1)
            score = 1.0 - (distance / 2.0) if self.hnsw_config.distance_metric == "cosine" else 1.0 / (1.0 + distance)
            
            search_results.append(SearchResult(
                document=Document(id=doc_id, content=content, metadata=metadata),
                score=score,
                rank=idx + 1
            ))
        
        return search_results
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        self._collection.delete(ids=document_ids)
        logger.info(f"Deleted {len(document_ids)} documents")
        return True
    
    def get_document_count(self) -> int:
        return self._collection.count()
    
    def persist(self) -> None:
        """Chroma自动持久化"""
        pass


class FAISSVectorStore(BaseVectorStore):
    """FAISS向量数据库实现 - 支持HNSW索引"""
    
    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "./faiss_db",
        embedding_dimension: int = 384,
        hnsw_config: Optional[HNSWConfig] = None
    ):
        super().__init__(collection_name, persist_directory, embedding_dimension, hnsw_config)
        self._documents: Dict[str, Document] = {}
        self._index = None
        self._init_index()
    
    def _init_index(self):
        """初始化FAISS HNSW索引"""
        import faiss
        
        index_path = os.path.join(self.persist_directory, f"{self.collection_name}.index")
        
        if os.path.exists(index_path):
            self._index = faiss.read_index(index_path)
            logger.info(f"Loaded existing FAISS index from {index_path}")
        else:
            # 创建HNSW索引
            if self.hnsw_config.distance_metric == "cosine":
                # 余弦相似度使用内积，需要归一化
                self._index = faiss.IndexHNSWFlat(
                    self.embedding_dimension,
                    self.hnsw_config.M,
                    faiss.METRIC_INNER_PRODUCT
                )
            else:
                self._index = faiss.IndexHNSWFlat(
                    self.embedding_dimension,
                    self.hnsw_config.M,
                    faiss.METRIC_L2
                )
            
            self._index.hnsw.efConstruction = self.hnsw_config.ef_construction
            self._index.hnsw.efSearch = self.hnsw_config.ef_search
            
            os.makedirs(self.persist_directory, exist_ok=True)
            logger.info(f"Created new FAISS HNSW index")
    
    def add_documents(self, documents: List[Document]) -> int:
        """批量添加文档到FAISS"""
        if not documents:
            return 0
        
        embeddings = []
        for doc in documents:
            if doc.embedding is not None:
                self._documents[doc.id] = doc
                
                # 余弦相似度需要归一化
                if self.hnsw_config.distance_metric == "cosine":
                    norm = np.linalg.norm(doc.embedding)
                    normalized_emb = np.array(doc.embedding) / norm if norm > 0 else doc.embedding
                    embeddings.append(normalized_emb)
                else:
                    embeddings.append(doc.embedding)
        
        if embeddings:
            embeddings_np = np.array(embeddings, dtype=np.float32)
            self._index.add(embeddings_np)
        
        logger.info(f"Added {len(embeddings)} documents to FAISS")
        return len(embeddings)
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """FAISS向量搜索"""
        # 归一化查询向量
        query_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        if self.hnsw_config.distance_metric == "cosine":
            norm = np.linalg.norm(query_np)
            query_np = query_np / norm if norm > 0 else query_np
        
        distances, indices = self._index.search(query_np, top_k)
        
        # 获取文档ID列表
        doc_ids = list(self._documents.keys())
        search_results = []
        
        for rank, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx < len(doc_ids) and idx >= 0:
                doc_id = doc_ids[idx]
                doc = self._documents[doc_id]
                
                # 内积即为余弦相似度
                score = float(distance) if self.hnsw_config.distance_metric == "cosine" else 1.0 / (1.0 + distance)
                
                # 应用元数据过滤
                if filter_metadata:
                    match = all(doc.metadata.get(k) == v for k, v in filter_metadata.items())
                    if not match:
                        continue
                
                search_results.append(SearchResult(
                    document=doc,
                    score=score,
                    rank=rank + 1
                ))
        
        return search_results
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """FAISS不支持原地删除，需要重建索引"""
        for doc_id in document_ids:
            self._documents.pop(doc_id, None)
        logger.warning("FAISS delete requires index rebuild, documents removed from cache only")
        return True
    
    def get_document_count(self) -> int:
        return len(self._documents)
    
    def persist(self) -> None:
        """持久化FAISS索引"""
        import faiss
        import json
        
        index_path = os.path.join(self.persist_directory, f"{self.collection_name}.index")
        docs_path = os.path.join(self.persist_directory, f"{self.collection_name}_docs.json")
        
        faiss.write_index(self._index, index_path)
        
        # 保存文档元数据
        docs_data = {
            doc_id: {
                "content": doc.content,
                "metadata": doc.metadata
            }
            for doc_id, doc in self._documents.items()
        }
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"FAISS index persisted to {self.persist_directory}")


class VectorStoreFactory:
    """向量存储工厂类"""
    
    @staticmethod
    def create(
        store_type: Union[str, VectorStoreType],
        collection_name: str,
        persist_directory: Optional[str] = None,
        embedding_dimension: int = 384,
        hnsw_config: Optional[HNSWConfig] = None
    ) -> BaseVectorStore:
        """创建向量存储实例"""
        if isinstance(store_type, str):
            store_type = VectorStoreType(store_type.lower())
        
        if persist_directory is None:
            persist_directory = f"./{store_type.value}_db"
        
        if store_type == VectorStoreType.CHROMA:
            return ChromaVectorStore(
                collection_name=collection_name,
                persist_directory=persist_directory,
                embedding_dimension=embedding_dimension,
                hnsw_config=hnsw_config
            )
        elif store_type == VectorStoreType.FAISS:
            return FAISSVectorStore(
                collection_name=collection_name,
                persist_directory=persist_directory,
                embedding_dimension=embedding_dimension,
                hnsw_config=hnsw_config
            )
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")
