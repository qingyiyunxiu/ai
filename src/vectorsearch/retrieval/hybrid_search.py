"""
混合检索引擎 - 核心技术展示
1. BM25关键词检索 (稀疏检索)
2. 稠密向量检索 (语义检索)
3. RRF融合策略 (Reciprocal Rank Fusion)
4. 查询重写与扩展
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict
import re
import math

from loguru import logger

from ..core.vector_store import Document, SearchResult, BaseVectorStore
from ..core.embeddings import BaseEmbeddingModel


@dataclass
class RetrievalResult:
    """检索结果"""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None
    fused_score: Optional[float] = None
    rank: int = 0


class BM25Retriever:
    """BM25关键词检索器 - 传统IR检索"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # 词频饱和度参数
        self.b = b    # 长度归一化参数
        self.documents: List[Document] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_freq: Dict[str, int] = defaultdict(int)  # 文档频率
        self.term_freq: List[Dict[str, int]] = []  # 每个文档的词频
        self._tokenizer = lambda text: re.findall(r'\w+', text.lower())
    
    def index_documents(self, documents: List[Document]) -> None:
        """构建BM25索引"""
        self.documents = documents
        self.doc_lengths = []
        self.term_freq = []
        self.doc_freq.clear()
        
        for doc in documents:
            tokens = self._tokenizer(doc.content)
            self.doc_lengths.append(len(tokens))
            
            tf = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            self.term_freq.append(tf)
            
            for token in set(tokens):
                self.doc_freq[token] += 1
        
        self.avg_doc_length = sum(self.doc_lengths) / len(documents) if documents else 0
        logger.info(f"BM25 indexed {len(documents)} documents, vocabulary size: {len(self.doc_freq)}")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """BM25检索，返回(文档索引, 分数)"""
        query_tokens = self._tokenizer(query)
        if not query_tokens:
            return []
        
        scores = []
        N = len(self.documents)
        
        for doc_idx in range(N):
            score = 0.0
            doc_len = self.doc_lengths[doc_idx]
            
            for token in query_tokens:
                if token not in self.doc_freq:
                    continue
                
                # IDF计算
                idf = math.log((N - self.doc_freq[token] + 0.5) / (self.doc_freq[token] + 0.5) + 1)
                
                # TF计算
                tf = self.term_freq[doc_idx].get(token, 0)
                
                # BM25公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                score += idf * numerator / denominator
            
            if score > 0:
                scores.append((doc_idx, score))
        
        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class RRFFuser:
    """RRF (Reciprocal Rank Fusion) 结果融合器
    无参数融合策略，无需调参，工业界标准做法
    """
    
    def __init__(self, k: int = 60):
        self.k = k  # RRF常数，通常取60
    
    def fuse(
        self,
        ranked_lists: List[List[Tuple[str, float]]],
        weights: Optional[List[float]] = None
    ) -> List[Tuple[str, float]]:
        """
        融合多个排序列表
        ranked_lists: 每个元素是 [(doc_id, score), ...] 的排序列表
        weights: 每个检索器的权重，默认均等
        """
        if weights is None:
            weights = [1.0] * len(ranked_lists)
        
        fused_scores: Dict[str, float] = defaultdict(float)
        
        for ranked_list, weight in zip(ranked_lists, weights):
            for rank, (doc_id, _) in enumerate(ranked_list, 1):
                fused_scores[doc_id] += weight * (1.0 / (self.k + rank))
        
        # 按融合分数排序
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results


class QueryRewriter:
    """查询重写与扩展模块"""
    
    def __init__(self):
        # 停用词表
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'what', 'when', 'where', 'who', 'why', 'how', 'which', 'this',
            '的', '是', '在', '和', '与', '或', '了', '我', '你', '他'
        }
    
    def rewrite(self, query: str) -> List[str]:
        """生成多个查询变体"""
        queries = [query]
        
        # 1. 去除停用词
        tokens = query.lower().split()
        filtered = ' '.join([t for t in tokens if t not in self.stopwords])
        if filtered and filtered != query:
            queries.append(filtered)
        
        # 2. 关键词提取
        keywords = self._extract_keywords(query)
        if keywords:
            queries.append(' '.join(keywords))
        
        return list(set(queries))
    
    def _extract_keywords(self, query: str) -> List[str]:
        """提取查询中的核心关键词"""
        tokens = re.findall(r'\w+', query.lower())
        keywords = [t for t in tokens if t not in self.stopwords and len(t) > 2]
        return keywords[:5]


class HybridSearchEngine:
    """混合检索引擎 - 主入口类"""
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_model: BaseEmbeddingModel,
        enable_bm25: bool = True,
        enable_rrf: bool = True,
        rrf_k: int = 60
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.enable_bm25 = enable_bm25
        self.enable_rrf = enable_rrf
        
        self.bm25_retriever = BM25Retriever()
        self.rrf_fuser = RRFFuser(k=rrf_k)
        self.query_rewriter = QueryRewriter()
        
        self._document_map: Dict[str, Document] = {}
        self._documents_indexed = False
    
    def index_documents(self, documents: List[Document]) -> None:
        """索引文档到混合检索系统"""
        # 1. 向量化并存储到向量数据库
        for doc in documents:
            if doc.embedding is None:
                doc.embedding = self.embedding_model.encode(doc.content)
            self._document_map[doc.id] = doc
        
        self.vector_store.add_documents(documents)
        
        # 2. 构建BM25索引
        if self.enable_bm25:
            self.bm25_retriever.index_documents(documents)
        
        self._documents_indexed = True
        logger.info(f"Hybrid search engine indexed {len(documents)} documents")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        hybrid_weights: Tuple[float, float] = (0.5, 0.5),  # (BM25权重, 向量权重)
        enable_query_rewrite: bool = False,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        混合检索主入口
        hybrid_weights: BM25和向量检索的权重，默认各0.5
        """
        if not self._documents_indexed:
            logger.warning("No documents indexed!")
            return []
        
        # 查询重写与扩展
        queries = self.query_rewriter.rewrite(query) if enable_query_rewrite else [query]
        
        all_vector_results: List[Tuple[str, float]] = []
        all_bm25_results: List[Tuple[str, float]] = []
        
        for q in queries:
            # 1. 稠密向量检索
            query_embedding = self.embedding_model.encode(q)
            vector_results = self.vector_store.search(
                query_embedding,
                top_k=top_k * 2,
                filter_metadata=filter_metadata
            )
            all_vector_results.extend([
                (r.document.id, r.score) for r in vector_results
            ])
            
            # 2. BM25稀疏检索
            if self.enable_bm25:
                bm25_raw = self.bm25_retriever.search(q, top_k=top_k * 2)
                for doc_idx, score in bm25_raw:
                    doc = self.bm25_retriever.documents[doc_idx]
                    all_bm25_results.append((doc.id, score))
        
        # 去重并保留最高分
        vector_results_dict = {}
        for doc_id, score in all_vector_results:
            if doc_id not in vector_results_dict or score > vector_results_dict[doc_id]:
                vector_results_dict[doc_id] = score
        vector_ranked = sorted(vector_results_dict.items(), key=lambda x: x[1], reverse=True)
        
        bm25_results_dict = {}
        for doc_id, score in all_bm25_results:
            if doc_id not in bm25_results_dict or score > bm25_results_dict[doc_id]:
                bm25_results_dict[doc_id] = score
        bm25_ranked = sorted(bm25_results_dict.items(), key=lambda x: x[1], reverse=True)
        
        # 3. RRF融合
        if self.enable_rrf and self.enable_bm25:
            fused_ranked = self.rrf_fuser.fuse(
                [bm25_ranked, vector_ranked],
                weights=hybrid_weights
            )
        else:
            fused_ranked = vector_ranked
        
        # 构建最终结果
        results = []
        for rank, (doc_id, fused_score) in enumerate(fused_ranked[:top_k], 1):
            doc = self._document_map[doc_id]
            results.append(RetrievalResult(
                document_id=doc_id,
                content=doc.content,
                metadata=doc.metadata,
                bm25_score=bm25_results_dict.get(doc_id),
                vector_score=vector_results_dict.get(doc_id),
                fused_score=fused_score,
                rank=rank
            ))
        
        return results
