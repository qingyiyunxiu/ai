"""
重排序模块 - Cross-Encoder实现
使用Cross-Encoder进行精细化重排序，提升检索相关性
"""

from typing import List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

from loguru import logger


@dataclass
class RerankResult:
    """重排序结果"""
    content: str
    score: float
    rank: int
    metadata: dict = None


class BaseReranker(ABC):
    """重排序器基类"""
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """对文档进行重排序"""
        pass


class CrossEncoderReranker(BaseReranker):
    """Cross-Encoder重排序器
    使用 sentence-transformers 的 CrossEncoder 模型
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        max_length: int = 512
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._model = None
        self._load_model()
    
    def _load_model(self):
        """加载Cross-Encoder模型"""
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading CrossEncoder model: {self.model_name}")
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length
            )
            logger.info("CrossEncoder model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load CrossEncoder: {e}, using fallback")
            self._model = None
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        重排序文档
        query: 查询文本
        documents: 待重排序的文档列表
        top_k: 返回前k个结果，None返回全部
        """
        if not documents:
            return []
        
        if self._model is None:
            return self._fallback_rerank(query, documents, top_k)
        
        # 构建(query, document)对
        pairs = [[query, doc] for doc in documents]
        
        # 预测分数
        scores = self._model.predict(pairs)
        
        # 排序
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 构建结果
        results = []
        for rank, (doc, score) in enumerate(scored_docs[:top_k], 1):
            results.append(RerankResult(
                content=doc,
                score=float(score),
                rank=rank
            ))
        
        return results
    
    def _fallback_rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """降级方案：基于关键词匹配的简单重排序"""
        query_terms = set(query.lower().split())
        results = []
        
        for doc in documents:
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms & doc_terms)
            score = overlap / max(len(query_terms), 1)
            results.append((doc, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        return [
            RerankResult(content=doc, score=score, rank=rank)
            for rank, (doc, score) in enumerate(results[:top_k], 1)
        ]


class SimpleReranker(BaseReranker):
    """轻量级重排序器 - 基于BM25和关键词匹配"""
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored = []
        for doc in documents:
            doc_lower = doc.lower()
            
            # 精确匹配分数
            exact_match = 2.0 if query_lower in doc_lower else 0.0
            
            # 词覆盖分数
            matched_words = sum(1 for w in query_words if w in doc_lower)
            coverage = matched_words / len(query_words) if query_words else 0
            
            score = exact_match + coverage
            scored.append((doc, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [
            RerankResult(content=doc, score=score, rank=rank)
            for rank, (doc, score) in enumerate(scored[:top_k], 1)
        ]


class RerankerFactory:
    """重排序器工厂"""
    
    @staticmethod
    def create(
        reranker_type: str = "cross-encoder",
        **kwargs
    ) -> BaseReranker:
        """创建重排序器实例"""
        reranker_type = reranker_type.lower()
        
        if reranker_type in ["cross-encoder", "crossencoder", "ce"]:
            return CrossEncoderReranker(**kwargs)
        elif reranker_type in ["simple", "keyword"]:
            return SimpleReranker()
        else:
            raise ValueError(f"Unsupported reranker type: {reranker_type}")
