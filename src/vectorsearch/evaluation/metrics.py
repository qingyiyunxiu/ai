"""
检索评估指标模块
实现工业界标准评估指标：
- nDCG (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- Precision@k, Recall@k
- MAP (Mean Average Precision)
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import math
from collections import defaultdict

from loguru import logger


@dataclass
class EvaluationResult:
    """评估结果"""
    ndcg: float
    ndcg_at_k: Dict[int, float]
    mrr: float
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    map_score: float
    query_count: int


class RetrievalMetrics:
    """检索评估指标计算器"""
    
    @staticmethod
    def dcg(relevances: List[int], k: Optional[int] = None) -> float:
        """
        计算DCG (Discounted Cumulative Gain)
        relevances: 相关性标签列表，按排序顺序 [3, 2, 1, 0, ...]
        k: 评估前k个结果
        """
        if k is not None:
            relevances = relevances[:k]
        
        score = 0.0
        for i, rel in enumerate(relevances, 1):
            # DCG公式: rel_i / log2(i + 1)
            score += rel / math.log2(i + 1)
        return score
    
    @staticmethod
    def ndcg(relevances: List[int], k: Optional[int] = None) -> float:
        """
        计算nDCG (Normalized Discounted Cumulative Gain)
        归一化到[0, 1]区间
        """
        if not relevances:
            return 0.0
        
        actual_dcg = RetrievalMetrics.dcg(relevances, k)
        # 理想排序（按相关性降序）
        ideal_relevances = sorted(relevances, reverse=True)
        ideal_dcg = RetrievalMetrics.dcg(ideal_relevances, k)
        
        if ideal_dcg == 0:
            return 0.0
        
        return actual_dcg / ideal_dcg
    
    @staticmethod
    def reciprocal_rank(relevant_ranks: List[int]) -> float:
        """
        计算RR (Reciprocal Rank)
        relevant_ranks: 第一个相关文档的位置列表（从1开始）
        """
        if not relevant_ranks:
            return 0.0
        first_rank = min(relevant_ranks)
        return 1.0 / first_rank
    
    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: set, k: int) -> float:
        """
        计算Precision@k
        retrieved: 检索到的文档ID列表（按排序）
        relevant: 相关文档ID集合
        k: 前k个结果
        """
        if k <= 0:
            return 0.0
        
        retrieved_k = retrieved[:k]
        hits = sum(1 for doc_id in retrieved_k if doc_id in relevant)
        return hits / min(k, len(retrieved_k)) if retrieved_k else 0.0
    
    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: set, k: int) -> float:
        """
        计算Recall@k
        """
        if not relevant:
            return 0.0
        
        retrieved_k = retrieved[:k]
        hits = sum(1 for doc_id in retrieved_k if doc_id in relevant)
        return hits / len(relevant)
    
    @staticmethod
    def average_precision(retrieved: List[str], relevant: set) -> float:
        """
        计算AP (Average Precision)
        """
        if not relevant:
            return 0.0
        
        precisions = []
        for k, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                precisions.append(RetrievalMetrics.precision_at_k(retrieved, relevant, k))
        
        if not precisions:
            return 0.0
        
        return sum(precisions) / len(relevant)


class RetrievalEvaluator:
    """检索系统评估器"""
    
    def __init__(self):
        self.metrics = RetrievalMetrics()
    
    def evaluate(
        self,
        queries_results: List[Dict[str, Any]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> EvaluationResult:
        """
        批量评估多个查询
        queries_results格式: [
            {
                "query": "查询文本",
                "retrieved": ["doc1", "doc2", ...],  # 检索结果（按排序）
                "relevant": {"doc1", "doc3", ...},    # 相关文档集合
                "relevances": [3, 2, 0, ...]          # 相关性分级（可选）
            },
            ...
        ]
        """
        ndcg_scores = []
        ndcg_at_k_scores = defaultdict(list)
        rr_scores = []
        precision_scores = defaultdict(list)
        recall_scores = defaultdict(list)
        ap_scores = []
        
        for qr in queries_results:
            retrieved = qr["retrieved"]
            relevant = qr["relevant"]
            
            # 1. 计算相关性分级（如果没有提供则使用二进制）
            if "relevances" in qr:
                relevances = qr["relevances"]
            else:
                relevances = [1 if doc_id in relevant else 0 for doc_id in retrieved]
            
            # 2. nDCG
            ndcg = self.metrics.ndcg(relevances)
            ndcg_scores.append(ndcg)
            
            for k in k_values:
                ndcg_at_k_scores[k].append(self.metrics.ndcg(relevances, k))
            
            # 3. MRR - 找到第一个相关文档的位置
            relevant_ranks = []
            for rank, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant:
                    relevant_ranks.append(rank)
                    break
            rr_scores.append(self.metrics.reciprocal_rank(relevant_ranks))
            
            # 4. Precision@k, Recall@k
            for k in k_values:
                precision_scores[k].append(
                    self.metrics.precision_at_k(retrieved, relevant, k)
                )
                recall_scores[k].append(
                    self.metrics.recall_at_k(retrieved, relevant, k)
                )
            
            # 5. MAP
            ap_scores.append(self.metrics.average_precision(retrieved, relevant))
        
        # 计算平均值
        return EvaluationResult(
            ndcg=sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
            ndcg_at_k={k: sum(scores) / len(scores) for k, scores in ndcg_at_k_scores.items()},
            mrr=sum(rr_scores) / len(rr_scores) if rr_scores else 0.0,
            precision_at_k={k: sum(scores) / len(scores) for k, scores in precision_scores.items()},
            recall_at_k={k: sum(scores) / len(scores) for k, scores in recall_scores.items()},
            map_score=sum(ap_scores) / len(ap_scores) if ap_scores else 0.0,
            query_count=len(queries_results)
        )
    
    def print_report(self, result: EvaluationResult) -> None:
        """打印评估报告"""
        print("=" * 60)
        print("检索系统评估报告")
        print("=" * 60)
        print(f"查询数量: {result.query_count}")
        print()
        print(f"nDCG (整体): {result.ndcg:.4f}")
        print()
        print("nDCG@k:")
        for k, score in sorted(result.ndcg_at_k.items()):
            print(f"  nDCG@{k}: {score:.4f}")
        print()
        print(f"MRR: {result.mrr:.4f}")
        print(f"MAP: {result.map_score:.4f}")
        print()
        print("Precision@k:")
        for k, score in sorted(result.precision_at_k.items()):
            print(f"  P@{k}: {score:.4f}")
        print()
        print("Recall@k:")
        for k, score in sorted(result.recall_at_k.items()):
            print(f"  R@{k}: {score:.4f}")
        print("=" * 60)
