"""
工具函数模块
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """保存JSONL文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_sample_documents() -> List[Dict[str, str]]:
    """加载示例文档用于演示"""
    return [
        {
            "content": "向量数据库是专门用于存储和检索向量嵌入的数据库系统，支持高效的近似最近邻搜索。",
            "metadata": {"category": "database", "topic": "vector"}
        },
        {
            "content": "Chroma是一个开源的向量数据库，提供简单易用的Python API，支持持久化存储。",
            "metadata": {"category": "database", "topic": "chroma"}
        },
        {
            "content": "FAISS是Facebook开发的高效相似度搜索库，支持多种索引结构包括HNSW。",
            "metadata": {"category": "library", "topic": "faiss"}
        },
        {
            "content": "BM25是一种基于概率检索模型的关键词匹配算法，广泛应用于传统搜索引擎。",
            "metadata": {"category": "algorithm", "topic": "bm25"}
        },
        {
            "content": "RRF（Reciprocal Rank Fusion）是一种无参数的结果融合算法，用于混合检索系统。",
            "metadata": {"category": "algorithm", "topic": "rrf"}
        },
        {
            "content": "RAG（检索增强生成）结合了检索系统和大语言模型，提供基于事实的问答能力。",
            "metadata": {"category": "ai", "topic": "rag"}
        },
        {
            "content": "HNSW（Hierarchical Navigable Small World）是一种高效的ANN索引结构，构建多层图结构。",
            "metadata": {"category": "algorithm", "topic": "hnsw"}
        },
        {
            "content": "Cross-Encoder是一种重排序模型，直接计算查询和文档的匹配分数。",
            "metadata": {"category": "model", "topic": "cross-encoder"}
        },
        {
            "content": "GraphRAG通过构建知识图谱，实现基于实体和关系的智能检索。",
            "metadata": {"category": "ai", "topic": "graphrag"}
        },
        {
            "content": "nDCG和MRR是信息检索领域常用的评估指标，衡量排序质量。",
            "metadata": {"category": "evaluation", "topic": "metrics"}
        }
    ]
