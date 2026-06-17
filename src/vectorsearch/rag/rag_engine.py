"""
RAG检索引擎 - 检索增强生成
支持基础RAG和GraphRAG两种模式
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from ..core.vector_store import Document, BaseVectorStore
from ..core.embeddings import BaseEmbeddingModel
from ..retrieval.hybrid_search import HybridSearchEngine
from ..rerank.cross_encoder import BaseReranker


@dataclass
class RAGResponse:
    """RAG响应结构"""
    answer: str
    retrieved_contexts: List[Dict[str, Any]]
    retrieval_scores: List[float]
    prompt_template: str


class TextSplitter:
    """文本分块器 - 用于文档预处理"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", " "]
    
    def split_text(self, text: str) -> List[str]:
        """递归分割文本"""
        chunks = []
        self._split_recursive(text, chunks, 0)
        return chunks
    
    def _split_recursive(self, text: str, chunks: List[str], depth: int = 0):
        if len(text) <= self.chunk_size:
            chunks.append(text.strip())
            return
        
        if depth >= len(self.separators):
            # 强制分割
            mid = len(text) // 2
            self._split_recursive(text[:mid], chunks, depth)
            self._split_recursive(text[mid:], chunks, depth)
            return
        
        separator = self.separators[depth]
        splits = text.split(separator)
        
        current_chunk = ""
        for split in splits:
            if len(current_chunk) + len(split) + len(separator) <= self.chunk_size:
                current_chunk += (separator if current_chunk else "") + split
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # 如果单个split太大，递归分割
                if len(split) > self.chunk_size:
                    self._split_recursive(split, chunks, depth + 1)
                else:
                    current_chunk = split
        
        if current_chunk:
            chunks.append(current_chunk.strip())


class RAGEngine:
    """基础RAG引擎"""
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_model: BaseEmbeddingModel,
        reranker: Optional[BaseReranker] = None,
        use_hybrid_search: bool = True
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.text_splitter = TextSplitter()
        
        if use_hybrid_search:
            self.search_engine = HybridSearchEngine(
                vector_store=vector_store,
                embedding_model=embedding_model
            )
        else:
            self.search_engine = None
        
        self._context_template = """
基于以下上下文信息回答问题：

上下文：
{context}

问题：{question}

请根据上下文信息，给出准确、简洁的回答。如果上下文没有相关信息，请说明"根据提供的信息无法回答该问题"。
"""
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        split_text: bool = True
    ) -> int:
        """添加文档到RAG系统"""
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        all_documents = []
        doc_id = 0
        
        for text, metadata in zip(texts, metadatas):
            if split_text:
                chunks = self.text_splitter.split_text(text)
            else:
                chunks = [text]
            
            for chunk in chunks:
                embedding = self.embedding_model.encode(chunk)
                doc = Document(
                    id=f"doc_{doc_id}",
                    content=chunk,
                    metadata={**metadata, "chunk_id": doc_id},
                    embedding=embedding
                )
                all_documents.append(doc)
                doc_id += 1
        
        # 添加到向量数据库
        if self.search_engine:
            self.search_engine.index_documents(all_documents)
        else:
            self.vector_store.add_documents(all_documents)
        
        logger.info(f"Added {len(all_documents)} document chunks to RAG")
        return len(all_documents)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """检索相关上下文"""
        # 第一阶段检索
        if self.search_engine:
            results = self.search_engine.search(query, top_k=top_k * 2)
            contexts = [
                {"content": r.content, "score": r.fused_score, "metadata": r.metadata}
                for r in results
            ]
        else:
            query_embedding = self.embedding_model.encode(query)
            results = self.vector_store.search(query_embedding, top_k=top_k * 2)
            contexts = [
                {"content": r.document.content, "score": r.score, "metadata": r.document.metadata}
                for r in results
            ]
        
        # 第二阶段重排序
        if use_rerank and self.reranker and contexts:
            contents = [c["content"] for c in contexts]
            reranked = self.reranker.rerank(query, contents, top_k=top_k)
            
            # 映射回原始上下文
            content_map = {c["content"]: c for c in contexts}
            contexts = [
                {
                    "content": r.content,
                    "score": r.score,
                    "rerank_score": r.score,
                    "metadata": content_map.get(r.content, {}).get("metadata", {})
                }
                for r in reranked
            ]
        else:
            contexts = contexts[:top_k]
        
        return contexts
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        custom_prompt: Optional[str] = None
    ) -> RAGResponse:
        """执行RAG查询"""
        # 检索上下文
        contexts = self.retrieve(question, top_k=top_k)
        
        # 构建提示词
        context_text = "\n---\n".join([
            f"[{i+1}] {c['content']}"
            for i, c in enumerate(contexts)
        ])
        
        prompt_template = custom_prompt or self._context_template
        prompt = prompt_template.format(context=context_text, question=question)
        
        # 这里可以集成LLM生成回答
        # 简化版：返回检索到的上下文和提示词模板
        answer = self._generate_answer(question, contexts)
        
        return RAGResponse(
            answer=answer,
            retrieved_contexts=contexts,
            retrieval_scores=[c.get("score", 0) for c in contexts],
            prompt_template=prompt
        )
    
    def _generate_answer(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        """简化版回答生成（无LLM依赖）"""
        if not contexts:
            return "未找到相关信息。"
        
        relevant_info = "\n".join([
            f"• {c['content'][:200]}..." if len(c['content']) > 200 else f"• {c['content']}"
            for c in contexts[:3]
        ])
        
        return f"检索到以下相关信息：\n\n{relevant_info}"


class GraphRAGEngine(RAGEngine):
    """GraphRAG引擎 - 实体提取 + 知识图谱 + 图遍历
    实现微软GraphRAG的核心思想
    """
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_model: BaseEmbeddingModel,
        reranker: Optional[BaseReranker] = None
    ):
        super().__init__(vector_store, embedding_model, reranker)
        self.entities: Dict[str, List[str]] = {}  # 实体 -> 文档ID列表
        self.relationships: Dict[str, List[str]] = {}  # 实体关系
    
    def extract_entities(self, text: str) -> List[str]:
        """简单实体提取 - 基于NLP规则
        实际项目中可使用spaCy/LLM进行更精准提取
        """
        import re
        
        # 简单规则：提取大写开头的名词短语
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # 专有名词
            r'\b(?:Python|Java|C\+\+|JavaScript|SQL|AI|ML)\b',  # 技术术语
        ]
        
        entities = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.update(matches)
        
        return list(entities)
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        split_text: bool = True
    ) -> int:
        """GraphRAG文档添加 - 同时构建实体图谱"""
        doc_count = super().add_documents(texts, metadatas, split_text)
        
        # 提取实体并构建图谱
        if self.search_engine:
            for doc_id, doc in self.search_engine._document_map.items():
                entities = self.extract_entities(doc.content)
                for entity in entities:
                    if entity not in self.entities:
                        self.entities[entity] = []
                    self.entities[entity].append(doc_id)
        
        logger.info(f"GraphRAG indexed {len(self.entities)} unique entities")
        return doc_count
    
    def retrieve_with_graph(
        self,
        query: str,
        top_k: int = 5,
        graph_expansion: int = 2
    ) -> List[Dict[str, Any]]:
        """基于图谱的检索 - 查询实体 -> 扩展关联实体 -> 定位文档"""
        # 1. 从查询中提取实体
        query_entities = self.extract_entities(query)
        
        # 2. 获取相关文档
        related_doc_ids = set()
        for entity in query_entities:
            if entity in self.entities:
                related_doc_ids.update(self.entities[entity])
        
        # 3. 基础向量检索作为补充
        base_results = self.retrieve(query, top_k=top_k)
        
        # 4. 融合结果
        if related_doc_ids and self.search_engine:
            graph_contexts = []
            for doc_id in related_doc_ids:
                if doc_id in self.search_engine._document_map:
                    doc = self.search_engine._document_map[doc_id]
                    graph_contexts.append({
                        "content": doc.content,
                        "score": 1.0,
                        "source": "graph",
                        "metadata": doc.metadata
                    })
            
            # 合并并去重
            seen_contents = set()
            merged = []
            
            for ctx in graph_contexts + base_results:
                if ctx["content"] not in seen_contents:
                    seen_contents.add(ctx["content"])
                    merged.append(ctx)
            
            return merged[:top_k]
        
        return base_results
