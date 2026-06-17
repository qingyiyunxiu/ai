"""
嵌入模型模块 - 支持多种嵌入模型
包括Sentence-BERT、OpenAI嵌入等
"""

from typing import List, Optional, Union
from abc import ABC, abstractmethod
import numpy as np
from loguru import logger


class BaseEmbeddingModel(ABC):
    """嵌入模型基类"""
    
    @abstractmethod
    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> Union[List[float], List[List[float]]]:
        """编码文本为向量"""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """嵌入维度"""
        pass


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """Sentence-Transformer嵌入模型"""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize_embeddings: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._model = None
        self._dimension = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Loading SentenceTransformer model: {self.model_name}")
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded, embedding dimension: {self._dimension}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True
    ) -> Union[List[float], List[List[float]]]:
        """编码文本"""
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=normalize or self.normalize_embeddings,
            convert_to_numpy=True
        )
        
        result = embeddings.tolist()
        return result[0] if single_input else result
    
    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIEmbedding(BaseEmbeddingModel):
    """OpenAI嵌入模型"""
    
    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """初始化OpenAI客户端"""
        from openai import OpenAI
        import os
        
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        base_url = self.base_url or os.getenv("OPENAI_BASE_URL")
        
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"OpenAI embedding client initialized")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True
    ) -> Union[List[float], List[List[float]]]:
        """调用OpenAI API编码"""
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        response = self._client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        
        embeddings = [item.embedding for item in response.data]
        
        if normalize:
            embeddings = [
                (np.array(e) / np.linalg.norm(e)).tolist()
                for e in embeddings
            ]
        
        return embeddings[0] if single_input else embeddings
    
    @property
    def dimension(self) -> int:
        dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
        return dimensions.get(self.model_name, 1536)


class EmbeddingFactory:
    """嵌入模型工厂"""
    
    @staticmethod
    def create(
        model_type: str = "sentence-transformer",
        model_name: Optional[str] = None,
        **kwargs
    ) -> BaseEmbeddingModel:
        """创建嵌入模型实例"""
        model_type = model_type.lower()
        
        if model_type in ["sentence-transformer", "st", "local"]:
            return SentenceTransformerEmbedding(
                model_name=model_name or "all-MiniLM-L6-v2",
                **kwargs
            )
        elif model_type in ["openai", "ada"]:
            return OpenAIEmbedding(
                model_name=model_name or "text-embedding-ada-002",
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported embedding model type: {model_type}")
