from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from openai import OpenAI

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig, make_cache_embed

logger = get_logger(__name__)

class BGEM3EmbeddingModel(BaseEmbeddingModel):
    """
    BGE-M3 embedding model implementation using local ollama deployment.
    
    BGE-M3 is a versatile text embedding model that supports:
    - Multi-Functionality: dense retrieval, sparse retrieval, multi-vector retrieval
    - Multi-Linguality: 100+ languages
    - Multi-Granularity: up to 8192 tokens
    
    This implementation uses OpenAI-compatible API through ollama.
    """

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        # 设置默认模型名称或使用传入的模型名称
        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(
                f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")
        else:
            self.embedding_model_name = "bge-m3"  # ollama中部署的模型名称

        # 设置embedding维度（BGE-M3的标准维度）
        self.embedding_dim = 1024

        self._init_embedding_config()

        # 初始化ollama客户端
        logger.debug(
            f"Initializing {self.__class__.__name__}'s embedding model with params: {self.embedding_config.model_init_params}")

        # 使用配置的base_url，如果没有配置则使用默认的ollama地址
        base_url = getattr(self.global_config, 'embedding_base_url', "http://localhost:11434/v1")
        self.client = OpenAI(
            base_url=base_url,
            api_key="ollama"  # ollama不需要真实的API key，但OpenAI客户端需要
        )

    def _init_embedding_config(self) -> None:
        """
        Extract embedding model-specific parameters to init the EmbeddingConfig.
        配置BGE-M3模型的参数
        """

        # 使用配置的base_url，如果没有配置则使用默认的ollama地址
        base_url = getattr(self.global_config, 'embedding_base_url', "http://localhost:11434/v1")
        
        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "embedding_dim": self.embedding_dim,
            "model_init_params": {
                "base_url": base_url,
                "model_name": self.embedding_model_name,
                "api_key": "ollama",
                "max_seq_length": 8192,  # BGE-M3支持的最大序列长度
                "trust_remote_code": True,
                "device_map": "auto",
            },
            "encode_params": {
                "max_length": min(self.global_config.embedding_max_seq_len, 8192),
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32,
                "truncate": True,  # 自动截断过长的文本
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        对文本列表进行编码，返回嵌入向量
        
        Args:
            texts: 要编码的文本列表
            
        Returns:
            numpy数组，形状为 (n_texts, embedding_dim)
        """
        # 文本预处理
        texts = [t.replace("\n", " ") for t in texts]
        texts = [t if t != '' else ' ' for t in texts]
        
        try:
            # 调用ollama API
            response = self.client.embeddings.create(
                input=texts, 
                model=self.embedding_model_name
            )
            
            # 提取嵌入向量
            results = np.array([v.embedding for v in response.data])
            
            logger.debug(f"Generated embeddings for {len(texts)} texts, shape: {results.shape}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during encoding: {e}")
            base_url = getattr(self.global_config, 'embedding_base_url', "http://localhost:11434/v1")
            logger.error(f"Make sure ollama is running and bge-m3 model is available at {base_url}")
            raise

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        批量编码文本，支持大规模文本处理
        
        Args:
            texts: 要编码的文本列表
            **kwargs: 额外的编码参数
            
        Returns:
            numpy数组，形状为 (n_texts, embedding_dim)
        """
        if isinstance(texts, str): 
            texts = [texts]

        # 深拷贝配置参数
        params = deepcopy(self.embedding_config.encode_params)
        if kwargs: 
            params.update(kwargs)

        # 处理指令参数（BGE-M3支持指令引导）
        if "instruction" in kwargs:
            if kwargs["instruction"] != '':
                # BGE-M3的指令格式
                params["instruction"] = f"Instruct: {kwargs['instruction']}\nQuery: "
                # 为每个文本添加指令前缀
                texts = [f"{params['instruction']}{text}" for text in texts]

        logger.debug(f"Calling {self.__class__.__name__} with:\n{params}")

        batch_size = params.pop("batch_size", 16)

        # 根据批大小决定处理方式
        if len(texts) <= batch_size:
            results = self.encode(texts)
        else:
            # 分批处理
            pbar = tqdm(total=len(texts), desc="Batch Encoding BGE-M3")
            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    batch_results = self.encode(batch)
                    results.append(batch_results)
                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                    # 可以选择继续处理其他批次或抛出异常
                    raise
                pbar.update(len(batch))
            pbar.close()
            
            # 合并所有批次的结果
            results = np.concatenate(results, axis=0)

        # 转换为numpy数组
        if isinstance(results, torch.Tensor):
            results = results.cpu().numpy()

        # 归一化处理
        if self.embedding_config.norm:
            # L2归一化
            results = (results.T / np.linalg.norm(results, axis=1)).T
            logger.debug("Applied L2 normalization to embeddings")

        logger.info(f"Successfully encoded {len(texts)} texts with BGE-M3, final shape: {results.shape}")
        
        return results

    def get_query_doc_scores(self, query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """
        计算查询向量与文档向量之间的相似度分数
        
        Args:
            query_vec: 查询向量，形状为 (embedding_dim,)
            doc_vecs: 文档向量矩阵，形状为 (n_docs, embedding_dim)
            
        Returns:
            相似度分数数组，形状为 (n_docs,)
        """
        # 使用余弦相似度计算分数
        if len(query_vec.shape) == 1:
            query_vec = query_vec.reshape(1, -1)
        
        # 计算余弦相似度
        scores = np.dot(doc_vecs, query_vec.T).flatten()
        
        # 如果向量已经归一化，上面的计算就是余弦相似度
        # 如果没有归一化，需要除以向量的模长
        if not self.embedding_config.norm:
            query_norm = np.linalg.norm(query_vec)
            doc_norms = np.linalg.norm(doc_vecs, axis=1)
            scores = scores / (query_norm * doc_norms)
        
        return scores 