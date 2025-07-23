import numpy as np
import chromadb
from typing import List, Dict, Any, Optional, Union
import logging
import os
from copy import deepcopy

logger = logging.getLogger(__name__)

class ChromaStore:
    """
    使用Chroma作为向量数据库的存储类，替代原来的EmbeddingStore
    """
    
    def __init__(self, embedding_model, db_path: str, batch_size: int, namespace: str):
        """
        初始化ChromaStore
        
        Args:
            embedding_model: embedding模型实例
            db_path: Chroma数据库路径
            batch_size: 批处理大小
            namespace: 命名空间，用于区分不同类型的embedding
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace
        self.db_path = db_path
        
        # 创建数据库目录
        os.makedirs(db_path, exist_ok=True)
        
        # 初始化Chroma客户端
        self.client = chromadb.PersistentClient(path=db_path)
        
        # 创建collection (不预设维度，让Chroma自动适应)
        self.collection = self.client.get_or_create_collection(
            name=f"{namespace}_embeddings",
            metadata={"hnsw:space": "cosine"}  # 使用cosine距离
        )
        
        # 缓存映射关系
        self._id_cache = None
        self._text_to_id_cache = None
        
        logger.info(f"ChromaStore initialized for {namespace} at {db_path}")
    
    def insert_strings(self, texts: List[str]) -> None:
        """
        插入文本字符串，自动生成embedding
        
        Args:
            texts: 要插入的文本列表
        """
        if not texts:
            return
            
        # 生成hash ID
        from .utils.misc_utils import compute_mdhash_id
        hash_ids = [compute_mdhash_id(content=text, prefix=f"{self.namespace}-") for text in texts]
        
        # 检查哪些是新的
        existing_ids = set(self.get_all_ids())
        new_texts = []
        new_ids = []
        
        for text, hash_id in zip(texts, hash_ids):
            if hash_id not in existing_ids:
                new_texts.append(text)
                new_ids.append(hash_id)
        
        if not new_texts:
            logger.info(f"No new texts to insert for {self.namespace}")
            return
            
        logger.info(f"Inserting {len(new_texts)} new texts for {self.namespace}")
        
        # 分批处理
        for i in range(0, len(new_texts), self.batch_size):
            batch_texts = new_texts[i:i + self.batch_size]
            batch_ids = new_ids[i:i + self.batch_size]
            
            # 生成embeddings
            if self.embedding_model is not None:
                embeddings = self.embedding_model.batch_encode(batch_texts, norm=True)
                embeddings = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
            else:
                # 如果没有embedding模型，使用零向量占位（自动适应维度）
                embeddings = None
            
            # 插入到Chroma
            if embeddings is not None:
                self.collection.upsert(
                    embeddings=embeddings,
                    documents=batch_texts,
                    ids=batch_ids
                )
            else:
                # 如果没有embedding模型，只插入文档，不插入embeddings
                self.collection.upsert(
                    documents=batch_texts,
                    ids=batch_ids
                )
        
        # 清除缓存
        self._clear_cache()
        logger.info(f"Successfully inserted {len(new_texts)} texts for {self.namespace}")
    
    def get_all_ids(self) -> List[str]:
        """获取所有ID"""
        if self._id_cache is None:
            try:
                result = self.collection.get()
                self._id_cache = result['ids'] if result['ids'] else []
            except Exception as e:
                logger.warning(f"Failed to get all ids: {e}")
                self._id_cache = []
        return deepcopy(self._id_cache)
    
    def get_all_texts(self) -> List[str]:
        """获取所有文本"""
        try:
            result = self.collection.get()
            return result['documents'] if result['documents'] else []
        except Exception as e:
            logger.warning(f"Failed to get all texts: {e}")
            return []
    
    def get_all_id_to_rows(self) -> Dict[str, Dict[str, str]]:
        """获取ID到行数据的映射"""
        try:
            result = self.collection.get()
            if not result['ids']:
                return {}
            
            id_to_rows = {}
            for i, (id_, doc) in enumerate(zip(result['ids'], result['documents'])):
                id_to_rows[id_] = {
                    'hash_id': id_,
                    'content': doc
                }
            return id_to_rows
        except Exception as e:
            logger.warning(f"Failed to get id to rows mapping: {e}")
            return {}
    
    def get_row(self, hash_id: str) -> Dict[str, str]:
        """获取单个行数据"""
        try:
            result = self.collection.get(ids=[hash_id])
            if result['ids'] and len(result['ids']) > 0:
                return {
                    'hash_id': hash_id,
                    'content': result['documents'][0]
                }
            else:
                raise KeyError(f"Hash ID {hash_id} not found")
        except Exception as e:
            logger.error(f"Failed to get row for {hash_id}: {e}")
            raise
    
    def get_rows(self, hash_ids: List[str]) -> Dict[str, Dict[str, str]]:
        """获取多个行数据"""
        if not hash_ids:
            return {}
        
        try:
            result = self.collection.get(ids=hash_ids)
            rows = {}
            for i, id_ in enumerate(result['ids']):
                rows[id_] = {
                    'hash_id': id_,
                    'content': result['documents'][i]
                }
            return rows
        except Exception as e:
            logger.error(f"Failed to get rows: {e}")
            return {}
    
    def get_embeddings(self, hash_ids: List[str]) -> List[List[float]]:
        """
        获取embeddings，这里实际上不会加载到内存，而是在搜索时按需使用
        为了兼容性，这里返回空列表
        """
        return []
    
    def search(self, query_embedding: Union[List[float], np.ndarray], k: int = 10) -> tuple:
        """
        搜索最相似的向量
        
        Args:
            query_embedding: 查询向量
            k: 返回top-k结果
            
        Returns:
            (indices, distances): 索引和距离
        """
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        try:
            # 获取collection中的文档数量
            total_count = self.collection.count()
            if total_count == 0:
                return ([], [])
                
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, total_count)
            )
            
            # 转换为numpy格式以兼容原有代码
            if results['ids'] and len(results['ids']) > 0:
                # 获取所有ID，计算索引
                all_ids = self.get_all_ids()
                indices = []
                distances = []
                
                for id_, distance in zip(results['ids'][0], results['distances'][0]):
                    if id_ in all_ids:
                        indices.append(all_ids.index(id_))
                        distances.append(1.0 - distance)  # 转换为相似度
                    
                return np.array(indices), np.array(distances)
            else:
                return np.array([]), np.array([])
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return np.array([]), np.array([])
    
    def delete(self, hash_ids: List[str]) -> None:
        """删除指定的embedding"""
        if not hash_ids:
            return
            
        try:
            # 过滤出存在的ID
            existing_ids = set(self.get_all_ids())
            ids_to_delete = [id_ for id_ in hash_ids if id_ in existing_ids]
            
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                self._clear_cache()
                logger.info(f"Deleted {len(ids_to_delete)} records from {self.namespace}")
        except Exception as e:
            logger.error(f"Failed to delete records: {e}")
    
    def count(self) -> int:
        """获取记录数量"""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to get count: {e}")
            return 0
    
    def _clear_cache(self):
        """清除缓存"""
        self._id_cache = None
        self._text_to_id_cache = None
    
    def get_missing_string_hash_ids(self, texts: List[str]) -> Dict[str, str]:
        """获取缺失的字符串hash ID"""
        from .utils.misc_utils import compute_mdhash_id
        
        existing_ids = set(self.get_all_ids())
        missing = {}
        
        for text in texts:
            hash_id = compute_mdhash_id(content=text, prefix=f"{self.namespace}-")
            if hash_id not in existing_ids:
                missing[hash_id] = text
                
        return missing

    @property 
    def text_to_hash_id(self):
        """兼容性属性：文本到hash_id的映射"""
        if self._text_to_id_cache is None:
            result = self.collection.get()
            if result['ids'] and result['documents']:
                self._text_to_id_cache = {doc: id_ for doc, id_ in zip(result['documents'], result['ids'])}
            else:
                self._text_to_id_cache = {}
        return self._text_to_id_cache
    
    def search_text(self, query: str, k: int = 10) -> List[tuple]:
        """
        使用文本查询搜索
        
        Args:
            query: 查询文本
            k: 返回top-k结果
            
        Returns:
            List[(doc_id, score)]: 文档ID和相似度分数的列表
        """
        if self.embedding_model is None:
            logger.warning("No embedding model available for text search")
            return []
        
        try:
            # 生成查询embedding
            query_embedding = self.embedding_model.batch_encode([query], norm=True)[0]
            
            # 搜索
            indices, distances = self.search(query_embedding, k)
            
            # 组合结果
            results = []
            for idx, dist in zip(indices, distances):
                results.append((idx, dist))
            
            return results
        except Exception as e:
            logger.error(f"Failed to search text: {e}")
            return [] 