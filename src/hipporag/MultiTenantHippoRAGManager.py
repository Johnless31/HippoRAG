"""
Multi-tenant Manager for HippoRAG
管理多个租户的HippoRAG实例，提供租户隔离和资源管理功能
"""

import os
import logging
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from .utils.config_utils import BaseConfig
from .HippoRAGByChroma import HippoRAGByChroma
import json
import time

logger = logging.getLogger(__name__)

DEFAULT_LLM = 'deepseek-v3'
DEFAULT_LLM_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
DEFAULT_EMBEDDING_MODEL_NAME = 'bge-m3'
DEFAULT_EMBEDDING_BASE_URL = 'http://localhost:11434/v1'


class MultiTenantHippoRAGManager:
    """
    多租户HippoRAG管理器
    功能:
    1. 管理多个租户的HippoRAG实例
    2. 提供租户隔离
    3. 资源管理和优化
    """
    def __init__(self, base_save_dir:str | None = None, 
                 max_concurrent_tenants: int = 600, 
                 auto_cleanup: bool = True):
        if base_save_dir is None:
            raise ValueError("tenant_config_path is required")
        self.base_save_dir = base_save_dir
        self.tenant_config_path = os.path.join(base_save_dir, 'tenant_config.json') # 所有租户的配置文件
        self.tenant_configs = {}
        self.max_concurrent_tenants = max_concurrent_tenants
        self.auto_cleanup = auto_cleanup
        # 租户实例缓存
        self._tenant_instances: Dict[str, HippoRAGByChroma] = {} # 活跃的租户
        self._access_times: Dict[str, float] = {} # 租户访问时间
        
        # 全局锁
        self._manager_lock = threading.RLock()
        # 确保基础目录存在
        os.makedirs(self.base_save_dir, exist_ok=True)
        logger.info(f"MultiTenantHippoRAGManager initialized with base_dir: {base_save_dir}")
        self.load_tenant_configs()

    def load_tenant_configs(self):
        """加载租户配置"""
        if not os.path.exists(self.tenant_config_path):
            logger.warning(f"Tenant config file not found at {self.tenant_config_path}")
            return
        
        with open(self.tenant_config_path, 'r') as f:
            self.tenant_configs = json.load(f)
    
    def save_tenant_configs(self):
        """保存租户配置"""
        with open(self.tenant_config_path, 'w') as f:
            json.dump(self.tenant_configs, f, indent=4)
        
    
    def get_tenant_config(self, 
                          tenant_id:str, 
                          llm_name:str = DEFAULT_LLM, 
                          llm_base_url:str = DEFAULT_LLM_BASE_URL,
                          embedding_model_name:str = DEFAULT_EMBEDDING_MODEL_NAME,
                          embedding_base_url:str = DEFAULT_EMBEDDING_BASE_URL) -> HippoRAGByChroma:
        with self._manager_lock:
            # 如果不存在，则创建租户配置
            if tenant_id not in self.tenant_configs:
                self.tenant_configs[tenant_id] = {
                    "tenant_id": tenant_id,
                    "save_dir": os.path.join(self.base_save_dir, tenant_id),
                    "llm_name": llm_name,
                    "llm_base_url": llm_base_url,
                    "embedding_model_name": embedding_model_name,
                    "embedding_base_url": embedding_base_url
                }
                self.save_tenant_configs()
            tenant_config = self.tenant_configs[tenant_id]
            # 记录访问时间
            self._access_times[tenant_id] = time.time()
            # 如果实例不存在，创建租户实例
            if tenant_id not in self._tenant_instances:
                # 检查是否需要清理旧实例
                if len(self._tenant_instances) >= self.max_concurrent_tenants:
                    self._cleanup_least_recently_used()
                # 创建租户实例
                logger.debug(f"Created new instance for tenant: {tenant_id}")
                config = BaseConfig()
                config.save_dir = os.path.join(self.base_save_dir, tenant_id)
                config.llm_name = tenant_config['llm_name']
                config.llm_base_url = tenant_config['llm_base_url']
                config.embedding_model_name = tenant_config['embedding_model_name']
                config.embedding_base_url = tenant_config['embedding_base_url']
                instance = HippoRAGByChroma(global_config=config)
                self._tenant_instances[tenant_id] = instance
            return self._tenant_instances[tenant_id]
    
    def _cleanup_least_recently_used(self):
        """清理最久未使用的租户实例"""
        if not self._access_times:
            return
            
        # 找到最久未使用的租户
        oldest_tenant = min(self._access_times.items(), key=lambda x: x[1])[0]
        
        logger.info(f"Cleaning up least recently used tenant: {oldest_tenant}")
        self._remove_tenant_instance(oldest_tenant)

    
    def _remove_tenant_instance(self, tenant_id: str):
        """
        移除租户实例（不删除数据）
        
        Args:
            tenant_id (str): 租户ID
        """
        with self._manager_lock:
            if tenant_id in self._tenant_instances:
                del self._tenant_instances[tenant_id]
                logger.info(f"Removed tenant instance: {tenant_id}")
            
            if tenant_id in self._access_times:
                del self._access_times[tenant_id]
    
    def list_active_tenants(self) -> List[str]:
        """
        列出当前活跃的租户
        
        Returns:
            List[str]: 活跃租户ID列表
        """
        with self._manager_lock:
            return list(self._tenant_instances.keys())
        
    def cleanup_all_instances(self):
        """清理所有租户实例（不删除数据）"""
        with self._manager_lock:
            tenant_ids = list(self._tenant_instances.keys())
            for tenant_id in tenant_ids:
                self._remove_tenant_instance(tenant_id)
            logger.info("Cleaned up all tenant instances")

    def __enter__(self):
        """Context manager入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager出口，清理资源"""
        if self.auto_cleanup:
            self.cleanup_all_instances()
        
    
                  