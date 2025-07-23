#!/usr/bin/env python3
"""
HippoRAGByChroma集成测试
测试使用ChromaDB替代原有EmbeddingStore的内存优化效果
"""

import os
import sys
import time
import psutil
import traceback
from typing import List, Dict, Any, Tuple

# 导入必要的模块
from src.hipporag.MultiTenantHippoRAGManager import MultiTenantHippoRAGManager

def get_memory_usage():
    """获取当前内存使用情况（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def print_memory_usage(label: str, initial_memory: float = None):
    """打印内存使用情况"""
    current_memory = get_memory_usage()
    if initial_memory is not None:
        diff = current_memory - initial_memory
        print(f"📊 {label}: {current_memory:.2f} MB ({diff:+.2f} MB)")
    else:
        print(f"📊 {label}: {current_memory:.2f} MB")
    return current_memory

def check_environment():
    """检查环境和依赖"""
    print("🔍 检查环境...")
    
    # 检查ChromaDB
    try:
        import chromadb
        print("✅ ChromaDB 已安装")
    except ImportError:
        print("❌ ChromaDB 未安装，请运行: pip install chromadb")
        return False
    
    return True

def test_hipporag_chroma():
    """测试HippoRAGByChroma"""
    
    print("🚀 开始测试 HippoRAGByChroma")
    initial_memory = print_memory_usage("初始内存使用")
    
    try:
        tenant_id = 'tenant_1'
        base_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
        manager = MultiTenantHippoRAGManager(base_save_dir=base_save_dir)
        hippo_chroma = manager.get_tenant_config(tenant_id)
        
        # 初始化HippoRAGByChroma
        
        test_docs = [
            "俊佐是一个人工智能研究员，他住在北京。",
            "俊佐是一家公司的创始人。",
        ]
        
        hippo_chroma.index(test_docs)
        
        
        # 测试检索
        print("\n🔍 测试检索...")
        queries = [
            "俊佐有哪些好朋友？",
            "谁住在深圳？",
        ]
        
        retrieval_results = hippo_chroma.retrieve(queries=queries, num_to_retrieve=3)
        # 显示检索结果
        print("\n📋 检索结果:")
        for i, result in enumerate(retrieval_results):
            print(f"   查询 {i+1}: {result.question}")
            print(f"   检索到的文档数: {len(result.docs)}")
            for j, doc in enumerate(result.docs[:2]):  # 显示前2个结果
                print(f"      {j+1}. {doc}")
            
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        print(f"   - 详细错误: {traceback.format_exc()}")
        return False

def main():
    test_hipporag_chroma()

if __name__ == "__main__":
    main()