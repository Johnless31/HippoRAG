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
from src.hipporag.HippoRAGByChroma import HippoRAGByChroma
from src.hipporag.utils.config_utils import BaseConfig

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
        # 参考demo_ollama_aq.py的配置
        config = BaseConfig()
        config.save_dir = './outputs'
        config.llm_name = 'deepseek-v3'  # 使用深度求索模型
        config.llm_base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        config.embedding_model_name = 'bge-m3'  # 使用本地embedding模型
        config.embedding_base_url = 'http://localhost:11434/v1'
        
        print("📝 配置信息:")
        print(f"   - LLM: {config.llm_name}")
        print(f"   - LLM Base URL: {config.llm_base_url}")
        print(f"   - Embedding: {config.embedding_model_name}")
        print(f"   - Embedding Base URL: {config.embedding_base_url}")
        print(f"   - 输出目录: {config.save_dir}")
        
        # 初始化HippoRAGByChroma
        print("\n🔧 初始化 HippoRAGByChroma...")
        start_time = time.time()
        
        hippo_chroma = HippoRAGByChroma(global_config=config)
        
        init_time = time.time() - start_time
        after_init_memory = print_memory_usage("初始化完成", initial_memory)
        
        print(f"✅ 初始化完成！")
        print(f"   - 时间: {init_time:.2f}s")
        print(f"   - 内存: {after_init_memory:.2f} MB (+{after_init_memory-initial_memory:.2f} MB)")
        
        # 测试文档索引
        print("\n📚 测试文档索引...")
        test_docs = [
            "俊佐是一个人工智能研究员，他住在北京。"
        ]
        
        start_time = time.time()
        hippo_chroma.index(test_docs)
        index_time = time.time() - start_time
        
        after_index_memory = print_memory_usage("索引完成", initial_memory)
        
        print(f"✅ 索引完成！")
        print(f"   - 时间: {index_time:.2f}s")
        print(f"   - 文档数量: {len(test_docs)}")
        print(f"   - 内存: {after_index_memory:.2f} MB (+{after_index_memory-initial_memory:.2f} MB)")
        
        # 测试检索
        print("\n🔍 测试检索...")
        queries = [
            "俊佐有哪些好朋友？",
            "谁住在深圳？",
        ]
        
        start_time = time.time()
        retrieval_results = hippo_chroma.retrieve(queries=queries, num_to_retrieve=3)
        retrieval_time = time.time() - start_time
        
        after_retrieval_memory = print_memory_usage("检索完成", initial_memory)
        
        print(f"✅ 检索完成！")
        print(f"   - 时间: {retrieval_time:.2f}s")
        print(f"   - 查询数量: {len(queries)}")
        print(f"   - 内存: {after_retrieval_memory:.2f} MB (+{after_retrieval_memory-initial_memory:.2f} MB)")
        
        # 显示检索结果
        print("\n📋 检索结果:")
        for i, result in enumerate(retrieval_results):
            print(f"   查询 {i+1}: {result.question}")
            print(f"   检索到的文档数: {len(result.docs)}")
            for j, doc in enumerate(result.docs[:2]):  # 显示前2个结果
                print(f"      {j+1}. {doc}")
        
        # 内存使用总结
        print("\n" + "=" * 50)
        print("📊 内存使用总结:")
        print(f"   - 初始内存: {initial_memory:.2f} MB")
        print(f"   - 初始化后: {after_init_memory:.2f} MB (+{after_init_memory-initial_memory:.2f} MB)")
        print(f"   - 索引完成: {after_index_memory:.2f} MB (+{after_index_memory-initial_memory:.2f} MB)")
        print(f"   - 检索完成: {after_retrieval_memory:.2f} MB (+{after_retrieval_memory-initial_memory:.2f} MB)")
        
        # 性能总结
        print("\n⚡ 性能总结:")
        print(f"   - 初始化时间: {init_time:.2f}s")
        print(f"   - 索引时间: {index_time:.2f}s")
        print(f"   - 检索时间: {retrieval_time:.2f}s")
        print(f"   - 平均每文档索引时间: {index_time/len(test_docs)*1000:.1f}ms")
        print(f"   - 平均每查询检索时间: {retrieval_time/len(queries)*1000:.1f}ms")
        
        # 成功标准检查
        total_memory_increase = after_retrieval_memory - initial_memory
        success_criteria = [
            total_memory_increase < 1000,  # 总内存增长小于1GB
            index_time < 60,               # 索引时间小于60秒
            retrieval_time < 30,           # 检索时间小于30秒
            len(retrieval_results) == len(queries),  # 所有查询都有结果
            all(len(result.docs) > 0 for result in retrieval_results)  # 所有查询都返回了文档
        ]
        
        print("\n✅ 成功标准检查:")
        print(f"   - 内存增长 < 1GB: {'✅' if success_criteria[0] else '❌'} ({total_memory_increase:.2f} MB)")
        print(f"   - 索引时间 < 60s: {'✅' if success_criteria[1] else '❌'} ({index_time:.2f}s)")
        print(f"   - 检索时间 < 30s: {'✅' if success_criteria[2] else '❌'} ({retrieval_time:.2f}s)")
        print(f"   - 查询结果完整: {'✅' if success_criteria[3] else '❌'} ({len(retrieval_results)}/{len(queries)})")
        print(f"   - 文档检索成功: {'✅' if success_criteria[4] else '❌'}")
        
        if all(success_criteria):
            print("\n🎉 所有测试通过！HippoRAGByChroma成功运行")
            print("💡 关键优势:")
            print("   - 内存优化: 使用ChromaDB避免大量embedding存储在内存中")
            print("   - 高效检索: 向量数据库提供快速相似度搜索")
            print("   - 可扩展性: 支持大规模文档集合")
            return True
        else:
            failed_tests = [i for i, passed in enumerate(success_criteria) if not passed]
            print(f"\n⚠️  部分测试未通过: {failed_tests}")
            return False
            
    except Exception as e:
        error_memory = print_memory_usage("错误时内存", initial_memory)
        print(f"❌ 测试失败: {str(e)}")
        print(f"   - 详细错误: {traceback.format_exc()}")
        return False

def main():
    """主函数"""
    print("🧪 HippoRAGByChroma 集成测试")
    print("=" * 50)
    
    # 检查环境
    if not check_environment():
        print("=" * 50)
        print("❌ 环境检查失败！请先安装必要的依赖")
        return False
    
    # 运行测试
    success = test_hipporag_chroma()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 测试完成！HippoRAGByChroma集成成功")
        return True
    else:
        print("❌ 测试失败！请检查错误信息")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 