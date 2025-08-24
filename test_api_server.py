#!/usr/bin/env python3
"""
HippoRAG API服务器测试脚本
测试所有可用的接口功能
"""

import requests
import json
import time
from typing import Dict, Any

# API服务器配置
BASE_URL = "http://localhost:6200"
TENANT_ID = "test_tenant_001"

def test_index_api():
    """测试索引接口"""
    print("🔍 测试索引接口...")
    
    url = f"{BASE_URL}/index"
    data = {
        "tenant_id": TENANT_ID,
        "docs": [
            "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。",
            "深度学习是机器学习的一个分支，使用神经网络来模拟人脑的工作方式。",
            "自然语言处理是人工智能的一个领域，专注于计算机理解和生成人类语言的能力。"
        ]
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 索引接口测试失败: {e}")
        return False

def test_retrieve_api():
    """测试检索接口"""
    print("\n🔍 测试检索接口...")
    
    url = f"{BASE_URL}/retrieve"
    data = {
        "tenant_id": TENANT_ID,
        "querys": [
            "什么是人工智能？",
            "机器学习和深度学习有什么区别？",
            "自然语言处理的应用有哪些？"
        ]
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        # 检查检索结果
        if result.get('code') == 0 and 'data' in result:
            docs = result['data'].get('docs', [])
            print(f"📚 检索到 {len(docs)} 个查询结果:")
            for i, doc_list in enumerate(docs):
                print(f"  查询 {i+1}: {len(doc_list)} 个文档")
                for j, doc in enumerate(doc_list[:2]):  # 只显示前2个文档
                    print(f"    文档 {j+1}: {str(doc)[:100]}...")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 检索接口测试失败: {e}")
        return False

def test_list_tenants_api():
    """测试租户列表接口"""
    print("\n🔍 测试租户列表接口...")
    
    url = f"{BASE_URL}/tenants"
    
    try:
        response = requests.get(url)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        # 检查租户信息
        if result.get('code') == 0 and 'data' in result:
            tenants = result['data'].get('tenants', [])
            print(f"📋 当前有 {len(tenants)} 个租户:")
            for tenant in tenants:
                status = "🟢 活跃" if tenant['active'] else "🔴 非活跃"
                print(f"  {tenant['tenant_id']}: {status}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 租户列表接口测试失败: {e}")
        return False

def test_delete_api():
    """测试删除接口"""
    print("\n🗑️ 测试删除接口...")
    
    url = f"{BASE_URL}/delete"
    data = {
        "tenant_id": TENANT_ID
    }
    
    try:
        response = requests.delete(url, json=data)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 删除接口测试失败: {e}")
        return False

def test_full_workflow():
    """测试完整的工作流程"""
    print("🚀 开始完整工作流程测试...")
    print("=" * 50)
    
    # 1. 创建索引
    print("📚 步骤1: 创建文档索引")
    if not test_index_api():
        print("❌ 索引创建失败，终止测试")
        return False
    
    # 等待一下让索引完成
    print("⏳ 等待索引完成...")
    time.sleep(5)
    
    # 2. 测试检索
    print("\n🔍 步骤2: 测试文档检索")
    if not test_retrieve_api():
        print("❌ 检索测试失败")
        return False
    
    # 3. 查看租户状态
    print("\n📋 步骤3: 查看租户状态")
    if not test_list_tenants_api():
        print("❌ 租户列表获取失败")
        return False
    
    # 4. 删除租户
    print("\n🗑️ 步骤4: 删除租户")
    if not test_delete_api():
        print("❌ 租户删除失败")
        return False
    
    # 5. 再次查看租户状态
    print("\n📋 步骤5: 查看删除后的租户状态")
    if not test_list_tenants_api():
        print("❌ 租户列表获取失败")
        return False
    
    # 6. 重新创建索引
    print("\n📚 步骤6: 重新创建索引")
    if not test_index_api():
        print("❌ 重新索引失败")
        return False
    
    # 7. 再次删除租户
    print("\n🗑️ 步骤7: 再次删除租户")
    if not test_delete_api():
        print("❌ 再次删除失败")
        return False
    
    print("\n🎉 完整工作流程测试完成！")
    return True

def main():
    """主函数"""
    print("🧪 HippoRAG API服务器测试")
    print("=" * 50)
    
    # 检查服务器是否运行
    try:
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code == 200:
            print("✅ API服务器正在运行")
        else:
            print("❌ API服务器响应异常")
            return
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到API服务器，请确保服务器正在运行")
        print("💡 运行命令: python api_server.py")
        return
    
    # 运行完整测试
    success = test_full_workflow()
    
    if success:
        print("\n🎉 所有测试通过！")
    else:
        print("\n❌ 部分测试失败！")

if __name__ == "__main__":
    main() 