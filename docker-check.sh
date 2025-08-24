#!/bin/bash

# HippoRAG Docker服务检查脚本
# 只在需要时手动运行，避免频繁检查

set -e

echo "🔍 检查HippoRAG Docker服务状态..."

# 使用完整配置
COMPOSE_FILE="docker-compose.yml"
echo "🔧 使用配置: $COMPOSE_FILE"

# 检查服务运行状态
echo "📊 服务运行状态:"
docker-compose -f $COMPOSE_FILE ps

echo ""

# 检查API服务可用性
echo "💚 检查API服务可用性..."
if curl -f http://localhost:6200/tenants > /dev/null 2>&1; then
    echo "✅ HippoRAG API服务运行正常"
    echo "📍 API地址: http://localhost:6200"
    echo "📚 API文档: http://localhost:6200/docs"
    
    # 显示租户信息
    echo ""
    echo "📋 当前租户信息:"
    curl -s http://localhost:6200/tenants | python3 -m json.tool 2>/dev/null || echo "无法解析租户信息"
else
    echo "❌ HippoRAG API服务不可用"
    echo "💡 请检查服务是否正常启动"
fi

echo ""
echo "🎉 检查完成！"
