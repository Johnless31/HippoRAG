#!/bin/bash

# HippoRAG Docker停止脚本

set -e

echo "🛑 停止HippoRAG Docker服务..."

# 使用完整配置
COMPOSE_FILE="docker-compose.yml"
echo "🔧 使用配置: $COMPOSE_FILE"

# 停止服务
echo "🛑 停止服务..."
docker-compose -f $COMPOSE_FILE down

echo "✅ 服务已停止"

# 询问是否清理数据
read -p "🗑️  是否清理所有数据？这将删除所有租户数据 (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  清理数据..."
    docker-compose -f $COMPOSE_FILE down -v
    rm -rf data/*
    echo "✅ 数据已清理"
else
    echo "💾 数据已保留"
fi

echo "🎉 操作完成！"
