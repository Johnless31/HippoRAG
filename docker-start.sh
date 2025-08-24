#!/bin/bash

# HippoRAG Docker启动脚本

set -e

echo "🚀 启动HippoRAG Docker服务..."

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

# 检查docker-compose是否安装
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose未安装，请先安装docker-compose"
    exit 1
fi

# 使用完整配置
COMPOSE_FILE="docker-compose.yml"
echo "🔧 使用配置: $COMPOSE_FILE"

# 创建必要的目录
echo "📁 创建必要的目录..."
mkdir -p data logs

# 检查.env文件
if [ ! -f .env ]; then
    echo "⚠️  未找到.env文件，将使用默认配置"
    echo "💡 如需自定义配置，请复制env.example为.env并修改"
else
    echo "✅ 找到.env文件，使用自定义配置"
fi

# 构建镜像
echo "🔨 构建Docker镜像..."
docker-compose -f $COMPOSE_FILE build

# 启动服务
echo "🚀 启动服务..."
docker-compose -f $COMPOSE_FILE up -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 10

# 检查服务状态
echo "🔍 检查服务状态..."
docker-compose -f $COMPOSE_FILE ps

# 显示服务信息
echo ""
echo "🎉 启动完成！"
echo "📍 HippoRAG API地址: http://localhost:6200"
echo "📚 API文档地址: http://localhost:6200/docs"
echo ""
echo "📋 可用命令:"
echo "   - 查看日志: docker-compose -f $COMPOSE_FILE logs -f"
echo "   - 停止服务: docker-compose -f $COMPOSE_FILE down"
echo "   - 重启服务: docker-compose -f $COMPOSE_FILE restart"
echo "   - 查看状态: docker-compose -f $COMPOSE_FILE ps"
echo "   - 检查服务: ./docker-check.sh"
