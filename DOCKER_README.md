# HippoRAG Docker 部署指南

## 快速开始

### 1. 环境准备
- 安装 [Docker](https://docs.docker.com/get-docker/)
- 安装 [Docker Compose](https://docs.docker.com/compose/install/)

### 2. 配置环境变量（可选）
```bash
# 复制环境变量模板
cp env.example .env

# 编辑.env文件，设置你的配置
vim .env
```

### 3. 启动服务
```bash
# 使用启动脚本（推荐）
./docker-start.sh

# 或手动启动
docker-compose up -d
```

### 4. 验证服务
```bash
# 检查服务状态
./docker-check.sh

# 或直接访问API
curl http://localhost:6200/tenants
```

## 📁 文件说明

- **`Dockerfile`**: HippoRAG API服务的Docker镜像定义
- **`docker-compose.yml`**: 完整的服务编排配置
- **`env.example`**: 环境变量配置模板
- **`docker-start.sh`**: 一键启动脚本
- **`docker-stop.sh`**: 停止服务脚本
- **`docker-check.sh`**: 服务状态检查脚本
- **`.dockerignore`**: Docker构建忽略文件

## 🔧 配置说明

### 服务组成
- **hipporag**: 主API服务 (端口6200)
- **ollama**: 本地模型服务 (端口11434)
- **chromadb**: 向量数据库 (端口8000)

### 环境变量
主要配置项在 `.env` 文件中：

- **`OPENAI_API_KEY`**: OpenAI API密钥（本地模型可设置为任意值）
- **`DEFAULT_LLM`**: 使用的LLM模型名称
- **`DEFAULT_LLM_BASE_URL`**: LLM服务地址（本地模型设置为localhost）
- **`DEFAULT_EMBEDDING_MODEL_NAME`**: 使用的嵌入模型名称
- **`DEFAULT_EMBEDDING_BASE_URL`**: 嵌入服务地址

### 服务端口
- **HippoRAG API**: 6200
- **Ollama**: 11434
- **ChromaDB**: 8000

## 📊 服务状态

### 查看服务状态
```bash
docker-compose ps
```

### 查看日志
```bash
# 查看所有服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f hipporag
```

### 手动检查服务状态
```bash
# 使用检查脚本（推荐）
./docker-check.sh

# 直接检查API服务
curl http://localhost:6200/tenants
```

## 🛠️ 常用命令

```bash
# 启动服务
./docker-start.sh

# 停止服务
./docker-stop.sh

# 重启服务
docker-compose restart

# 查看日志
docker-compose logs -f

# 进入容器
docker-compose exec hipporag bash
```

## 🚨 故障排除

如果遇到问题，请：
1. 检查日志：`docker-compose logs`
2. 查看状态：`docker-compose ps`
3. 重启服务：`docker-compose restart`
4. 提交Issue到项目仓库