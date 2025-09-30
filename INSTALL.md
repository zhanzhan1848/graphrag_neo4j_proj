# GraphRAG 系统安装指南

本文档提供了 GraphRAG 系统的详细安装指南，包括环境准备、安装步骤、配置说明和常见问题解决方案。

## 📋 系统要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **CPU** | 4 核心 | 8 核心+ |
| **内存** | 8GB RAM | 16GB+ RAM |
| **存储** | 20GB 可用空间 | 100GB+ SSD |
| **网络** | 稳定的互联网连接 | 高速网络连接 |

### 软件要求

| 软件 | 版本要求 | 说明 |
|------|---------|------|
| **操作系统** | Linux/macOS/Windows | 推荐 Ubuntu 20.04+ |
| **Docker** | 20.10+ | 容器运行时 |
| **Docker Compose** | 2.0+ | 服务编排工具 |
| **Git** | 2.0+ | 版本控制 |
| **Python** | 3.12+ | 后端开发环境 |
| **uv** | 最新版本 | Python 包管理器（开发必需） |
| **Node.js** | 16+ | 前端开发（暂缓） |

> **注意**: 基础服务（PostgreSQL, Neo4j, Redis 等）通过 Docker Compose 统一部署，无需单独安装

## 🚀 快速安装

### 方式一：Docker Compose（推荐）

这是最简单的安装方式，**统一部署所有基础服务**：

```bash
# 1. 克隆项目
git clone https://github.com/your-org/GraphRAG_NEO_IMG.git
cd GraphRAG_NEO_IMG

# 2. 复制环境配置文件
cp .env.example .env

# 3. 启动所有基础服务（一键部署）
docker-compose up -d

# 4. 等待服务启动完成（约2-3分钟）
docker-compose logs -f

# 5. 验证安装
curl http://localhost:8000/health
```

**包含的基础服务**：
- **PostgreSQL**: 文档元数据存储
- **Neo4j**: 知识图谱数据库
- **Redis**: 缓存和任务队列
- **MinIO**: 对象存储服务
- **Weaviate**: 向量数据库
- **MinerU OCR**: OCR 识别服务（可选）
- **Prometheus + Grafana**: 监控服务

> **优势**: 无需单独安装和配置各个数据库，一条命令完成所有基础设施部署

### 方式二：开发环境安装

适合需要进行代码开发的用户（**优先后端开发**）：

```bash
# 1. 克隆项目
git clone https://github.com/your-org/GraphRAG_NEO_IMG.git
cd GraphRAG_NEO_IMG

# 2. 启动基础服务（PostgreSQL, Neo4j, Redis 等）
docker-compose -f docker-compose.dev.yml up -d

# 3. 安装 uv 包管理器（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或者使用 pip: pip install uv

# 4. 设置后端开发环境
cd backend
uv venv                    # 创建虚拟环境
source .venv/bin/activate  # 激活虚拟环境 (Windows: .venv\Scripts\activate)
uv pip install -r requirements.txt  # 安装依赖

# 5. 启动后端开发服务器
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# 6. （可选）前端开发环境
cd ../frontend
npm install
npm start
```

> **开发重点**: 当前阶段专注于后端 API 开发，前端开发可暂缓

## ⚙️ 详细配置

### 环境变量配置

编辑 `.env` 文件，配置以下关键参数：

```bash
# 基础配置
PROJECT_NAME=GraphRAG
VERSION=1.0.0
DEBUG=false

# API 配置
API_HOST=0.0.0.0
API_PORT=8000
API_PREFIX=/api

# 数据库配置
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=graphrag
POSTGRES_USER=graphrag
POSTGRES_PASSWORD=your_secure_password

# Neo4j 配置
NEO4J_HOST=neo4j
NEO4J_PORT=7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_password

# Redis 配置
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=your_secure_password

# MinIO 配置
MINIO_HOST=minio
MINIO_PORT=9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=your_secure_password

# Weaviate 配置
WEAVIATE_HOST=weaviate
WEAVIATE_PORT=8080

# AI 服务配置
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-3.5-turbo

# MinerU OCR 服务配置
MINERU_OCR_ENABLED=true
MINERU_OCR_MODE=remote  # remote 或 local
MINERU_OCR_REMOTE_URL=https://api.mineru.com/v1/ocr
MINERU_OCR_API_KEY=your_mineru_api_key
MINERU_OCR_LOCAL_HOST=mineru-ocr
MINERU_OCR_LOCAL_PORT=8080

# 安全配置
SECRET_KEY=your_very_secure_secret_key
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=1440
```

### 服务端口配置

| 服务 | 默认端口 | 说明 | 可修改 |
|------|---------|------|--------|
| FastAPI | 8000 | API 服务 | ✅ |
| PostgreSQL | 5432 | 关系数据库 | ✅ |
| Neo4j | 7474/7687 | 图数据库 | ✅ |
| Redis | 6379 | 缓存和队列 | ✅ |
| MinIO | 9000/9001 | 对象存储 | ✅ |
| Weaviate | 8080 | 向量数据库 | ✅ |
| Grafana | 3001 | 监控面板 | ✅ |
| Prometheus | 9090 | 监控数据 | ✅ |

### 数据持久化配置

默认情况下，所有数据都会持久化到本地目录：

```bash
# 数据目录结构
data/
├── postgres/          # PostgreSQL 数据
├── neo4j/            # Neo4j 数据
├── redis/            # Redis 数据
├── minio/            # MinIO 数据
├── weaviate/         # Weaviate 数据
└── logs/             # 应用日志
```

## 🔧 高级安装选项

### 生产环境部署

对于生产环境，建议使用以下配置（**基础服务仍通过 Docker Compose 统一管理**）：

```bash
# 1. 使用生产配置文件
cp docker-compose.prod.yml docker-compose.yml

# 2. 配置生产环境变量
cp .env.prod .env

# 3. 启动生产环境基础服务
docker-compose up -d

# 4. 配置资源限制和健康检查
docker-compose -f docker-compose.prod.yml up -d
```

**生产环境基础服务配置特点**：
- **资源限制**: 为每个服务设置合理的 CPU 和内存限制
- **健康检查**: 自动监控服务状态并重启异常服务
- **数据备份**: 自动备份 PostgreSQL 和 Neo4j 数据
- **日志管理**: 统一日志收集和轮转
- **安全配置**: 强化数据库密码和网络安全

### 反向代理配置

```bash
# 配置反向代理（Nginx）
sudo apt install nginx
sudo cp configs/nginx/graphrag.conf /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/graphrag.conf /etc/nginx/sites-enabled/
sudo systemctl reload nginx
```

### Kubernetes 部署

对于大规模部署，可以使用 Kubernetes：

```bash
# 1. 安装 kubectl 和 helm
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# 2. 部署到 Kubernetes
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/deployments/
kubectl apply -f k8s/services/

# 3. 验证部署
kubectl get pods -n graphrag
kubectl get services -n graphrag
```

## 🔍 安装验证

### 健康检查

安装完成后，执行以下命令验证系统状态：

```bash
# 1. 检查所有服务状态
docker-compose ps

# 2. 检查 API 健康状态
curl http://localhost:8000/health

# 3. 检查数据库连接
curl http://localhost:8000/health/database

# 4. 检查各个组件
curl http://localhost:8000/health/detailed
```

预期输出：
```json
{
  "status": "healthy",
  "timestamp": "2024-12-19T10:00:00Z",
  "services": {
    "api": "healthy",
    "postgres": "healthy",
    "neo4j": "healthy",
    "redis": "healthy",
    "minio": "healthy",
    "weaviate": "healthy",
    "mineru_ocr": "healthy"
  }
}
```

### 功能测试

```bash
# 1. 测试文档上传
curl -X POST "http://localhost:8000/api/documents/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test-document.pdf"

# 2. 测试 OCR 功能（图片文档）
curl -X POST "http://localhost:8000/api/documents/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test-image.png" \
     -F "use_ocr=true"

# 3. 测试知识查询
curl -X POST "http://localhost:8000/api/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "测试查询", "limit": 5}'

# 4. 测试图谱浏览
curl "http://localhost:8000/api/graph/entities?limit=10"

# 5. 测试 MinerU OCR 服务状态
curl "http://localhost:8000/api/ocr/status"
```

## 🛠️ 故障排除

### 常见问题

#### 1. 端口冲突

**问题**: 服务启动失败，提示端口被占用

**解决方案**:
```bash
# 查看端口占用
sudo netstat -tlnp | grep :8000

# 修改端口配置
vim .env
# 修改 API_PORT=8001

# 重启服务
docker-compose down
docker-compose up -d
```

#### 2. 内存不足

**问题**: 服务启动缓慢或失败

**解决方案**:
```bash
# 检查内存使用
free -h
docker stats

# 调整服务资源限制
vim docker-compose.yml
# 添加或修改 mem_limit 配置

# 重启服务
docker-compose restart
```

#### 3. 数据库连接失败

**问题**: API 无法连接到数据库

**解决方案**:
```bash
# 检查数据库服务状态
docker-compose logs postgres
docker-compose logs neo4j

# 检查网络连接
docker network ls
docker network inspect graphrag_default

# 重置数据库
docker-compose down -v
docker-compose up -d
```

#### 4. 权限问题

**问题**: 文件权限错误

**解决方案**:
```bash
# 修复数据目录权限
sudo chown -R $USER:$USER data/
chmod -R 755 data/

# 修复 Docker socket 权限（Linux）
sudo usermod -aG docker $USER
newgrp docker
```

### 日志查看

```bash
# 查看所有服务日志
docker-compose logs

# 查看特定服务日志
docker-compose logs api
docker-compose logs postgres
docker-compose logs neo4j

# 实时查看日志
docker-compose logs -f api

# 查看最近的日志
docker-compose logs --tail=100 api
```

### 性能调优

#### 数据库优化

```bash
# PostgreSQL 优化
echo "shared_preload_libraries = 'pg_stat_statements'" >> data/postgres/postgresql.conf
echo "max_connections = 200" >> data/postgres/postgresql.conf
echo "shared_buffers = 256MB" >> data/postgres/postgresql.conf

# Neo4j 优化
echo "dbms.memory.heap.initial_size=1G" >> data/neo4j/conf/neo4j.conf
echo "dbms.memory.heap.max_size=2G" >> data/neo4j/conf/neo4j.conf
echo "dbms.memory.pagecache.size=1G" >> data/neo4j/conf/neo4j.conf
```

#### 应用优化

```bash
# 增加 Worker 进程数
vim docker-compose.yml
# 修改 CELERY_WORKERS=4

# 调整 API 并发数
vim docker-compose.yml
# 添加 --workers 4 到 uvicorn 命令
```

## 📚 下一步

安装完成后，建议按以下顺序进行：

1. **[用户指南](docs/user-guide/README.md)** - 学习基本使用方法
2. **[API 文档](docs/api/README.md)** - 了解 API 接口
3. **[开发指南](docs/development/README.md)** - 进行二次开发
4. **[部署指南](docs/deployment/README.md)** - 生产环境部署

## 🆘 获取帮助

如果遇到安装问题，可以通过以下方式获取帮助：

- **文档**: 查看 [完整文档](docs/README.md)
- **Issues**: 提交 [GitHub Issue](https://github.com/your-org/GraphRAG_NEO_IMG/issues)
- **讨论**: 参与 [GitHub Discussions](https://github.com/your-org/GraphRAG_NEO_IMG/discussions)
- **邮箱**: 发送邮件到 support@graphrag.com

---

**祝您使用愉快！** 🎉