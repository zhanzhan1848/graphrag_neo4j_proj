# GraphRAG 基础服务 Docker 部署指南

## 概述

本文档详细说明如何使用 Docker Compose 部署 GraphRAG 知识库系统的基础服务，包括 PostgreSQL、Neo4j、Redis、MinIO、Weaviate 和 MinerU。

## 系统要求

### 硬件要求
- **CPU**: 4 核心或以上
- **内存**: 8GB RAM 或以上（推荐 16GB）
- **存储**: 20GB 可用磁盘空间或以上
- **网络**: 稳定的互联网连接

### 软件要求
- **操作系统**: macOS 10.15+, Ubuntu 18.04+, Windows 10+
- **Docker**: 20.10.0 或以上版本
- **Docker Compose**: 2.0.0 或以上版本

## 快速开始

### 1. 环境准备

```bash
# 克隆项目（如果还没有）
git clone <repository-url>
cd GraphRAG_NEO_IMG

# 复制环境变量配置文件
cp .env.example .env

# 根据需要修改 .env 文件中的配置
vim .env
```

### 2. 启动服务

```bash
# 启动所有基础服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看服务日志
docker-compose logs -f
```

### 3. 验证服务

```bash
# 使用 Shell 脚本测试（推荐）
./scripts/test-services.sh

# 或使用 Python 脚本测试（需要安装依赖）
pip install -r scripts/requirements.txt
python scripts/test-services.py
```

## 服务详细配置

### PostgreSQL 数据库

**用途**: 存储文档元数据、文本块和系统配置信息

**默认配置**:
- 端口: 5432
- 数据库: graphrag
- 用户名: graphrag
- 密码: graphrag123

**连接测试**:
```bash
# 使用 psql 连接
psql -h localhost -p 5432 -U graphrag -d graphrag

# 或使用 Docker 容器内的 psql
docker-compose exec postgres psql -U graphrag -d graphrag
```

**数据持久化**: 数据存储在 Docker 卷 `postgres_data` 中

### Neo4j 图数据库

**用途**: 存储实体关系图和知识图谱

**默认配置**:
- HTTP 端口: 7474
- Bolt 端口: 7687
- 用户名: neo4j
- 密码: neo4j123

**Web 界面访问**:
```
http://localhost:7474
```

**连接测试**:
```bash
# 使用 cypher-shell 连接
docker-compose exec neo4j cypher-shell -u neo4j -p neo4j123
```

**数据持久化**: 数据存储在 Docker 卷 `neo4j_data` 中

### Redis 缓存

**用途**: 缓存查询结果、会话管理和临时数据存储

**默认配置**:
- 端口: 6379
- 密码: redis123
- 数据库: 0

**连接测试**:
```bash
# 使用 redis-cli 连接
redis-cli -h localhost -p 6379 -a redis123

# 或使用 Docker 容器内的 redis-cli
docker-compose exec redis redis-cli -a redis123
```

**数据持久化**: 数据存储在 Docker 卷 `redis_data` 中

### MinIO 对象存储

**用途**: 存储原始文档、处理后的文件和媒体资源

**默认配置**:
- API 端口: 9000
- 控制台端口: 9001
- 访问密钥: minioadmin
- 秘密密钥: minioadmin123

**Web 控制台访问**:
```
http://localhost:9001
```

**数据持久化**: 数据存储在 Docker 卷 `minio_data` 中

### Weaviate 向量数据库

**用途**: 语义搜索、向量存储和相似性检索

**默认配置**:
- 端口: 8080
- 认证: 匿名访问已启用

**API 访问**:
```
http://localhost:8080/v1
```

**连接测试**:
```bash
# 检查服务状态
curl http://localhost:8080/v1/.well-known/ready

# 获取元数据
curl http://localhost:8080/v1/meta
```

**数据持久化**: 数据存储在 Docker 卷 `weaviate_data` 中

### MinerU 文档解析服务

**用途**: PDF 解析、OCR 识别和文档结构化处理

**默认配置**:
- 端口: 8501

**Web 界面访问**:
```
http://localhost:8501
```

**数据持久化**: 
- 输入数据: Docker 卷 `mineru_data`
- 输出数据: Docker 卷 `mineru_output`

## 常用操作

### 服务管理

```bash
# 启动所有服务
docker-compose up -d

# 停止所有服务
docker-compose down

# 重启特定服务
docker-compose restart postgres

# 查看服务状态
docker-compose ps

# 查看服务日志
docker-compose logs -f [service_name]

# 进入服务容器
docker-compose exec [service_name] bash
```

### 数据管理

```bash
# 备份 PostgreSQL 数据
docker-compose exec postgres pg_dump -U graphrag graphrag > backup.sql

# 恢复 PostgreSQL 数据
docker-compose exec -T postgres psql -U graphrag graphrag < backup.sql

# 清理所有数据（谨慎操作）
docker-compose down -v
```

### 监控和调试

```bash
# 查看资源使用情况
docker stats

# 查看容器详细信息
docker-compose exec [service_name] ps aux

# 查看网络连接
docker network ls
docker network inspect graphrag_neo_img_graphrag_network
```

## 故障排除

### 常见问题

#### 1. 端口冲突
**问题**: 服务启动失败，提示端口已被占用

**解决方案**:
```bash
# 检查端口占用
lsof -i :5432  # PostgreSQL
lsof -i :7474  # Neo4j HTTP
lsof -i :6379  # Redis
lsof -i :9000  # MinIO
lsof -i :8080  # Weaviate
lsof -i :8501  # MinerU

# 修改 .env 文件中的端口配置
# 或停止占用端口的进程
```

#### 2. 内存不足
**问题**: 服务启动缓慢或异常退出

**解决方案**:
```bash
# 检查系统内存使用
free -h  # Linux
vm_stat  # macOS

# 调整 Docker 内存限制
# 在 Docker Desktop 设置中增加内存分配

# 或在 docker-compose.yml 中添加内存限制
```

#### 3. 数据卷权限问题
**问题**: 服务无法写入数据

**解决方案**:
```bash
# 检查数据卷权限
docker volume inspect graphrag_neo_img_postgres_data

# 修复权限（如果需要）
docker-compose exec postgres chown -R postgres:postgres /var/lib/postgresql/data
```

#### 4. 网络连接问题
**问题**: 服务之间无法通信

**解决方案**:
```bash
# 检查网络配置
docker network ls
docker network inspect graphrag_neo_img_graphrag_network

# 重新创建网络
docker-compose down
docker-compose up -d
```

### 日志分析

```bash
# 查看所有服务日志
docker-compose logs

# 查看特定服务日志
docker-compose logs postgres
docker-compose logs neo4j
docker-compose logs redis
docker-compose logs minio
docker-compose logs weaviate
docker-compose logs mineru

# 实时跟踪日志
docker-compose logs -f --tail=100
```

### 性能优化

#### PostgreSQL 优化
```bash
# 在 docker-compose.yml 中添加性能参数
environment:
  - POSTGRES_INITDB_ARGS=--data-checksums
command: >
  postgres
  -c shared_preload_libraries=pg_stat_statements
  -c max_connections=200
  -c shared_buffers=256MB
  -c effective_cache_size=1GB
```

#### Neo4j 优化
```bash
# 调整内存配置
environment:
  - NEO4J_dbms_memory_heap_initial__size=1G
  - NEO4J_dbms_memory_heap_max__size=4G
  - NEO4J_dbms_memory_pagecache_size=2G
```

#### Redis 优化
```bash
# 调整内存策略
command: >
  redis-server
  --requirepass redis123
  --maxmemory 512mb
  --maxmemory-policy allkeys-lru
  --save 900 1
```

## 安全配置

### 生产环境建议

1. **修改默认密码**:
   ```bash
   # 在 .env 文件中设置强密码
   POSTGRES_PASSWORD=your-strong-password
   NEO4J_PASSWORD=your-strong-password
   REDIS_PASSWORD=your-strong-password
   MINIO_ROOT_PASSWORD=your-strong-password
   ```

2. **网络安全**:
   ```yaml
   # 在 docker-compose.yml 中限制端口暴露
   ports:
     - "127.0.0.1:5432:5432"  # 只允许本地访问
   ```

3. **数据加密**:
   ```bash
   # 启用 PostgreSQL SSL
   environment:
     - POSTGRES_INITDB_ARGS=--auth-host=scram-sha-256
   ```

4. **访问控制**:
   ```bash
   # 配置防火墙规则
   sudo ufw allow from 192.168.1.0/24 to any port 5432
   ```

## 备份和恢复

### 自动备份脚本

```bash
#!/bin/bash
# backup.sh - 自动备份脚本

BACKUP_DIR="/backup/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# 备份 PostgreSQL
docker-compose exec -T postgres pg_dump -U graphrag graphrag > $BACKUP_DIR/postgres.sql

# 备份 Neo4j
docker-compose exec neo4j neo4j-admin dump --database=graphrag --to=/tmp/neo4j-backup.dump
docker cp $(docker-compose ps -q neo4j):/tmp/neo4j-backup.dump $BACKUP_DIR/

# 备份 Redis
docker-compose exec redis redis-cli -a redis123 --rdb /tmp/redis-backup.rdb
docker cp $(docker-compose ps -q redis):/tmp/redis-backup.rdb $BACKUP_DIR/
```

### 恢复数据

```bash
#!/bin/bash
# restore.sh - 数据恢复脚本

BACKUP_DIR="/backup/20240101"  # 指定备份日期

# 恢复 PostgreSQL
docker-compose exec -T postgres psql -U graphrag graphrag < $BACKUP_DIR/postgres.sql

# 恢复 Neo4j
docker cp $BACKUP_DIR/neo4j-backup.dump $(docker-compose ps -q neo4j):/tmp/
docker-compose exec neo4j neo4j-admin load --database=graphrag --from=/tmp/neo4j-backup.dump

# 恢复 Redis
docker cp $BACKUP_DIR/redis-backup.rdb $(docker-compose ps -q redis):/tmp/
docker-compose restart redis
```

## 扩展配置

### 添加新服务

如需添加新的服务到 Docker Compose 配置中：

1. 在 `docker-compose.yml` 中添加服务定义
2. 更新 `.env.example` 文件添加相关环境变量
3. 更新测试脚本 `scripts/test-services.py`
4. 更新本文档

### 集群部署

对于生产环境的集群部署，建议：

1. 使用 Docker Swarm 或 Kubernetes
2. 配置负载均衡器
3. 设置数据复制和高可用
4. 实施监控和告警

## 相关文档

- [API 文档](../api/README.md)
- [开发指南](../development/README.md)
- [监控配置](../monitoring/README.md)
- [故障排除](../troubleshooting/README.md)

## 支持

如遇到问题，请：

1. 查看本文档的故障排除部分
2. 检查服务日志
3. 运行健康检查脚本
4. 提交 Issue 到项目仓库