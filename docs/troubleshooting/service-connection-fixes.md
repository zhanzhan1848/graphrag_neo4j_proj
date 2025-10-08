# 服务连接问题修复指南

## 概述

本文档记录了在 GraphRAG 系统中遇到的服务连接问题及其解决方案。

## 问题描述

在实施 API 健康检查功能时，发现 Neo4j 和 MinIO 服务连接失败的问题：

- Neo4j 显示为 "disconnected"
- MinIO 显示为 "disconnected"

## 根本原因分析

### 1. 端口映射不一致

**问题**: Docker Compose 配置中的端口映射与应用配置不匹配

- Neo4j 在 docker-compose.yml 中映射为 `7688:7687`（避免端口冲突）
- 但应用配置中仍使用默认端口 `7687`

### 2. 环境变量配置不一致

**问题**: API 容器内的环境变量与 docker-compose.yml 配置不同步

- MinIO 配置使用了 `MINIO_ENDPOINT` 而不是 `MINIO_HOST` 和 `MINIO_PORT`
- Neo4j 端口配置在容器内外不一致

## 解决方案

### 1. 修复 Neo4j 连接配置

#### 修改应用配置文件

在 `app/core/config.py` 中更新 Neo4j 端口配置：

```python
# Neo4j 配置
NEO4J_HOST: str = Field(default="localhost", description="Neo4j 主机地址")
NEO4J_PORT: int = Field(default=7688, description="Neo4j Bolt 端口")  # 修改为映射后的端口
NEO4J_HTTP_PORT: str = Field(default="7475", description="Neo4j HTTP 端口")  # 修改为映射后的端口
NEO4J_BOLT_PORT: str = Field(default="7688", description="Neo4j Bolt 端口")  # 修改为映射后的端口
```

#### 修改 Docker Compose 环境变量

在 `docker-compose.yml` 中的 API 服务配置：

```yaml
# Neo4j 配置
NEO4J_HOST: neo4j
NEO4J_PORT: 7687  # 容器内部端口
NEO4J_USER: ${NEO4J_USER:-neo4j}
NEO4J_PASSWORD: ${NEO4J_PASSWORD:-neo4j123}
NEO4J_DATABASE: ${NEO4J_DATABASE:-graphrag}
```

### 2. 修复 MinIO 连接配置

#### 修改 Docker Compose 环境变量

在 `docker-compose.yml` 中的 API 服务配置：

```yaml
# MinIO 配置
MINIO_HOST: minio  # 使用服务名
MINIO_PORT: 9000   # 容器内部端口
MINIO_ACCESS_KEY: ${MINIO_ROOT_USER:-minioadmin}
MINIO_SECRET_KEY: ${MINIO_ROOT_PASSWORD:-minioadmin123}
MINIO_SECURE: "false"
MINIO_BUCKET: ${MINIO_BUCKET:-graphrag}
```

### 3. 修复 MinIO 异步连接问题

#### 更新健康检查代码

在 `app/utils/health_check.py` 中修复 MinIO 连接方法：

```python
async def check_minio(self) -> str:
    """
    检查 MinIO 对象存储服务连接状态
    
    Returns:
        str: 连接状态 ("connected", "disconnected", "error")
    """
    try:
        # 创建 MinIO 客户端
        client = Minio(
            f"{settings.MINIO_HOST}:{settings.MINIO_PORT}",
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False  # 本地开发环境使用 HTTP
        )
        
        # 测试连接 - 列出存储桶
        # 使用异步方式运行同步的 MinIO 操作
        loop = asyncio.get_event_loop()

## 4. Neo4j 会话资源泄漏问题修复

### 问题描述

在系统运行过程中出现大量 `ResourceWarning: Unclosed AsyncSession` 警告，表明 Neo4j 异步会话未被正确关闭，导致资源泄漏。

### 根本原因分析

#### 1. 错误的异步上下文管理器使用

**问题**: 在 `GraphService` 中错误使用了 `async with` 语法：

```python
# 错误的用法
async with self.connection_manager.get_session() as session:
    # 数据库操作
```

**原因**: `get_session()` 方法返回的是 `AsyncSession` 对象，而不是异步上下文管理器。

#### 2. 缺少会话关闭机制

**问题**: 手动获取的会话没有在 `finally` 块中正确关闭：

```python
# 有问题的代码
session = await self.connection_manager.get_session()
try:
    # 数据库操作
    result = await session.run(query, parameters)
    return result
except Exception as e:
    logger.error(f"操作失败: {str(e)}")
    raise
# 缺少 finally 块来关闭会话
```

### 解决方案

#### 1. 修复会话管理模式

将所有 `async with` 语法替换为手动会话管理：

```python
# 修复后的代码
session = await self.connection_manager.get_session()
try:
    # 数据库操作
    result = await session.run(query, parameters)
    return result
except Exception as e:
    logger.error(f"操作失败: {str(e)}")
    raise
finally:
    await session.close()  # 确保会话关闭
```

#### 2. 修复的具体位置

在 `app/services/graph_service.py` 中修复了以下 12 个方法的会话管理：

**GraphConnectionManager 类**:
- `health_check()` - 第293行

**GraphNodeManager 类**:
- `create_node()` - 第399行
- `get_node()` - 第439行  
- `update_node()` - 第509行
- `delete_node()` - 第562行
- `find_nodes()` - 第632行

**GraphRelationshipManager 类**:
- `create_relationship()` - 第729行
- `get_relationship()` - 第776行
- `update_relationship()` - 第852行
- `delete_relationship()` - 第899行
- `find_relationships()` - 第990行

**GraphService 类**:
- `execute_cypher()` - 第1307行

#### 3. 修复验证

修复完成后验证结果：
- ✅ 所有 12 个 `session` 调用都有对应的 `finally` 块
- ✅ 所有位置都添加了 `await session.close()` 调用
- ✅ 移除了所有错误的 `async with` 异步上下文管理器调用
- ✅ 应用启动时不再出现 `ResourceWarning: Unclosed AsyncSession` 警告

### 最佳实践

#### 1. Neo4j 会话管理规范

```python
# 推荐的会话管理模式
async def database_operation(self):
    """数据库操作的标准模式"""
    session = await self.connection_manager.get_session()
    try:
        # 执行数据库操作
        result = await session.run(query, parameters)
        # 处理结果
        return processed_result
    except Exception as e:
        # 错误处理
        logger.error(f"数据库操作失败: {str(e)}")
        raise
    finally:
        # 确保会话关闭
        await session.close()
```

#### 2. 避免的错误模式

```python
# 错误模式 1: 使用 async with（不适用于 AsyncSession）
async with self.connection_manager.get_session() as session:
    # 这会导致错误

# 错误模式 2: 忘记关闭会话
session = await self.connection_manager.get_session()
result = await session.run(query)
return result  # 会话未关闭，导致资源泄漏

# 错误模式 3: 只在异常情况下关闭会话
session = await self.connection_manager.get_session()
try:
    result = await session.run(query)
    await session.close()  # 只在成功时关闭
    return result
except Exception as e:
    # 异常时会话未关闭
    raise
```

### 监控和预防

#### 1. 资源泄漏监控

在应用启动时启用资源警告监控：

```python
import warnings
import tracemalloc

# 启用内存跟踪
tracemalloc.start()

# 启用资源警告
warnings.filterwarnings("always", category=ResourceWarning)
```

#### 2. 代码审查检查点

在代码审查时检查以下要点：
- [ ] 所有 `session = await ... get_session()` 调用都有对应的 `finally` 块
- [ ] `finally` 块中包含 `await session.close()` 调用
- [ ] 没有使用 `async with` 语法处理 `AsyncSession` 对象
- [ ] 异常处理不会阻止会话关闭
        buckets = await loop.run_in_executor(None, client.list_buckets)
        
        logger.debug(f"MinIO 连接正常，发现 {len(buckets)} 个存储桶")
        return "connected"
        
    except S3Error as e:
        logger.error(f"MinIO S3 错误: {e}")
        return "disconnected"
    except Exception as e:
        logger.error(f"MinIO 连接检查失败: {e}")
        return "disconnected"
```

### 4. 修复 Weaviate 健康检查配置

#### 更新 Docker Compose 健康检查

在 `docker-compose.yml` 中优化 Weaviate 健康检查：

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8080/v1/.well-known/ready"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s  # 增加启动等待时间
```

## 验证步骤

### 1. 重启服务

```bash
# 重新创建 API 容器以应用环境变量更改
docker-compose rm -f api
docker-compose up -d api --no-deps
```

### 2. 检查环境变量

```bash
# 验证容器内环境变量
docker exec graphrag_api env | grep -E "(MINIO|NEO4J)" | sort
```

### 3. 测试健康检查

```bash
# 测试 API 健康检查端点
curl -s http://localhost:8000/api/v1/status | jq .
```

## 预期结果

修复后的健康检查应该返回：

```json
{
  "status": "healthy",
  "timestamp": "2025-10-01T16:20:26.090881Z",
  "uptime": "0d 0h 0m",
  "version": "1.0.0",
  "environment": "development",
  "services": {
    "postgres": "connected",
    "neo4j": "connected",
    "redis": "connected",
    "weaviate": "connected",
    "minio": "connected"
  },
  "system": {
    "cpu_usage": "2.0%",
    "memory_usage": "40.9%",
    "disk_usage": "46.1%"
  }
}
```

## 关键要点

1. **容器内外端口区分**: 容器内部使用标准端口，外部映射可以不同
2. **环境变量一致性**: 确保 docker-compose.yml 中的环境变量与应用代码中使用的变量名一致
3. **异步操作处理**: 对于同步的第三方库，需要使用 `run_in_executor` 在异步环境中运行
4. **健康检查优化**: 适当调整健康检查的间隔和超时时间，特别是对于启动较慢的服务

## 相关文件

- `app/core/config.py` - 应用配置
- `app/utils/health_check.py` - 健康检查实现
- `docker-compose.yml` - 服务编排配置
- `docs/api/health-check.md` - API 文档