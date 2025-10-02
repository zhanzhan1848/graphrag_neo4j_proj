# GraphRAG API 健康检查接口文档

## 概述

GraphRAG API 提供了完整的健康检查接口，用于监控系统中各个基础服务的连接状态和系统资源使用情况。

## API 接口

### GET /api/v1/status

获取 API 运行状态，包括所有基础服务的连接状态和系统资源监控。

#### 响应格式

```json
{
    "status": "healthy|degraded|unhealthy|error",
    "timestamp": "2024-01-01T00:00:00Z",
    "uptime": "1d 2h 30m",
    "version": "1.0.0",
    "environment": "development",
    "services": {
        "postgres": "connected|disconnected|timeout|error",
        "neo4j": "connected|disconnected|timeout|error",
        "redis": "connected|disconnected|timeout|error",
        "weaviate": "connected|disconnected|timeout|error",
        "minio": "connected|disconnected|timeout|error"
    },
    "system": {
        "cpu_usage": "15.2%",
        "memory_usage": "45.8%",
        "disk_usage": "23.1%"
    }
}
```

#### 状态说明

**整体状态 (status)**
- `healthy`: 所有服务连接正常
- `degraded`: 部分服务连接异常或超时
- `unhealthy`: 有服务完全无法连接
- `error`: 健康检查过程中发生异常

**服务状态 (services)**
- `connected`: 服务连接正常
- `disconnected`: 服务无法连接
- `timeout`: 服务连接超时
- `error`: 服务检查过程中发生错误

## 监控的服务

### 1. PostgreSQL (postgres)
- **用途**: 文档元数据和文本块存储
- **检查方式**: 执行简单 SQL 查询 `SELECT 1`
- **连接配置**: 使用环境变量中的 PostgreSQL 配置

### 2. Neo4j (neo4j)
- **用途**: 实体关系图存储
- **检查方式**: 验证连接并执行 Cypher 查询 `RETURN 1`
- **连接配置**: 使用 Bolt 协议连接

### 3. Redis (redis)
- **用途**: 缓存和会话管理
- **检查方式**: 执行 PING 命令
- **连接配置**: 支持密码认证

### 4. Weaviate (weaviate)
- **用途**: 向量数据库和语义搜索
- **检查方式**: 检查服务就绪状态
- **连接配置**: HTTP 协议连接

### 5. MinIO (minio)
- **用途**: 对象存储服务
- **检查方式**: 列出存储桶操作
- **连接配置**: S3 兼容 API

## 系统资源监控

使用 `psutil` 库监控系统资源：

- **CPU 使用率**: 实时 CPU 使用百分比
- **内存使用率**: 物理内存使用百分比
- **磁盘使用率**: 根分区磁盘使用百分比

## 实现细节

### 健康检查器类

位置: `app/utils/health_check.py`

主要功能：
- 并发检查所有服务状态（10秒超时）
- 系统资源监控
- 应用运行时间计算
- 异常处理和错误恢复

### 配置管理

所有服务连接配置通过 `app/core/config.py` 管理，支持：
- 环境变量配置
- .env 文件配置
- 默认值设置
- 配置验证

## 使用示例

### 检查 API 状态

```bash
curl -X GET "http://localhost:8000/api/v1/status" \
     -H "accept: application/json"
```

### 响应示例

```json
{
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00Z",
    "uptime": "0d 2h 15m",
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
        "cpu_usage": "12.5%",
        "memory_usage": "38.2%",
        "disk_usage": "45.7%"
    }
}
```

## 故障排除

### 常见问题

1. **服务连接超时**
   - 检查服务是否正在运行
   - 验证网络连接和端口配置
   - 检查防火墙设置

2. **认证失败**
   - 验证用户名和密码配置
   - 检查服务的认证设置

3. **配置错误**
   - 检查环境变量设置
   - 验证 .env 文件配置
   - 确认服务端口和地址

### 调试建议

1. 查看应用日志获取详细错误信息
2. 使用 Docker Compose 检查服务状态
3. 手动测试各服务的连接
4. 验证配置文件的正确性

## 扩展功能

健康检查模块支持以下扩展：

1. **添加新服务检查**: 在 `HealthChecker` 类中添加新的检查方法
2. **自定义超时设置**: 调整各服务的连接超时时间
3. **监控指标扩展**: 添加更多系统资源监控项
4. **告警集成**: 集成外部监控和告警系统