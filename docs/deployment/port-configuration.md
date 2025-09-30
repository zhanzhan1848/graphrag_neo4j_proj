# 端口配置说明

## 服务端口映射

为了避免与本地服务的端口冲突，本项目使用了以下端口配置：

### PostgreSQL
- **容器端口**: 5432
- **主机端口**: 5432
- **用途**: 数据库连接

### Neo4j
- **HTTP 端口**: 
  - 容器端口: 7474
  - 主机端口: 7475 (避免与本地 Neo4j 冲突)
  - 用途: Neo4j 浏览器界面
- **Bolt 端口**:
  - 容器端口: 7687
  - 主机端口: 7688 (避免与本地 Neo4j 冲突)
  - 用途: Neo4j 数据库连接

### Redis
- **容器端口**: 6379
- **主机端口**: 6379
- **用途**: 缓存服务

### MinIO
- **API 端口**:
  - 容器端口: 9000
  - 主机端口: 9000
  - 用途: S3 兼容 API
- **控制台端口**:
  - 容器端口: 9001
  - 主机端口: 9001
  - 用途: MinIO 管理界面

### Weaviate
- **HTTP 端口**:
  - 容器端口: 8080
  - 主机端口: 8080
  - 用途: REST API
- **gRPC 端口**:
  - 容器端口: 50051
  - 主机端口: 50051
  - 用途: gRPC API

## 访问地址

### Neo4j 浏览器
- **URL**: http://localhost:7475
- **用户名**: neo4j
- **密码**: neo4j123 (默认)

### MinIO 控制台
- **URL**: http://localhost:9001
- **用户名**: minioadmin (默认)
- **密码**: minioadmin (默认)

### Weaviate API
- **REST API**: http://localhost:8080
- **健康检查**: http://localhost:8080/v1/.well-known/ready
- **元数据**: http://localhost:8080/v1/meta

## 连接字符串

### Neo4j 连接
```python
# Python 示例
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7688",
    auth=("neo4j", "neo4j123")
)
```

### PostgreSQL 连接
```python
# Python 示例
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="graphrag",
    user="postgres",
    password="postgres123"
)
```

### Redis 连接
```python
# Python 示例
import redis

r = redis.Redis(
    host='localhost',
    port=6379,
    decode_responses=True
)
```

## 端口冲突解决

如果遇到端口冲突，可以修改 `docker-compose.yml` 中的端口映射：

```yaml
ports:
  - "新端口:容器端口"
```

**注意**: 修改端口后，需要相应更新应用程序中的连接配置。

## 防火墙配置

如果需要从外部访问这些服务，请确保防火墙允许相应端口的访问：

```bash
# macOS 示例 (如果启用了防火墙)
sudo pfctl -f /etc/pf.conf
```

## 安全建议

1. **生产环境**: 修改默认密码
2. **网络隔离**: 使用 Docker 网络进行服务间通信
3. **端口限制**: 只暴露必要的端口到主机
4. **访问控制**: 配置适当的访问控制策略