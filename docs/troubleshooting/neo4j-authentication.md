# Neo4j 认证问题排查指南

## 问题描述

在使用 Neo4j 浏览器界面 `http://localhost:7475` 登录时，可能会遇到以下认证错误：

```
Neo.ClientError.Security.Unauthorized: Unsupported authentication token, scheme 'basic' is not supported.
```

## 问题分析

这个错误通常出现在以下情况：

1. **Neo4j 5.x 版本变更**：Neo4j 5.x 对认证机制进行了调整
2. **数据库初始化问题**：认证配置在数据库首次启动时设置，后续修改可能不生效
3. **配置参数错误**：使用了不正确的环境变量名称
4. **数据卷持久化**：旧的认证配置被持久化在数据卷中

## 解决方案

### 1. 完整重置配置

确保 `docker-compose.yml` 中的 Neo4j 配置正确：

```yaml
neo4j:
  image: neo4j:5.15-community
  environment:
    NEO4J_AUTH: ${NEO4J_USER:-neo4j}/${NEO4J_PASSWORD:-neo4j123}
    NEO4J_dbms_default__database: ${NEO4J_DATABASE:-graphrag}
    # 禁用严格配置验证以避免未知配置项错误
    NEO4J_server_config_strict__validation_enabled: "false"
    # 明确启用认证和基本认证方案
    NEO4J_dbms_security_auth__enabled: "true"
```

### 2. 重置数据卷（重要步骤）

由于认证配置在首次启动时设置，需要删除旧数据卷：

```bash
# 停止 Neo4j 服务
docker-compose down neo4j

# 删除数据卷（这会清除所有数据）
docker volume rm graphrag_neo_img_neo4j_data

# 重新启动服务
docker-compose up -d neo4j
```

**⚠️ 警告**: 删除数据卷会清除所有 Neo4j 数据，请确保已备份重要数据。

### 3. 端口配置

为避免与本地 Neo4j 服务冲突，使用不同的端口：

```yaml
ports:
  - "7475:7474"  # HTTP (Neo4j 浏览器)
  - "7688:7687"  # Bolt (数据库连接)
```

## 验证步骤

### 1. 检查容器状态

```bash
docker-compose ps neo4j
```

### 2. 查看启动日志

```bash
docker-compose logs neo4j
```

确认看到以下关键信息：
- `Changed password for user 'neo4j'`
- `Bolt enabled on 0.0.0.0:7687`
- `HTTP enabled on 0.0.0.0:7474`
- `Started.`

### 3. 测试容器内连接

```bash
docker exec graphrag_neo4j cypher-shell -a bolt://localhost:7687 -u neo4j -p neo4j123 "RETURN 'Authentication test successful' AS result;"
```

### 4. 测试浏览器访问

1. 打开浏览器访问: http://localhost:7475
2. 在连接 URL 中输入: `bolt://localhost:7688`
3. 用户名: `neo4j`
4. 密码: `neo4j123`

## 连接方法

### Neo4j 浏览器登录

1. **访问地址**: http://localhost:7475
2. **连接设置**:
   - Connect URL: `bolt://localhost:7688`
   - Authentication type: `Username / Password`
   - Username: `neo4j`
   - Password: `neo4j123`

### 编程语言连接

#### Python 示例

```python
from neo4j import GraphDatabase

# 连接配置
uri = "bolt://localhost:7688"
username = "neo4j"
password = "neo4j123"

# 创建驱动
driver = GraphDatabase.driver(uri, auth=(username, password))

# 测试连接
def test_connection():
    with driver.session() as session:
        result = session.run("RETURN 'Hello, Neo4j!' AS message")
        record = result.single()
        print(record["message"])

test_connection()
driver.close()
```

#### Java 示例

```java
import org.neo4j.driver.*;

public class Neo4jConnection {
    public static void main(String[] args) {
        String uri = "bolt://localhost:7688";
        String username = "neo4j";
        String password = "neo4j123";
        
        try (Driver driver = GraphDatabase.driver(uri, AuthTokens.basic(username, password))) {
            try (Session session = driver.session()) {
                Result result = session.run("RETURN 'Hello, Neo4j!' AS message");
                Record record = result.single();
                System.out.println(record.get("message").asString());
            }
        }
    }
}
```

#### JavaScript/Node.js 示例

```javascript
const neo4j = require('neo4j-driver');

const uri = 'bolt://localhost:7688';
const username = 'neo4j';
const password = 'neo4j123';

const driver = neo4j.driver(uri, neo4j.auth.basic(username, password));

async function testConnection() {
    const session = driver.session();
    try {
        const result = await session.run("RETURN 'Hello, Neo4j!' AS message");
        const record = result.records[0];
        console.log(record.get('message'));
    } finally {
        await session.close();
    }
}

testConnection().then(() => {
    driver.close();
});
```

## 常见问题

### 1. 浏览器连接失败

**症状**: 在 Neo4j 浏览器中输入认证信息后仍然报错

**解决方案**:
1. 确认使用正确的连接 URL: `bolt://localhost:7688`
2. 确认用户名密码: `neo4j` / `neo4j123`
3. 检查容器是否完全启动
4. 重置数据卷并重新启动

### 2. 连接超时

**症状**: 连接时出现超时错误

**解决方案**:
- 检查容器状态: `docker-compose ps neo4j`
- 检查端口映射是否正确
- 确认防火墙设置

### 3. 认证配置不生效

**症状**: 修改配置后认证问题仍然存在

**解决方案**:
- 必须删除数据卷重新初始化
- 认证配置只在首次启动时生效

### 4. 配置警告

Neo4j 5.x 可能会显示配置警告，这些通常不影响功能：

```
WARN  Use of deprecated setting 'dbms.default_database'. It is replaced by 'initial.dbms.default_database'.
```

这些警告可以忽略，或者更新为新的配置参数名称。

## 重要提示

1. **数据备份**: 删除数据卷前请备份重要数据
2. **首次启动**: 认证配置只在数据库首次启动时生效
3. **端口冲突**: 确保端口没有被其他服务占用
4. **完全启动**: 等待 Neo4j 完全启动后再尝试连接

## 相关链接

- [Neo4j 官方文档](https://neo4j.com/docs/)
- [Neo4j Docker 镜像](https://hub.docker.com/_/neo4j)
- [Neo4j 浏览器指南](https://neo4j.com/developer/neo4j-browser/)
- [Neo4j 认证配置](https://neo4j.com/docs/operations-manual/current/authentication-authorization/)