# 阶段0：基础架构搭建

## 阶段目标

搭建项目的基础架构，包括Docker环境、数据库服务、基础API框架，确保开发环境可以正常运行。

## 核心任务

### 1. 环境搭建
- [x] 创建项目目录结构
- [ ] 配置Docker Compose环境
- [ ] 搭建PostgreSQL + pgvector
- [ ] 搭建Neo4j图数据库
- [ ] 搭建Redis缓存服务
- [ ] 搭建MinIO对象存储
- [ ] 配置Weaviate向量数据库

### 2. 基础服务
- [ ] 创建FastAPI应用框架
- [ ] 实现基础路由和中间件
- [ ] 配置数据库连接
- [ ] 实现健康检查接口
- [ ] 配置日志系统

### 3. 数据模型
- [ ] 设计PostgreSQL数据表结构
- [ ] 创建数据库迁移脚本
- [ ] 设计Neo4j图模型
- [ ] 配置Weaviate Schema
- [ ] 实现基础数据访问层

## 详细任务清单

```mermaid
gantt
    title 阶段0开发计划
    dateFormat  YYYY-MM-DD
    section 环境搭建
    项目结构创建    :done, struct, 2024-01-01, 1d
    Docker配置     :active, docker, after struct, 2d
    数据库搭建     :db, after docker, 3d
    服务配置       :service, after db, 2d
    
    section 基础服务
    FastAPI框架    :api, after service, 2d
    路由实现       :route, after api, 2d
    数据库连接     :conn, after route, 1d
    健康检查       :health, after conn, 1d
    
    section 数据模型
    表结构设计     :schema, after health, 2d
    迁移脚本       :migrate, after schema, 1d
    图模型设计     :graph, after migrate, 2d
    数据访问层     :dal, after graph, 2d
```

## 技术栈

### 核心技术
- **容器化**: Docker, Docker Compose
- **后端框架**: FastAPI
- **数据库**: PostgreSQL 15 + pgvector
- **图数据库**: Neo4j 5.8
- **向量数据库**: Weaviate
- **对象存储**: MinIO
- **缓存**: Redis 7

### 开发工具
- **Python**: 3.9+
- **包管理**: Poetry/pip
- **代码质量**: Black, isort, flake8
- **测试**: pytest

## 项目结构

```
GraphRAG_NEO_IMG/
├── docker-compose.yml          # Docker编排文件
├── .env                        # 环境变量配置
├── app/                        # FastAPI应用
│   ├── Dockerfile             # 应用容器配置
│   ├── requirements.txt       # Python依赖
│   ├── main.py               # 应用入口
│   ├── api/                  # API路由
│   ├── core/                 # 核心配置
│   ├── db/                   # 数据库相关
│   ├── models/               # 数据模型
│   └── services/             # 业务服务
├── postgres/                   # PostgreSQL配置
│   └── Dockerfile            # 带pgvector的镜像
├── scripts/                    # 初始化脚本
│   ├── init_db.sql           # 数据库初始化
│   └── init_neo4j.cypher     # Neo4j初始化
├── docs/                       # 文档目录
└── tests/                      # 测试文件
```

## 环境配置

### Docker Compose配置
```yaml
version: "3.8"
services:
  postgres:
    build: ./postgres
    environment:
      POSTGRES_USER: rguser
      POSTGRES_PASSWORD: rgpass
      POSTGRES_DB: graphrag
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  neo4j:
    image: neo4j:5.8
    environment:
      NEO4J_AUTH: neo4j/neo4jpass
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4jdata:/data

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  minio:
    image: minio/minio
    command: server /data
    environment:
      MINIO_ROOT_USER: minio
      MINIO_ROOT_PASSWORD: minio123
    ports:
      - "9000:9000"
    volumes:
      - minio_data:/data

  weaviate:
    image: semitechnologies/weaviate:latest
    environment:
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: "true"
      PERSISTENCE_DATA_PATH: /var/lib/weaviate
    ports:
      - "8080:8080"
    volumes:
      - weaviate_data:/var/lib/weaviate

  backend:
    build: ./app
    depends_on:
      - postgres
      - neo4j
      - redis
      - minio
      - weaviate
    environment:
      DATABASE_URL: postgresql://rguser:rgpass@postgres:5432/graphrag
      NEO4J_URI: bolt://neo4j:7687
      NEO4J_USER: neo4j
      NEO4J_PASSWORD: neo4jpass
      REDIS_URL: redis://redis:6379/0
      MINIO_ENDPOINT: minio:9000
      WEAVIATE_URL: http://weaviate:8080
    ports:
      - "8000:8000"
    volumes:
      - ./app:/app

volumes:
  pgdata:
  neo4jdata:
  minio_data:
  weaviate_data:
```

## 验收标准

### 功能验收
- [ ] 所有服务能够正常启动
- [ ] 数据库连接正常
- [ ] API健康检查通过
- [ ] 基础CRUD接口可用
- [ ] 日志系统正常工作

### 性能验收
- [ ] 服务启动时间 < 30秒
- [ ] API响应时间 < 100ms
- [ ] 数据库连接池正常工作
- [ ] 内存使用 < 2GB

### 质量验收
- [ ] 代码覆盖率 > 80%
- [ ] 所有测试用例通过
- [ ] 代码质量检查通过
- [ ] 文档完整性检查通过

## 下一阶段

完成阶段0后，进入[阶段1：MVP功能实现](../phase1/README.md)，开始实现核心的文档处理和知识抽取功能。

## 常见问题

### Q: Docker服务启动失败怎么办？
A: 检查端口占用情况，确保所需端口未被占用。使用`docker-compose logs`查看详细错误信息。

### Q: 数据库连接失败？
A: 确认数据库服务已启动，检查连接字符串和认证信息是否正确。

### Q: 如何重置开发环境？
A: 运行`docker-compose down -v`清理所有容器和数据卷，然后重新启动。