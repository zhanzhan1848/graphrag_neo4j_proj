# GraphRAG 测试环境配置指南

本文档详细说明如何配置和管理 GraphRAG 项目的测试环境。

## 📋 目录

- [环境要求](#环境要求)
- [本地开发环境](#本地开发环境)
- [Docker环境](#docker环境)
- [CI/CD环境](#cicd环境)
- [数据库配置](#数据库配置)
- [环境变量](#环境变量)
- [故障排除](#故障排除)

## 🔧 环境要求

### 系统要求

- **操作系统**: macOS 10.15+, Ubuntu 20.04+, Windows 10+
- **Python**: 3.10+ (推荐 3.11)
- **内存**: 最少 8GB RAM (推荐 16GB)
- **存储**: 最少 10GB 可用空间

### 必需软件

```bash
# Python 和包管理
python3.11
pip
virtualenv 或 conda

# 数据库
postgresql (15+)
neo4j (5.15+)
redis (7+)

# 开发工具
git
docker (可选)
docker-compose (可选)
```

## 🏠 本地开发环境

### 1. 克隆项目

```bash
git clone <repository-url>
cd GraphRAG_NEO_IMG
```

### 2. 创建虚拟环境

```bash
# 使用 venv
python3.11 -m venv venv
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate     # Windows

# 使用 conda
conda create -n graphrag python=3.11
conda activate graphrag
```

### 3. 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装测试依赖
pip install -r requirements-test.txt

# 安装开发依赖
pip install -r requirements-dev.txt
```

### 4. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑环境变量
vim .env
```

### 5. 自动化设置

使用提供的脚本自动设置环境：

```bash
# 完整环境设置
./scripts/setup_test_env.sh --all

# 仅设置数据库
./scripts/setup_test_env.sh --db-only

# 使用Docker
./scripts/setup_test_env.sh --docker

# 重置环境
./scripts/setup_test_env.sh --reset
```

## 🐳 Docker环境

### 1. 使用Docker Compose

```bash
# 启动所有服务
docker-compose up -d

# 仅启动数据库服务
docker-compose up -d postgres neo4j redis

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 2. Docker环境配置

```yaml
# docker-compose.test.yml
version: '3.8'

services:
  postgres-test:
    image: postgres:15
    environment:
      POSTGRES_DB: graphrag_test
      POSTGRES_USER: test_user
      POSTGRES_PASSWORD: test_password
    ports:
      - "5433:5432"
    volumes:
      - postgres_test_data:/var/lib/postgresql/data
      - ./init-scripts/postgres:/docker-entrypoint-initdb.d
    
  neo4j-test:
    image: neo4j:5.15
    environment:
      NEO4J_AUTH: neo4j/test_password
      NEO4J_PLUGINS: '["apoc"]'
    ports:
      - "7688:7687"
      - "7475:7474"
    volumes:
      - neo4j_test_data:/data
      
  redis-test:
    image: redis:7-alpine
    ports:
      - "6380:6379"
    volumes:
      - redis_test_data:/data

volumes:
  postgres_test_data:
  neo4j_test_data:
  redis_test_data:
```

### 3. 测试容器管理

```bash
# 启动测试环境
docker-compose -f docker-compose.test.yml up -d

# 运行测试
docker-compose -f docker-compose.test.yml exec app pytest

# 清理测试环境
docker-compose -f docker-compose.test.yml down -v
```

## 🔄 CI/CD环境

### GitHub Actions配置

测试环境在CI/CD中自动配置，包括：

1. **服务容器**: PostgreSQL, Neo4j, Redis
2. **Python环境**: 多版本矩阵测试
3. **依赖缓存**: 加速构建过程
4. **并行执行**: 不同测试类型并行运行

### 环境变量配置

在GitHub仓库设置中配置以下secrets：

```bash
# 数据库配置
DATABASE_URL
NEO4J_URI
NEO4J_USER
NEO4J_PASSWORD
REDIS_URL

# API密钥
AZURE_OPENAI_API_KEY
AZURE_OPENAI_ENDPOINT

# 其他配置
SECRET_KEY
ENVIRONMENT=testing
```

## 🗄️ 数据库配置

### PostgreSQL配置

```bash
# 创建测试数据库
createdb graphrag_test

# 安装pgvector扩展
psql -d graphrag_test -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 运行迁移
python -m alembic upgrade head
```

### Neo4j配置

```bash
# 启动Neo4j
neo4j start

# 创建约束和索引
python scripts/init_neo4j.py

# 验证连接
cypher-shell -u neo4j -p your_password "RETURN 1"
```

### Redis配置

```bash
# 启动Redis
redis-server

# 测试连接
redis-cli ping
```

## 🔐 环境变量

### 测试环境变量

```bash
# .env.test
ENVIRONMENT=testing
DEBUG=true
TESTING=true

# 数据库配置
DATABASE_URL=postgresql://test_user:test_password@localhost:5432/graphrag_test
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=test_password
REDIS_URL=redis://localhost:6379/1

# API配置
API_V1_STR=/api/v1
SECRET_KEY=test-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# 文件存储
UPLOAD_DIR=./test_uploads
MAX_FILE_SIZE=100MB

# 日志配置
LOG_LEVEL=DEBUG
LOG_FORMAT=detailed

# 外部服务（测试时使用mock）
AZURE_OPENAI_API_KEY=test-key
AZURE_OPENAI_ENDPOINT=https://test.openai.azure.com/
AZURE_OPENAI_API_VERSION=2023-12-01-preview
```

### 环境变量优先级

1. 系统环境变量
2. `.env.test` 文件
3. `.env.local` 文件
4. `.env` 文件
5. 默认值

## 🧪 测试数据管理

### 测试数据库

```bash
# 创建测试数据
python -c "
from tests.fixtures.test_data import TestDataFactory
from app.core.database import SessionLocal

with SessionLocal() as session:
    factory = TestDataFactory(session)
    factory.create_sample_data()
    session.commit()
"

# 清理测试数据
python -c "
from app.core.database import engine, Base
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)
"
```

### 测试文件

```bash
# 创建测试文件目录
mkdir -p test_data/documents
mkdir -p test_data/uploads

# 生成测试文件
python tests/fixtures/generate_test_files.py
```

## 🔍 环境验证

### 验证脚本

```bash
# 运行环境检查
python scripts/check_test_env.py

# 或使用make命令
make check-env

# 或使用测试脚本
./scripts/run_tests.sh --check-env
```

### 手动验证

```python
# test_environment.py
import os
import asyncio
from app.core.database import engine, SessionLocal
from app.core.config import settings
from neo4j import GraphDatabase
import redis

async def verify_environment():
    """验证测试环境配置"""
    
    # 检查环境变量
    assert os.getenv('TESTING') == 'true'
    assert settings.environment == 'testing'
    
    # 检查PostgreSQL连接
    with SessionLocal() as session:
        result = session.execute("SELECT 1")
        assert result.scalar() == 1
    
    # 检查Neo4j连接
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password)
    )
    with driver.session() as session:
        result = session.run("RETURN 1 as test")
        assert result.single()["test"] == 1
    driver.close()
    
    # 检查Redis连接
    r = redis.from_url(settings.redis_url)
    assert r.ping()
    
    print("✅ 所有环境检查通过")

if __name__ == "__main__":
    asyncio.run(verify_environment())
```

## 🚨 故障排除

### 常见问题

#### 1. 数据库连接失败

```bash
# 检查服务状态
pg_isready -h localhost -p 5432
neo4j status
redis-cli ping

# 检查端口占用
lsof -i :5432
lsof -i :7687
lsof -i :6379

# 重启服务
brew services restart postgresql
neo4j restart
brew services restart redis
```

#### 2. 权限问题

```bash
# PostgreSQL权限
sudo -u postgres createuser -s $(whoami)
createdb graphrag_test

# 文件权限
chmod +x scripts/*.sh
```

#### 3. 依赖冲突

```bash
# 清理pip缓存
pip cache purge

# 重新安装依赖
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# 使用新的虚拟环境
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 4. Docker问题

```bash
# 清理Docker资源
docker-compose down -v
docker system prune -f

# 重建镜像
docker-compose build --no-cache

# 查看容器日志
docker-compose logs -f postgres
docker-compose logs -f neo4j
```

### 调试技巧

1. **使用详细日志**:
   ```bash
   export LOG_LEVEL=DEBUG
   pytest -v -s
   ```

2. **单独测试组件**:
   ```bash
   pytest tests/unit/test_database/ -v
   pytest tests/integration/test_api/ -v
   ```

3. **使用调试器**:
   ```python
   import pdb; pdb.set_trace()
   ```

4. **检查测试覆盖率**:
   ```bash
   pytest --cov=app --cov-report=html
   open htmlcov/index.html
   ```

## 📊 性能优化

### 测试性能优化

1. **并行测试**:
   ```bash
   pytest -n auto  # 使用所有CPU核心
   pytest -n 4     # 使用4个进程
   ```

2. **测试分组**:
   ```bash
   pytest -m "not slow"  # 跳过慢速测试
   pytest -m "unit"      # 只运行单元测试
   ```

3. **数据库优化**:
   ```python
   # 使用事务回滚而不是删除数据
   @pytest.fixture(autouse=True)
   def db_transaction(db_session):
       transaction = db_session.begin()
       yield
       transaction.rollback()
   ```

4. **缓存优化**:
   ```bash
   # 使用pytest缓存
   pytest --cache-clear  # 清理缓存
   pytest --lf          # 只运行上次失败的测试
   ```

## 📝 最佳实践

1. **环境隔离**: 每个测试类型使用独立的数据库
2. **数据清理**: 测试后自动清理数据
3. **配置管理**: 使用环境变量管理配置
4. **服务健康检查**: 测试前验证服务可用性
5. **资源监控**: 监控测试过程中的资源使用
6. **文档更新**: 及时更新环境配置文档

---

更多信息请参考：
- [测试编写指南](test_guide.md)
- [CI/CD配置指南](ci_cd_guide.md)
- [项目README](../../README.md)