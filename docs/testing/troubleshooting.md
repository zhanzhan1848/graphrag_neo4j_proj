# GraphRAG 测试故障排除指南

本文档提供了 GraphRAG 项目测试过程中常见问题的解决方案和调试技巧。

## 📋 目录

- [常见测试错误](#常见测试错误)
- [数据库相关问题](#数据库相关问题)
- [API测试问题](#api测试问题)
- [性能测试问题](#性能测试问题)
- [CI/CD问题](#cicd问题)
- [调试技巧](#调试技巧)
- [日志分析](#日志分析)

## 🚨 常见测试错误

### 1. 导入错误 (ImportError)

**错误信息**:
```
ImportError: No module named 'app'
```

**解决方案**:
```bash
# 确保PYTHONPATH正确设置
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 或在pytest.ini中配置
# testpaths = tests
# python_paths = .

# 检查__init__.py文件是否存在
find . -name "__init__.py" -type f
```

### 2. 配置错误 (ConfigurationError)

**错误信息**:
```
pydantic.error_wrappers.ValidationError: DATABASE_URL field required
```

**解决方案**:
```bash
# 检查环境变量
env | grep -E "(DATABASE_URL|NEO4J|REDIS)"

# 复制并配置环境文件
cp .env.example .env
vim .env

# 或设置测试环境变量
export TESTING=true
export DATABASE_URL="postgresql://test_user:test_password@localhost:5432/graphrag_test"
```

### 3. 数据库连接错误

**错误信息**:
```
sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) could not connect to server
```

**解决方案**:
```bash
# 检查PostgreSQL服务状态
pg_isready -h localhost -p 5432

# 启动PostgreSQL服务
brew services start postgresql  # macOS
sudo systemctl start postgresql # Linux

# 检查连接参数
psql -h localhost -p 5432 -U test_user -d graphrag_test

# 创建测试数据库
createdb graphrag_test
```

### 4. 测试数据冲突

**错误信息**:
```
IntegrityError: duplicate key value violates unique constraint
```

**解决方案**:
```python
# 使用事务回滚
@pytest.fixture(autouse=True)
def db_transaction(db_session):
    transaction = db_session.begin()
    yield
    transaction.rollback()

# 或清理测试数据
@pytest.fixture(autouse=True)
def cleanup_data(db_session):
    yield
    db_session.query(Entity).delete()
    db_session.query(Document).delete()
    db_session.commit()
```

### 5. 异步测试错误

**错误信息**:
```
RuntimeError: There is no current event loop in thread
```

**解决方案**:
```python
# 使用pytest-asyncio
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result is not None

# 或配置pytest.ini
# asyncio_mode = auto
```

## 🗄️ 数据库相关问题

### PostgreSQL问题

#### 1. 连接池耗尽

**错误信息**:
```
sqlalchemy.exc.TimeoutError: QueuePool limit of size 5 overflow 10 reached
```

**解决方案**:
```python
# 配置连接池
from sqlalchemy import create_engine

engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=30,
    pool_timeout=30,
    pool_recycle=3600
)

# 确保连接正确关闭
@pytest.fixture
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
```

#### 2. 锁等待超时

**错误信息**:
```
psycopg2.errors.LockNotAvailable: could not obtain lock on row
```

**解决方案**:
```python
# 使用较短的事务
with db_session.begin():
    # 快速操作
    pass

# 避免长时间持有锁
def test_with_timeout():
    with pytest.timeout(30):  # 30秒超时
        # 测试代码
        pass
```

### Neo4j问题

#### 1. 认证失败

**错误信息**:
```
neo4j.exceptions.AuthError: The client is unauthorized due to authentication failure
```

**解决方案**:
```bash
# 重置Neo4j密码
neo4j-admin set-initial-password new_password

# 检查认证配置
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_password

# 测试连接
cypher-shell -u neo4j -p your_password "RETURN 1"
```

#### 2. 事务冲突

**错误信息**:
```
neo4j.exceptions.TransientError: Transaction was rolled back
```

**解决方案**:
```python
# 使用重试机制
from neo4j import GraphDatabase
import time

def with_retry(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except TransientError:
            if attempt == max_retries - 1:
                raise
            time.sleep(0.1 * (2 ** attempt))  # 指数退避
```

## 🌐 API测试问题

### 1. 认证问题

**错误信息**:
```
401 Unauthorized: Invalid authentication credentials
```

**解决方案**:
```python
# 创建测试用户和token
@pytest.fixture
def auth_headers(test_client):
    # 创建测试用户
    response = test_client.post("/api/v1/auth/register", json={
        "email": "test@example.com",
        "password": "testpassword"
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

# 使用认证头
def test_protected_endpoint(test_client, auth_headers):
    response = test_client.get("/api/v1/protected", headers=auth_headers)
    assert response.status_code == 200
```

### 2. 请求超时

**错误信息**:
```
httpx.ReadTimeout: The read operation timed out
```

**解决方案**:
```python
# 增加超时时间
import httpx

@pytest.fixture
def test_client():
    with httpx.Client(
        app=app,
        base_url="http://testserver",
        timeout=30.0  # 30秒超时
    ) as client:
        yield client

# 或使用异步客户端
@pytest.fixture
async def async_client():
    async with httpx.AsyncClient(
        app=app,
        base_url="http://testserver",
        timeout=30.0
    ) as client:
        yield client
```

### 3. JSON序列化错误

**错误信息**:
```
TypeError: Object of type datetime is not JSON serializable
```

**解决方案**:
```python
# 使用自定义JSON编码器
import json
from datetime import datetime

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# 在测试中使用
response_data = json.loads(response.content, cls=DateTimeEncoder)
```

## ⚡ 性能测试问题

### 1. 内存泄漏

**症状**: 测试运行时内存持续增长

**解决方案**:
```python
# 使用内存分析工具
import tracemalloc

@pytest.fixture(autouse=True)
def memory_monitor():
    tracemalloc.start()
    yield
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")

# 显式清理资源
@pytest.fixture
def large_dataset():
    data = create_large_dataset()
    yield data
    del data  # 显式删除
    gc.collect()  # 强制垃圾回收
```

### 2. 测试运行缓慢

**解决方案**:
```bash
# 使用并行测试
pytest -n auto

# 分析慢速测试
pytest --durations=10

# 跳过慢速测试
pytest -m "not slow"

# 使用测试缓存
pytest --lf  # 只运行上次失败的测试
pytest --ff  # 先运行上次失败的测试
```

## 🔄 CI/CD问题

### 1. GitHub Actions超时

**错误信息**:
```
The job running on runner GitHub Actions 2 has exceeded the maximum execution time of 360 minutes
```

**解决方案**:
```yaml
# 增加超时时间
jobs:
  test:
    timeout-minutes: 60  # 增加到60分钟
    
# 或优化测试执行
- name: Run tests
  run: |
    pytest tests/unit/ -n auto  # 并行执行
    pytest tests/integration/ -x  # 遇到错误立即停止
```

### 2. 服务启动失败

**错误信息**:
```
Service postgres didn't start properly
```

**解决方案**:
```yaml
services:
  postgres:
    image: postgres:15
    env:
      POSTGRES_PASSWORD: test_password
    options: >-
      --health-cmd pg_isready
      --health-interval 10s
      --health-timeout 5s
      --health-retries 10  # 增加重试次数
    ports:
      - 5432:5432

# 添加等待步骤
- name: Wait for services
  run: |
    sleep 30
    pg_isready -h localhost -p 5432
```

### 3. 依赖缓存失效

**解决方案**:
```yaml
- name: Cache dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-
      
# 或清理缓存
- name: Clear cache
  run: |
    pip cache purge
    rm -rf ~/.cache/pip
```

## 🔍 调试技巧

### 1. 使用调试器

```python
# 在测试中设置断点
import pdb

def test_complex_function():
    result = complex_function()
    pdb.set_trace()  # 设置断点
    assert result is not None

# 使用ipdb (更好的调试器)
import ipdb
ipdb.set_trace()
```

### 2. 详细日志输出

```python
# 配置测试日志
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 在测试中使用日志
def test_with_logging():
    logger = logging.getLogger(__name__)
    logger.debug("开始测试")
    
    result = function_under_test()
    logger.debug(f"结果: {result}")
    
    assert result is not None
```

### 3. 测试数据检查

```python
# 打印测试数据
def test_data_inspection(db_session):
    entities = db_session.query(Entity).all()
    print(f"实体数量: {len(entities)}")
    for entity in entities[:5]:  # 只打印前5个
        print(f"实体: {entity.name} - {entity.entity_type}")
    
    assert len(entities) > 0
```

### 4. 性能分析

```python
# 使用cProfile
import cProfile
import pstats

def test_performance():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 执行测试代码
    result = expensive_function()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # 打印前10个最耗时的函数
```

## 📊 日志分析

### 1. 结构化日志

```python
# 使用结构化日志
import structlog

logger = structlog.get_logger()

def test_with_structured_logs():
    logger.info("测试开始", test_name="test_example", user_id=123)
    
    try:
        result = function_under_test()
        logger.info("测试成功", result=result)
    except Exception as e:
        logger.error("测试失败", error=str(e), exc_info=True)
        raise
```

### 2. 日志聚合

```bash
# 收集所有测试日志
pytest --log-file=test.log --log-file-level=DEBUG

# 分析错误日志
grep -i error test.log | head -20

# 统计警告
grep -c warning test.log
```

### 3. 实时日志监控

```bash
# 实时查看日志
tail -f test.log

# 过滤特定内容
tail -f test.log | grep -i "database"

# 使用多个终端窗口监控不同组件
tail -f logs/api.log      # 终端1: API日志
tail -f logs/database.log # 终端2: 数据库日志
tail -f logs/neo4j.log    # 终端3: Neo4j日志
```

## 🛠️ 工具推荐

### 1. 调试工具

- **ipdb**: 增强的Python调试器
- **pytest-pdb**: pytest调试插件
- **pytest-xdist**: 并行测试执行
- **pytest-cov**: 代码覆盖率分析

### 2. 性能分析

- **py-spy**: Python性能分析器
- **memory-profiler**: 内存使用分析
- **pytest-benchmark**: 性能基准测试
- **locust**: 负载测试工具

### 3. 日志分析

- **loguru**: 现代Python日志库
- **structlog**: 结构化日志
- **elk-stack**: 日志聚合和分析
- **grafana**: 日志可视化

## 📝 最佳实践

1. **早期发现问题**: 在开发过程中频繁运行测试
2. **隔离问题**: 使用最小化的测试用例重现问题
3. **记录解决方案**: 将解决方案添加到文档中
4. **自动化检查**: 使用CI/CD自动检测常见问题
5. **监控趋势**: 跟踪测试失败率和性能趋势
6. **团队协作**: 分享调试经验和解决方案

## 🆘 获取帮助

如果遇到本文档未涵盖的问题：

1. 检查项目的GitHub Issues
2. 查看相关组件的官方文档
3. 在团队聊天中寻求帮助
4. 创建详细的问题报告

---

更多信息请参考：
- [测试编写指南](test_guide.md)
- [环境配置指南](environment_setup.md)
- [CI/CD配置指南](ci_cd_guide.md)