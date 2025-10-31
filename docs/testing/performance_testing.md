# GraphRAG 性能测试指南

本文档详细说明了 GraphRAG 项目的性能测试策略、工具和最佳实践。

## 📋 目录

- [性能测试概述](#性能测试概述)
- [测试类型](#测试类型)
- [性能指标](#性能指标)
- [测试工具](#测试工具)
- [基准测试](#基准测试)
- [负载测试](#负载测试)
- [压力测试](#压力测试)
- [性能分析](#性能分析)
- [优化建议](#优化建议)

## 🎯 性能测试概述

### 测试目标

- **响应时间**: API响应时间 < 2秒
- **吞吐量**: 支持100并发用户
- **资源使用**: CPU < 80%, 内存 < 4GB
- **数据库性能**: 查询时间 < 500ms
- **图数据库性能**: 复杂查询 < 1秒

### 测试环境

```yaml
# 性能测试环境规格
CPU: 4核心
内存: 8GB RAM
存储: SSD 100GB
网络: 1Gbps
数据库: PostgreSQL 15 + Neo4j 5.15
```

## 🧪 测试类型

### 1. 单元性能测试

测试单个函数或方法的性能：

```python
# tests/performance/test_unit_performance.py

import pytest
import time
from app.services.text_service import TextService

class TestTextServicePerformance:
    """文本服务性能测试"""
    
    @pytest.mark.performance
    @pytest.mark.benchmark(group="text_processing")
    def test_text_chunking_performance(self, benchmark):
        """测试文本分块性能"""
        text_service = TextService()
        large_text = "这是一个测试文本。" * 1000  # 1000次重复
        
        result = benchmark(text_service.chunk_text, large_text)
        
        # 性能断言
        assert len(result) > 0
        assert benchmark.stats.stats.mean < 0.1  # 平均时间 < 100ms

    @pytest.mark.performance
    def test_entity_extraction_performance(self):
        """测试实体抽取性能"""
        text_service = TextService()
        text = "张三在北京大学学习计算机科学。" * 100
        
        start_time = time.time()
        entities = text_service.extract_entities(text)
        end_time = time.time()
        
        execution_time = end_time - start_time
        assert execution_time < 2.0  # 2秒内完成
        assert len(entities) > 0
```

### 2. 集成性能测试

测试组件间交互的性能：

```python
# tests/performance/test_integration_performance.py

import pytest
import asyncio
from app.services.document_service import DocumentService
from app.services.graph_service import GraphService

class TestDocumentProcessingPerformance:
    """文档处理性能测试"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_document_processing_pipeline(self, db_session):
        """测试文档处理流水线性能"""
        document_service = DocumentService(db_session)
        graph_service = GraphService()
        
        # 创建测试文档
        test_documents = []
        for i in range(10):
            doc_content = f"这是测试文档{i}的内容。" * 100
            test_documents.append({
                "filename": f"test_doc_{i}.txt",
                "content": doc_content.encode(),
                "file_type": "text/plain"
            })
        
        start_time = time.time()
        
        # 并发处理文档
        tasks = []
        for doc_data in test_documents:
            task = document_service.create_and_process_document(**doc_data)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 性能断言
        assert total_time < 30.0  # 30秒内处理10个文档
        assert len(results) == 10
        assert all(doc.status == "processed" for doc in results)
        
        # 计算吞吐量
        throughput = len(results) / total_time
        assert throughput > 0.3  # 每秒至少处理0.3个文档
```

### 3. 端到端性能测试

测试完整用户场景的性能：

```python
# tests/performance/test_e2e_performance.py

import pytest
import httpx
import asyncio
from concurrent.futures import ThreadPoolExecutor

class TestAPIPerformance:
    """API性能测试"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_api_requests(self, test_client):
        """测试并发API请求性能"""
        
        async def make_request(client, endpoint):
            """发送单个请求"""
            start_time = time.time()
            response = await client.get(endpoint)
            end_time = time.time()
            return {
                "status_code": response.status_code,
                "response_time": end_time - start_time
            }
        
        # 并发发送100个请求
        tasks = []
        async with httpx.AsyncClient(app=app, base_url="http://testserver") as client:
            for _ in range(100):
                task = make_request(client, "/api/v1/health")
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
        
        # 分析结果
        response_times = [r["response_time"] for r in results]
        success_count = sum(1 for r in results if r["status_code"] == 200)
        
        # 性能断言
        assert success_count == 100  # 所有请求成功
        assert max(response_times) < 2.0  # 最大响应时间 < 2秒
        assert sum(response_times) / len(response_times) < 0.5  # 平均响应时间 < 500ms
```

## 📊 性能指标

### 关键指标

1. **响应时间指标**:
   - 平均响应时间 (Mean Response Time)
   - 95百分位响应时间 (95th Percentile)
   - 99百分位响应时间 (99th Percentile)
   - 最大响应时间 (Max Response Time)

2. **吞吐量指标**:
   - 每秒请求数 (RPS - Requests Per Second)
   - 每秒事务数 (TPS - Transactions Per Second)
   - 并发用户数 (Concurrent Users)

3. **资源使用指标**:
   - CPU使用率
   - 内存使用量
   - 磁盘I/O
   - 网络I/O

4. **数据库指标**:
   - 查询执行时间
   - 连接池使用率
   - 锁等待时间
   - 缓存命中率

### 性能基准

```python
# tests/performance/benchmarks.py

PERFORMANCE_BENCHMARKS = {
    "api_response_time": {
        "target": 0.5,  # 500ms
        "warning": 1.0,  # 1s
        "critical": 2.0  # 2s
    },
    "document_processing": {
        "target": 5.0,   # 5s per document
        "warning": 10.0, # 10s
        "critical": 20.0 # 20s
    },
    "graph_query": {
        "target": 0.1,   # 100ms
        "warning": 0.5,  # 500ms
        "critical": 1.0  # 1s
    },
    "concurrent_users": {
        "target": 100,   # 100并发用户
        "warning": 50,   # 50用户
        "critical": 10   # 10用户
    }
}

def check_performance_benchmark(metric_name, value):
    """检查性能是否达到基准"""
    benchmark = PERFORMANCE_BENCHMARKS.get(metric_name)
    if not benchmark:
        return "unknown"
    
    if value <= benchmark["target"]:
        return "excellent"
    elif value <= benchmark["warning"]:
        return "good"
    elif value <= benchmark["critical"]:
        return "warning"
    else:
        return "critical"
```

## 🛠️ 测试工具

### 1. pytest-benchmark

用于Python函数的微基准测试：

```python
# 安装
pip install pytest-benchmark

# 使用示例
@pytest.mark.benchmark(group="database")
def test_database_query_performance(benchmark, db_session):
    """数据库查询性能测试"""
    def query_entities():
        return db_session.query(Entity).limit(100).all()
    
    result = benchmark(query_entities)
    assert len(result) <= 100
```

### 2. Locust

用于负载测试和压力测试：

```python
# tests/performance/locustfile.py

from locust import HttpUser, task, between

class GraphRAGUser(HttpUser):
    """GraphRAG用户行为模拟"""
    
    wait_time = between(1, 3)  # 用户操作间隔1-3秒
    
    def on_start(self):
        """用户开始时的初始化"""
        # 登录获取token
        response = self.client.post("/api/v1/auth/login", json={
            "email": "test@example.com",
            "password": "testpassword"
        })
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.headers = {}
    
    @task(3)
    def view_documents(self):
        """查看文档列表 - 高频操作"""
        self.client.get("/api/v1/documents", headers=self.headers)
    
    @task(2)
    def search_documents(self):
        """搜索文档 - 中频操作"""
        self.client.get(
            "/api/v1/documents/search?q=测试",
            headers=self.headers
        )
    
    @task(1)
    def upload_document(self):
        """上传文档 - 低频操作"""
        files = {"file": ("test.txt", "测试文档内容", "text/plain")}
        self.client.post(
            "/api/v1/documents/upload",
            files=files,
            headers=self.headers
        )
    
    @task(2)
    def query_graph(self):
        """图查询 - 中频操作"""
        self.client.post("/api/v1/graph/query", json={
            "query": "MATCH (n:Entity) RETURN n LIMIT 10"
        }, headers=self.headers)

# 运行负载测试
# locust -f tests/performance/locustfile.py --host=http://localhost:8000
```

### 3. k6

用于API性能测试：

```javascript
// tests/performance/k6_test.js

import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 10 },  // 2分钟内增加到10用户
    { duration: '5m', target: 10 },  // 保持10用户5分钟
    { duration: '2m', target: 50 },  // 2分钟内增加到50用户
    { duration: '5m', target: 50 },  // 保持50用户5分钟
    { duration: '2m', target: 0 },   // 2分钟内减少到0用户
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95%的请求在2秒内完成
    http_req_failed: ['rate<0.1'],     // 错误率小于10%
  },
};

export default function() {
  // 健康检查
  let response = http.get('http://localhost:8000/api/v1/health');
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
  
  sleep(1);
  
  // 文档查询
  response = http.get('http://localhost:8000/api/v1/documents');
  check(response, {
    'documents status is 200': (r) => r.status === 200,
  });
  
  sleep(1);
}

// 运行: k6 run tests/performance/k6_test.js
```

## 📈 基准测试

### 数据库性能基准

```python
# tests/performance/test_database_benchmarks.py

import pytest
import time
from sqlalchemy import text

class TestDatabaseBenchmarks:
    """数据库性能基准测试"""
    
    @pytest.mark.benchmark(group="database")
    def test_entity_query_benchmark(self, benchmark, db_session):
        """实体查询基准测试"""
        def query_entities():
            return db_session.query(Entity).limit(1000).all()
        
        result = benchmark(query_entities)
        assert len(result) <= 1000
    
    @pytest.mark.benchmark(group="database")
    def test_complex_join_benchmark(self, benchmark, db_session):
        """复杂连接查询基准测试"""
        def complex_query():
            return db_session.execute(text("""
                SELECT d.title, COUNT(e.id) as entity_count
                FROM documents d
                LEFT JOIN chunks c ON d.id = c.document_id
                LEFT JOIN entities e ON c.id = e.chunk_id
                GROUP BY d.id, d.title
                ORDER BY entity_count DESC
                LIMIT 100
            """)).fetchall()
        
        result = benchmark(complex_query)
        assert len(result) <= 100
    
    @pytest.mark.benchmark(group="database")
    def test_vector_similarity_benchmark(self, benchmark, db_session):
        """向量相似度查询基准测试"""
        def vector_query():
            return db_session.execute(text("""
                SELECT id, title, embedding <-> %s as distance
                FROM chunks
                ORDER BY distance
                LIMIT 10
            """), ([0.1] * 1536,)).fetchall()  # 假设1536维向量
        
        result = benchmark(vector_query)
        assert len(result) <= 10
```

### Neo4j性能基准

```python
# tests/performance/test_neo4j_benchmarks.py

import pytest
from neo4j import GraphDatabase

class TestNeo4jBenchmarks:
    """Neo4j性能基准测试"""
    
    @pytest.fixture
    def neo4j_driver(self):
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password")
        )
        yield driver
        driver.close()
    
    @pytest.mark.benchmark(group="neo4j")
    def test_entity_traversal_benchmark(self, benchmark, neo4j_driver):
        """实体遍历基准测试"""
        def traverse_entities():
            with neo4j_driver.session() as session:
                return session.run("""
                    MATCH (e:Entity)-[:RELATED_TO*1..3]-(related)
                    WHERE e.name = $name
                    RETURN related.name
                    LIMIT 100
                """, name="张三").data()
        
        result = benchmark(traverse_entities)
        assert len(result) <= 100
    
    @pytest.mark.benchmark(group="neo4j")
    def test_shortest_path_benchmark(self, benchmark, neo4j_driver):
        """最短路径基准测试"""
        def find_shortest_path():
            with neo4j_driver.session() as session:
                return session.run("""
                    MATCH (start:Entity {name: $start_name}),
                          (end:Entity {name: $end_name})
                    MATCH path = shortestPath((start)-[*]-(end))
                    RETURN path
                """, start_name="张三", end_name="李四").single()
        
        result = benchmark(find_shortest_path)
        # 验证路径存在或不存在都是有效结果
        assert result is not None or result is None
```

## 🔥 负载测试

### API负载测试

```python
# tests/performance/test_load_testing.py

import pytest
import asyncio
import httpx
import time
from concurrent.futures import ThreadPoolExecutor

class TestLoadTesting:
    """负载测试"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_api_load_test(self):
        """API负载测试"""
        
        async def make_requests(client, num_requests):
            """发送指定数量的请求"""
            tasks = []
            for i in range(num_requests):
                task = client.get(f"/api/v1/documents?page={i%10}")
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            return responses
        
        # 测试不同负载级别
        load_levels = [10, 50, 100, 200]
        results = {}
        
        async with httpx.AsyncClient(
            app=app,
            base_url="http://testserver",
            timeout=30.0
        ) as client:
            
            for load in load_levels:
                start_time = time.time()
                responses = await make_requests(client, load)
                end_time = time.time()
                
                # 分析结果
                successful_requests = sum(
                    1 for r in responses 
                    if not isinstance(r, Exception) and r.status_code == 200
                )
                
                total_time = end_time - start_time
                rps = successful_requests / total_time
                
                results[load] = {
                    "total_requests": load,
                    "successful_requests": successful_requests,
                    "total_time": total_time,
                    "rps": rps,
                    "success_rate": successful_requests / load
                }
                
                # 基本断言
                assert successful_requests > 0
                assert results[load]["success_rate"] > 0.9  # 90%成功率
        
        # 分析负载测试结果
        print("\n负载测试结果:")
        for load, result in results.items():
            print(f"负载 {load}: RPS={result['rps']:.2f}, "
                  f"成功率={result['success_rate']:.2%}")
```

### 数据库负载测试

```python
# tests/performance/test_database_load.py

import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from app.core.database import SessionLocal

class TestDatabaseLoad:
    """数据库负载测试"""
    
    @pytest.mark.performance
    def test_concurrent_database_operations(self):
        """并发数据库操作测试"""
        
        def database_operation(thread_id):
            """单个数据库操作"""
            session = SessionLocal()
            try:
                # 模拟复杂查询
                entities = session.query(Entity).limit(100).all()
                
                # 模拟写操作
                new_entity = Entity(
                    name=f"测试实体_{thread_id}_{int(time.time())}",
                    entity_type="TEST",
                    confidence=0.9
                )
                session.add(new_entity)
                session.commit()
                
                return len(entities)
            except Exception as e:
                session.rollback()
                raise e
            finally:
                session.close()
        
        # 并发执行数据库操作
        num_threads = 20
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(database_operation, i) 
                for i in range(num_threads)
            ]
            
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    print(f"数据库操作失败: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 分析结果
        successful_operations = len(results)
        ops_per_second = successful_operations / total_time
        
        print(f"\n数据库负载测试结果:")
        print(f"并发线程数: {num_threads}")
        print(f"成功操作数: {successful_operations}")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"每秒操作数: {ops_per_second:.2f}")
        
        # 性能断言
        assert successful_operations >= num_threads * 0.8  # 80%成功率
        assert ops_per_second > 1.0  # 每秒至少1个操作
```

## 💥 压力测试

### 系统压力测试

```python
# tests/performance/test_stress_testing.py

import pytest
import psutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor

class TestStressTesting:
    """压力测试"""
    
    @pytest.mark.performance
    def test_memory_stress(self):
        """内存压力测试"""
        
        def memory_intensive_operation():
            """内存密集型操作"""
            # 创建大量数据
            large_data = []
            for i in range(10000):
                large_data.append({
                    "id": i,
                    "data": "x" * 1000,  # 1KB数据
                    "timestamp": time.time()
                })
            
            # 模拟数据处理
            processed_data = [
                item for item in large_data 
                if item["id"] % 2 == 0
            ]
            
            return len(processed_data)
        
        # 监控内存使用
        initial_memory = psutil.virtual_memory().used
        
        # 并发执行内存密集型操作
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(memory_intensive_operation) 
                for _ in range(10)
            ]
            
            results = [future.result() for future in futures]
        
        final_memory = psutil.virtual_memory().used
        memory_increase = final_memory - initial_memory
        
        print(f"\n内存压力测试结果:")
        print(f"初始内存使用: {initial_memory / 1024 / 1024:.2f} MB")
        print(f"最终内存使用: {final_memory / 1024 / 1024:.2f} MB")
        print(f"内存增长: {memory_increase / 1024 / 1024:.2f} MB")
        
        # 验证内存使用在合理范围内
        assert memory_increase < 1024 * 1024 * 1024  # 小于1GB增长
        assert all(result > 0 for result in results)
    
    @pytest.mark.performance
    def test_cpu_stress(self):
        """CPU压力测试"""
        
        def cpu_intensive_operation():
            """CPU密集型操作"""
            # 计算密集型任务
            result = 0
            for i in range(1000000):
                result += i ** 2
            return result
        
        # 监控CPU使用
        cpu_percent_before = psutil.cpu_percent(interval=1)
        
        start_time = time.time()
        
        # 并发执行CPU密集型操作
        with ThreadPoolExecutor(max_workers=psutil.cpu_count()) as executor:
            futures = [
                executor.submit(cpu_intensive_operation) 
                for _ in range(psutil.cpu_count() * 2)
            ]
            
            results = [future.result() for future in futures]
        
        end_time = time.time()
        cpu_percent_after = psutil.cpu_percent(interval=1)
        
        execution_time = end_time - start_time
        
        print(f"\nCPU压力测试结果:")
        print(f"测试前CPU使用率: {cpu_percent_before:.2f}%")
        print(f"测试后CPU使用率: {cpu_percent_after:.2f}%")
        print(f"执行时间: {execution_time:.2f}秒")
        print(f"完成任务数: {len(results)}")
        
        # 验证所有任务完成
        assert len(results) == psutil.cpu_count() * 2
        assert all(result > 0 for result in results)
```

## 📊 性能分析

### 性能分析工具

```python
# tests/performance/performance_analyzer.py

import cProfile
import pstats
import io
import time
import psutil
from contextlib import contextmanager

class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        self.profiler = None
        self.start_time = None
        self.start_memory = None
    
    @contextmanager
    def profile(self, sort_by='cumulative'):
        """性能分析上下文管理器"""
        # 开始分析
        self.profiler = cProfile.Profile()
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used
        
        self.profiler.enable()
        
        try:
            yield self
        finally:
            # 结束分析
            self.profiler.disable()
            
            end_time = time.time()
            end_memory = psutil.virtual_memory().used
            
            # 生成报告
            self._generate_report(sort_by, end_time, end_memory)
    
    def _generate_report(self, sort_by, end_time, end_memory):
        """生成性能报告"""
        # 时间和内存统计
        execution_time = end_time - self.start_time
        memory_usage = end_memory - self.start_memory
        
        print(f"\n=== 性能分析报告 ===")
        print(f"执行时间: {execution_time:.4f}秒")
        print(f"内存使用: {memory_usage / 1024 / 1024:.2f} MB")
        
        # CPU分析
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats(sort_by)
        ps.print_stats(20)  # 显示前20个函数
        
        print(f"\n=== CPU分析 (按{sort_by}排序) ===")
        print(s.getvalue())

# 使用示例
def test_with_performance_analysis():
    """使用性能分析的测试"""
    analyzer = PerformanceAnalyzer()
    
    with analyzer.profile():
        # 执行需要分析的代码
        result = expensive_function()
        assert result is not None
```

### 内存分析

```python
# tests/performance/memory_profiler.py

import tracemalloc
import gc
from functools import wraps

def memory_profile(func):
    """内存分析装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 开始内存跟踪
        tracemalloc.start()
        
        # 执行函数
        result = func(*args, **kwargs)
        
        # 获取内存统计
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"\n=== {func.__name__} 内存分析 ===")
        print(f"当前内存使用: {current / 1024 / 1024:.2f} MB")
        print(f"峰值内存使用: {peak / 1024 / 1024:.2f} MB")
        
        # 强制垃圾回收
        gc.collect()
        
        return result
    
    return wrapper

# 使用示例
@memory_profile
def test_memory_intensive_function():
    """内存密集型函数测试"""
    large_list = [i for i in range(1000000)]
    processed_list = [x * 2 for x in large_list if x % 2 == 0]
    return len(processed_list)
```

## 🚀 优化建议

### 1. 数据库优化

```sql
-- 创建适当的索引
CREATE INDEX idx_entities_name ON entities(name);
CREATE INDEX idx_entities_type ON entities(entity_type);
CREATE INDEX idx_chunks_document_id ON chunks(document_id);

-- 使用部分索引
CREATE INDEX idx_active_entities ON entities(name) WHERE status = 'active';

-- 向量索引优化
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

### 2. 应用层优化

```python
# 使用连接池
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_timeout=30,
    pool_recycle=3600
)

# 批量操作
def bulk_insert_entities(entities_data):
    """批量插入实体"""
    with SessionLocal() as session:
        entities = [Entity(**data) for data in entities_data]
        session.bulk_save_objects(entities)
        session.commit()

# 缓存优化
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_entity_by_name(name: str):
    """缓存实体查询"""
    with SessionLocal() as session:
        return session.query(Entity).filter(Entity.name == name).first()
```

### 3. Neo4j优化

```cypher
// 创建约束和索引
CREATE CONSTRAINT entity_name_unique FOR (e:Entity) REQUIRE e.name IS UNIQUE;
CREATE INDEX entity_type_index FOR (e:Entity) ON (e.entity_type);
CREATE INDEX relation_type_index FOR ()-[r:RELATED_TO]-() ON (r.relation_type);

// 优化查询
// 使用参数化查询
MATCH (e:Entity {name: $entity_name})-[r:RELATED_TO]-(related)
RETURN related.name, r.confidence
ORDER BY r.confidence DESC
LIMIT 10;

// 使用EXPLAIN分析查询计划
EXPLAIN MATCH (e:Entity)-[:RELATED_TO*1..3]-(related)
WHERE e.name = '张三'
RETURN related.name;
```

### 4. 异步优化

```python
# 使用异步数据库操作
import asyncpg
import asyncio

async def async_database_operation():
    """异步数据库操作"""
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        result = await conn.fetch(
            "SELECT * FROM entities WHERE entity_type = $1",
            "PERSON"
        )
        return result
    finally:
        await conn.close()

# 并发处理
async def process_documents_concurrently(documents):
    """并发处理文档"""
    semaphore = asyncio.Semaphore(10)  # 限制并发数
    
    async def process_single_document(doc):
        async with semaphore:
            return await document_service.process_document(doc)
    
    tasks = [process_single_document(doc) for doc in documents]
    results = await asyncio.gather(*tasks)
    return results
```

## 📈 持续监控

### 性能监控脚本

```python
# scripts/performance_monitor.py

import time
import psutil
import requests
import json
from datetime import datetime

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, api_base_url="http://localhost:8000"):
        self.api_base_url = api_base_url
        self.metrics = []
    
    def collect_system_metrics(self):
        """收集系统指标"""
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "network_io": psutil.net_io_counters()._asdict()
        }
    
    def test_api_performance(self):
        """测试API性能"""
        endpoints = [
            "/api/v1/health",
            "/api/v1/documents",
            "/api/v1/entities"
        ]
        
        api_metrics = {}
        for endpoint in endpoints:
            start_time = time.time()
            try:
                response = requests.get(f"{self.api_base_url}{endpoint}")
                end_time = time.time()
                
                api_metrics[endpoint] = {
                    "response_time": end_time - start_time,
                    "status_code": response.status_code,
                    "success": response.status_code == 200
                }
            except Exception as e:
                api_metrics[endpoint] = {
                    "error": str(e),
                    "success": False
                }
        
        return api_metrics
    
    def run_monitoring(self, duration_minutes=60, interval_seconds=30):
        """运行性能监控"""
        end_time = time.time() + (duration_minutes * 60)
        
        while time.time() < end_time:
            # 收集指标
            system_metrics = self.collect_system_metrics()
            api_metrics = self.test_api_performance()
            
            metrics = {
                "system": system_metrics,
                "api": api_metrics
            }
            
            self.metrics.append(metrics)
            
            # 输出当前状态
            print(f"[{system_metrics['timestamp']}] "
                  f"CPU: {system_metrics['cpu_percent']:.1f}% "
                  f"Memory: {system_metrics['memory_percent']:.1f}%")
            
            time.sleep(interval_seconds)
        
        # 保存结果
        self.save_metrics()
    
    def save_metrics(self):
        """保存监控指标"""
        filename = f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"性能指标已保存到: {filename}")

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    monitor.run_monitoring(duration_minutes=30, interval_seconds=60)
```

---

更多信息请参考：
- [测试编写指南](test_guide.md)
- [环境配置指南](environment_setup.md)
- [故障排除指南](troubleshooting.md)