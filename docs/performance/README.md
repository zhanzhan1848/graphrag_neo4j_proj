# 性能优化指南

## 概述

本文档提供了GraphRAG知识库系统的全面性能优化指南，涵盖数据库优化、应用层优化、缓存策略、负载均衡等关键性能领域。

## 性能架构

```mermaid
graph TB
    subgraph "负载均衡层"
        LB[Nginx负载均衡<br/>连接池 + 缓存]
        CDN[CDN<br/>静态资源缓存]
    end
    
    subgraph "应用层"
        API1[FastAPI实例1<br/>异步处理]
        API2[FastAPI实例2<br/>异步处理]
        API3[FastAPI实例3<br/>异步处理]
        
        WORKER1[Celery Worker 1<br/>任务队列]
        WORKER2[Celery Worker 2<br/>任务队列]
        WORKER3[Celery Worker 3<br/>任务队列]
    end
    
    subgraph "缓存层"
        REDIS_CACHE[Redis缓存<br/>查询结果缓存]
        REDIS_SESSION[Redis会话<br/>用户会话存储]
        MEMCACHED[Memcached<br/>对象缓存]
    end
    
    subgraph "数据存储层"
        PG_MASTER[(PostgreSQL主库<br/>读写分离)]
        PG_SLAVE1[(PostgreSQL从库1<br/>只读查询)]
        PG_SLAVE2[(PostgreSQL从库2<br/>只读查询)]
        
        NEO4J_CLUSTER[(Neo4j集群<br/>图查询优化)]
        WEAVIATE_CLUSTER[(Weaviate集群<br/>向量检索)]
        
        MINIO_CLUSTER[(MinIO集群<br/>对象存储)]
    end
    
    subgraph "监控层"
        PROMETHEUS[Prometheus<br/>指标收集]
        GRAFANA[Grafana<br/>性能监控]
        JAEGER[Jaeger<br/>链路追踪]
    end
    
    CDN --> LB
    LB --> API1
    LB --> API2
    LB --> API3
    
    API1 --> REDIS_CACHE
    API2 --> REDIS_CACHE
    API3 --> REDIS_CACHE
    
    API1 --> REDIS_SESSION
    API2 --> REDIS_SESSION
    API3 --> REDIS_SESSION
    
    API1 --> PG_MASTER
    API1 --> PG_SLAVE1
    API1 --> PG_SLAVE2
    
    API2 --> PG_MASTER
    API2 --> PG_SLAVE1
    API2 --> PG_SLAVE2
    
    API3 --> PG_MASTER
    API3 --> PG_SLAVE1
    API3 --> PG_SLAVE2
    
    API1 --> NEO4J_CLUSTER
    API2 --> NEO4J_CLUSTER
    API3 --> NEO4J_CLUSTER
    
    API1 --> WEAVIATE_CLUSTER
    API2 --> WEAVIATE_CLUSTER
    API3 --> WEAVIATE_CLUSTER
    
    WORKER1 --> REDIS_CACHE
    WORKER2 --> REDIS_CACHE
    WORKER3 --> REDIS_CACHE
    
    WORKER1 --> PG_MASTER
    WORKER2 --> PG_MASTER
    WORKER3 --> PG_MASTER
    
    WORKER1 --> NEO4J_CLUSTER
    WORKER2 --> NEO4J_CLUSTER
    WORKER3 --> NEO4J_CLUSTER
    
    PROMETHEUS --> API1
    PROMETHEUS --> API2
    PROMETHEUS --> API3
    PROMETHEUS --> PG_MASTER
    PROMETHEUS --> NEO4J_CLUSTER
    PROMETHEUS --> REDIS_CACHE
    
    GRAFANA --> PROMETHEUS
    JAEGER --> API1
    JAEGER --> API2
    JAEGER --> API3
```

## 1. 数据库性能优化

### 1.1 PostgreSQL优化配置

```sql
-- PostgreSQL性能优化配置
-- scripts/postgres_performance.sql

-- 内存配置
ALTER SYSTEM SET shared_buffers = '4GB';                    -- 共享缓冲区
ALTER SYSTEM SET effective_cache_size = '12GB';             -- 有效缓存大小
ALTER SYSTEM SET work_mem = '256MB';                        -- 工作内存
ALTER SYSTEM SET maintenance_work_mem = '1GB';              -- 维护工作内存
ALTER SYSTEM SET wal_buffers = '64MB';                      -- WAL缓冲区

-- 检查点配置
ALTER SYSTEM SET checkpoint_completion_target = 0.9;        -- 检查点完成目标
ALTER SYSTEM SET checkpoint_timeout = '15min';              -- 检查点超时
ALTER SYSTEM SET max_wal_size = '4GB';                      -- 最大WAL大小
ALTER SYSTEM SET min_wal_size = '1GB';                      -- 最小WAL大小

-- 连接配置
ALTER SYSTEM SET max_connections = 200;                     -- 最大连接数
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements'; -- 预加载库

-- 查询优化
ALTER SYSTEM SET random_page_cost = 1.1;                    -- 随机页面成本
ALTER SYSTEM SET seq_page_cost = 1.0;                       -- 顺序页面成本
ALTER SYSTEM SET cpu_tuple_cost = 0.01;                     -- CPU元组成本
ALTER SYSTEM SET cpu_index_tuple_cost = 0.005;              -- CPU索引元组成本
ALTER SYSTEM SET cpu_operator_cost = 0.0025;                -- CPU操作符成本

-- 并发配置
ALTER SYSTEM SET max_worker_processes = 8;                  -- 最大工作进程
ALTER SYSTEM SET max_parallel_workers = 8;                  -- 最大并行工作者
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;       -- 每个Gather的最大并行工作者
ALTER SYSTEM SET max_parallel_maintenance_workers = 4;      -- 最大并行维护工作者

-- 日志配置
ALTER SYSTEM SET log_min_duration_statement = 1000;         -- 记录慢查询（1秒以上）
ALTER SYSTEM SET log_checkpoints = on;                      -- 记录检查点
ALTER SYSTEM SET log_connections = on;                      -- 记录连接
ALTER SYSTEM SET log_disconnections = on;                   -- 记录断开连接
ALTER SYSTEM SET log_lock_waits = on;                       -- 记录锁等待

-- 统计信息配置
ALTER SYSTEM SET track_activities = on;                     -- 跟踪活动
ALTER SYSTEM SET track_counts = on;                         -- 跟踪计数
ALTER SYSTEM SET track_io_timing = on;                      -- 跟踪IO时间
ALTER SYSTEM SET track_functions = 'all';                   -- 跟踪函数

-- 重新加载配置
SELECT pg_reload_conf();

-- 创建性能监控视图
CREATE OR REPLACE VIEW performance_stats AS
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation,
    most_common_vals,
    most_common_freqs
FROM pg_stats 
WHERE schemaname = 'public'
ORDER BY tablename, attname;

-- 创建慢查询监控视图
CREATE OR REPLACE VIEW slow_queries AS
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements 
ORDER BY total_time DESC
LIMIT 20;
```

### 1.2 数据库索引优化

```sql
-- 数据库索引优化
-- scripts/create_indexes.sql

-- 文档表索引
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_owner_id 
ON documents(owner_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_status 
ON documents(status);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_created_at 
ON documents(created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_file_type 
ON documents(file_type);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_title_gin 
ON documents USING gin(to_tsvector('english', title));

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_content_gin 
ON documents USING gin(to_tsvector('english', content));

-- 复合索引
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_owner_status_created 
ON documents(owner_id, status, created_at DESC);

-- 文档块表索引
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_document_id 
ON document_chunks(document_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_chunk_index 
ON document_chunks(chunk_index);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_content_gin 
ON document_chunks USING gin(to_tsvector('english', content));

-- 向量索引（使用pgvector扩展）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_embedding_cosine 
ON document_chunks USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- 知识实体表索引
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_entities_name 
ON knowledge_entities(name);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_entities_type 
ON knowledge_entities(entity_type);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_entities_document_id 
ON knowledge_entities(document_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_entities_name_gin 
ON knowledge_entities USING gin(to_tsvector('english', name));

-- 知识关系表索引
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_relations_source_id 
ON knowledge_relations(source_entity_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_relations_target_id 
ON knowledge_relations(target_entity_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_relations_type 
ON knowledge_relations(relation_type);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_relations_confidence 
ON knowledge_relations(confidence DESC);

-- 复合索引
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_relations_source_type 
ON knowledge_relations(source_entity_id, relation_type);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_relations_target_type 
ON knowledge_relations(target_entity_id, relation_type);

-- 用户表索引
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email 
ON users(email);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_username 
ON users(username);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_is_active 
ON users(is_active);

-- 查询历史表索引
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_query_history_user_id 
ON query_history(user_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_query_history_created_at 
ON query_history(created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_query_history_query_gin 
ON query_history USING gin(to_tsvector('english', query_text));

-- 分区表索引（按时间分区）
-- 为大表创建分区以提高查询性能
CREATE TABLE query_history_y2024m01 PARTITION OF query_history 
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE query_history_y2024m02 PARTITION OF query_history 
FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- 为每个分区创建索引
CREATE INDEX idx_query_history_y2024m01_user_id 
ON query_history_y2024m01(user_id);

CREATE INDEX idx_query_history_y2024m02_user_id 
ON query_history_y2024m02(user_id);

-- 索引使用情况监控
CREATE OR REPLACE VIEW index_usage_stats AS
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_tup_read,
    idx_tup_fetch,
    idx_scan,
    CASE 
        WHEN idx_scan = 0 THEN 'Unused'
        WHEN idx_scan < 10 THEN 'Low Usage'
        WHEN idx_scan < 100 THEN 'Medium Usage'
        ELSE 'High Usage'
    END as usage_level
FROM pg_stat_user_indexes 
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;

-- 表大小监控
CREATE OR REPLACE VIEW table_sizes AS
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
    pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) as indexes_size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### 1.3 Neo4j性能优化

```conf
# neo4j/neo4j.conf - Neo4j性能优化配置

# 内存配置
dbms.memory.heap.initial_size=4G
dbms.memory.heap.max_size=4G
dbms.memory.pagecache.size=2G

# 事务配置
dbms.transaction.timeout=60s
dbms.transaction.concurrent.maximum=1000
dbms.transaction.bookmark_ready_timeout=30s

# 查询配置
cypher.default_language_version=5
cypher.forbid_exhaustive_shortestpath=true
cypher.forbid_shortestpath_common_nodes=true

# 连接池配置
dbms.connector.bolt.thread_pool_min_size=5
dbms.connector.bolt.thread_pool_max_size=400
dbms.connector.bolt.thread_pool_keep_alive=5m

# 索引配置
dbms.index_sampling.background_enabled=true
dbms.index_sampling.sample_size_limit=1000000

# 日志配置
dbms.logs.query.enabled=true
dbms.logs.query.threshold=1s
dbms.logs.query.parameter_logging_enabled=true

# 性能监控
metrics.enabled=true
metrics.csv.enabled=true
metrics.csv.interval=30s
metrics.jmx.enabled=true

# 并发配置
dbms.cypher.parallel_runtime_support=all
dbms.cypher.runtime=parallel

# 缓存配置
dbms.query_cache_size=1000
dbms.query_plan_cache_size=1000
```

```cypher
// Neo4j索引和约束创建
// scripts/neo4j_indexes.cypher

// 创建节点索引
CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.entity_id);

CREATE INDEX document_id_index IF NOT EXISTS FOR (d:Document) ON (d.document_id);
CREATE INDEX document_title_index IF NOT EXISTS FOR (d:Document) ON (d.title);

CREATE INDEX chunk_id_index IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_id);
CREATE INDEX chunk_index_index IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_index);

// 创建全文索引
CREATE FULLTEXT INDEX entity_fulltext_index IF NOT EXISTS 
FOR (e:Entity) ON EACH [e.name, e.description];

CREATE FULLTEXT INDEX document_fulltext_index IF NOT EXISTS 
FOR (d:Document) ON EACH [d.title, d.content];

// 创建约束
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE;
CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE;
CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE;

// 创建复合索引
CREATE INDEX entity_type_confidence_index IF NOT EXISTS FOR (e:Entity) ON (e.type, e.confidence);
CREATE INDEX relation_type_confidence_index IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.type, r.confidence);

// 性能监控查询
// 查看索引使用情况
CALL db.indexes() YIELD name, state, populationPercent, type
RETURN name, state, populationPercent, type
ORDER BY name;

// 查看约束
CALL db.constraints() YIELD name, description
RETURN name, description;

// 查看查询性能
CALL dbms.queryJmx("org.neo4j:instance=kernel#0,name=Queries") 
YIELD attributes
RETURN attributes;
```

## 2. 应用层性能优化

### 2.1 FastAPI性能优化

```python
# src/core/performance/optimization.py
"""
应用层性能优化工具
提供缓存、连接池、异步处理等优化功能
"""

import asyncio
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union
import redis
import json
import hashlib
from contextlib import asynccontextmanager
import aioredis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self):
        """初始化性能优化器"""
        self.redis_client = None
        self.db_engine = None
        self.connection_pools = {}
        
    async def initialize(self):
        """初始化连接池和缓存"""
        # 初始化Redis连接池
        self.redis_client = aioredis.from_url(
            "redis://localhost:6379",
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
            retry_on_timeout=True
        )
        
        # 初始化数据库连接池
        self.db_engine = create_async_engine(
            "postgresql+asyncpg://user:password@localhost/graphrag",
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )
        
        self.async_session = sessionmaker(
            self.db_engine, 
            class_=AsyncSession, 
            expire_on_commit=False
        )
    
    def cache_result(self, ttl: int = 300, key_prefix: str = "cache"):
        """
        结果缓存装饰器
        
        Args:
            ttl: 缓存过期时间（秒）
            key_prefix: 缓存键前缀
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = self._generate_cache_key(
                    key_prefix, func.__name__, args, kwargs
                )
                
                # 尝试从缓存获取结果
                cached_result = await self.redis_client.get(cache_key)
                if cached_result:
                    return json.loads(cached_result)
                
                # 执行函数并缓存结果
                result = await func(*args, **kwargs)
                await self.redis_client.setex(
                    cache_key, ttl, json.dumps(result, default=str)
                )
                
                return result
            return wrapper
        return decorator
    
    def rate_limit(self, max_calls: int, window: int):
        """
        速率限制装饰器
        
        Args:
            max_calls: 最大调用次数
            window: 时间窗口（秒）
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # 获取客户端标识（这里简化处理）
                client_id = kwargs.get('client_id', 'default')
                rate_key = f"rate_limit:{func.__name__}:{client_id}"
                
                # 检查速率限制
                current_calls = await self.redis_client.get(rate_key)
                if current_calls and int(current_calls) >= max_calls:
                    raise Exception("Rate limit exceeded")
                
                # 增加调用计数
                pipe = self.redis_client.pipeline()
                pipe.incr(rate_key)
                pipe.expire(rate_key, window)
                await pipe.execute()
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def async_batch_process(self, batch_size: int = 100):
        """
        异步批处理装饰器
        
        Args:
            batch_size: 批处理大小
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(items: list, *args, **kwargs):
                results = []
                
                # 分批处理
                for i in range(0, len(items), batch_size):
                    batch = items[i:i + batch_size]
                    
                    # 并发处理批次
                    tasks = [
                        func(item, *args, **kwargs) 
                        for item in batch
                    ]
                    batch_results = await asyncio.gather(*tasks)
                    results.extend(batch_results)
                
                return results
            return wrapper
        return decorator
    
    def performance_monitor(self):
        """性能监控装饰器"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    # 记录性能指标
                    await self._record_performance_metric(
                        func.__name__, execution_time, "success"
                    )
                    
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    
                    # 记录错误指标
                    await self._record_performance_metric(
                        func.__name__, execution_time, "error"
                    )
                    
                    raise e
            return wrapper
        return decorator
    
    async def _record_performance_metric(self, function_name: str, 
                                       execution_time: float, status: str):
        """记录性能指标"""
        metric_key = f"performance:{function_name}"
        
        # 记录执行时间
        await self.redis_client.lpush(
            f"{metric_key}:times", 
            json.dumps({
                "timestamp": time.time(),
                "execution_time": execution_time,
                "status": status
            })
        )
        
        # 保留最近1000条记录
        await self.redis_client.ltrim(f"{metric_key}:times", 0, 999)
        
        # 更新统计信息
        await self.redis_client.hincrby(f"{metric_key}:stats", "total_calls", 1)
        await self.redis_client.hincrby(f"{metric_key}:stats", f"{status}_calls", 1)
        
        # 更新平均执行时间
        current_avg = await self.redis_client.hget(f"{metric_key}:stats", "avg_time")
        if current_avg:
            current_avg = float(current_avg)
            total_calls = await self.redis_client.hget(f"{metric_key}:stats", "total_calls")
            new_avg = (current_avg * (int(total_calls) - 1) + execution_time) / int(total_calls)
        else:
            new_avg = execution_time
        
        await self.redis_client.hset(f"{metric_key}:stats", "avg_time", new_avg)
    
    def _generate_cache_key(self, prefix: str, func_name: str, 
                          args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        # 创建参数的哈希值
        params_str = f"{args}_{sorted(kwargs.items())}"
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        
        return f"{prefix}:{func_name}:{params_hash}"
    
    @asynccontextmanager
    async def get_db_session(self):
        """获取数据库会话的上下文管理器"""
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

# 创建全局性能优化器实例
performance_optimizer = PerformanceOptimizer()

# 便捷装饰器
cache_result = performance_optimizer.cache_result
rate_limit = performance_optimizer.rate_limit
async_batch_process = performance_optimizer.async_batch_process
performance_monitor = performance_optimizer.performance_monitor
```

### 2.2 异步处理优化

```python
# src/core/performance/async_processing.py
"""
异步处理优化
提供高效的异步任务处理和并发控制
"""

import asyncio
import aiofiles
from typing import List, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from dataclasses import dataclass
import time

@dataclass
class TaskResult:
    """任务结果数据类"""
    task_id: str
    result: Any
    execution_time: float
    status: str
    error: Optional[str] = None

class AsyncTaskProcessor:
    """异步任务处理器"""
    
    def __init__(self, max_workers: int = None):
        """
        初始化异步任务处理器
        
        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.semaphore = asyncio.Semaphore(self.max_workers * 2)
    
    async def process_tasks_concurrently(
        self, 
        tasks: List[Callable], 
        max_concurrent: int = 10
    ) -> List[TaskResult]:
        """
        并发处理任务列表
        
        Args:
            tasks: 任务列表
            max_concurrent: 最大并发数
            
        Returns:
            任务结果列表
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_task(task_func: Callable, task_id: str) -> TaskResult:
            async with semaphore:
                start_time = time.time()
                
                try:
                    if asyncio.iscoroutinefunction(task_func):
                        result = await task_func()
                    else:
                        # 在线程池中执行同步函数
                        result = await asyncio.get_event_loop().run_in_executor(
                            self.thread_executor, task_func
                        )
                    
                    execution_time = time.time() - start_time
                    
                    return TaskResult(
                        task_id=task_id,
                        result=result,
                        execution_time=execution_time,
                        status="success"
                    )
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    
                    return TaskResult(
                        task_id=task_id,
                        result=None,
                        execution_time=execution_time,
                        status="error",
                        error=str(e)
                    )
        
        # 创建任务
        task_coroutines = [
            process_single_task(task, f"task_{i}")
            for i, task in enumerate(tasks)
        ]
        
        # 并发执行所有任务
        results = await asyncio.gather(*task_coroutines, return_exceptions=True)
        
        return results
    
    async def process_cpu_intensive_tasks(
        self, 
        tasks: List[Callable], 
        use_processes: bool = True
    ) -> List[TaskResult]:
        """
        处理CPU密集型任务
        
        Args:
            tasks: 任务列表
            use_processes: 是否使用进程池
            
        Returns:
            任务结果列表
        """
        executor = self.process_executor if use_processes else self.thread_executor
        
        async def process_task(task_func: Callable, task_id: str) -> TaskResult:
            start_time = time.time()
            
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    executor, task_func
                )
                
                execution_time = time.time() - start_time
                
                return TaskResult(
                    task_id=task_id,
                    result=result,
                    execution_time=execution_time,
                    status="success"
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                return TaskResult(
                    task_id=task_id,
                    result=None,
                    execution_time=execution_time,
                    status="error",
                    error=str(e)
                )
        
        # 创建任务
        task_coroutines = [
            process_task(task, f"cpu_task_{i}")
            for i, task in enumerate(tasks)
        ]
        
        # 执行任务
        results = await asyncio.gather(*task_coroutines)
        
        return results
    
    async def process_io_intensive_tasks(
        self, 
        tasks: List[Callable]
    ) -> List[TaskResult]:
        """
        处理IO密集型任务
        
        Args:
            tasks: 任务列表
            
        Returns:
            任务结果列表
        """
        async def process_task(task_func: Callable, task_id: str) -> TaskResult:
            async with self.semaphore:
                start_time = time.time()
                
                try:
                    if asyncio.iscoroutinefunction(task_func):
                        result = await task_func()
                    else:
                        result = await asyncio.get_event_loop().run_in_executor(
                            None, task_func
                        )
                    
                    execution_time = time.time() - start_time
                    
                    return TaskResult(
                        task_id=task_id,
                        result=result,
                        execution_time=execution_time,
                        status="success"
                    )
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    
                    return TaskResult(
                        task_id=task_id,
                        result=None,
                        execution_time=execution_time,
                        status="error",
                        error=str(e)
                    )
        
        # 创建任务
        task_coroutines = [
            process_task(task, f"io_task_{i}")
            for i, task in enumerate(tasks)
        ]
        
        # 并发执行所有任务
        results = await asyncio.gather(*task_coroutines)
        
        return results
    
    async def batch_file_processing(
        self, 
        file_paths: List[str], 
        process_func: Callable,
        batch_size: int = 10
    ) -> List[TaskResult]:
        """
        批量文件处理
        
        Args:
            file_paths: 文件路径列表
            process_func: 处理函数
            batch_size: 批处理大小
            
        Returns:
            处理结果列表
        """
        results = []
        
        for i in range(0, len(file_paths), batch_size):
            batch_files = file_paths[i:i + batch_size]
            
            async def process_file_batch():
                batch_results = []
                
                for file_path in batch_files:
                    start_time = time.time()
                    
                    try:
                        async with aiofiles.open(file_path, 'rb') as file:
                            content = await file.read()
                        
                        # 处理文件内容
                        if asyncio.iscoroutinefunction(process_func):
                            result = await process_func(content, file_path)
                        else:
                            result = await asyncio.get_event_loop().run_in_executor(
                                self.thread_executor, process_func, content, file_path
                            )
                        
                        execution_time = time.time() - start_time
                        
                        batch_results.append(TaskResult(
                            task_id=file_path,
                            result=result,
                            execution_time=execution_time,
                            status="success"
                        ))
                        
                    except Exception as e:
                        execution_time = time.time() - start_time
                        
                        batch_results.append(TaskResult(
                            task_id=file_path,
                            result=None,
                            execution_time=execution_time,
                            status="error",
                            error=str(e)
                        ))
                
                return batch_results
            
            batch_results = await process_file_batch()
            results.extend(batch_results)
        
        return results
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'thread_executor'):
            self.thread_executor.shutdown(wait=False)
        if hasattr(self, 'process_executor'):
            self.process_executor.shutdown(wait=False)

# 创建全局异步任务处理器
async_processor = AsyncTaskProcessor()
```

## 3. 缓存策略优化

### 3.1 多层缓存架构

```python
# src/core/performance/caching.py
"""
多层缓存系统
实现内存缓存、Redis缓存、数据库缓存的多层架构
"""

import asyncio
import json
import time
from typing import Any, Dict, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import hashlib
import pickle
from functools import wraps
import aioredis
from cachetools import TTLCache, LRUCache

class CacheLevel(Enum):
    """缓存级别"""
    MEMORY = "memory"      # 内存缓存
    REDIS = "redis"        # Redis缓存
    DATABASE = "database"  # 数据库缓存

@dataclass
class CacheConfig:
    """缓存配置"""
    ttl: int = 300                    # 过期时间（秒）
    max_size: int = 1000             # 最大缓存条目数
    levels: list = None              # 缓存级别
    key_prefix: str = "cache"        # 键前缀
    serialize_method: str = "json"   # 序列化方法

class MultiLevelCache:
    """多层缓存管理器"""
    
    def __init__(self):
        """初始化多层缓存"""
        # 内存缓存（L1）
        self.memory_cache = TTLCache(maxsize=1000, ttl=300)
        self.lru_cache = LRUCache(maxsize=5000)
        
        # Redis缓存（L2）
        self.redis_client = None
        
        # 缓存统计
        self.stats = {
            "memory_hits": 0,
            "memory_misses": 0,
            "redis_hits": 0,
            "redis_misses": 0,
            "database_hits": 0,
            "database_misses": 0
        }
    
    async def initialize_redis(self, redis_url: str = "redis://localhost:6379"):
        """初始化Redis连接"""
        self.redis_client = aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20
        )
    
    def cache_multi_level(self, config: CacheConfig):
        """
        多层缓存装饰器
        
        Args:
            config: 缓存配置
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = self._generate_cache_key(
                    config.key_prefix, func.__name__, args, kwargs
                )
                
                # 按层级查找缓存
                levels = config.levels or [CacheLevel.MEMORY, CacheLevel.REDIS]
                
                for level in levels:
                    cached_value = await self._get_from_cache(cache_key, level)
                    if cached_value is not None:
                        # 回填到更高层级的缓存
                        await self._backfill_cache(cache_key, cached_value, level, levels)
                        return cached_value
                
                # 缓存未命中，执行函数
                result = await func(*args, **kwargs)
                
                # 存储到所有层级的缓存
                await self._store_to_all_levels(cache_key, result, config, levels)
                
                return result
            
            return wrapper
        return decorator
    
    async def _get_from_cache(self, key: str, level: CacheLevel) -> Optional[Any]:
        """从指定层级获取缓存"""
        try:
            if level == CacheLevel.MEMORY:
                # 先尝试TTL缓存
                if key in self.memory_cache:
                    self.stats["memory_hits"] += 1
                    return self.memory_cache[key]
                
                # 再尝试LRU缓存
                if key in self.lru_cache:
                    self.stats["memory_hits"] += 1
                    return self.lru_cache[key]
                
                self.stats["memory_misses"] += 1
                return None
            
            elif level == CacheLevel.REDIS:
                if self.redis_client:
                    cached_data = await self.redis_client.get(key)
                    if cached_data:
                        self.stats["redis_hits"] += 1
                        return json.loads(cached_data)
                
                self.stats["redis_misses"] += 1
                return None
            
            elif level == CacheLevel.DATABASE:
                # 这里可以实现数据库级别的缓存查询
                # 例如查询缓存表或使用数据库的查询缓存
                self.stats["database_misses"] += 1
                return None
        
        except Exception as e:
            print(f"缓存获取错误 ({level.value}): {e}")
            return None
    
    async def _store_to_all_levels(self, key: str, value: Any, 
                                 config: CacheConfig, levels: list):
        """存储到所有层级的缓存"""
        for level in levels:
            await self._store_to_cache(key, value, level, config)
    
    async def _store_to_cache(self, key: str, value: Any, 
                            level: CacheLevel, config: CacheConfig):
        """存储到指定层级的缓存"""
        try:
            if level == CacheLevel.MEMORY:
                # 存储到TTL缓存
                self.memory_cache[key] = value
                
                # 同时存储到LRU缓存
                self.lru_cache[key] = value
            
            elif level == CacheLevel.REDIS:
                if self.redis_client:
                    serialized_value = json.dumps(value, default=str)
                    await self.redis_client.setex(key, config.ttl, serialized_value)
            
            elif level == CacheLevel.DATABASE:
                # 这里可以实现数据库级别的缓存存储
                pass
        
        except Exception as e:
            print(f"缓存存储错误 ({level.value}): {e}")
    
    async def _backfill_cache(self, key: str, value: Any, 
                            found_level: CacheLevel, all_levels: list):
        """回填缓存到更高层级"""
        level_priority = {
            CacheLevel.MEMORY: 0,
            CacheLevel.REDIS: 1,
            CacheLevel.DATABASE: 2
        }
        
        found_priority = level_priority[found_level]
        
        # 回填到优先级更高的层级
        for level in all_levels:
            if level_priority[level] < found_priority:
                config = CacheConfig()  # 使用默认配置
                await self._store_to_cache(key, value, level, config)
    
    def _generate_cache_key(self, prefix: str, func_name: str, 
                          args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        # 创建参数的哈希值
        params_str = f"{args}_{sorted(kwargs.items())}"
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        
        return f"{prefix}:{func_name}:{params_hash}"
    
    async def invalidate_cache(self, pattern: str):
        """
        失效缓存
        
        Args:
            pattern: 缓存键模式
        """
        # 清理内存缓存
        keys_to_remove = [key for key in self.memory_cache.keys() if pattern in key]
        for key in keys_to_remove:
            del self.memory_cache[key]
        
        keys_to_remove = [key for key in self.lru_cache.keys() if pattern in key]
        for key in keys_to_remove:
            del self.lru_cache[key]
        
        # 清理Redis缓存
        if self.redis_client:
            keys = await self.redis_client.keys(f"*{pattern}*")
            if keys:
                await self.redis_client.delete(*keys)
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_requests = sum(self.stats.values())
        
        if total_requests == 0:
            return self.stats
        
        # 计算命中率
        memory_hit_rate = (self.stats["memory_hits"] / 
                          (self.stats["memory_hits"] + self.stats["memory_misses"])) * 100
        
        redis_hit_rate = (self.stats["redis_hits"] / 
                         (self.stats["redis_hits"] + self.stats["redis_misses"])) * 100
        
        return {
            **self.stats,
            "memory_hit_rate": f"{memory_hit_rate:.2f}%",
            "redis_hit_rate": f"{redis_hit_rate:.2f}%",
            "memory_cache_size": len(self.memory_cache),
            "lru_cache_size": len(self.lru_cache)
        }
    
    async def warm_up_cache(self, warm_up_functions: list):
        """
        缓存预热
        
        Args:
            warm_up_functions: 预热函数列表
        """
        print("开始缓存预热...")
        
        for func in warm_up_functions:
            try:
                await func()
                print(f"预热完成: {func.__name__}")
            except Exception as e:
                print(f"预热失败 {func.__name__}: {e}")
        
        print("缓存预热完成")

# 创建全局多层缓存实例
multi_cache = MultiLevelCache()

# 便捷的缓存装饰器
def cache_result(ttl: int = 300, levels: list = None, key_prefix: str = "cache"):
    """便捷的缓存装饰器"""
    config = CacheConfig(
        ttl=ttl,
        levels=levels or [CacheLevel.MEMORY, CacheLevel.REDIS],
        key_prefix=key_prefix
    )
    return multi_cache.cache_multi_level(config)

# 特定用途的缓存装饰器
def cache_query_result(ttl: int = 600):
    """查询结果缓存"""
    return cache_result(ttl=ttl, key_prefix="query")

def cache_document_processing(ttl: int = 3600):
    """文档处理结果缓存"""
    return cache_result(ttl=ttl, key_prefix="document")

def cache_embedding_result(ttl: int = 7200):
    """向量嵌入结果缓存"""
    return cache_result(ttl=ttl, key_prefix="embedding")
```

## 4. 负载均衡和扩展

### 4.1 Nginx负载均衡配置

```nginx
# nginx/nginx.conf
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # 日志格式
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for" '
                    'rt=$request_time uct="$upstream_connect_time" '
                    'uht="$upstream_header_time" urt="$upstream_response_time"';
    
    access_log /var/log/nginx/access.log main;
    
    # 基本配置
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 100M;
    
    # Gzip压缩
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;
    
    # 上游服务器配置
    upstream graphrag_api {
        least_conn;
        server api1:8000 max_fails=3 fail_timeout=30s;
        server api2:8000 max_fails=3 fail_timeout=30s;
        server api3:8000 max_fails=3 fail_timeout=30s;
        
        # 健康检查
        keepalive 32;
        keepalive_requests 100;
        keepalive_timeout 60s;
    }
    
    # 限流配置
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=upload_limit:10m rate=2r/s;
    
    # 连接限制
    limit_conn_zone $binary_remote_addr zone=conn_limit:10m;
    
    server {
        listen 80;
        server_name graphrag.example.com;
        
        # 重定向到HTTPS
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name graphrag.example.com;
        
        # SSL配置
        ssl_certificate /etc/ssl/certs/graphrag.crt;
        ssl_certificate_key /etc/ssl/private/graphrag.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;
        
        # 安全头
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        
        # 连接限制
        limit_conn conn_limit 20;
        
        # API路由
        location /api/ {
            limit_req zone=api_limit burst=20 nodelay;
            
            proxy_pass http://graphrag_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # 超时配置
            proxy_connect_timeout 30s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
            
            # 缓冲配置
            proxy_buffering on;
            proxy_buffer_size 4k;
            proxy_buffers 8 4k;
            proxy_busy_buffers_size 8k;
        }
        
        # 文件上传路由
        location /api/documents/upload {
            limit_req zone=upload_limit burst=5 nodelay;
            
            proxy_pass http://graphrag_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # 上传超时配置
            proxy_connect_timeout 30s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
            
            # 禁用缓冲以支持大文件上传
            proxy_request_buffering off;
        }
        
        # 静态文件缓存
        location /static/ {
            alias /var/www/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
            
            # 启用gzip
            gzip_static on;
        }
        
        # 健康检查
        location /health {
            proxy_pass http://graphrag_api/health;
            access_log off;
        }
        
        # 监控端点
        location /metrics {
            proxy_pass http://graphrag_api/metrics;
            
            # 限制访问
            allow 10.0.0.0/8;
            allow 172.16.0.0/12;
            allow 192.168.0.0/16;
            deny all;
        }
    }
}
```

### 4.2 Kubernetes水平扩展配置

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: graphrag-api-hpa
  namespace: graphrag-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: graphrag-api
  
  minReplicas: 3
  maxReplicas: 20
  
  metrics:
  # CPU使用率
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  
  # 内存使用率
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  
  # 自定义指标：请求率
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  
  # 自定义指标：响应时间
  - type: Pods
    pods:
      metric:
        name: http_request_duration_seconds
      target:
        type: AverageValue
        averageValue: "0.5"
  
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Min
    
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 4
        periodSeconds: 60
      selectPolicy: Max

---
# Worker HPA配置
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: graphrag-worker-hpa
  namespace: graphrag-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: graphrag-worker
  
  minReplicas: 2
  maxReplicas: 10
  
  metrics:
  # CPU使用率
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 80
  
  # 内存使用率
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 85
  
  # 队列长度
  - type: External
    external:
      metric:
        name: celery_queue_length
        selector:
          matchLabels:
            queue: "default"
      target:
        type: AverageValue
        averageValue: "10"
  
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 600  # 10分钟稳定期
      policies:
      - type: Pods
        value: 1
        periodSeconds: 120
    
    scaleUp:
      stabilizationWindowSeconds: 120  # 2分钟稳定期
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60
```

## 5. 性能监控和分析

### 5.1 性能指标收集

```python
# src/core/performance/metrics.py
"""
性能指标收集和分析
提供详细的性能监控和分析功能
"""

import time
import psutil
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import aioredis
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import logging

@dataclass
class PerformanceMetric:
    """性能指标数据类"""
    timestamp: datetime
    metric_name: str
    metric_type: str  # counter, gauge, histogram
    value: float
    labels: Dict[str, str]
    unit: str = ""

class MetricsCollector:
    """性能指标收集器"""
    
    def __init__(self):
        """初始化指标收集器"""
        self.registry = CollectorRegistry()
        self.redis_client = None
        
        # 定义指标
        self.request_counter = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'active_connections',
            'Active database connections',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            ['type'],
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.cache_operations = Counter(
            'cache_operations_total',
            'Cache operations',
            ['operation', 'result'],
            registry=self.registry
        )
        
        self.task_queue_size = Gauge(
            'task_queue_size',
            'Task queue size',
            ['queue'],
            registry=self.registry
        )
        
        # 性能数据存储
        self.metrics_history = []
        self.max_history_size = 10000
    
    async def initialize_redis(self, redis_url: str = "redis://localhost:6379"):
        """初始化Redis连接"""
        self.redis_client = aioredis.from_url(redis_url, db=2)
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """
        记录HTTP请求指标
        
        Args:
            method: HTTP方法
            endpoint: 端点
            status: 状态码
            duration: 请求时长
        """
        self.request_counter.labels(
            method=method, 
            endpoint=endpoint, 
            status=str(status)
        ).inc()
        
        self.request_duration.labels(
            method=method, 
            endpoint=endpoint
        ).observe(duration)
    
    def record_cache_operation(self, operation: str, result: str):
        """
        记录缓存操作指标
        
        Args:
            operation: 操作类型（get, set, delete）
            result: 操作结果（hit, miss, success, error）
        """
        self.cache_operations.labels(
            operation=operation, 
            result=result
        ).inc()
    
    async def collect_system_metrics(self):
        """收集系统指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        self.cpu_usage.set(cpu_percent)
        
        # 内存使用情况
        memory = psutil.virtual_memory()
        self.memory_usage.labels(type='used').set(memory.used)
        self.memory_usage.labels(type='available').set(memory.available)
        self.memory_usage.labels(type='total').set(memory.total)
        
        # 磁盘使用情况
        disk = psutil.disk_usage('/')
        
        # 网络IO
        network = psutil.net_io_counters()
        
        # 进程信息
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # 创建性能指标记录
        metrics = [
            PerformanceMetric(
                timestamp=datetime.now(),
                metric_name="cpu_usage",
                metric_type="gauge",
                value=cpu_percent,
                labels={"host": "localhost"},
                unit="percent"
            ),
            PerformanceMetric(
                timestamp=datetime.now(),
                metric_name="memory_usage",
                metric_type="gauge",
                value=memory.used,
                labels={"type": "used", "host": "localhost"},
                unit="bytes"
            ),
            PerformanceMetric(
                timestamp=datetime.now(),
                metric_name="disk_usage",
                metric_type="gauge",
                value=disk.used,
                labels={"mount": "/", "host": "localhost"},
                unit="bytes"
            ),
            PerformanceMetric(
                timestamp=datetime.now(),
                metric_name="network_bytes_sent",
                metric_type="counter",
                value=network.bytes_sent,
                labels={"direction": "sent", "host": "localhost"},
                unit="bytes"
            ),
            PerformanceMetric(
                timestamp=datetime.now(),
                metric_name="network_bytes_recv",
                metric_type="counter",
                value=network.bytes_recv,
                labels={"direction": "received", "host": "localhost"},
                unit="bytes"
            )
        ]
        
        # 存储指标到历史记录
        self.metrics_history.extend(metrics)
        
        # 限制历史记录大小
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
        
        # 存储到Redis
        if self.redis_client:
            for metric in metrics:
                await self._store_metric_to_redis(metric)
    
    async def _store_metric_to_redis(self, metric: PerformanceMetric):
        """存储指标到Redis"""
        try:
            key = f"metrics:{metric.metric_name}:{int(metric.timestamp.timestamp())}"
            value = json.dumps(asdict(metric), default=str)
            
            # 存储指标数据，设置1小时过期时间
            await self.redis_client.setex(key, 3600, value)
            
            # 添加到时间序列
            series_key = f"metrics_series:{metric.metric_name}"
            await self.redis_client.zadd(
                series_key, 
                {key: metric.timestamp.timestamp()}
            )
            
            # 清理过期的时间序列数据（保留24小时）
            cutoff_time = (datetime.now() - timedelta(hours=24)).timestamp()
            await self.redis_client.zremrangebyscore(series_key, 0, cutoff_time)
            
        except Exception as e:
            logging.error(f"存储指标到Redis失败: {e}")
    
    async def get_metrics_summary(self, 
                                time_range: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """
        获取指标摘要
        
        Args:
            time_range: 时间范围
            
        Returns:
            指标摘要
        """
        cutoff_time = datetime.now() - time_range
        
        # 过滤时间范围内的指标
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        # 按指标名称分组
        metrics_by_name = {}
        for metric in recent_metrics:
            if metric.metric_name not in metrics_by_name:
                metrics_by_name[metric.metric_name] = []
            metrics_by_name[metric.metric_name].append(metric.value)
        
        # 计算统计信息
        summary = {}
        for name, values in metrics_by_name.items():
            if values:
                summary[name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "latest": values[-1] if values else 0
                }
        
        return summary
    
    async def analyze_performance_trends(self) -> Dict[str, Any]:
        """分析性能趋势"""
        if len(self.metrics_history) < 10:
            return {"error": "数据不足，无法分析趋势"}
        
        # 分析CPU使用率趋势
        cpu_metrics = [
            m for m in self.metrics_history 
            if m.metric_name == "cpu_usage"
        ]
        
        # 分析内存使用率趋势
        memory_metrics = [
            m for m in self.metrics_history 
            if m.metric_name == "memory_usage" and m.labels.get("type") == "used"
        ]
        
        trends = {}
        
        if len(cpu_metrics) >= 5:
            cpu_values = [m.value for m in cpu_metrics[-10:]]
            trends["cpu_trend"] = self._calculate_trend(cpu_values)
        
        if len(memory_metrics) >= 5:
            memory_values = [m.value for m in memory_metrics[-10:]]
            trends["memory_trend"] = self._calculate_trend(memory_values)
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """计算趋势"""
        if len(values) < 2:
            return {"trend": "insufficient_data"}
        
        # 简单的线性趋势计算
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        
        if slope > 0.1:
            trend = "increasing"
        elif slope < -0.1:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "slope": slope,
            "current": values[-1],
            "average": sum(values) / len(values),
            "change_rate": ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0
        }

# 创建全局指标收集器
metrics_collector = MetricsCollector()
```

### 5.2 性能监控仪表板配置

```yaml
# monitoring/grafana-dashboard.json
{
  "dashboard": {
    "id": null,
    "title": "GraphRAG Performance Dashboard",
    "tags": ["graphrag", "performance"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "API请求率",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ],
        "yAxes": [
          {
            "label": "请求/秒",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 0
        }
      },
      {
        "id": 2,
        "title": "响应时间",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ],
        "yAxes": [
          {
            "label": "秒",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 0
        }
      },
      {
        "id": 3,
        "title": "系统资源使用率",
        "type": "graph",
        "targets": [
          {
            "expr": "cpu_usage_percent",
            "legendFormat": "CPU使用率"
          },
          {
            "expr": "(memory_usage_bytes{type=\"used\"} / memory_usage_bytes{type=\"total\"}) * 100",
            "legendFormat": "内存使用率"
          }
        ],
        "yAxes": [
          {
            "label": "百分比",
            "min": 0,
            "max": 100
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 24,
          "x": 0,
          "y": 8
        }
      },
      {
        "id": 4,
        "title": "数据库连接池",
        "type": "singlestat",
        "targets": [
          {
            "expr": "active_connections",
            "legendFormat": "活跃连接"
          }
        ],
        "gridPos": {
          "h": 4,
          "w": 6,
          "x": 0,
          "y": 16
        }
      },
      {
        "id": 5,
        "title": "缓存命中率",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(cache_operations_total{result=\"hit\"}[5m]) / rate(cache_operations_total[5m]) * 100",
            "legendFormat": "命中率"
          }
        ],
        "gridPos": {
          "h": 4,
          "w": 6,
          "x": 6,
          "y": 16
        }
      },
      {
        "id": 6,
        "title": "任务队列长度",
        "type": "graph",
        "targets": [
          {
            "expr": "task_queue_size",
            "legendFormat": "{{queue}}"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 16
        }
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
```

## 6. 性能测试和基准测试

### 6.1 负载测试脚本

```javascript
// tests/performance/load_test.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// 自定义指标
export let errorRate = new Rate('errors');
export let responseTime = new Trend('response_time');

// 测试配置
export let options = {
  stages: [
    { duration: '2m', target: 10 },   // 预热阶段
    { duration: '5m', target: 50 },   // 负载增加
    { duration: '10m', target: 100 }, // 稳定负载
    { duration: '5m', target: 200 },  // 峰值负载
    { duration: '5m', target: 0 },    // 负载下降
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95%的请求在2秒内完成
    http_req_failed: ['rate<0.1'],     // 错误率低于10%
    errors: ['rate<0.1'],              // 自定义错误率低于10%
  },
};

const BASE_URL = 'http://localhost:8000';

// 测试数据
const testQueries = [
  '什么是人工智能？',
  '机器学习的基本原理',
  '深度学习和神经网络',
  '自然语言处理技术',
  '计算机视觉应用'
];

const testDocuments = [
  {
    title: '测试文档1',
    content: '这是一个用于性能测试的文档内容...',
    file_type: 'text'
  },
  {
    title: '测试文档2', 
    content: '另一个测试文档的内容，包含更多信息...',
    file_type: 'text'
  }
];

export default function() {
  // 随机选择测试场景
  const scenario = Math.random();
  
  if (scenario < 0.4) {
    // 40% - 查询测试
    testQuery();
  } else if (scenario < 0.7) {
    // 30% - 文档上传测试
    testDocumentUpload();
  } else if (scenario < 0.9) {
    // 20% - 实体查询测试
    testEntityQuery();
  } else {
    // 10% - 关系查询测试
    testRelationQuery();
  }
  
  sleep(1);
}

function testQuery() {
  const query = testQueries[Math.floor(Math.random() * testQueries.length)];
  
  const response = http.post(`${BASE_URL}/api/query`, 
    JSON.stringify({
      query: query,
      limit: 10
    }), 
    {
      headers: { 'Content-Type': 'application/json' },
      timeout: '30s'
    }
  );
  
  const success = check(response, {
    'query status is 200': (r) => r.status === 200,
    'query response time < 5s': (r) => r.timings.duration < 5000,
    'query has results': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.results && body.results.length > 0;
      } catch (e) {
        return false;
      }
    }
  });
  
  errorRate.add(!success);
  responseTime.add(response.timings.duration);
}

function testDocumentUpload() {
  const doc = testDocuments[Math.floor(Math.random() * testDocuments.length)];
  
  const response = http.post(`${BASE_URL}/api/documents`, 
    JSON.stringify(doc), 
    {
      headers: { 'Content-Type': 'application/json' },
      timeout: '60s'
    }
  );
  
  const success = check(response, {
    'upload status is 201': (r) => r.status === 201,
    'upload response time < 10s': (r) => r.timings.duration < 10000,
    'upload returns document id': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.document_id !== undefined;
      } catch (e) {
        return false;
      }
    }
  });
  
  errorRate.add(!success);
  responseTime.add(response.timings.duration);
}

function testEntityQuery() {
  const response = http.get(`${BASE_URL}/api/entities?limit=20`, {
    timeout: '15s'
  });
  
  const success = check(response, {
    'entity query status is 200': (r) => r.status === 200,
    'entity query response time < 3s': (r) => r.timings.duration < 3000,
    'entity query has results': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.entities && Array.isArray(body.entities);
      } catch (e) {
        return false;
      }
    }
  });
  
  errorRate.add(!success);
  responseTime.add(response.timings.duration);
}

function testRelationQuery() {
  const response = http.get(`${BASE_URL}/api/relations?limit=20`, {
    timeout: '15s'
  });
  
  const success = check(response, {
    'relation query status is 200': (r) => r.status === 200,
    'relation query response time < 3s': (r) => r.timings.duration < 3000,
    'relation query has results': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.relations && Array.isArray(body.relations);
      } catch (e) {
        return false;
      }
    }
  });
  
  errorRate.add(!success);
  responseTime.add(response.timings.duration);
}

// 测试结束后的清理函数
export function teardown(data) {
  console.log('负载测试完成');
  console.log(`总请求数: ${data.http_reqs}`);
  console.log(`平均响应时间: ${data.http_req_duration.avg}ms`);
  console.log(`错误率: ${data.http_req_failed.rate * 100}%`);
}
```

## 7. 性能优化最佳实践

### 7.1 代码级优化建议

```python
# src/core/performance/best_practices.py
"""
性能优化最佳实践
提供代码级别的性能优化建议和工具
"""

import asyncio
import functools
from typing import Any, Callable, List, Dict
import time
import cProfile
import pstats
from memory_profiler import profile
import tracemalloc

class PerformanceOptimizationGuide:
    """性能优化指南"""
    
    @staticmethod
    def optimize_database_queries():
        """数据库查询优化建议"""
        return {
            "索引优化": [
                "为经常查询的字段创建索引",
                "使用复合索引优化多字段查询",
                "定期分析索引使用情况，删除无用索引",
                "使用部分索引减少索引大小"
            ],
            "查询优化": [
                "使用EXPLAIN分析查询计划",
                "避免SELECT *，只查询需要的字段",
                "使用批量操作减少数据库往返",
                "合理使用JOIN，避免N+1查询问题"
            ],
            "连接池优化": [
                "配置合适的连接池大小",
                "设置连接超时和空闲超时",
                "使用连接池监控工具",
                "实现连接池健康检查"
            ]
        }
    
    @staticmethod
    def optimize_caching_strategy():
        """缓存策略优化建议"""
        return {
            "缓存层级": [
                "L1: 应用内存缓存（最快）",
                "L2: Redis分布式缓存（中等）", 
                "L3: 数据库查询缓存（较慢）"
            ],
            "缓存策略": [
                "热点数据使用LRU缓存",
                "静态数据使用TTL缓存",
                "大对象使用压缩存储",
                "实现缓存预热机制"
            ],
            "缓存失效": [
                "使用版本号控制缓存一致性",
                "实现主动缓存失效",
                "设置合理的过期时间",
                "监控缓存命中率"
            ]
        }
    
    @staticmethod
    def optimize_async_processing():
        """异步处理优化建议"""
        return {
            "并发控制": [
                "使用信号量限制并发数",
                "实现任务队列避免过载",
                "使用连接池管理资源",
                "合理设置超时时间"
            ],
            "任务调度": [
                "CPU密集型任务使用进程池",
                "IO密集型任务使用线程池",
                "长时间任务使用异步队列",
                "实现任务优先级机制"
            ],
            "错误处理": [
                "实现重试机制",
                "使用断路器模式",
                "记录详细的错误日志",
                "实现优雅降级"
            ]
        }

def profile_performance(func: Callable) -> Callable:
    """性能分析装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 开始内存跟踪
        tracemalloc.start()
        
        # 开始CPU分析
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            
            # 停止CPU分析
            profiler.disable()
            
            # 获取内存使用情况
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # 输出性能报告
            print(f"\n=== 性能分析报告: {func.__name__} ===")
            print(f"执行时间: {end_time - start_time:.4f} 秒")
            print(f"当前内存使用: {current / 1024 / 1024:.2f} MB")
            print(f"峰值内存使用: {peak / 1024 / 1024:.2f} MB")
            
            # 输出CPU分析结果
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(10)  # 显示前10个最耗时的函数
    
    return wrapper

@profile  # memory_profiler装饰器
def memory_intensive_function():
    """内存密集型函数示例"""
    # 这个装饰器会逐行分析内存使用情况
    data = []
    for i in range(100000):
        data.append(i * i)
    return data

# 性能优化工具函数
class PerformanceTools:
    """性能优化工具集"""
    
    @staticmethod
    async def benchmark_function(func: Callable, *args, iterations: int = 100, **kwargs):
        """
        函数基准测试
        
        Args:
            func: 要测试的函数
            iterations: 迭代次数
            *args, **kwargs: 函数参数
            
        Returns:
            基准测试结果
        """
        times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            
            if asyncio.iscoroutinefunction(func):
                await func(*args, **kwargs)
            else:
                func(*args, **kwargs)
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return {
            "iterations": iterations,
            "total_time": sum(times),
            "average_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "median_time": sorted(times)[len(times) // 2]
        }
    
    @staticmethod
    def compare_algorithms(algorithms: Dict[str, Callable], test_data: Any, iterations: int = 10):
        """
        算法性能比较
        
        Args:
            algorithms: 算法字典 {名称: 函数}
            test_data: 测试数据
            iterations: 迭代次数
            
        Returns:
            比较结果
        """
        results = {}
        
        for name, algorithm in algorithms.items():
            times = []
            
            for _ in range(iterations):
                start_time = time.perf_counter()
                algorithm(test_data)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            results[name] = {
                "average_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times)
            }
        
        # 按平均时间排序
        sorted_results = sorted(results.items(), key=lambda x: x[1]["average_time"])
        
        print("\n=== 算法性能比较结果 ===")
        for i, (name, stats) in enumerate(sorted_results):
            print(f"{i+1}. {name}: {stats['average_time']:.6f}s (平均)")
        
        return results
    
    @staticmethod
    def memory_usage_tracker(func: Callable):
        """内存使用跟踪装饰器"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            # 执行前的内存使用
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            # 执行后的内存使用
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_diff = memory_after - memory_before
            
            print(f"\n=== 内存使用跟踪: {func.__name__} ===")
            print(f"执行前内存: {memory_before:.2f} MB")
            print(f"执行后内存: {memory_after:.2f} MB")
            print(f"内存变化: {memory_diff:+.2f} MB")
            
            return result
        
        return wrapper

# 使用示例
if __name__ == "__main__":
    # 性能分析示例
    @profile_performance
    def example_function():
        """示例函数"""
        data = [i ** 2 for i in range(100000)]
        return sum(data)
    
    # 执行性能分析
    result = example_function()
    
    # 算法比较示例
    def bubble_sort(arr):
        """冒泡排序"""
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr
    
    def quick_sort(arr):
        """快速排序"""
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quick_sort(left) + middle + quick_sort(right)
    
    # 比较排序算法
    test_data = list(range(1000, 0, -1))  # 逆序数组
    algorithms = {
        "冒泡排序": bubble_sort,
        "快速排序": quick_sort,
        "内置排序": sorted
    }
    
    PerformanceTools.compare_algorithms(algorithms, test_data.copy(), iterations=5)
```

## 总结

本性能优化指南提供了GraphRAG知识库系统的全面性能优化策略，包括：

1. **数据库优化**：PostgreSQL和Neo4j的配置优化、索引策略
2. **应用层优化**：FastAPI性能优化、异步处理、缓存策略
3. **多层缓存架构**：内存缓存、Redis缓存、数据库缓存
4. **负载均衡和扩展**：Nginx配置、Kubernetes HPA
5. **性能监控**：指标收集、Grafana仪表板、性能分析
6. **负载测试**：k6测试脚本、基准测试工具
7. **最佳实践**：代码级优化建议、性能分析工具

通过实施这些优化策略，可以显著提升系统的性能、可扩展性和稳定性，确保GraphRAG系统能够处理大规模的知识库数据和高并发的用户请求。