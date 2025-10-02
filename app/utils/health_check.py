#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 服务健康检查工具模块
============================

本模块提供各种基础服务的连接状态检查功能，包括：
1. PostgreSQL 数据库连接检查
2. Neo4j 图数据库连接检查
3. Redis 缓存服务连接检查
4. Weaviate 向量数据库连接检查
5. MinIO 对象存储服务连接检查
6. 系统资源监控（CPU、内存、磁盘使用率）

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import asyncio
import logging
import psutil
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager

# 数据库和服务连接库
import asyncpg
import redis.asyncio as redis
import weaviate
from neo4j import AsyncGraphDatabase
from minio import Minio
from minio.error import S3Error

from app.core.config import settings

logger = logging.getLogger(__name__)


class HealthChecker:
    """
    服务健康检查器
    
    提供各种基础服务的连接状态检查和系统资源监控功能。
    """
    
    def __init__(self):
        """初始化健康检查器"""
        self.start_time = time.time()
        
    async def check_all_services(self) -> Dict[str, Any]:
        """
        检查所有服务的健康状态
        
        Returns:
            Dict[str, Any]: 包含所有服务状态的字典
        """
        logger.info("开始检查所有服务健康状态")
        
        # 并发检查所有服务
        tasks = {
            "postgres": self.check_postgres(),
            "neo4j": self.check_neo4j(),
            "redis": self.check_redis(),
            "weaviate": self.check_weaviate(),
            "minio": self.check_minio(),
        }
        
        # 等待所有检查完成
        results = {}
        for service_name, task in tasks.items():
            try:
                results[service_name] = await asyncio.wait_for(task, timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning(f"{service_name} 服务检查超时")
                results[service_name] = "timeout"
            except Exception as e:
                logger.error(f"{service_name} 服务检查失败: {e}")
                results[service_name] = "error"
        
        logger.info(f"服务健康检查完成: {results}")
        return results
    
    async def check_postgres(self) -> str:
        """
        检查 PostgreSQL 数据库连接状态
        
        Returns:
            str: 连接状态 ("connected", "disconnected", "error")
        """
        try:
            # 构建连接字符串
            connection_string = (
                f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
                f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
            )
            
            # 尝试连接数据库
            conn = await asyncpg.connect(connection_string)
            
            # 执行简单查询测试连接
            result = await conn.fetchval("SELECT 1")
            await conn.close()
            
            if result == 1:
                logger.debug("PostgreSQL 连接正常")
                return "connected"
            else:
                logger.warning("PostgreSQL 查询结果异常")
                return "error"
                
        except Exception as e:
            logger.error(f"PostgreSQL 连接检查失败: {e}")
            return "disconnected"
    
    async def check_neo4j(self) -> str:
        """
        检查 Neo4j 图数据库连接状态
        
        Returns:
            str: 连接状态 ("connected", "disconnected", "error")
        """
        try:
            # 构建连接 URI
            uri = f"bolt://{settings.NEO4J_HOST}:{settings.NEO4J_PORT}"
            
            # 创建驱动程序
            driver = AsyncGraphDatabase.driver(
                uri,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            
            # 验证连接
            await driver.verify_connectivity()
            
            # 执行简单查询测试
            async with driver.session() as session:
                result = await session.run("RETURN 1 as test")
                record = await result.single()
                
                if record and record["test"] == 1:
                    logger.debug("Neo4j 连接正常")
                    await driver.close()
                    return "connected"
                else:
                    logger.warning("Neo4j 查询结果异常")
                    await driver.close()
                    return "error"
                    
        except Exception as e:
            logger.error(f"Neo4j 连接检查失败: {e}")
            return "disconnected"
    
    async def check_redis(self) -> str:
        """
        检查 Redis 缓存服务连接状态
        
        Returns:
            str: 连接状态 ("connected", "disconnected", "error")
        """
        try:
            # 创建 Redis 连接
            redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                db=settings.REDIS_DB,
                decode_responses=True
            )
            
            # 测试连接
            pong = await redis_client.ping()
            await redis_client.aclose()
            
            if pong:
                logger.debug("Redis 连接正常")
                return "connected"
            else:
                logger.warning("Redis ping 失败")
                return "error"
                
        except Exception as e:
            logger.error(f"Redis 连接检查失败: {e}")
            return "disconnected"
    
    async def check_weaviate(self) -> str:
        """
        检查 Weaviate 向量数据库连接状态
        
        Returns:
            str: 连接状态 ("connected", "disconnected", "error")
        """
        try:
            # 创建 Weaviate 客户端
            client = weaviate.Client(
                url=f"{settings.WEAVIATE_SCHEME}://{settings.WEAVIATE_HOST}:{settings.WEAVIATE_PORT}"
            )
            
            # 检查连接状态
            is_ready = client.is_ready()
            
            if is_ready:
                logger.debug("Weaviate 连接正常")
                return "connected"
            else:
                logger.warning("Weaviate 服务未就绪")
                return "disconnected"
                
        except Exception as e:
            logger.error(f"Weaviate 连接检查失败: {e}")
            return "disconnected"
    
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
            buckets = await loop.run_in_executor(None, client.list_buckets)
            
            logger.debug(f"MinIO 连接正常，发现 {len(buckets)} 个存储桶")
            return "connected"
            
        except S3Error as e:
            logger.error(f"MinIO S3 错误: {e}")
            return "disconnected"
        except Exception as e:
            logger.error(f"MinIO 连接检查失败: {e}")
            return "disconnected"
    
    def get_system_resources(self) -> Dict[str, str]:
        """
        获取系统资源使用情况
        
        Returns:
            Dict[str, str]: 系统资源使用情况
        """
        try:
            # 获取 CPU 使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 获取内存使用情况
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # 获取磁盘使用情况
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            logger.debug(f"系统资源: CPU={cpu_percent}%, 内存={memory_percent}%, 磁盘={disk_percent:.1f}%")
            
            return {
                "cpu_usage": f"{cpu_percent:.1f}%",
                "memory_usage": f"{memory_percent:.1f}%",
                "disk_usage": f"{disk_percent:.1f}%"
            }
            
        except Exception as e:
            logger.error(f"获取系统资源信息失败: {e}")
            return {
                "cpu_usage": "unknown",
                "memory_usage": "unknown",
                "disk_usage": "unknown"
            }
    
    def get_uptime(self) -> str:
        """
        获取应用运行时间
        
        Returns:
            str: 运行时间字符串
        """
        try:
            uptime_seconds = int(time.time() - self.start_time)
            
            days = uptime_seconds // 86400
            hours = (uptime_seconds % 86400) // 3600
            minutes = (uptime_seconds % 3600) // 60
            
            return f"{days}d {hours}h {minutes}m"
            
        except Exception as e:
            logger.error(f"计算运行时间失败: {e}")
            return "unknown"


# 全局健康检查器实例
health_checker = HealthChecker()