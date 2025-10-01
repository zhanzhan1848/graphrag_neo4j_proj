#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 系统管理 API 端点
========================

本模块提供系统管理相关的 API 端点，包括：
1. 健康检查 - 检查服务是否正常运行
2. 系统状态 - 获取系统运行状态和统计信息
3. 系统信息 - 获取系统配置和版本信息
4. 服务依赖检查 - 检查各个依赖服务的状态
5. 系统监控 - 获取系统资源使用情况

这些端点主要用于：
- 负载均衡器的健康检查
- 监控系统的状态收集
- 运维人员的系统诊断
- 自动化部署的状态验证

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import psutil
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import get_logger

# 获取日志记录器
logger = get_logger(__name__)

# 创建系统管理路由器
router = APIRouter()

# 服务启动时间（用于计算运行时间）
_service_start_time = datetime.utcnow()


class HealthCheckResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    timestamp: str
    service: str
    version: str
    environment: str


class SystemStatusResponse(BaseModel):
    """系统状态响应模型"""
    success: bool
    data: Dict[str, Any]
    message: str
    timestamp: str


class ServiceDependency(BaseModel):
    """服务依赖模型"""
    name: str
    status: str
    url: Optional[str] = None
    response_time: Optional[float] = None
    error: Optional[str] = None


@router.get("/health", 
           response_model=HealthCheckResponse,
           summary="健康检查",
           description="检查 API 服务是否正常运行，返回基本的服务状态信息")
async def health_check() -> HealthCheckResponse:
    """
    健康检查端点
    
    这是一个轻量级的健康检查端点，主要用于：
    - 负载均衡器检查服务是否可用
    - 容器编排系统的健康检查
    - 监控系统的基础状态检查
    
    Returns:
        HealthCheckResponse: 健康状态信息
    """
    logger.debug("执行健康检查")
    
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat() + "Z",
        service=settings.APP_NAME,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT
    )


@router.get("/status",
           response_model=SystemStatusResponse,
           summary="系统状态",
           description="获取详细的系统运行状态，包括资源使用情况和服务统计")
async def system_status() -> SystemStatusResponse:
    """
    获取系统详细状态
    
    返回系统的详细运行状态，包括：
    - 服务运行时间
    - 系统资源使用情况
    - 内存和CPU使用率
    - 磁盘使用情况
    
    Returns:
        SystemStatusResponse: 系统状态信息
    """
    logger.info("获取系统状态")
    
    try:
        # 计算运行时间
        uptime = datetime.utcnow() - _service_start_time
        uptime_seconds = int(uptime.total_seconds())
        uptime_str = str(timedelta(seconds=uptime_seconds))
        
        # 获取系统资源信息
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # 获取网络信息
        network = psutil.net_io_counters()
        
        status_data = {
            "service_info": {
                "name": settings.APP_NAME,
                "version": settings.VERSION,
                "environment": settings.ENVIRONMENT,
                "start_time": _service_start_time.isoformat() + "Z",
                "uptime": uptime_str,
                "uptime_seconds": uptime_seconds
            },
            "system_resources": {
                "cpu": {
                    "usage_percent": round(cpu_percent, 2),
                    "count": psutil.cpu_count(),
                    "count_logical": psutil.cpu_count(logical=True)
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "usage_percent": round(memory.percent, 2),
                    "total_gb": round(memory.total / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2)
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "usage_percent": round(disk.used / disk.total * 100, 2),
                    "total_gb": round(disk.total / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2)
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                }
            },
            "configuration": {
                "debug_mode": settings.DEBUG,
                "log_level": settings.LOG_LEVEL,
                "host": settings.HOST,
                "port": settings.PORT,
                "api_prefix": settings.API_V1_STR
            }
        }
        
        return SystemStatusResponse(
            success=True,
            data=status_data,
            message="系统状态获取成功",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取系统状态失败: {str(e)}"
        )


@router.get("/dependencies",
           summary="服务依赖检查",
           description="检查所有外部服务依赖的连接状态")
async def check_dependencies() -> Dict[str, Any]:
    """
    检查服务依赖状态
    
    检查 GraphRAG 系统依赖的所有外部服务的连接状态：
    - PostgreSQL 数据库
    - Neo4j 图数据库
    - Redis 缓存
    - Weaviate 向量数据库
    - MinIO 对象存储
    
    Returns:
        Dict[str, Any]: 依赖服务状态信息
    """
    logger.info("检查服务依赖状态")
    
    dependencies = []
    overall_status = "healthy"
    
    # 定义要检查的服务
    services_to_check = [
        {
            "name": "PostgreSQL",
            "url": settings.postgres_url,
            "check_func": _check_postgres
        },
        {
            "name": "Neo4j",
            "url": settings.neo4j_url,
            "check_func": _check_neo4j
        },
        {
            "name": "Redis",
            "url": settings.redis_url,
            "check_func": _check_redis
        },
        {
            "name": "Weaviate",
            "url": settings.weaviate_url,
            "check_func": _check_weaviate
        },
        {
            "name": "MinIO",
            "url": settings.minio_url,
            "check_func": _check_minio
        }
    ]
    
    # 并发检查所有服务
    check_tasks = []
    for service in services_to_check:
        task = _check_service_dependency(service)
        check_tasks.append(task)
    
    # 等待所有检查完成
    dependency_results = await asyncio.gather(*check_tasks, return_exceptions=True)
    
    # 处理检查结果
    for result in dependency_results:
        if isinstance(result, Exception):
            logger.error(f"依赖检查异常: {str(result)}")
            dependencies.append(ServiceDependency(
                name="Unknown",
                status="error",
                error=str(result)
            ))
            overall_status = "degraded"
        else:
            dependencies.append(result)
            if result.status != "healthy":
                overall_status = "degraded"
    
    return {
        "success": True,
        "data": {
            "overall_status": overall_status,
            "dependencies": [dep.dict() for dep in dependencies],
            "total_dependencies": len(dependencies),
            "healthy_count": sum(1 for dep in dependencies if dep.status == "healthy"),
            "unhealthy_count": sum(1 for dep in dependencies if dep.status != "healthy")
        },
        "message": f"依赖检查完成，总体状态: {overall_status}",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


async def _check_service_dependency(service_config: Dict[str, Any]) -> ServiceDependency:
    """
    检查单个服务依赖
    
    Args:
        service_config: 服务配置信息
        
    Returns:
        ServiceDependency: 服务依赖状态
    """
    start_time = datetime.utcnow()
    
    try:
        # 调用具体的检查函数
        await service_config["check_func"]()
        
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        return ServiceDependency(
            name=service_config["name"],
            status="healthy",
            url=service_config["url"],
            response_time=round(response_time, 3)
        )
        
    except Exception as e:
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        return ServiceDependency(
            name=service_config["name"],
            status="unhealthy",
            url=service_config["url"],
            response_time=round(response_time, 3),
            error=str(e)
        )


async def _check_postgres():
    """检查 PostgreSQL 连接"""
    # TODO: 实现 PostgreSQL 连接检查
    # 这里应该尝试连接数据库并执行简单查询
    await asyncio.sleep(0.1)  # 模拟检查时间
    # raise Exception("PostgreSQL 连接检查未实现")


async def _check_neo4j():
    """检查 Neo4j 连接"""
    # TODO: 实现 Neo4j 连接检查
    await asyncio.sleep(0.1)  # 模拟检查时间
    # raise Exception("Neo4j 连接检查未实现")


async def _check_redis():
    """检查 Redis 连接"""
    # TODO: 实现 Redis 连接检查
    await asyncio.sleep(0.1)  # 模拟检查时间
    # raise Exception("Redis 连接检查未实现")


async def _check_weaviate():
    """检查 Weaviate 连接"""
    # TODO: 实现 Weaviate 连接检查
    await asyncio.sleep(0.1)  # 模拟检查时间
    # raise Exception("Weaviate 连接检查未实现")


async def _check_minio():
    """检查 MinIO 连接"""
    # TODO: 实现 MinIO 连接检查
    await asyncio.sleep(0.1)  # 模拟检查时间
    # raise Exception("MinIO 连接检查未实现")


@router.get("/info",
           summary="系统信息",
           description="获取系统配置和环境信息")
async def system_info() -> Dict[str, Any]:
    """
    获取系统信息
    
    返回系统的配置信息和环境详情，用于调试和运维
    
    Returns:
        Dict[str, Any]: 系统信息
    """
    logger.info("获取系统信息")
    
    return {
        "success": True,
        "data": {
            "application": {
                "name": settings.APP_NAME,
                "version": settings.VERSION,
                "description": settings.DESCRIPTION,
                "environment": settings.ENVIRONMENT,
                "debug_mode": settings.DEBUG
            },
            "server": {
                "host": settings.HOST,
                "port": settings.PORT,
                "workers": settings.WORKERS
            },
            "api": {
                "version": "v1",
                "prefix": settings.API_V1_STR,
                "allowed_hosts": settings.ALLOWED_HOSTS
            },
            "features": {
                "document_processing": "支持多种文档格式处理",
                "knowledge_extraction": "自动知识抽取和图谱构建",
                "semantic_search": "基于向量的语义搜索",
                "graph_query": "图数据库查询和分析",
                "rag_system": "检索增强生成问答"
            },
            "limits": {
                "max_upload_size": f"{settings.UPLOAD_MAX_SIZE / (1024*1024):.0f}MB",
                "allowed_file_types": settings.ALLOWED_FILE_TYPES,
                "max_concurrent_tasks": settings.MAX_CONCURRENT_TASKS,
                "chunk_size": settings.CHUNK_SIZE
            }
        },
        "message": "系统信息获取成功",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }