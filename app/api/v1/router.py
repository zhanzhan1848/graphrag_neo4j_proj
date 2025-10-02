#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG API v1 路由器
====================

本模块定义了 GraphRAG 系统的 v1 版本 API 路由，包括：
1. 系统管理相关接口
2. 文档管理相关接口
3. 知识图谱相关接口
4. 搜索和查询相关接口
5. 用户认证相关接口

所有 API 接口都遵循 RESTful 设计原则，并提供完整的错误处理和日志记录。

路由结构：
- /api/v1/system - 系统管理接口
- /api/v1/documents - 文档管理接口
- /api/v1/knowledge - 知识图谱接口
- /api/v1/search - 搜索查询接口
- /api/v1/auth - 用户认证接口

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import Dict, Any

from app.core.config import settings
from app.core.logging import get_logger
from app.api.v1.endpoints import system, graphrag

# 获取日志记录器
logger = get_logger(__name__)

# 创建 API 路由器
api_router = APIRouter()

# 包含系统管理路由
api_router.include_router(
    system.router,
    prefix="/system",
    tags=["系统管理"]
)

# 包含 GraphRAG 路由
api_router.include_router(
    graphrag.router,
    prefix="/graphrag",
    tags=["GraphRAG"]
)

# TODO: 添加其他路由模块
# api_router.include_router(
#     documents.router,
#     prefix="/documents",
#     tags=["文档管理"]
# )
# 
# api_router.include_router(
#     knowledge.router,
#     prefix="/knowledge",
#     tags=["知识图谱"]
# )
# 
# api_router.include_router(
#     search.router,
#     prefix="/search",
#     tags=["搜索查询"]
# )
# 
# api_router.include_router(
#     auth.router,
#     prefix="/auth",
#     tags=["用户认证"]
# )


@api_router.get("/", tags=["API信息"])
async def api_info() -> Dict[str, Any]:
    """
    获取 API 基本信息
    
    返回 GraphRAG API 的基本信息，包括版本、描述、功能等。
    
    Returns:
        Dict[str, Any]: API 基本信息
        
    Example:
        ```json
        {
            "name": "GraphRAG API",
            "version": "1.0.0",
            "description": "知识图谱增强检索生成系统 API",
            "status": "running",
            "timestamp": "2024-01-01T00:00:00Z",
            "features": [
                "文档处理与存储",
                "实体关系抽取",
                "知识图谱构建",
                "语义检索",
                "RAG 问答"
            ],
            "endpoints": {
                "docs": "/docs",
                "redoc": "/redoc",
                "openapi": "/openapi.json"
            }
        }
        ```
    """
    logger.info("API 信息请求")
    
    return {
        "name": "GraphRAG API",
        "version": settings.VERSION,
        "description": settings.DESCRIPTION,
        "status": "running",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "features": [
            "文档处理与存储",
            "实体关系抽取", 
            "知识图谱构建",
            "语义检索",
            "RAG 问答"
        ],
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc", 
            "openapi": "/openapi.json"
        }
    }


@api_router.get("/status", tags=["API信息"])
async def api_status() -> Dict[str, Any]:
    """
    获取 API 运行状态
    
    返回 GraphRAG API 的详细运行状态，包括系统资源、数据库连接等。
    
    Returns:
        Dict[str, Any]: API 运行状态信息
        
    Example:
        ```json
        {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",
            "uptime": "1d 2h 30m",
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
                "cpu_usage": "15.2%",
                "memory_usage": "45.8%",
                "disk_usage": "23.1%"
            }
        }
        ```
    """
    logger.info("API 状态检查请求")
    
    # 导入健康检查器
    from app.utils.health_check import health_checker
    
    try:
        # 检查所有服务状态
        services_status = await health_checker.check_all_services()
        
        # 获取系统资源信息
        system_resources = health_checker.get_system_resources()
        
        # 获取运行时间
        uptime = health_checker.get_uptime()
        
        # 判断整体健康状态
        # 如果所有服务都连接正常，则状态为 healthy
        all_connected = all(status == "connected" for status in services_status.values())
        overall_status = "healthy" if all_connected else "degraded"
        
        # 如果有服务完全无法连接，则状态为 unhealthy
        any_disconnected = any(status == "disconnected" for status in services_status.values())
        if any_disconnected:
            overall_status = "unhealthy"
        
        logger.info(f"API 状态检查完成，整体状态: {overall_status}")
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "uptime": uptime,
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "services": services_status,
            "system": system_resources
        }
        
    except Exception as e:
        logger.error(f"API 状态检查失败: {e}")
        
        # 如果检查过程中出现异常，返回错误状态
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "uptime": "unknown",
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "services": {
                "postgres": "error",
                "neo4j": "error",
                "redis": "error",
                "weaviate": "error",
                "minio": "error"
            },
            "system": {
                "cpu_usage": "unknown",
                "memory_usage": "unknown",
                "disk_usage": "unknown"
            },
            "error": str(e)
        }