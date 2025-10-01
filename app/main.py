#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 知识库系统 - FastAPI 主应用入口
==================================

本文件是 GraphRAG 系统的主要 FastAPI 应用入口，负责：
1. 初始化 FastAPI 应用实例
2. 配置中间件（日志记录、CORS、异常处理等）
3. 注册路由模块
4. 配置应用元数据和文档

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import time
import uuid
from datetime import datetime
from typing import Dict, Any

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.middleware.logging_middleware import LoggingMiddleware
from app.api.v1.router import api_router


# 设置日志系统
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理器
    
    负责应用启动和关闭时的初始化和清理工作：
    - 启动时：初始化数据库连接、缓存等
    - 关闭时：清理资源、关闭连接等
    """
    # 启动时执行
    logger.info("GraphRAG API 服务启动中...")
    logger.info(f"服务配置: {settings.APP_NAME} v{settings.VERSION}")
    logger.info(f"运行环境: {settings.ENVIRONMENT}")
    logger.info(f"调试模式: {settings.DEBUG}")
    
    yield
    
    # 关闭时执行
    logger.info("GraphRAG API 服务正在关闭...")


def create_application() -> FastAPI:
    """
    创建并配置 FastAPI 应用实例
    
    Returns:
        FastAPI: 配置完成的 FastAPI 应用实例
    """
    # 创建 FastAPI 应用实例
    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.DESCRIPTION,
        version=settings.VERSION,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
        lifespan=lifespan
    )
    
    # 配置 CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 添加日志记录中间件
    app.add_middleware(LoggingMiddleware)
    
    # 注册 API 路由
    app.include_router(api_router, prefix=settings.API_V1_STR)
    
    # 添加全局异常处理器
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        全局异常处理器
        
        捕获所有未处理的异常，记录日志并返回统一的错误响应
        """
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        logger.error(
            f"未处理的异常",
            extra={
                'request_id': request_id,
                'method': request.method,
                'url': str(request.url),
                'exception': str(exc),
                'exception_type': type(exc).__name__
            },
            exc_info=True
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "服务器内部错误，请稍后重试"
                },
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "request_id": request_id
            }
        )
    
    # 添加健康检查端点
    @app.get("/health", tags=["系统"])
    async def health_check() -> Dict[str, Any]:
        """
        健康检查端点
        
        用于检查服务是否正常运行，通常被负载均衡器或监控系统调用
        
        Returns:
            Dict[str, Any]: 健康状态信息
        """
        return {
            "success": True,
            "data": {
                "status": "healthy",
                "service": settings.APP_NAME,
                "version": settings.VERSION,
                "environment": settings.ENVIRONMENT,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            },
            "message": "服务运行正常"
        }
    
    # 添加根路径端点
    @app.get("/", tags=["系统"])
    async def root() -> Dict[str, Any]:
        """
        根路径端点
        
        返回 API 基本信息和文档链接
        
        Returns:
            Dict[str, Any]: API 基本信息
        """
        return {
            "success": True,
            "data": {
                "service": settings.APP_NAME,
                "version": settings.VERSION,
                "description": settings.DESCRIPTION,
                "docs_url": "/docs" if settings.DEBUG else None,
                "api_prefix": settings.API_V1_STR
            },
            "message": "欢迎使用 GraphRAG 知识库系统 API"
        }
    
    logger.info("FastAPI 应用创建完成")
    return app


# 创建应用实例
app = create_application()


if __name__ == "__main__":
    """
    直接运行时的入口点
    
    用于开发环境直接启动服务，生产环境建议使用 gunicorn 或其他 WSGI 服务器
    """
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug"
    )