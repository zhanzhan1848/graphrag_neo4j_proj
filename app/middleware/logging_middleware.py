#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 系统日志记录中间件
=========================

本模块实现 FastAPI 中间件，负责：
1. 记录所有 HTTP 请求和响应的详细信息
2. 测量和记录每个请求的处理时间
3. 生成唯一的请求 ID 用于追踪
4. 记录请求体和响应体（可配置）
5. 异常处理和错误日志记录
6. 性能监控和统计信息

支持的功能：
- 请求/响应日志记录
- 执行时间统计
- 错误追踪
- 性能监控
- 请求去重检测

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import time
import uuid
import json
from typing import Callable, Dict, Any, Optional
from datetime import datetime

from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.config import settings
from app.core.logging import get_logger, get_access_logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    日志记录中间件
    
    拦截所有 HTTP 请求，记录详细的访问日志和性能指标
    """
    
    def __init__(self, app: ASGIApp):
        """
        初始化日志中间件
        
        Args:
            app: ASGI 应用实例
        """
        super().__init__(app)
        self.logger = get_logger(__name__)
        self.access_logger = get_access_logger()
        
        # 不需要记录详细日志的路径
        self.skip_paths = {
            "/health",
            "/metrics",
            "/favicon.ico"
        }
        
        # 不需要记录请求体的路径（通常是文件上传等大数据请求）
        self.skip_body_paths = {
            "/api/v1/documents/upload",
            "/api/v1/files/upload"
        }
        
        # 敏感信息字段，需要脱敏处理
        self.sensitive_fields = {
            "password", "token", "secret", "key", "authorization",
            "x-api-key", "x-auth-token", "cookie"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        处理请求的主要方法
        
        Args:
            request: HTTP 请求对象
            call_next: 下一个中间件或路由处理器
            
        Returns:
            Response: HTTP 响应对象
        """
        # 生成唯一的请求 ID
        request_id = str(uuid.uuid4())
        
        # 将请求 ID 添加到请求状态中，供其他地方使用
        request.state.request_id = request_id
        
        # 记录请求开始时间
        start_time = time.time()
        
        # 获取请求信息
        request_info = await self._extract_request_info(request, request_id)
        
        # 记录请求开始日志
        if not self._should_skip_logging(request.url.path):
            self.access_logger.info(
                "请求开始",
                extra={
                    **request_info,
                    "event": "request_start"
                }
            )
        
        # 处理请求
        response = None
        exception_info = None
        
        try:
            response = await call_next(request)
        except Exception as exc:
            # 记录异常信息
            exception_info = {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "exception_traceback": self._format_exception(exc)
            }
            
            self.logger.error(
                f"请求处理异常: {str(exc)}",
                extra={
                    **request_info,
                    **exception_info,
                    "event": "request_exception"
                },
                exc_info=True
            )
            
            # 重新抛出异常，让全局异常处理器处理
            raise
        
        finally:
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 获取响应信息
            response_info = self._extract_response_info(response) if response else {}
            
            # 记录请求完成日志
            if not self._should_skip_logging(request.url.path):
                log_data = {
                    **request_info,
                    **response_info,
                    "process_time": round(process_time, 4),
                    "event": "request_complete"
                }
                
                if exception_info:
                    log_data.update(exception_info)
                
                # 根据响应状态码选择日志级别
                if response and response.status_code >= 500:
                    self.access_logger.error("请求完成 - 服务器错误", extra=log_data)
                elif response and response.status_code >= 400:
                    self.access_logger.warning("请求完成 - 客户端错误", extra=log_data)
                else:
                    self.access_logger.info("请求完成", extra=log_data)
            
            # 记录性能监控日志
            self._log_performance_metrics(request_info, process_time, response)
        
        # 添加响应头
        if response:
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(round(process_time, 4))
        
        return response
    
    async def _extract_request_info(self, request: Request, request_id: str) -> Dict[str, Any]:
        """
        提取请求信息
        
        Args:
            request: HTTP 请求对象
            request_id: 请求 ID
            
        Returns:
            Dict[str, Any]: 请求信息字典
        """
        # 基础请求信息
        request_info = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent", ""),
            "content_type": request.headers.get("content-type", ""),
            "content_length": request.headers.get("content-length", 0)
        }
        
        # 添加请求头（脱敏处理）
        request_info["headers"] = self._sanitize_headers(dict(request.headers))
        
        # 添加请求体（如果需要且不是大文件上传）
        if (request.method in ["POST", "PUT", "PATCH"] and 
            request.url.path not in self.skip_body_paths and
            request_info["content_length"] and 
            int(request_info["content_length"]) < 10240):  # 小于 10KB
            
            try:
                body = await self._get_request_body(request)
                if body:
                    request_info["body"] = self._sanitize_body(body)
            except Exception as e:
                self.logger.warning(f"无法读取请求体: {str(e)}")
        
        return request_info
    
    def _extract_response_info(self, response: Response) -> Dict[str, Any]:
        """
        提取响应信息
        
        Args:
            response: HTTP 响应对象
            
        Returns:
            Dict[str, Any]: 响应信息字典
        """
        if not response:
            return {}
        
        response_info = {
            "status_code": response.status_code,
            "response_headers": dict(response.headers),
            "content_type": response.headers.get("content-type", ""),
        }
        
        # 如果是小响应且不是流式响应，记录响应体
        if (hasattr(response, 'body') and 
            not isinstance(response, StreamingResponse) and
            len(getattr(response, 'body', b'')) < 10240):  # 小于 10KB
            
            try:
                body = getattr(response, 'body', b'')
                if body:
                    response_info["body"] = body.decode('utf-8')[:1000]  # 最多记录 1000 字符
            except Exception as e:
                self.logger.debug(f"无法读取响应体: {str(e)}")
        
        return response_info
    
    async def _get_request_body(self, request: Request) -> Optional[str]:
        """
        获取请求体内容
        
        Args:
            request: HTTP 请求对象
            
        Returns:
            Optional[str]: 请求体内容
        """
        try:
            body = await request.body()
            if body:
                return body.decode('utf-8')
        except Exception:
            pass
        return None
    
    def _get_client_ip(self, request: Request) -> str:
        """
        获取客户端真实 IP 地址
        
        Args:
            request: HTTP 请求对象
            
        Returns:
            str: 客户端 IP 地址
        """
        # 尝试从各种头部获取真实 IP
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # 返回直接连接的 IP
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"
    
    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        对请求头进行脱敏处理
        
        Args:
            headers: 原始请求头字典
            
        Returns:
            Dict[str, str]: 脱敏后的请求头字典
        """
        sanitized = {}
        for key, value in headers.items():
            if key.lower() in self.sensitive_fields:
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = value
        return sanitized
    
    def _sanitize_body(self, body: str) -> str:
        """
        对请求体进行脱敏处理
        
        Args:
            body: 原始请求体
            
        Returns:
            str: 脱敏后的请求体
        """
        try:
            # 尝试解析为 JSON
            data = json.loads(body)
            if isinstance(data, dict):
                for key in data:
                    if key.lower() in self.sensitive_fields:
                        data[key] = "***REDACTED***"
                return json.dumps(data, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            pass
        
        # 如果不是 JSON，直接返回（限制长度）
        return body[:1000] if len(body) > 1000 else body
    
    def _should_skip_logging(self, path: str) -> bool:
        """
        判断是否应该跳过日志记录
        
        Args:
            path: 请求路径
            
        Returns:
            bool: 是否跳过
        """
        return path in self.skip_paths
    
    def _format_exception(self, exc: Exception) -> str:
        """
        格式化异常信息
        
        Args:
            exc: 异常对象
            
        Returns:
            str: 格式化后的异常信息
        """
        import traceback
        return traceback.format_exc()
    
    def _log_performance_metrics(self, request_info: Dict[str, Any], 
                                process_time: float, response: Optional[Response]):
        """
        记录性能监控指标
        
        Args:
            request_info: 请求信息
            process_time: 处理时间
            response: 响应对象
        """
        # 只记录超过阈值的慢请求
        slow_request_threshold = 1.0  # 1秒
        
        if process_time > slow_request_threshold:
            self.logger.warning(
                f"慢请求检测",
                extra={
                    "request_id": request_info.get("request_id"),
                    "method": request_info.get("method"),
                    "path": request_info.get("path"),
                    "process_time": process_time,
                    "status_code": response.status_code if response else None,
                    "event": "slow_request"
                }
            )
        
        # 记录错误请求
        if response and response.status_code >= 400:
            self.logger.info(
                f"错误请求",
                extra={
                    "request_id": request_info.get("request_id"),
                    "method": request_info.get("method"),
                    "path": request_info.get("path"),
                    "status_code": response.status_code,
                    "process_time": process_time,
                    "event": "error_request"
                }
            )