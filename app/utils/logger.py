#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 日志工具
===============

本模块提供了统一的日志记录功能。

日志功能：
- 结构化日志记录
- 多级别日志支持
- 文件和控制台输出
- 日志轮转和清理
- 性能监控日志
- 错误追踪日志

日志配置：
- 可配置的日志级别
- 可配置的输出格式
- 可配置的输出目标
- 可配置的日志轮转

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import os
import sys
import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
from contextvars import ContextVar

from app.core.config import settings

# 上下文变量用于追踪请求
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)


class JSONFormatter(logging.Formatter):
    """
    JSON 格式化器
    
    将日志记录格式化为 JSON 格式。
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录
        
        Args:
            record: 日志记录
            
        Returns:
            格式化后的 JSON 字符串
        """
        # 基础日志信息
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
            'process': record.process,
        }
        
        # 添加上下文信息
        request_id = request_id_var.get()
        if request_id:
            log_entry['request_id'] = request_id
        
        user_id = user_id_var.get()
        if user_id:
            log_entry['user_id'] = user_id
        
        # 添加异常信息
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # 添加额外字段
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # 添加性能信息
        if hasattr(record, 'duration'):
            log_entry['duration'] = record.duration
        
        if hasattr(record, 'status_code'):
            log_entry['status_code'] = record.status_code
        
        return json.dumps(log_entry, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """
    彩色格式化器
    
    为控制台输出添加颜色。
    """
    
    # 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m'        # 重置
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录并添加颜色
        
        Args:
            record: 日志记录
            
        Returns:
            格式化后的彩色字符串
        """
        # 获取颜色
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # 格式化时间
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # 构建日志消息
        message = f"{color}[{timestamp}] {record.levelname:8} {record.name:20} | {record.getMessage()}{reset}"
        
        # 添加上下文信息
        context_parts = []
        request_id = request_id_var.get()
        if request_id:
            context_parts.append(f"req:{request_id[:8]}")
        
        user_id = user_id_var.get()
        if user_id:
            context_parts.append(f"user:{user_id}")
        
        if context_parts:
            message += f" [{', '.join(context_parts)}]"
        
        # 添加位置信息
        message += f" ({record.filename}:{record.lineno})"
        
        # 添加异常信息
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"
        
        return message


class GraphRAGLogger:
    """
    GraphRAG 日志记录器
    
    提供统一的日志记录接口。
    """
    
    _loggers: Dict[str, logging.Logger] = {}
    _configured = False
    
    @classmethod
    def configure(cls) -> None:
        """
        配置日志系统
        """
        if cls._configured:
            return
        
        # 创建日志目录
        log_dir = Path(settings.LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
        
        # 清除现有处理器
        root_logger.handlers.clear()
        
        # 控制台处理器
        if True:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
            
            if settings.LOG_FORMAT == 'json':
                console_handler.setFormatter(JSONFormatter())
            else:
                console_handler.setFormatter(ColoredFormatter())
            
            root_logger.addHandler(console_handler)
        
        # 文件处理器
        if True:
            # 应用日志文件
            app_log_file = log_dir / 'app.log'
            app_handler = logging.handlers.RotatingFileHandler(
                app_log_file,
                maxBytes=settings.LOG_MAX_SIZE,
                backupCount=settings.LOG_BACKUP_COUNT,
                encoding='utf-8'
            )
            app_handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
            app_handler.setFormatter(JSONFormatter())
            root_logger.addHandler(app_handler)
            
            # 错误日志文件
            error_log_file = log_dir / 'error.log'
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=settings.LOG_MAX_SIZE,
                backupCount=settings.LOG_BACKUP_COUNT,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(JSONFormatter())
            root_logger.addHandler(error_handler)
        
        # 设置第三方库日志级别
        logging.getLogger('uvicorn').setLevel(logging.INFO)
        logging.getLogger('fastapi').setLevel(logging.INFO)
        logging.getLogger('sqlalchemy').setLevel(logging.WARNING)
        logging.getLogger('neo4j').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)
        
        cls._configured = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        获取日志记录器
        
        Args:
            name: 日志记录器名称
            
        Returns:
            日志记录器实例
        """
        if not cls._configured:
            cls.configure()
        
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
        
        return cls._loggers[name]


class LoggerAdapter(logging.LoggerAdapter):
    """
    日志适配器
    
    为日志记录添加额外的上下文信息。
    """
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        处理日志消息
        
        Args:
            msg: 日志消息
            kwargs: 关键字参数
            
        Returns:
            处理后的消息和参数
        """
        # 添加额外字段
        extra = kwargs.get('extra', {})
        
        # 添加上下文信息
        request_id = request_id_var.get()
        if request_id:
            extra['request_id'] = request_id
        
        user_id = user_id_var.get()
        if user_id:
            extra['user_id'] = user_id
        
        # 添加适配器的额外信息
        if self.extra:
            extra.update(self.extra)
        
        if extra:
            kwargs['extra'] = {'extra_fields': extra}
        
        return msg, kwargs


def get_logger(name: str, **extra_fields) -> Union[logging.Logger, LoggerAdapter]:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
        **extra_fields: 额外字段
        
    Returns:
        日志记录器或适配器
    """
    logger = GraphRAGLogger.get_logger(name)
    
    if extra_fields:
        return LoggerAdapter(logger, extra_fields)
    
    return logger


def set_request_context(request_id: str, user_id: Optional[str] = None) -> None:
    """
    设置请求上下文
    
    Args:
        request_id: 请求ID
        user_id: 用户ID
    """
    request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)


def clear_request_context() -> None:
    """
    清除请求上下文
    """
    request_id_var.set(None)
    user_id_var.set(None)


def log_performance(
    logger: logging.Logger,
    operation: str,
    duration: float,
    status: str = "success",
    **extra_fields
) -> None:
    """
    记录性能日志
    
    Args:
        logger: 日志记录器
        operation: 操作名称
        duration: 持续时间（秒）
        status: 状态
        **extra_fields: 额外字段
    """
    extra = {
        'operation': operation,
        'duration': duration,
        'status': status,
        **extra_fields
    }
    
    logger.info(
        f"Performance: {operation} completed in {duration:.3f}s with status {status}",
        extra={'extra_fields': extra}
    )


def log_api_request(
    logger: logging.Logger,
    method: str,
    path: str,
    status_code: int,
    duration: float,
    **extra_fields
) -> None:
    """
    记录 API 请求日志
    
    Args:
        logger: 日志记录器
        method: HTTP 方法
        path: 请求路径
        status_code: 状态码
        duration: 持续时间（秒）
        **extra_fields: 额外字段
    """
    extra = {
        'method': method,
        'path': path,
        'status_code': status_code,
        'duration': duration,
        **extra_fields
    }
    
    level = logging.INFO
    if status_code >= 400:
        level = logging.WARNING
    if status_code >= 500:
        level = logging.ERROR
    
    logger.log(
        level,
        f"API: {method} {path} -> {status_code} ({duration:.3f}s)",
        extra={'extra_fields': extra}
    )


def log_database_query(
    logger: logging.Logger,
    query_type: str,
    table: str,
    duration: float,
    rows_affected: Optional[int] = None,
    **extra_fields
) -> None:
    """
    记录数据库查询日志
    
    Args:
        logger: 日志记录器
        query_type: 查询类型
        table: 表名
        duration: 持续时间（秒）
        rows_affected: 影响行数
        **extra_fields: 额外字段
    """
    extra = {
        'query_type': query_type,
        'table': table,
        'duration': duration,
        **extra_fields
    }
    
    if rows_affected is not None:
        extra['rows_affected'] = rows_affected
    
    logger.debug(
        f"DB: {query_type} on {table} ({duration:.3f}s, {rows_affected or 0} rows)",
        extra={'extra_fields': extra}
    )


# 初始化日志系统
GraphRAGLogger.configure()