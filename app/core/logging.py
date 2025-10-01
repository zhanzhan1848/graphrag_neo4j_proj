#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 系统日志配置模块
=======================

本模块提供完整的日志系统配置，包括：
1. 多种日志格式化器（开发环境彩色输出、生产环境结构化输出）
2. 多种日志处理器（控制台、文件、错误文件、访问日志）
3. 日志轮转和归档
4. 请求上下文过滤器
5. 性能监控日志

支持的日志输出：
- 控制台输出（开发环境彩色）
- 文件输出（按日期轮转）
- 错误日志单独记录
- 访问日志记录
- 结构化日志（JSON格式）

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import logging
import logging.handlers
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import structlog
from colorlog import ColoredFormatter
from pythonjsonlogger import jsonlogger

from app.core.config import settings


class ColoredFormatter(logging.Formatter):
    """
    彩色日志格式化器
    
    为不同级别的日志添加颜色，提升开发环境的可读性
    """
    
    # ANSI 颜色代码
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
        格式化日志记录，添加颜色
        
        Args:
            record: 日志记录对象
            
        Returns:
            str: 格式化后的日志字符串
        """
        # 获取颜色代码
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # 添加颜色到级别名称
        record.levelname = f"{color}{record.levelname}{reset}"
        
        # 格式化消息
        formatted = super().format(record)
        
        return formatted


class StructuredFormatter(jsonlogger.JsonFormatter):
    """
    结构化 JSON 日志格式化器
    
    将日志记录转换为结构化的 JSON 格式，便于日志分析和监控
    """
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]):
        """
        添加自定义字段到日志记录
        
        Args:
            log_record: 日志记录字典
            record: 原始日志记录对象
            message_dict: 消息字典
        """
        super().add_fields(log_record, record, message_dict)
        
        # 添加时间戳
        if not log_record.get('timestamp'):
            log_record['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        
        # 添加日志级别
        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname
        
        # 添加模块信息
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        
        # 添加进程和线程信息
        log_record['process_id'] = record.process
        log_record['thread_id'] = record.thread
        
        # 如果有异常信息，添加异常详情
        if record.exc_info:
            log_record['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }


class RequestContextFilter(logging.Filter):
    """
    请求上下文过滤器
    
    为日志记录添加请求相关的上下文信息，如请求ID、用户ID等
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        过滤日志记录，添加请求上下文
        
        Args:
            record: 日志记录对象
            
        Returns:
            bool: 是否通过过滤
        """
        # 尝试从当前上下文获取请求信息
        # 这里可以集成 contextvars 或其他上下文管理工具
        
        # 添加默认值
        if not hasattr(record, 'request_id'):
            record.request_id = 'N/A'
        
        if not hasattr(record, 'user_id'):
            record.user_id = 'N/A'
        
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = 'N/A'
        
        return True


def setup_console_handler() -> logging.StreamHandler:
    """
    设置控制台日志处理器
    
    Returns:
        logging.StreamHandler: 配置好的控制台处理器
    """
    handler = logging.StreamHandler(sys.stdout)
    
    if settings.ENVIRONMENT == "development":
        # 开发环境使用彩色格式
        formatter = ColoredFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        # 生产环境使用结构化格式
        formatter = StructuredFormatter(
            fmt='%(timestamp)s %(level)s %(name)s %(message)s'
        )
    
    handler.setFormatter(formatter)
    handler.setLevel(settings.LOG_LEVEL)
    
    return handler


def setup_file_handler() -> logging.handlers.RotatingFileHandler:
    """
    设置文件日志处理器
    
    Returns:
        logging.handlers.RotatingFileHandler: 配置好的文件处理器
    """
    # 确保日志目录存在
    log_file_path = Path(settings.LOG_FILE_PATH)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建轮转文件处理器
    handler = logging.handlers.RotatingFileHandler(
        filename=settings.LOG_FILE_PATH,
        maxBytes=settings.LOG_MAX_SIZE,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    
    # 使用结构化格式
    formatter = StructuredFormatter(
        fmt='%(timestamp)s %(level)s %(name)s %(message)s'
    )
    
    handler.setFormatter(formatter)
    handler.setLevel(settings.LOG_LEVEL)
    
    return handler


def setup_error_file_handler() -> logging.handlers.RotatingFileHandler:
    """
    设置错误日志文件处理器
    
    专门记录 ERROR 和 CRITICAL 级别的日志
    
    Returns:
        logging.handlers.RotatingFileHandler: 配置好的错误文件处理器
    """
    # 错误日志文件路径
    error_log_path = Path(settings.LOG_FILE_PATH).parent / "error.log"
    
    handler = logging.handlers.RotatingFileHandler(
        filename=str(error_log_path),
        maxBytes=settings.LOG_MAX_SIZE,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    
    # 只记录 ERROR 和 CRITICAL 级别
    handler.setLevel(logging.ERROR)
    
    # 使用结构化格式
    formatter = StructuredFormatter(
        fmt='%(timestamp)s %(level)s %(name)s %(message)s'
    )
    
    handler.setFormatter(formatter)
    
    return handler


def setup_access_log_handler() -> logging.handlers.RotatingFileHandler:
    """
    设置访问日志处理器
    
    专门记录 HTTP 请求访问日志
    
    Returns:
        logging.handlers.RotatingFileHandler: 配置好的访问日志处理器
    """
    # 访问日志文件路径
    access_log_path = Path(settings.LOG_FILE_PATH).parent / "access.log"
    
    handler = logging.handlers.RotatingFileHandler(
        filename=str(access_log_path),
        maxBytes=settings.LOG_MAX_SIZE,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    
    # 使用结构化格式
    formatter = StructuredFormatter(
        fmt='%(timestamp)s %(level)s %(name)s %(message)s'
    )
    
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    
    return handler


def setup_logging():
    """
    设置整个应用的日志系统
    
    配置根日志记录器和各种处理器
    """
    # 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.LOG_LEVEL)
    
    # 清除现有处理器
    root_logger.handlers.clear()
    
    # 添加请求上下文过滤器
    context_filter = RequestContextFilter()
    root_logger.addFilter(context_filter)
    
    # 添加控制台处理器
    console_handler = setup_console_handler()
    root_logger.addHandler(console_handler)
    
    # 添加文件处理器
    file_handler = setup_file_handler()
    root_logger.addHandler(file_handler)
    
    # 添加错误文件处理器
    error_handler = setup_error_file_handler()
    root_logger.addHandler(error_handler)
    
    # 设置特定模块的日志级别
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    
    # 禁用一些第三方库的详细日志
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # 记录日志系统初始化完成
    logger = logging.getLogger(__name__)
    logger.info("日志系统初始化完成", extra={
        'log_level': settings.LOG_LEVEL,
        'log_file': settings.LOG_FILE_PATH,
        'environment': settings.ENVIRONMENT
    })


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志记录器
    
    Args:
        name: 日志记录器名称，通常使用 __name__
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    return logging.getLogger(name)


def get_access_logger() -> logging.Logger:
    """
    获取访问日志记录器
    
    Returns:
        logging.Logger: 访问日志记录器
    """
    logger = logging.getLogger("access")
    
    # 如果还没有添加访问日志处理器，则添加
    if not any(isinstance(h, logging.handlers.RotatingFileHandler) 
               and "access.log" in str(h.baseFilename) for h in logger.handlers):
        access_handler = setup_access_log_handler()
        logger.addHandler(access_handler)
        logger.setLevel(logging.INFO)
        # 防止日志传播到根记录器，避免重复记录
        logger.propagate = False
    
    return logger


class LoggerMixin:
    """
    日志记录器混入类
    
    为其他类提供便捷的日志记录功能
    """
    
    @property
    def logger(self) -> logging.Logger:
        """获取当前类的日志记录器"""
        return get_logger(self.__class__.__module__ + "." + self.__class__.__name__)
    
    def log_info(self, message: str, **kwargs):
        """记录信息日志"""
        self.logger.info(message, extra=kwargs)
    
    def log_warning(self, message: str, **kwargs):
        """记录警告日志"""
        self.logger.warning(message, extra=kwargs)
    
    def log_error(self, message: str, **kwargs):
        """记录错误日志"""
        self.logger.error(message, extra=kwargs)
    
    def log_debug(self, message: str, **kwargs):
        """记录调试日志"""
        self.logger.debug(message, extra=kwargs)