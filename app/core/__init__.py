#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 系统核心模块
===================

本模块包含 GraphRAG 系统的核心组件和配置：

1. config.py - 应用配置管理
   - 环境变量读取
   - 配置验证
   - 数据库连接配置
   - 服务配置

2. logging.py - 日志系统
   - 多种日志输出方式
   - 日志轮转和归档
   - 结构化日志记录
   - 性能监控日志

3. security.py - 安全相关（待实现）
   - 认证和授权
   - 密码加密
   - JWT 令牌管理
   - API 密钥管理

4. database.py - 数据库连接（待实现）
   - PostgreSQL 连接池
   - Neo4j 连接管理
   - Redis 连接池
   - 数据库迁移

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from .config import settings
from .logging import setup_logging, get_logger

__all__ = [
    "settings",
    "setup_logging", 
    "get_logger"
]