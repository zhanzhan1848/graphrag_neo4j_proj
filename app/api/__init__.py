#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 系统 API 模块
====================

本模块是 GraphRAG 系统的 API 层，负责处理所有的 HTTP 请求和响应。

包含的子模块：
- v1: API 版本 1 的实现
- v2: API 版本 2 的实现（未来版本）

API 设计原则：
1. RESTful 设计风格
2. 统一的请求/响应格式
3. 完整的错误处理
4. 详细的日志记录
5. 版本化管理
6. 安全认证和授权

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

# 当前主要使用 v1 版本
from .v1 import api_router

__all__ = [
    "api_router"
]