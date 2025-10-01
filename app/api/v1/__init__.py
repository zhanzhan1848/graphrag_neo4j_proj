#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 系统 API v1 模块
=======================

本模块是 GraphRAG 系统 API v1 版本的主要模块，包含：
1. 主路由配置 (router.py)
2. 各功能端点实现 (endpoints/)
3. 请求/响应模型 (models/)
4. 业务逻辑依赖 (dependencies/)

API v1 提供的主要功能：
- 系统管理和监控
- 文档上传和处理
- 知识抽取和存储
- 语义搜索和查询
- 图数据库操作
- RAG 问答系统

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from .router import api_router

__all__ = [
    "api_router"
]