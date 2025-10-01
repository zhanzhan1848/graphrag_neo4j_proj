#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 系统 API v1 端点模块
===========================

本模块包含 GraphRAG 系统 API v1 版本的所有端点实现：

1. system - 系统管理端点
   - 健康检查
   - 系统状态
   - 服务依赖检查
   - 系统信息

2. documents - 文档管理端点（待实现）
   - 文档上传
   - 文档处理
   - 文档查询
   - 文档删除

3. knowledge - 知识查询端点（待实现）
   - 语义搜索
   - 图查询
   - RAG 问答

4. entities - 实体管理端点（待实现）
   - 实体查询
   - 实体创建
   - 实体更新
   - 实体删除

5. relations - 关系管理端点（待实现）
   - 关系查询
   - 关系创建
   - 关系更新
   - 关系删除

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

# 导入已实现的端点模块
from . import system

# 定义可导出的模块
__all__ = [
    "system"
]