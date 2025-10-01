#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 知识库系统
==================

GraphRAG 是一个基于图数据库和向量检索的智能知识管理平台，提供：

核心功能：
1. 文档处理 - 支持多种格式文档的解析和处理
2. 知识抽取 - 自动从文档中抽取实体、关系和断言
3. 图谱构建 - 基于 Neo4j 的知识图谱存储和管理
4. 语义搜索 - 基于 Weaviate 的向量化语义检索
5. RAG 问答 - 检索增强生成的智能问答系统
6. 溯源追踪 - 完整的知识来源和证据链追踪

技术架构：
- FastAPI - 高性能 Web 框架
- PostgreSQL - 关系数据库存储
- Neo4j - 图数据库存储
- Weaviate - 向量数据库
- Redis - 缓存和队列
- MinIO - 对象存储

模块结构：
- api/ - API 接口层
- core/ - 核心配置和工具
- middleware/ - 中间件组件
- services/ - 业务逻辑服务
- models/ - 数据模型定义
- utils/ - 工具函数

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
许可证: MIT
"""

__version__ = "1.0.0"
__author__ = "GraphRAG Team"
__email__ = "team@graphrag.com"
__description__ = "GraphRAG 知识库系统 - 基于图数据库和向量检索的智能知识管理平台"

# 应用元数据
__app_name__ = "GraphRAG Knowledge Base"
__app_version__ = __version__
__api_version__ = "v1"