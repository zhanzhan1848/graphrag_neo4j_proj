#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型单元测试包
============

测试 GraphRAG 系统的数据模型。

测试模块：
- test_database: 数据库模型测试
  - test_base: 基础模型测试
  - test_documents: 文档模型测试
  - test_chunks: 文本块模型测试
  - test_entities: 实体模型测试
  - test_relations: 关系模型测试
  - test_images: 图像模型测试

- test_schemas: API 模式测试
  - test_base: 基础模式测试
  - test_documents: 文档模式测试
  - test_chunks: 文本块模式测试
  - test_entities: 实体模式测试
  - test_relations: 关系模式测试
  - test_graph: 图模式测试
  - test_search: 搜索模式测试

- test_graph: 图模型测试
  - test_base: 图基础模型测试
  - test_nodes: 图节点测试
  - test_relationships: 图关系测试
  - test_queries: 图查询测试

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

__all__ = [
    "test_database",
    "test_schemas", 
    "test_graph"
]