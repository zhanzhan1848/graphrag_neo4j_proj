#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 数据库模型包
===================

本包包含 GraphRAG 系统的 PostgreSQL 数据库模型定义，使用 SQLAlchemy ORM。

包含的模型：
- BaseModel - 基础模型类，提供通用字段和方法
- Document - 文档模型
- Chunk - 文本块模型  
- Entity - 实体模型
- Relation - 关系模型
- Image - 图像模型

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from .base import BaseModel
from .documents import Document
from .chunks import Chunk
from .entities import Entity
from .relations import Relation
from .images import Image

__all__ = [
    "BaseModel",
    "Document",
    "Chunk", 
    "Entity",
    "Relation",
    "Image",
]