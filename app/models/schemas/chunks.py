#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 文本块 API 模式
=======================

本模块定义了文本块相关的 API 模式（Pydantic 模型）。

模式说明：
- ChunkCreate: 文本块创建请求模式
- ChunkUpdate: 文本块更新请求模式
- ChunkResponse: 文本块响应模式
- ChunkListResponse: 文本块列表响应模式
- ChunkSearchRequest: 文本块搜索请求模式
- ChunkSearchResponse: 文本块搜索响应模式

字段说明：
- content: 文本块内容
- document_id: 所属文档ID
- chunk_index: 文本块索引
- start_pos/end_pos: 在原文档中的位置
- embedding: 向量嵌入
- metadata: 元数据信息

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import Field, validator

from .base import (
    BaseSchema,
    IDMixin,
    TimestampMixin,
    MetadataMixin,
    PaginatedResponse
)


class ChunkCreate(BaseSchema):
    """
    文本块创建请求模式
    
    用于创建新的文本块。
    """
    content: str = Field(
        ...,
        description="文本块内容",
        min_length=1,
        max_length=10000
    )
    
    document_id: UUID = Field(
        ...,
        description="所属文档的唯一标识符"
    )
    
    chunk_index: int = Field(
        ...,
        description="文本块在文档中的索引位置",
        ge=0
    )
    
    start_pos: int = Field(
        ...,
        description="文本块在原文档中的起始位置",
        ge=0
    )
    
    end_pos: int = Field(
        ...,
        description="文本块在原文档中的结束位置",
        ge=0
    )
    
    embedding: Optional[List[float]] = Field(
        None,
        description="文本块的向量嵌入"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="文本块的元数据信息"
    )
    
    @validator('end_pos')
    def validate_positions(cls, v, values):
        """验证位置参数的有效性"""
        if 'start_pos' in values and v <= values['start_pos']:
            raise ValueError('end_pos must be greater than start_pos')
        return v


class ChunkUpdate(BaseSchema):
    """
    文本块更新请求模式
    
    用于更新现有的文本块。
    """
    content: Optional[str] = Field(
        None,
        description="文本块内容",
        min_length=1,
        max_length=10000
    )
    
    embedding: Optional[List[float]] = Field(
        None,
        description="文本块的向量嵌入"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="文本块的元数据信息"
    )


class ChunkResponse(IDMixin, TimestampMixin, MetadataMixin, BaseSchema):
    """
    文本块响应模式
    
    返回文本块的完整信息。
    """
    content: str = Field(
        ...,
        description="文本块内容"
    )
    
    document_id: UUID = Field(
        ...,
        description="所属文档的唯一标识符"
    )
    
    chunk_index: int = Field(
        ...,
        description="文本块在文档中的索引位置"
    )
    
    start_pos: int = Field(
        ...,
        description="文本块在原文档中的起始位置"
    )
    
    end_pos: int = Field(
        ...,
        description="文本块在原文档中的结束位置"
    )
    
    embedding: Optional[List[float]] = Field(
        None,
        description="文本块的向量嵌入"
    )
    
    # 关联信息
    document_title: Optional[str] = Field(
        None,
        description="所属文档的标题"
    )
    
    entities_count: int = Field(
        default=0,
        description="文本块中包含的实体数量",
        ge=0
    )
    
    relations_count: int = Field(
        default=0,
        description="文本块中包含的关系数量",
        ge=0
    )


class ChunkListResponse(PaginatedResponse[ChunkResponse]):
    """
    文本块列表响应模式
    
    返回分页的文本块列表。
    """
    pass


class ChunkSearchRequest(BaseSchema):
    """
    文本块搜索请求模式
    
    用于搜索文本块。
    """
    query: str = Field(
        ...,
        description="搜索查询字符串",
        min_length=1,
        max_length=500
    )
    
    document_ids: Optional[List[UUID]] = Field(
        None,
        description="限制搜索的文档ID列表"
    )
    
    search_type: str = Field(
        default="semantic",
        description="搜索类型：semantic（语义搜索）、keyword（关键词搜索）、hybrid（混合搜索）"
    )
    
    similarity_threshold: float = Field(
        default=0.7,
        description="相似度阈值",
        ge=0.0,
        le=1.0
    )
    
    max_results: int = Field(
        default=10,
        description="最大返回结果数",
        ge=1,
        le=100
    )
    
    include_embedding: bool = Field(
        default=False,
        description="是否包含向量嵌入信息"
    )
    
    @validator('search_type')
    def validate_search_type(cls, v):
        """验证搜索类型"""
        allowed_types = {'semantic', 'keyword', 'hybrid'}
        if v not in allowed_types:
            raise ValueError(f'search_type must be one of {allowed_types}')
        return v

class ChunkSearchResult(BaseSchema):
    """
    文本块搜索结果项
    
    单个搜索结果的详细信息。
    """
    chunk: ChunkResponse = Field(
        ...,
        description="文本块信息"
    )
    
    score: float = Field(
        ...,
        description="相似度分数",
        ge=0.0,
        le=1.0
    )
    
    highlights: Optional[List[str]] = Field(
        None,
        description="高亮显示的文本片段"
    )
    
    match_type: str = Field(
        ...,
        description="匹配类型：semantic、keyword、hybrid"
    )

class ChunkSearchResponse(BaseSchema):
    """
    文本块搜索响应模式
    
    返回搜索结果。
    """
    query: str = Field(
        ...,
        description="原始搜索查询"
    )
    
    total_results: int = Field(
        ...,
        description="总结果数量",
        ge=0
    )
    
    search_time_ms: float = Field(
        ...,
        description="搜索耗时（毫秒）",
        ge=0
    )
    
    results: List[ChunkSearchResult] = Field(
        default_factory=list,
        description="搜索结果列表"
    )
