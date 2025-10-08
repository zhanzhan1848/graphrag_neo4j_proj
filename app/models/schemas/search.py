#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 搜索 API 模式
=====================

本模块定义了搜索相关的 API 模式（Pydantic 模型）。

模式说明：
- SearchRequest: 通用搜索请求模式
- SearchResponse: 通用搜索响应模式
- SemanticSearchRequest: 语义搜索请求模式
- SemanticSearchResponse: 语义搜索响应模式
- HybridSearchRequest: 混合搜索请求模式
- HybridSearchResponse: 混合搜索响应模式

字段说明：
- query: 搜索查询字符串
- search_type: 搜索类型
- filters: 搜索过滤条件
- results: 搜索结果列表
- facets: 分面统计信息

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import Field, validator

from .base import BaseSchema


class SearchType(str, Enum):
    """
    搜索类型枚举
    
    定义系统支持的搜索类型。
    """
    KEYWORD = "keyword"         # 关键词搜索
    SEMANTIC = "semantic"       # 语义搜索
    HYBRID = "hybrid"          # 混合搜索
    GRAPH = "graph"            # 图搜索
    FUZZY = "fuzzy"            # 模糊搜索


class SearchScope(str, Enum):
    """
    搜索范围枚举
    
    定义搜索的数据范围。
    """
    ALL = "all"                # 全部
    DOCUMENTS = "documents"    # 文档
    CHUNKS = "chunks"          # 文本块
    ENTITIES = "entities"      # 实体
    RELATIONS = "relations"    # 关系


class SearchRequest(BaseSchema):
    """
    通用搜索请求模式
    
    用于通用搜索功能。
    """
    query: str = Field(
        ...,
        description="搜索查询字符串",
        min_length=1,
        max_length=1000
    )
    
    search_type: SearchType = Field(
        default=SearchType.HYBRID,
        description="搜索类型"
    )
    
    search_scope: SearchScope = Field(
        default=SearchScope.ALL,
        description="搜索范围"
    )
    
    filters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="搜索过滤条件"
    )
    
    max_results: int = Field(
        default=20,
        description="最大返回结果数",
        ge=1,
        le=100
    )
    
    offset: int = Field(
        default=0,
        description="结果偏移量",
        ge=0
    )
    
    include_highlights: bool = Field(
        default=True,
        description="是否包含高亮信息"
    )
    
    include_facets: bool = Field(
        default=False,
        description="是否包含分面统计信息"
    )


class SearchResult(BaseSchema):
    """
    搜索结果项
    
    单个搜索结果的详细信息。
    """
    id: UUID = Field(
        ...,
        description="结果项的唯一标识符"
    )
    
    type: str = Field(
        ...,
        description="结果类型：document、chunk、entity、relation"
    )
    
    title: str = Field(
        ...,
        description="结果标题"
    )
    
    content: str = Field(
        ...,
        description="结果内容"
    )
    
    score: float = Field(
        ...,
        description="相关性分数",
        ge=0.0,
        le=1.0
    )
    
    highlights: Optional[Dict[str, List[str]]] = Field(
        None,
        description="高亮显示的字段和文本片段"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="结果的元数据信息"
    )


class SearchResponse(BaseSchema):
    """
    通用搜索响应模式
    
    返回搜索结果。
    """
    query: str = Field(
        ...,
        description="原始搜索查询"
    )
    
    search_type: SearchType = Field(
        ...,
        description="搜索类型"
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
    
    results: List[SearchResult] = Field(
        default_factory=list,
        description="搜索结果列表"
    )
    
    facets: Optional[Dict[str, Any]] = Field(
        None,
        description="分面统计信息"
    )
    
    suggestions: Optional[List[str]] = Field(
        None,
        description="搜索建议"
    )


class SemanticSearchRequest(BaseSchema):
    """
    语义搜索请求模式
    
    用于语义搜索功能。
    """
    query: str = Field(
        ...,
        description="搜索查询字符串",
        min_length=1,
        max_length=1000
    )
    
    embedding_model: Optional[str] = Field(
        None,
        description="指定使用的嵌入模型"
    )
    
    similarity_threshold: float = Field(
        default=0.7,
        description="相似度阈值",
        ge=0.0,
        le=1.0
    )
    
    search_scope: SearchScope = Field(
        default=SearchScope.CHUNKS,
        description="搜索范围"
    )
    
    filters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="搜索过滤条件"
    )
    
    max_results: int = Field(
        default=10,
        description="最大返回结果数",
        ge=1,
        le=50
    )
    
    include_embeddings: bool = Field(
        default=False,
        description="是否包含向量嵌入信息"
    )
    
    rerank: bool = Field(
        default=True,
        description="是否对结果进行重排序"
    )


class SemanticSearchResult(BaseSchema):
    """
    语义搜索结果项
    
    单个语义搜索结果的详细信息。
    """
    id: UUID = Field(
        ...,
        description="结果项的唯一标识符"
    )
    
    type: str = Field(
        ...,
        description="结果类型"
    )
    
    content: str = Field(
        ...,
        description="结果内容"
    )
    
    similarity_score: float = Field(
        ...,
        description="语义相似度分数",
        ge=0.0,
        le=1.0
    )
    
    embedding: Optional[List[float]] = Field(
        None,
        description="结果的向量嵌入"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="结果的元数据信息"
    )


class SemanticSearchResponse(BaseSchema):
    """
    语义搜索响应模式
    
    返回语义搜索结果。
    """
    query: str = Field(
        ...,
        description="原始搜索查询"
    )
    
    query_embedding: Optional[List[float]] = Field(
        None,
        description="查询的向量嵌入"
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
    
    results: List[SemanticSearchResult] = Field(
        default_factory=list,
        description="语义搜索结果列表"
    )
    
    similarity_distribution: Optional[Dict[str, float]] = Field(
        None,
        description="相似度分布统计"
    )


class HybridSearchRequest(BaseSchema):
    """
    混合搜索请求模式
    
    结合关键词搜索和语义搜索。
    """
    query: str = Field(
        ...,
        description="搜索查询字符串",
        min_length=1,
        max_length=1000
    )
    
    keyword_weight: float = Field(
        default=0.3,
        description="关键词搜索权重",
        ge=0.0,
        le=1.0
    )
    
    semantic_weight: float = Field(
        default=0.7,
        description="语义搜索权重",
        ge=0.0,
        le=1.0
    )
    
    similarity_threshold: float = Field(
        default=0.6,
        description="语义相似度阈值",
        ge=0.0,
        le=1.0
    )
    
    search_scope: SearchScope = Field(
        default=SearchScope.ALL,
        description="搜索范围"
    )
    
    filters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="搜索过滤条件"
    )
    
    max_results: int = Field(
        default=20,
        description="最大返回结果数",
        ge=1,
        le=100
    )
    
    rerank: bool = Field(
        default=True,
        description="是否对结果进行重排序"
    )
    
    @validator('semantic_weight')
    def validate_weights_sum(cls, v, values):
        """验证权重之和"""
        if 'keyword_weight' in values:
            total_weight = v + values['keyword_weight']
            if abs(total_weight - 1.0) > 0.001:
                raise ValueError('keyword_weight and semantic_weight must sum to 1.0')
        return v


class HybridSearchResult(BaseSchema):
    """
    混合搜索结果项
    
    单个混合搜索结果的详细信息。
    """
    id: UUID = Field(
        ...,
        description="结果项的唯一标识符"
    )
    
    type: str = Field(
        ...,
        description="结果类型"
    )
    
    content: str = Field(
        ...,
        description="结果内容"
    )
    
    hybrid_score: float = Field(
        ...,
        description="混合搜索分数",
        ge=0.0,
        le=1.0
    )
    
    keyword_score: Optional[float] = Field(
        None,
        description="关键词搜索分数",
        ge=0.0,
        le=1.0
    )
    
    semantic_score: Optional[float] = Field(
        None,
        description="语义搜索分数",
        ge=0.0,
        le=1.0
    )
    
    highlights: Optional[Dict[str, List[str]]] = Field(
        None,
        description="高亮显示的字段和文本片段"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="结果的元数据信息"
    )
    
    match_type: str = Field(
        ...,
        description="匹配类型：keyword、semantic、both"
    )


class HybridSearchResponse(BaseSchema):
    """
    混合搜索响应模式
    
    返回混合搜索结果。
    """
    query: str = Field(
        ...,
        description="原始搜索查询"
    )
    
    keyword_weight: float = Field(
        ...,
        description="关键词搜索权重"
    )
    
    semantic_weight: float = Field(
        ...,
        description="语义搜索权重"
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
    
    results: List[HybridSearchResult] = Field(
        default_factory=list,
        description="混合搜索结果列表"
    )
    
    keyword_results_count: int = Field(
        ...,
        description="关键词搜索结果数量",
        ge=0
    )
    
    semantic_results_count: int = Field(
        ...,
        description="语义搜索结果数量",
        ge=0
    )


