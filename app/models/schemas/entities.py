#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 实体 API 模式
=====================

本模块定义了实体相关的 API 模式（Pydantic 模型）。

模式说明：
- EntityCreate: 实体创建请求模式
- EntityUpdate: 实体更新请求模式
- EntityResponse: 实体响应模式
- EntityListResponse: 实体列表响应模式
- EntitySearchRequest: 实体搜索请求模式
- EntitySearchResponse: 实体搜索响应模式

字段说明：
- name: 实体名称
- entity_type: 实体类型
- description: 实体描述
- properties: 实体属性
- confidence: 置信度
- source_chunks: 来源文本块

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from datetime import datetime
from enum import Enum
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


class EntityType(str, Enum):
    """
    实体类型枚举
    
    定义系统支持的实体类型。
    """
    PERSON = "PERSON"           # 人物
    ORGANIZATION = "ORG"        # 组织机构
    LOCATION = "LOCATION"       # 地点
    EVENT = "EVENT"             # 事件
    CONCEPT = "CONCEPT"         # 概念
    PRODUCT = "PRODUCT"         # 产品
    TECHNOLOGY = "TECH"         # 技术
    DATE = "DATE"               # 日期
    NUMBER = "NUMBER"           # 数字
    MISC = "MISC"               # 其他


class EntityCreate(BaseSchema):
    """
    实体创建请求模式
    
    用于创建新的实体。
    """
    name: str = Field(
        ...,
        description="实体名称",
        min_length=1,
        max_length=200
    )
    
    entity_type: EntityType = Field(
        ...,
        description="实体类型"
    )
    
    description: Optional[str] = Field(
        None,
        description="实体描述",
        max_length=1000
    )
    
    properties: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="实体的属性信息"
    )
    
    confidence: float = Field(
        default=1.0,
        description="实体识别的置信度",
        ge=0.0,
        le=1.0
    )
    
    source_chunks: Optional[List[UUID]] = Field(
        default_factory=list,
        description="实体来源的文本块ID列表"
    )
    
    aliases: Optional[List[str]] = Field(
        default_factory=list,
        description="实体的别名列表"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="实体的元数据信息"
    )


class EntityUpdate(BaseSchema):
    """
    实体更新请求模式
    
    用于更新现有的实体。
    """
    name: Optional[str] = Field(
        None,
        description="实体名称",
        min_length=1,
        max_length=200
    )
    
    entity_type: Optional[EntityType] = Field(
        None,
        description="实体类型"
    )
    
    description: Optional[str] = Field(
        None,
        description="实体描述",
        max_length=1000
    )
    
    properties: Optional[Dict[str, Any]] = Field(
        None,
        description="实体的属性信息"
    )
    
    confidence: Optional[float] = Field(
        None,
        description="实体识别的置信度",
        ge=0.0,
        le=1.0
    )
    
    aliases: Optional[List[str]] = Field(
        None,
        description="实体的别名列表"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="实体的元数据信息"
    )


class EntityResponse(IDMixin, TimestampMixin, MetadataMixin, BaseSchema):
    """
    实体响应模式
    
    返回实体的完整信息。
    """
    name: str = Field(
        ...,
        description="实体名称"
    )
    
    entity_type: EntityType = Field(
        ...,
        description="实体类型"
    )
    
    description: Optional[str] = Field(
        None,
        description="实体描述"
    )
    
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="实体的属性信息"
    )
    
    confidence: float = Field(
        ...,
        description="实体识别的置信度"
    )
    
    aliases: List[str] = Field(
        default_factory=list,
        description="实体的别名列表"
    )
    
    # 统计信息
    mentions_count: int = Field(
        default=0,
        description="实体在文档中的提及次数",
        ge=0
    )
    
    relations_count: int = Field(
        default=0,
        description="与该实体相关的关系数量",
        ge=0
    )
    
    documents_count: int = Field(
        default=0,
        description="包含该实体的文档数量",
        ge=0
    )
    
    # 关联信息
    source_chunks: List[UUID] = Field(
        default_factory=list,
        description="实体来源的文本块ID列表"
    )
    
    related_entities: Optional[List['EntitySummary']] = Field(
        None,
        description="相关实体的简要信息"
    )


class EntitySummary(BaseSchema):
    """
    实体摘要信息
    
    用于在关联信息中显示实体的简要信息。
    """
    id: UUID = Field(
        ...,
        description="实体唯一标识符"
    )
    
    name: str = Field(
        ...,
        description="实体名称"
    )
    
    entity_type: EntityType = Field(
        ...,
        description="实体类型"
    )
    
    confidence: float = Field(
        ...,
        description="实体识别的置信度"
    )


class EntityListResponse(PaginatedResponse[EntityResponse]):
    """
    实体列表响应模式
    
    返回分页的实体列表。
    """
    pass


class EntitySearchRequest(BaseSchema):
    """
    实体搜索请求模式
    
    用于搜索实体。
    """
    query: str = Field(
        ...,
        description="搜索查询字符串",
        min_length=1,
        max_length=500
    )
    
    entity_types: Optional[List[EntityType]] = Field(
        None,
        description="限制搜索的实体类型列表"
    )
    
    confidence_threshold: float = Field(
        default=0.5,
        description="置信度阈值",
        ge=0.0,
        le=1.0
    )
    
    search_fields: List[str] = Field(
        default=["name", "description", "aliases"],
        description="搜索字段列表"
    )
    
    include_related: bool = Field(
        default=False,
        description="是否包含相关实体信息"
    )
    
    max_results: int = Field(
        default=20,
        description="最大返回结果数",
        ge=1,
        le=100
    )
    
    @validator('search_fields')
    def validate_search_fields(cls, v):
        """验证搜索字段"""
        allowed_fields = {'name', 'description', 'aliases', 'properties'}
        invalid_fields = set(v) - allowed_fields
        if invalid_fields:
            raise ValueError(f'Invalid search fields: {invalid_fields}')
        return v

class EntitySearchResult(BaseSchema):
    """
    实体搜索结果项
    
    单个搜索结果的详细信息。
    """
    entity: EntityResponse = Field(
        ...,
        description="实体信息"
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
    
    match_fields: List[str] = Field(
        default_factory=list,
        description="匹配的字段列表"
    )

class EntitySearchResponse(BaseSchema):
    """
    实体搜索响应模式
    
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
    
    results: List[EntitySearchResult] = Field(
        default_factory=list,
        description="搜索结果列表"
    )
    
    facets: Optional[Dict[str, Any]] = Field(
        None,
        description="搜索结果的分面统计信息"
    )


# 更新前向引用
EntityResponse.model_rebuild()