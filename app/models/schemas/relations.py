#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 关系 API 模式
=====================

本模块定义了关系相关的 API 模式（Pydantic 模型）。

模式说明：
- RelationCreate: 关系创建请求模式
- RelationUpdate: 关系更新请求模式
- RelationResponse: 关系响应模式
- RelationListResponse: 关系列表响应模式
- RelationSearchRequest: 关系搜索请求模式
- RelationSearchResponse: 关系搜索响应模式

字段说明：
- source_entity_id: 源实体ID
- target_entity_id: 目标实体ID
- relation_type: 关系类型
- description: 关系描述
- properties: 关系属性
- confidence: 置信度
- evidence: 证据信息

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


class RelationType(str, Enum):
    """
    关系类型枚举
    
    定义系统支持的关系类型。
    """
    # 基础关系
    RELATED_TO = "RELATED_TO"           # 相关
    PART_OF = "PART_OF"                 # 属于
    CONTAINS = "CONTAINS"               # 包含
    LOCATED_IN = "LOCATED_IN"           # 位于
    
    # 人物关系
    WORKS_FOR = "WORKS_FOR"             # 工作于
    FOUNDED = "FOUNDED"                 # 创立
    LEADS = "LEADS"                     # 领导
    COLLABORATES_WITH = "COLLABORATES_WITH"  # 合作
    
    # 时间关系
    OCCURS_ON = "OCCURS_ON"             # 发生于
    BEFORE = "BEFORE"                   # 之前
    AFTER = "AFTER"                     # 之后
    DURING = "DURING"                   # 期间
    
    # 因果关系
    CAUSES = "CAUSES"                   # 导致
    RESULTS_IN = "RESULTS_IN"           # 结果是
    INFLUENCES = "INFLUENCES"           # 影响
    
    # 技术关系
    USES = "USES"                       # 使用
    IMPLEMENTS = "IMPLEMENTS"           # 实现
    DEPENDS_ON = "DEPENDS_ON"           # 依赖于
    
    # 其他
    SIMILAR_TO = "SIMILAR_TO"           # 相似于
    OPPOSITE_TO = "OPPOSITE_TO"         # 相反于
    CUSTOM = "CUSTOM"                   # 自定义


class RelationCreate(BaseSchema):
    """
    关系创建请求模式
    
    用于创建新的关系。
    """
    source_entity_id: UUID = Field(
        ...,
        description="源实体的唯一标识符"
    )
    
    target_entity_id: UUID = Field(
        ...,
        description="目标实体的唯一标识符"
    )
    
    relation_type: RelationType = Field(
        ...,
        description="关系类型"
    )
    
    description: Optional[str] = Field(
        None,
        description="关系描述",
        max_length=1000
    )
    
    properties: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="关系的属性信息"
    )
    
    confidence: float = Field(
        default=1.0,
        description="关系识别的置信度",
        ge=0.0,
        le=1.0
    )
    
    evidence: Optional[List[UUID]] = Field(
        default_factory=list,
        description="支持该关系的证据文本块ID列表"
    )
    
    weight: float = Field(
        default=1.0,
        description="关系权重",
        ge=0.0
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="关系的元数据信息"
    )
    
    @validator('target_entity_id')
    def validate_different_entities(cls, v, values):
        """验证源实体和目标实体不能相同"""
        if 'source_entity_id' in values and v == values['source_entity_id']:
            raise ValueError('source_entity_id and target_entity_id must be different')
        return v


class RelationUpdate(BaseSchema):
    """
    关系更新请求模式
    
    用于更新现有的关系。
    """
    relation_type: Optional[RelationType] = Field(
        None,
        description="关系类型"
    )
    
    description: Optional[str] = Field(
        None,
        description="关系描述",
        max_length=1000
    )
    
    properties: Optional[Dict[str, Any]] = Field(
        None,
        description="关系的属性信息"
    )
    
    confidence: Optional[float] = Field(
        None,
        description="关系识别的置信度",
        ge=0.0,
        le=1.0
    )
    
    weight: Optional[float] = Field(
        None,
        description="关系权重",
        ge=0.0
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="关系的元数据信息"
    )


class RelationResponse(IDMixin, TimestampMixin, MetadataMixin, BaseSchema):
    """
    关系响应模式
    
    返回关系的完整信息。
    """
    source_entity_id: UUID = Field(
        ...,
        description="源实体的唯一标识符"
    )
    
    target_entity_id: UUID = Field(
        ...,
        description="目标实体的唯一标识符"
    )
    
    relation_type: RelationType = Field(
        ...,
        description="关系类型"
    )
    
    description: Optional[str] = Field(
        None,
        description="关系描述"
    )
    
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="关系的属性信息"
    )
    
    confidence: float = Field(
        ...,
        description="关系识别的置信度"
    )
    
    weight: float = Field(
        ...,
        description="关系权重"
    )
    
    # 关联实体信息
    source_entity: Optional['EntitySummary'] = Field(
        None,
        description="源实体的简要信息"
    )
    
    target_entity: Optional['EntitySummary'] = Field(
        None,
        description="目标实体的简要信息"
    )
    
    # 证据信息
    evidence: List[UUID] = Field(
        default_factory=list,
        description="支持该关系的证据文本块ID列表"
    )
    
    evidence_count: int = Field(
        default=0,
        description="证据数量",
        ge=0
    )


class RelationListResponse(PaginatedResponse[RelationResponse]):
    """
    关系列表响应模式
    
    返回分页的关系列表。
    """
    pass


class RelationSearchRequest(BaseSchema):
    """
    关系搜索请求模式
    
    用于搜索关系。
    """
    query: Optional[str] = Field(
        None,
        description="搜索查询字符串",
        max_length=500
    )
    
    source_entity_id: Optional[UUID] = Field(
        None,
        description="源实体ID过滤"
    )
    
    target_entity_id: Optional[UUID] = Field(
        None,
        description="目标实体ID过滤"
    )
    
    relation_types: Optional[List[RelationType]] = Field(
        None,
        description="关系类型过滤列表"
    )
    
    confidence_threshold: float = Field(
        default=0.5,
        description="置信度阈值",
        ge=0.0,
        le=1.0
    )
    
    weight_threshold: float = Field(
        default=0.0,
        description="权重阈值",
        ge=0.0
    )
    
    include_entities: bool = Field(
        default=True,
        description="是否包含关联实体信息"
    )
    
    include_evidence: bool = Field(
        default=False,
        description="是否包含证据信息"
    )
    
    max_results: int = Field(
        default=20,
        description="最大返回结果数",
        ge=1,
        le=100
    )

class RelationSearchResult(BaseSchema):
    """
    关系搜索结果项
    
    单个搜索结果的详细信息。
    """
    relation: RelationResponse = Field(
        ...,
        description="关系信息"
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
    
    match_reason: Optional[str] = Field(
        None,
        description="匹配原因说明"
    )

class RelationSearchResponse(BaseSchema):
    """
    关系搜索响应模式
    
    返回搜索结果。
    """
    query: Optional[str] = Field(
        None,
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
    
    results: List[RelationSearchResult] = Field(
        default_factory=list,
        description="搜索结果列表"
    )
    
    facets: Optional[Dict[str, Any]] = Field(
        None,
        description="搜索结果的分面统计信息"
    )


class RelationPath(BaseSchema):
    """
    关系路径模式
    
    用于表示实体间的关系路径。
    """
    start_entity_id: UUID = Field(
        ...,
        description="起始实体ID"
    )
    
    end_entity_id: UUID = Field(
        ...,
        description="结束实体ID"
    )
    
    path_length: int = Field(
        ...,
        description="路径长度（关系数量）",
        ge=1
    )
    
    relations: List[RelationResponse] = Field(
        ...,
        description="路径中的关系列表"
    )
    
    entities: List['EntitySummary'] = Field(
        ...,
        description="路径中的实体列表"
    )
    
    total_weight: float = Field(
        ...,
        description="路径总权重"
    )
    
    confidence: float = Field(
        ...,
        description="路径置信度"
    )


# 导入实体摘要类型（避免循环导入）
from .entities import EntitySummary

# 更新前向引用
RelationResponse.model_rebuild()