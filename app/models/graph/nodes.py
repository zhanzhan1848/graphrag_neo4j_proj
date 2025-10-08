#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 图节点模型
==================

本模块定义了 GraphRAG 系统的图节点模型（Neo4j）。

节点类型说明：
- EntityNode: 实体节点，表示从文档中提取的实体
- DocumentNode: 文档节点，表示原始文档
- ChunkNode: 文本块节点，表示文档的文本片段
- ConceptNode: 概念节点，表示抽象概念
- PersonNode: 人物节点，表示人物实体
- OrganizationNode: 组织节点，表示组织机构
- LocationNode: 地点节点，表示地理位置
- EventNode: 事件节点，表示事件实体
- TopicNode: 主题节点，表示主题分类

字段说明：
- id: 节点唯一标识符
- name: 节点名称
- description: 节点描述
- properties: 节点属性字典
- embedding: 向量嵌入
- confidence: 置信度分数

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from neomodel import (
    StructuredNode,
    StringProperty,
    DateTimeProperty,
    JSONProperty,
    FloatProperty,
    IntegerProperty,
    BooleanProperty,
    UniqueIdProperty,
    ArrayProperty,
    RelationshipTo,
    RelationshipFrom,
    Relationship
)

from .base import BaseNode


class EntityNode(BaseNode):
    """
    实体节点
    
    表示从文档中提取的命名实体，如人物、地点、组织等。
    """
    
    # 基础属性
    name = StringProperty(required=True, index=True)
    entity_type = StringProperty(required=True, index=True)
    description = StringProperty()
    
    # 实体特定属性
    aliases = ArrayProperty(StringProperty(), default=[])
    canonical_name = StringProperty()
    
    # 置信度和统计
    confidence = FloatProperty(default=0.0)
    mention_count = IntegerProperty(default=0)
    document_count = IntegerProperty(default=0)
    
    # 向量嵌入
    embedding = ArrayProperty(FloatProperty(), default=[])
    embedding_model = StringProperty()
    
    # 验证和标准化
    is_verified = BooleanProperty(default=False)
    verification_source = StringProperty()
    
    # 关系定义
    mentioned_in = RelationshipTo('ChunkNode', 'MENTIONED_IN')
    appears_in = RelationshipTo('DocumentNode', 'APPEARS_IN')
    similar_to = RelationshipTo('EntityNode', 'SIMILAR_TO')
    relates_to = RelationshipTo('EntityNode', 'RELATES_TO')
    
    class Meta:
        app_label = 'graphrag'


class DocumentNode(BaseNode):
    """
    文档节点
    
    表示系统中的原始文档。
    """
    
    # 基础属性
    title = StringProperty(required=True, index=True)
    file_name = StringProperty(required=True)
    file_path = StringProperty(required=True)
    
    # 文档元数据
    file_type = StringProperty(required=True)
    file_size = IntegerProperty()
    mime_type = StringProperty()
    
    # 内容属性
    content_hash = StringProperty(unique_index=True)
    language = StringProperty(default='zh')
    encoding = StringProperty(default='utf-8')
    
    # 处理状态
    processing_status = StringProperty(default='pending')
    processed_at = DateTimeProperty()
    
    # 统计信息
    word_count = IntegerProperty(default=0)
    chunk_count = IntegerProperty(default=0)
    entity_count = IntegerProperty(default=0)
    
    # 质量评估
    quality_score = FloatProperty(default=0.0)
    readability_score = FloatProperty(default=0.0)
    
    # 关系定义
    contains = RelationshipTo('ChunkNode', 'CONTAINS')
    mentions = RelationshipTo('EntityNode', 'MENTIONS')
    similar_to = RelationshipTo('DocumentNode', 'SIMILAR_TO')
    
    class Meta:
        app_label = 'graphrag'


class ChunkNode(BaseNode):
    """
    文本块节点
    
    表示文档的文本片段，是文本处理的基本单位。
    """
    
    # 基础属性
    content = StringProperty(required=True)
    chunk_index = IntegerProperty(required=True)
    
    # 位置信息
    start_position = IntegerProperty()
    end_position = IntegerProperty()
    page_number = IntegerProperty()
    
    # 内容属性
    content_hash = StringProperty(unique_index=True)
    word_count = IntegerProperty(default=0)
    char_count = IntegerProperty(default=0)
    
    # 向量嵌入
    embedding = ArrayProperty(FloatProperty(), default=[])
    embedding_model = StringProperty()
    
    # 语义属性
    summary = StringProperty()
    keywords = ArrayProperty(StringProperty(), default=[])
    topics = ArrayProperty(StringProperty(), default=[])
    
    # 质量评估
    quality_score = FloatProperty(default=0.0)
    coherence_score = FloatProperty(default=0.0)
    
    # 关系定义
    part_of = RelationshipTo('DocumentNode', 'PART_OF')
    mentions = RelationshipTo('EntityNode', 'MENTIONS')
    similar_to = RelationshipTo('ChunkNode', 'SIMILAR_TO')
    follows = RelationshipTo('ChunkNode', 'FOLLOWS')
    
    class Meta:
        app_label = 'graphrag'


class ConceptNode(BaseNode):
    """
    概念节点
    
    表示抽象概念或主题。
    """
    
    # 基础属性
    name = StringProperty(required=True, index=True)
    concept_type = StringProperty(required=True, index=True)
    description = StringProperty()
    
    # 概念层次
    level = IntegerProperty(default=0)
    parent_concept = StringProperty()
    
    # 统计信息
    mention_count = IntegerProperty(default=0)
    document_count = IntegerProperty(default=0)
    
    # 向量嵌入
    embedding = ArrayProperty(FloatProperty(), default=[])
    embedding_model = StringProperty()
    
    # 重要性评分
    importance_score = FloatProperty(default=0.0)
    centrality_score = FloatProperty(default=0.0)
    
    # 关系定义
    mentioned_in = RelationshipTo('ChunkNode', 'MENTIONED_IN')
    related_to = RelationshipTo('ConceptNode', 'RELATED_TO')
    part_of = RelationshipTo('ConceptNode', 'PART_OF')
    
    class Meta:
        app_label = 'graphrag'


class PersonNode(EntityNode):
    """
    人物节点
    
    表示人物实体，继承自 EntityNode。
    """
    
    # 人物特定属性
    full_name = StringProperty()
    first_name = StringProperty()
    last_name = StringProperty()
    
    # 职业信息
    occupation = StringProperty()
    title = StringProperty()
    organization = StringProperty()
    
    # 个人信息
    birth_date = StringProperty()
    death_date = StringProperty()
    nationality = StringProperty()
    
    # 联系信息
    email = StringProperty()
    phone = StringProperty()
    
    # 关系定义
    works_for = RelationshipTo('OrganizationNode', 'WORKS_FOR')
    located_in = RelationshipTo('LocationNode', 'LOCATED_IN')
    knows = RelationshipTo('PersonNode', 'KNOWS')
    
    class Meta:
        app_label = 'graphrag'


class OrganizationNode(EntityNode):
    """
    组织节点
    
    表示组织机构实体，继承自 EntityNode。
    """
    
    # 组织特定属性
    full_name = StringProperty()
    short_name = StringProperty()
    organization_type = StringProperty()
    
    # 组织信息
    industry = StringProperty()
    founded_date = StringProperty()
    headquarters = StringProperty()
    
    # 规模信息
    employee_count = IntegerProperty()
    revenue = StringProperty()
    
    # 联系信息
    website = StringProperty()
    email = StringProperty()
    phone = StringProperty()
    
    # 关系定义
    located_in = RelationshipTo('LocationNode', 'LOCATED_IN')
    part_of = RelationshipTo('OrganizationNode', 'PART_OF')
    competes_with = RelationshipTo('OrganizationNode', 'COMPETES_WITH')
    
    class Meta:
        app_label = 'graphrag'


class LocationNode(EntityNode):
    """
    地点节点
    
    表示地理位置实体，继承自 EntityNode。
    """
    
    # 地点特定属性
    full_name = StringProperty()
    location_type = StringProperty()  # city, country, region, etc.
    
    # 地理信息
    latitude = FloatProperty()
    longitude = FloatProperty()
    altitude = FloatProperty()
    
    # 行政信息
    country = StringProperty()
    state_province = StringProperty()
    city = StringProperty()
    postal_code = StringProperty()
    
    # 统计信息
    population = IntegerProperty()
    area = FloatProperty()
    
    # 关系定义
    contains = RelationshipTo('LocationNode', 'CONTAINS')
    part_of = RelationshipTo('LocationNode', 'PART_OF')
    near = RelationshipTo('LocationNode', 'NEAR')
    
    class Meta:
        app_label = 'graphrag'


class EventNode(EntityNode):
    """
    事件节点
    
    表示事件实体，继承自 EntityNode。
    """
    
    # 事件特定属性
    event_type = StringProperty(required=True)
    title = StringProperty()
    
    # 时间信息
    start_date = StringProperty()
    end_date = StringProperty()
    duration = StringProperty()
    
    # 地点信息
    location = StringProperty()
    venue = StringProperty()
    
    # 参与者信息
    organizer = StringProperty()
    participants = ArrayProperty(StringProperty(), default=[])
    
    # 影响评估
    impact_score = FloatProperty(default=0.0)
    significance = StringProperty()
    
    # 关系定义
    occurs_at = RelationshipTo('LocationNode', 'OCCURS_AT')
    involves = RelationshipTo('PersonNode', 'INVOLVES')
    organized_by = RelationshipTo('OrganizationNode', 'ORGANIZED_BY')
    
    class Meta:
        app_label = 'graphrag'


class TopicNode(BaseNode):
    """
    主题节点
    
    表示主题分类或标签。
    """
    
    # 基础属性
    name = StringProperty(required=True, index=True)
    topic_type = StringProperty(required=True)
    description = StringProperty()
    
    # 层次信息
    level = IntegerProperty(default=0)
    parent_topic = StringProperty()
    
    # 统计信息
    document_count = IntegerProperty(default=0)
    chunk_count = IntegerProperty(default=0)
    
    # 向量嵌入
    embedding = ArrayProperty(FloatProperty(), default=[])
    embedding_model = StringProperty()
    
    # 重要性评分
    importance_score = FloatProperty(default=0.0)
    popularity_score = FloatProperty(default=0.0)
    
    # 关系定义
    contains = RelationshipTo('TopicNode', 'CONTAINS')
    related_to = RelationshipTo('TopicNode', 'RELATED_TO')
    applied_to = RelationshipTo('DocumentNode', 'APPLIED_TO')
    
    class Meta:
        app_label = 'graphrag'