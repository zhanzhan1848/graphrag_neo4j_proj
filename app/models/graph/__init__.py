#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 图数据库模型包
======================

本模块定义了 GraphRAG 系统的图数据库模型（Neo4j）。

模块结构：
- base: 基础图模型定义
- nodes: 图节点模型
- relationships: 图关系模型
- queries: 图查询工具

模型说明：
- BaseNode: 基础节点类
- BaseRelationship: 基础关系类
- EntityNode: 实体节点
- DocumentNode: 文档节点
- ChunkNode: 文本块节点
- ConceptNode: 概念节点
- PersonNode: 人物节点
- OrganizationNode: 组织节点
- LocationNode: 地点节点

关系说明：
- CONTAINS: 包含关系
- MENTIONS: 提及关系
- RELATES_TO: 相关关系
- PART_OF: 部分关系
- SIMILAR_TO: 相似关系

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from .base import (
    BaseNode,
    BaseRelationship,
    GraphModel
)

from .nodes import (
    EntityNode,
    DocumentNode,
    ChunkNode,
    ConceptNode,
    PersonNode,
    OrganizationNode,
    LocationNode,
    EventNode,
    TopicNode
)

from .relationships import (
    ContainsRelationship,
    MentionsRelationship,
    RelatesToRelationship,
    PartOfRelationship,
    SimilarToRelationship,
    LocatedInRelationship,
    WorksForRelationship,
    ParticipatesInRelationship,
    InfluencesRelationship,
    DependsOnRelationship
)

from .queries import (
    GraphQueryBuilder,
    EntityQuery,
    RelationshipQuery,
    PathQuery,
    NeighborQuery,
    SimilarityQuery
)

__all__ = [
    # 基础模型
    "BaseNode",
    "BaseRelationship", 
    "GraphModel",
    
    # 节点模型
    "EntityNode",
    "DocumentNode",
    "ChunkNode",
    "ConceptNode",
    "PersonNode",
    "OrganizationNode",
    "LocationNode",
    "EventNode",
    "TopicNode",
    
    # 关系模型
    "ContainsRelationship",
    "MentionsRelationship",
    "RelatesToRelationship",
    "PartOfRelationship",
    "SimilarToRelationship",
    "LocatedInRelationship",
    "WorksForRelationship",
    "ParticipatesInRelationship",
    "InfluencesRelationship",
    "DependsOnRelationship",
    
    # 查询工具
    "GraphQueryBuilder",
    "EntityQuery",
    "RelationshipQuery",
    "PathQuery",
    "NeighborQuery",
    "SimilarityQuery"
]