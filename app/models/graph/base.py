#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 图数据库基础模型
========================

本模块定义了 GraphRAG 系统的图数据库基础模型。

模型说明：
- BaseNode: 基础节点类，所有图节点的基类
- BaseRelationship: 基础关系类，所有图关系的基类
- GraphModel: 图模型基类，提供通用的图操作方法

字段说明：
- id: 节点/关系的唯一标识符
- created_at/updated_at: 创建和更新时间
- properties: 节点/关系的属性字典
- labels: 节点标签列表
- type: 关系类型

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

from neomodel import (
    StructuredNode,
    StructuredRel,
    StringProperty,
    DateTimeProperty,
    JSONProperty,
    FloatProperty,
    IntegerProperty,
    BooleanProperty,
    UniqueIdProperty,
    ArrayProperty
)


class GraphModel:
    """
    图模型基类
    
    提供通用的图操作方法和属性。
    """
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            字典表示
        """
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, datetime):
                    result[key] = value.isoformat()
                elif isinstance(value, UUID):
                    result[key] = str(value)
                else:
                    result[key] = value
        return result
    
    def update_properties(self, properties: Dict[str, Any]) -> None:
        """
        更新属性
        
        Args:
            properties: 属性字典
        """
        for key, value in properties.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @classmethod
    def get_labels(cls) -> List[str]:
        """
        获取节点标签
        
        Returns:
            标签列表
        """
        if hasattr(cls, '__label__'):
            return [cls.__label__]
        return [cls.__name__]
    
    @classmethod
    def get_relationship_type(cls) -> str:
        """
        获取关系类型
        
        Returns:
            关系类型字符串
        """
        if hasattr(cls, '__type__'):
            return cls.__type__
        return cls.__name__.upper()


class BaseNode(StructuredNode, GraphModel):
    """
    基础节点类
    
    所有图节点的基类，包含通用字段和方法。
    """
    
    # 唯一标识符
    uid = UniqueIdProperty()
    
    # 外部ID（对应关系数据库中的ID）
    external_id = StringProperty(
        unique_index=True,
        required=False
    )
    
    # 基本属性
    name = StringProperty(
        required=True,
        index=True
    )
    
    display_name = StringProperty(
        required=False
    )
    
    description = StringProperty(
        required=False
    )
    
    # 时间戳
    created_at = DateTimeProperty(
        default_now=True,
        index=True
    )
    
    updated_at = DateTimeProperty(
        default_now=True
    )
    
    # 元数据
    metadata = JSONProperty(
        default=dict
    )
    
    # 属性字典
    properties = JSONProperty(
        default=dict
    )
    
    # 标签和分类
    labels = ArrayProperty(
        StringProperty(),
        default=list
    )
    
    categories = ArrayProperty(
        StringProperty(),
        default=list
    )
    
    tags = ArrayProperty(
        StringProperty(),
        default=list
    )
    
    # 统计信息
    degree = IntegerProperty(
        default=0,
        index=True
    )
    
    in_degree = IntegerProperty(
        default=0
    )
    
    out_degree = IntegerProperty(
        default=0
    )
    
    # 重要性评分
    importance_score = FloatProperty(
        default=0.0,
        index=True
    )
    
    centrality_score = FloatProperty(
        default=0.0
    )
    
    # 状态
    is_active = BooleanProperty(
        default=True,
        index=True
    )
    
    is_verified = BooleanProperty(
        default=False
    )
    
    def __str__(self) -> str:
        """返回节点的字符串表示"""
        return f"<{self.__class__.__name__}(uid={self.uid}, name='{self.name}')>"
    
    def __repr__(self) -> str:
        """返回节点的详细表示"""
        return self.__str__()
    
    def save(self):
        """保存节点时更新时间戳"""
        self.updated_at = datetime.utcnow()
        return super().save()
    
    def add_label(self, label: str) -> None:
        """
        添加标签
        
        Args:
            label: 标签名称
        """
        if not self.labels:
            self.labels = []
        if label not in self.labels:
            self.labels.append(label)
    
    def remove_label(self, label: str) -> None:
        """
        移除标签
        
        Args:
            label: 标签名称
        """
        if self.labels and label in self.labels:
            self.labels.remove(label)
    
    def add_tag(self, tag: str) -> None:
        """
        添加标签
        
        Args:
            tag: 标签名称
        """
        if not self.tags:
            self.tags = []
        if tag not in self.tags:
            self.tags.append(tag)
    
    def set_property(self, key: str, value: Any) -> None:
        """
        设置属性
        
        Args:
            key: 属性键
            value: 属性值
        """
        if not self.properties:
            self.properties = {}
        self.properties[key] = value
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """
        获取属性
        
        Args:
            key: 属性键
            default: 默认值
            
        Returns:
            属性值
        """
        if not self.properties:
            return default
        return self.properties.get(key, default)
    
    def update_scores(self, importance: float = None, centrality: float = None) -> None:
        """
        更新评分
        
        Args:
            importance: 重要性评分
            centrality: 中心性评分
        """
        if importance is not None:
            self.importance_score = max(0.0, min(1.0, importance))
        if centrality is not None:
            self.centrality_score = max(0.0, min(1.0, centrality))
    
    def update_degree_stats(self) -> None:
        """更新度数统计"""
        # 这里需要实际查询图数据库来计算度数
        # 暂时使用占位符实现
        pass
    
    @classmethod
    def find_by_name(cls, name: str):
        """
        根据名称查找节点
        
        Args:
            name: 节点名称
            
        Returns:
            节点实例或None
        """
        try:
            return cls.nodes.get(name=name)
        except cls.DoesNotExist:
            return None
    
    @classmethod
    def find_by_external_id(cls, external_id: str):
        """
        根据外部ID查找节点
        
        Args:
            external_id: 外部ID
            
        Returns:
            节点实例或None
        """
        try:
            return cls.nodes.get(external_id=external_id)
        except cls.DoesNotExist:
            return None


class BaseRelationship(StructuredRel, GraphModel):
    """
    基础关系类
    
    所有图关系的基类，包含通用字段和方法。
    """
    
    # 唯一标识符
    uid = UniqueIdProperty()
    
    # 外部ID（对应关系数据库中的ID）
    external_id = StringProperty(
        required=False
    )
    
    # 基本属性
    name = StringProperty(
        required=False
    )
    
    description = StringProperty(
        required=False
    )
    
    # 时间戳
    created_at = DateTimeProperty(
        default_now=True,
        index=True
    )
    
    updated_at = DateTimeProperty(
        default_now=True
    )
    
    # 元数据
    metadata = JSONProperty(
        default=dict
    )
    
    # 属性字典
    properties = JSONProperty(
        default=dict
    )
    
    # 权重和评分
    weight = FloatProperty(
        default=1.0,
        index=True
    )
    
    confidence = FloatProperty(
        default=0.0,
        index=True
    )
    
    strength = FloatProperty(
        default=0.0
    )
    
    # 证据信息
    evidence_text = StringProperty(
        required=False
    )
    
    evidence_count = IntegerProperty(
        default=1
    )
    
    # 来源信息
    source_document_id = StringProperty(
        required=False,
        index=True
    )
    
    source_chunk_id = StringProperty(
        required=False,
        index=True
    )
    
    # 状态
    is_active = BooleanProperty(
        default=True,
        index=True
    )
    
    is_verified = BooleanProperty(
        default=False
    )
    
    def __str__(self) -> str:
        """返回关系的字符串表示"""
        return f"<{self.__class__.__name__}(uid={self.uid}, weight={self.weight})>"
    
    def __repr__(self) -> str:
        """返回关系的详细表示"""
        return self.__str__()
    
    def save(self):
        """保存关系时更新时间戳"""
        self.updated_at = datetime.utcnow()
        return super().save()
    
    def set_property(self, key: str, value: Any) -> None:
        """
        设置属性
        
        Args:
            key: 属性键
            value: 属性值
        """
        if not self.properties:
            self.properties = {}
        self.properties[key] = value
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """
        获取属性
        
        Args:
            key: 属性键
            default: 默认值
            
        Returns:
            属性值
        """
        if not self.properties:
            return default
        return self.properties.get(key, default)
    
    def update_confidence(self, confidence: float) -> None:
        """
        更新置信度
        
        Args:
            confidence: 置信度值（0.0-1.0）
        """
        self.confidence = max(0.0, min(1.0, confidence))
    
    def add_evidence(self, text: str, document_id: str = None, chunk_id: str = None) -> None:
        """
        添加证据
        
        Args:
            text: 证据文本
            document_id: 文档ID
            chunk_id: 文本块ID
        """
        if text:
            self.evidence_text = text
        if document_id:
            self.source_document_id = document_id
        if chunk_id:
            self.source_chunk_id = chunk_id
        self.evidence_count += 1
    
    def update_weight(self, weight: float) -> None:
        """
        更新权重
        
        Args:
            weight: 权重值
        """
        self.weight = max(0.0, weight)
    
    def calculate_strength(self) -> float:
        """
        计算关系强度
        
        Returns:
            关系强度值
        """
        # 基于权重、置信度和证据数量计算强度
        strength = (self.weight * self.confidence * 
                   min(1.0, self.evidence_count / 10.0))
        self.strength = strength
        return strength
    
    @classmethod
    def get_type(cls) -> str:
        """
        获取关系类型
        
        Returns:
            关系类型字符串
        """
        return cls.get_relationship_type()


# 图数据库配置
class GraphConfig:
    """
    图数据库配置类
    
    定义图数据库的连接和操作配置。
    """
    
    # 默认批处理大小
    DEFAULT_BATCH_SIZE = 1000
    
    # 默认查询限制
    DEFAULT_QUERY_LIMIT = 100
    
    # 默认相似度阈值
    DEFAULT_SIMILARITY_THRESHOLD = 0.7
    
    # 默认重要性阈值
    DEFAULT_IMPORTANCE_THRESHOLD = 0.5
    
    # 节点类型映射
    NODE_TYPE_MAPPING = {
        'entity': 'EntityNode',
        'document': 'DocumentNode',
        'chunk': 'ChunkNode',
        'concept': 'ConceptNode',
        'person': 'PersonNode',
        'organization': 'OrganizationNode',
        'location': 'LocationNode',
        'event': 'EventNode',
        'topic': 'TopicNode'
    }
    
    # 关系类型映射
    RELATIONSHIP_TYPE_MAPPING = {
        'contains': 'CONTAINS',
        'mentions': 'MENTIONS',
        'relates_to': 'RELATES_TO',
        'part_of': 'PART_OF',
        'similar_to': 'SIMILAR_TO',
        'located_in': 'LOCATED_IN',
        'works_for': 'WORKS_FOR',
        'participates_in': 'PARTICIPATES_IN',
        'influences': 'INFLUENCES',
        'depends_on': 'DEPENDS_ON'
    }