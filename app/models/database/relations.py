#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 关系数据库模型
=====================

本模块定义了关系相关的数据库模型。

模型说明：
- Relation: 关系模型，存储实体之间的关系信息
- 支持多种关系类型：工作关系、从属关系、位置关系等
- 包含关系的类型、描述、置信度、证据等信息
- 支持关系的可追溯性，每个关系都能回溯到源文档和文本块
- 与实体、文档、文本块建立关联关系

字段说明：
- relation_type: 关系类型
- description: 关系描述
- properties: 关系属性（JSON格式）
- confidence: 抽取置信度
- evidence_text: 支持证据文本
- source_entity_id: 主体实体ID
- target_entity_id: 客体实体ID

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from sqlalchemy import Column, String, Text, Integer, Float, ForeignKey, Index, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSON
from pgvector.sqlalchemy import Vector

from .base import BaseModel


class Relation(BaseModel):
    """
    关系模型
    
    存储实体之间的关系信息，包括关系类型、描述、置信度和证据。
    每个关系都能追溯到源文档和文本块，确保可解释性。
    """
    __tablename__ = "relations"
    
    # 关联信息
    document_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("documents.id", ondelete="CASCADE"), 
        nullable=False,
        index=True,
        comment="所属文档ID"
    )
    
    evidence_chunk_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("chunks.id", ondelete="CASCADE"),
        index=True,
        comment="证据文本块ID"
    )
    
    # 关系实体
    source_entity_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("entities.id", ondelete="CASCADE"), 
        nullable=False,
        index=True,
        comment="主体实体ID"
    )
    
    target_entity_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("entities.id", ondelete="CASCADE"), 
        nullable=False,
        index=True,
        comment="客体实体ID"
    )
    
    # 关系基本信息
    relation_type = Column(
        String(100), 
        nullable=False,
        index=True,
        comment="关系类型：WORKS_WITH/LOCATED_IN/PART_OF/SIMILAR_TO等"
    )
    
    description = Column(
        Text,
        comment="关系描述"
    )
    
    # 关系属性（JSON格式存储结构化信息）
    properties = Column(
        JSON,
        default=dict,
        comment="关系属性（JSON格式）"
    )
    
    # 抽取信息
    extraction_method = Column(
        String(50),
        comment="抽取方法：RE/LLM/MANUAL等"
    )
    
    confidence = Column(
        Float,
        default=0.0,
        comment="抽取置信度（0.0-1.0）"
    )
    
    # 证据信息
    evidence_text = Column(
        Text,
        comment="支持证据文本"
    )
    
    evidence_start_pos = Column(
        Integer,
        comment="证据在文本块中的起始位置"
    )
    
    evidence_end_pos = Column(
        Integer,
        comment="证据在文本块中的结束位置"
    )
    
    # 关系方向和强度
    is_directed = Column(
        Boolean,
        default=True,
        comment="是否为有向关系"
    )
    
    strength = Column(
        Float,
        comment="关系强度（0.0-1.0）"
    )
    
    # 时间信息
    temporal_info = Column(
        JSON,
        comment="时间信息（开始时间、结束时间等）"
    )
    
    # 向量嵌入
    embedding = Column(
        Vector(1536),
        comment="关系向量嵌入"
    )
    
    embedding_model = Column(
        String(100),
        default="text-embedding-ada-002",
        comment="嵌入模型名称"
    )
    
    # 验证状态
    is_verified = Column(
        Boolean,
        default=False,
        comment="是否已人工验证"
    )
    
    verification_score = Column(
        Float,
        comment="验证评分（0.0-1.0）"
    )
    
    # 重要性评分
    importance_score = Column(
        Float,
        comment="重要性评分（0.0-1.0）"
    )
    
    # 分类和标签
    categories = Column(
        ARRAY(String),
        default=list,
        comment="分类列表"
    )
    
    tags = Column(
        ARRAY(String),
        default=list,
        comment="标签列表"
    )
    
    # 关联关系
    document = relationship(
        "Document", 
        back_populates="relations"
    )
    
    evidence_chunk = relationship(
        "Chunk", 
        back_populates="relations"
    )
    
    source_entity = relationship(
        "Entity", 
        foreign_keys=[source_entity_id],
        back_populates="source_relations"
    )
    
    target_entity = relationship(
        "Entity", 
        foreign_keys=[target_entity_id],
        back_populates="target_relations"
    )
    
    def __repr__(self) -> str:
        """返回关系的字符串表示"""
        return f"<Relation(id={self.id}, type='{self.relation_type}', source={self.source_entity_id}, target={self.target_entity_id})>"
    
    @property
    def relation_triple(self) -> tuple:
        """获取关系三元组（主体，关系，客体）"""
        return (self.source_entity_id, self.relation_type, self.target_entity_id)
    
    @property
    def is_high_confidence(self) -> bool:
        """检查是否为高置信度关系"""
        return self.confidence >= 0.8
    
    @property
    def is_important(self) -> bool:
        """检查是否为重要关系"""
        return self.importance_score and self.importance_score >= 0.7
    
    def update_property(self, key: str, value) -> None:
        """
        更新关系属性
        
        Args:
            key: 属性键
            value: 属性值
        """
        if not self.properties:
            self.properties = {}
        self.properties[key] = value
    
    def get_property(self, key: str, default=None):
        """
        获取关系属性值
        
        Args:
            key: 属性键
            default: 默认值
            
        Returns:
            属性值
        """
        if not self.properties:
            return default
        return self.properties.get(key, default)
    
    def set_temporal_info(self, start_time: str = None, end_time: str = None, duration: str = None) -> None:
        """
        设置时间信息
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            duration: 持续时间
        """
        if not self.temporal_info:
            self.temporal_info = {}
        
        if start_time:
            self.temporal_info["start_time"] = start_time
        if end_time:
            self.temporal_info["end_time"] = end_time
        if duration:
            self.temporal_info["duration"] = duration
    
    def mark_as_verified(self, score: float = 1.0) -> None:
        """
        标记为已验证
        
        Args:
            score: 验证评分
        """
        self.is_verified = True
        self.verification_score = score
    
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
    
    def get_reverse_relation_type(self) -> str:
        """
        获取反向关系类型
        
        Returns:
            str: 反向关系类型
        """
        # 定义一些常见的反向关系映射
        reverse_mapping = {
            "WORKS_WITH": "WORKS_WITH",  # 对称关系
            "LOCATED_IN": "CONTAINS",
            "CONTAINS": "LOCATED_IN",
            "PART_OF": "HAS_PART",
            "HAS_PART": "PART_OF",
            "MANAGES": "MANAGED_BY",
            "MANAGED_BY": "MANAGES",
            "OWNS": "OWNED_BY",
            "OWNED_BY": "OWNS",
            "SIMILAR_TO": "SIMILAR_TO",  # 对称关系
        }
        
        return reverse_mapping.get(self.relation_type, f"REVERSE_OF_{self.relation_type}")


# 创建索引
Index("idx_relations_source_target", Relation.source_entity_id, Relation.target_entity_id)
Index("idx_relations_type_confidence", Relation.relation_type, Relation.confidence)
Index("idx_relations_document_chunk", Relation.document_id, Relation.evidence_chunk_id)
Index("idx_relations_embedding", Relation.embedding, postgresql_using="ivfflat")
Index("idx_relations_properties", Relation.properties, postgresql_using="gin")
Index("idx_relations_importance", Relation.importance_score)
Index("idx_relations_verified", Relation.is_verified, Relation.verification_score)
Index("idx_relations_temporal", Relation.temporal_info, postgresql_using="gin")