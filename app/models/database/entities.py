#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 实体数据库模型
=====================

本模块定义了实体相关的数据库模型。

模型说明：
- Entity: 实体模型，存储从文档中抽取的实体信息
- 支持多种实体类型：人物、组织、地点、概念等
- 包含实体的标准化名称、显示名称、类型、属性等
- 支持实体链接和消歧
- 与文档、文本块、关系建立关联关系

字段说明：
- canonical_name: 标准化名称（用于去重和链接）
- display_name: 显示名称
- entity_type: 实体类型
- description: 实体描述
- properties: 实体属性（JSON格式）
- confidence: 抽取置信度
- embedding: 向量嵌入

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from sqlalchemy import Column, String, Text, Integer, Float, ForeignKey, Index, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSON
from pgvector.sqlalchemy import Vector

from .base import BaseModel


class Entity(BaseModel):
    """
    实体模型
    
    存储从文档中抽取的实体信息，包括人物、组织、地点、概念等。
    支持实体链接、消歧和语义检索。
    """
    __tablename__ = "entities"
    
    # 关联信息
    document_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("documents.id", ondelete="CASCADE"), 
        nullable=False,
        index=True,
        comment="所属文档ID"
    )
    
    chunk_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("chunks.id", ondelete="CASCADE"),
        index=True,
        comment="所属文本块ID"
    )
    
    # 实体基本信息
    canonical_name = Column(
        String(200), 
        nullable=False,
        index=True,
        comment="标准化名称（用于去重和链接）"
    )
    
    display_name = Column(
        String(200), 
        nullable=False,
        comment="显示名称"
    )
    
    entity_type = Column(
        String(50), 
        nullable=False,
        index=True,
        comment="实体类型：PERSON/ORGANIZATION/LOCATION/CONCEPT/EVENT等"
    )
    
    description = Column(
        Text,
        comment="实体描述"
    )
    
    # 实体属性（JSON格式存储结构化信息）
    properties = Column(
        JSON,
        default=dict,
        comment="实体属性（JSON格式）"
    )
    
    # 别名和同义词
    aliases = Column(
        ARRAY(String),
        default=list,
        comment="别名列表"
    )
    
    synonyms = Column(
        ARRAY(String),
        default=list,
        comment="同义词列表"
    )
    
    # 抽取信息
    extraction_method = Column(
        String(50),
        comment="抽取方法：NER/LLM/MANUAL等"
    )
    
    confidence = Column(
        Float,
        default=0.0,
        comment="抽取置信度（0.0-1.0）"
    )
    
    # 位置信息
    start_pos = Column(
        Integer,
        comment="在文本块中的起始位置"
    )
    
    end_pos = Column(
        Integer,
        comment="在文本块中的结束位置"
    )
    
    # 向量嵌入
    embedding = Column(
        Vector(1536),
        comment="实体向量嵌入"
    )
    
    embedding_model = Column(
        String(100),
        default="text-embedding-ada-002",
        comment="嵌入模型名称"
    )
    
    # 统计信息
    mention_count = Column(
        Integer,
        default=1,
        comment="在文档中的提及次数"
    )
    
    importance_score = Column(
        Float,
        comment="重要性评分（0.0-1.0）"
    )
    
    # 验证状态
    is_verified = Column(
        Boolean,
        default=False,
        comment="是否已人工验证"
    )
    
    is_linked = Column(
        Boolean,
        default=False,
        comment="是否已链接到知识库"
    )
    
    # 外部链接
    external_id = Column(
        String(100),
        comment="外部知识库ID（如Wikidata ID）"
    )
    
    external_url = Column(
        String(500),
        comment="外部链接URL"
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
        back_populates="entities"
    )
    
    chunk = relationship(
        "Chunk", 
        back_populates="entities"
    )
    
    # 作为主体的关系
    source_relations = relationship(
        "Relation", 
        foreign_keys="Relation.source_entity_id",
        back_populates="source_entity",
        lazy="dynamic"
    )
    
    # 作为客体的关系
    target_relations = relationship(
        "Relation", 
        foreign_keys="Relation.target_entity_id",
        back_populates="target_entity",
        lazy="dynamic"
    )
    
    def __repr__(self) -> str:
        """返回实体的字符串表示"""
        return f"<Entity(id={self.id}, name='{self.canonical_name}', type='{self.entity_type}')>"
    
    @property
    def all_relations(self):
        """获取所有相关的关系（作为主体或客体）"""
        return list(self.source_relations) + list(self.target_relations)
    
    @property
    def relation_count(self) -> int:
        """获取关系总数"""
        return self.source_relations.count() + self.target_relations.count()
    
    def add_alias(self, alias: str) -> None:
        """
        添加别名
        
        Args:
            alias: 别名
        """
        if not self.aliases:
            self.aliases = []
        if alias not in self.aliases:
            self.aliases.append(alias)
    
    def add_synonym(self, synonym: str) -> None:
        """
        添加同义词
        
        Args:
            synonym: 同义词
        """
        if not self.synonyms:
            self.synonyms = []
        if synonym not in self.synonyms:
            self.synonyms.append(synonym)
    
    def update_property(self, key: str, value) -> None:
        """
        更新属性
        
        Args:
            key: 属性键
            value: 属性值
        """
        if not self.properties:
            self.properties = {}
        self.properties[key] = value
    
    def get_property(self, key: str, default=None):
        """
        获取属性值
        
        Args:
            key: 属性键
            default: 默认值
            
        Returns:
            属性值
        """
        if not self.properties:
            return default
        return self.properties.get(key, default)
    
    def increment_mention_count(self) -> None:
        """增加提及次数"""
        self.mention_count += 1
    
    def mark_as_verified(self) -> None:
        """标记为已验证"""
        self.is_verified = True
    
    def link_to_external(self, external_id: str, external_url: str = None) -> None:
        """
        链接到外部知识库
        
        Args:
            external_id: 外部ID
            external_url: 外部URL
        """
        self.external_id = external_id
        self.external_url = external_url
        self.is_linked = True


# 创建索引
Index("idx_entities_canonical_name_type", Entity.canonical_name, Entity.entity_type)
Index("idx_entities_document_chunk", Entity.document_id, Entity.chunk_id)
Index("idx_entities_embedding", Entity.embedding, postgresql_using="ivfflat")
Index("idx_entities_properties", Entity.properties, postgresql_using="gin")
Index("idx_entities_confidence", Entity.confidence)
Index("idx_entities_importance", Entity.importance_score)
Index("idx_entities_mention_count", Entity.mention_count)
Index("idx_entities_verified_linked", Entity.is_verified, Entity.is_linked)