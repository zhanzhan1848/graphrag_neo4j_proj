#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 文本块数据库模型
=======================

本模块定义了文本块相关的数据库模型。

模型说明：
- Chunk: 文本块模型，存储文档分块后的文本片段
- 每个文本块包含原始文本、向量嵌入、位置信息等
- 支持向量相似度搜索和语义检索
- 与文档、实体、关系建立关联关系

字段说明：
- content: 文本块内容
- chunk_index: 在文档中的序号
- start_pos: 在原文档中的起始位置
- end_pos: 在原文档中的结束位置
- token_count: 词元数量
- embedding: 向量嵌入
- embedding_model: 嵌入模型名称
- summary: 文本块摘要

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from sqlalchemy import Column, String, Text, Integer, Float, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from pgvector.sqlalchemy import Vector

from .base import BaseModel


class Chunk(BaseModel):
    """
    文本块模型
    
    存储文档分块后的文本片段，包括原始文本、向量嵌入和位置信息。
    用于支持语义检索和 RAG（检索增强生成）功能。
    """
    __tablename__ = "chunks"
    
    # 关联文档
    document_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("documents.id", ondelete="CASCADE"), 
        nullable=False,
        index=True,
        comment="所属文档ID"
    )
    
    # 文本内容
    content = Column(
        Text, 
        nullable=False,
        comment="文本块内容"
    )
    
    summary = Column(
        Text,
        comment="文本块摘要"
    )
    
    # 位置信息
    chunk_index = Column(
        Integer, 
        nullable=False,
        comment="在文档中的序号（从0开始）"
    )
    
    start_pos = Column(
        Integer,
        comment="在原文档中的起始字符位置"
    )
    
    end_pos = Column(
        Integer,
        comment="在原文档中的结束字符位置"
    )
    
    # 文本统计
    token_count = Column(
        Integer,
        comment="词元数量"
    )
    
    char_count = Column(
        Integer,
        comment="字符数量"
    )
    
    word_count = Column(
        Integer,
        comment="单词数量"
    )
    
    # 向量嵌入
    embedding = Column(
        Vector(1536),  # OpenAI text-embedding-ada-002 的维度
        comment="文本向量嵌入"
    )
    
    embedding_model = Column(
        String(100),
        default="text-embedding-ada-002",
        comment="嵌入模型名称"
    )
    
    # 语言和类型
    language = Column(
        String(10),
        comment="文本语言"
    )
    
    chunk_type = Column(
        String(50),
        default="text",
        comment="文本块类型：text/title/header/table/list等"
    )
    
    # 质量评分
    quality_score = Column(
        Float,
        comment="文本质量评分（0.0-1.0）"
    )
    
    coherence_score = Column(
        Float,
        comment="连贯性评分（0.0-1.0）"
    )
    
    # 处理状态
    status = Column(
        String(20),
        default="pending",
        index=True,
        comment="处理状态：pending/processing/completed/failed"
    )
    
    # 标签和分类
    tags = Column(
        ARRAY(String),
        default=list,
        comment="标签列表"
    )
    
    categories = Column(
        ARRAY(String),
        default=list,
        comment="分类列表"
    )
    
    # 关联关系
    document = relationship(
        "Document", 
        back_populates="chunks"
    )
    
    entities = relationship(
        "Entity", 
        back_populates="chunk",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    relations = relationship(
        "Relation", 
        back_populates="evidence_chunk",
        lazy="dynamic"
    )
    
    def __repr__(self) -> str:
        """返回文本块的字符串表示"""
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"<Chunk(id={self.id}, document_id={self.document_id}, index={self.chunk_index}, content='{content_preview}')>"
    
    @property
    def content_preview(self) -> str:
        """获取内容预览（前100个字符）"""
        return self.content[:100] + "..." if len(self.content) > 100 else self.content
    
    @property
    def is_embedded(self) -> bool:
        """检查是否已生成向量嵌入"""
        return self.embedding is not None
    
    @property
    def is_processed(self) -> bool:
        """检查是否已处理完成"""
        return self.status == "completed"
    
    def calculate_similarity(self, other_embedding: list) -> float:
        """
        计算与另一个向量的余弦相似度
        
        Args:
            other_embedding: 另一个向量嵌入
            
        Returns:
            float: 相似度分数（-1到1之间）
        """
        if not self.embedding or not other_embedding:
            return 0.0
        
        # 这里应该使用向量数据库的相似度计算功能
        # 或者使用 numpy 等库计算余弦相似度
        # 暂时返回占位符
        return 0.0
    
    def update_embedding(self, embedding: list, model_name: str = None) -> None:
        """
        更新向量嵌入
        
        Args:
            embedding: 向量嵌入
            model_name: 嵌入模型名称
        """
        self.embedding = embedding
        if model_name:
            self.embedding_model = model_name
        self.status = "completed"
    
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
    
    def remove_tag(self, tag: str) -> None:
        """
        移除标签
        
        Args:
            tag: 标签名称
        """
        if self.tags and tag in self.tags:
            self.tags.remove(tag)


# 创建索引
Index("idx_chunks_document_id_index", Chunk.document_id, Chunk.chunk_index)
Index("idx_chunks_embedding", Chunk.embedding, postgresql_using="ivfflat")
Index("idx_chunks_content_gin", Chunk.content, postgresql_using="gin", postgresql_ops={"content": "gin_trgm_ops"})
Index("idx_chunks_status_created", Chunk.status, Chunk.created_at)
Index("idx_chunks_quality_score", Chunk.quality_score)
Index("idx_chunks_token_count", Chunk.token_count)