#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 文档数据库模型
=====================

本模块定义了文档相关的数据库模型。

模型说明：
- Document: 文档主表，存储文档基本信息和元数据
- 支持多种文档类型：PDF、TXT、Markdown、HTML、DOCX 等
- 包含文档处理状态跟踪
- 与文本块、实体、关系、图像等模型建立关联关系

字段说明：
- title: 文档标题
- content: 文档原始内容（可选，大文档可能不存储）
- file_path: 文件存储路径
- file_type: 文件类型
- file_size: 文件大小（字节）
- language: 文档语言
- status: 处理状态（pending/processing/completed/failed）
- hash_value: 文件哈希值，用于去重
- source_url: 原始来源URL（如果适用）

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from sqlalchemy import Column, String, Text, Integer, Float, Boolean, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ARRAY

from .base import BaseModel


class Document(BaseModel):
    """
    文档模型
    
    存储上传到系统中的文档信息，包括文档内容、元数据和处理状态。
    每个文档可以包含多个文本块、实体、关系和图像。
    """
    __tablename__ = "documents"
    
    # 基本信息
    title = Column(
        String(500), 
        nullable=False, 
        index=True,
        comment="文档标题"
    )
    
    description = Column(
        Text,
        comment="文档描述"
    )
    
    content = Column(
        Text,
        comment="文档原始内容（可选）"
    )
    
    # 文件信息
    file_path = Column(
        String(1000), 
        nullable=False,
        comment="文件存储路径"
    )
    
    file_type = Column(
        String(50), 
        nullable=False,
        index=True,
        comment="文件类型（pdf, txt, md, html, docx等）"
    )
    
    file_size = Column(
        Integer,
        comment="文件大小（字节）"
    )
    
    hash_value = Column(
        String(64),
        unique=True,
        index=True,
        comment="文件哈希值，用于去重"
    )
    
    # 文档属性
    language = Column(
        String(10), 
        default="zh",
        index=True,
        comment="文档语言代码"
    )
    
    author = Column(
        String(200),
        comment="文档作者"
    )
    
    source_url = Column(
        String(1000),
        comment="原始来源URL"
    )
    
    keywords = Column(
        ARRAY(String),
        default=list,
        comment="关键词列表"
    )
    
    # 处理状态
    status = Column(
        String(20), 
        default="pending",
        index=True,
        comment="处理状态：pending/processing/completed/failed"
    )
    
    processing_progress = Column(
        Float,
        default=0.0,
        comment="处理进度（0.0-1.0）"
    )
    
    error_message = Column(
        Text,
        comment="错误信息（如果处理失败）"
    )
    
    # 统计信息
    chunk_count = Column(
        Integer,
        default=0,
        comment="文本块数量"
    )
    
    entity_count = Column(
        Integer,
        default=0,
        comment="实体数量"
    )
    
    relation_count = Column(
        Integer,
        default=0,
        comment="关系数量"
    )
    
    image_count = Column(
        Integer,
        default=0,
        comment="图像数量"
    )
    
    # 质量评分
    quality_score = Column(
        Float,
        comment="文档质量评分（0.0-1.0）"
    )
    
    # 是否公开
    is_public = Column(
        Boolean,
        default=False,
        comment="是否公开可见"
    )
    
    # 关联关系
    chunks = relationship(
        "Chunk", 
        back_populates="document", 
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    entities = relationship(
        "Entity", 
        back_populates="document", 
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    relations = relationship(
        "Relation", 
        back_populates="document", 
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    images = relationship(
        "Image", 
        back_populates="document", 
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    def __repr__(self) -> str:
        """返回文档的字符串表示"""
        return f"<Document(id={self.id}, title='{self.title}', status='{self.status}')>"
    
    @property
    def is_processed(self) -> bool:
        """检查文档是否已处理完成"""
        return self.status == "completed"
    
    @property
    def is_processing(self) -> bool:
        """检查文档是否正在处理中"""
        return self.status == "processing"
    
    @property
    def has_error(self) -> bool:
        """检查文档是否处理失败"""
        return self.status == "failed"
    
    def update_processing_progress(self, progress: float, status: str = None) -> None:
        """
        更新处理进度
        
        Args:
            progress: 进度值（0.0-1.0）
            status: 可选的状态更新
        """
        self.processing_progress = max(0.0, min(1.0, progress))
        if status:
            self.status = status
    
    def mark_as_completed(self) -> None:
        """标记文档处理完成"""
        self.status = "completed"
        self.processing_progress = 1.0
        self.error_message = None
    
    def mark_as_failed(self, error_message: str) -> None:
        """
        标记文档处理失败
        
        Args:
            error_message: 错误信息
        """
        self.status = "failed"
        self.error_message = error_message


# 创建索引
Index("idx_documents_title_status", Document.title, Document.status)
Index("idx_documents_file_type_language", Document.file_type, Document.language)
Index("idx_documents_created_at", Document.created_at)
Index("idx_documents_quality_score", Document.quality_score)