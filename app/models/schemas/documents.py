#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 文档 API 模式
=====================

本模块定义了文档相关的 API 模式（Pydantic 模型）。

模式说明：
- DocumentCreate: 文档创建请求模式
- DocumentUpdate: 文档更新请求模式
- DocumentResponse: 文档响应模式
- DocumentStatus: 文档状态枚举
- DocumentUploadResponse: 文档上传响应模式
- DocumentListResponse: 文档列表响应模式

字段说明：
- title: 文档标题
- content: 文档内容
- file_path: 文件路径
- status: 处理状态
- metadata: 元数据信息

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import Field, validator, HttpUrl

from .base import (
    BaseSchema,
    IDMixin,
    TimestampMixin,
    MetadataMixin,
    PaginatedResponse
)


class DocumentStatus(str, Enum):
    """
    文档状态枚举
    
    定义文档处理的各个阶段状态。
    """
    PENDING = "pending"          # 待处理
    UPLOADING = "uploading"      # 上传中
    UPLOADED = "uploaded"        # 已上传
    PROCESSING = "processing"    # 处理中
    CHUNKING = "chunking"        # 分块中
    EMBEDDING = "embedding"      # 向量化中
    EXTRACTING = "extracting"    # 抽取中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"            # 处理失败
    ARCHIVED = "archived"        # 已归档


class DocumentType(str, Enum):
    """
    文档类型枚举
    
    定义支持的文档类型。
    """
    PDF = "pdf"
    TXT = "txt"
    MARKDOWN = "markdown"
    HTML = "html"
    DOCX = "docx"
    DOC = "doc"
    RTF = "rtf"
    ODT = "odt"
    EPUB = "epub"
    IMAGE = "image"
    OTHER = "other"


class DocumentCreate(BaseSchema):
    """
    文档创建请求模式
    
    用于创建新文档的请求数据。
    """
    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="文档标题"
    )
    
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="文档描述"
    )
    
    content: Optional[str] = Field(
        None,
        description="文档内容（如果直接提供文本）"
    )
    
    file_path: Optional[str] = Field(
        None,
        description="文件路径（如果上传文件）"
    )
    
    file_name: Optional[str] = Field(
        None,
        max_length=255,
        description="原始文件名"
    )
    
    file_size: Optional[int] = Field(
        None,
        ge=0,
        description="文件大小（字节）"
    )
    
    mime_type: Optional[str] = Field(
        None,
        max_length=100,
        description="MIME类型"
    )
    
    document_type: Optional[DocumentType] = Field(
        None,
        description="文档类型"
    )
    
    language: Optional[str] = Field(
        "zh",
        max_length=10,
        description="文档语言代码"
    )
    
    source_url: Optional[HttpUrl] = Field(
        None,
        description="文档来源URL"
    )
    
    author: Optional[str] = Field(
        None,
        max_length=200,
        description="作者"
    )
    
    tags: Optional[List[str]] = Field(
        default_factory=list,
        description="标签列表"
    )
    
    categories: Optional[List[str]] = Field(
        default_factory=list,
        description="分类列表"
    )
    
    is_public: bool = Field(
        default=False,
        description="是否公开"
    )
    
    auto_process: bool = Field(
        default=True,
        description="是否自动处理"
    )
    
    processing_options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="处理选项"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="元数据"
    )
    
    @validator('tags', 'categories')
    def validate_lists(cls, v):
        """验证列表字段"""
        if v is None:
            return []
        return [item.strip() for item in v if item.strip()]
    
    @validator('title')
    def validate_title(cls, v):
        """验证标题"""
        return v.strip()


class DocumentUpdate(BaseSchema):
    """
    文档更新请求模式
    
    用于更新现有文档的请求数据。
    """
    title: Optional[str] = Field(
        None,
        min_length=1,
        max_length=500,
        description="文档标题"
    )
    
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="文档描述"
    )
    
    author: Optional[str] = Field(
        None,
        max_length=200,
        description="作者"
    )
    
    tags: Optional[List[str]] = Field(
        None,
        description="标签列表"
    )
    
    categories: Optional[List[str]] = Field(
        None,
        description="分类列表"
    )
    
    is_public: Optional[bool] = Field(
        None,
        description="是否公开"
    )
    
    status: Optional[DocumentStatus] = Field(
        None,
        description="文档状态"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="元数据"
    )
    
    @validator('tags', 'categories')
    def validate_lists(cls, v):
        """验证列表字段"""
        if v is None:
            return None
        return [item.strip() for item in v if item.strip()]
    
    @validator('title')
    def validate_title(cls, v):
        """验证标题"""
        if v is not None:
            return v.strip()
        return v


class DocumentResponse(IDMixin, TimestampMixin, BaseSchema):
    """
    文档响应模式
    
    用于返回文档信息的响应数据。
    """
    title: str = Field(
        description="文档标题"
    )
    
    description: Optional[str] = Field(
        None,
        description="文档描述"
    )
    
    file_name: Optional[str] = Field(
        None,
        description="原始文件名"
    )
    
    file_path: Optional[str] = Field(
        None,
        description="文件路径"
    )
    
    file_size: Optional[int] = Field(
        None,
        description="文件大小（字节）"
    )
    
    mime_type: Optional[str] = Field(
        None,
        description="MIME类型"
    )
    
    document_type: Optional[DocumentType] = Field(
        None,
        description="文档类型"
    )
    
    language: Optional[str] = Field(
        None,
        description="文档语言"
    )
    
    source_url: Optional[str] = Field(
        None,
        description="来源URL"
    )
    
    author: Optional[str] = Field(
        None,
        description="作者"
    )
    
    status: DocumentStatus = Field(
        description="处理状态"
    )
    
    processing_progress: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="处理进度（0.0-1.0）"
    )
    
    error_message: Optional[str] = Field(
        None,
        description="错误信息"
    )
    
    # 统计信息
    total_pages: Optional[int] = Field(
        None,
        description="总页数"
    )
    
    total_chunks: int = Field(
        default=0,
        description="文本块数量"
    )
    
    total_entities: int = Field(
        default=0,
        description="实体数量"
    )
    
    total_relations: int = Field(
        default=0,
        description="关系数量"
    )
    
    total_images: int = Field(
        default=0,
        description="图像数量"
    )
    
    # 内容统计
    char_count: Optional[int] = Field(
        None,
        description="字符数"
    )
    
    word_count: Optional[int] = Field(
        None,
        description="单词数"
    )
    
    token_count: Optional[int] = Field(
        None,
        description="词元数"
    )
    
    # 分类和标签
    tags: List[str] = Field(
        default_factory=list,
        description="标签列表"
    )
    
    categories: List[str] = Field(
        default_factory=list,
        description="分类列表"
    )
    
    # 权限和可见性
    is_public: bool = Field(
        default=False,
        description="是否公开"
    )
    
    # 元数据
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="元数据"
    )
    
    # 处理时间
    processed_at: Optional[datetime] = Field(
        None,
        description="处理完成时间"
    )
    
    @property
    def is_processed(self) -> bool:
        """检查是否已处理完成"""
        return self.status == DocumentStatus.COMPLETED
    
    @property
    def has_error(self) -> bool:
        """检查是否有错误"""
        return self.status == DocumentStatus.FAILED
    
    @property
    def file_size_mb(self) -> Optional[float]:
        """获取文件大小（MB）"""
        if self.file_size:
            return self.file_size / (1024 * 1024)
        return None


class DocumentUploadResponse(BaseSchema):
    """
    文档上传响应模式
    
    用于返回文档上传结果的响应数据。
    """
    document_id: UUID = Field(
        description="文档ID"
    )
    
    upload_url: Optional[str] = Field(
        None,
        description="上传URL（如果使用预签名URL）"
    )
    
    file_path: str = Field(
        description="文件存储路径"
    )
    
    status: DocumentStatus = Field(
        description="上传状态"
    )
    
    message: str = Field(
        description="上传消息"
    )
    
    processing_started: bool = Field(
        default=False,
        description="是否已开始处理"
    )


class DocumentListResponse(PaginatedResponse[DocumentResponse]):
    """
    文档列表响应模式
    
    用于返回文档列表的分页响应数据。
    """
    pass


class DocumentSearchRequest(BaseSchema):
    """
    文档搜索请求模式
    
    用于搜索文档的请求参数。
    """
    query: Optional[str] = Field(
        None,
        description="搜索关键词"
    )
    
    status: Optional[List[DocumentStatus]] = Field(
        None,
        description="状态过滤"
    )
    
    document_type: Optional[List[DocumentType]] = Field(
        None,
        description="类型过滤"
    )
    
    tags: Optional[List[str]] = Field(
        None,
        description="标签过滤"
    )
    
    categories: Optional[List[str]] = Field(
        None,
        description="分类过滤"
    )
    
    author: Optional[str] = Field(
        None,
        description="作者过滤"
    )
    
    language: Optional[str] = Field(
        None,
        description="语言过滤"
    )
    
    is_public: Optional[bool] = Field(
        None,
        description="公开状态过滤"
    )
    
    created_after: Optional[datetime] = Field(
        None,
        description="创建时间起始"
    )
    
    created_before: Optional[datetime] = Field(
        None,
        description="创建时间结束"
    )
    
    min_file_size: Optional[int] = Field(
        None,
        ge=0,
        description="最小文件大小"
    )
    
    max_file_size: Optional[int] = Field(
        None,
        ge=0,
        description="最大文件大小"
    )


class DocumentStatsResponse(BaseSchema):
    """
    文档统计响应模式
    
    用于返回文档统计信息的响应数据。
    """
    total_documents: int = Field(
        description="文档总数"
    )
    
    status_counts: Dict[str, int] = Field(
        description="各状态文档数量"
    )
    
    type_counts: Dict[str, int] = Field(
        description="各类型文档数量"
    )
    
    total_file_size: int = Field(
        description="文件总大小（字节）"
    )
    
    total_chunks: int = Field(
        description="文本块总数"
    )
    
    total_entities: int = Field(
        description="实体总数"
    )
    
    total_relations: int = Field(
        description="关系总数"
    )
    
    avg_processing_time: Optional[float] = Field(
        None,
        description="平均处理时间（秒）"
    )
    
    recent_uploads: int = Field(
        description="最近24小时上传数量"
    )