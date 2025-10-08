#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 图像数据库模型
=====================

本模块定义了图像相关的数据库模型。

模型说明：
- Image: 图像模型，存储文档中的图像信息
- 支持多种图像格式：PNG、JPG、GIF、SVG等
- 包含图像的基本信息、OCR文本、视觉特征等
- 支持图像内容分析和跨模态检索
- 与文档建立关联关系

字段说明：
- file_path: 图像文件路径
- file_size: 文件大小
- width/height: 图像尺寸
- format: 图像格式
- ocr_text: OCR识别的文本
- visual_features: 视觉特征向量
- description: 图像描述

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from sqlalchemy import Column, String, Text, Integer, Float, ForeignKey, Index, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSON
from pgvector.sqlalchemy import Vector

from .base import BaseModel


class Image(BaseModel):
    """
    图像模型
    
    存储文档中的图像信息，包括基本属性、OCR文本、视觉特征等。
    支持图像内容分析和跨模态检索。
    """
    __tablename__ = "images"
    
    # 关联文档
    document_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("documents.id", ondelete="CASCADE"), 
        nullable=False,
        index=True,
        comment="所属文档ID"
    )
    
    # 文件基本信息
    file_path = Column(
        String(1000), 
        nullable=False,
        comment="图像文件存储路径"
    )
    
    original_filename = Column(
        String(255),
        comment="原始文件名"
    )
    
    file_size = Column(
        Integer,
        comment="文件大小（字节）"
    )
    
    file_format = Column(
        String(20),
        comment="图像格式：png/jpg/gif/svg等"
    )
    
    mime_type = Column(
        String(50),
        comment="MIME类型"
    )
    
    # 图像属性
    width = Column(
        Integer,
        comment="图像宽度（像素）"
    )
    
    height = Column(
        Integer,
        comment="图像高度（像素）"
    )
    
    aspect_ratio = Column(
        Float,
        comment="宽高比"
    )
    
    color_mode = Column(
        String(20),
        comment="颜色模式：RGB/RGBA/CMYK/Grayscale等"
    )
    
    # 位置信息
    page_number = Column(
        Integer,
        comment="在文档中的页码（如果适用）"
    )
    
    position_x = Column(
        Float,
        comment="在页面中的X坐标"
    )
    
    position_y = Column(
        Float,
        comment="在页面中的Y坐标"
    )
    
    # OCR和文本信息
    ocr_text = Column(
        Text,
        comment="OCR识别的文本内容"
    )
    
    ocr_confidence = Column(
        Float,
        comment="OCR识别置信度（0.0-1.0）"
    )
    
    ocr_language = Column(
        String(10),
        comment="OCR识别的语言"
    )
    
    # 图像描述和标注
    caption = Column(
        Text,
        comment="图像标题或说明"
    )
    
    description = Column(
        Text,
        comment="图像描述"
    )
    
    alt_text = Column(
        Text,
        comment="替代文本"
    )
    
    # 视觉特征
    visual_features = Column(
        Vector(512),  # 常见的视觉特征向量维度
        comment="视觉特征向量"
    )
    
    feature_model = Column(
        String(100),
        comment="特征提取模型名称"
    )
    
    # 图像分析结果
    detected_objects = Column(
        JSON,
        default=list,
        comment="检测到的对象列表"
    )
    
    detected_text_regions = Column(
        JSON,
        default=list,
        comment="检测到的文本区域"
    )
    
    color_palette = Column(
        ARRAY(String),
        default=list,
        comment="主要颜色调色板"
    )
    
    # 质量评估
    quality_score = Column(
        Float,
        comment="图像质量评分（0.0-1.0）"
    )
    
    clarity_score = Column(
        Float,
        comment="清晰度评分（0.0-1.0）"
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
    
    # 分类和标签
    image_type = Column(
        String(50),
        comment="图像类型：photo/diagram/chart/table/logo等"
    )
    
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
    
    # 版权和来源
    copyright_info = Column(
        String(500),
        comment="版权信息"
    )
    
    source_url = Column(
        String(1000),
        comment="原始来源URL"
    )
    
    # 是否公开
    is_public = Column(
        Boolean,
        default=False,
        comment="是否公开可见"
    )
    
    # 关联关系
    document = relationship(
        "Document", 
        back_populates="images"
    )
    
    def __repr__(self) -> str:
        """返回图像的字符串表示"""
        return f"<Image(id={self.id}, document_id={self.document_id}, filename='{self.original_filename}')>"
    
    @property
    def is_processed(self) -> bool:
        """检查是否已处理完成"""
        return self.status == "completed"
    
    @property
    def has_ocr_text(self) -> bool:
        """检查是否包含OCR文本"""
        return bool(self.ocr_text and self.ocr_text.strip())
    
    @property
    def has_visual_features(self) -> bool:
        """检查是否已提取视觉特征"""
        return self.visual_features is not None
    
    @property
    def file_size_mb(self) -> float:
        """获取文件大小（MB）"""
        if self.file_size:
            return self.file_size / (1024 * 1024)
        return 0.0
    
    def calculate_aspect_ratio(self) -> None:
        """计算并设置宽高比"""
        if self.width and self.height:
            self.aspect_ratio = self.width / self.height
    
    def add_detected_object(self, object_info: dict) -> None:
        """
        添加检测到的对象
        
        Args:
            object_info: 对象信息字典，包含类型、置信度、边界框等
        """
        if not self.detected_objects:
            self.detected_objects = []
        self.detected_objects.append(object_info)
    
    def add_text_region(self, region_info: dict) -> None:
        """
        添加文本区域
        
        Args:
            region_info: 文本区域信息字典，包含文本、位置、置信度等
        """
        if not self.detected_text_regions:
            self.detected_text_regions = []
        self.detected_text_regions.append(region_info)
    
    def update_visual_features(self, features: list, model_name: str = None) -> None:
        """
        更新视觉特征
        
        Args:
            features: 视觉特征向量
            model_name: 特征提取模型名称
        """
        self.visual_features = features
        if model_name:
            self.feature_model = model_name
    
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
    
    def set_quality_scores(self, quality: float, clarity: float = None) -> None:
        """
        设置质量评分
        
        Args:
            quality: 整体质量评分
            clarity: 清晰度评分
        """
        self.quality_score = max(0.0, min(1.0, quality))
        if clarity is not None:
            self.clarity_score = max(0.0, min(1.0, clarity))
    
    def mark_as_completed(self) -> None:
        """标记处理完成"""
        self.status = "completed"
        self.processing_progress = 1.0
        self.error_message = None
    
    def mark_as_failed(self, error_message: str) -> None:
        """
        标记处理失败
        
        Args:
            error_message: 错误信息
        """
        self.status = "failed"
        self.error_message = error_message


# 创建索引
Index("idx_images_document_id", Image.document_id)
Index("idx_images_file_format", Image.file_format)
Index("idx_images_status", Image.status)
Index("idx_images_visual_features", Image.visual_features, postgresql_using="ivfflat")
Index("idx_images_ocr_text", Image.ocr_text, postgresql_using="gin", postgresql_ops={"ocr_text": "gin_trgm_ops"})
Index("idx_images_quality_score", Image.quality_score)
Index("idx_images_page_number", Image.page_number)
Index("idx_images_size", Image.width, Image.height)