#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 图像 API 模式
=====================

本模块定义了图像相关的 API 模式（Pydantic 模型）。

模式说明：
- ImageCreate: 图像创建请求模式
- ImageUpdate: 图像更新请求模式
- ImageResponse: 图像响应模式
- ImageListResponse: 图像列表响应模式
- ImageUploadResponse: 图像上传响应模式
- ImageAnalysisResponse: 图像分析响应模式

字段说明：
- file_path: 图像文件路径
- file_name: 图像文件名
- file_size: 文件大小
- mime_type: MIME 类型
- width/height: 图像尺寸
- analysis_results: 分析结果

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import Field, validator, HttpUrl

from .base import (
    BaseSchema,
    IDMixin,
    TimestampMixin,
    MetadataMixin,
    PaginatedResponse
)


class ImageFormat(str, Enum):
    """
    图像格式枚举
    
    定义系统支持的图像格式。
    """
    JPEG = "jpeg"
    JPG = "jpg"
    PNG = "png"
    GIF = "gif"
    BMP = "bmp"
    TIFF = "tiff"
    WEBP = "webp"
    SVG = "svg"


class ImageStatus(str, Enum):
    """
    图像状态枚举
    
    定义图像处理的各个阶段状态。
    """
    UPLOADED = "uploaded"           # 已上传
    PROCESSING = "processing"       # 处理中
    ANALYZED = "analyzed"           # 已分析
    INDEXED = "indexed"             # 已索引
    ERROR = "error"                 # 错误
    DELETED = "deleted"             # 已删除


class ImageAnalysisType(str, Enum):
    """
    图像分析类型枚举
    
    定义图像分析的类型。
    """
    OCR = "ocr"                     # 光学字符识别
    OBJECT_DETECTION = "object_detection"  # 物体检测
    SCENE_ANALYSIS = "scene_analysis"      # 场景分析
    TEXT_EXTRACTION = "text_extraction"    # 文本提取
    METADATA_EXTRACTION = "metadata_extraction"  # 元数据提取
    SIMILARITY = "similarity"       # 相似性分析


class ImageCreate(BaseSchema):
    """
    图像创建请求模式
    
    用于创建新的图像记录。
    """
    file_name: str = Field(
        ...,
        description="图像文件名",
        min_length=1,
        max_length=255
    )
    
    file_path: str = Field(
        ...,
        description="图像文件路径",
        min_length=1,
        max_length=500
    )
    
    file_size: int = Field(
        ...,
        description="文件大小（字节）",
        ge=0
    )
    
    mime_type: str = Field(
        ...,
        description="MIME 类型",
        pattern=r'^image/.*'
    )
    
    format: ImageFormat = Field(
        ...,
        description="图像格式"
    )
    
    width: Optional[int] = Field(
        None,
        description="图像宽度（像素）",
        ge=1
    )
    
    height: Optional[int] = Field(
        None,
        description="图像高度（像素）",
        ge=1
    )
    
    document_id: Optional[UUID] = Field(
        None,
        description="关联的文档ID"
    )
    
    description: Optional[str] = Field(
        None,
        description="图像描述",
        max_length=1000
    )
    
    tags: Optional[List[str]] = Field(
        default_factory=list,
        description="图像标签列表"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="图像的元数据信息"
    )


class ImageUpdate(BaseSchema):
    """
    图像更新请求模式
    
    用于更新现有的图像记录。
    """
    file_name: Optional[str] = Field(
        None,
        description="图像文件名",
        min_length=1,
        max_length=255
    )
    
    description: Optional[str] = Field(
        None,
        description="图像描述",
        max_length=1000
    )
    
    tags: Optional[List[str]] = Field(
        None,
        description="图像标签列表"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="图像的元数据信息"
    )


class ImageResponse(IDMixin, TimestampMixin, MetadataMixin, BaseSchema):
    """
    图像响应模式
    
    返回图像的完整信息。
    """
    file_name: str = Field(
        ...,
        description="图像文件名"
    )
    
    file_path: str = Field(
        ...,
        description="图像文件路径"
    )
    
    file_size: int = Field(
        ...,
        description="文件大小（字节）"
    )
    
    mime_type: str = Field(
        ...,
        description="MIME 类型"
    )
    
    format: ImageFormat = Field(
        ...,
        description="图像格式"
    )
    
    width: Optional[int] = Field(
        None,
        description="图像宽度（像素）"
    )
    
    height: Optional[int] = Field(
        None,
        description="图像高度（像素）"
    )
    
    status: ImageStatus = Field(
        ...,
        description="图像处理状态"
    )
    
    document_id: Optional[UUID] = Field(
        None,
        description="关联的文档ID"
    )
    
    description: Optional[str] = Field(
        None,
        description="图像描述"
    )
    
    tags: List[str] = Field(
        default_factory=list,
        description="图像标签列表"
    )
    
    # 分析结果
    ocr_text: Optional[str] = Field(
        None,
        description="OCR 提取的文本内容"
    )
    
    detected_objects: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="检测到的物体列表"
    )
    
    scene_description: Optional[str] = Field(
        None,
        description="场景描述"
    )
    
    # 统计信息
    analysis_count: int = Field(
        default=0,
        description="分析次数",
        ge=0
    )
    
    view_count: int = Field(
        default=0,
        description="查看次数",
        ge=0
    )
    
    # URL 信息
    download_url: Optional[HttpUrl] = Field(
        None,
        description="图像下载URL"
    )
    
    thumbnail_url: Optional[HttpUrl] = Field(
        None,
        description="缩略图URL"
    )


class ImageListResponse(PaginatedResponse[ImageResponse]):
    """
    图像列表响应模式
    
    返回分页的图像列表。
    """
    pass


class ImageUploadResponse(BaseSchema):
    """
    图像上传响应模式
    
    返回图像上传结果。
    """
    image_id: UUID = Field(
        ...,
        description="图像唯一标识符"
    )
    
    file_name: str = Field(
        ...,
        description="上传的文件名"
    )
    
    file_size: int = Field(
        ...,
        description="文件大小（字节）"
    )
    
    upload_time: datetime = Field(
        ...,
        description="上传时间"
    )
    
    status: ImageStatus = Field(
        ...,
        description="上传状态"
    )
    
    download_url: Optional[HttpUrl] = Field(
        None,
        description="图像下载URL"
    )
    
    thumbnail_url: Optional[HttpUrl] = Field(
        None,
        description="缩略图URL"
    )
    
    message: Optional[str] = Field(
        None,
        description="上传结果消息"
    )


class ImageAnalysisRequest(BaseSchema):
    """
    图像分析请求模式
    
    用于请求图像分析。
    """
    image_id: UUID = Field(
        ...,
        description="图像唯一标识符"
    )
    
    analysis_types: List[ImageAnalysisType] = Field(
        ...,
        description="分析类型列表",
        min_items=1
    )
    
    options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="分析选项"
    )
    
    force_reanalysis: bool = Field(
        default=False,
        description="是否强制重新分析"
    )


class ImageAnalysisResponse(BaseSchema):
    """
    图像分析响应模式
    
    返回图像分析结果。
    """
    image_id: UUID = Field(
        ...,
        description="图像唯一标识符"
    )
    
    analysis_time: datetime = Field(
        ...,
        description="分析时间"
    )
    
    processing_time_ms: float = Field(
        ...,
        description="处理耗时（毫秒）",
        ge=0
    )
    
    results: Dict[ImageAnalysisType, Any] = Field(
        ...,
        description="分析结果字典"
    )
    
    confidence_scores: Optional[Dict[ImageAnalysisType, float]] = Field(
        None,
        description="置信度分数"
    )
    
    errors: Optional[List[str]] = Field(
        None,
        description="分析过程中的错误信息"
    )
    
    status: str = Field(
        ...,
        description="分析状态：success、partial、failed"
    )


class OCRResult(BaseSchema):
    """
    OCR 结果模式
    
    光学字符识别的结果。
    """
    text: str = Field(
        ...,
        description="识别出的文本内容"
    )
    
    confidence: float = Field(
        ...,
        description="识别置信度",
        ge=0.0,
        le=1.0
    )
    
    bounding_boxes: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="文本区域边界框"
    )
    
    language: Optional[str] = Field(
        None,
        description="识别的语言"
    )
    
    word_count: int = Field(
        ...,
        description="单词数量",
        ge=0
    )


class DetectedObject(BaseSchema):
    """
    检测到的物体
    
    单个检测物体的信息。
    """
    label: str = Field(
        ...,
        description="物体标签"
    )
    
    confidence: float = Field(
        ...,
        description="检测置信度",
        ge=0.0,
        le=1.0
    )
    
    bounding_box: Dict[str, float] = Field(
        ...,
        description="边界框坐标 {x, y, width, height}"
    )
    
    attributes: Optional[Dict[str, Any]] = Field(
        None,
        description="物体属性"
    )


class ObjectDetectionResult(BaseSchema):
    """
    物体检测结果模式
    
    物体检测的结果。
    """
    objects: List[DetectedObject] = Field(
        ...,
        description="检测到的物体列表"
    )
    
    total_objects: int = Field(
        ...,
        description="检测到的物体总数",
        ge=0
    )
    
    processing_time_ms: float = Field(
        ...,
        description="检测耗时（毫秒）",
        ge=0
    )


class ImageSearchRequest(BaseSchema):
    """
    图像搜索请求模式
    
    用于搜索图像。
    """
    query: Optional[str] = Field(
        None,
        description="文本查询",
        max_length=500
    )
    
    image_query: Optional[UUID] = Field(
        None,
        description="以图搜图的参考图像ID"
    )
    
    tags: Optional[List[str]] = Field(
        None,
        description="标签过滤"
    )
    
    formats: Optional[List[ImageFormat]] = Field(
        None,
        description="格式过滤"
    )
    
    size_range: Optional[Dict[str, int]] = Field(
        None,
        description="尺寸范围过滤 {min_width, max_width, min_height, max_height}"
    )
    
    similarity_threshold: float = Field(
        default=0.7,
        description="相似度阈值",
        ge=0.0,
        le=1.0
    )
    
    max_results: int = Field(
        default=20,
        description="最大返回结果数",
        ge=1,
        le=100
    )


class ImageSearchResult(BaseSchema):
    """
    图像搜索结果项
    
    单个图像搜索结果的详细信息。
    """
    image: ImageResponse = Field(
        ...,
        description="图像信息"
    )
    
    score: float = Field(
        ...,
        description="相关性分数",
        ge=0.0,
        le=1.0
    )
    
    match_reason: Optional[str] = Field(
        None,
        description="匹配原因说明"
    )
    
    highlights: Optional[List[str]] = Field(
        None,
        description="高亮匹配的内容"
    )
    

class ImageSearchResponse(BaseSchema):
    """
    图像搜索响应模式
    
    返回图像搜索结果。
    """
    query: Optional[str] = Field(
        None,
        description="原始文本查询"
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
    
    results: List[ImageSearchResult] = Field(
        default_factory=list,
        description="搜索结果列表"
    )