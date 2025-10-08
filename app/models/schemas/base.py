#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 基础 API 模式
=====================

本模块定义了 GraphRAG 系统的基础 API 模式（Pydantic 模型）。

模式说明：
- BaseSchema: 基础模式类，包含通用字段和方法
- PaginationParams: 分页参数模式
- PaginatedResponse: 分页响应模式
- StatusResponse: 状态响应模式
- ErrorResponse: 错误响应模式

字段说明：
- id: 唯一标识符
- created_at/updated_at: 创建和更新时间
- metadata: 元数据字典
- page/size: 分页参数
- total/items: 分页响应数据

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict
from pydantic.alias_generators import to_camel


# 泛型类型变量
T = TypeVar('T')


class BaseSchema(BaseModel):
    """
    基础模式类
    
    所有 API 模式的基类，包含通用字段和配置。
    """
    model_config = ConfigDict(
        # 使用驼峰命名法作为别名
        alias_generator=to_camel,
        # 允许通过别名填充字段
        populate_by_name=True,
        # 验证赋值
        validate_assignment=True,
        # 使用枚举值而不是名称
        use_enum_values=True,
        # 序列化时排除未设置的字段
        exclude_unset=True,
        # 允许额外字段
        extra='forbid'
    )
    
    def to_dict(self, **kwargs) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            字典表示
        """
        return self.model_dump(**kwargs)
    
    def to_json(self, **kwargs) -> str:
        """
        转换为JSON字符串
        
        Returns:
            JSON字符串
        """
        return self.model_dump_json(**kwargs)


class TimestampMixin(BaseSchema):
    """
    时间戳混入类
    
    包含创建时间和更新时间字段。
    """
    created_at: Optional[datetime] = Field(
        None,
        description="创建时间"
    )
    
    updated_at: Optional[datetime] = Field(
        None,
        description="更新时间"
    )


class MetadataMixin(BaseSchema):
    """
    元数据混入类
    
    包含元数据字段。
    """
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="元数据字典"
    )


class IDMixin(BaseSchema):
    """
    ID混入类
    
    包含唯一标识符字段。
    """
    id: Optional[UUID] = Field(
        None,
        description="唯一标识符"
    )


class PaginationParams(BaseSchema):
    """
    分页参数模式
    
    用于API请求的分页参数。
    """
    page: int = Field(
        default=1,
        ge=1,
        description="页码，从1开始"
    )
    
    size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="每页大小，最大100"
    )
    
    @property
    def offset(self) -> int:
        """计算偏移量"""
        return (self.page - 1) * self.size
    
    @property
    def limit(self) -> int:
        """获取限制数量"""
        return self.size


class SortParams(BaseSchema):
    """
    排序参数模式
    
    用于API请求的排序参数。
    """
    sort_by: Optional[str] = Field(
        None,
        description="排序字段"
    )
    
    sort_order: Optional[str] = Field(
        default="asc",
        pattern="^(asc|desc)$",
        description="排序顺序：asc或desc"
    )


class FilterParams(BaseSchema):
    """
    过滤参数模式
    
    用于API请求的过滤参数。
    """
    filters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="过滤条件字典"
    )
    
    search: Optional[str] = Field(
        None,
        description="搜索关键词"
    )


class PaginatedResponse(BaseSchema, Generic[T]):
    """
    分页响应模式
    
    用于返回分页数据的通用响应格式。
    """
    items: List[T] = Field(
        default_factory=list,
        description="数据项列表"
    )
    
    total: int = Field(
        default=0,
        ge=0,
        description="总数量"
    )
    
    page: int = Field(
        default=1,
        ge=1,
        description="当前页码"
    )
    
    size: int = Field(
        default=20,
        ge=1,
        description="每页大小"
    )
    
    pages: int = Field(
        default=0,
        ge=0,
        description="总页数"
    )
    
    has_next: bool = Field(
        default=False,
        description="是否有下一页"
    )
    
    has_prev: bool = Field(
        default=False,
        description="是否有上一页"
    )
    
    @classmethod
    def create(
        cls,
        items: List[T],
        total: int,
        page: int = 1,
        size: int = 20
    ) -> "PaginatedResponse[T]":
        """
        创建分页响应
        
        Args:
            items: 数据项列表
            total: 总数量
            page: 当前页码
            size: 每页大小
            
        Returns:
            分页响应实例
        """
        pages = (total + size - 1) // size if total > 0 else 0
        has_next = page < pages
        has_prev = page > 1
        
        return cls(
            items=items,
            total=total,
            page=page,
            size=size,
            pages=pages,
            has_next=has_next,
            has_prev=has_prev
        )


class StatusResponse(BaseSchema):
    """
    状态响应模式
    
    用于返回操作状态的响应格式。
    """
    success: bool = Field(
        description="操作是否成功"
    )
    
    message: Optional[str] = Field(
        None,
        description="状态消息"
    )
    
    code: Optional[str] = Field(
        None,
        description="状态代码"
    )
    
    data: Optional[Dict[str, Any]] = Field(
        None,
        description="附加数据"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="响应时间戳"
    )
    
    @classmethod
    def success_response(
        cls,
        message: str = "操作成功",
        data: Optional[Dict[str, Any]] = None,
        code: Optional[str] = None
    ) -> "StatusResponse":
        """
        创建成功响应
        
        Args:
            message: 成功消息
            data: 附加数据
            code: 状态代码
            
        Returns:
            成功状态响应
        """
        return cls(
            success=True,
            message=message,
            data=data,
            code=code
        )
    
    @classmethod
    def error_response(
        cls,
        message: str = "操作失败",
        data: Optional[Dict[str, Any]] = None,
        code: Optional[str] = None
    ) -> "StatusResponse":
        """
        创建错误响应
        
        Args:
            message: 错误消息
            data: 附加数据
            code: 错误代码
            
        Returns:
            错误状态响应
        """
        return cls(
            success=False,
            message=message,
            data=data,
            code=code
        )


class ErrorResponse(BaseSchema):
    """
    错误响应模式
    
    用于返回错误信息的响应格式。
    """
    error: str = Field(
        description="错误类型"
    )
    
    message: str = Field(
        description="错误消息"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="错误详情"
    )
    
    code: Optional[int] = Field(
        None,
        description="错误代码"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="错误时间戳"
    )
    
    request_id: Optional[str] = Field(
        None,
        description="请求ID"
    )
    
    @classmethod
    def validation_error(
        cls,
        message: str = "输入验证失败",
        details: Optional[Dict[str, Any]] = None
    ) -> "ErrorResponse":
        """
        创建验证错误响应
        
        Args:
            message: 错误消息
            details: 错误详情
            
        Returns:
            验证错误响应
        """
        return cls(
            error="ValidationError",
            message=message,
            details=details,
            code=400
        )
    
    @classmethod
    def not_found_error(
        cls,
        message: str = "资源未找到",
        details: Optional[Dict[str, Any]] = None
    ) -> "ErrorResponse":
        """
        创建未找到错误响应
        
        Args:
            message: 错误消息
            details: 错误详情
            
        Returns:
            未找到错误响应
        """
        return cls(
            error="NotFoundError",
            message=message,
            details=details,
            code=404
        )
    
    @classmethod
    def internal_error(
        cls,
        message: str = "内部服务器错误",
        details: Optional[Dict[str, Any]] = None
    ) -> "ErrorResponse":
        """
        创建内部错误响应
        
        Args:
            message: 错误消息
            details: 错误详情
            
        Returns:
            内部错误响应
        """
        return cls(
            error="InternalError",
            message=message,
            details=details,
            code=500
        )


class HealthResponse(BaseSchema):
    """
    健康检查响应模式
    
    用于系统健康检查的响应格式。
    """
    status: str = Field(
        description="健康状态：healthy/unhealthy/degraded"
    )
    
    version: Optional[str] = Field(
        None,
        description="系统版本"
    )
    
    uptime: Optional[float] = Field(
        None,
        description="运行时间（秒）"
    )
    
    checks: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="各组件健康检查结果"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="检查时间戳"
    )


class BatchRequest(BaseSchema, Generic[T]):
    """
    批量请求模式
    
    用于批量操作的请求格式。
    """
    items: List[T] = Field(
        description="批量操作的数据项列表"
    )
    
    options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="批量操作选项"
    )


class BatchResponse(BaseSchema, Generic[T]):
    """
    批量响应模式
    
    用于批量操作的响应格式。
    """
    success_count: int = Field(
        default=0,
        ge=0,
        description="成功处理的数量"
    )
    
    error_count: int = Field(
        default=0,
        ge=0,
        description="处理失败的数量"
    )
    
    total_count: int = Field(
        default=0,
        ge=0,
        description="总处理数量"
    )
    
    results: List[T] = Field(
        default_factory=list,
        description="处理结果列表"
    )
    
    errors: List[ErrorResponse] = Field(
        default_factory=list,
        description="错误信息列表"
    )