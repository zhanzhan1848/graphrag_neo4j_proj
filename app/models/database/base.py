#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 数据库基础模型
=====================

本模块定义了数据库模型的基类，提供通用的字段和方法。

主要功能：
1. 定义通用字段（id、创建时间、更新时间、元数据）
2. 提供模型转换方法（to_dict、update_from_dict）
3. 统一的表命名规则
4. 通用的审计字段

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Column, String, DateTime, JSON, MetaData
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.sql import func

# 创建元数据对象，用于表命名约定
metadata = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s"
    }
)

# 创建声明性基类
Base = declarative_base(metadata=metadata)


class BaseModel(Base):
    """
    数据库模型基类
    
    提供所有数据库模型的通用字段和方法：
    - id: UUID 主键
    - created_at: 创建时间
    - updated_at: 更新时间  
    - extra_data: JSON 额外数据字段
    
    方法：
    - to_dict(): 转换为字典格式
    - update_from_dict(): 从字典更新模型属性
    """
    __abstract__ = True
    
    @declared_attr
    def __tablename__(cls) -> str:
        """
        自动生成表名
        
        将类名转换为小写下划线格式作为表名
        例如: DocumentChunk -> document_chunk
        """
        import re
        # 将驼峰命名转换为下划线命名
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', cls.__name__)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    
    # 通用字段
    id = Column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4,
        comment="主键ID"
    )
    
    created_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False,
        comment="创建时间"
    )
    
    updated_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(),
        onupdate=func.now(),
        comment="更新时间"
    )
    
    extra_data = Column(
        JSON,
        default=dict,
        comment="额外数据信息"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将模型转换为字典格式
        
        Returns:
            Dict[str, Any]: 包含所有字段的字典
        """
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            # 处理特殊类型
            if isinstance(value, datetime):
                value = value.isoformat()
            elif isinstance(value, uuid.UUID):
                value = str(value)
            result[column.name] = value
        return result
    
    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """
        从字典更新模型属性
        
        Args:
            data: 包含要更新字段的字典
        """
        for key, value in data.items():
            if hasattr(self, key) and key != 'id':  # 不允许更新主键
                setattr(self, key, value)
    
    def __repr__(self) -> str:
        """
        返回模型的字符串表示
        
        Returns:
            str: 模型的字符串表示
        """
        return f"<{self.__class__.__name__}(id={self.id})>"
    
    @classmethod
    def get_table_name(cls) -> str:
        """
        获取表名
        
        Returns:
            str: 表名
        """
        return cls.__tablename__
    
    @classmethod
    def get_columns(cls) -> list:
        """
        获取所有列名
        
        Returns:
            list: 列名列表
        """
        return [column.name for column in cls.__table__.columns]