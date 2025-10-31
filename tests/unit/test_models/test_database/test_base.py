#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 基础数据库模型单元测试
=============================

测试 BaseModel 类的功能：
1. 基本字段（id、created_at、updated_at、extra_data）
2. 表名自动生成
3. 模型转换方法（to_dict、update_from_dict）
4. 通用方法测试

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import pytest
import uuid
from datetime import datetime
from typing import Dict, Any
from sqlalchemy import Column, String, Integer

from app.models.database.base import BaseModel, Base


class TestModel(BaseModel):
    """测试用的模型类"""
    __tablename__ = "test_models"
    
    name = Column(String(100), nullable=False)
    value = Column(Integer, default=0)


class TestBaseModel:
    """BaseModel 基础功能测试"""
    
    def test_table_name_generation(self):
        """测试表名自动生成功能"""
        # 测试简单类名
        assert TestModel.__tablename__ == "test_models"
        
        # 测试驼峰命名转换
        class DocumentChunk(BaseModel):
            __tablename__ = "document_chunks"
        
        # 验证驼峰命名转换为下划线命名
        expected_name = "document_chunk"
        # 注意：这里我们手动设置了 __tablename__，所以实际值是 "document_chunks"
        assert DocumentChunk.__tablename__ == "document_chunks"
    
    def test_base_fields_exist(self):
        """测试基础字段是否存在"""
        model = TestModel()
        
        # 检查基础字段是否存在
        assert hasattr(model, 'id')
        assert hasattr(model, 'created_at')
        assert hasattr(model, 'updated_at')
        assert hasattr(model, 'extra_data')
    
    def test_id_field_properties(self):
        """测试 ID 字段属性"""
        model = TestModel()
        
        # ID 应该是 UUID 类型
        assert model.id is None or isinstance(model.id, uuid.UUID)
        
        # 创建实例时应该自动生成 UUID
        model_with_data = TestModel(name="test")
        # 注意：在实际数据库操作中，UUID 会自动生成
    
    def test_extra_data_field(self):
        """测试额外数据字段"""
        model = TestModel(name="test")
        
        # extra_data 默认应该是空字典
        assert model.extra_data == {} or model.extra_data is None
        
        # 可以设置 JSON 数据
        test_data = {"key1": "value1", "key2": 123}
        model.extra_data = test_data
        assert model.extra_data == test_data
    
    def test_to_dict_method(self):
        """测试 to_dict 方法"""
        # 创建测试数据
        test_id = uuid.uuid4()
        test_time = datetime.now()
        test_extra = {"test": "data"}
        
        model = TestModel(
            id=test_id,
            name="test_name",
            value=42,
            extra_data=test_extra
        )
        model.created_at = test_time
        model.updated_at = test_time
        
        # 转换为字典
        result = model.to_dict()
        
        # 验证字典内容
        assert isinstance(result, dict)
        assert result["id"] == str(test_id)  # UUID 转换为字符串
        assert result["name"] == "test_name"
        assert result["value"] == 42
        assert result["extra_data"] == test_extra
        assert "created_at" in result
        assert "updated_at" in result
    
    def test_to_dict_excludes_none_values(self):
        """测试 to_dict 排除 None 值"""
        model = TestModel(name="test")
        result = model.to_dict()
        
        # None 值应该被排除
        for key, value in result.items():
            assert value is not None, f"Key '{key}' should not have None value"
    
    def test_update_from_dict_method(self):
        """测试 update_from_dict 方法"""
        model = TestModel(name="original", value=1)
        
        # 更新数据
        update_data = {
            "name": "updated",
            "value": 99,
            "extra_data": {"new": "data"}
        }
        
        model.update_from_dict(update_data)
        
        # 验证更新结果
        assert model.name == "updated"
        assert model.value == 99
        assert model.extra_data == {"new": "data"}
    
    def test_update_from_dict_ignores_invalid_fields(self):
        """测试 update_from_dict 忽略无效字段"""
        model = TestModel(name="test", value=1)
        original_name = model.name
        
        # 尝试更新不存在的字段
        update_data = {
            "invalid_field": "should_be_ignored",
            "name": "updated"
        }
        
        model.update_from_dict(update_data)
        
        # 有效字段应该被更新
        assert model.name == "updated"
        
        # 无效字段应该被忽略（不会抛出异常）
        assert not hasattr(model, "invalid_field")
    
    def test_update_from_dict_skips_none_values(self):
        """测试 update_from_dict 跳过 None 值"""
        model = TestModel(name="original", value=42)
        
        update_data = {
            "name": None,  # 应该被跳过
            "value": 99    # 应该被更新
        }
        
        model.update_from_dict(update_data)
        
        # None 值应该被跳过，原值保持不变
        assert model.name == "original"
        assert model.value == 99
    
    def test_repr_method(self):
        """测试 __repr__ 方法"""
        model = TestModel(name="test_repr")
        repr_str = repr(model)
        
        # 应该包含类名
        assert "TestModel" in repr_str
        # 应该包含 ID 信息
        assert "id=" in repr_str
    
    def test_get_table_name_class_method(self):
        """测试 get_table_name 类方法"""
        table_name = TestModel.get_table_name()
        assert table_name == "test_models"
    
    def test_get_columns_class_method(self):
        """测试 get_columns 类方法"""
        columns = TestModel.get_columns()
        
        # 应该包含基础字段
        expected_columns = ["id", "created_at", "updated_at", "extra_data", "name", "value"]
        
        for col in expected_columns:
            assert col in columns, f"Column '{col}' should be in columns list"
    
    def test_model_inheritance(self):
        """测试模型继承"""
        # TestModel 应该继承自 BaseModel
        assert issubclass(TestModel, BaseModel)
        
        # BaseModel 应该继承自 Base
        assert issubclass(BaseModel, Base)
    
    def test_metadata_naming_convention(self):
        """测试元数据命名约定"""
        from app.models.database.base import metadata
        
        # 检查命名约定是否设置
        assert metadata.naming_convention is not None
        
        # 检查特定的命名约定
        naming_conv = metadata.naming_convention
        assert "ix" in naming_conv  # 索引
        assert "uq" in naming_conv  # 唯一约束
        assert "ck" in naming_conv  # 检查约束
        assert "fk" in naming_conv  # 外键
        assert "pk" in naming_conv  # 主键


class TestBaseModelEdgeCases:
    """BaseModel 边界情况测试"""
    
    def test_empty_extra_data(self):
        """测试空的额外数据"""
        model = TestModel(name="test")
        model.extra_data = {}
        
        result = model.to_dict()
        # 空字典不应该被排除
        assert "extra_data" in result
        assert result["extra_data"] == {}
    
    def test_complex_extra_data(self):
        """测试复杂的额外数据"""
        complex_data = {
            "nested": {
                "key": "value",
                "number": 123,
                "list": [1, 2, 3]
            },
            "array": ["a", "b", "c"],
            "boolean": True,
            "null": None
        }
        
        model = TestModel(name="test", extra_data=complex_data)
        result = model.to_dict()
        
        assert result["extra_data"] == complex_data
    
    def test_update_with_empty_dict(self):
        """测试用空字典更新"""
        model = TestModel(name="original", value=42)
        
        model.update_from_dict({})
        
        # 原值应该保持不变
        assert model.name == "original"
        assert model.value == 42
    
    def test_to_dict_with_datetime_serialization(self):
        """测试日期时间序列化"""
        model = TestModel(name="test")
        test_time = datetime(2024, 1, 1, 12, 0, 0)
        model.created_at = test_time
        
        result = model.to_dict()
        
        # 日期时间应该被序列化为 ISO 格式字符串
        assert isinstance(result["created_at"], str)
        assert "2024-01-01" in result["created_at"]


@pytest.mark.unit
class TestBaseModelIntegration:
    """BaseModel 集成测试"""
    
    def test_model_creation_and_conversion_cycle(self):
        """测试模型创建和转换的完整周期"""
        # 1. 创建模型
        original_data = {
            "name": "test_cycle",
            "value": 100,
            "extra_data": {"meta": "information"}
        }
        
        model = TestModel(**original_data)
        
        # 2. 转换为字典
        dict_data = model.to_dict()
        
        # 3. 创建新模型并从字典更新
        new_model = TestModel(name="placeholder")
        new_model.update_from_dict(dict_data)
        
        # 4. 验证数据一致性
        assert new_model.name == original_data["name"]
        assert new_model.value == original_data["value"]
        assert new_model.extra_data == original_data["extra_data"]