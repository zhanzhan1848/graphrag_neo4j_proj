#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异常类单元测试
=============

测试 GraphRAG 系统的自定义异常类。

测试内容：
- 异常初始化
- 异常属性
- 异常继承关系
- 异常序列化

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import pytest
from typing import Dict, Any

from app.utils.exceptions import (
    GraphRAGException,
    DatabaseError,
    DatabaseConnectionError,
    FileStorageError,
    FileValidationError,
    DocumentError,
    DocumentNotFoundError,
    EntityError,
    RelationError,
    GraphDatabaseError,
    VectorError,
    APIError,
    ValidationError
)


class TestGraphRAGException:
    """测试基础异常类"""
    
    def test_basic_initialization(self):
        """测试基本初始化"""
        message = "测试错误消息"
        exception = GraphRAGException(message)
        
        assert str(exception) == message
        assert exception.message == message
        assert exception.error_code is None
        assert exception.details is None
    
    def test_initialization_with_error_code(self):
        """测试带错误代码的初始化"""
        message = "测试错误消息"
        error_code = "TEST_ERROR"
        exception = GraphRAGException(message, error_code=error_code)
        
        assert exception.message == message
        assert exception.error_code == error_code
        assert exception.details is None
    
    def test_initialization_with_details(self):
        """测试带详情的初始化"""
        message = "测试错误消息"
        details = {"key": "value", "number": 123}
        exception = GraphRAGException(message, details=details)
        
        assert exception.message == message
        assert exception.error_code is None
        assert exception.details == details
    
    def test_full_initialization(self):
        """测试完整初始化"""
        message = "测试错误消息"
        error_code = "TEST_ERROR"
        details = {"key": "value"}
        
        exception = GraphRAGException(
            message, 
            error_code=error_code, 
            details=details
        )
        
        assert exception.message == message
        assert exception.error_code == error_code
        assert exception.details == details
    
    def test_to_dict(self):
        """测试字典序列化"""
        message = "测试错误消息"
        error_code = "TEST_ERROR"
        details = {"key": "value"}
        
        exception = GraphRAGException(
            message, 
            error_code=error_code, 
            details=details
        )
        
        result = exception.to_dict()
        
        expected = {
            "message": message,
            "error_code": error_code,
            "details": details
        }
        
        assert result == expected
    
    def test_to_dict_minimal(self):
        """测试最小字典序列化"""
        message = "测试错误消息"
        exception = GraphRAGException(message)
        
        result = exception.to_dict()
        
        expected = {
            "message": message,
            "error_code": None,
            "details": None
        }
        
        assert result == expected


class TestDatabaseExceptions:
    """测试数据库异常类"""
    
    def test_database_error_inheritance(self):
        """测试数据库异常继承"""
        exception = DatabaseError("数据库错误")
        assert isinstance(exception, GraphRAGException)
        assert isinstance(exception, DatabaseError)
    
    def test_database_connection_error_inheritance(self):
        """测试数据库连接异常继承"""
        exception = DatabaseConnectionError("连接失败")
        assert isinstance(exception, DatabaseError)
        assert isinstance(exception, GraphRAGException)
    
    def test_database_error_with_details(self):
        """测试带详情的数据库异常"""
        details = {
            "host": "localhost",
            "port": 5432,
            "database": "test_db"
        }
        exception = DatabaseError(
            "连接数据库失败", 
            error_code="DB_CONNECTION_FAILED",
            details=details
        )
        
        assert exception.message == "连接数据库失败"
        assert exception.error_code == "DB_CONNECTION_FAILED"
        assert exception.details == details


class TestFileExceptions:
    """测试文件异常类"""
    
    def test_file_storage_error_inheritance(self):
        """测试文件存储异常继承"""
        exception = FileStorageError("文件存储错误")
        assert isinstance(exception, GraphRAGException)
    
    def test_file_validation_error_inheritance(self):
        """测试文件验证异常继承"""
        exception = FileValidationError("文件验证失败")
        assert isinstance(exception, FileStorageError)
        assert isinstance(exception, GraphRAGException)
    
    def test_file_validation_error_with_file_info(self):
        """测试带文件信息的验证异常"""
        details = {
            "filename": "test.pdf",
            "size": 1024000,
            "mime_type": "application/pdf",
            "max_size": 500000
        }
        exception = FileValidationError(
            "文件大小超出限制",
            error_code="FILE_TOO_LARGE",
            details=details
        )
        
        assert "文件大小超出限制" in exception.message
        assert exception.details["filename"] == "test.pdf"
        assert exception.details["size"] == 1024000


class TestDocumentExceptions:
    """测试文档异常类"""
    
    def test_document_error_inheritance(self):
        """测试文档异常继承"""
        exception = DocumentError("文档错误")
        assert isinstance(exception, GraphRAGException)
    
    def test_document_not_found_error(self):
        """测试文档未找到异常"""
        document_id = "123e4567-e89b-12d3-a456-426614174000"
        exception = DocumentNotFoundError(
            f"文档 {document_id} 未找到",
            details={"document_id": document_id}
        )
        
        assert isinstance(exception, DocumentError)
        assert document_id in exception.message
        assert exception.details["document_id"] == document_id


class TestEntityExceptions:
    """测试实体异常类"""
    
    def test_entity_error_inheritance(self):
        """测试实体异常继承"""
        exception = EntityError("实体错误")
        assert isinstance(exception, GraphRAGException)
    
    def test_entity_error_with_entity_info(self):
        """测试带实体信息的异常"""
        details = {
            "entity_name": "测试实体",
            "entity_type": "PERSON",
            "confidence": 0.95
        }
        exception = EntityError(
            "实体处理失败",
            error_code="ENTITY_PROCESSING_FAILED",
            details=details
        )
        
        assert exception.details["entity_name"] == "测试实体"
        assert exception.details["entity_type"] == "PERSON"


class TestRelationExceptions:
    """测试关系异常类"""
    
    def test_relation_error_inheritance(self):
        """测试关系异常继承"""
        exception = RelationError("关系错误")
        assert isinstance(exception, GraphRAGException)
    
    def test_relation_error_with_relation_info(self):
        """测试带关系信息的异常"""
        details = {
            "source_entity": "实体A",
            "target_entity": "实体B",
            "relation_type": "WORKS_AT",
            "confidence": 0.85
        }
        exception = RelationError(
            "关系抽取失败",
            error_code="RELATION_EXTRACTION_FAILED",
            details=details
        )
        
        assert exception.details["source_entity"] == "实体A"
        assert exception.details["relation_type"] == "WORKS_AT"


class TestGraphDatabaseExceptions:
    """测试图数据库异常类"""
    
    def test_graph_database_error_inheritance(self):
        """测试图数据库异常继承"""
        exception = GraphDatabaseError("图数据库错误")
        assert isinstance(exception, GraphRAGException)
    
    def test_graph_database_error_with_query_info(self):
        """测试带查询信息的图数据库异常"""
        details = {
            "query": "MATCH (n:Person) RETURN n",
            "parameters": {"name": "张三"},
            "database": "neo4j"
        }
        exception = GraphDatabaseError(
            "图查询执行失败",
            error_code="GRAPH_QUERY_FAILED",
            details=details
        )
        
        assert "MATCH" in exception.details["query"]
        assert exception.details["parameters"]["name"] == "张三"


class TestVectorExceptions:
    """测试向量异常类"""
    
    def test_vector_error_inheritance(self):
        """测试向量异常继承"""
        exception = VectorError("向量错误")
        assert isinstance(exception, GraphRAGException)
    
    def test_vector_error_with_embedding_info(self):
        """测试带嵌入信息的向量异常"""
        details = {
            "text": "测试文本",
            "model": "text-embedding-ada-002",
            "dimension": 1536
        }
        exception = VectorError(
            "向量生成失败",
            error_code="VECTOR_GENERATION_FAILED",
            details=details
        )
        
        assert exception.details["text"] == "测试文本"
        assert exception.details["dimension"] == 1536


class TestAPIExceptions:
    """测试 API 异常类"""
    
    def test_api_error_inheritance(self):
        """测试 API 异常继承"""
        exception = APIError("API 错误")
        assert isinstance(exception, GraphRAGException)
    
    def test_validation_error_inheritance(self):
        """测试验证异常继承"""
        exception = ValidationError("验证失败")
        assert isinstance(exception, APIError)
        assert isinstance(exception, GraphRAGException)
    
    def test_validation_error_with_field_info(self):
        """测试带字段信息的验证异常"""
        details = {
            "field": "title",
            "value": "",
            "constraint": "min_length=1",
            "message": "标题不能为空"
        }
        exception = ValidationError(
            "字段验证失败",
            error_code="FIELD_VALIDATION_FAILED",
            details=details
        )
        
        assert exception.details["field"] == "title"
        assert exception.details["constraint"] == "min_length=1"


class TestExceptionChaining:
    """测试异常链"""
    
    def test_exception_chaining(self):
        """测试异常链"""
        try:
            try:
                raise ValueError("原始错误")
            except ValueError as e:
                raise DocumentError("文档处理失败") from e
        except DocumentError as doc_error:
            assert doc_error.__cause__ is not None
            assert isinstance(doc_error.__cause__, ValueError)
            assert str(doc_error.__cause__) == "原始错误"
    
    def test_nested_exception_details(self):
        """测试嵌套异常详情"""
        original_error = "数据库连接超时"
        
        try:
            raise DatabaseConnectionError(original_error)
        except DatabaseConnectionError as db_error:
            details = {
                "original_error": str(db_error),
                "retry_count": 3,
                "timeout": 30
            }
            doc_error = DocumentError(
                "文档保存失败",
                error_code="DOCUMENT_SAVE_FAILED",
                details=details
            )
            
            assert doc_error.details["original_error"] == original_error
            assert doc_error.details["retry_count"] == 3