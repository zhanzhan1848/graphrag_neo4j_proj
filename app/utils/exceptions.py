#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 自定义异常
=================

本模块定义了 GraphRAG 系统的自定义异常类。

异常分类：
- 基础异常：系统级别的异常
- 数据库异常：数据库操作相关异常
- 文件异常：文件操作相关异常
- 文档异常：文档处理相关异常
- 实体异常：实体处理相关异常
- 关系异常：关系处理相关异常
- 图数据库异常：Neo4j 操作相关异常
- 向量异常：向量处理相关异常
- API 异常：API 调用相关异常

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from typing import Optional, Dict, Any


class GraphRAGException(Exception):
    """
    GraphRAG 基础异常类
    
    所有自定义异常的基类。
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        初始化异常
        
        Args:
            message: 错误消息
            error_code: 错误代码
            details: 错误详情
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            异常信息字典
        """
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


# 数据库相关异常
class DatabaseError(GraphRAGException):
    """数据库操作异常"""
    pass


class DatabaseConnectionError(DatabaseError):
    """数据库连接异常"""
    pass


class DatabaseQueryError(DatabaseError):
    """数据库查询异常"""
    pass


class DatabaseTransactionError(DatabaseError):
    """数据库事务异常"""
    pass


# 文件相关异常
class FileStorageError(GraphRAGException):
    """文件存储异常"""
    pass


class FileValidationError(FileStorageError):
    """文件验证异常"""
    pass


class FileNotFoundError(FileStorageError):
    """文件不存在异常"""
    pass


class FilePermissionError(FileStorageError):
    """文件权限异常"""
    pass


class FileSizeError(FileValidationError):
    """文件大小异常"""
    pass


class FileTypeError(FileValidationError):
    """文件类型异常"""
    pass


# 文档相关异常
class DocumentError(GraphRAGException):
    """文档处理异常"""
    pass


class DocumentNotFoundError(DocumentError):
    """文档不存在异常"""
    pass


class DocumentValidationError(DocumentError):
    """文档验证异常"""
    pass


class DocumentPermissionError(DocumentError):
    """文档权限异常"""
    pass


class DocumentProcessingError(DocumentError):
    """文档处理异常"""
    pass


class DocumentParsingError(DocumentProcessingError):
    """文档解析异常"""
    pass


class DocumentChunkingError(DocumentProcessingError):
    """文档分块异常"""
    pass


# 实体相关异常
class EntityError(GraphRAGException):
    """实体处理异常"""
    pass


class EntityNotFoundError(EntityError):
    """实体不存在异常"""
    pass


class EntityValidationError(EntityError):
    """实体验证异常"""
    pass


class EntityExtractionError(EntityError):
    """实体抽取异常"""
    pass


class EntityLinkingError(EntityError):
    """实体链接异常"""
    pass


# 关系相关异常
class RelationError(GraphRAGException):
    """关系处理异常"""
    pass


class RelationNotFoundError(RelationError):
    """关系不存在异常"""
    pass


class RelationValidationError(RelationError):
    """关系验证异常"""
    pass


class RelationExtractionError(RelationError):
    """关系抽取异常"""
    pass


# 图数据库相关异常
class GraphDatabaseError(GraphRAGException):
    """图数据库异常"""
    pass


class GraphConnectionError(GraphDatabaseError):
    """图数据库连接异常"""
    pass


class GraphQueryError(GraphDatabaseError):
    """图查询异常"""
    pass


class GraphNodeError(GraphDatabaseError):
    """图节点异常"""
    pass


class GraphRelationshipError(GraphDatabaseError):
    """图关系异常"""
    pass


class GraphSchemaError(GraphDatabaseError):
    """图模式异常"""
    pass


class GraphTransactionError(GraphDatabaseError):
    """图事务异常"""
    pass


class GraphIndexError(GraphDatabaseError):
    """图索引异常"""
    pass


class GraphConstraintError(GraphDatabaseError):
    """图约束异常"""
    pass


class GraphPerformanceError(GraphDatabaseError):
    """图性能异常"""
    pass


class GraphValidationError(GraphDatabaseError):
    """图验证异常"""
    pass


class GraphTraversalError(GraphDatabaseError):
    """图遍历异常"""
    pass


class GraphAnalysisError(GraphDatabaseError):
    """图分析异常"""
    pass


class RelationLinkingError(RelationError):
    """关系链接异常"""
    pass


# 向量相关异常
class VectorError(GraphRAGException):
    """向量处理异常"""
    pass


class VectorEmbeddingError(VectorError):
    """向量嵌入异常"""
    pass


class VectorSearchError(VectorError):
    """向量搜索异常"""
    pass


class VectorIndexError(VectorError):
    """向量索引异常"""
    pass


class VectorDimensionError(VectorError):
    """向量维度异常"""
    pass


# 文本处理相关异常
class TextProcessingError(GraphRAGException):
    """文本处理异常"""
    pass


class TextChunkingError(TextProcessingError):
    """文本分块异常"""
    pass


class TextCleaningError(TextProcessingError):
    """文本清理异常"""
    pass


class TextNormalizationError(TextProcessingError):
    """文本标准化异常"""
    pass


class LanguageDetectionError(TextProcessingError):
    """语言检测异常"""
    pass


# API 相关异常
class APIError(GraphRAGException):
    """API 异常"""
    pass


class AuthenticationError(APIError):
    """认证异常"""
    pass


class AuthorizationError(APIError):
    """授权异常"""
    pass


class ValidationError(APIError):
    """验证异常"""
    pass


class RateLimitError(APIError):
    """频率限制异常"""
    pass


class ServiceUnavailableError(APIError):
    """服务不可用异常"""
    pass


# 搜索相关异常
class SearchError(GraphRAGException):
    """搜索异常"""
    pass


class SearchQueryError(SearchError):
    """搜索查询异常"""
    pass


class SearchIndexError(SearchError):
    """搜索索引异常"""
    pass


class SearchResultError(SearchError):
    """搜索结果异常"""
    pass


# RAG 相关异常
class RAGError(GraphRAGException):
    """RAG 异常"""
    pass


class RetrievalError(RAGError):
    """检索异常"""
    pass


class GenerationError(RAGError):
    """生成异常"""
    pass


class ContextError(RAGError):
    """上下文异常"""
    pass


# 配置相关异常
class ConfigurationError(GraphRAGException):
    """配置异常"""
    pass


class SettingsError(ConfigurationError):
    """设置异常"""
    pass


class EnvironmentError(ConfigurationError):
    """环境异常"""
    pass


# 外部服务异常
class ExternalServiceError(GraphRAGException):
    """外部服务异常"""
    pass


class LLMServiceError(ExternalServiceError):
    """LLM 服务异常"""
    pass


class EmbeddingServiceError(ExternalServiceError):
    """嵌入服务异常"""
    pass


class OCRServiceError(ExternalServiceError):
    """OCR 服务异常"""
    pass


# 缓存相关异常
class CacheError(GraphRAGException):
    """缓存异常"""
    pass


class CacheConnectionError(CacheError):
    """缓存连接异常"""
    pass


class CacheKeyError(CacheError):
    """缓存键异常"""
    pass


class CacheSerializationError(CacheError):
    """缓存序列化异常"""
    pass


# 任务相关异常
class TaskError(GraphRAGException):
    """任务异常"""
    pass


class TaskNotFoundError(TaskError):
    """任务不存在异常"""
    pass


class TaskExecutionError(TaskError):
    """任务执行异常"""
    pass


class TaskTimeoutError(TaskError):
    """任务超时异常"""
    pass


class TaskCancellationError(TaskError):
    """任务取消异常"""
    pass


# 监控相关异常
class MonitoringError(GraphRAGException):
    """监控异常"""
    pass


class MetricsError(MonitoringError):
    """指标异常"""
    pass


class HealthCheckError(MonitoringError):
    """健康检查异常"""
    pass


class AlertError(MonitoringError):
    """告警异常"""
    pass