#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG API 模式包
==================

本模块定义了 GraphRAG 系统的 API 模式（Pydantic 模型）。

模块结构：
- base: 基础模式定义
- documents: 文档相关的请求/响应模式
- chunks: 文本块相关的模式
- entities: 实体相关的模式
- relations: 关系相关的模式
- images: 图像相关的模式
- search: 搜索相关的模式
- graph: 图查询相关的模式

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from .base import (
    BaseSchema,
    PaginationParams,
    PaginatedResponse,
    StatusResponse,
    ErrorResponse
)

from .documents import (
    DocumentCreate,
    DocumentUpdate,
    DocumentResponse,
    DocumentStatus,
    DocumentUploadResponse,
    DocumentListResponse
)

from .chunks import (
    ChunkCreate,
    ChunkUpdate,
    ChunkResponse,
    ChunkListResponse,
    ChunkSearchRequest,
    ChunkSearchResponse
)

from .entities import (
    EntityCreate,
    EntityUpdate,
    EntityResponse,
    EntityListResponse,
    EntitySearchRequest,
    EntitySearchResponse
)

from .relations import (
    RelationCreate,
    RelationUpdate,
    RelationResponse,
    RelationListResponse,
    RelationSearchRequest,
    RelationSearchResponse
)

from .images import (
    ImageCreate,
    ImageUpdate,
    ImageResponse,
    ImageListResponse,
    ImageUploadResponse,
    ImageAnalysisResponse
)

from .search import (
    SearchRequest,
    SearchResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
    HybridSearchRequest,
    HybridSearchResponse
)

from .graph import (
    GraphQueryRequest,
    GraphQueryResponse,
    GraphNodeResponse,
    GraphRelationResponse,
    GraphPathResponse,
    GraphStatsResponse
)

__all__ = [
    # 基础模式
    "BaseSchema",
    "PaginationParams", 
    "PaginatedResponse",
    "StatusResponse",
    "ErrorResponse",
    
    # 文档模式
    "DocumentCreate",
    "DocumentUpdate", 
    "DocumentResponse",
    "DocumentStatus",
    "DocumentUploadResponse",
    "DocumentListResponse",
    
    # 文本块模式
    "ChunkCreate",
    "ChunkUpdate",
    "ChunkResponse", 
    "ChunkListResponse",
    "ChunkSearchRequest",
    "ChunkSearchResponse",
    
    # 实体模式
    "EntityCreate",
    "EntityUpdate",
    "EntityResponse",
    "EntityListResponse", 
    "EntitySearchRequest",
    "EntitySearchResponse",
    
    # 关系模式
    "RelationCreate",
    "RelationUpdate",
    "RelationResponse",
    "RelationListResponse",
    "RelationSearchRequest", 
    "RelationSearchResponse",
    
    # 图像模式
    "ImageCreate",
    "ImageUpdate",
    "ImageResponse",
    "ImageListResponse",
    "ImageUploadResponse",
    "ImageAnalysisResponse",
    
    # 搜索模式
    "SearchRequest",
    "SearchResponse",
    "SemanticSearchRequest",
    "SemanticSearchResponse", 
    "HybridSearchRequest",
    "HybridSearchResponse",
    
    # 图查询模式
    "GraphQueryRequest",
    "GraphQueryResponse",
    "GraphNodeResponse",
    "GraphRelationResponse",
    "GraphPathResponse",
    "GraphStatsResponse"
]