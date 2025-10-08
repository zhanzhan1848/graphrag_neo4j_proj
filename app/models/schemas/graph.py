#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 图查询 API 模式
=======================

本模块定义了图查询相关的 API 模式（Pydantic 模型）。

模式说明：
- GraphQueryRequest: 图查询请求模式
- GraphQueryResponse: 图查询响应模式
- GraphNodeResponse: 图节点响应模式
- GraphRelationResponse: 图关系响应模式
- GraphPathResponse: 图路径响应模式
- GraphStatsResponse: 图统计响应模式

字段说明：
- cypher_query: Cypher 查询语句
- parameters: 查询参数
- nodes: 图节点列表
- relationships: 图关系列表
- paths: 图路径列表
- statistics: 图统计信息

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import Field, validator

from .base import BaseSchema


class GraphQueryType(str, Enum):
    """
    图查询类型枚举
    
    定义系统支持的图查询类型。
    """
    CYPHER = "cypher"           # Cypher 查询
    PATTERN = "pattern"         # 模式匹配
    PATH = "path"               # 路径查询
    NEIGHBORHOOD = "neighborhood"  # 邻域查询
    SUBGRAPH = "subgraph"       # 子图查询
    TRAVERSAL = "traversal"     # 遍历查询


class GraphOutputFormat(str, Enum):
    """
    图输出格式枚举
    
    定义图查询结果的输出格式。
    """
    JSON = "json"               # JSON 格式
    GRAPH = "graph"             # 图格式
    TABLE = "table"             # 表格格式
    TREE = "tree"               # 树形格式


class GraphQueryRequest(BaseSchema):
    """
    图查询请求模式
    
    用于执行图数据库查询。
    """
    query: str = Field(
        ...,
        description="图查询语句（Cypher 或自然语言）",
        min_length=1,
        max_length=5000
    )
    
    query_type: GraphQueryType = Field(
        default=GraphQueryType.CYPHER,
        description="查询类型"
    )
    
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="查询参数"
    )
    
    output_format: GraphOutputFormat = Field(
        default=GraphOutputFormat.GRAPH,
        description="输出格式"
    )
    
    limit: int = Field(
        default=100,
        description="结果限制数量",
        ge=1,
        le=1000
    )
    
    include_metadata: bool = Field(
        default=True,
        description="是否包含节点和关系的元数据"
    )
    
    include_statistics: bool = Field(
        default=False,
        description="是否包含查询统计信息"
    )
    
    timeout_seconds: int = Field(
        default=30,
        description="查询超时时间（秒）",
        ge=1,
        le=300
    )


class GraphNodeResponse(BaseSchema):
    """
    图节点响应模式
    
    表示图中的一个节点。
    """
    id: Union[UUID, str, int] = Field(
        ...,
        description="节点唯一标识符"
    )
    
    labels: List[str] = Field(
        ...,
        description="节点标签列表"
    )
    
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="节点属性"
    )
    
    # 扩展信息
    degree: Optional[int] = Field(
        None,
        description="节点度数（连接的关系数量）",
        ge=0
    )
    
    in_degree: Optional[int] = Field(
        None,
        description="入度（指向该节点的关系数量）",
        ge=0
    )
    
    out_degree: Optional[int] = Field(
        None,
        description="出度（从该节点出发的关系数量）",
        ge=0
    )
    
    centrality_scores: Optional[Dict[str, float]] = Field(
        None,
        description="中心性分数"
    )


class GraphRelationResponse(BaseSchema):
    """
    图关系响应模式
    
    表示图中的一个关系。
    """
    id: Union[UUID, str, int] = Field(
        ...,
        description="关系唯一标识符"
    )
    
    type: str = Field(
        ...,
        description="关系类型"
    )
    
    start_node_id: Union[UUID, str, int] = Field(
        ...,
        description="起始节点ID"
    )
    
    end_node_id: Union[UUID, str, int] = Field(
        ...,
        description="结束节点ID"
    )
    
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="关系属性"
    )
    
    # 关联节点信息
    start_node: Optional[GraphNodeResponse] = Field(
        None,
        description="起始节点信息"
    )
    
    end_node: Optional[GraphNodeResponse] = Field(
        None,
        description="结束节点信息"
    )


class GraphPathResponse(BaseSchema):
    """
    图路径响应模式
    
    表示图中的一条路径。
    """
    length: int = Field(
        ...,
        description="路径长度（关系数量）",
        ge=0
    )
    
    nodes: List[GraphNodeResponse] = Field(
        ...,
        description="路径中的节点列表"
    )
    
    relationships: List[GraphRelationResponse] = Field(
        ...,
        description="路径中的关系列表"
    )
    
    start_node_id: Union[UUID, str, int] = Field(
        ...,
        description="路径起始节点ID"
    )
    
    end_node_id: Union[UUID, str, int] = Field(
        ...,
        description="路径结束节点ID"
    )
    
    total_weight: Optional[float] = Field(
        None,
        description="路径总权重"
    )
    
    path_score: Optional[float] = Field(
        None,
        description="路径评分"
    )


class GraphStatsResponse(BaseSchema):
    """
    图统计响应模式
    
    提供图的统计信息。
    """
    total_nodes: int = Field(
        ...,
        description="总节点数量",
        ge=0
    )
    
    total_relationships: int = Field(
        ...,
        description="总关系数量",
        ge=0
    )
    
    node_labels: Dict[str, int] = Field(
        default_factory=dict,
        description="节点标签统计"
    )
    
    relationship_types: Dict[str, int] = Field(
        default_factory=dict,
        description="关系类型统计"
    )
    
    density: Optional[float] = Field(
        None,
        description="图密度",
        ge=0.0,
        le=1.0
    )
    
    average_degree: Optional[float] = Field(
        None,
        description="平均度数",
        ge=0.0
    )
    
    connected_components: Optional[int] = Field(
        None,
        description="连通分量数量",
        ge=0
    )
    
    diameter: Optional[int] = Field(
        None,
        description="图直径（最长最短路径）",
        ge=0
    )
    
    clustering_coefficient: Optional[float] = Field(
        None,
        description="聚类系数",
        ge=0.0,
        le=1.0
    )


class GraphNeighborhoodRequest(BaseSchema):
    """
    图邻域查询请求模式
    
    用于查询节点的邻域信息。
    """
    node_id: Union[UUID, str, int] = Field(
        ...,
        description="中心节点ID"
    )
    
    max_depth: int = Field(
        default=2,
        description="最大遍历深度",
        ge=1,
        le=5
    )
    
    relationship_types: Optional[List[str]] = Field(
        None,
        description="限制的关系类型列表"
    )
    
    node_labels: Optional[List[str]] = Field(
        None,
        description="限制的节点标签列表"
    )
    
    max_nodes: int = Field(
        default=100,
        description="最大返回节点数",
        ge=1,
        le=500
    )
    
    include_center_node: bool = Field(
        default=True,
        description="是否包含中心节点"
    )


class GraphSubgraphRequest(BaseSchema):
    """
    图子图查询请求模式
    
    用于提取图的子图。
    """
    node_ids: List[Union[UUID, str, int]] = Field(
        ...,
        description="子图节点ID列表",
        min_items=1,
        max_items=100
    )
    
    include_relationships: bool = Field(
        default=True,
        description="是否包含节点间的关系"
    )
    
    relationship_types: Optional[List[str]] = Field(
        None,
        description="限制的关系类型列表"
    )
    
    expand_depth: int = Field(
        default=0,
        description="扩展深度（包含邻近节点）",
        ge=0,
        le=3
    )


class GraphAnalysisRequest(BaseSchema):
    """
    图分析请求模式
    
    用于执行图分析算法。
    """
    algorithm: str = Field(
        ...,
        description="分析算法名称"
    )
    
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="算法参数"
    )
    
    node_filter: Optional[Dict[str, Any]] = Field(
        None,
        description="节点过滤条件"
    )
    
    relationship_filter: Optional[Dict[str, Any]] = Field(
        None,
        description="关系过滤条件"
    )
    
    @validator('algorithm')
    def validate_algorithm(cls, v):
        """验证算法名称"""
        allowed_algorithms = {
            'pagerank', 'betweenness_centrality', 'closeness_centrality',
            'degree_centrality', 'community_detection', 'shortest_path',
            'clustering_coefficient', 'connected_components'
        }
        if v not in allowed_algorithms:
            raise ValueError(f'algorithm must be one of {allowed_algorithms}')
        return v


class GraphAnalysisResponse(BaseSchema):
    """
    图分析响应模式
    
    返回图分析结果。
    """
    algorithm: str = Field(
        ...,
        description="分析算法名称"
    )
    
    execution_time_ms: float = Field(
        ...,
        description="算法执行时间（毫秒）",
        ge=0
    )
    
    results: Dict[str, Any] = Field(
        ...,
        description="分析结果"
    )
    
    node_scores: Optional[Dict[Union[UUID, str, int], float]] = Field(
        None,
        description="节点分数"
    )
    
    summary: Optional[Dict[str, Any]] = Field(
        None,
        description="结果摘要"
    )

class GraphQueryResponse(BaseSchema):
    """
    图查询响应模式
    
    返回图查询结果。
    """
    query: str = Field(
        ...,
        description="原始查询语句"
    )
    
    query_type: GraphQueryType = Field(
        ...,
        description="查询类型"
    )
    
    execution_time_ms: float = Field(
        ...,
        description="查询执行时间（毫秒）",
        ge=0
    )
    
    total_results: int = Field(
        ...,
        description="总结果数量",
        ge=0
    )
    
    nodes: List[GraphNodeResponse] = Field(
        default_factory=list,
        description="图节点列表"
    )
    
    relationships: List[GraphRelationResponse] = Field(
        default_factory=list,
        description="图关系列表"
    )
    
    paths: Optional[List[GraphPathResponse]] = Field(
        None,
        description="图路径列表"
    )
    
    statistics: Optional[GraphStatsResponse] = Field(
        None,
        description="查询统计信息"
    )
    
    raw_results: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="原始查询结果"
    )