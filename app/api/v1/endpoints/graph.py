#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 图查询 API 端点
=======================

本模块提供图查询相关的 API 端点，包括：
1. 图遍历 - 节点和关系的遍历查询
2. 路径查找 - 最短路径、所有路径查询
3. 社区检测 - 图中社区结构分析
4. 中心性分析 - 节点重要性分析
5. 相似性搜索 - 结构相似性查询
6. 图统计 - 图的统计信息和度量
7. 子图提取 - 基于条件的子图提取
8. 图可视化 - 图数据的可视化支持

所有端点都支持复杂的图查询和分析，并提供详细的结果和统计信息。

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import uuid

from app.core.logging import get_logger
from app.services.graph_service import GraphService
from app.services.graph_query_service import GraphQueryService
from app.utils.exceptions import (
    GraphQueryError,
    GraphConnectionError,
    ValidationError
)

logger = get_logger(__name__)

# 创建路由器
router = APIRouter()


# Pydantic 模型定义

class NodeQuery(BaseModel):
    """节点查询模型"""
    labels: Optional[List[str]] = Field(None, description="节点标签列表")
    properties: Optional[Dict[str, Any]] = Field(None, description="节点属性过滤条件")
    limit: int = Field(default=100, ge=1, le=10000, description="返回结果数量限制")
    skip: int = Field(default=0, ge=0, description="跳过的结果数量")
    order_by: Optional[str] = Field(None, description="排序字段")
    order_direction: str = Field(default="ASC", pattern="^(ASC|DESC)$", description="排序方向")


class RelationshipQuery(BaseModel):
    """关系查询模型"""
    types: Optional[List[str]] = Field(None, description="关系类型列表")
    properties: Optional[Dict[str, Any]] = Field(None, description="关系属性过滤条件")
    source_labels: Optional[List[str]] = Field(None, description="源节点标签")
    target_labels: Optional[List[str]] = Field(None, description="目标节点标签")
    limit: int = Field(default=100, ge=1, le=10000, description="返回结果数量限制")
    skip: int = Field(default=0, ge=0, description="跳过的结果数量")


class PathQuery(BaseModel):
    """路径查询模型"""
    source_id: str = Field(..., description="源节点ID")
    target_id: str = Field(..., description="目标节点ID")
    max_depth: int = Field(default=5, ge=1, le=10, description="最大路径深度")
    relationship_types: Optional[List[str]] = Field(None, description="允许的关系类型")
    direction: str = Field(default="BOTH", pattern="^(OUTGOING|INCOMING|BOTH)$", description="关系方向")
    find_all: bool = Field(default=False, description="是否查找所有路径")
    max_paths: int = Field(default=10, ge=1, le=100, description="最大路径数量")


class CommunityQuery(BaseModel):
    """社区检测查询模型"""
    algorithm: str = Field(default="louvain", pattern="^(louvain|leiden|label_propagation)$", description="社区检测算法")
    node_labels: Optional[List[str]] = Field(None, description="参与检测的节点标签")
    relationship_types: Optional[List[str]] = Field(None, description="参与检测的关系类型")
    resolution: float = Field(default=1.0, ge=0.1, le=10.0, description="分辨率参数")
    min_community_size: int = Field(default=3, ge=1, description="最小社区大小")
    max_communities: int = Field(default=100, ge=1, le=1000, description="最大社区数量")


class CentralityQuery(BaseModel):
    """中心性分析查询模型"""
    algorithm: str = Field(default="degree", pattern="^(degree|betweenness|closeness|pagerank|eigenvector)$", description="中心性算法")
    node_labels: Optional[List[str]] = Field(None, description="参与分析的节点标签")
    relationship_types: Optional[List[str]] = Field(None, description="参与分析的关系类型")
    direction: str = Field(default="BOTH", pattern="^(OUTGOING|INCOMING|BOTH)$", description="关系方向")
    top_k: int = Field(default=20, ge=1, le=1000, description="返回前K个节点")
    damping_factor: float = Field(default=0.85, ge=0.1, le=1.0, description="PageRank阻尼因子")


class SimilarityQuery(BaseModel):
    """相似性查询模型"""
    node_id: str = Field(..., description="参考节点ID")
    algorithm: str = Field(default="jaccard", pattern="^(jaccard|cosine|overlap|pearson)$", description="相似性算法")
    similarity_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="相似性阈值")
    top_k: int = Field(default=20, ge=1, le=1000, description="返回前K个相似节点")
    node_labels: Optional[List[str]] = Field(None, description="候选节点标签")
    relationship_types: Optional[List[str]] = Field(None, description="用于计算相似性的关系类型")


class SubgraphQuery(BaseModel):
    """子图查询模型"""
    center_node_id: str = Field(..., description="中心节点ID")
    max_depth: int = Field(default=2, ge=1, le=5, description="子图最大深度")
    max_nodes: int = Field(default=100, ge=1, le=1000, description="子图最大节点数")
    node_labels: Optional[List[str]] = Field(None, description="包含的节点标签")
    relationship_types: Optional[List[str]] = Field(None, description="包含的关系类型")
    direction: str = Field(default="BOTH", pattern="^(OUTGOING|INCOMING|BOTH)$", description="关系方向")
    include_properties: bool = Field(default=True, description="是否包含属性")


class CypherQuery(BaseModel):
    """Cypher查询模型"""
    query: str = Field(..., min_length=1, max_length=10000, description="Cypher查询语句")
    parameters: Optional[Dict[str, Any]] = Field(None, description="查询参数")
    limit: Optional[int] = Field(None, ge=1, le=10000, description="结果限制")
    explain: bool = Field(default=False, description="是否返回执行计划")
    profile: bool = Field(default=False, description="是否返回性能分析")
    
    @validator('query')
    def validate_query(cls, v):
        # 基本的安全检查，防止危险操作
        dangerous_keywords = ['DELETE', 'REMOVE', 'SET', 'CREATE', 'MERGE', 'DROP']
        query_upper = v.upper()
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                raise ValueError(f'查询中不允许包含 {keyword} 操作')
        return v


class GraphResponse(BaseModel):
    """图查询响应模型"""
    success: bool = Field(..., description="是否成功")
    query_id: str = Field(..., description="查询ID")
    message: str = Field(..., description="响应消息")
    data: Optional[Dict[str, Any]] = Field(None, description="查询结果数据")
    metadata: Optional[Dict[str, Any]] = Field(None, description="查询元数据")
    timestamp: str = Field(..., description="时间戳")


# 依赖注入

async def get_graph_service() -> GraphService:
    """获取图服务实例"""
    from app.services.graph_service import GraphService
    return GraphService()


async def get_graph_query_service() -> GraphQueryService:
    """获取图查询服务实例"""
    from app.services.graph_query_service import GraphQueryService
    return GraphQueryService()


# API 端点

@router.post("/nodes/search", response_model=GraphResponse, tags=["节点查询"])
async def search_nodes(
    query: NodeQuery,
    graph_service: GraphService = Depends(get_graph_service)
) -> GraphResponse:
    """
    搜索图中的节点
    
    Args:
        query: 节点查询条件
        graph_service: 图服务
        
    Returns:
        GraphResponse: 查询响应
        
    Example:
        ```json
        {
            "labels": ["Entity", "Person"],
            "properties": {"name": "张三"},
            "limit": 50,
            "order_by": "name"
        }
        ```
    """
    try:
        logger.info(f"开始节点搜索，标签: {query.labels}")
        
        query_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # 构建Cypher查询
        cypher_parts = []
        params = {}
        
        # 构建节点匹配模式
        if query.labels:
            label_str = ":".join(query.labels)
            cypher_parts.append(f"MATCH (n:{label_str})")
        else:
            cypher_parts.append("MATCH (n)")
        
        # 添加属性过滤
        if query.properties:
            where_conditions = []
            for key, value in query.properties.items():
                param_key = f"prop_{key}"
                where_conditions.append(f"n.{key} = ${param_key}")
                params[param_key] = value
            
            if where_conditions:
                cypher_parts.append(f"WHERE {' AND '.join(where_conditions)}")
        
        # 添加返回和排序
        cypher_parts.append("RETURN n")
        
        if query.order_by:
            cypher_parts.append(f"ORDER BY n.{query.order_by} {query.order_direction}")
        
        # 添加分页
        if query.skip > 0:
            cypher_parts.append(f"SKIP {query.skip}")
        cypher_parts.append(f"LIMIT {query.limit}")
        
        cypher_query = " ".join(cypher_parts)
        
        # 执行查询
        result = await graph_service.execute_cypher(cypher_query, params)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # 处理结果
        nodes = []
        for record in result.records:
            node = record["n"]
            node_dict = {
                "id": node.element_id,
                "labels": list(node.labels),
                "properties": dict(node)
            }
            nodes.append(node_dict)
        
        response_data = {
            "nodes": nodes,
            "total_count": len(nodes),
            "query": cypher_query,
            "parameters": params
        }
        
        metadata = {
            "processing_time": processing_time,
            "query_stats": result.summary.counters if hasattr(result, 'summary') else None
        }
        
        logger.info(f"节点搜索完成，查询ID: {query_id}，找到节点数: {len(nodes)}")
        
        return GraphResponse(
            success=True,
            query_id=query_id,
            message=f"找到 {len(nodes)} 个节点",
            data=response_data,
            metadata=metadata,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"节点搜索失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"节点搜索失败: {str(e)}"
        )


@router.post("/relationships/search", response_model=GraphResponse, tags=["关系查询"])
async def search_relationships(
    query: RelationshipQuery,
    graph_service: GraphService = Depends(get_graph_service)
) -> GraphResponse:
    """
    搜索图中的关系
    
    Args:
        query: 关系查询条件
        graph_service: 图服务
        
    Returns:
        GraphResponse: 查询响应
        
    Example:
        ```json
        {
            "types": ["KNOWS", "WORKS_FOR"],
            "properties": {"since": "2020"},
            "source_labels": ["Person"],
            "target_labels": ["Organization"],
            "limit": 50
        }
        ```
    """
    try:
        logger.info(f"开始关系搜索，类型: {query.types}")
        
        query_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # 构建Cypher查询
        cypher_parts = []
        params = {}
        
        # 构建关系匹配模式
        source_pattern = "s"
        if query.source_labels:
            source_pattern = f"s:{':'.join(query.source_labels)}"
        
        target_pattern = "t"
        if query.target_labels:
            target_pattern = f"t:{':'.join(query.target_labels)}"
        
        rel_pattern = "r"
        if query.types:
            rel_pattern = f"r:{':'.join(query.types)}"
        
        cypher_parts.append(f"MATCH ({source_pattern})-[{rel_pattern}]->({target_pattern})")
        
        # 添加属性过滤
        if query.properties:
            where_conditions = []
            for key, value in query.properties.items():
                param_key = f"rel_prop_{key}"
                where_conditions.append(f"r.{key} = ${param_key}")
                params[param_key] = value
            
            if where_conditions:
                cypher_parts.append(f"WHERE {' AND '.join(where_conditions)}")
        
        # 添加返回
        cypher_parts.append("RETURN s, r, t")
        
        # 添加分页
        if query.skip > 0:
            cypher_parts.append(f"SKIP {query.skip}")
        cypher_parts.append(f"LIMIT {query.limit}")
        
        cypher_query = " ".join(cypher_parts)
        
        # 执行查询
        result = await graph_service.execute_cypher(cypher_query, params)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # 处理结果
        relationships = []
        for record in result.records:
            source = record["s"]
            rel = record["r"]
            target = record["t"]
            
            rel_dict = {
                "id": rel.element_id,
                "type": rel.type,
                "properties": dict(rel),
                "source": {
                    "id": source.element_id,
                    "labels": list(source.labels),
                    "properties": dict(source)
                },
                "target": {
                    "id": target.element_id,
                    "labels": list(target.labels),
                    "properties": dict(target)
                }
            }
            relationships.append(rel_dict)
        
        response_data = {
            "relationships": relationships,
            "total_count": len(relationships),
            "query": cypher_query,
            "parameters": params
        }
        
        metadata = {
            "processing_time": processing_time,
            "query_stats": result.summary.counters if hasattr(result, 'summary') else None
        }
        
        logger.info(f"关系搜索完成，查询ID: {query_id}，找到关系数: {len(relationships)}")
        
        return GraphResponse(
            success=True,
            query_id=query_id,
            message=f"找到 {len(relationships)} 个关系",
            data=response_data,
            metadata=metadata,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"关系搜索失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"关系搜索失败: {str(e)}"
        )


@router.post("/paths/find", response_model=GraphResponse, tags=["路径查询"])
async def find_paths(
    query: PathQuery,
    graph_query_service: GraphQueryService = Depends(get_graph_query_service)
) -> GraphResponse:
    """
    查找节点间的路径
    
    Args:
        query: 路径查询条件
        graph_query_service: 图查询服务
        
    Returns:
        GraphResponse: 查询响应
        
    Example:
        ```json
        {
            "source_id": "node_123",
            "target_id": "node_456",
            "max_depth": 3,
            "relationship_types": ["KNOWS", "WORKS_WITH"],
            "find_all": false,
            "max_paths": 5
        }
        ```
    """
    try:
        logger.info(f"开始路径查找，源节点: {query.source_id}，目标节点: {query.target_id}")
        
        query_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # 执行路径查找
        if query.find_all:
            result = await graph_query_service.find_all_paths(
                source_id=query.source_id,
                target_id=query.target_id,
                max_depth=query.max_depth,
                relationship_types=query.relationship_types,
                direction=query.direction,
                max_paths=query.max_paths
            )
        else:
            result = await graph_query_service.find_shortest_path(
                source_id=query.source_id,
                target_id=query.target_id,
                max_depth=query.max_depth,
                relationship_types=query.relationship_types,
                direction=query.direction
            )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # 处理结果
        paths_data = []
        if hasattr(result, 'paths'):
            for path in result.paths:
                path_dict = {
                    "length": path.length,
                    "nodes": [{"id": node.element_id, "labels": list(node.labels), "properties": dict(node)} 
                             for node in path.nodes],
                    "relationships": [{"id": rel.element_id, "type": rel.type, "properties": dict(rel)} 
                                    for rel in path.relationships],
                    "weight": getattr(path, 'weight', None)
                }
                paths_data.append(path_dict)
        
        response_data = {
            "paths": paths_data,
            "total_count": len(paths_data),
            "source_id": query.source_id,
            "target_id": query.target_id,
            "search_stats": result.stats if hasattr(result, 'stats') else None
        }
        
        metadata = {
            "processing_time": processing_time,
            "max_depth": query.max_depth,
            "find_all": query.find_all
        }
        
        logger.info(f"路径查找完成，查询ID: {query_id}，找到路径数: {len(paths_data)}")
        
        return GraphResponse(
            success=True,
            query_id=query_id,
            message=f"找到 {len(paths_data)} 条路径",
            data=response_data,
            metadata=metadata,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"路径查找失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"路径查找失败: {str(e)}"
        )


@router.post("/communities/detect", response_model=GraphResponse, tags=["社区检测"])
async def detect_communities(
    query: CommunityQuery,
    graph_query_service: GraphQueryService = Depends(get_graph_query_service)
) -> GraphResponse:
    """
    检测图中的社区结构
    
    Args:
        query: 社区检测查询条件
        graph_query_service: 图查询服务
        
    Returns:
        GraphResponse: 查询响应
        
    Example:
        ```json
        {
            "algorithm": "louvain",
            "node_labels": ["Person", "Organization"],
            "relationship_types": ["KNOWS", "WORKS_FOR"],
            "resolution": 1.0,
            "min_community_size": 5
        }
        ```
    """
    try:
        logger.info(f"开始社区检测，算法: {query.algorithm}")
        
        query_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # 执行社区检测
        result = await graph_query_service.detect_communities(
            algorithm=query.algorithm,
            node_labels=query.node_labels,
            relationship_types=query.relationship_types,
            resolution=query.resolution,
            min_community_size=query.min_community_size,
            max_communities=query.max_communities
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # 处理结果
        communities_data = []
        if hasattr(result, 'communities'):
            for community in result.communities:
                community_dict = {
                    "id": community.id,
                    "size": community.size,
                    "nodes": [{"id": node.element_id, "labels": list(node.labels)} 
                             for node in community.nodes],
                    "density": getattr(community, 'density', None),
                    "modularity": getattr(community, 'modularity', None)
                }
                communities_data.append(community_dict)
        
        response_data = {
            "communities": communities_data,
            "total_communities": len(communities_data),
            "algorithm": query.algorithm,
            "detection_stats": result.stats if hasattr(result, 'stats') else None
        }
        
        metadata = {
            "processing_time": processing_time,
            "resolution": query.resolution,
            "min_community_size": query.min_community_size
        }
        
        logger.info(f"社区检测完成，查询ID: {query_id}，检测到社区数: {len(communities_data)}")
        
        return GraphResponse(
            success=True,
            query_id=query_id,
            message=f"检测到 {len(communities_data)} 个社区",
            data=response_data,
            metadata=metadata,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"社区检测失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"社区检测失败: {str(e)}"
        )


@router.post("/centrality/analyze", response_model=GraphResponse, tags=["中心性分析"])
async def analyze_centrality(
    query: CentralityQuery,
    graph_query_service: GraphQueryService = Depends(get_graph_query_service)
) -> GraphResponse:
    """
    分析节点的中心性
    
    Args:
        query: 中心性分析查询条件
        graph_query_service: 图查询服务
        
    Returns:
        GraphResponse: 查询响应
        
    Example:
        ```json
        {
            "algorithm": "pagerank",
            "node_labels": ["Person"],
            "relationship_types": ["KNOWS"],
            "top_k": 10,
            "damping_factor": 0.85
        }
        ```
    """
    try:
        logger.info(f"开始中心性分析，算法: {query.algorithm}")
        
        query_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # 执行中心性分析
        result = await graph_query_service.analyze_centrality(
            algorithm=query.algorithm,
            node_labels=query.node_labels,
            relationship_types=query.relationship_types,
            direction=query.direction,
            top_k=query.top_k,
            damping_factor=query.damping_factor
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # 处理结果
        centrality_data = []
        if hasattr(result, 'centrality_scores'):
            for score in result.centrality_scores:
                score_dict = {
                    "node": {
                        "id": score.node.element_id,
                        "labels": list(score.node.labels),
                        "properties": dict(score.node)
                    },
                    "score": score.score,
                    "rank": score.rank
                }
                centrality_data.append(score_dict)
        
        response_data = {
            "centrality_scores": centrality_data,
            "total_nodes": len(centrality_data),
            "algorithm": query.algorithm,
            "analysis_stats": result.stats if hasattr(result, 'stats') else None
        }
        
        metadata = {
            "processing_time": processing_time,
            "top_k": query.top_k,
            "damping_factor": query.damping_factor if query.algorithm == "pagerank" else None
        }
        
        logger.info(f"中心性分析完成，查询ID: {query_id}，分析节点数: {len(centrality_data)}")
        
        return GraphResponse(
            success=True,
            query_id=query_id,
            message=f"分析了 {len(centrality_data)} 个节点的中心性",
            data=response_data,
            metadata=metadata,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"中心性分析失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"中心性分析失败: {str(e)}"
        )


@router.post("/similarity/search", response_model=GraphResponse, tags=["相似性搜索"])
async def search_similarity(
    query: SimilarityQuery,
    graph_query_service: GraphQueryService = Depends(get_graph_query_service)
) -> GraphResponse:
    """
    搜索相似的节点
    
    Args:
        query: 相似性搜索查询条件
        graph_query_service: 图查询服务
        
    Returns:
        GraphResponse: 查询响应
        
    Example:
        ```json
        {
            "node_id": "node_123",
            "algorithm": "jaccard",
            "similarity_threshold": 0.3,
            "top_k": 10,
            "node_labels": ["Person"]
        }
        ```
    """
    try:
        logger.info(f"开始相似性搜索，参考节点: {query.node_id}，算法: {query.algorithm}")
        
        query_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # 执行相似性搜索
        result = await graph_query_service.search_similarity(
            node_id=query.node_id,
            algorithm=query.algorithm,
            similarity_threshold=query.similarity_threshold,
            top_k=query.top_k,
            node_labels=query.node_labels,
            relationship_types=query.relationship_types
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # 处理结果
        similarity_data = []
        if hasattr(result, 'similar_nodes'):
            for similar in result.similar_nodes:
                similar_dict = {
                    "node": {
                        "id": similar.node.element_id,
                        "labels": list(similar.node.labels),
                        "properties": dict(similar.node)
                    },
                    "similarity_score": similar.similarity_score,
                    "common_neighbors": getattr(similar, 'common_neighbors', None),
                    "rank": similar.rank
                }
                similarity_data.append(similar_dict)
        
        response_data = {
            "similar_nodes": similarity_data,
            "total_count": len(similarity_data),
            "reference_node_id": query.node_id,
            "algorithm": query.algorithm,
            "search_stats": result.stats if hasattr(result, 'stats') else None
        }
        
        metadata = {
            "processing_time": processing_time,
            "similarity_threshold": query.similarity_threshold,
            "top_k": query.top_k
        }
        
        logger.info(f"相似性搜索完成，查询ID: {query_id}，找到相似节点数: {len(similarity_data)}")
        
        return GraphResponse(
            success=True,
            query_id=query_id,
            message=f"找到 {len(similarity_data)} 个相似节点",
            data=response_data,
            metadata=metadata,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"相似性搜索失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"相似性搜索失败: {str(e)}"
        )


@router.post("/subgraph/extract", response_model=GraphResponse, tags=["子图提取"])
async def extract_subgraph(
    query: SubgraphQuery,
    graph_query_service: GraphQueryService = Depends(get_graph_query_service)
) -> GraphResponse:
    """
    提取子图
    
    Args:
        query: 子图提取查询条件
        graph_query_service: 图查询服务
        
    Returns:
        GraphResponse: 查询响应
        
    Example:
        ```json
        {
            "center_node_id": "node_123",
            "max_depth": 2,
            "max_nodes": 50,
            "node_labels": ["Person", "Organization"],
            "relationship_types": ["KNOWS", "WORKS_FOR"],
            "include_properties": true
        }
        ```
    """
    try:
        logger.info(f"开始子图提取，中心节点: {query.center_node_id}，最大深度: {query.max_depth}")
        
        query_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # 执行子图提取
        result = await graph_query_service.extract_subgraph(
            center_node_id=query.center_node_id,
            max_depth=query.max_depth,
            max_nodes=query.max_nodes,
            node_labels=query.node_labels,
            relationship_types=query.relationship_types,
            direction=query.direction
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # 处理结果
        nodes_data = []
        relationships_data = []
        
        if hasattr(result, 'subgraph'):
            subgraph = result.subgraph
            
            # 处理节点
            for node in subgraph.nodes:
                node_dict = {
                    "id": node.element_id,
                    "labels": list(node.labels)
                }
                if query.include_properties:
                    node_dict["properties"] = dict(node)
                nodes_data.append(node_dict)
            
            # 处理关系
            for rel in subgraph.relationships:
                rel_dict = {
                    "id": rel.element_id,
                    "type": rel.type,
                    "source_id": rel.start_node.element_id,
                    "target_id": rel.end_node.element_id
                }
                if query.include_properties:
                    rel_dict["properties"] = dict(rel)
                relationships_data.append(rel_dict)
        
        response_data = {
            "nodes": nodes_data,
            "relationships": relationships_data,
            "node_count": len(nodes_data),
            "relationship_count": len(relationships_data),
            "center_node_id": query.center_node_id,
            "extraction_stats": result.stats if hasattr(result, 'stats') else None
        }
        
        metadata = {
            "processing_time": processing_time,
            "max_depth": query.max_depth,
            "max_nodes": query.max_nodes
        }
        
        logger.info(f"子图提取完成，查询ID: {query_id}，节点数: {len(nodes_data)}，关系数: {len(relationships_data)}")
        
        return GraphResponse(
            success=True,
            query_id=query_id,
            message=f"提取子图包含 {len(nodes_data)} 个节点和 {len(relationships_data)} 个关系",
            data=response_data,
            metadata=metadata,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"子图提取失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"子图提取失败: {str(e)}"
        )


@router.post("/cypher/execute", response_model=GraphResponse, tags=["Cypher查询"])
async def execute_cypher(
    query: CypherQuery,
    graph_service: GraphService = Depends(get_graph_service)
) -> GraphResponse:
    """
    执行自定义Cypher查询
    
    Args:
        query: Cypher查询
        graph_service: 图服务
        
    Returns:
        GraphResponse: 查询响应
        
    Example:
        ```json
        {
            "query": "MATCH (n:Person) WHERE n.age > $min_age RETURN n LIMIT 10",
            "parameters": {"min_age": 25},
            "explain": false,
            "profile": false
        }
        ```
    """
    try:
        logger.info(f"开始执行Cypher查询")
        
        query_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # 执行Cypher查询
        result = await graph_service.execute_cypher(
            query.query,
            query.parameters or {}
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # 处理结果
        records_data = []
        for record in result.records:
            record_dict = {}
            for key in record.keys():
                value = record[key]
                # 处理Neo4j对象
                if hasattr(value, 'element_id'):  # Node or Relationship
                    if hasattr(value, 'labels'):  # Node
                        record_dict[key] = {
                            "id": value.element_id,
                            "labels": list(value.labels),
                            "properties": dict(value)
                        }
                    else:  # Relationship
                        record_dict[key] = {
                            "id": value.element_id,
                            "type": value.type,
                            "properties": dict(value)
                        }
                else:
                    record_dict[key] = value
            records_data.append(record_dict)
        
        response_data = {
            "records": records_data,
            "total_count": len(records_data),
            "query": query.query,
            "parameters": query.parameters
        }
        
        metadata = {
            "processing_time": processing_time,
            "query_stats": result.summary.counters if hasattr(result, 'summary') else None
        }
        
        # 添加执行计划或性能分析
        if query.explain or query.profile:
            if hasattr(result, 'summary') and hasattr(result.summary, 'plan'):
                metadata["execution_plan"] = result.summary.plan
            if hasattr(result, 'summary') and hasattr(result.summary, 'profile'):
                metadata["performance_profile"] = result.summary.profile
        
        logger.info(f"Cypher查询执行完成，查询ID: {query_id}，返回记录数: {len(records_data)}")
        
        return GraphResponse(
            success=True,
            query_id=query_id,
            message=f"查询返回 {len(records_data)} 条记录",
            data=response_data,
            metadata=metadata,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Cypher查询执行失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cypher查询执行失败: {str(e)}"
        )


@router.get("/stats", tags=["统计信息"])
async def get_graph_stats(
    graph_service: GraphService = Depends(get_graph_service)
) -> Dict[str, Any]:
    """
    获取图统计信息
    
    Args:
        graph_service: 图服务
        
    Returns:
        Dict[str, Any]: 图统计信息
        
    Example:
        ```json
        {
            "node_count": 10000,
            "relationship_count": 25000,
            "node_labels": ["Person", "Organization", "Document"],
            "relationship_types": ["KNOWS", "WORKS_FOR", "MENTIONS"]
        }
        ```
    """
    try:
        logger.info("获取图统计信息")
        
        # 获取图统计
        stats = await graph_service.get_graph_stats()
        
        return {
            "node_count": stats.node_count,
            "relationship_count": stats.relationship_count,
            "node_labels": stats.node_labels,
            "relationship_types": stats.relationship_types,
            "database_info": stats.database_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取图统计信息失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取图统计信息失败: {str(e)}"
        )


@router.get("/health", tags=["健康检查"])
async def health_check(
    graph_service: GraphService = Depends(get_graph_service),
    graph_query_service: GraphQueryService = Depends(get_graph_query_service)
) -> Dict[str, Any]:
    """
    图查询服务健康检查
    
    Args:
        graph_service: 图服务
        graph_query_service: 图查询服务
        
    Returns:
        Dict[str, Any]: 健康状态
    """
    try:
        logger.info("执行图查询服务健康检查")
        
        # 检查图服务
        graph_health = await graph_service.health_check()
        
        # 检查图查询服务
        query_health = await graph_query_service.health_check()
        
        # 判断整体健康状态
        overall_status = "healthy"
        if (graph_health.get("status") != "healthy" or 
            query_health.get("status") != "healthy"):
            overall_status = "unhealthy"
        
        health_status = {
            "status": overall_status,
            "services": {
                "graph_service": graph_health.get("status", "unknown"),
                "graph_query_service": query_health.get("status", "unknown")
            },
            "details": {
                "graph_service": graph_health,
                "graph_query_service": query_health
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }