#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 图查询和遍历服务
========================

本模块实现了图数据库的高级查询和遍历功能。

服务功能：
- 路径查找和分析
- 社区检测和聚类
- 中心性分析
- 图遍历算法
- 子图提取
- 图模式匹配
- 相似性查询
- 图统计分析

支持的查询类型：
- 最短路径查询
- 所有路径查询
- 深度优先遍历
- 广度优先遍历
- PageRank 算法
- 社区检测算法
- 相似节点查找
- 关系强度分析

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict, deque
from uuid import UUID, uuid4
import math

from app.services.graph_service import (
    GraphService, 
    GraphNode, 
    GraphRelationship, 
    GraphPath,
    GraphConnectionManager
)
from app.utils.exceptions import (
    GraphQueryError,
    GraphTraversalError,
    GraphAnalysisError
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PathResult:
    """路径查询结果"""
    paths: List[GraphPath]
    total_paths: int
    shortest_path_length: Optional[int]
    longest_path_length: Optional[int]
    average_path_length: Optional[float]
    query_time: float


@dataclass
class CommunityResult:
    """社区检测结果"""
    communities: List[List[str]]  # 每个社区包含的节点ID列表
    modularity: float
    total_communities: int
    largest_community_size: int
    smallest_community_size: int
    average_community_size: float


@dataclass
class CentralityResult:
    """中心性分析结果"""
    node_scores: Dict[str, float]  # 节点ID -> 中心性分数
    top_nodes: List[Tuple[str, float]]  # 排序后的前N个节点
    algorithm: str
    total_nodes: int
    max_score: float
    min_score: float
    average_score: float


@dataclass
class SimilarityResult:
    """相似性查询结果"""
    similar_nodes: List[Tuple[str, float]]  # 节点ID -> 相似度分数
    query_node_id: str
    similarity_threshold: float
    algorithm: str
    total_similar_nodes: int


@dataclass
class SubgraphResult:
    """子图提取结果"""
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]
    total_nodes: int
    total_relationships: int
    density: float  # 图密度
    diameter: Optional[int]  # 图直径


class PathFinder:
    """
    路径查找器
    
    实现各种路径查找算法。
    """
    
    def __init__(self, graph_service: GraphService):
        """
        初始化路径查找器
        
        Args:
            graph_service: 图服务实例
        """
        self.graph_service = graph_service
    
    async def find_shortest_path(
        self,
        start_node_id: str,
        end_node_id: str,
        max_depth: int = 10,
        relationship_types: Optional[List[str]] = None
    ) -> Optional[GraphPath]:
        """
        查找最短路径
        
        Args:
            start_node_id: 起始节点ID
            end_node_id: 结束节点ID
            max_depth: 最大搜索深度
            relationship_types: 关系类型过滤
            
        Returns:
            Optional[GraphPath]: 最短路径，如果不存在则返回None
        """
        try:
            # 构建 Cypher 查询
            rel_filter = ""
            if relationship_types:
                rel_types = "|".join(relationship_types)
                rel_filter = f":{rel_types}"
            
            query = f"""
            MATCH path = shortestPath((start {{id: $start_id}})-[{rel_filter}*1..{max_depth}]-(end {{id: $end_id}}))
            RETURN path
            """
            
            params = {
                "start_id": start_node_id,
                "end_id": end_node_id
            }
            
            result = await self.graph_service.execute_cypher(query, params)
            
            if not result:
                return None
            
            # 解析路径
            path_data = result[0]["path"]
            return await self._parse_path(path_data)
            
        except Exception as e:
            logger.error(f"查找最短路径失败: {str(e)}")
            raise GraphTraversalError(f"查找最短路径失败: {str(e)}")
    
    async def find_all_paths(
        self,
        start_node_id: str,
        end_node_id: str,
        max_depth: int = 5,
        relationship_types: Optional[List[str]] = None,
        limit: int = 100
    ) -> PathResult:
        """
        查找所有路径
        
        Args:
            start_node_id: 起始节点ID
            end_node_id: 结束节点ID
            max_depth: 最大搜索深度
            relationship_types: 关系类型过滤
            limit: 结果限制数量
            
        Returns:
            PathResult: 路径查询结果
        """
        try:
            start_time = time.time()
            
            # 构建 Cypher 查询
            rel_filter = ""
            if relationship_types:
                rel_types = "|".join(relationship_types)
                rel_filter = f":{rel_types}"
            
            query = f"""
            MATCH path = (start {{id: $start_id}})-[{rel_filter}*1..{max_depth}]-(end {{id: $end_id}})
            RETURN path
            LIMIT {limit}
            """
            
            params = {
                "start_id": start_node_id,
                "end_id": end_node_id
            }
            
            result = await self.graph_service.execute_cypher(query, params)
            
            # 解析所有路径
            paths = []
            for record in result:
                path = await self._parse_path(record["path"])
                if path:
                    paths.append(path)
            
            # 计算统计信息
            query_time = time.time() - start_time
            path_lengths = [path.length for path in paths]
            
            return PathResult(
                paths=paths,
                total_paths=len(paths),
                shortest_path_length=min(path_lengths) if path_lengths else None,
                longest_path_length=max(path_lengths) if path_lengths else None,
                average_path_length=sum(path_lengths) / len(path_lengths) if path_lengths else None,
                query_time=query_time
            )
            
        except Exception as e:
            logger.error(f"查找所有路径失败: {str(e)}")
            raise GraphTraversalError(f"查找所有路径失败: {str(e)}")
    
    async def find_paths_with_constraints(
        self,
        start_node_id: str,
        end_node_id: str,
        node_constraints: Optional[Dict[str, Any]] = None,
        relationship_constraints: Optional[Dict[str, Any]] = None,
        max_depth: int = 5,
        limit: int = 50
    ) -> PathResult:
        """
        带约束条件的路径查找
        
        Args:
            start_node_id: 起始节点ID
            end_node_id: 结束节点ID
            node_constraints: 节点约束条件
            relationship_constraints: 关系约束条件
            max_depth: 最大搜索深度
            limit: 结果限制数量
            
        Returns:
            PathResult: 路径查询结果
        """
        try:
            start_time = time.time()
            
            # 构建约束条件
            where_clauses = []
            params = {
                "start_id": start_node_id,
                "end_id": end_node_id
            }
            
            # 节点约束
            if node_constraints:
                for key, value in node_constraints.items():
                    param_name = f"node_{key}"
                    where_clauses.append(f"ALL(n IN nodes(path) WHERE n.{key} = ${param_name})")
                    params[param_name] = value
            
            # 关系约束
            if relationship_constraints:
                for key, value in relationship_constraints.items():
                    param_name = f"rel_{key}"
                    where_clauses.append(f"ALL(r IN relationships(path) WHERE r.{key} = ${param_name})")
                    params[param_name] = value
            
            # 构建查询
            where_clause = ""
            if where_clauses:
                where_clause = f"WHERE {' AND '.join(where_clauses)}"
            
            query = f"""
            MATCH path = (start {{id: $start_id}})-[*1..{max_depth}]-(end {{id: $end_id}})
            {where_clause}
            RETURN path
            LIMIT {limit}
            """
            
            result = await self.graph_service.execute_cypher(query, params)
            
            # 解析路径
            paths = []
            for record in result:
                path = await self._parse_path(record["path"])
                if path:
                    paths.append(path)
            
            # 计算统计信息
            query_time = time.time() - start_time
            path_lengths = [path.length for path in paths]
            
            return PathResult(
                paths=paths,
                total_paths=len(paths),
                shortest_path_length=min(path_lengths) if path_lengths else None,
                longest_path_length=max(path_lengths) if path_lengths else None,
                average_path_length=sum(path_lengths) / len(path_lengths) if path_lengths else None,
                query_time=query_time
            )
            
        except Exception as e:
            logger.error(f"带约束条件的路径查找失败: {str(e)}")
            raise GraphTraversalError(f"带约束条件的路径查找失败: {str(e)}")
    
    async def _parse_path(self, path_data: Any) -> Optional[GraphPath]:
        """
        解析路径数据
        
        Args:
            path_data: Neo4j 路径数据
            
        Returns:
            Optional[GraphPath]: 解析后的路径
        """
        try:
            # 这里需要根据 Neo4j Python 驱动的实际返回格式来解析
            # 由于这是示例代码，我们创建一个简化的解析逻辑
            
            nodes = []
            relationships = []
            
            # 假设 path_data 有 nodes 和 relationships 属性
            if hasattr(path_data, 'nodes') and hasattr(path_data, 'relationships'):
                for node_data in path_data.nodes:
                    node = GraphNode(
                        id=node_data.get("id", str(uuid4())),
                        labels=list(node_data.labels) if hasattr(node_data, 'labels') else [],
                        properties=dict(node_data)
                    )
                    nodes.append(node)
                
                for rel_data in path_data.relationships:
                    relationship = GraphRelationship(
                        id=rel_data.get("id", str(uuid4())),
                        type=rel_data.type if hasattr(rel_data, 'type') else "UNKNOWN",
                        start_node_id=str(rel_data.start_node.id) if hasattr(rel_data, 'start_node') else "",
                        end_node_id=str(rel_data.end_node.id) if hasattr(rel_data, 'end_node') else "",
                        properties=dict(rel_data)
                    )
                    relationships.append(relationship)
            
            return GraphPath(
                nodes=nodes,
                relationships=relationships,
                length=len(relationships)
            )
            
        except Exception as e:
            logger.error(f"解析路径数据失败: {str(e)}")
            return None


class CommunityDetector:
    """
    社区检测器
    
    实现图的社区检测算法。
    """
    
    def __init__(self, graph_service: GraphService):
        """
        初始化社区检测器
        
        Args:
            graph_service: 图服务实例
        """
        self.graph_service = graph_service
    
    async def detect_communities_louvain(
        self,
        node_labels: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        resolution: float = 1.0
    ) -> CommunityResult:
        """
        使用 Louvain 算法进行社区检测
        
        Args:
            node_labels: 节点标签过滤
            relationship_types: 关系类型过滤
            resolution: 分辨率参数
            
        Returns:
            CommunityResult: 社区检测结果
        """
        try:
            # 构建节点过滤条件
            node_filter = ""
            if node_labels:
                labels_str = ":".join(node_labels)
                node_filter = f":{labels_str}"
            
            # 构建关系过滤条件
            rel_filter = ""
            if relationship_types:
                rel_types = "|".join(relationship_types)
                rel_filter = f":{rel_types}"
            
            # 使用 GDS 库进行社区检测（如果可用）
            # 这里提供一个简化的实现
            query = f"""
            CALL {{
                MATCH (n{node_filter})-[r{rel_filter}]-(m{node_filter})
                RETURN n.id as node_id, collect(DISTINCT m.id) as neighbors
            }}
            RETURN node_id, neighbors
            """
            
            result = await self.graph_service.execute_cypher(query)
            
            # 构建邻接表
            adjacency = {}
            for record in result:
                node_id = record["node_id"]
                neighbors = record["neighbors"]
                adjacency[node_id] = set(neighbors)
            
            # 简化的社区检测算法（基于连通分量）
            communities = []
            visited = set()
            
            for node_id in adjacency:
                if node_id not in visited:
                    community = self._find_connected_component(node_id, adjacency, visited)
                    if community:
                        communities.append(list(community))
            
            # 计算模块度（简化版本）
            modularity = self._calculate_modularity(communities, adjacency)
            
            # 计算统计信息
            community_sizes = [len(community) for community in communities]
            
            return CommunityResult(
                communities=communities,
                modularity=modularity,
                total_communities=len(communities),
                largest_community_size=max(community_sizes) if community_sizes else 0,
                smallest_community_size=min(community_sizes) if community_sizes else 0,
                average_community_size=sum(community_sizes) / len(community_sizes) if community_sizes else 0
            )
            
        except Exception as e:
            logger.error(f"Louvain 社区检测失败: {str(e)}")
            raise GraphAnalysisError(f"Louvain 社区检测失败: {str(e)}")
    
    def _find_connected_component(
        self,
        start_node: str,
        adjacency: Dict[str, Set[str]],
        visited: Set[str]
    ) -> Set[str]:
        """
        查找连通分量
        
        Args:
            start_node: 起始节点
            adjacency: 邻接表
            visited: 已访问节点集合
            
        Returns:
            Set[str]: 连通分量中的节点集合
        """
        component = set()
        queue = deque([start_node])
        
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            
            visited.add(node)
            component.add(node)
            
            # 添加邻居节点
            for neighbor in adjacency.get(node, set()):
                if neighbor not in visited:
                    queue.append(neighbor)
        
        return component
    
    def _calculate_modularity(
        self,
        communities: List[List[str]],
        adjacency: Dict[str, Set[str]]
    ) -> float:
        """
        计算模块度
        
        Args:
            communities: 社区列表
            adjacency: 邻接表
            
        Returns:
            float: 模块度值
        """
        try:
            # 简化的模块度计算
            total_edges = sum(len(neighbors) for neighbors in adjacency.values()) // 2
            
            if total_edges == 0:
                return 0.0
            
            modularity = 0.0
            
            for community in communities:
                # 社区内部边数
                internal_edges = 0
                community_set = set(community)
                
                for node in community:
                    neighbors = adjacency.get(node, set())
                    internal_edges += len(neighbors & community_set)
                
                internal_edges //= 2  # 每条边被计算了两次
                
                # 社区的度数总和
                community_degree = sum(len(adjacency.get(node, set())) for node in community)
                
                # 模块度贡献
                modularity += (internal_edges / total_edges) - (community_degree / (2 * total_edges)) ** 2
            
            return modularity
            
        except Exception as e:
            logger.error(f"计算模块度失败: {str(e)}")
            return 0.0


class CentralityAnalyzer:
    """
    中心性分析器
    
    实现各种中心性算法。
    """
    
    def __init__(self, graph_service: GraphService):
        """
        初始化中心性分析器
        
        Args:
            graph_service: 图服务实例
        """
        self.graph_service = graph_service
    
    async def calculate_degree_centrality(
        self,
        node_labels: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        top_k: int = 10
    ) -> CentralityResult:
        """
        计算度中心性
        
        Args:
            node_labels: 节点标签过滤
            relationship_types: 关系类型过滤
            top_k: 返回前K个节点
            
        Returns:
            CentralityResult: 中心性分析结果
        """
        try:
            # 构建过滤条件
            node_filter = ""
            if node_labels:
                labels_str = ":".join(node_labels)
                node_filter = f":{labels_str}"
            
            rel_filter = ""
            if relationship_types:
                rel_types = "|".join(relationship_types)
                rel_filter = f":{rel_types}"
            
            # 计算度中心性
            query = f"""
            MATCH (n{node_filter})
            OPTIONAL MATCH (n)-[r{rel_filter}]-()
            WITH n, count(r) as degree
            RETURN n.id as node_id, degree
            ORDER BY degree DESC
            """
            
            result = await self.graph_service.execute_cypher(query)
            
            # 处理结果
            node_scores = {}
            for record in result:
                node_id = record["node_id"]
                degree = record["degree"]
                node_scores[node_id] = float(degree)
            
            # 排序并获取前K个
            sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
            top_nodes = sorted_nodes[:top_k]
            
            # 计算统计信息
            scores = list(node_scores.values())
            
            return CentralityResult(
                node_scores=node_scores,
                top_nodes=top_nodes,
                algorithm="degree_centrality",
                total_nodes=len(node_scores),
                max_score=max(scores) if scores else 0.0,
                min_score=min(scores) if scores else 0.0,
                average_score=sum(scores) / len(scores) if scores else 0.0
            )
            
        except Exception as e:
            logger.error(f"计算度中心性失败: {str(e)}")
            raise GraphAnalysisError(f"计算度中心性失败: {str(e)}")
    
    async def calculate_betweenness_centrality(
        self,
        node_labels: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        top_k: int = 10,
        sample_size: Optional[int] = None
    ) -> CentralityResult:
        """
        计算介数中心性
        
        Args:
            node_labels: 节点标签过滤
            relationship_types: 关系类型过滤
            top_k: 返回前K个节点
            sample_size: 采样大小（用于大图优化）
            
        Returns:
            CentralityResult: 中心性分析结果
        """
        try:
            # 这里提供一个简化的介数中心性实现
            # 在实际应用中，建议使用 Neo4j GDS 库
            
            # 获取所有节点
            node_filter = ""
            if node_labels:
                labels_str = ":".join(node_labels)
                node_filter = f":{labels_str}"
            
            nodes_query = f"MATCH (n{node_filter}) RETURN n.id as node_id"
            nodes_result = await self.graph_service.execute_cypher(nodes_query)
            node_ids = [record["node_id"] for record in nodes_result]
            
            # 采样节点（如果指定了采样大小）
            if sample_size and len(node_ids) > sample_size:
                import random
                node_ids = random.sample(node_ids, sample_size)
            
            # 计算介数中心性（简化版本）
            betweenness_scores = defaultdict(float)
            
            # 对每对节点计算最短路径
            for i, source in enumerate(node_ids):
                for target in node_ids[i+1:]:
                    paths = await self._find_all_shortest_paths(source, target, relationship_types)
                    
                    if paths:
                        # 计算每个中间节点的贡献
                        for path in paths:
                            path_length = len(path)
                            if path_length > 2:  # 至少有一个中间节点
                                contribution = 1.0 / len(paths)  # 平均分配到所有最短路径
                                
                                # 为路径上的中间节点增加分数
                                for j in range(1, path_length - 1):
                                    intermediate_node = path[j]
                                    betweenness_scores[intermediate_node] += contribution
            
            # 标准化分数
            n = len(node_ids)
            if n > 2:
                normalization_factor = 2.0 / ((n - 1) * (n - 2))
                for node_id in betweenness_scores:
                    betweenness_scores[node_id] *= normalization_factor
            
            # 确保所有节点都有分数
            for node_id in node_ids:
                if node_id not in betweenness_scores:
                    betweenness_scores[node_id] = 0.0
            
            # 排序并获取前K个
            sorted_nodes = sorted(betweenness_scores.items(), key=lambda x: x[1], reverse=True)
            top_nodes = sorted_nodes[:top_k]
            
            # 计算统计信息
            scores = list(betweenness_scores.values())
            
            return CentralityResult(
                node_scores=dict(betweenness_scores),
                top_nodes=top_nodes,
                algorithm="betweenness_centrality",
                total_nodes=len(betweenness_scores),
                max_score=max(scores) if scores else 0.0,
                min_score=min(scores) if scores else 0.0,
                average_score=sum(scores) / len(scores) if scores else 0.0
            )
            
        except Exception as e:
            logger.error(f"计算介数中心性失败: {str(e)}")
            raise GraphAnalysisError(f"计算介数中心性失败: {str(e)}")
    
    async def _find_all_shortest_paths(
        self,
        source: str,
        target: str,
        relationship_types: Optional[List[str]] = None
    ) -> List[List[str]]:
        """
        查找两个节点间的所有最短路径
        
        Args:
            source: 源节点ID
            target: 目标节点ID
            relationship_types: 关系类型过滤
            
        Returns:
            List[List[str]]: 所有最短路径（节点ID列表）
        """
        try:
            rel_filter = ""
            if relationship_types:
                rel_types = "|".join(relationship_types)
                rel_filter = f":{rel_types}"
            
            query = f"""
            MATCH path = allShortestPaths((source {{id: $source}})-[{rel_filter}*]-(target {{id: $target}}))
            RETURN [node in nodes(path) | node.id] as path_nodes
            """
            
            params = {"source": source, "target": target}
            result = await self.graph_service.execute_cypher(query, params)
            
            return [record["path_nodes"] for record in result]
            
        except Exception as e:
            logger.error(f"查找最短路径失败: {str(e)}")
            return []


class SimilaritySearcher:
    """
    相似性搜索器
    
    实现基于图结构的相似性搜索。
    """
    
    def __init__(self, graph_service: GraphService):
        """
        初始化相似性搜索器
        
        Args:
            graph_service: 图服务实例
        """
        self.graph_service = graph_service
    
    async def find_similar_nodes_by_structure(
        self,
        node_id: str,
        similarity_threshold: float = 0.5,
        max_results: int = 10,
        depth: int = 2
    ) -> SimilarityResult:
        """
        基于结构相似性查找相似节点
        
        Args:
            node_id: 查询节点ID
            similarity_threshold: 相似度阈值
            max_results: 最大结果数量
            depth: 搜索深度
            
        Returns:
            SimilarityResult: 相似性查询结果
        """
        try:
            # 获取查询节点的邻居结构
            query_structure = await self._get_node_structure(node_id, depth)
            
            # 获取所有候选节点
            candidates_query = """
            MATCH (n)
            WHERE n.id <> $node_id
            RETURN n.id as candidate_id
            """
            
            candidates_result = await self.graph_service.execute_cypher(
                candidates_query, 
                {"node_id": node_id}
            )
            
            # 计算相似度
            similar_nodes = []
            
            for record in candidates_result:
                candidate_id = record["candidate_id"]
                candidate_structure = await self._get_node_structure(candidate_id, depth)
                
                # 计算结构相似度
                similarity = self._calculate_structural_similarity(query_structure, candidate_structure)
                
                if similarity >= similarity_threshold:
                    similar_nodes.append((candidate_id, similarity))
            
            # 排序并限制结果数量
            similar_nodes.sort(key=lambda x: x[1], reverse=True)
            similar_nodes = similar_nodes[:max_results]
            
            return SimilarityResult(
                similar_nodes=similar_nodes,
                query_node_id=node_id,
                similarity_threshold=similarity_threshold,
                algorithm="structural_similarity",
                total_similar_nodes=len(similar_nodes)
            )
            
        except Exception as e:
            logger.error(f"结构相似性搜索失败: {str(e)}")
            raise GraphQueryError(f"结构相似性搜索失败: {str(e)}")
    
    async def _get_node_structure(self, node_id: str, depth: int) -> Dict[str, Any]:
        """
        获取节点的结构信息
        
        Args:
            node_id: 节点ID
            depth: 搜索深度
            
        Returns:
            Dict[str, Any]: 节点结构信息
        """
        try:
            query = f"""
            MATCH (center {{id: $node_id}})
            OPTIONAL MATCH path = (center)-[*1..{depth}]-(neighbor)
            RETURN 
                center,
                collect(DISTINCT type(relationships(path)[0])) as relationship_types,
                collect(DISTINCT labels(neighbor)) as neighbor_labels,
                count(DISTINCT neighbor) as neighbor_count
            """
            
            result = await self.graph_service.execute_cypher(query, {"node_id": node_id})
            
            if not result:
                return {}
            
            record = result[0]
            
            return {
                "relationship_types": record["relationship_types"],
                "neighbor_labels": record["neighbor_labels"],
                "neighbor_count": record["neighbor_count"]
            }
            
        except Exception as e:
            logger.error(f"获取节点结构失败: {str(e)}")
            return {}
    
    def _calculate_structural_similarity(
        self,
        structure1: Dict[str, Any],
        structure2: Dict[str, Any]
    ) -> float:
        """
        计算结构相似度
        
        Args:
            structure1: 结构1
            structure2: 结构2
            
        Returns:
            float: 相似度分数 (0-1)
        """
        try:
            if not structure1 or not structure2:
                return 0.0
            
            # 关系类型相似度
            rel_types1 = set(structure1.get("relationship_types", []))
            rel_types2 = set(structure2.get("relationship_types", []))
            
            rel_similarity = 0.0
            if rel_types1 or rel_types2:
                intersection = len(rel_types1 & rel_types2)
                union = len(rel_types1 | rel_types2)
                rel_similarity = intersection / union if union > 0 else 0.0
            
            # 邻居标签相似度
            labels1 = set()
            labels2 = set()
            
            for label_list in structure1.get("neighbor_labels", []):
                labels1.update(label_list)
            
            for label_list in structure2.get("neighbor_labels", []):
                labels2.update(label_list)
            
            label_similarity = 0.0
            if labels1 or labels2:
                intersection = len(labels1 & labels2)
                union = len(labels1 | labels2)
                label_similarity = intersection / union if union > 0 else 0.0
            
            # 邻居数量相似度
            count1 = structure1.get("neighbor_count", 0)
            count2 = structure2.get("neighbor_count", 0)
            
            count_similarity = 0.0
            if count1 > 0 or count2 > 0:
                max_count = max(count1, count2)
                min_count = min(count1, count2)
                count_similarity = min_count / max_count if max_count > 0 else 0.0
            
            # 综合相似度（加权平均）
            total_similarity = (
                rel_similarity * 0.4 +
                label_similarity * 0.4 +
                count_similarity * 0.2
            )
            
            return total_similarity
            
        except Exception as e:
            logger.error(f"计算结构相似度失败: {str(e)}")
            return 0.0


class GraphQueryService:
    """
    图查询和遍历服务
    
    提供完整的图查询和分析功能。
    """
    
    def __init__(self, graph_service: GraphService):
        """
        初始化图查询服务
        
        Args:
            graph_service: 图服务实例
        """
        self.graph_service = graph_service
        
        # 初始化各个组件
        self.path_finder = PathFinder(graph_service)
        self.community_detector = CommunityDetector(graph_service)
        self.centrality_analyzer = CentralityAnalyzer(graph_service)
        self.similarity_searcher = SimilaritySearcher(graph_service)
        
        # 服务统计
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "path_queries": 0,
            "community_queries": 0,
            "centrality_queries": 0,
            "similarity_queries": 0,
            "last_query_time": None
        }
        
        logger.info("图查询和遍历服务初始化完成")
    
    # 路径查询方法
    async def find_shortest_path(
        self,
        start_node_id: str,
        end_node_id: str,
        max_depth: int = 10,
        relationship_types: Optional[List[str]] = None
    ) -> Optional[GraphPath]:
        """查找最短路径"""
        try:
            self.stats["total_queries"] += 1
            self.stats["path_queries"] += 1
            
            result = await self.path_finder.find_shortest_path(
                start_node_id, end_node_id, max_depth, relationship_types
            )
            
            self.stats["successful_queries"] += 1
            self.stats["last_query_time"] = datetime.utcnow().isoformat()
            
            return result
        except Exception as e:
            self.stats["failed_queries"] += 1
            raise
    
    async def find_all_paths(
        self,
        start_node_id: str,
        end_node_id: str,
        max_depth: int = 5,
        relationship_types: Optional[List[str]] = None,
        limit: int = 100
    ) -> PathResult:
        """查找所有路径"""
        try:
            self.stats["total_queries"] += 1
            self.stats["path_queries"] += 1
            
            result = await self.path_finder.find_all_paths(
                start_node_id, end_node_id, max_depth, relationship_types, limit
            )
            
            self.stats["successful_queries"] += 1
            self.stats["last_query_time"] = datetime.utcnow().isoformat()
            
            return result
        except Exception as e:
            self.stats["failed_queries"] += 1
            raise
    
    # 社区检测方法
    async def detect_communities(
        self,
        algorithm: str = "louvain",
        node_labels: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        **kwargs
    ) -> CommunityResult:
        """社区检测"""
        try:
            self.stats["total_queries"] += 1
            self.stats["community_queries"] += 1
            
            if algorithm.lower() == "louvain":
                result = await self.community_detector.detect_communities_louvain(
                    node_labels, relationship_types, kwargs.get("resolution", 1.0)
                )
            else:
                raise ValueError(f"不支持的社区检测算法: {algorithm}")
            
            self.stats["successful_queries"] += 1
            self.stats["last_query_time"] = datetime.utcnow().isoformat()
            
            return result
        except Exception as e:
            self.stats["failed_queries"] += 1
            raise
    
    # 中心性分析方法
    async def calculate_centrality(
        self,
        algorithm: str = "degree",
        node_labels: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        top_k: int = 10,
        **kwargs
    ) -> CentralityResult:
        """中心性分析"""
        try:
            self.stats["total_queries"] += 1
            self.stats["centrality_queries"] += 1
            
            if algorithm.lower() == "degree":
                result = await self.centrality_analyzer.calculate_degree_centrality(
                    node_labels, relationship_types, top_k
                )
            elif algorithm.lower() == "betweenness":
                result = await self.centrality_analyzer.calculate_betweenness_centrality(
                    node_labels, relationship_types, top_k, kwargs.get("sample_size")
                )
            else:
                raise ValueError(f"不支持的中心性算法: {algorithm}")
            
            self.stats["successful_queries"] += 1
            self.stats["last_query_time"] = datetime.utcnow().isoformat()
            
            return result
        except Exception as e:
            self.stats["failed_queries"] += 1
            raise
    
    # 相似性搜索方法
    async def find_similar_nodes(
        self,
        node_id: str,
        algorithm: str = "structural",
        similarity_threshold: float = 0.5,
        max_results: int = 10,
        **kwargs
    ) -> SimilarityResult:
        """相似性搜索"""
        try:
            self.stats["total_queries"] += 1
            self.stats["similarity_queries"] += 1
            
            if algorithm.lower() == "structural":
                result = await self.similarity_searcher.find_similar_nodes_by_structure(
                    node_id, similarity_threshold, max_results, kwargs.get("depth", 2)
                )
            else:
                raise ValueError(f"不支持的相似性算法: {algorithm}")
            
            self.stats["successful_queries"] += 1
            self.stats["last_query_time"] = datetime.utcnow().isoformat()
            
            return result
        except Exception as e:
            self.stats["failed_queries"] += 1
            raise
    
    # 子图提取
    async def extract_subgraph(
        self,
        center_node_ids: List[str],
        max_depth: int = 2,
        node_labels: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        max_nodes: int = 1000
    ) -> SubgraphResult:
        """
        提取子图
        
        Args:
            center_node_ids: 中心节点ID列表
            max_depth: 最大深度
            node_labels: 节点标签过滤
            relationship_types: 关系类型过滤
            max_nodes: 最大节点数量
            
        Returns:
            SubgraphResult: 子图提取结果
        """
        try:
            self.stats["total_queries"] += 1
            
            # 构建过滤条件
            node_filter = ""
            if node_labels:
                labels_str = ":".join(node_labels)
                node_filter = f":{labels_str}"
            
            rel_filter = ""
            if relationship_types:
                rel_types = "|".join(relationship_types)
                rel_filter = f":{rel_types}"
            
            # 构建中心节点条件
            center_ids_str = ", ".join([f"'{node_id}'" for node_id in center_node_ids])
            
            # 查询子图
            query = f"""
            MATCH (center{node_filter})
            WHERE center.id IN [{center_ids_str}]
            OPTIONAL MATCH path = (center)-[{rel_filter}*1..{max_depth}]-(neighbor{node_filter})
            WITH collect(DISTINCT center) + collect(DISTINCT neighbor) as all_nodes,
                 collect(DISTINCT relationships(path)) as all_rels
            UNWIND all_nodes as node
            WITH collect(DISTINCT node) as nodes, all_rels
            UNWIND all_rels as rel_list
            UNWIND rel_list as rel
            WITH nodes, collect(DISTINCT rel) as relationships
            RETURN nodes[0..{max_nodes}] as limited_nodes, relationships
            """
            
            result = await self.graph_service.execute_cypher(query)
            
            if not result:
                return SubgraphResult(
                    nodes=[],
                    relationships=[],
                    total_nodes=0,
                    total_relationships=0,
                    density=0.0,
                    diameter=None
                )
            
            # 解析结果
            record = result[0]
            nodes_data = record.get("limited_nodes", [])
            relationships_data = record.get("relationships", [])
            
            # 转换为 GraphNode 和 GraphRelationship 对象
            nodes = []
            for node_data in nodes_data:
                if node_data:  # 过滤空值
                    node = GraphNode(
                        id=node_data.get("id", str(uuid4())),
                        labels=list(node_data.labels) if hasattr(node_data, 'labels') else [],
                        properties=dict(node_data)
                    )
                    nodes.append(node)
            
            relationships = []
            for rel_data in relationships_data:
                if rel_data:  # 过滤空值
                    relationship = GraphRelationship(
                        id=rel_data.get("id", str(uuid4())),
                        type=rel_data.type if hasattr(rel_data, 'type') else "UNKNOWN",
                        start_node_id=str(rel_data.start_node.id) if hasattr(rel_data, 'start_node') else "",
                        end_node_id=str(rel_data.end_node.id) if hasattr(rel_data, 'end_node') else "",
                        properties=dict(rel_data)
                    )
                    relationships.append(relationship)
            
            # 计算图密度
            n_nodes = len(nodes)
            n_relationships = len(relationships)
            
            density = 0.0
            if n_nodes > 1:
                max_possible_edges = n_nodes * (n_nodes - 1) / 2
                density = n_relationships / max_possible_edges
            
            # 计算图直径（简化版本）
            diameter = None
            if n_nodes > 0:
                diameter = max_depth  # 简化为最大搜索深度
            
            self.stats["successful_queries"] += 1
            self.stats["last_query_time"] = datetime.utcnow().isoformat()
            
            return SubgraphResult(
                nodes=nodes,
                relationships=relationships,
                total_nodes=n_nodes,
                total_relationships=n_relationships,
                density=density,
                diameter=diameter
            )
            
        except Exception as e:
            self.stats["failed_queries"] += 1
            logger.error(f"提取子图失败: {str(e)}")
            raise GraphQueryError(f"提取子图失败: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        stats = self.stats.copy()
        
        # 计算成功率
        if stats["total_queries"] > 0:
            stats["success_rate"] = stats["successful_queries"] / stats["total_queries"]
        else:
            stats["success_rate"] = 0.0
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 基础图服务健康检查
            graph_health = await self.graph_service.health_check()
            
            if graph_health["status"] != "healthy":
                return {
                    "status": "unhealthy",
                    "graph_service": graph_health,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # 测试基本查询功能
            try:
                # 简单的节点计数查询
                result = await self.graph_service.execute_cypher("MATCH (n) RETURN count(n) as node_count")
                
                if result:
                    return {
                        "status": "healthy",
                        "graph_service": graph_health,
                        "query_test": "passed",
                        "node_count": result[0]["node_count"],
                        "stats": self.get_stats(),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                else:
                    return {
                        "status": "degraded",
                        "graph_service": graph_health,
                        "query_test": "failed",
                        "error": "查询返回空结果",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
            except Exception as e:
                return {
                    "status": "degraded",
                    "graph_service": graph_health,
                    "query_test": "failed",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }