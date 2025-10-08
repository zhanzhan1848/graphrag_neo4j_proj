#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 图查询工具
==================

本模块定义了 GraphRAG 系统的图查询工具类。

查询工具说明：
- GraphQueryBuilder: 图查询构建器，提供流式查询构建接口
- EntityQuery: 实体查询工具，专门用于实体相关查询
- RelationshipQuery: 关系查询工具，专门用于关系相关查询
- PathQuery: 路径查询工具，用于查找节点间的路径
- NeighborQuery: 邻居查询工具，用于查找节点的邻居
- SimilarityQuery: 相似性查询工具，用于相似性搜索

功能说明：
- 支持复杂的图查询构建
- 提供高级查询接口
- 支持查询优化和缓存
- 提供查询结果分析

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID

from neomodel import db
from neomodel.exceptions import DoesNotExist

from .nodes import (
    EntityNode,
    DocumentNode,
    ChunkNode,
    ConceptNode,
    PersonNode,
    OrganizationNode,
    LocationNode,
    EventNode,
    TopicNode
)
from .relationships import (
    ContainsRelationship,
    MentionsRelationship,
    RelatesToRelationship,
    PartOfRelationship,
    SimilarToRelationship
)


class GraphQueryBuilder:
    """
    图查询构建器
    
    提供流式查询构建接口，支持复杂的图查询操作。
    """
    
    def __init__(self):
        """
        初始化查询构建器
        """
        self.query_parts = []
        self.parameters = {}
        self.return_clause = ""
        self.order_clause = ""
        self.limit_clause = ""
        self.where_conditions = []
        
    def match(self, pattern: str, **params) -> 'GraphQueryBuilder':
        """
        添加 MATCH 子句
        
        Args:
            pattern: 匹配模式
            **params: 查询参数
            
        Returns:
            GraphQueryBuilder: 查询构建器实例
        """
        self.query_parts.append(f"MATCH {pattern}")
        self.parameters.update(params)
        return self
        
    def where(self, condition: str, **params) -> 'GraphQueryBuilder':
        """
        添加 WHERE 条件
        
        Args:
            condition: 条件表达式
            **params: 查询参数
            
        Returns:
            GraphQueryBuilder: 查询构建器实例
        """
        self.where_conditions.append(condition)
        self.parameters.update(params)
        return self
        
    def return_nodes(self, *nodes) -> 'GraphQueryBuilder':
        """
        设置返回节点
        
        Args:
            *nodes: 要返回的节点变量
            
        Returns:
            GraphQueryBuilder: 查询构建器实例
        """
        self.return_clause = f"RETURN {', '.join(nodes)}"
        return self
        
    def order_by(self, field: str, desc: bool = False) -> 'GraphQueryBuilder':
        """
        设置排序
        
        Args:
            field: 排序字段
            desc: 是否降序
            
        Returns:
            GraphQueryBuilder: 查询构建器实例
        """
        direction = "DESC" if desc else "ASC"
        self.order_clause = f"ORDER BY {field} {direction}"
        return self
        
    def limit(self, count: int) -> 'GraphQueryBuilder':
        """
        设置结果限制
        
        Args:
            count: 结果数量限制
            
        Returns:
            GraphQueryBuilder: 查询构建器实例
        """
        self.limit_clause = f"LIMIT {count}"
        return self
        
    def build(self) -> Tuple[str, Dict[str, Any]]:
        """
        构建最终查询
        
        Returns:
            Tuple[str, Dict[str, Any]]: 查询语句和参数
        """
        query_parts = self.query_parts.copy()
        
        if self.where_conditions:
            query_parts.append(f"WHERE {' AND '.join(self.where_conditions)}")
            
        if self.return_clause:
            query_parts.append(self.return_clause)
            
        if self.order_clause:
            query_parts.append(self.order_clause)
            
        if self.limit_clause:
            query_parts.append(self.limit_clause)
            
        query = " ".join(query_parts)
        return query, self.parameters
        
    def execute(self) -> List[Dict[str, Any]]:
        """
        执行查询
        
        Returns:
            List[Dict[str, Any]]: 查询结果
        """
        query, params = self.build()
        results, _ = db.cypher_query(query, params)
        return results


class EntityQuery:
    """
    实体查询工具
    
    专门用于实体相关的查询操作。
    """
    
    @staticmethod
    def find_by_name(name: str, entity_type: Optional[str] = None) -> List[EntityNode]:
        """
        根据名称查找实体
        
        Args:
            name: 实体名称
            entity_type: 实体类型（可选）
            
        Returns:
            List[EntityNode]: 匹配的实体列表
        """
        query = EntityNode.nodes
        
        if entity_type:
            query = query.filter(name__icontains=name, entity_type=entity_type)
        else:
            query = query.filter(name__icontains=name)
            
        return list(query)
        
    @staticmethod
    def find_similar_entities(entity_id: str, similarity_threshold: float = 0.7, 
                            limit: int = 10) -> List[Tuple[EntityNode, float]]:
        """
        查找相似实体
        
        Args:
            entity_id: 实体ID
            similarity_threshold: 相似度阈值
            limit: 结果数量限制
            
        Returns:
            List[Tuple[EntityNode, float]]: 相似实体和相似度分数
        """
        query = """
        MATCH (e1:EntityNode {id: $entity_id})-[r:SIMILAR_TO]-(e2:EntityNode)
        WHERE r.similarity_score >= $threshold
        RETURN e2, r.similarity_score as score
        ORDER BY score DESC
        LIMIT $limit
        """
        
        results, _ = db.cypher_query(query, {
            'entity_id': entity_id,
            'threshold': similarity_threshold,
            'limit': limit
        })
        
        similar_entities = []
        for row in results:
            entity_data = row[0]
            score = row[1]
            entity = EntityNode.inflate(entity_data)
            similar_entities.append((entity, score))
            
        return similar_entities
        
    @staticmethod
    def get_entity_mentions(entity_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取实体的所有提及
        
        Args:
            entity_id: 实体ID
            limit: 结果数量限制
            
        Returns:
            List[Dict[str, Any]]: 提及信息列表
        """
        query = """
        MATCH (e:EntityNode {id: $entity_id})<-[m:MENTIONS]-(c:ChunkNode)
        RETURN c, m
        ORDER BY m.confidence DESC
        LIMIT $limit
        """
        
        results, _ = db.cypher_query(query, {
            'entity_id': entity_id,
            'limit': limit
        })
        
        mentions = []
        for row in results:
            chunk_data = row[0]
            mention_data = row[1]
            
            chunk = ChunkNode.inflate(chunk_data)
            mention_info = {
                'chunk': chunk,
                'mention_text': mention_data.get('mention_text'),
                'confidence': mention_data.get('confidence'),
                'position': mention_data.get('start_position'),
                'context': mention_data.get('context_before', '') + mention_data.get('context_after', '')
            }
            mentions.append(mention_info)
            
        return mentions
        
    @staticmethod
    def get_entity_relationships(entity_id: str, relation_types: Optional[List[str]] = None,
                               limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取实体的所有关系
        
        Args:
            entity_id: 实体ID
            relation_types: 关系类型过滤（可选）
            limit: 结果数量限制
            
        Returns:
            List[Dict[str, Any]]: 关系信息列表
        """
        if relation_types:
            type_filter = f"AND type(r) IN {relation_types}"
        else:
            type_filter = ""
            
        query = f"""
        MATCH (e1:EntityNode {{id: $entity_id}})-[r]-(e2:EntityNode)
        WHERE e1 <> e2 {type_filter}
        RETURN e2, r, type(r) as relation_type
        ORDER BY r.weight DESC, r.confidence DESC
        LIMIT $limit
        """
        
        results, _ = db.cypher_query(query, {
            'entity_id': entity_id,
            'limit': limit
        })
        
        relationships = []
        for row in results:
            target_entity_data = row[0]
            relation_data = row[1]
            relation_type = row[2]
            
            target_entity = EntityNode.inflate(target_entity_data)
            relationship_info = {
                'target_entity': target_entity,
                'relation_type': relation_type,
                'weight': relation_data.get('weight', 0.0),
                'confidence': relation_data.get('confidence', 0.0),
                'properties': dict(relation_data)
            }
            relationships.append(relationship_info)
            
        return relationships


class RelationshipQuery:
    """
    关系查询工具
    
    专门用于关系相关的查询操作。
    """
    
    @staticmethod
    def find_relationships_between(entity1_id: str, entity2_id: str) -> List[Dict[str, Any]]:
        """
        查找两个实体间的所有关系
        
        Args:
            entity1_id: 第一个实体ID
            entity2_id: 第二个实体ID
            
        Returns:
            List[Dict[str, Any]]: 关系信息列表
        """
        query = """
        MATCH (e1:EntityNode {id: $entity1_id})-[r]-(e2:EntityNode {id: $entity2_id})
        RETURN r, type(r) as relation_type, 
               CASE WHEN startNode(r) = e1 THEN 'outgoing' ELSE 'incoming' END as direction
        ORDER BY r.weight DESC
        """
        
        results, _ = db.cypher_query(query, {
            'entity1_id': entity1_id,
            'entity2_id': entity2_id
        })
        
        relationships = []
        for row in results:
            relation_data = row[0]
            relation_type = row[1]
            direction = row[2]
            
            relationship_info = {
                'relation_type': relation_type,
                'direction': direction,
                'weight': relation_data.get('weight', 0.0),
                'confidence': relation_data.get('confidence', 0.0),
                'properties': dict(relation_data)
            }
            relationships.append(relationship_info)
            
        return relationships
        
    @staticmethod
    def find_strongest_relationships(relation_type: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        查找最强的关系
        
        Args:
            relation_type: 关系类型
            limit: 结果数量限制
            
        Returns:
            List[Dict[str, Any]]: 关系信息列表
        """
        query = f"""
        MATCH (e1)-[r:{relation_type}]-(e2)
        RETURN e1, e2, r
        ORDER BY r.weight DESC, r.confidence DESC
        LIMIT $limit
        """
        
        results, _ = db.cypher_query(query, {'limit': limit})
        
        relationships = []
        for row in results:
            entity1_data = row[0]
            entity2_data = row[1]
            relation_data = row[2]
            
            entity1 = EntityNode.inflate(entity1_data)
            entity2 = EntityNode.inflate(entity2_data)
            
            relationship_info = {
                'entity1': entity1,
                'entity2': entity2,
                'weight': relation_data.get('weight', 0.0),
                'confidence': relation_data.get('confidence', 0.0),
                'properties': dict(relation_data)
            }
            relationships.append(relationship_info)
            
        return relationships


class PathQuery:
    """
    路径查询工具
    
    用于查找节点间的路径。
    """
    
    @staticmethod
    def find_shortest_path(start_entity_id: str, end_entity_id: str, 
                          max_depth: int = 5) -> Optional[List[Dict[str, Any]]]:
        """
        查找最短路径
        
        Args:
            start_entity_id: 起始实体ID
            end_entity_id: 结束实体ID
            max_depth: 最大深度
            
        Returns:
            Optional[List[Dict[str, Any]]]: 路径信息，如果不存在则返回None
        """
        query = f"""
        MATCH path = shortestPath((start:EntityNode {{id: $start_id}})-[*1..{max_depth}]-(end:EntityNode {{id: $end_id}}))
        RETURN path, length(path) as path_length
        ORDER BY path_length
        LIMIT 1
        """
        
        results, _ = db.cypher_query(query, {
            'start_id': start_entity_id,
            'end_id': end_entity_id
        })
        
        if not results:
            return None
            
        path_data = results[0][0]
        path_length = results[0][1]
        
        # 解析路径
        path_info = []
        nodes = path_data.nodes
        relationships = path_data.relationships
        
        for i, node in enumerate(nodes):
            entity = EntityNode.inflate(node)
            step_info = {
                'step': i,
                'entity': entity,
                'relationship': None
            }
            
            if i < len(relationships):
                rel_data = relationships[i]
                step_info['relationship'] = {
                    'type': rel_data.type,
                    'properties': dict(rel_data)
                }
                
            path_info.append(step_info)
            
        return path_info
        
    @staticmethod
    def find_all_paths(start_entity_id: str, end_entity_id: str, 
                      max_depth: int = 3, limit: int = 10) -> List[List[Dict[str, Any]]]:
        """
        查找所有路径
        
        Args:
            start_entity_id: 起始实体ID
            end_entity_id: 结束实体ID
            max_depth: 最大深度
            limit: 结果数量限制
            
        Returns:
            List[List[Dict[str, Any]]]: 所有路径信息列表
        """
        query = f"""
        MATCH path = (start:EntityNode {{id: $start_id}})-[*1..{max_depth}]-(end:EntityNode {{id: $end_id}})
        RETURN path, length(path) as path_length
        ORDER BY path_length
        LIMIT $limit
        """
        
        results, _ = db.cypher_query(query, {
            'start_id': start_entity_id,
            'end_id': end_entity_id,
            'limit': limit
        })
        
        all_paths = []
        for row in results:
            path_data = row[0]
            
            # 解析路径
            path_info = []
            nodes = path_data.nodes
            relationships = path_data.relationships
            
            for i, node in enumerate(nodes):
                entity = EntityNode.inflate(node)
                step_info = {
                    'step': i,
                    'entity': entity,
                    'relationship': None
                }
                
                if i < len(relationships):
                    rel_data = relationships[i]
                    step_info['relationship'] = {
                        'type': rel_data.type,
                        'properties': dict(rel_data)
                    }
                    
                path_info.append(step_info)
                
            all_paths.append(path_info)
            
        return all_paths


class NeighborQuery:
    """
    邻居查询工具
    
    用于查找节点的邻居。
    """
    
    @staticmethod
    def get_direct_neighbors(entity_id: str, relation_types: Optional[List[str]] = None,
                           limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取直接邻居
        
        Args:
            entity_id: 实体ID
            relation_types: 关系类型过滤（可选）
            limit: 结果数量限制
            
        Returns:
            List[Dict[str, Any]]: 邻居信息列表
        """
        if relation_types:
            type_filter = f"AND type(r) IN {relation_types}"
        else:
            type_filter = ""
            
        query = f"""
        MATCH (e:EntityNode {{id: $entity_id}})-[r]-(neighbor:EntityNode)
        WHERE e <> neighbor {type_filter}
        RETURN neighbor, r, type(r) as relation_type
        ORDER BY r.weight DESC
        LIMIT $limit
        """
        
        results, _ = db.cypher_query(query, {
            'entity_id': entity_id,
            'limit': limit
        })
        
        neighbors = []
        for row in results:
            neighbor_data = row[0]
            relation_data = row[1]
            relation_type = row[2]
            
            neighbor = EntityNode.inflate(neighbor_data)
            neighbor_info = {
                'entity': neighbor,
                'relation_type': relation_type,
                'weight': relation_data.get('weight', 0.0),
                'confidence': relation_data.get('confidence', 0.0),
                'properties': dict(relation_data)
            }
            neighbors.append(neighbor_info)
            
        return neighbors
        
    @staticmethod
    def get_k_hop_neighbors(entity_id: str, k: int = 2, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取K跳邻居
        
        Args:
            entity_id: 实体ID
            k: 跳数
            limit: 结果数量限制
            
        Returns:
            List[Dict[str, Any]]: K跳邻居信息列表
        """
        query = f"""
        MATCH path = (e:EntityNode {{id: $entity_id}})-[*1..{k}]-(neighbor:EntityNode)
        WHERE e <> neighbor
        RETURN DISTINCT neighbor, length(path) as distance
        ORDER BY distance, neighbor.name
        LIMIT $limit
        """
        
        results, _ = db.cypher_query(query, {
            'entity_id': entity_id,
            'limit': limit
        })
        
        neighbors = []
        for row in results:
            neighbor_data = row[0]
            distance = row[1]
            
            neighbor = EntityNode.inflate(neighbor_data)
            neighbor_info = {
                'entity': neighbor,
                'distance': distance
            }
            neighbors.append(neighbor_info)
            
        return neighbors


class SimilarityQuery:
    """
    相似性查询工具
    
    用于相似性搜索和分析。
    """
    
    @staticmethod
    def find_similar_by_embedding(entity_id: str, similarity_threshold: float = 0.7,
                                limit: int = 20) -> List[Tuple[EntityNode, float]]:
        """
        基于嵌入向量查找相似实体
        
        Args:
            entity_id: 实体ID
            similarity_threshold: 相似度阈值
            limit: 结果数量限制
            
        Returns:
            List[Tuple[EntityNode, float]]: 相似实体和相似度分数
        """
        # 这里需要实现向量相似度计算
        # 由于Neo4j的向量相似度查询比较复杂，这里提供一个简化版本
        query = """
        MATCH (e1:EntityNode {id: $entity_id})
        MATCH (e2:EntityNode)
        WHERE e1 <> e2 AND size(e1.embedding) > 0 AND size(e2.embedding) > 0
        WITH e1, e2, 
             reduce(dot = 0.0, i in range(0, size(e1.embedding)-1) | 
                    dot + e1.embedding[i] * e2.embedding[i]) as dot_product,
             sqrt(reduce(norm1 = 0.0, i in range(0, size(e1.embedding)-1) | 
                         norm1 + e1.embedding[i] * e1.embedding[i])) as norm1,
             sqrt(reduce(norm2 = 0.0, i in range(0, size(e2.embedding)-1) | 
                         norm2 + e2.embedding[i] * e2.embedding[i])) as norm2
        WITH e2, dot_product / (norm1 * norm2) as similarity
        WHERE similarity >= $threshold
        RETURN e2, similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """
        
        results, _ = db.cypher_query(query, {
            'entity_id': entity_id,
            'threshold': similarity_threshold,
            'limit': limit
        })
        
        similar_entities = []
        for row in results:
            entity_data = row[0]
            similarity = row[1]
            
            entity = EntityNode.inflate(entity_data)
            similar_entities.append((entity, similarity))
            
        return similar_entities
        
    @staticmethod
    def cluster_similar_entities(entity_ids: List[str], 
                               similarity_threshold: float = 0.8) -> List[List[str]]:
        """
        聚类相似实体
        
        Args:
            entity_ids: 实体ID列表
            similarity_threshold: 相似度阈值
            
        Returns:
            List[List[str]]: 聚类结果，每个子列表包含相似的实体ID
        """
        # 简化的聚类实现
        clusters = []
        processed = set()
        
        for entity_id in entity_ids:
            if entity_id in processed:
                continue
                
            # 查找与当前实体相似的所有实体
            similar_entities = SimilarityQuery.find_similar_by_embedding(
                entity_id, similarity_threshold
            )
            
            cluster = [entity_id]
            for similar_entity, _ in similar_entities:
                if similar_entity.id in entity_ids and similar_entity.id not in processed:
                    cluster.append(similar_entity.id)
                    processed.add(similar_entity.id)
                    
            processed.add(entity_id)
            clusters.append(cluster)
            
        return clusters