#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 图数据库操作服务
========================

本模块实现了 Neo4j 图数据库的核心操作功能。

服务功能：
- 数据库连接管理
- 节点 CRUD 操作
- 关系 CRUD 操作
- 图查询和遍历
- 批量操作
- 事务管理
- 索引管理
- 性能监控

支持的节点类型：
- Entity: 实体节点
- Document: 文档节点
- Chunk: 文本块节点
- Person: 人物节点
- Organization: 机构节点
- Concept: 概念节点
- Location: 地点节点

支持的关系类型：
- RELATED_TO: 相关关系
- IS_A: 是一个关系
- PART_OF: 部分关系
- WORKS_AT: 工作于
- LOCATED_IN: 位于
- COLLABORATES_WITH: 合作关系
- DEVELOPS: 开发关系
- USES: 使用关系
- CITES: 引用关系
- MENTIONS: 提及关系

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
from uuid import UUID, uuid4
import hashlib

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession, AsyncTransaction
from neo4j.exceptions import Neo4jError, ServiceUnavailable, AuthError, ClientError

from app.core.config import settings
from app.models.database.entities import Entity
from app.models.database.relations import Relation
from app.models.database.documents import Document
from app.models.database.chunks import Chunk
# from app.models.schemas.entities import EntityCreate, EntityUpdate
# from app.models.schemas.relations import RelationCreate, RelationUpdate
from app.utils.exceptions import (
    GraphDatabaseError,
    GraphConnectionError,
    GraphQueryError,
    GraphTransactionError,
    GraphIndexError,
    DatabaseError
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = settings


@dataclass
class GraphNode:
    """图节点"""
    id: str
    labels: List[str]
    properties: Dict[str, Any]
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at


@dataclass
class GraphRelationship:
    """图关系"""
    id: str
    type: str
    start_node_id: str
    end_node_id: str
    properties: Dict[str, Any]
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at


@dataclass
class GraphPath:
    """图路径"""
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]
    length: int
    
    def __post_init__(self):
        self.length = len(self.relationships)


@dataclass
class GraphStats:
    """图统计信息"""
    total_nodes: int
    total_relationships: int
    node_types: Dict[str, int]
    relationship_types: Dict[str, int]
    database_size: str
    last_updated: str


class GraphConnectionManager:
    """
    图数据库连接管理器
    
    管理 Neo4j 数据库连接的生命周期。
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        max_connection_lifetime: int = 3600,
        max_connection_pool_size: int = 100,
        connection_acquisition_timeout: int = 60
    ):
        """
        初始化连接管理器
        
        Args:
            uri: Neo4j 连接 URI
            username: 用户名
            password: 密码
            database: 数据库名称
            max_connection_lifetime: 最大连接生命周期（秒）
            max_connection_pool_size: 最大连接池大小
            connection_acquisition_timeout: 连接获取超时时间（秒）
        """
        self.uri = uri or f"bolt://{settings.NEO4J_HOST}:{settings.NEO4J_PORT}"
        self.username = username or settings.NEO4J_USER
        self.password = password or settings.NEO4J_PASSWORD
        self.database = database or settings.NEO4J_DATABASE
        
        # 连接配置
        self.max_connection_lifetime = max_connection_lifetime
        self.max_connection_pool_size = max_connection_pool_size
        self.connection_acquisition_timeout = connection_acquisition_timeout
        
        # 驱动实例
        self.driver: Optional[AsyncDriver] = None
        self.is_connected = False
        
        # 连接统计
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "failed_connections": 0,
            "last_connection_time": None,
            "last_error": None
        }
        
        logger.info(f"图数据库连接管理器初始化 - URI: {self.uri}, 数据库: {self.database}")
    
    async def connect(self) -> bool:
        """
        连接到 Neo4j 数据库
        
        Returns:
            bool: 连接是否成功
            
        Raises:
            GraphConnectionError: 连接失败
        """
        try:
            if self.is_connected and self.driver:
                logger.debug("数据库已连接")
                return True
            
            logger.info("正在连接到 Neo4j 数据库...")
            
            # 创建驱动
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_lifetime=self.max_connection_lifetime,
                max_connection_pool_size=self.max_connection_pool_size,
                connection_acquisition_timeout=self.connection_acquisition_timeout
            )
            
            # 验证连接
            await self.driver.verify_connectivity()
            
            # 测试查询
            async with self.driver.session(database=self.database) as session:
                result = await session.run("RETURN 1 as test")
                record = await result.single()
                
                if not record or record["test"] != 1:
                    raise GraphConnectionError("连接测试失败")
            
            self.is_connected = True
            self.connection_stats["total_connections"] += 1
            self.connection_stats["last_connection_time"] = datetime.utcnow().isoformat()
            
            logger.info("Neo4j 数据库连接成功")
            return True
            
        except ServiceUnavailable as e:
            error_msg = f"Neo4j 服务不可用: {str(e)}"
            logger.error(error_msg)
            self.connection_stats["failed_connections"] += 1
            self.connection_stats["last_error"] = error_msg
            raise GraphConnectionError(error_msg)
        except AuthError as e:
            error_msg = f"Neo4j 认证失败: {str(e)}"
            logger.error(error_msg)
            self.connection_stats["failed_connections"] += 1
            self.connection_stats["last_error"] = error_msg
            raise GraphConnectionError(error_msg)
        except Exception as e:
            error_msg = f"Neo4j 连接失败: {str(e)}"
            logger.error(error_msg)
            self.connection_stats["failed_connections"] += 1
            self.connection_stats["last_error"] = error_msg
            raise GraphConnectionError(error_msg)
    
    async def disconnect(self):
        """断开数据库连接"""
        try:
            if self.driver:
                await self.driver.close()
                self.driver = None
                self.is_connected = False
                logger.info("Neo4j 数据库连接已断开")
        except Exception as e:
            logger.error(f"断开连接失败: {str(e)}")
    
    async def get_session(self) -> AsyncSession:
        """
        获取数据库会话
        
        Returns:
            AsyncSession: 数据库会话
            
        Raises:
            GraphConnectionError: 连接失败
        """
        if not self.is_connected or not self.driver:
            await self.connect()
        
        if not self.driver:
            raise GraphConnectionError("无法获取数据库驱动")
        
        return self.driver.session(database=self.database)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            Dict[str, Any]: 健康状态信息
        """
        try:
            if not self.is_connected:
                await self.connect()
            
            # 执行测试查询
            session = await self.get_session()
            try:
                start_time = time.time()
                result = await session.run("RETURN 1 as test")
                record = await result.single()
                query_time = time.time() - start_time
            finally:
                await session.close()
                
                if record and record["test"] == 1:
                    return {
                        "status": "healthy",
                        "connected": True,
                        "query_time": query_time,
                        "database": self.database,
                        "uri": self.uri,
                        "stats": self.connection_stats,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "connected": False,
                        "error": "测试查询失败",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


class GraphNodeManager:
    """
    图节点管理器
    
    管理图数据库中的节点操作。
    """
    
    def __init__(self, connection_manager: GraphConnectionManager):
        """
        初始化节点管理器
        
        Args:
            connection_manager: 连接管理器
        """
        self.connection_manager = connection_manager
        
        # 节点标签映射
        self.label_mapping = {
            "entity": "Entity",
            "document": "Document",
            "chunk": "Chunk",
            "person": "Person",
            "organization": "Organization",
            "concept": "Concept",
            "location": "Location"
        }
    
    async def create_node(
        self,
        labels: Union[str, List[str]],
        properties: Dict[str, Any],
        node_id: Optional[str] = None
    ) -> GraphNode:
        """
        创建节点
        
        Args:
            labels: 节点标签
            properties: 节点属性
            node_id: 节点 ID（可选）
            
        Returns:
            GraphNode: 创建的节点
            
        Raises:
            GraphQueryError: 查询失败
        """
        try:
            # 处理标签
            if isinstance(labels, str):
                labels = [labels]
            
            # 生成节点 ID
            if not node_id:
                node_id = str(uuid4())
            
            # 添加系统属性
            properties = properties.copy()
            properties["id"] = node_id
            properties["created_at"] = datetime.utcnow().isoformat()
            properties["updated_at"] = properties["created_at"]
            
            # 构建 Cypher 查询
            labels_str = ":".join(labels)
            query = f"""
            CREATE (n:{labels_str} $properties)
            RETURN n
            """
            
            # 执行创建节点查询
            session = await self.connection_manager.get_session()
            try:
                result = await session.run(query, properties=properties)
                record = await result.single()
                
                if not record:
                    raise GraphQueryError("节点创建失败")
                
                node_data = dict(record["n"])
                
                return GraphNode(
                    id=node_data["id"],
                    labels=labels,
                    properties=node_data,
                    created_at=node_data.get("created_at"),
                    updated_at=node_data.get("updated_at")
                )
            finally:
                await session.close()
                
        except Exception as e:
            logger.error(f"创建节点失败: {str(e)}")
            raise GraphQueryError(f"创建节点失败: {str(e)}")
    
    async def get_node(self, node_id: str) -> Optional[GraphNode]:
        """
        获取节点
        
        Args:
            node_id: 节点 ID
            
        Returns:
            Optional[GraphNode]: 节点信息
        """
        try:
            query = """
            MATCH (n {id: $node_id})
            RETURN n, labels(n) as labels
            """
            
            session = await self.connection_manager.get_session()
            try:
                result = await session.run(query, node_id=node_id)
                record = await result.single()
                
                if not record:
                    return None
                
                node_data = dict(record["n"])
                labels = record["labels"]
                
                return GraphNode(
                    id=node_data["id"],
                    labels=labels,
                    properties=node_data,
                    created_at=node_data.get("created_at"),
                    updated_at=node_data.get("updated_at")
                )
            finally:
                await session.close()
                
        except Exception as e:
            logger.error(f"获取节点失败: {str(e)}")
            raise GraphQueryError(f"获取节点失败: {str(e)}")
    
    async def update_node(
        self,
        node_id: str,
        properties: Dict[str, Any],
        merge: bool = True
    ) -> Optional[GraphNode]:
        """
        更新节点
        
        Args:
            node_id: 节点 ID
            properties: 更新的属性
            merge: 是否合并属性（True）还是替换（False）
            
        Returns:
            Optional[GraphNode]: 更新后的节点
        """
        try:
            # 添加更新时间
            properties = properties.copy()
            properties["updated_at"] = datetime.utcnow().isoformat()
            
            if merge:
                # 合并属性
                set_clauses = []
                for key, value in properties.items():
                    set_clauses.append(f"n.{key} = ${key}")
                
                query = f"""
                MATCH (n {{id: $node_id}})
                SET {', '.join(set_clauses)}
                RETURN n, labels(n) as labels
                """
                
                params = {"node_id": node_id, **properties}
            else:
                # 替换属性
                query = """
                MATCH (n {id: $node_id})
                SET n = $properties
                RETURN n, labels(n) as labels
                """
                
                params = {"node_id": node_id, "properties": properties}
            
            session = await self.connection_manager.get_session()

            
            try:
                result = await session.run(query, **params)
                record = await result.single()
                
                if not record:
                    return None
                
                node_data = dict(record["n"])
                labels = record["labels"]
                
                return GraphNode(
                    id=node_data["id"],
                    labels=labels,
                    properties=node_data,
                    created_at=node_data.get("created_at"),
                    updated_at=node_data.get("updated_at")
                )
            except Exception as e:
                logger.error(f"更新节点失败: {str(e)}")
                raise GraphQueryError(f"更新节点失败: {str(e)}")
            finally:
                await session.close()
                
        except Exception as e:
            logger.error(f"更新节点失败: {str(e)}")
            raise GraphQueryError(f"更新节点失败: {str(e)}")
    
    async def delete_node(self, node_id: str, delete_relationships: bool = True) -> bool:
        """
        删除节点
        
        Args:
            node_id: 节点 ID
            delete_relationships: 是否同时删除关系
            
        Returns:
            bool: 删除是否成功
        """
        try:
            if delete_relationships:
                query = """
                MATCH (n {id: $node_id})
                DETACH DELETE n
                """
            else:
                query = """
                MATCH (n {id: $node_id})
                DELETE n
                """
            
            session = await self.connection_manager.get_session()

            
            try:
                result = await session.run(query, node_id=node_id)
                summary = await result.consume()
                
                return summary.counters.nodes_deleted > 0
            except Exception as e:
                logger.error(f"删除节点失败: {str(e)}")
                raise GraphQueryError(f"删除节点失败: {str(e)}")
            finally:
                await session.close()
                
        except Exception as e:
            logger.error(f"删除节点失败: {str(e)}")
            raise GraphQueryError(f"删除节点失败: {str(e)}")
    
    async def find_nodes(
        self,
        labels: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        skip: int = 0
    ) -> List[GraphNode]:
        """
        查找节点
        
        Args:
            labels: 节点标签过滤
            properties: 属性过滤
            limit: 限制数量
            skip: 跳过数量
            
        Returns:
            List[GraphNode]: 节点列表
        """
        try:
            # 构建查询条件
            match_parts = []
            where_parts = []
            params = {"limit": limit, "skip": skip}
            
            # 标签条件
            if labels:
                labels_str = ":".join(labels)
                match_parts.append(f"(n:{labels_str})")
            else:
                match_parts.append("(n)")
            
            # 属性条件
            if properties:
                for key, value in properties.items():
                    where_parts.append(f"n.{key} = ${key}")
                    params[key] = value
            
            # 构建完整查询
            query_parts = [f"MATCH {' '.join(match_parts)}"]
            
            if where_parts:
                query_parts.append(f"WHERE {' AND '.join(where_parts)}")
            
            query_parts.extend([
                "RETURN n, labels(n) as labels",
                "SKIP $skip",
                "LIMIT $limit"
            ])
            
            query = "\n".join(query_parts)
            
            session = await self.connection_manager.get_session()

            
            try:
                result = await session.run(query, **params)
                records = await result.data()
                
                nodes = []
                for record in records:
                    node_data = dict(record["n"])
                    labels = record["labels"]
                    
                    node = GraphNode(
                        id=node_data["id"],
                        labels=labels,
                        properties=node_data,
                        created_at=node_data.get("created_at"),
                        updated_at=node_data.get("updated_at")
                    )
                    nodes.append(node)
                
                return nodes
            except Exception as e:
                logger.error(f"查找节点失败: {str(e)}")
                raise GraphQueryError(f"查找节点失败: {str(e)}")
            finally:
                await session.close()
                
        except Exception as e:
            logger.error(f"查找节点失败: {str(e)}")
            raise GraphQueryError(f"查找节点失败: {str(e)}")


class GraphRelationshipManager:
    """
    图关系管理器
    
    管理图数据库中的关系操作。
    """
    
    def __init__(self, connection_manager: GraphConnectionManager):
        """
        初始化关系管理器
        
        Args:
            connection_manager: 连接管理器
        """
        self.connection_manager = connection_manager
    
    async def create_relationship(
        self,
        start_node_id: str,
        end_node_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
        relationship_id: Optional[str] = None
    ) -> GraphRelationship:
        """
        创建关系
        
        Args:
            start_node_id: 起始节点 ID
            end_node_id: 结束节点 ID
            relationship_type: 关系类型
            properties: 关系属性
            relationship_id: 关系 ID（可选）
            
        Returns:
            GraphRelationship: 创建的关系
        """
        try:
            # 生成关系 ID
            if not relationship_id:
                relationship_id = str(uuid4())
            
            # 处理属性
            if not properties:
                properties = {}
            
            properties = properties.copy()
            properties["id"] = relationship_id
            properties["created_at"] = datetime.utcnow().isoformat()
            properties["updated_at"] = properties["created_at"]
            
            # 构建查询
            query = f"""
            MATCH (start {{id: $start_node_id}}), (end {{id: $end_node_id}})
            CREATE (start)-[r:{relationship_type} $properties]->(end)
            RETURN r
            """
            
            params = {
                "start_node_id": start_node_id,
                "end_node_id": end_node_id,
                "properties": properties
            }
            
            session = await self.connection_manager.get_session()

            
            try:
                result = await session.run(query, **params)
                record = await result.single()
                
                if not record:
                    raise GraphQueryError("关系创建失败")
                
                rel_data = dict(record["r"])
                
                return GraphRelationship(
                    id=rel_data["id"],
                    type=relationship_type,
                    start_node_id=start_node_id,
                    end_node_id=end_node_id,
                    properties=rel_data,
                    created_at=rel_data.get("created_at"),
                    updated_at=rel_data.get("updated_at")
                )
            except Exception as e:
                logger.error(f"创建关系失败: {str(e)}")
                raise GraphQueryError(f"创建关系失败: {str(e)}")
            finally:
                await session.close()
                
        except Exception as e:
            logger.error(f"创建关系失败: {str(e)}")
            raise GraphQueryError(f"创建关系失败: {str(e)}")
    
    async def get_relationship(self, relationship_id: str) -> Optional[GraphRelationship]:
        """
        获取关系
        
        Args:
            relationship_id: 关系 ID
            
        Returns:
            Optional[GraphRelationship]: 关系信息
        """
        try:
            query = """
            MATCH (start)-[r {id: $relationship_id}]->(end)
            RETURN r, type(r) as rel_type, start.id as start_id, end.id as end_id
            """
            
            session = await self.connection_manager.get_session()

            
            try:
                result = await session.run(query, relationship_id=relationship_id)
                record = await result.single()
                
                if not record:
                    return None
                
                rel_data = dict(record["r"])
                
                return GraphRelationship(
                    id=rel_data["id"],
                    type=record["rel_type"],
                    start_node_id=record["start_id"],
                    end_node_id=record["end_id"],
                    properties=rel_data,
                    created_at=rel_data.get("created_at"),
                    updated_at=rel_data.get("updated_at")
                )
            except Exception as e:
                logger.error(f"获取关系失败: {str(e)}")
                raise GraphQueryError(f"获取关系失败: {str(e)}")
            finally:
                await session.close()
                
        except Exception as e:
            logger.error(f"获取关系失败: {str(e)}")
            raise GraphQueryError(f"获取关系失败: {str(e)}")
    
    async def update_relationship(
        self,
        relationship_id: str,
        properties: Dict[str, Any],
        merge: bool = True
    ) -> Optional[GraphRelationship]:
        """
        更新关系
        
        Args:
            relationship_id: 关系 ID
            properties: 更新的属性
            merge: 是否合并属性
            
        Returns:
            Optional[GraphRelationship]: 更新后的关系
        """
        try:
            # 添加更新时间
            properties = properties.copy()
            properties["updated_at"] = datetime.utcnow().isoformat()
            
            if merge:
                # 合并属性
                set_clauses = []
                for key, value in properties.items():
                    set_clauses.append(f"r.{key} = ${key}")
                
                query = f"""
                MATCH (start)-[r {{id: $relationship_id}}]->(end)
                SET {', '.join(set_clauses)}
                RETURN r, type(r) as rel_type, start.id as start_id, end.id as end_id
                """
                
                params = {"relationship_id": relationship_id, **properties}
            else:
                # 替换属性
                query = """
                MATCH (start)-[r {id: $relationship_id}]->(end)
                SET r = $properties
                RETURN r, type(r) as rel_type, start.id as start_id, end.id as end_id
                """
                
                params = {"relationship_id": relationship_id, "properties": properties}
            
            session = await self.connection_manager.get_session()

            
            try:
                result = await session.run(query, **params)
                record = await result.single()
                
                if not record:
                    return None
                
                rel_data = dict(record["r"])
                
                return GraphRelationship(
                    id=rel_data["id"],
                    type=record["rel_type"],
                    start_node_id=record["start_id"],
                    end_node_id=record["end_id"],
                    properties=rel_data,
                    created_at=rel_data.get("created_at"),
                    updated_at=rel_data.get("updated_at")
                )
            except Exception as e:
                logger.error(f"更新关系失败: {str(e)}")
                raise GraphQueryError(f"更新关系失败: {str(e)}")
                
        except Exception as e:
            logger.error(f"更新关系失败: {str(e)}")
            raise GraphQueryError(f"更新关系失败: {str(e)}")
        finally:
            await session.close()
    
    async def delete_relationship(self, relationship_id: str) -> bool:
        """
        删除关系
        
        Args:
            relationship_id: 关系 ID
            
        Returns:
            bool: 删除是否成功
        """
        try:
            query = """
            MATCH ()-[r {id: $relationship_id}]->()
            DELETE r
            """
            
            session = await self.connection_manager.get_session()

            
            try:
                result = await session.run(query, relationship_id=relationship_id)
                summary = await result.consume()
                
                return summary.counters.relationships_deleted > 0
            except Exception as e:
                logger.error(f"删除关系失败: {str(e)}")
                raise GraphQueryError(f"删除关系失败: {str(e)}")
            finally:
                await session.close()
                
        except Exception as e:
            logger.error(f"删除关系失败: {str(e)}")
            raise GraphQueryError(f"删除关系失败: {str(e)}")
    
    async def find_relationships(
        self,
        start_node_id: Optional[str] = None,
        end_node_id: Optional[str] = None,
        relationship_types: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        skip: int = 0
    ) -> List[GraphRelationship]:
        """
        查找关系
        
        Args:
            start_node_id: 起始节点 ID
            end_node_id: 结束节点 ID
            relationship_types: 关系类型过滤
            properties: 属性过滤
            limit: 限制数量
            skip: 跳过数量
            
        Returns:
            List[GraphRelationship]: 关系列表
        """
        try:
            # 构建查询条件
            match_parts = []
            where_parts = []
            params = {"limit": limit, "skip": skip}
            
            # 节点条件
            if start_node_id and end_node_id:
                match_parts.append("(start {id: $start_node_id})-[r]->(end {id: $end_node_id})")
                params["start_node_id"] = start_node_id
                params["end_node_id"] = end_node_id
            elif start_node_id:
                match_parts.append("(start {id: $start_node_id})-[r]->(end)")
                params["start_node_id"] = start_node_id
            elif end_node_id:
                match_parts.append("(start)-[r]->(end {id: $end_node_id})")
                params["end_node_id"] = end_node_id
            else:
                match_parts.append("(start)-[r]->(end)")
            
            # 关系类型条件
            if relationship_types:
                type_conditions = []
                for i, rel_type in enumerate(relationship_types):
                    type_param = f"rel_type_{i}"
                    type_conditions.append(f"type(r) = ${type_param}")
                    params[type_param] = rel_type
                
                where_parts.append(f"({' OR '.join(type_conditions)})")
            
            # 属性条件
            if properties:
                for key, value in properties.items():
                    where_parts.append(f"r.{key} = ${key}")
                    params[key] = value
            
            # 构建完整查询
            query_parts = [f"MATCH {' '.join(match_parts)}"]
            
            if where_parts:
                query_parts.append(f"WHERE {' AND '.join(where_parts)}")
            
            query_parts.extend([
                "RETURN r, type(r) as rel_type, start.id as start_id, end.id as end_id",
                "SKIP $skip",
                "LIMIT $limit"
            ])
            
            query = "\n".join(query_parts)
            
            session = await self.connection_manager.get_session()

            
            try:
                result = await session.run(query, **params)
                records = await result.data()
                
                relationships = []
                for record in records:
                    rel_data = dict(record["r"])
                    
                    relationship = GraphRelationship(
                        id=rel_data["id"],
                        type=record["rel_type"],
                        start_node_id=record["start_id"],
                        end_node_id=record["end_id"],
                        properties=rel_data,
                        created_at=rel_data.get("created_at"),
                        updated_at=rel_data.get("updated_at")
                    )
                    relationships.append(relationship)
                
                return relationships
            except Exception as e:
                logger.error(f"查找关系失败: {str(e)}")
                raise GraphQueryError(f"查找关系失败: {str(e)}")
            finally:
                await session.close()
                
        except Exception as e:
            logger.error(f"查找关系失败: {str(e)}")
            raise GraphQueryError(f"查找关系失败: {str(e)}")


class GraphService:
    """
    图数据库服务
    
    提供完整的图数据库操作功能。
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None
    ):
        """
        初始化图服务
        
        Args:
            uri: Neo4j 连接 URI
            username: 用户名
            password: 密码
            database: 数据库名称
        """
        # 初始化连接管理器
        self.connection_manager = GraphConnectionManager(
            uri=uri,
            username=username,
            password=password,
            database=database
        )
        
        # 初始化管理器
        self.node_manager = GraphNodeManager(self.connection_manager)
        self.relationship_manager = GraphRelationshipManager(self.connection_manager)
        
        # 服务统计
        self.stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "node_operations": 0,
            "relationship_operations": 0,
            "query_operations": 0,
            "last_operation_time": None
        }
        
        logger.info("图数据库服务初始化完成")
    
    async def connect(self) -> bool:
        """连接到数据库"""
        return await self.connection_manager.connect()
    
    async def disconnect(self):
        """断开数据库连接"""
        await self.connection_manager.disconnect()
    
    # 节点操作
    async def create_node(
        self,
        labels: Union[str, List[str]],
        properties: Dict[str, Any],
        node_id: Optional[str] = None
    ) -> GraphNode:
        """创建节点"""
        try:
            self.stats["total_operations"] += 1
            self.stats["node_operations"] += 1
            
            result = await self.node_manager.create_node(labels, properties, node_id)
            
            self.stats["successful_operations"] += 1
            self.stats["last_operation_time"] = datetime.utcnow().isoformat()
            
            return result
        except Exception as e:
            self.stats["failed_operations"] += 1
            raise
    
    async def get_node(self, node_id: str) -> Optional[GraphNode]:
        """获取节点"""
        try:
            self.stats["total_operations"] += 1
            self.stats["query_operations"] += 1
            
            result = await self.node_manager.get_node(node_id)
            
            self.stats["successful_operations"] += 1
            self.stats["last_operation_time"] = datetime.utcnow().isoformat()
            
            return result
        except Exception as e:
            self.stats["failed_operations"] += 1
            raise
    
    async def update_node(
        self,
        node_id: str,
        properties: Dict[str, Any],
        merge: bool = True
    ) -> Optional[GraphNode]:
        """更新节点"""
        try:
            self.stats["total_operations"] += 1
            self.stats["node_operations"] += 1
            
            result = await self.node_manager.update_node(node_id, properties, merge)
            
            self.stats["successful_operations"] += 1
            self.stats["last_operation_time"] = datetime.utcnow().isoformat()
            
            return result
        except Exception as e:
            self.stats["failed_operations"] += 1
            raise
    
    async def delete_node(self, node_id: str, delete_relationships: bool = True) -> bool:
        """删除节点"""
        try:
            self.stats["total_operations"] += 1
            self.stats["node_operations"] += 1
            
            result = await self.node_manager.delete_node(node_id, delete_relationships)
            
            self.stats["successful_operations"] += 1
            self.stats["last_operation_time"] = datetime.utcnow().isoformat()
            
            return result
        except Exception as e:
            self.stats["failed_operations"] += 1
            raise
    
    async def find_nodes(
        self,
        labels: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        skip: int = 0
    ) -> List[GraphNode]:
        """查找节点"""
        try:
            self.stats["total_operations"] += 1
            self.stats["query_operations"] += 1
            
            result = await self.node_manager.find_nodes(labels, properties, limit, skip)
            
            self.stats["successful_operations"] += 1
            self.stats["last_operation_time"] = datetime.utcnow().isoformat()
            
            return result
        except Exception as e:
            self.stats["failed_operations"] += 1
            raise
    
    # 关系操作
    async def create_relationship(
        self,
        start_node_id: str,
        end_node_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
        relationship_id: Optional[str] = None
    ) -> GraphRelationship:
        """创建关系"""
        try:
            self.stats["total_operations"] += 1
            self.stats["relationship_operations"] += 1
            
            result = await self.relationship_manager.create_relationship(
                start_node_id, end_node_id, relationship_type, properties, relationship_id
            )
            
            self.stats["successful_operations"] += 1
            self.stats["last_operation_time"] = datetime.utcnow().isoformat()
            
            return result
        except Exception as e:
            self.stats["failed_operations"] += 1
            raise
    
    async def get_relationship(self, relationship_id: str) -> Optional[GraphRelationship]:
        """获取关系"""
        try:
            self.stats["total_operations"] += 1
            self.stats["query_operations"] += 1
            
            result = await self.relationship_manager.get_relationship(relationship_id)
            
            self.stats["successful_operations"] += 1
            self.stats["last_operation_time"] = datetime.utcnow().isoformat()
            
            return result
        except Exception as e:
            self.stats["failed_operations"] += 1
            raise
    
    async def update_relationship(
        self,
        relationship_id: str,
        properties: Dict[str, Any],
        merge: bool = True
    ) -> Optional[GraphRelationship]:
        """更新关系"""
        try:
            self.stats["total_operations"] += 1
            self.stats["relationship_operations"] += 1
            
            result = await self.relationship_manager.update_relationship(
                relationship_id, properties, merge
            )
            
            self.stats["successful_operations"] += 1
            self.stats["last_operation_time"] = datetime.utcnow().isoformat()
            
            return result
        except Exception as e:
            self.stats["failed_operations"] += 1
            raise
    
    async def delete_relationship(self, relationship_id: str) -> bool:
        """删除关系"""
        try:
            self.stats["total_operations"] += 1
            self.stats["relationship_operations"] += 1
            
            result = await self.relationship_manager.delete_relationship(relationship_id)
            
            self.stats["successful_operations"] += 1
            self.stats["last_operation_time"] = datetime.utcnow().isoformat()
            
            return result
        except Exception as e:
            self.stats["failed_operations"] += 1
            raise
    
    async def find_relationships(
        self,
        start_node_id: Optional[str] = None,
        end_node_id: Optional[str] = None,
        relationship_types: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        skip: int = 0
    ) -> List[GraphRelationship]:
        """查找关系"""
        try:
            self.stats["total_operations"] += 1
            self.stats["query_operations"] += 1
            
            result = await self.relationship_manager.find_relationships(
                start_node_id, end_node_id, relationship_types, properties, limit, skip
            )
            
            self.stats["successful_operations"] += 1
            self.stats["last_operation_time"] = datetime.utcnow().isoformat()
            
            return result
        except Exception as e:
            self.stats["failed_operations"] += 1
            raise
    
    # 高级查询操作
    async def execute_cypher(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        执行 Cypher 查询
        
        Args:
            query: Cypher 查询语句
            parameters: 查询参数
            
        Returns:
            List[Dict[str, Any]]: 查询结果
        """
        try:
            self.stats["total_operations"] += 1
            self.stats["query_operations"] += 1
            
            if not parameters:
                parameters = {}
            
            session = await self.connection_manager.get_session()

            
            try:
                result = await session.run(query, **parameters)
                records = await result.data()
                
                self.stats["successful_operations"] += 1
                self.stats["last_operation_time"] = datetime.utcnow().isoformat()
                
                return records
            except Exception as e:
                self.stats["failed_operations"] += 1
                logger.error(f"Cypher 查询执行失败: {str(e)}")
                raise GraphQueryError(f"Cypher 查询执行失败: {str(e)}")
            finally:
                await session.close()
                
        except Exception as e:
            self.stats["failed_operations"] += 1
            logger.error(f"Cypher 查询执行失败: {str(e)}")
            raise GraphQueryError(f"Cypher 查询执行失败: {str(e)}")
    
    async def get_graph_stats(self) -> GraphStats:
        """
        获取图统计信息
        
        Returns:
            GraphStats: 图统计信息
        """
        try:
            # 节点统计
            node_count_query = "MATCH (n) RETURN count(n) as total_nodes"
            node_result = await self.execute_cypher(node_count_query)
            total_nodes = node_result[0]["total_nodes"] if node_result else 0
            
            # 关系统计
            rel_count_query = "MATCH ()-[r]->() RETURN count(r) as total_relationships"
            rel_result = await self.execute_cypher(rel_count_query)
            total_relationships = rel_result[0]["total_relationships"] if rel_result else 0
            
            # 节点类型统计
            node_types_query = """
            MATCH (n)
            UNWIND labels(n) as label
            RETURN label, count(*) as count
            ORDER BY count DESC
            """
            node_types_result = await self.execute_cypher(node_types_query)
            node_types = {record["label"]: record["count"] for record in node_types_result}
            
            # 关系类型统计
            rel_types_query = """
            MATCH ()-[r]->()
            RETURN type(r) as rel_type, count(*) as count
            ORDER BY count DESC
            """
            rel_types_result = await self.execute_cypher(rel_types_query)
            relationship_types = {record["rel_type"]: record["count"] for record in rel_types_result}
            
            return GraphStats(
                total_nodes=total_nodes,
                total_relationships=total_relationships,
                node_types=node_types,
                relationship_types=relationship_types,
                database_size="N/A",  # Neo4j 不直接提供数据库大小
                last_updated=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            logger.error(f"获取图统计信息失败: {str(e)}")
            raise GraphQueryError(f"获取图统计信息失败: {str(e)}")
    
    async def create_indexes(self) -> Dict[str, bool]:
        """
        创建推荐的索引
        
        Returns:
            Dict[str, bool]: 索引创建结果
        """
        try:
            indexes = {
                "entity_id_index": "CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.id)",
                "entity_name_index": "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                "document_id_index": "CREATE INDEX document_id_index IF NOT EXISTS FOR (d:Document) ON (d.id)",
                "chunk_id_index": "CREATE INDEX chunk_id_index IF NOT EXISTS FOR (c:Chunk) ON (c.id)",
                "relationship_id_index": "CREATE INDEX relationship_id_index IF NOT EXISTS FOR ()-[r]-() ON (r.id)"
            }
            
            results = {}
            
            for index_name, query in indexes.items():
                try:
                    await self.execute_cypher(query)
                    results[index_name] = True
                    logger.info(f"索引 {index_name} 创建成功")
                except Exception as e:
                    results[index_name] = False
                    logger.error(f"索引 {index_name} 创建失败: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"创建索引失败: {str(e)}")
            raise GraphIndexError(f"创建索引失败: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        stats = self.stats.copy()
        
        # 计算成功率
        if stats["total_operations"] > 0:
            stats["success_rate"] = stats["successful_operations"] / stats["total_operations"]
        else:
            stats["success_rate"] = 0.0
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 连接健康检查
            connection_health = await self.connection_manager.health_check()
            
            if connection_health["status"] != "healthy":
                return {
                    "status": "unhealthy",
                    "connection": connection_health,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # 基本操作测试
            try:
                # 测试节点操作
                test_node = await self.create_node(
                    labels=["Test"],
                    properties={"name": "health_check_test", "test": True}
                )
                
                # 测试查询操作
                found_node = await self.get_node(test_node.id)
                
                # 清理测试节点
                await self.delete_node(test_node.id)
                
                if found_node and found_node.id == test_node.id:
                    return {
                        "status": "healthy",
                        "connection": connection_health,
                        "operations": "working",
                        "stats": self.get_stats(),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                else:
                    return {
                        "status": "degraded",
                        "connection": connection_health,
                        "operations": "partial",
                        "error": "节点操作测试失败",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
            except Exception as e:
                return {
                    "status": "degraded",
                    "connection": connection_health,
                    "operations": "failed",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }