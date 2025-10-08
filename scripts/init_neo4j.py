#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neo4j 图数据库初始化脚本
========================

本脚本用于初始化 Neo4j 图数据库，包括：
- 创建数据库结构
- 设置索引和约束
- 创建基础节点类型
- 设置图模式
- 初始化系统配置

功能特性：
- 自动检测数据库状态
- 幂等性操作（可重复执行）
- 详细的日志记录
- 错误处理和回滚
- 性能优化配置

支持的节点类型：
- Document: 文档节点
- Chunk: 文本块节点
- Entity: 实体节点
- Person: 人物实体
- Organization: 组织实体
- Concept: 概念实体
- Location: 地点实体
- Event: 事件实体

支持的关系类型：
- CONTAINS: 包含关系
- MENTIONS: 提及关系
- RELATES_TO: 相关关系
- PART_OF: 部分关系
- LOCATED_IN: 位于关系
- WORKS_FOR: 工作关系
- PARTICIPATES_IN: 参与关系

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import asyncio
import sys
import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import get_settings
from app.services.graph_service import GraphService, GraphConnectionManager
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Neo4jInitializer:
    """
    Neo4j 数据库初始化器
    
    负责初始化图数据库的结构和配置。
    """
    
    def __init__(self):
        """初始化 Neo4j 初始化器"""
        self.settings = get_settings()
        self.connection_manager = None
        self.graph_service = None
        
        # 初始化统计
        self.stats = {
            "start_time": None,
            "end_time": None,
            "duration": 0,
            "constraints_created": 0,
            "indexes_created": 0,
            "nodes_created": 0,
            "relationships_created": 0,
            "errors": []
        }
        
        # 节点标签定义
        self.node_labels = [
            "Document",
            "Chunk", 
            "Entity",
            "Person",
            "Organization",
            "Concept",
            "Location",
            "Event",
            "Topic",
            "Keyword"
        ]
        
        # 关系类型定义
        self.relationship_types = [
            "CONTAINS",
            "MENTIONS", 
            "RELATES_TO",
            "PART_OF",
            "LOCATED_IN",
            "WORKS_FOR",
            "PARTICIPATES_IN",
            "SIMILAR_TO",
            "REFERENCES",
            "CITES",
            "FOLLOWS",
            "PRECEDES"
        ]
        
        # 约束定义
        self.constraints = [
            # 唯一性约束
            ("Document", "id", "UNIQUE"),
            ("Document", "file_path", "UNIQUE"),
            ("Chunk", "id", "UNIQUE"),
            ("Entity", "id", "UNIQUE"),
            ("Person", "id", "UNIQUE"),
            ("Organization", "id", "UNIQUE"),
            ("Concept", "id", "UNIQUE"),
            ("Location", "id", "UNIQUE"),
            ("Event", "id", "UNIQUE"),
            ("Topic", "id", "UNIQUE"),
            ("Keyword", "id", "UNIQUE"),
            
            # 存在性约束
            ("Document", "title", "EXISTS"),
            ("Chunk", "content", "EXISTS"),
            ("Entity", "name", "EXISTS"),
            ("Person", "name", "EXISTS"),
            ("Organization", "name", "EXISTS"),
            ("Concept", "name", "EXISTS")
        ]
        
        # 索引定义
        self.indexes = [
            # 单属性索引
            ("Document", ["title"]),
            ("Document", ["created_at"]),
            ("Document", ["file_type"]),
            ("Chunk", ["chunk_index"]),
            ("Chunk", ["created_at"]),
            ("Entity", ["name"]),
            ("Entity", ["entity_type"]),
            ("Entity", ["confidence"]),
            ("Person", ["name"]),
            ("Organization", ["name"]),
            ("Concept", ["name"]),
            ("Location", ["name"]),
            ("Event", ["name"]),
            ("Topic", ["name"]),
            ("Keyword", ["name"]),
            
            # 复合索引
            ("Document", ["file_type", "created_at"]),
            ("Entity", ["entity_type", "confidence"]),
            ("Chunk", ["document_id", "chunk_index"])
        ]
    
    async def initialize(self) -> Dict[str, Any]:
        """
        执行完整的数据库初始化
        
        Returns:
            Dict[str, Any]: 初始化结果统计
        """
        try:
            self.stats["start_time"] = datetime.utcnow()
            logger.info("开始初始化 Neo4j 图数据库...")
            
            # 1. 建立连接
            await self._setup_connection()
            
            # 2. 检查数据库状态
            await self._check_database_status()
            
            # 3. 清理旧数据（如果需要）
            # await self._cleanup_if_needed()
            
            # 4. 创建约束
            await self._create_constraints()
            
            # 5. 创建索引
            await self._create_indexes()
            
            # 6. 创建基础节点
            await self._create_base_nodes()
            
            # 7. 设置图模式
            await self._setup_graph_schema()
            
            # 8. 验证初始化结果
            await self._verify_initialization()
            
            self.stats["end_time"] = datetime.utcnow()
            self.stats["duration"] = (
                self.stats["end_time"] - self.stats["start_time"]
            ).total_seconds()
            
            logger.info(f"Neo4j 数据库初始化完成，耗时 {self.stats['duration']:.2f} 秒")
            
            return self.stats
            
        except Exception as e:
            self.stats["errors"].append(str(e))
            logger.error(f"Neo4j 数据库初始化失败: {str(e)}")
            raise
        finally:
            # 清理连接
            if self.connection_manager:
                await self.connection_manager.close()
    
    async def _setup_connection(self):
        """建立数据库连接"""
        try:
            logger.info("建立 Neo4j 数据库连接...")
            
            self.connection_manager = GraphConnectionManager(
                uri=f"bolt://{self.settings.NEO4J_HOST}:{self.settings.NEO4J_BOLT_PORT}",
                user=self.settings.NEO4J_USER,
                password=self.settings.NEO4J_PASSWORD,
                database=self.settings.NEO4J_DATABASE
            )
            
            await self.connection_manager.connect()
            
            self.graph_service = GraphService(self.connection_manager)
            
            logger.info("Neo4j 数据库连接建立成功")
            
        except Exception as e:
            logger.error(f"建立 Neo4j 连接失败: {str(e)}")
            raise
    
    async def _check_database_status(self):
        """检查数据库状态"""
        try:
            logger.info("检查数据库状态...")
            
            # 检查数据库版本
            version_query = "CALL dbms.components() YIELD name, versions RETURN name, versions"
            version_result = await self.graph_service.execute_cypher(version_query)
            
            if version_result:
                for record in version_result:
                    if record["name"] == "Neo4j Kernel":
                        logger.info(f"Neo4j 版本: {record['versions'][0]}")
            
            # 检查现有数据
            stats_query = """
            CALL {
                MATCH (n) RETURN count(n) as node_count
            }
            CALL {
                MATCH ()-[r]->() RETURN count(r) as rel_count
            }
            RETURN node_count, rel_count
            """
            
            stats_result = await self.graph_service.execute_cypher(stats_query)
            
            if stats_result:
                record = stats_result[0]
                node_count = record["node_count"]
                rel_count = record["rel_count"]
                
                logger.info(f"当前数据库状态 - 节点数: {node_count}, 关系数: {rel_count}")
                
                if node_count > 0 or rel_count > 0:
                    logger.warning("数据库中已存在数据，将进行增量初始化")
            
        except Exception as e:
            logger.error(f"检查数据库状态失败: {str(e)}")
            raise
    
    async def _create_constraints(self):
        """创建约束"""
        try:
            logger.info("创建数据库约束...")
            
            for label, property_name, constraint_type in self.constraints:
                try:
                    if constraint_type == "UNIQUE":
                        constraint_name = f"unique_{label.lower()}_{property_name}"
                        query = f"""
                        CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
                        FOR (n:{label})
                        REQUIRE n.{property_name} IS UNIQUE
                        """
                    elif constraint_type == "EXISTS":
                        constraint_name = f"exists_{label.lower()}_{property_name}"
                        query = f"""
                        CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
                        FOR (n:{label})
                        REQUIRE n.{property_name} IS NOT NULL
                        """
                    else:
                        logger.warning(f"不支持的约束类型: {constraint_type}")
                        continue
                    
                    await self.graph_service.execute_cypher(query)
                    self.stats["constraints_created"] += 1
                    
                    logger.debug(f"创建约束: {constraint_name}")
                    
                except Exception as e:
                    error_msg = f"创建约束失败 ({label}.{property_name}): {str(e)}"
                    logger.warning(error_msg)
                    self.stats["errors"].append(error_msg)
            
            logger.info(f"约束创建完成，共创建 {self.stats['constraints_created']} 个约束")
            
        except Exception as e:
            logger.error(f"创建约束失败: {str(e)}")
            raise
    
    async def _create_indexes(self):
        """创建索引"""
        try:
            logger.info("创建数据库索引...")
            
            for label, properties in self.indexes:
                try:
                    # 生成索引名称
                    props_str = "_".join(properties)
                    index_name = f"idx_{label.lower()}_{props_str}"
                    
                    # 构建索引查询
                    if len(properties) == 1:
                        # 单属性索引
                        query = f"""
                        CREATE INDEX {index_name} IF NOT EXISTS
                        FOR (n:{label})
                        ON (n.{properties[0]})
                        """
                    else:
                        # 复合索引
                        props_list = ", ".join([f"n.{prop}" for prop in properties])
                        query = f"""
                        CREATE INDEX {index_name} IF NOT EXISTS
                        FOR (n:{label})
                        ON ({props_list})
                        """
                    
                    await self.graph_service.execute_cypher(query)
                    self.stats["indexes_created"] += 1
                    
                    logger.debug(f"创建索引: {index_name}")
                    
                except Exception as e:
                    error_msg = f"创建索引失败 ({label}.{properties}): {str(e)}"
                    logger.warning(error_msg)
                    self.stats["errors"].append(error_msg)
            
            logger.info(f"索引创建完成，共创建 {self.stats['indexes_created']} 个索引")
            
        except Exception as e:
            logger.error(f"创建索引失败: {str(e)}")
            raise
    
    async def _create_base_nodes(self):
        """创建基础节点"""
        try:
            logger.info("创建基础系统节点...")
            
            # 创建系统配置节点
            system_config = {
                "id": "system_config",
                "name": "System Configuration",
                "version": "1.0.0",
                "created_at": datetime.utcnow().isoformat(),
                "description": "GraphRAG 系统配置节点"
            }
            
            config_query = """
            MERGE (config:SystemConfig {id: $id})
            SET config += $properties
            RETURN config
            """
            
            await self.graph_service.execute_cypher(
                config_query, 
                {"id": system_config["id"], "properties": system_config}
            )
            
            self.stats["nodes_created"] += 1
            
            # 创建根文档节点（用于组织文档层次结构）
            root_doc = {
                "id": "root_document",
                "title": "Root Document",
                "file_type": "system",
                "created_at": datetime.utcnow().isoformat(),
                "description": "根文档节点，用于组织文档层次结构"
            }
            
            root_query = """
            MERGE (root:Document {id: $id})
            SET root += $properties
            RETURN root
            """
            
            await self.graph_service.execute_cypher(
                root_query,
                {"id": root_doc["id"], "properties": root_doc}
            )
            
            self.stats["nodes_created"] += 1
            
            # 创建实体类型节点（用于实体分类）
            entity_types = [
                "Person", "Organization", "Location", "Event", 
                "Concept", "Topic", "Keyword", "Date", "Number"
            ]
            
            for entity_type in entity_types:
                type_node = {
                    "id": f"entity_type_{entity_type.lower()}",
                    "name": entity_type,
                    "category": "entity_type",
                    "created_at": datetime.utcnow().isoformat(),
                    "description": f"{entity_type} 实体类型"
                }
                
                type_query = """
                MERGE (et:EntityType {id: $id})
                SET et += $properties
                RETURN et
                """
                
                await self.graph_service.execute_cypher(
                    type_query,
                    {"id": type_node["id"], "properties": type_node}
                )
                
                self.stats["nodes_created"] += 1
            
            logger.info(f"基础节点创建完成，共创建 {self.stats['nodes_created']} 个节点")
            
        except Exception as e:
            logger.error(f"创建基础节点失败: {str(e)}")
            raise
    
    async def _setup_graph_schema(self):
        """设置图模式"""
        try:
            logger.info("设置图数据库模式...")
            
            # 创建关系类型定义节点
            for rel_type in self.relationship_types:
                rel_def = {
                    "id": f"rel_type_{rel_type.lower()}",
                    "name": rel_type,
                    "category": "relationship_type",
                    "created_at": datetime.utcnow().isoformat(),
                    "description": f"{rel_type} 关系类型定义"
                }
                
                rel_query = """
                MERGE (rt:RelationshipType {id: $id})
                SET rt += $properties
                RETURN rt
                """
                
                await self.graph_service.execute_cypher(
                    rel_query,
                    {"id": rel_def["id"], "properties": rel_def}
                )
                
                self.stats["nodes_created"] += 1
            
            # 创建模式关系（连接系统配置和类型定义）
            schema_queries = [
                # 系统配置 -> 实体类型
                """
                MATCH (config:SystemConfig {id: 'system_config'})
                MATCH (et:EntityType)
                MERGE (config)-[:DEFINES]->(et)
                """,
                
                # 系统配置 -> 关系类型
                """
                MATCH (config:SystemConfig {id: 'system_config'})
                MATCH (rt:RelationshipType)
                MERGE (config)-[:DEFINES]->(rt)
                """
            ]
            
            for query in schema_queries:
                result = await self.graph_service.execute_cypher(query)
                # 统计创建的关系数量
                if result:
                    self.stats["relationships_created"] += len(result)
            
            logger.info("图数据库模式设置完成")
            
        except Exception as e:
            logger.error(f"设置图模式失败: {str(e)}")
            raise
    
    async def _verify_initialization(self):
        """验证初始化结果"""
        try:
            logger.info("验证初始化结果...")
            
            # 检查约束
            constraints_query = "SHOW CONSTRAINTS"
            constraints_result = await self.graph_service.execute_cypher(constraints_query)
            actual_constraints = len(constraints_result) if constraints_result else 0
            
            logger.info(f"实际创建的约束数量: {actual_constraints}")
            
            # 检查索引
            indexes_query = "SHOW INDEXES"
            indexes_result = await self.graph_service.execute_cypher(indexes_query)
            actual_indexes = len(indexes_result) if indexes_result else 0
            
            logger.info(f"实际创建的索引数量: {actual_indexes}")
            
            # 检查节点和关系
            stats_query = """
            CALL {
                MATCH (n) RETURN count(n) as node_count
            }
            CALL {
                MATCH ()-[r]->() RETURN count(r) as rel_count
            }
            CALL {
                MATCH (n) RETURN labels(n) as labels
            }
            WITH node_count, rel_count, collect(DISTINCT labels) as all_labels
            UNWIND all_labels as label_list
            UNWIND label_list as label
            WITH node_count, rel_count, collect(DISTINCT label) as unique_labels
            RETURN node_count, rel_count, unique_labels
            """
            
            stats_result = await self.graph_service.execute_cypher(stats_query)
            
            if stats_result:
                record = stats_result[0]
                final_node_count = record["node_count"]
                final_rel_count = record["rel_count"]
                unique_labels = record["unique_labels"]
                
                logger.info(f"最终统计 - 节点数: {final_node_count}, 关系数: {final_rel_count}")
                logger.info(f"节点标签: {unique_labels}")
            
            # 测试基本查询性能
            start_time = time.time()
            test_query = "MATCH (n) RETURN count(n) as total"
            await self.graph_service.execute_cypher(test_query)
            query_time = time.time() - start_time
            
            logger.info(f"基本查询性能测试: {query_time:.3f} 秒")
            
            if query_time > 1.0:
                logger.warning("查询性能较慢，建议检查索引配置")
            
            logger.info("初始化验证完成")
            
        except Exception as e:
            logger.error(f"验证初始化结果失败: {str(e)}")
            raise
    
    async def reset_database(self, confirm: bool = False):
        """
        重置数据库（危险操作）
        
        Args:
            confirm: 确认执行重置
        """
        if not confirm:
            logger.warning("数据库重置需要确认参数 confirm=True")
            return
        
        try:
            logger.warning("开始重置 Neo4j 数据库...")
            
            # 删除所有节点和关系
            delete_query = "MATCH (n) DETACH DELETE n"
            await self.graph_service.execute_cypher(delete_query)
            
            # 删除所有约束
            constraints_query = "SHOW CONSTRAINTS"
            constraints_result = await self.graph_service.execute_cypher(constraints_query)
            
            if constraints_result:
                for constraint in constraints_result:
                    constraint_name = constraint.get("name")
                    if constraint_name:
                        drop_query = f"DROP CONSTRAINT {constraint_name} IF EXISTS"
                        await self.graph_service.execute_cypher(drop_query)
            
            # 删除所有索引
            indexes_query = "SHOW INDEXES"
            indexes_result = await self.graph_service.execute_cypher(indexes_query)
            
            if indexes_result:
                for index in indexes_result:
                    index_name = index.get("name")
                    if index_name and not index_name.startswith("system"):
                        drop_query = f"DROP INDEX {index_name} IF EXISTS"
                        await self.graph_service.execute_cypher(drop_query)
            
            logger.warning("数据库重置完成")
            
        except Exception as e:
            logger.error(f"重置数据库失败: {str(e)}")
            raise


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Neo4j 图数据库初始化脚本")
    parser.add_argument(
        "--reset", 
        action="store_true", 
        help="重置数据库（删除所有数据）"
    )
    parser.add_argument(
        "--confirm-reset", 
        action="store_true", 
        help="确认重置数据库"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="详细日志输出"
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        initializer = Neo4jInitializer()
        
        if args.reset:
            if not args.confirm_reset:
                print("错误: 重置数据库需要 --confirm-reset 参数")
                sys.exit(1)
            
            print("警告: 即将删除所有数据库内容！")
            response = input("请输入 'YES' 确认重置: ")
            
            if response != "YES":
                print("重置操作已取消")
                sys.exit(0)
            
            await initializer.reset_database(confirm=True)
            print("数据库重置完成")
        
        # 执行初始化
        stats = await initializer.initialize()
        
        # 输出统计信息
        print("\n" + "="*50)
        print("Neo4j 数据库初始化完成")
        print("="*50)
        print(f"耗时: {stats['duration']:.2f} 秒")
        print(f"创建约束: {stats['constraints_created']} 个")
        print(f"创建索引: {stats['indexes_created']} 个")
        print(f"创建节点: {stats['nodes_created']} 个")
        print(f"创建关系: {stats['relationships_created']} 个")
        
        if stats['errors']:
            print(f"错误数量: {len(stats['errors'])}")
            for error in stats['errors']:
                print(f"  - {error}")
        
        print("="*50)
        
    except KeyboardInterrupt:
        print("\n初始化被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n初始化失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())