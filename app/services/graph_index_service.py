#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 图数据库索引和约束管理服务
==================================

本模块实现了图数据库的索引和约束管理功能。

服务功能：
- 索引管理（创建、删除、监控）
- 约束管理（唯一性、存在性约束）
- 性能分析和优化建议
- 索引使用统计
- 自动索引优化
- 查询性能监控

支持的索引类型：
- 单属性索引
- 复合索引
- 全文索引
- 向量索引（如果支持）

支持的约束类型：
- 唯一性约束
- 存在性约束
- 节点键约束
- 关系存在性约束

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import json
import re

from app.services.graph_service import GraphService
from app.utils.exceptions import (
    GraphIndexError,
    GraphConstraintError,
    GraphPerformanceError
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class IndexType(Enum):
    """索引类型枚举"""
    BTREE = "BTREE"
    RANGE = "RANGE"
    TEXT = "TEXT"
    POINT = "POINT"
    FULLTEXT = "FULLTEXT"
    VECTOR = "VECTOR"


class ConstraintType(Enum):
    """约束类型枚举"""
    UNIQUE = "UNIQUE"
    EXISTS = "EXISTS"
    NODE_KEY = "NODE_KEY"
    RELATIONSHIP_EXISTS = "RELATIONSHIP_EXISTS"


@dataclass
class IndexInfo:
    """索引信息"""
    name: str
    labels: List[str]
    properties: List[str]
    type: str
    state: str
    population_percent: float
    unique_values: Optional[int]
    size: Optional[int]
    created_at: Optional[str]
    last_updated: Optional[str]
    usage_count: int = 0
    last_used: Optional[str] = None


@dataclass
class ConstraintInfo:
    """约束信息"""
    name: str
    type: str
    labels: List[str]
    properties: List[str]
    description: str
    created_at: Optional[str]
    violations: int = 0
    last_checked: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """性能指标"""
    query_time: float
    index_hits: int
    index_misses: int
    rows_examined: int
    rows_returned: int
    memory_usage: Optional[int]
    cpu_usage: Optional[float]
    timestamp: str


@dataclass
class OptimizationSuggestion:
    """优化建议"""
    type: str  # "create_index", "drop_index", "create_constraint", etc.
    priority: str  # "high", "medium", "low"
    description: str
    impact: str
    query: str
    estimated_benefit: Optional[str] = None


class IndexManager:
    """
    索引管理器
    
    负责图数据库索引的创建、删除和监控。
    """
    
    def __init__(self, graph_service: GraphService):
        """
        初始化索引管理器
        
        Args:
            graph_service: 图服务实例
        """
        self.graph_service = graph_service
        self.index_cache = {}
        self.usage_stats = defaultdict(int)
        self.last_cache_update = None
    
    async def create_index(
        self,
        name: str,
        labels: List[str],
        properties: List[str],
        index_type: IndexType = IndexType.BTREE,
        if_not_exists: bool = True
    ) -> bool:
        """
        创建索引
        
        Args:
            name: 索引名称
            labels: 节点标签列表
            properties: 属性列表
            index_type: 索引类型
            if_not_exists: 如果不存在才创建
            
        Returns:
            bool: 创建是否成功
        """
        try:
            # 构建标签字符串
            if len(labels) == 1:
                label_str = labels[0]
            else:
                label_str = ":".join(labels)
            
            # 构建属性字符串
            if len(properties) == 1:
                prop_str = f"n.{properties[0]}"
            else:
                prop_str = ", ".join([f"n.{prop}" for prop in properties])
            
            # 构建创建索引的查询
            if_not_exists_str = "IF NOT EXISTS" if if_not_exists else ""
            
            if index_type == IndexType.FULLTEXT:
                # 全文索引
                query = f"""
                CREATE FULLTEXT INDEX {name} {if_not_exists_str}
                FOR (n:{label_str})
                ON EACH [{', '.join([f'n.{prop}' for prop in properties])}]
                """
            elif index_type == IndexType.VECTOR:
                # 向量索引（Neo4j 5.x+）
                if len(properties) != 1:
                    raise ValueError("向量索引只支持单个属性")
                
                query = f"""
                CREATE VECTOR INDEX {name} {if_not_exists_str}
                FOR (n:{label_str})
                ON (n.{properties[0]})
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
                """
            else:
                # 标准索引
                query = f"""
                CREATE INDEX {name} {if_not_exists_str}
                FOR (n:{label_str})
                ON ({prop_str})
                """
            
            await self.graph_service.execute_cypher(query)
            
            # 清除缓存
            self.index_cache.clear()
            self.last_cache_update = None
            
            logger.info(f"成功创建索引: {name}")
            return True
            
        except Exception as e:
            logger.error(f"创建索引失败 ({name}): {str(e)}")
            raise GraphIndexError(f"创建索引失败: {str(e)}")
    
    async def drop_index(self, name: str, if_exists: bool = True) -> bool:
        """
        删除索引
        
        Args:
            name: 索引名称
            if_exists: 如果存在才删除
            
        Returns:
            bool: 删除是否成功
        """
        try:
            if_exists_str = "IF EXISTS" if if_exists else ""
            query = f"DROP INDEX {name} {if_exists_str}"
            
            await self.graph_service.execute_cypher(query)
            
            # 清除缓存
            if name in self.index_cache:
                del self.index_cache[name]
            
            logger.info(f"成功删除索引: {name}")
            return True
            
        except Exception as e:
            logger.error(f"删除索引失败 ({name}): {str(e)}")
            raise GraphIndexError(f"删除索引失败: {str(e)}")
    
    async def list_indexes(self, refresh_cache: bool = False) -> List[IndexInfo]:
        """
        列出所有索引
        
        Args:
            refresh_cache: 是否刷新缓存
            
        Returns:
            List[IndexInfo]: 索引信息列表
        """
        try:
            # 检查缓存
            if (not refresh_cache and 
                self.last_cache_update and 
                (datetime.utcnow() - self.last_cache_update).seconds < 300):  # 5分钟缓存
                return list(self.index_cache.values())
            
            query = "SHOW INDEXES"
            result = await self.graph_service.execute_cypher(query)
            
            indexes = []
            for record in result:
                index_info = IndexInfo(
                    name=record.get("name", ""),
                    labels=record.get("labelsOrTypes", []),
                    properties=record.get("properties", []),
                    type=record.get("type", ""),
                    state=record.get("state", ""),
                    population_percent=record.get("populationPercent", 0.0),
                    unique_values=record.get("uniqueValues"),
                    size=record.get("size"),
                    created_at=record.get("createdAt"),
                    last_updated=record.get("lastUpdated"),
                    usage_count=self.usage_stats.get(record.get("name", ""), 0)
                )
                indexes.append(index_info)
                self.index_cache[index_info.name] = index_info
            
            self.last_cache_update = datetime.utcnow()
            return indexes
            
        except Exception as e:
            logger.error(f"列出索引失败: {str(e)}")
            raise GraphIndexError(f"列出索引失败: {str(e)}")
    
    async def get_index_usage_stats(self, index_name: str) -> Dict[str, Any]:
        """
        获取索引使用统计
        
        Args:
            index_name: 索引名称
            
        Returns:
            Dict[str, Any]: 使用统计信息
        """
        try:
            # 这里需要根据 Neo4j 版本和可用的监控功能来实现
            # 由于不是所有版本都支持详细的索引统计，我们提供一个基础实现
            
            query = f"""
            SHOW INDEXES
            WHERE name = '{index_name}'
            """
            
            result = await self.graph_service.execute_cypher(query)
            
            if not result:
                return {"error": f"索引 {index_name} 不存在"}
            
            index_data = result[0]
            
            return {
                "name": index_name,
                "state": index_data.get("state", "unknown"),
                "population_percent": index_data.get("populationPercent", 0.0),
                "unique_values": index_data.get("uniqueValues"),
                "size": index_data.get("size"),
                "usage_count": self.usage_stats.get(index_name, 0),
                "last_used": None  # 需要额外的监控来获取
            }
            
        except Exception as e:
            logger.error(f"获取索引使用统计失败 ({index_name}): {str(e)}")
            return {"error": str(e)}
    
    async def rebuild_index(self, index_name: str) -> bool:
        """
        重建索引
        
        Args:
            index_name: 索引名称
            
        Returns:
            bool: 重建是否成功
        """
        try:
            # 获取索引信息
            indexes = await self.list_indexes(refresh_cache=True)
            target_index = None
            
            for index in indexes:
                if index.name == index_name:
                    target_index = index
                    break
            
            if not target_index:
                raise ValueError(f"索引 {index_name} 不存在")
            
            # 删除索引
            await self.drop_index(index_name)
            
            # 等待一段时间确保删除完成
            await asyncio.sleep(1)
            
            # 重新创建索引
            await self.create_index(
                name=target_index.name,
                labels=target_index.labels,
                properties=target_index.properties,
                index_type=IndexType(target_index.type) if target_index.type in [t.value for t in IndexType] else IndexType.BTREE
            )
            
            logger.info(f"成功重建索引: {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"重建索引失败 ({index_name}): {str(e)}")
            raise GraphIndexError(f"重建索引失败: {str(e)}")


class ConstraintManager:
    """
    约束管理器
    
    负责图数据库约束的创建、删除和监控。
    """
    
    def __init__(self, graph_service: GraphService):
        """
        初始化约束管理器
        
        Args:
            graph_service: 图服务实例
        """
        self.graph_service = graph_service
        self.constraint_cache = {}
        self.last_cache_update = None
    
    async def create_constraint(
        self,
        name: str,
        constraint_type: ConstraintType,
        labels: List[str],
        properties: List[str],
        if_not_exists: bool = True
    ) -> bool:
        """
        创建约束
        
        Args:
            name: 约束名称
            constraint_type: 约束类型
            labels: 节点标签列表
            properties: 属性列表
            if_not_exists: 如果不存在才创建
            
        Returns:
            bool: 创建是否成功
        """
        try:
            # 构建标签字符串
            if len(labels) == 1:
                label_str = labels[0]
            else:
                label_str = ":".join(labels)
            
            # 构建属性字符串
            if len(properties) == 1:
                prop_str = f"n.{properties[0]}"
            else:
                prop_str = ", ".join([f"n.{prop}" for prop in properties])
            
            if_not_exists_str = "IF NOT EXISTS" if if_not_exists else ""
            
            # 构建创建约束的查询
            if constraint_type == ConstraintType.UNIQUE:
                if len(properties) == 1:
                    query = f"""
                    CREATE CONSTRAINT {name} {if_not_exists_str}
                    FOR (n:{label_str})
                    REQUIRE n.{properties[0]} IS UNIQUE
                    """
                else:
                    query = f"""
                    CREATE CONSTRAINT {name} {if_not_exists_str}
                    FOR (n:{label_str})
                    REQUIRE ({prop_str}) IS UNIQUE
                    """
            elif constraint_type == ConstraintType.EXISTS:
                if len(properties) == 1:
                    query = f"""
                    CREATE CONSTRAINT {name} {if_not_exists_str}
                    FOR (n:{label_str})
                    REQUIRE n.{properties[0]} IS NOT NULL
                    """
                else:
                    # 多属性存在性约束
                    conditions = " AND ".join([f"n.{prop} IS NOT NULL" for prop in properties])
                    query = f"""
                    CREATE CONSTRAINT {name} {if_not_exists_str}
                    FOR (n:{label_str})
                    REQUIRE {conditions}
                    """
            elif constraint_type == ConstraintType.NODE_KEY:
                query = f"""
                CREATE CONSTRAINT {name} {if_not_exists_str}
                FOR (n:{label_str})
                REQUIRE ({prop_str}) IS NODE KEY
                """
            else:
                raise ValueError(f"不支持的约束类型: {constraint_type}")
            
            await self.graph_service.execute_cypher(query)
            
            # 清除缓存
            self.constraint_cache.clear()
            self.last_cache_update = None
            
            logger.info(f"成功创建约束: {name}")
            return True
            
        except Exception as e:
            logger.error(f"创建约束失败 ({name}): {str(e)}")
            raise GraphConstraintError(f"创建约束失败: {str(e)}")
    
    async def drop_constraint(self, name: str, if_exists: bool = True) -> bool:
        """
        删除约束
        
        Args:
            name: 约束名称
            if_exists: 如果存在才删除
            
        Returns:
            bool: 删除是否成功
        """
        try:
            if_exists_str = "IF EXISTS" if if_exists else ""
            query = f"DROP CONSTRAINT {name} {if_exists_str}"
            
            await self.graph_service.execute_cypher(query)
            
            # 清除缓存
            if name in self.constraint_cache:
                del self.constraint_cache[name]
            
            logger.info(f"成功删除约束: {name}")
            return True
            
        except Exception as e:
            logger.error(f"删除约束失败 ({name}): {str(e)}")
            raise GraphConstraintError(f"删除约束失败: {str(e)}")
    
    async def list_constraints(self, refresh_cache: bool = False) -> List[ConstraintInfo]:
        """
        列出所有约束
        
        Args:
            refresh_cache: 是否刷新缓存
            
        Returns:
            List[ConstraintInfo]: 约束信息列表
        """
        try:
            # 检查缓存
            if (not refresh_cache and 
                self.last_cache_update and 
                (datetime.utcnow() - self.last_cache_update).seconds < 300):  # 5分钟缓存
                return list(self.constraint_cache.values())
            
            query = "SHOW CONSTRAINTS"
            result = await self.graph_service.execute_cypher(query)
            
            constraints = []
            for record in result:
                constraint_info = ConstraintInfo(
                    name=record.get("name", ""),
                    type=record.get("type", ""),
                    labels=record.get("labelsOrTypes", []),
                    properties=record.get("properties", []),
                    description=record.get("description", ""),
                    created_at=record.get("createdAt")
                )
                constraints.append(constraint_info)
                self.constraint_cache[constraint_info.name] = constraint_info
            
            self.last_cache_update = datetime.utcnow()
            return constraints
            
        except Exception as e:
            logger.error(f"列出约束失败: {str(e)}")
            raise GraphConstraintError(f"列出约束失败: {str(e)}")
    
    async def validate_constraints(self) -> Dict[str, Any]:
        """
        验证约束完整性
        
        Returns:
            Dict[str, Any]: 验证结果
        """
        try:
            constraints = await self.list_constraints(refresh_cache=True)
            validation_results = {
                "total_constraints": len(constraints),
                "valid_constraints": 0,
                "invalid_constraints": 0,
                "violations": [],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            for constraint in constraints:
                try:
                    # 检查约束是否有效（这里提供基础检查）
                    if constraint.type == "UNIQUENESS":
                        # 检查唯一性约束
                        label = constraint.labels[0] if constraint.labels else ""
                        prop = constraint.properties[0] if constraint.properties else ""
                        
                        if label and prop:
                            check_query = f"""
                            MATCH (n:{label})
                            WHERE n.{prop} IS NOT NULL
                            WITH n.{prop} as value, count(*) as cnt
                            WHERE cnt > 1
                            RETURN value, cnt
                            LIMIT 10
                            """
                            
                            violations = await self.graph_service.execute_cypher(check_query)
                            
                            if violations:
                                validation_results["invalid_constraints"] += 1
                                validation_results["violations"].append({
                                    "constraint": constraint.name,
                                    "type": "uniqueness_violation",
                                    "violations": len(violations),
                                    "examples": violations[:5]
                                })
                            else:
                                validation_results["valid_constraints"] += 1
                    else:
                        # 其他类型的约束暂时标记为有效
                        validation_results["valid_constraints"] += 1
                        
                except Exception as e:
                    validation_results["invalid_constraints"] += 1
                    validation_results["violations"].append({
                        "constraint": constraint.name,
                        "type": "validation_error",
                        "error": str(e)
                    })
            
            return validation_results
            
        except Exception as e:
            logger.error(f"验证约束失败: {str(e)}")
            raise GraphConstraintError(f"验证约束失败: {str(e)}")


class PerformanceAnalyzer:
    """
    性能分析器
    
    分析查询性能并提供优化建议。
    """
    
    def __init__(self, graph_service: GraphService):
        """
        初始化性能分析器
        
        Args:
            graph_service: 图服务实例
        """
        self.graph_service = graph_service
        self.query_history = []
        self.performance_metrics = []
    
    async def analyze_query_performance(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        explain: bool = True
    ) -> Dict[str, Any]:
        """
        分析查询性能
        
        Args:
            query: Cypher 查询
            params: 查询参数
            explain: 是否使用 EXPLAIN
            
        Returns:
            Dict[str, Any]: 性能分析结果
        """
        try:
            start_time = time.time()
            
            # 执行 EXPLAIN 或 PROFILE
            if explain:
                explain_query = f"EXPLAIN {query}"
            else:
                explain_query = f"PROFILE {query}"
            
            result = await self.graph_service.execute_cypher(explain_query, params or {})
            
            execution_time = time.time() - start_time
            
            # 解析执行计划
            plan_analysis = self._analyze_execution_plan(result)
            
            performance_result = {
                "query": query,
                "execution_time": execution_time,
                "plan_analysis": plan_analysis,
                "timestamp": datetime.utcnow().isoformat(),
                "suggestions": self._generate_performance_suggestions(query, plan_analysis)
            }
            
            # 记录到历史
            self.query_history.append(performance_result)
            
            # 限制历史记录数量
            if len(self.query_history) > 1000:
                self.query_history = self.query_history[-1000:]
            
            return performance_result
            
        except Exception as e:
            logger.error(f"分析查询性能失败: {str(e)}")
            raise GraphPerformanceError(f"分析查询性能失败: {str(e)}")
    
    def _analyze_execution_plan(self, plan_result: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析执行计划
        
        Args:
            plan_result: 执行计划结果
            
        Returns:
            Dict[str, Any]: 执行计划分析
        """
        try:
            analysis = {
                "total_db_hits": 0,
                "estimated_rows": 0,
                "operators": [],
                "index_usage": [],
                "warnings": []
            }
            
            # 这里需要根据 Neo4j 返回的执行计划格式来解析
            # 由于格式可能因版本而异，我们提供一个基础的解析逻辑
            
            for record in plan_result:
                # 提取操作符信息
                if "operatorType" in record:
                    operator = {
                        "type": record.get("operatorType"),
                        "estimated_rows": record.get("estimatedRows", 0),
                        "db_hits": record.get("dbHits", 0)
                    }
                    analysis["operators"].append(operator)
                    analysis["total_db_hits"] += operator["db_hits"]
                    analysis["estimated_rows"] += operator["estimated_rows"]
                
                # 检查索引使用
                if "index" in str(record).lower():
                    analysis["index_usage"].append(record)
                
                # 检查警告
                if "warning" in str(record).lower() or "scan" in str(record).lower():
                    analysis["warnings"].append(str(record))
            
            return analysis
            
        except Exception as e:
            logger.error(f"分析执行计划失败: {str(e)}")
            return {"error": str(e)}
    
    def _generate_performance_suggestions(
        self,
        query: str,
        plan_analysis: Dict[str, Any]
    ) -> List[OptimizationSuggestion]:
        """
        生成性能优化建议
        
        Args:
            query: 查询语句
            plan_analysis: 执行计划分析
            
        Returns:
            List[OptimizationSuggestion]: 优化建议列表
        """
        suggestions = []
        
        try:
            # 检查是否有全表扫描
            if any("scan" in op.get("type", "").lower() for op in plan_analysis.get("operators", [])):
                suggestions.append(OptimizationSuggestion(
                    type="create_index",
                    priority="high",
                    description="检测到全表扫描，建议创建索引",
                    impact="可能显著提升查询性能",
                    query="",  # 需要根据具体情况生成
                    estimated_benefit="50-90% 性能提升"
                ))
            
            # 检查数据库命中次数
            total_db_hits = plan_analysis.get("total_db_hits", 0)
            if total_db_hits > 10000:
                suggestions.append(OptimizationSuggestion(
                    type="optimize_query",
                    priority="medium",
                    description=f"数据库命中次数过高 ({total_db_hits})，建议优化查询",
                    impact="减少数据库访问次数",
                    query="",
                    estimated_benefit="20-50% 性能提升"
                ))
            
            # 检查是否使用了索引
            if not plan_analysis.get("index_usage"):
                # 分析查询中的 WHERE 条件
                where_match = re.search(r'WHERE\s+(.+?)(?:RETURN|ORDER|LIMIT|$)', query, re.IGNORECASE | re.DOTALL)
                if where_match:
                    suggestions.append(OptimizationSuggestion(
                        type="create_index",
                        priority="medium",
                        description="查询包含 WHERE 条件但未使用索引",
                        impact="可能提升过滤性能",
                        query="",
                        estimated_benefit="30-70% 性能提升"
                    ))
            
            # 检查 ORDER BY 子句
            if "ORDER BY" in query.upper() and not any("sort" in op.get("type", "").lower() for op in plan_analysis.get("operators", [])):
                suggestions.append(OptimizationSuggestion(
                    type="create_index",
                    priority="low",
                    description="查询包含 ORDER BY 但可能未使用排序索引",
                    impact="可能提升排序性能",
                    query="",
                    estimated_benefit="10-30% 性能提升"
                ))
            
        except Exception as e:
            logger.error(f"生成性能建议失败: {str(e)}")
        
        return suggestions
    
    async def get_slow_queries(
        self,
        threshold_seconds: float = 1.0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        获取慢查询列表
        
        Args:
            threshold_seconds: 慢查询阈值（秒）
            limit: 返回数量限制
            
        Returns:
            List[Dict[str, Any]]: 慢查询列表
        """
        slow_queries = [
            query for query in self.query_history
            if query.get("execution_time", 0) > threshold_seconds
        ]
        
        # 按执行时间排序
        slow_queries.sort(key=lambda x: x.get("execution_time", 0), reverse=True)
        
        return slow_queries[:limit]


class GraphIndexService:
    """
    图数据库索引和约束管理服务
    
    提供完整的索引和约束管理功能。
    """
    
    def __init__(self, graph_service: GraphService):
        """
        初始化图索引服务
        
        Args:
            graph_service: 图服务实例
        """
        self.graph_service = graph_service
        
        # 初始化各个管理器
        self.index_manager = IndexManager(graph_service)
        self.constraint_manager = ConstraintManager(graph_service)
        self.performance_analyzer = PerformanceAnalyzer(graph_service)
        
        # 服务统计
        self.stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "index_operations": 0,
            "constraint_operations": 0,
            "performance_analyses": 0,
            "last_operation_time": None
        }
        
        logger.info("图数据库索引和约束管理服务初始化完成")
    
    # 索引管理方法
    async def create_index(
        self,
        name: str,
        labels: List[str],
        properties: List[str],
        index_type: str = "BTREE",
        if_not_exists: bool = True
    ) -> bool:
        """创建索引"""
        try:
            self.stats["total_operations"] += 1
            self.stats["index_operations"] += 1
            
            index_type_enum = IndexType(index_type.upper())
            result = await self.index_manager.create_index(
                name, labels, properties, index_type_enum, if_not_exists
            )
            
            self.stats["successful_operations"] += 1
            self.stats["last_operation_time"] = datetime.utcnow().isoformat()
            
            return result
        except Exception as e:
            self.stats["failed_operations"] += 1
            raise
    
    async def drop_index(self, name: str, if_exists: bool = True) -> bool:
        """删除索引"""
        try:
            self.stats["total_operations"] += 1
            self.stats["index_operations"] += 1
            
            result = await self.index_manager.drop_index(name, if_exists)
            
            self.stats["successful_operations"] += 1
            self.stats["last_operation_time"] = datetime.utcnow().isoformat()
            
            return result
        except Exception as e:
            self.stats["failed_operations"] += 1
            raise
    
    async def list_indexes(self, refresh_cache: bool = False) -> List[Dict[str, Any]]:
        """列出所有索引"""
        try:
            self.stats["total_operations"] += 1
            
            indexes = await self.index_manager.list_indexes(refresh_cache)
            
            self.stats["successful_operations"] += 1
            self.stats["last_operation_time"] = datetime.utcnow().isoformat()
            
            return [asdict(index) for index in indexes]
        except Exception as e:
            self.stats["failed_operations"] += 1
            raise
    
    # 约束管理方法
    async def create_constraint(
        self,
        name: str,
        constraint_type: str,
        labels: List[str],
        properties: List[str],
        if_not_exists: bool = True
    ) -> bool:
        """创建约束"""
        try:
            self.stats["total_operations"] += 1
            self.stats["constraint_operations"] += 1
            
            constraint_type_enum = ConstraintType(constraint_type.upper())
            result = await self.constraint_manager.create_constraint(
                name, constraint_type_enum, labels, properties, if_not_exists
            )
            
            self.stats["successful_operations"] += 1
            self.stats["last_operation_time"] = datetime.utcnow().isoformat()
            
            return result
        except Exception as e:
            self.stats["failed_operations"] += 1
            raise
    
    async def drop_constraint(self, name: str, if_exists: bool = True) -> bool:
        """删除约束"""
        try:
            self.stats["total_operations"] += 1
            self.stats["constraint_operations"] += 1
            
            result = await self.constraint_manager.drop_constraint(name, if_exists)
            
            self.stats["successful_operations"] += 1
            self.stats["last_operation_time"] = datetime.utcnow().isoformat()
            
            return result
        except Exception as e:
            self.stats["failed_operations"] += 1
            raise
    
    async def list_constraints(self, refresh_cache: bool = False) -> List[Dict[str, Any]]:
        """列出所有约束"""
        try:
            self.stats["total_operations"] += 1
            
            constraints = await self.constraint_manager.list_constraints(refresh_cache)
            
            self.stats["successful_operations"] += 1
            self.stats["last_operation_time"] = datetime.utcnow().isoformat()
            
            return [asdict(constraint) for constraint in constraints]
        except Exception as e:
            self.stats["failed_operations"] += 1
            raise
    
    # 性能分析方法
    async def analyze_query_performance(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        explain: bool = True
    ) -> Dict[str, Any]:
        """分析查询性能"""
        try:
            self.stats["total_operations"] += 1
            self.stats["performance_analyses"] += 1
            
            result = await self.performance_analyzer.analyze_query_performance(
                query, params, explain
            )
            
            self.stats["successful_operations"] += 1
            self.stats["last_operation_time"] = datetime.utcnow().isoformat()
            
            return result
        except Exception as e:
            self.stats["failed_operations"] += 1
            raise
    
    async def get_optimization_suggestions(
        self,
        query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取优化建议"""
        try:
            suggestions = []
            
            if query:
                # 分析特定查询
                analysis = await self.analyze_query_performance(query)
                suggestions.extend(analysis.get("suggestions", []))
            else:
                # 分析整体性能
                # 检查缺失的索引
                missing_indexes = await self._suggest_missing_indexes()
                suggestions.extend(missing_indexes)
                
                # 检查未使用的索引
                unused_indexes = await self._suggest_unused_indexes()
                suggestions.extend(unused_indexes)
                
                # 检查约束问题
                constraint_issues = await self._suggest_constraint_fixes()
                suggestions.extend(constraint_issues)
            
            return [asdict(suggestion) for suggestion in suggestions]
            
        except Exception as e:
            logger.error(f"获取优化建议失败: {str(e)}")
            return []
    
    async def _suggest_missing_indexes(self) -> List[OptimizationSuggestion]:
        """建议缺失的索引"""
        suggestions = []
        
        try:
            # 分析常用的查询模式
            # 这里提供一个基础的实现，实际应用中可以基于查询日志来分析
            
            # 检查常用属性是否有索引
            common_properties = ["id", "name", "created_at", "updated_at", "type"]
            
            for prop in common_properties:
                # 检查是否已有索引
                existing_indexes = await self.index_manager.list_indexes()
                has_index = any(prop in index.properties for index in existing_indexes)
                
                if not has_index:
                    # 检查该属性的使用频率
                    usage_query = f"""
                    MATCH (n)
                    WHERE n.{prop} IS NOT NULL
                    RETURN count(n) as usage_count
                    """
                    
                    result = await self.graph_service.execute_cypher(usage_query)
                    
                    if result and result[0]["usage_count"] > 100:  # 使用频率阈值
                        suggestions.append(OptimizationSuggestion(
                            type="create_index",
                            priority="medium",
                            description=f"属性 '{prop}' 使用频繁但缺少索引",
                            impact="可能提升查询性能",
                            query=f"CREATE INDEX idx_{prop} FOR (n) ON (n.{prop})",
                            estimated_benefit="20-50% 性能提升"
                        ))
            
        except Exception as e:
            logger.error(f"分析缺失索引失败: {str(e)}")
        
        return suggestions
    
    async def _suggest_unused_indexes(self) -> List[OptimizationSuggestion]:
        """建议删除未使用的索引"""
        suggestions = []
        
        try:
            indexes = await self.index_manager.list_indexes()
            
            for index in indexes:
                # 检查索引使用情况（这里提供简化的检查）
                if index.usage_count == 0 and index.name.startswith("idx_"):
                    suggestions.append(OptimizationSuggestion(
                        type="drop_index",
                        priority="low",
                        description=f"索引 '{index.name}' 似乎未被使用",
                        impact="减少存储空间和维护开销",
                        query=f"DROP INDEX {index.name}",
                        estimated_benefit="减少存储开销"
                    ))
            
        except Exception as e:
            logger.error(f"分析未使用索引失败: {str(e)}")
        
        return suggestions
    
    async def _suggest_constraint_fixes(self) -> List[OptimizationSuggestion]:
        """建议约束修复"""
        suggestions = []
        
        try:
            # 验证约束
            validation_result = await self.constraint_manager.validate_constraints()
            
            for violation in validation_result.get("violations", []):
                if violation["type"] == "uniqueness_violation":
                    suggestions.append(OptimizationSuggestion(
                        type="fix_constraint",
                        priority="high",
                        description=f"约束 '{violation['constraint']}' 存在违规数据",
                        impact="数据完整性问题",
                        query="",  # 需要根据具体情况生成清理查询
                        estimated_benefit="提升数据质量"
                    ))
            
        except Exception as e:
            logger.error(f"分析约束问题失败: {str(e)}")
        
        return suggestions
    
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
            # 基础图服务健康检查
            graph_health = await self.graph_service.health_check()
            
            if graph_health["status"] != "healthy":
                return {
                    "status": "unhealthy",
                    "graph_service": graph_health,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # 测试索引和约束功能
            try:
                # 列出索引和约束
                indexes = await self.list_indexes()
                constraints = await self.list_constraints()
                
                return {
                    "status": "healthy",
                    "graph_service": graph_health,
                    "indexes_count": len(indexes),
                    "constraints_count": len(constraints),
                    "stats": self.get_stats(),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                return {
                    "status": "degraded",
                    "graph_service": graph_health,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }