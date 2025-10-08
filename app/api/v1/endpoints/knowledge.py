#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 知识抽取 API 端点
========================

本模块提供知识抽取相关的 API 端点，包括：
1. 实体抽取 - 从文本中抽取实体
2. 关系抽取 - 从文本中抽取关系
3. 引用抽取 - 从文本中抽取引用和证据
4. 批量知识抽取 - 批量处理多个文档
5. 抽取结果管理 - 查询、更新、删除抽取结果

所有端点都支持异步处理和错误处理，并提供详细的日志记录。

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import uuid
import asyncio

from app.core.logging import get_logger
from app.services.entity_service import EntityService
from app.services.relation_service import RelationService
from app.utils.exceptions import (
    EntityExtractionError,
    RelationExtractionError,
    ValidationError
)

logger = get_logger(__name__)

# 创建路由器
router = APIRouter()


# Pydantic 模型定义

class TextInput(BaseModel):
    """文本输入模型"""
    text: str = Field(..., min_length=1, max_length=50000, description="待处理的文本")
    language: str = Field(default="zh", description="文本语言")
    document_id: Optional[str] = Field(None, description="关联的文档ID")
    chunk_id: Optional[str] = Field(None, description="关联的文本块ID")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('文本内容不能为空')
        return v.strip()


class EntityExtractionRequest(BaseModel):
    """实体抽取请求模型"""
    text: str = Field(..., min_length=1, max_length=50000, description="待抽取的文本")
    language: str = Field(default="zh", description="文本语言")
    entity_types: Optional[List[str]] = Field(None, description="指定抽取的实体类型")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="置信度阈值")
    max_entities: int = Field(default=100, ge=1, le=1000, description="最大抽取实体数量")
    include_context: bool = Field(default=True, description="是否包含上下文信息")
    document_id: Optional[str] = Field(None, description="关联的文档ID")
    chunk_id: Optional[str] = Field(None, description="关联的文本块ID")


class RelationExtractionRequest(BaseModel):
    """关系抽取请求模型"""
    text: str = Field(..., min_length=1, max_length=50000, description="待抽取的文本")
    language: str = Field(default="zh", description="文本语言")
    relation_types: Optional[List[str]] = Field(None, description="指定抽取的关系类型")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="置信度阈值")
    max_relations: int = Field(default=100, ge=1, le=1000, description="最大抽取关系数量")
    include_entities: bool = Field(default=True, description="是否同时抽取实体")
    document_id: Optional[str] = Field(None, description="关联的文档ID")
    chunk_id: Optional[str] = Field(None, description="关联的文本块ID")


class BatchExtractionRequest(BaseModel):
    """批量抽取请求模型"""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="待抽取的文本列表")
    language: str = Field(default="zh", description="文本语言")
    extract_entities: bool = Field(default=True, description="是否抽取实体")
    extract_relations: bool = Field(default=True, description="是否抽取关系")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="置信度阈值")
    document_ids: Optional[List[str]] = Field(None, description="关联的文档ID列表")
    chunk_ids: Optional[List[str]] = Field(None, description="关联的文本块ID列表")


class ExtractionResponse(BaseModel):
    """抽取响应模型"""
    success: bool = Field(..., description="是否成功")
    task_id: str = Field(..., description="任务ID")
    message: str = Field(..., description="响应消息")
    data: Optional[Dict[str, Any]] = Field(None, description="抽取结果数据")
    timestamp: str = Field(..., description="时间戳")


class EntityResponse(BaseModel):
    """实体响应模型"""
    entities: List[Dict[str, Any]] = Field(..., description="抽取的实体列表")
    total_count: int = Field(..., description="实体总数")
    processing_time: float = Field(..., description="处理时间（秒）")
    confidence_stats: Dict[str, Any] = Field(..., description="置信度统计")


class RelationResponse(BaseModel):
    """关系响应模型"""
    relations: List[Dict[str, Any]] = Field(..., description="抽取的关系列表")
    entities: Optional[List[Dict[str, Any]]] = Field(None, description="相关实体列表")
    total_count: int = Field(..., description="关系总数")
    processing_time: float = Field(..., description="处理时间（秒）")
    confidence_stats: Dict[str, Any] = Field(..., description="置信度统计")


# 依赖注入

async def get_entity_service() -> EntityService:
    """获取实体服务实例"""
    # 这里应该从依赖注入容器获取服务实例
    # 暂时返回一个新实例，实际应用中应该使用单例
    from app.services.entity_service import EntityService
    return EntityService()


async def get_relation_service() -> RelationService:
    """获取关系服务实例"""
    # 这里应该从依赖注入容器获取服务实例
    # 暂时返回一个新实例，实际应用中应该使用单例
    from app.services.relation_service import RelationService
    return RelationService()


# API 端点

@router.post("/extract-entities", response_model=ExtractionResponse, tags=["实体抽取"])
async def extract_entities(
    request: EntityExtractionRequest,
    background_tasks: BackgroundTasks,
    entity_service: EntityService = Depends(get_entity_service)
) -> ExtractionResponse:
    """
    从文本中抽取实体
    
    Args:
        request: 实体抽取请求
        background_tasks: 后台任务
        entity_service: 实体服务
        
    Returns:
        ExtractionResponse: 抽取响应
        
    Raises:
        HTTPException: 当抽取失败时
        
    Example:
        ```json
        {
            "text": "苹果公司是一家美国跨国科技公司，总部位于加利福尼亚州库比蒂诺。",
            "language": "zh",
            "entity_types": ["ORGANIZATION", "LOCATION"],
            "confidence_threshold": 0.7,
            "max_entities": 50,
            "include_context": true
        }
        ```
    """
    try:
        logger.info(f"开始实体抽取任务，文本长度: {len(request.text)}")
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 执行实体抽取
        start_time = datetime.utcnow()
        
        result = await entity_service.extract_entities(
            text=request.text,
            language=request.language,
            entity_types=request.entity_types,
            confidence_threshold=request.confidence_threshold,
            max_entities=request.max_entities,
            document_id=request.document_id,
            chunk_id=request.chunk_id
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # 构建响应数据
        entities_data = []
        confidence_scores = []
        
        for entity in result.entities:
            entity_dict = {
                "text": entity.text,
                "type": entity.type,
                "confidence": entity.confidence,
                "start_pos": entity.start_pos,
                "end_pos": entity.end_pos,
                "context": entity.context if request.include_context else None,
                "properties": entity.properties,
                "normalized_form": entity.normalized_form
            }
            entities_data.append(entity_dict)
            confidence_scores.append(entity.confidence)
        
        # 计算置信度统计
        confidence_stats = {
            "mean": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            "min": min(confidence_scores) if confidence_scores else 0,
            "max": max(confidence_scores) if confidence_scores else 0,
            "count_high": len([s for s in confidence_scores if s >= 0.8]),
            "count_medium": len([s for s in confidence_scores if 0.5 <= s < 0.8]),
            "count_low": len([s for s in confidence_scores if s < 0.5])
        }
        
        response_data = {
            "entities": entities_data,
            "total_count": len(entities_data),
            "processing_time": processing_time,
            "confidence_stats": confidence_stats,
            "extraction_stats": result.stats
        }
        
        logger.info(f"实体抽取完成，任务ID: {task_id}，抽取实体数: {len(entities_data)}")
        
        return ExtractionResponse(
            success=True,
            task_id=task_id,
            message=f"成功抽取 {len(entities_data)} 个实体",
            data=response_data,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except EntityExtractionError as e:
        logger.error(f"实体抽取失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"实体抽取失败: {str(e)}"
        )
    except Exception as e:
        logger.error(f"实体抽取异常: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"服务器内部错误: {str(e)}"
        )


@router.post("/extract-relations", response_model=ExtractionResponse, tags=["关系抽取"])
async def extract_relations(
    request: RelationExtractionRequest,
    background_tasks: BackgroundTasks,
    relation_service: RelationService = Depends(get_relation_service)
) -> ExtractionResponse:
    """
    从文本中抽取关系
    
    Args:
        request: 关系抽取请求
        background_tasks: 后台任务
        relation_service: 关系服务
        
    Returns:
        ExtractionResponse: 抽取响应
        
    Raises:
        HTTPException: 当抽取失败时
        
    Example:
        ```json
        {
            "text": "苹果公司的CEO是蒂姆·库克，他于2011年接任这一职位。",
            "language": "zh",
            "relation_types": ["CEO_OF", "SUCCESSOR_OF"],
            "confidence_threshold": 0.7,
            "max_relations": 50,
            "include_entities": true
        }
        ```
    """
    try:
        logger.info(f"开始关系抽取任务，文本长度: {len(request.text)}")
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 执行关系抽取
        start_time = datetime.utcnow()
        
        result = await relation_service.extract_relations(
            text=request.text,
            language=request.language,
            relation_types=request.relation_types,
            confidence_threshold=request.confidence_threshold,
            max_relations=request.max_relations,
            document_id=request.document_id,
            chunk_id=request.chunk_id
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # 构建响应数据
        relations_data = []
        confidence_scores = []
        
        for relation in result.relations:
            relation_dict = {
                "subject": relation.subject,
                "predicate": relation.predicate,
                "object": relation.object,
                "confidence": relation.confidence,
                "context": relation.context,
                "properties": relation.properties,
                "evidence": relation.evidence
            }
            relations_data.append(relation_dict)
            confidence_scores.append(relation.confidence)
        
        # 计算置信度统计
        confidence_stats = {
            "mean": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            "min": min(confidence_scores) if confidence_scores else 0,
            "max": max(confidence_scores) if confidence_scores else 0,
            "count_high": len([s for s in confidence_scores if s >= 0.8]),
            "count_medium": len([s for s in confidence_scores if 0.5 <= s < 0.8]),
            "count_low": len([s for s in confidence_scores if s < 0.5])
        }
        
        response_data = {
            "relations": relations_data,
            "total_count": len(relations_data),
            "processing_time": processing_time,
            "confidence_stats": confidence_stats,
            "extraction_stats": result.stats
        }
        
        # 如果需要包含实体信息
        if request.include_entities and hasattr(result, 'entities'):
            entities_data = []
            for entity in result.entities:
                entity_dict = {
                    "text": entity.text,
                    "type": entity.type,
                    "confidence": entity.confidence,
                    "properties": entity.properties
                }
                entities_data.append(entity_dict)
            response_data["entities"] = entities_data
        
        logger.info(f"关系抽取完成，任务ID: {task_id}，抽取关系数: {len(relations_data)}")
        
        return ExtractionResponse(
            success=True,
            task_id=task_id,
            message=f"成功抽取 {len(relations_data)} 个关系",
            data=response_data,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except RelationExtractionError as e:
        logger.error(f"关系抽取失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"关系抽取失败: {str(e)}"
        )
    except Exception as e:
        logger.error(f"关系抽取异常: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"服务器内部错误: {str(e)}"
        )


@router.post("/extract-batch", response_model=ExtractionResponse, tags=["批量抽取"])
async def extract_batch(
    request: BatchExtractionRequest,
    background_tasks: BackgroundTasks,
    entity_service: EntityService = Depends(get_entity_service),
    relation_service: RelationService = Depends(get_relation_service)
) -> ExtractionResponse:
    """
    批量抽取实体和关系
    
    Args:
        request: 批量抽取请求
        background_tasks: 后台任务
        entity_service: 实体服务
        relation_service: 关系服务
        
    Returns:
        ExtractionResponse: 抽取响应
        
    Raises:
        HTTPException: 当抽取失败时
        
    Example:
        ```json
        {
            "texts": [
                "苹果公司是一家美国跨国科技公司。",
                "谷歌公司总部位于加利福尼亚州山景城。"
            ],
            "language": "zh",
            "extract_entities": true,
            "extract_relations": true,
            "confidence_threshold": 0.7
        }
        ```
    """
    try:
        logger.info(f"开始批量抽取任务，文本数量: {len(request.texts)}")
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 执行批量抽取
        start_time = datetime.utcnow()
        
        batch_results = {
            "entities": [],
            "relations": [],
            "processing_stats": []
        }
        
        # 并发处理多个文本
        tasks = []
        
        for i, text in enumerate(request.texts):
            document_id = request.document_ids[i] if request.document_ids and i < len(request.document_ids) else None
            chunk_id = request.chunk_ids[i] if request.chunk_ids and i < len(request.chunk_ids) else None
            
            if request.extract_entities:
                entity_task = entity_service.extract_entities(
                    text=text,
                    language=request.language,
                    confidence_threshold=request.confidence_threshold,
                    document_id=document_id,
                    chunk_id=chunk_id
                )
                tasks.append(("entity", i, entity_task))
            
            if request.extract_relations:
                relation_task = relation_service.extract_relations(
                    text=text,
                    language=request.language,
                    confidence_threshold=request.confidence_threshold,
                    document_id=document_id,
                    chunk_id=chunk_id
                )
                tasks.append(("relation", i, relation_task))
        
        # 等待所有任务完成
        results = await asyncio.gather(*[task[2] for task in tasks], return_exceptions=True)
        
        # 处理结果
        for (task_type, text_index, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error(f"批量抽取任务失败 (类型: {task_type}, 索引: {text_index}): {str(result)}")
                continue
            
            if task_type == "entity":
                for entity in result.entities:
                    entity_dict = {
                        "text_index": text_index,
                        "text": entity.text,
                        "type": entity.type,
                        "confidence": entity.confidence,
                        "start_pos": entity.start_pos,
                        "end_pos": entity.end_pos,
                        "properties": entity.properties
                    }
                    batch_results["entities"].append(entity_dict)
            
            elif task_type == "relation":
                for relation in result.relations:
                    relation_dict = {
                        "text_index": text_index,
                        "subject": relation.subject,
                        "predicate": relation.predicate,
                        "object": relation.object,
                        "confidence": relation.confidence,
                        "properties": relation.properties
                    }
                    batch_results["relations"].append(relation_dict)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # 构建统计信息
        stats = {
            "total_texts": len(request.texts),
            "total_entities": len(batch_results["entities"]),
            "total_relations": len(batch_results["relations"]),
            "processing_time": processing_time,
            "average_time_per_text": processing_time / len(request.texts) if request.texts else 0
        }
        
        batch_results["stats"] = stats
        
        logger.info(f"批量抽取完成，任务ID: {task_id}，实体数: {stats['total_entities']}，关系数: {stats['total_relations']}")
        
        return ExtractionResponse(
            success=True,
            task_id=task_id,
            message=f"成功处理 {len(request.texts)} 个文本",
            data=batch_results,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"批量抽取异常: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"服务器内部错误: {str(e)}"
        )


@router.get("/extraction-stats", tags=["统计信息"])
async def get_extraction_stats(
    entity_service: EntityService = Depends(get_entity_service),
    relation_service: RelationService = Depends(get_relation_service)
) -> Dict[str, Any]:
    """
    获取抽取统计信息
    
    Args:
        entity_service: 实体服务
        relation_service: 关系服务
        
    Returns:
        Dict[str, Any]: 统计信息
        
    Example:
        ```json
        {
            "entity_stats": {
                "total_extractions": 1000,
                "success_rate": 0.95,
                "average_entities_per_text": 5.2
            },
            "relation_stats": {
                "total_extractions": 800,
                "success_rate": 0.92,
                "average_relations_per_text": 3.1
            }
        }
        ```
    """
    try:
        logger.info("获取抽取统计信息")
        
        # 获取实体服务统计
        entity_stats = entity_service.get_stats()
        
        # 获取关系服务统计
        relation_stats = relation_service.get_stats()
        
        # 合并统计信息
        combined_stats = {
            "entity_stats": entity_stats,
            "relation_stats": relation_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return combined_stats
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取统计信息失败: {str(e)}"
        )


@router.get("/health", tags=["健康检查"])
async def health_check(
    entity_service: EntityService = Depends(get_entity_service),
    relation_service: RelationService = Depends(get_relation_service)
) -> Dict[str, Any]:
    """
    知识抽取服务健康检查
    
    Args:
        entity_service: 实体服务
        relation_service: 关系服务
        
    Returns:
        Dict[str, Any]: 健康状态
        
    Example:
        ```json
        {
            "status": "healthy",
            "services": {
                "entity_service": "healthy",
                "relation_service": "healthy"
            },
            "timestamp": "2024-01-01T00:00:00Z"
        }
        ```
    """
    try:
        logger.info("执行知识抽取服务健康检查")
        
        # 检查实体服务
        entity_health = await entity_service.health_check()
        
        # 检查关系服务
        relation_health = await relation_service.health_check()
        
        # 判断整体健康状态
        overall_status = "healthy"
        if (entity_health.get("status") != "healthy" or 
            relation_health.get("status") != "healthy"):
            overall_status = "unhealthy"
        
        health_status = {
            "status": overall_status,
            "services": {
                "entity_service": entity_health.get("status", "unknown"),
                "relation_service": relation_health.get("status", "unknown")
            },
            "details": {
                "entity_service": entity_health,
                "relation_service": relation_health
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