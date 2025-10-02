#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG API 端点
================

本模块提供 GraphRAG 相关的 API 端点，包括：
1. 文档处理和分析
2. 实体关系抽取
3. 知识图谱查询
4. RAG 问答
5. 服务健康检查

所有端点都集成了 Azure OpenAI 服务，支持异步处理和错误处理。

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from app.core.logging import get_logger
from app.services.graphrag_service import get_graphrag_service, GraphRAGService
from app.services.azure_openai_service import get_azure_openai_service, AzureOpenAIService

logger = get_logger(__name__)

# 创建路由器
router = APIRouter()


# Pydantic 模型定义
class DocumentProcessRequest(BaseModel):
    """文档处理请求模型"""
    text: str = Field(..., description="文档文本内容", min_length=1)
    document_id: Optional[str] = Field(None, description="文档ID，如果不提供将自动生成")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="文档元数据")


class KnowledgeGraphQueryRequest(BaseModel):
    """知识图谱查询请求模型"""
    query: str = Field(..., description="查询问题", min_length=1)
    max_results: Optional[int] = Field(10, description="最大结果数", ge=1, le=100)
    include_context: Optional[bool] = Field(True, description="是否包含上下文信息")


class RAGQueryRequest(BaseModel):
    """RAG 查询请求模型"""
    question: str = Field(..., description="用户问题", min_length=1)
    max_context_length: Optional[int] = Field(2000, description="最大上下文长度", ge=100, le=8000)
    temperature: Optional[float] = Field(0.7, description="生成温度", ge=0.0, le=2.0)


class EntityExtractionRequest(BaseModel):
    """实体抽取请求模型"""
    text: str = Field(..., description="输入文本", min_length=1)
    entity_types: Optional[List[str]] = Field(None, description="指定实体类型")


class RelationshipExtractionRequest(BaseModel):
    """关系抽取请求模型"""
    text: str = Field(..., description="输入文本", min_length=1)
    entities: Optional[List[Dict[str, Any]]] = Field(None, description="已知实体列表")


@router.post("/process-document", tags=["文档处理"])
async def process_document(
    request: DocumentProcessRequest,
    graphrag_service: GraphRAGService = Depends(get_graphrag_service)
) -> Dict[str, Any]:
    """
    处理文档并抽取知识
    
    对输入的文档进行分块、实体抽取、关系抽取和断言抽取，
    构建知识图谱并生成向量嵌入。
    
    Args:
        request: 文档处理请求
        
    Returns:
        处理结果，包括分块、实体、关系和断言信息
        
    Raises:
        HTTPException: 处理失败时抛出异常
    """
    try:
        logger.info(f"开始处理文档: {request.document_id}")
        
        # 生成文档ID（如果未提供）
        document_id = request.document_id or str(uuid.uuid4())
        
        # 处理文档
        result = await graphrag_service.process_document(document_id, request.text)
        
        # 添加请求元数据
        result["metadata"] = request.metadata
        result["request_timestamp"] = datetime.utcnow().isoformat()
        
        logger.info(f"文档处理完成: {document_id}")
        return result
        
    except Exception as e:
        logger.error(f"文档处理失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文档处理失败: {str(e)}"
        )


@router.post("/upload-document", tags=["文档处理"])
async def upload_document(
    file: UploadFile = File(..., description="上传的文档文件"),
    metadata: Optional[str] = Form(None, description="文档元数据（JSON格式）"),
    graphrag_service: GraphRAGService = Depends(get_graphrag_service)
) -> Dict[str, Any]:
    """
    上传并处理文档文件
    
    支持上传文本文件，自动提取内容并进行知识抽取。
    
    Args:
        file: 上传的文档文件
        metadata: 文档元数据（可选）
        
    Returns:
        处理结果
        
    Raises:
        HTTPException: 上传或处理失败时抛出异常
    """
    try:
        logger.info(f"开始处理上传文件: {file.filename}")
        
        # 检查文件类型
        if not file.content_type.startswith('text/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="仅支持文本文件"
            )
        
        # 读取文件内容
        content = await file.read()
        text = content.decode('utf-8')
        
        # 生成文档ID
        document_id = f"upload_{uuid.uuid4()}"
        
        # 处理文档
        result = await graphrag_service.process_document(document_id, text)
        
        # 添加文件信息
        result["file_info"] = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content)
        }
        
        if metadata:
            import json
            try:
                result["metadata"] = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning("元数据格式错误，忽略")
        
        logger.info(f"文件处理完成: {file.filename}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件处理失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文件处理失败: {str(e)}"
        )


@router.post("/query-knowledge-graph", tags=["知识图谱"])
async def query_knowledge_graph(
    request: KnowledgeGraphQueryRequest,
    graphrag_service: GraphRAGService = Depends(get_graphrag_service)
) -> Dict[str, Any]:
    """
    查询知识图谱
    
    基于自然语言查询知识图谱，返回相关的实体、关系和上下文信息。
    
    Args:
        request: 知识图谱查询请求
        
    Returns:
        查询结果，包括答案和相关上下文
        
    Raises:
        HTTPException: 查询失败时抛出异常
    """
    try:
        logger.info(f"执行知识图谱查询: {request.query}")
        
        result = await graphrag_service.query_knowledge_graph(
            query=request.query,
            max_results=request.max_results
        )
        
        # 添加请求信息
        result["request"] = {
            "query": request.query,
            "max_results": request.max_results,
            "include_context": request.include_context,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"知识图谱查询完成: {request.query}")
        return result
        
    except Exception as e:
        logger.error(f"知识图谱查询失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"知识图谱查询失败: {str(e)}"
        )


@router.post("/rag-query", tags=["RAG问答"])
async def rag_query(
    request: RAGQueryRequest,
    graphrag_service: GraphRAGService = Depends(get_graphrag_service)
) -> Dict[str, Any]:
    """
    RAG 问答查询
    
    基于检索增强生成（RAG）技术回答用户问题，
    结合知识图谱和向量检索提供准确答案。
    
    Args:
        request: RAG 查询请求
        
    Returns:
        问答结果，包括答案、来源和置信度
        
    Raises:
        HTTPException: 查询失败时抛出异常
    """
    try:
        logger.info(f"执行 RAG 查询: {request.question}")
        
        # 使用知识图谱查询作为 RAG 的基础
        result = await graphrag_service.query_knowledge_graph(
            query=request.question,
            max_results=10
        )
        
        # 添加 RAG 特定信息
        result["rag_config"] = {
            "max_context_length": request.max_context_length,
            "temperature": request.temperature
        }
        
        result["request"] = {
            "question": request.question,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"RAG 查询完成: {request.question}")
        return result
        
    except Exception as e:
        logger.error(f"RAG 查询失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG 查询失败: {str(e)}"
        )


@router.post("/extract-entities", tags=["知识抽取"])
async def extract_entities(
    request: EntityExtractionRequest,
    graphrag_service: GraphRAGService = Depends(get_graphrag_service)
) -> Dict[str, Any]:
    """
    实体抽取
    
    从输入文本中抽取命名实体，包括人物、组织、地点、概念等。
    
    Args:
        request: 实体抽取请求
        
    Returns:
        抽取的实体列表
        
    Raises:
        HTTPException: 抽取失败时抛出异常
    """
    try:
        logger.info("执行实体抽取")
        
        entities = await graphrag_service.extract_entities(request.text)
        
        result = {
            "entities": entities,
            "entity_count": len(entities),
            "request": {
                "text_length": len(request.text),
                "entity_types": request.entity_types,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        logger.info(f"实体抽取完成，抽取到 {len(entities)} 个实体")
        return result
        
    except Exception as e:
        logger.error(f"实体抽取失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"实体抽取失败: {str(e)}"
        )


@router.post("/extract-relationships", tags=["知识抽取"])
async def extract_relationships(
    request: RelationshipExtractionRequest,
    graphrag_service: GraphRAGService = Depends(get_graphrag_service)
) -> Dict[str, Any]:
    """
    关系抽取
    
    从输入文本中抽取实体之间的关系。
    
    Args:
        request: 关系抽取请求
        
    Returns:
        抽取的关系列表
        
    Raises:
        HTTPException: 抽取失败时抛出异常
    """
    try:
        logger.info("执行关系抽取")
        
        # 如果没有提供实体，先进行实体抽取
        entities = request.entities
        if not entities:
            entities = await graphrag_service.extract_entities(request.text)
        
        relationships = await graphrag_service.extract_relationships(request.text, entities)
        
        result = {
            "relationships": relationships,
            "relationship_count": len(relationships),
            "entities": entities,
            "entity_count": len(entities),
            "request": {
                "text_length": len(request.text),
                "provided_entities": bool(request.entities),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        logger.info(f"关系抽取完成，抽取到 {len(relationships)} 个关系")
        return result
        
    except Exception as e:
        logger.error(f"关系抽取失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"关系抽取失败: {str(e)}"
        )


@router.get("/health", tags=["系统状态"])
async def health_check(
    graphrag_service: GraphRAGService = Depends(get_graphrag_service),
    azure_openai_service: AzureOpenAIService = Depends(get_azure_openai_service)
) -> Dict[str, Any]:
    """
    GraphRAG 服务健康检查
    
    检查 GraphRAG 服务和相关依赖的健康状态。
    
    Returns:
        健康状态信息
    """
    try:
        logger.info("执行 GraphRAG 健康检查")
        
        # 检查 GraphRAG 服务
        graphrag_health = await graphrag_service.health_check()
        
        # 检查 Azure OpenAI 服务
        azure_openai_healthy = await azure_openai_service.health_check()
        
        # 获取模型信息
        model_info = azure_openai_service.get_model_info()
        
        result = {
            "status": "healthy" if graphrag_health["status"] == "healthy" and azure_openai_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "graphrag": graphrag_health,
                "azure_openai": {
                    "status": "healthy" if azure_openai_healthy else "unhealthy",
                    "models": model_info
                }
            }
        }
        
        logger.info(f"GraphRAG 健康检查完成: {result['status']}")
        return result
        
    except Exception as e:
        logger.error(f"GraphRAG 健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/models", tags=["系统信息"])
async def get_model_info(
    azure_openai_service: AzureOpenAIService = Depends(get_azure_openai_service)
) -> Dict[str, Any]:
    """
    获取模型配置信息
    
    返回当前使用的 Azure OpenAI 模型配置信息。
    
    Returns:
        模型配置信息
    """
    try:
        logger.info("获取模型配置信息")
        
        model_info = azure_openai_service.get_model_info()
        
        result = {
            "models": model_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info("模型配置信息获取完成")
        return result
        
    except Exception as e:
        logger.error(f"获取模型配置信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模型配置信息失败: {str(e)}"
        )