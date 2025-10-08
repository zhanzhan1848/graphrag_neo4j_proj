#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG RAG 问答 API 端点
========================

本模块提供基于检索增强生成（RAG）的问答 API 端点，包括：
1. 自然语言问答 - 基于知识库的智能问答
2. 多模态查询 - 支持文本、图像等多种输入
3. 上下文对话 - 支持多轮对话和上下文记忆
4. 引用追溯 - 提供答案的来源和证据
5. 个性化配置 - 支持不同的生成策略和参数
6. 批量问答 - 支持批量问题处理
7. 问答历史 - 查询和管理问答历史记录

所有端点都支持流式响应、实时反馈和详细的可追溯性信息。

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from datetime import datetime
import uuid
import json
import asyncio
from enum import Enum

from app.core.logging import get_logger
from app.services.graphrag_service import GraphRAGService
from app.services.embedding_service import EmbeddingService
from app.services.graph_service import GraphService
from app.services.azure_openai_service import AzureOpenAIService
from app.utils.exceptions import (
    RAGError,
    VectorSearchError,
    GraphQueryError
)

logger = get_logger(__name__)

# 创建路由器
router = APIRouter()


# 枚举定义

class QueryType(str, Enum):
    """查询类型枚举"""
    TEXT = "text"
    MULTIMODAL = "multimodal"
    STRUCTURED = "structured"
    CONVERSATIONAL = "conversational"


class ResponseFormat(str, Enum):
    """响应格式枚举"""
    JSON = "json"
    STREAM = "stream"
    MARKDOWN = "markdown"
    HTML = "html"


class RetrievalStrategy(str, Enum):
    """检索策略枚举"""
    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class GenerationMode(str, Enum):
    """生成模式枚举"""
    FACTUAL = "factual"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"


# Pydantic 模型定义

class RAGQueryRequest(BaseModel):
    """RAG 查询请求模型"""
    query: str = Field(..., min_length=1, max_length=2000, description="查询问题")
    query_type: QueryType = Field(default=QueryType.TEXT, description="查询类型")
    context: Optional[List[str]] = Field(None, description="上下文信息")
    conversation_id: Optional[str] = Field(None, description="对话ID")
    user_id: Optional[str] = Field(None, description="用户ID")
    
    # 检索配置
    retrieval_strategy: RetrievalStrategy = Field(default=RetrievalStrategy.HYBRID, description="检索策略")
    max_chunks: int = Field(default=10, ge=1, le=50, description="最大检索块数")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="相似度阈值")
    include_entities: bool = Field(default=True, description="是否包含实体信息")
    include_relations: bool = Field(default=True, description="是否包含关系信息")
    
    # 生成配置
    generation_mode: GenerationMode = Field(default=GenerationMode.FACTUAL, description="生成模式")
    max_tokens: int = Field(default=1000, ge=50, le=4000, description="最大生成token数")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="生成温度")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p采样")
    
    # 响应配置
    response_format: ResponseFormat = Field(default=ResponseFormat.JSON, description="响应格式")
    include_sources: bool = Field(default=True, description="是否包含来源信息")
    include_confidence: bool = Field(default=True, description="是否包含置信度")
    stream_response: bool = Field(default=False, description="是否流式响应")
    
    # 语言和本地化
    language: str = Field(default="zh", description="响应语言")
    locale: str = Field(default="zh-CN", description="本地化设置")


class MultimodalQueryRequest(BaseModel):
    """多模态查询请求模型"""
    text_query: Optional[str] = Field(None, description="文本查询")
    image_urls: Optional[List[str]] = Field(None, description="图像URL列表")
    audio_urls: Optional[List[str]] = Field(None, description="音频URL列表")
    document_ids: Optional[List[str]] = Field(None, description="文档ID列表")
    
    # 继承基础配置
    retrieval_strategy: RetrievalStrategy = Field(default=RetrievalStrategy.HYBRID, description="检索策略")
    generation_mode: GenerationMode = Field(default=GenerationMode.FACTUAL, description="生成模式")
    max_tokens: int = Field(default=1000, ge=50, le=4000, description="最大生成token数")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="生成温度")
    include_sources: bool = Field(default=True, description="是否包含来源信息")
    stream_response: bool = Field(default=False, description="是否流式响应")
    language: str = Field(default="zh", description="响应语言")


class ConversationRequest(BaseModel):
    """对话请求模型"""
    message: str = Field(..., min_length=1, max_length=2000, description="对话消息")
    conversation_id: str = Field(..., description="对话ID")
    user_id: Optional[str] = Field(None, description="用户ID")
    
    # 对话配置
    maintain_context: bool = Field(default=True, description="是否维护上下文")
    max_context_turns: int = Field(default=10, ge=1, le=50, description="最大上下文轮数")
    context_window_size: int = Field(default=4000, ge=1000, le=8000, description="上下文窗口大小")
    
    # 检索和生成配置
    retrieval_strategy: RetrievalStrategy = Field(default=RetrievalStrategy.ADAPTIVE, description="检索策略")
    generation_mode: GenerationMode = Field(default=GenerationMode.CONVERSATIONAL, description="生成模式")
    max_tokens: int = Field(default=800, ge=50, le=2000, description="最大生成token数")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0, description="生成温度")
    
    # 响应配置
    include_sources: bool = Field(default=True, description="是否包含来源信息")
    stream_response: bool = Field(default=True, description="是否流式响应")
    language: str = Field(default="zh", description="响应语言")


class BatchQueryRequest(BaseModel):
    """批量查询请求模型"""
    queries: List[str] = Field(..., min_items=1, max_items=20, description="查询列表")
    batch_id: Optional[str] = Field(None, description="批次ID")
    user_id: Optional[str] = Field(None, description="用户ID")
    
    # 批量处理配置
    parallel_processing: bool = Field(default=True, description="是否并行处理")
    max_concurrent: int = Field(default=5, ge=1, le=10, description="最大并发数")
    timeout_per_query: int = Field(default=30, ge=10, le=120, description="单个查询超时时间（秒）")
    
    # 检索和生成配置
    retrieval_strategy: RetrievalStrategy = Field(default=RetrievalStrategy.HYBRID, description="检索策略")
    generation_mode: GenerationMode = Field(default=GenerationMode.FACTUAL, description="生成模式")
    max_tokens: int = Field(default=500, ge=50, le=2000, description="最大生成token数")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="生成温度")
    
    # 响应配置
    include_sources: bool = Field(default=True, description="是否包含来源信息")
    language: str = Field(default="zh", description="响应语言")


class RAGResponse(BaseModel):
    """RAG 响应模型"""
    success: bool = Field(..., description="是否成功")
    query_id: str = Field(..., description="查询ID")
    answer: str = Field(..., description="生成的答案")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="置信度")
    
    # 检索信息
    retrieved_chunks: Optional[List[Dict[str, Any]]] = Field(None, description="检索到的文本块")
    retrieved_entities: Optional[List[Dict[str, Any]]] = Field(None, description="检索到的实体")
    retrieved_relations: Optional[List[Dict[str, Any]]] = Field(None, description="检索到的关系")
    
    # 来源信息
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="答案来源")
    citations: Optional[List[str]] = Field(None, description="引用信息")
    
    # 元数据
    processing_time: float = Field(..., description="处理时间（秒）")
    token_usage: Dict[str, int] = Field(..., description="Token使用统计")
    retrieval_stats: Dict[str, Any] = Field(..., description="检索统计")
    generation_stats: Dict[str, Any] = Field(..., description="生成统计")
    
    # 时间戳
    timestamp: str = Field(..., description="响应时间戳")


class ConversationResponse(BaseModel):
    """对话响应模型"""
    success: bool = Field(..., description="是否成功")
    conversation_id: str = Field(..., description="对话ID")
    message_id: str = Field(..., description="消息ID")
    response: str = Field(..., description="对话回复")
    
    # 上下文信息
    context_used: bool = Field(..., description="是否使用了上下文")
    context_turns: int = Field(..., description="上下文轮数")
    
    # 检索和来源信息
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="答案来源")
    retrieved_info: Optional[Dict[str, Any]] = Field(None, description="检索信息")
    
    # 统计信息
    processing_time: float = Field(..., description="处理时间（秒）")
    token_usage: Dict[str, int] = Field(..., description="Token使用统计")
    
    # 时间戳
    timestamp: str = Field(..., description="响应时间戳")


class BatchQueryResponse(BaseModel):
    """批量查询响应模型"""
    success: bool = Field(..., description="是否成功")
    batch_id: str = Field(..., description="批次ID")
    total_queries: int = Field(..., description="总查询数")
    successful_queries: int = Field(..., description="成功查询数")
    failed_queries: int = Field(..., description="失败查询数")
    
    # 查询结果
    results: List[Dict[str, Any]] = Field(..., description="查询结果列表")
    errors: List[Dict[str, Any]] = Field(..., description="错误信息列表")
    
    # 统计信息
    total_processing_time: float = Field(..., description="总处理时间（秒）")
    average_processing_time: float = Field(..., description="平均处理时间（秒）")
    total_token_usage: Dict[str, int] = Field(..., description="总Token使用统计")
    
    # 时间戳
    timestamp: str = Field(..., description="响应时间戳")


class QueryHistoryResponse(BaseModel):
    """查询历史响应模型"""
    queries: List[Dict[str, Any]] = Field(..., description="查询历史列表")
    total_count: int = Field(..., description="总查询数")
    page_info: Dict[str, Any] = Field(..., description="分页信息")
    stats: Dict[str, Any] = Field(..., description="统计信息")


# 依赖注入

async def get_rag_service() -> GraphRAGService:
    """获取 RAG 服务实例"""
    from app.core.dependencies import get_graphrag_service
    return await get_graphrag_service()


async def get_vector_service() -> EmbeddingService:
    """获取向量服务实例"""
    from app.core.dependencies import get_embedding_service
    return await get_embedding_service()


async def get_graph_service() -> GraphService:
    """获取图服务实例"""
    from app.core.dependencies import get_graph_service as get_graph_svc
    return await get_graph_svc()


async def get_llm_service() -> AzureOpenAIService:
    """获取 LLM 服务实例"""
    from app.core.dependencies import get_azure_openai_service
    return await get_azure_openai_service()


# 辅助函数

def generate_query_id() -> str:
    """生成查询ID"""
    return f"query_{uuid.uuid4().hex[:12]}"


def generate_conversation_id() -> str:
    """生成对话ID"""
    return f"conv_{uuid.uuid4().hex[:12]}"


def generate_batch_id() -> str:
    """生成批次ID"""
    return f"batch_{uuid.uuid4().hex[:12]}"


async def create_streaming_response(
    generator: AsyncGenerator[str, None],
    content_type: str = "text/plain"
) -> StreamingResponse:
    """创建流式响应"""
    return StreamingResponse(
        generator,
        media_type=content_type,
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# API 端点

@router.post("/query", response_model=None, tags=["RAG问答"])
async def rag_query(
    request: RAGQueryRequest,
    rag_service: GraphRAGService = Depends(get_rag_service)
) -> Union[RAGResponse, StreamingResponse]:
    """
    RAG 智能问答
    
    Args:
        request: RAG查询请求
        rag_service: RAG服务
        
    Returns:
        Union[RAGResponse, StreamingResponse]: 问答响应或流式响应
        
    Example:
        ```json
        {
            "query": "什么是机器学习？",
            "retrieval_strategy": "hybrid",
            "generation_mode": "factual",
            "max_tokens": 1000,
            "include_sources": true,
            "stream_response": false
        }
        ```
    """
    try:
        start_time = datetime.utcnow()
        query_id = generate_query_id()
        
        logger.info(f"开始RAG查询: {query_id}, 问题: {request.query[:100]}...")
        
        # 如果是流式响应
        if request.stream_response:
            async def stream_generator():
                try:
                    async for chunk in rag_service.stream_query(
                        query=request.query,
                        query_id=query_id,
                        config={
                            "retrieval_strategy": request.retrieval_strategy,
                            "generation_mode": request.generation_mode,
                            "max_tokens": request.max_tokens,
                            "temperature": request.temperature,
                            "top_p": request.top_p,
                            "max_chunks": request.max_chunks,
                            "similarity_threshold": request.similarity_threshold,
                            "include_entities": request.include_entities,
                            "include_relations": request.include_relations,
                            "include_sources": request.include_sources,
                            "include_confidence": request.include_confidence,
                            "language": request.language,
                            "context": request.context,
                            "conversation_id": request.conversation_id,
                            "user_id": request.user_id
                        }
                    ):
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                except Exception as e:
                    error_chunk = {
                        "type": "error",
                        "error": str(e),
                        "query_id": query_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
            
            return await create_streaming_response(
                stream_generator(),
                content_type="text/event-stream"
            )
        
        # 非流式响应
        result = await rag_service.query(
            query=request.query,
            query_id=query_id,
            config={
                "retrieval_strategy": request.retrieval_strategy,
                "generation_mode": request.generation_mode,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_chunks": request.max_chunks,
                "similarity_threshold": request.similarity_threshold,
                "include_entities": request.include_entities,
                "include_relations": request.include_relations,
                "include_sources": request.include_sources,
                "include_confidence": request.include_confidence,
                "language": request.language,
                "context": request.context,
                "conversation_id": request.conversation_id,
                "user_id": request.user_id
            }
        )
        
        # 计算处理时间
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"RAG查询完成: {query_id}, 处理时间: {processing_time:.2f}秒")
        
        return RAGResponse(
            success=True,
            query_id=query_id,
            answer=result.get("answer", ""),
            confidence=result.get("confidence"),
            retrieved_chunks=result.get("retrieved_chunks"),
            retrieved_entities=result.get("retrieved_entities"),
            retrieved_relations=result.get("retrieved_relations"),
            sources=result.get("sources"),
            citations=result.get("citations"),
            processing_time=processing_time,
            token_usage=result.get("token_usage", {}),
            retrieval_stats=result.get("retrieval_stats", {}),
            generation_stats=result.get("generation_stats", {}),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except RAGError as e:
        logger.error(f"RAG查询失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"RAG查询失败: {str(e)}"
        )
    except Exception as e:
        logger.error(f"RAG查询异常: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"服务器内部错误: {str(e)}"
        )


@router.post("/multimodal", response_model=None, tags=["多模态查询"])
async def multimodal_query(
    request: MultimodalQueryRequest,
    rag_service: GraphRAGService = Depends(get_rag_service)
) -> Union[RAGResponse, StreamingResponse]:
    """
    多模态智能查询
    
    Args:
        request: 多模态查询请求
        rag_service: RAG服务
        
    Returns:
        Union[RAGResponse, StreamingResponse]: 查询响应或流式响应
        
    Example:
        ```json
        {
            "text_query": "这张图片显示了什么？",
            "image_urls": ["http://example.com/image.jpg"],
            "retrieval_strategy": "hybrid",
            "generation_mode": "analytical",
            "include_sources": true
        }
        ```
    """
    try:
        start_time = datetime.utcnow()
        query_id = generate_query_id()
        
        logger.info(f"开始多模态查询: {query_id}")
        
        # 验证输入
        if not any([request.text_query, request.image_urls, request.audio_urls, request.document_ids]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="至少需要提供一种查询输入（文本、图像、音频或文档）"
            )
        
        # 如果是流式响应
        if request.stream_response:
            async def stream_generator():
                try:
                    async for chunk in rag_service.multimodal_stream_query(
                        text_query=request.text_query,
                        image_urls=request.image_urls,
                        audio_urls=request.audio_urls,
                        document_ids=request.document_ids,
                        query_id=query_id,
                        config={
                            "retrieval_strategy": request.retrieval_strategy,
                            "generation_mode": request.generation_mode,
                            "max_tokens": request.max_tokens,
                            "temperature": request.temperature,
                            "include_sources": request.include_sources,
                            "language": request.language
                        }
                    ):
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                except Exception as e:
                    error_chunk = {
                        "type": "error",
                        "error": str(e),
                        "query_id": query_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
            
            return await create_streaming_response(
                stream_generator(),
                content_type="text/event-stream"
            )
        
        # 非流式响应
        result = await rag_service.multimodal_query(
            text_query=request.text_query,
            image_urls=request.image_urls,
            audio_urls=request.audio_urls,
            document_ids=request.document_ids,
            query_id=query_id,
            config={
                "retrieval_strategy": request.retrieval_strategy,
                "generation_mode": request.generation_mode,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "include_sources": request.include_sources,
                "language": request.language
            }
        )
        
        # 计算处理时间
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"多模态查询完成: {query_id}, 处理时间: {processing_time:.2f}秒")
        
        return RAGResponse(
            success=True,
            query_id=query_id,
            answer=result.get("answer", ""),
            confidence=result.get("confidence"),
            retrieved_chunks=result.get("retrieved_chunks"),
            retrieved_entities=result.get("retrieved_entities"),
            retrieved_relations=result.get("retrieved_relations"),
            sources=result.get("sources"),
            citations=result.get("citations"),
            processing_time=processing_time,
            token_usage=result.get("token_usage", {}),
            retrieval_stats=result.get("retrieval_stats", {}),
            generation_stats=result.get("generation_stats", {}),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"多模态查询失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"多模态查询失败: {str(e)}"
        )


@router.post("/conversation", response_model=None, tags=["对话问答"])
async def conversation_query(
    request: ConversationRequest,
    rag_service: GraphRAGService = Depends(get_rag_service)
) -> Union[ConversationResponse, StreamingResponse]:
    """
    对话式问答
    
    Args:
        request: 对话请求
        rag_service: RAG服务
        
    Returns:
        Union[ConversationResponse, StreamingResponse]: 对话响应或流式响应
        
    Example:
        ```json
        {
            "message": "继续刚才的话题，能详细解释一下吗？",
            "conversation_id": "conv_abc123",
            "maintain_context": true,
            "max_context_turns": 10,
            "stream_response": true
        }
        ```
    """
    try:
        start_time = datetime.utcnow()
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        
        logger.info(f"开始对话查询: {request.conversation_id}, 消息: {message_id}")
        
        # 如果是流式响应
        if request.stream_response:
            async def stream_generator():
                try:
                    async for chunk in rag_service.conversation_stream_query(
                        message=request.message,
                        conversation_id=request.conversation_id,
                        message_id=message_id,
                        user_id=request.user_id,
                        config={
                            "maintain_context": request.maintain_context,
                            "max_context_turns": request.max_context_turns,
                            "context_window_size": request.context_window_size,
                            "retrieval_strategy": request.retrieval_strategy,
                            "generation_mode": request.generation_mode,
                            "max_tokens": request.max_tokens,
                            "temperature": request.temperature,
                            "include_sources": request.include_sources,
                            "language": request.language
                        }
                    ):
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                except Exception as e:
                    error_chunk = {
                        "type": "error",
                        "error": str(e),
                        "conversation_id": request.conversation_id,
                        "message_id": message_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
            
            return await create_streaming_response(
                stream_generator(),
                content_type="text/event-stream"
            )
        
        # 非流式响应
        result = await rag_service.conversation_query(
            message=request.message,
            conversation_id=request.conversation_id,
            message_id=message_id,
            user_id=request.user_id,
            config={
                "maintain_context": request.maintain_context,
                "max_context_turns": request.max_context_turns,
                "context_window_size": request.context_window_size,
                "retrieval_strategy": request.retrieval_strategy,
                "generation_mode": request.generation_mode,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "include_sources": request.include_sources,
                "language": request.language
            }
        )
        
        # 计算处理时间
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"对话查询完成: {request.conversation_id}, 处理时间: {processing_time:.2f}秒")
        
        return ConversationResponse(
            success=True,
            conversation_id=request.conversation_id,
            message_id=message_id,
            response=result.get("response", ""),
            context_used=result.get("context_used", False),
            context_turns=result.get("context_turns", 0),
            sources=result.get("sources"),
            retrieved_info=result.get("retrieved_info"),
            processing_time=processing_time,
            token_usage=result.get("token_usage", {}),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"对话查询失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"对话查询失败: {str(e)}"
        )


@router.post("/batch", response_model=BatchQueryResponse, tags=["批量问答"])
async def batch_query(
    request: BatchQueryRequest,
    background_tasks: BackgroundTasks,
    rag_service: GraphRAGService = Depends(get_rag_service)
) -> BatchQueryResponse:
    """
    批量问答查询
    
    Args:
        request: 批量查询请求
        background_tasks: 后台任务
        rag_service: RAG服务
        
    Returns:
        BatchQueryResponse: 批量查询响应
        
    Example:
        ```json
        {
            "queries": [
                "什么是机器学习？",
                "深度学习的应用有哪些？",
                "如何选择合适的算法？"
            ],
            "parallel_processing": true,
            "max_concurrent": 3,
            "include_sources": true
        }
        ```
    """
    try:
        start_time = datetime.utcnow()
        batch_id = request.batch_id or generate_batch_id()
        
        logger.info(f"开始批量查询: {batch_id}, 查询数量: {len(request.queries)}")
        
        # 执行批量查询
        results = await rag_service.batch_query(
            queries=request.queries,
            batch_id=batch_id,
            user_id=request.user_id,
            config={
                "parallel_processing": request.parallel_processing,
                "max_concurrent": request.max_concurrent,
                "timeout_per_query": request.timeout_per_query,
                "retrieval_strategy": request.retrieval_strategy,
                "generation_mode": request.generation_mode,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "include_sources": request.include_sources,
                "language": request.language
            }
        )
        
        # 计算统计信息
        total_processing_time = (datetime.utcnow() - start_time).total_seconds()
        successful_results = [r for r in results.get("results", []) if r.get("success")]
        failed_results = [r for r in results.get("results", []) if not r.get("success")]
        
        # 计算平均处理时间
        processing_times = [r.get("processing_time", 0) for r in successful_results]
        average_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # 计算总Token使用量
        total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for result in successful_results:
            token_usage = result.get("token_usage", {})
            for key in total_token_usage:
                total_token_usage[key] += token_usage.get(key, 0)
        
        logger.info(f"批量查询完成: {batch_id}, 成功: {len(successful_results)}, 失败: {len(failed_results)}")
        
        return BatchQueryResponse(
            success=True,
            batch_id=batch_id,
            total_queries=len(request.queries),
            successful_queries=len(successful_results),
            failed_queries=len(failed_results),
            results=results.get("results", []),
            errors=results.get("errors", []),
            total_processing_time=total_processing_time,
            average_processing_time=average_processing_time,
            total_token_usage=total_token_usage,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"批量查询失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量查询失败: {str(e)}"
        )


@router.get("/conversations/{conversation_id}", tags=["对话管理"])
async def get_conversation_history(
    conversation_id: str,
    limit: int = 50,
    skip: int = 0,
    rag_service: GraphRAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    获取对话历史
    
    Args:
        conversation_id: 对话ID
        limit: 返回消息数量限制
        skip: 跳过的消息数量
        rag_service: RAG服务
        
    Returns:
        Dict[str, Any]: 对话历史
        
    Example:
        ```
        GET /rag/conversations/conv_abc123?limit=20&skip=0
        ```
    """
    try:
        logger.info(f"获取对话历史: {conversation_id}")
        
        # 获取对话历史
        history = await rag_service.get_conversation_history(
            conversation_id=conversation_id,
            limit=limit,
            skip=skip
        )
        
        if not history:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"对话不存在: {conversation_id}"
            )
        
        return {
            "conversation_id": conversation_id,
            "messages": history.get("messages", []),
            "total_messages": history.get("total_messages", 0),
            "conversation_info": history.get("conversation_info", {}),
            "page_info": {
                "current_page": (skip // limit) + 1,
                "page_size": limit,
                "total_pages": ((history.get("total_messages", 0) - 1) // limit) + 1,
                "has_next": skip + limit < history.get("total_messages", 0),
                "has_prev": skip > 0
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取对话历史失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取对话历史失败: {str(e)}"
        )


@router.delete("/conversations/{conversation_id}", tags=["对话管理"])
async def delete_conversation(
    conversation_id: str,
    rag_service: GraphRAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    删除对话
    
    Args:
        conversation_id: 对话ID
        rag_service: RAG服务
        
    Returns:
        Dict[str, Any]: 删除结果
        
    Example:
        ```
        DELETE /rag/conversations/conv_abc123
        ```
    """
    try:
        logger.info(f"删除对话: {conversation_id}")
        
        # 删除对话
        result = await rag_service.delete_conversation(conversation_id)
        
        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"对话不存在: {conversation_id}"
            )
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "message": "对话删除成功",
            "deleted_messages": result.get("deleted_messages", 0),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"删除对话失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除对话失败: {str(e)}"
        )


@router.get("/history", response_model=QueryHistoryResponse, tags=["查询历史"])
async def get_query_history(
    user_id: Optional[str] = None,
    query_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 50,
    skip: int = 0,
    rag_service: GraphRAGService = Depends(get_rag_service)
) -> QueryHistoryResponse:
    """
    获取查询历史
    
    Args:
        user_id: 用户ID
        query_type: 查询类型
        date_from: 开始日期
        date_to: 结束日期
        limit: 返回结果数量限制
        skip: 跳过的结果数量
        rag_service: RAG服务
        
    Returns:
        QueryHistoryResponse: 查询历史响应
        
    Example:
        ```
        GET /rag/history?user_id=user123&limit=20&skip=0
        ```
    """
    try:
        logger.info(f"获取查询历史，用户: {user_id}")
        
        # 构建查询条件
        query_params = {
            "user_id": user_id,
            "query_type": query_type,
            "date_from": date_from,
            "date_to": date_to,
            "limit": limit,
            "skip": skip
        }
        
        # 获取查询历史
        history = await rag_service.get_query_history(query_params)
        
        # 构建分页信息
        page_info = {
            "current_page": (skip // limit) + 1,
            "page_size": limit,
            "total_count": history.get("total_count", 0),
            "total_pages": ((history.get("total_count", 0) - 1) // limit) + 1,
            "has_next": skip + limit < history.get("total_count", 0),
            "has_prev": skip > 0
        }
        
        return QueryHistoryResponse(
            queries=history.get("queries", []),
            total_count=history.get("total_count", 0),
            page_info=page_info,
            stats=history.get("stats", {})
        )
        
    except Exception as e:
        logger.error(f"获取查询历史失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取查询历史失败: {str(e)}"
        )


@router.get("/stats", tags=["统计信息"])
async def get_rag_stats(
    user_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    rag_service: GraphRAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    获取 RAG 统计信息
    
    Args:
        user_id: 用户ID
        date_from: 开始日期
        date_to: 结束日期
        rag_service: RAG服务
        
    Returns:
        Dict[str, Any]: 统计信息
        
    Example:
        ```json
        {
            "total_queries": 10000,
            "successful_queries": 9500,
            "average_response_time": 2.5,
            "token_usage": {"total": 1000000, "average": 100},
            "popular_topics": ["机器学习", "深度学习", "自然语言处理"]
        }
        ```
    """
    try:
        logger.info(f"获取RAG统计信息，用户: {user_id}")
        
        # 获取统计信息
        stats = await rag_service.get_stats({
            "user_id": user_id,
            "date_from": date_from,
            "date_to": date_to
        })
        
        return {
            "query_stats": stats.get("query_stats", {}),
            "performance_stats": stats.get("performance_stats", {}),
            "usage_stats": stats.get("usage_stats", {}),
            "popular_topics": stats.get("popular_topics", []),
            "error_stats": stats.get("error_stats", {}),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取RAG统计信息失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取RAG统计信息失败: {str(e)}"
        )


@router.get("/health", tags=["健康检查"])
async def health_check(
    rag_service: GraphRAGService = Depends(get_rag_service),
    vector_service: EmbeddingService = Depends(get_vector_service),
    graph_service: GraphService = Depends(get_graph_service),
    llm_service: AzureOpenAIService = Depends(get_llm_service)
) -> Dict[str, Any]:
    """
    RAG 服务健康检查
    
    Args:
        rag_service: RAG服务
        vector_service: 向量服务
        graph_service: 图服务
        llm_service: LLM服务
        
    Returns:
        Dict[str, Any]: 健康状态
    """
    try:
        logger.info("执行RAG服务健康检查")
        
        # 检查各个服务
        rag_health = await rag_service.health_check()
        vector_health = await vector_service.health_check()
        graph_health = await graph_service.health_check()
        llm_health = await llm_service.health_check()
        
        # 判断整体健康状态
        overall_status = "healthy"
        if any(health.get("status") != "healthy" for health in [
            rag_health, vector_health, graph_health, llm_health
        ]):
            overall_status = "unhealthy"
        
        health_status = {
            "status": overall_status,
            "services": {
                "rag_service": rag_health.get("status", "unknown"),
                "vector_service": vector_health.get("status", "unknown"),
                "graph_service": graph_health.get("status", "unknown"),
                "llm_service": llm_health.get("status", "unknown")
            },
            "details": {
                "rag_service": rag_health,
                "vector_service": vector_health,
                "graph_service": graph_health,
                "llm_service": llm_health
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