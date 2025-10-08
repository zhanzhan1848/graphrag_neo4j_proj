#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 文档管理 API 端点
=========================

本模块提供文档管理相关的 API 端点，包括：
1. 文档上传 - 支持多种格式的文档上传
2. 文档处理 - 文档解析、分块、向量化
3. 文档状态 - 查询文档处理状态和进度
4. 文档管理 - 文档的增删改查操作
5. 批量操作 - 批量上传和处理文档
6. 文档搜索 - 基于内容和元数据的文档搜索
7. 文档统计 - 文档库的统计信息

所有端点都支持异步处理、进度跟踪和详细的错误处理。

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import uuid
import os
import asyncio
from pathlib import Path

from app.core.logging import get_logger
from app.services.document_service import DocumentService
from app.services.file_storage_service import FileStorageService
from app.services.text_service import TextService
from app.utils.exceptions import (
    DocumentProcessingError,
    FileStorageError,
    ValidationError
)

logger = get_logger(__name__)

# 创建路由器
router = APIRouter()


# Pydantic 模型定义

class DocumentUploadRequest(BaseModel):
    """文档上传请求模型"""
    title: Optional[str] = Field(None, max_length=200, description="文档标题")
    description: Optional[str] = Field(None, max_length=1000, description="文档描述")
    tags: Optional[List[str]] = Field(None, description="文档标签")
    category: Optional[str] = Field(None, max_length=50, description="文档分类")
    language: str = Field(default="zh", description="文档语言")
    auto_process: bool = Field(default=True, description="是否自动处理文档")
    extract_entities: bool = Field(default=True, description="是否抽取实体")
    extract_relations: bool = Field(default=True, description="是否抽取关系")
    chunk_size: int = Field(default=1000, ge=100, le=5000, description="文本分块大小")
    chunk_overlap: int = Field(default=200, ge=0, le=1000, description="分块重叠大小")


class DocumentProcessRequest(BaseModel):
    """文档处理请求模型"""
    document_id: str = Field(..., description="文档ID")
    extract_entities: bool = Field(default=True, description="是否抽取实体")
    extract_relations: bool = Field(default=True, description="是否抽取关系")
    generate_embeddings: bool = Field(default=True, description="是否生成向量嵌入")
    chunk_size: int = Field(default=1000, ge=100, le=5000, description="文本分块大小")
    chunk_overlap: int = Field(default=200, ge=0, le=1000, description="分块重叠大小")
    force_reprocess: bool = Field(default=False, description="是否强制重新处理")


class DocumentQuery(BaseModel):
    """文档查询模型"""
    query: Optional[str] = Field(None, description="搜索关键词")
    category: Optional[str] = Field(None, description="文档分类")
    tags: Optional[List[str]] = Field(None, description="文档标签")
    language: Optional[str] = Field(None, description="文档语言")
    status: Optional[str] = Field(None, description="文档状态")
    date_from: Optional[str] = Field(None, description="开始日期")
    date_to: Optional[str] = Field(None, description="结束日期")
    limit: int = Field(default=20, ge=1, le=100, description="返回结果数量限制")
    skip: int = Field(default=0, ge=0, description="跳过的结果数量")
    order_by: str = Field(default="created_at", description="排序字段")
    order_direction: str = Field(default="DESC", pattern="^(ASC|DESC)$", description="排序方向")


class DocumentUpdateRequest(BaseModel):
    """文档更新请求模型"""
    title: Optional[str] = Field(None, max_length=200, description="文档标题")
    description: Optional[str] = Field(None, max_length=1000, description="文档描述")
    tags: Optional[List[str]] = Field(None, description="文档标签")
    category: Optional[str] = Field(None, max_length=50, description="文档分类")
    language: Optional[str] = Field(None, description="文档语言")


class BatchUploadRequest(BaseModel):
    """批量上传请求模型"""
    documents: List[DocumentUploadRequest] = Field(..., min_items=1, max_items=50, description="文档列表")
    auto_process: bool = Field(default=True, description="是否自动处理所有文档")
    parallel_processing: bool = Field(default=True, description="是否并行处理")
    max_concurrent: int = Field(default=5, ge=1, le=10, description="最大并发数")


class DocumentResponse(BaseModel):
    """文档响应模型"""
    success: bool = Field(..., description="是否成功")
    document_id: Optional[str] = Field(None, description="文档ID")
    message: str = Field(..., description="响应消息")
    data: Optional[Dict[str, Any]] = Field(None, description="响应数据")
    timestamp: str = Field(..., description="时间戳")


class DocumentListResponse(BaseModel):
    """文档列表响应模型"""
    documents: List[Dict[str, Any]] = Field(..., description="文档列表")
    total_count: int = Field(..., description="总文档数")
    page_info: Dict[str, Any] = Field(..., description="分页信息")
    filters: Dict[str, Any] = Field(..., description="应用的过滤条件")


class ProcessingStatus(BaseModel):
    """处理状态模型"""
    document_id: str = Field(..., description="文档ID")
    status: str = Field(..., description="处理状态")
    progress: float = Field(..., ge=0.0, le=1.0, description="处理进度")
    current_step: str = Field(..., description="当前处理步骤")
    total_steps: int = Field(..., description="总步骤数")
    completed_steps: int = Field(..., description="已完成步骤数")
    error_message: Optional[str] = Field(None, description="错误信息")
    started_at: str = Field(..., description="开始时间")
    updated_at: str = Field(..., description="更新时间")
    estimated_completion: Optional[str] = Field(None, description="预计完成时间")


# 依赖注入

async def get_document_service() -> DocumentService:
    """获取文档服务实例"""
    from app.services.document_service import DocumentService
    return DocumentService()


async def get_file_storage_service() -> FileStorageService:
    """获取文件存储服务实例"""
    from app.services.file_storage_service import FileStorageService
    return FileStorageService()


async def get_text_service() -> TextService:
    """获取文本服务实例"""
    from app.services.text_service import TextService
    return TextService()


# 辅助函数

def validate_file_type(filename: str) -> bool:
    """验证文件类型"""
    allowed_extensions = {'.pdf', '.txt', '.md', '.html', '.docx', '.doc', '.rtf'}
    file_ext = Path(filename).suffix.lower()
    return file_ext in allowed_extensions


def get_file_info(file: UploadFile) -> Dict[str, Any]:
    """获取文件信息"""
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": file.size if hasattr(file, 'size') else None
    }


# API 端点

@router.post("/upload", response_model=DocumentResponse, tags=["文档上传"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),  # JSON字符串
    category: Optional[str] = Form(None),
    language: str = Form("zh"),
    auto_process: bool = Form(True),
    extract_entities: bool = Form(True),
    extract_relations: bool = Form(True),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    document_service: DocumentService = Depends(get_document_service),
    file_storage_service: FileStorageService = Depends(get_file_storage_service)
) -> DocumentResponse:
    """
    上传单个文档
    
    Args:
        file: 上传的文件
        title: 文档标题
        description: 文档描述
        tags: 文档标签（JSON字符串）
        category: 文档分类
        language: 文档语言
        auto_process: 是否自动处理
        extract_entities: 是否抽取实体
        extract_relations: 是否抽取关系
        chunk_size: 文本分块大小
        chunk_overlap: 分块重叠大小
        document_service: 文档服务
        file_storage_service: 文件存储服务
        
    Returns:
        DocumentResponse: 上传响应
        
    Raises:
        HTTPException: 当上传失败时
    """
    try:
        logger.info(f"开始上传文档: {file.filename}")
        
        # 验证文件类型
        if not validate_file_type(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的文件类型: {Path(file.filename).suffix}"
            )
        
        # 验证文件大小（100MB限制）
        if hasattr(file, 'size') and file.size > 100 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="文件大小超过100MB限制"
            )
        
        # 生成文档ID
        document_id = str(uuid.uuid4())
        
        # 解析标签
        parsed_tags = []
        if tags:
            try:
                import json
                parsed_tags = json.loads(tags)
            except json.JSONDecodeError:
                parsed_tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
        
        # 读取文件内容
        file_content = await file.read()
        
        # 存储文件
        file_path = await file_storage_service.store_file(
            file_content=file_content,
            filename=file.filename,
            document_id=document_id
        )
        
        # 创建文档记录
        document_data = {
            "id": document_id,
            "title": title or file.filename,
            "description": description,
            "filename": file.filename,
            "file_path": file_path,
            "file_size": len(file_content),
            "content_type": file.content_type,
            "tags": parsed_tags,
            "category": category,
            "language": language,
            "status": "uploaded",
            "created_at": datetime.utcnow().isoformat()
        }
        
        # 保存文档信息
        document = await document_service.create_document(document_data)
        
        # 如果需要自动处理，添加后台任务
        if auto_process:
            background_tasks.add_task(
                process_document_background,
                document_id=document_id,
                extract_entities=extract_entities,
                extract_relations=extract_relations,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                document_service=document_service
            )
            
            # 更新状态为处理中
            await document_service.update_document_status(
                document_id, 
                "processing", 
                "文档已加入处理队列"
            )
        
        logger.info(f"文档上传成功: {document_id}")
        
        return DocumentResponse(
            success=True,
            document_id=document_id,
            message="文档上传成功" + ("，正在后台处理" if auto_process else ""),
            data={
                "document": document,
                "file_info": get_file_info(file),
                "auto_process": auto_process
            },
            timestamp=datetime.utcnow().isoformat()
        )
        
    except FileStorageError as e:
        logger.error(f"文件存储失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文件存储失败: {str(e)}"
        )
    except Exception as e:
        logger.error(f"文档上传失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文档上传失败: {str(e)}"
        )


async def process_document_background(
    document_id: str,
    extract_entities: bool,
    extract_relations: bool,
    chunk_size: int,
    chunk_overlap: int,
    document_service: DocumentService
):
    """后台处理文档的任务"""
    try:
        logger.info(f"开始后台处理文档: {document_id}")
        
        # 处理文档
        await document_service.process_document(
            document_id=document_id,
            extract_entities=extract_entities,
            extract_relations=extract_relations,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # 更新状态为完成
        await document_service.update_document_status(
            document_id, 
            "completed", 
            "文档处理完成"
        )
        
        logger.info(f"文档处理完成: {document_id}")
        
    except Exception as e:
        logger.error(f"文档处理失败: {document_id}, 错误: {str(e)}")
        
        # 更新状态为失败
        await document_service.update_document_status(
            document_id, 
            "failed", 
            f"文档处理失败: {str(e)}"
        )


@router.post("/process", response_model=DocumentResponse, tags=["文档处理"])
async def process_document(
    request: DocumentProcessRequest,
    background_tasks: BackgroundTasks,
    document_service: DocumentService = Depends(get_document_service)
) -> DocumentResponse:
    """
    处理已上传的文档
    
    Args:
        request: 文档处理请求
        background_tasks: 后台任务
        document_service: 文档服务
        
    Returns:
        DocumentResponse: 处理响应
        
    Example:
        ```json
        {
            "document_id": "doc_123",
            "extract_entities": true,
            "extract_relations": true,
            "generate_embeddings": true,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "force_reprocess": false
        }
        ```
    """
    try:
        logger.info(f"开始处理文档: {request.document_id}")
        
        # 检查文档是否存在
        document = await document_service.get_document(request.document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"文档不存在: {request.document_id}"
            )
        
        # 检查文档状态
        if document.get("status") == "processing" and not request.force_reprocess:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="文档正在处理中，请等待完成或使用强制重新处理"
            )
        
        # 更新状态为处理中
        await document_service.update_document_status(
            request.document_id, 
            "processing", 
            "开始处理文档"
        )
        
        # 添加后台处理任务
        background_tasks.add_task(
            process_document_background,
            document_id=request.document_id,
            extract_entities=request.extract_entities,
            extract_relations=request.extract_relations,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            document_service=document_service
        )
        
        logger.info(f"文档处理任务已启动: {request.document_id}")
        
        return DocumentResponse(
            success=True,
            document_id=request.document_id,
            message="文档处理任务已启动",
            data={
                "document_id": request.document_id,
                "processing_options": {
                    "extract_entities": request.extract_entities,
                    "extract_relations": request.extract_relations,
                    "generate_embeddings": request.generate_embeddings,
                    "chunk_size": request.chunk_size,
                    "chunk_overlap": request.chunk_overlap
                }
            },
            timestamp=datetime.utcnow().isoformat()
        )
        
    except DocumentProcessingError as e:
        logger.error(f"文档处理失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"文档处理失败: {str(e)}"
        )
    except Exception as e:
        logger.error(f"文档处理异常: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"服务器内部错误: {str(e)}"
        )


@router.get("/status/{document_id}", response_model=ProcessingStatus, tags=["文档状态"])
async def get_document_status(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service)
) -> ProcessingStatus:
    """
    获取文档处理状态
    
    Args:
        document_id: 文档ID
        document_service: 文档服务
        
    Returns:
        ProcessingStatus: 处理状态
        
    Example:
        ```json
        {
            "document_id": "doc_123",
            "status": "processing",
            "progress": 0.6,
            "current_step": "extracting_relations",
            "total_steps": 5,
            "completed_steps": 3
        }
        ```
    """
    try:
        logger.info(f"获取文档状态: {document_id}")
        
        # 获取文档状态
        status_info = await document_service.get_document_status(document_id)
        
        if not status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"文档不存在: {document_id}"
            )
        
        return ProcessingStatus(
            document_id=document_id,
            status=status_info.get("status", "unknown"),
            progress=status_info.get("progress", 0.0),
            current_step=status_info.get("current_step", ""),
            total_steps=status_info.get("total_steps", 0),
            completed_steps=status_info.get("completed_steps", 0),
            error_message=status_info.get("error_message"),
            started_at=status_info.get("started_at", ""),
            updated_at=status_info.get("updated_at", ""),
            estimated_completion=status_info.get("estimated_completion")
        )
        
    except Exception as e:
        logger.error(f"获取文档状态失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取文档状态失败: {str(e)}"
        )


@router.get("/list", response_model=DocumentListResponse, tags=["文档查询"])
async def list_documents(
    query: Optional[str] = None,
    category: Optional[str] = None,
    tags: Optional[str] = None,  # 逗号分隔的标签
    language: Optional[str] = None,
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 20,
    skip: int = 0,
    order_by: str = "created_at",
    order_direction: str = "DESC",
    document_service: DocumentService = Depends(get_document_service)
) -> DocumentListResponse:
    """
    查询文档列表
    
    Args:
        query: 搜索关键词
        category: 文档分类
        tags: 文档标签（逗号分隔）
        language: 文档语言
        status: 文档状态
        date_from: 开始日期
        date_to: 结束日期
        limit: 返回结果数量限制
        skip: 跳过的结果数量
        order_by: 排序字段
        order_direction: 排序方向
        document_service: 文档服务
        
    Returns:
        DocumentListResponse: 文档列表响应
        
    Example:
        ```
        GET /documents/list?query=机器学习&category=技术&limit=10&skip=0
        ```
    """
    try:
        logger.info(f"查询文档列表，关键词: {query}")
        
        # 解析标签
        parsed_tags = []
        if tags:
            parsed_tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
        
        # 构建查询条件
        query_params = {
            "query": query,
            "category": category,
            "tags": parsed_tags if parsed_tags else None,
            "language": language,
            "status": status,
            "date_from": date_from,
            "date_to": date_to,
            "limit": limit,
            "skip": skip,
            "order_by": order_by,
            "order_direction": order_direction
        }
        
        # 执行查询
        result = await document_service.search_documents(query_params)
        
        # 构建分页信息
        page_info = {
            "current_page": (skip // limit) + 1,
            "page_size": limit,
            "total_count": result.get("total_count", 0),
            "total_pages": ((result.get("total_count", 0) - 1) // limit) + 1 if result.get("total_count", 0) > 0 else 0,
            "has_next": skip + limit < result.get("total_count", 0),
            "has_prev": skip > 0
        }
        
        # 应用的过滤条件
        applied_filters = {k: v for k, v in query_params.items() if v is not None}
        
        logger.info(f"文档查询完成，找到 {len(result.get('documents', []))} 个文档")
        
        return DocumentListResponse(
            documents=result.get("documents", []),
            total_count=result.get("total_count", 0),
            page_info=page_info,
            filters=applied_filters
        )
        
    except Exception as e:
        logger.error(f"文档查询失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文档查询失败: {str(e)}"
        )


@router.get("/{document_id}", response_model=Dict[str, Any], tags=["文档查询"])
async def get_document(
    document_id: str,
    include_content: bool = False,
    include_chunks: bool = False,
    include_entities: bool = False,
    include_relations: bool = False,
    document_service: DocumentService = Depends(get_document_service)
) -> Dict[str, Any]:
    """
    获取单个文档详情
    
    Args:
        document_id: 文档ID
        include_content: 是否包含文档内容
        include_chunks: 是否包含文本块
        include_entities: 是否包含抽取的实体
        include_relations: 是否包含抽取的关系
        document_service: 文档服务
        
    Returns:
        Dict[str, Any]: 文档详情
        
    Example:
        ```
        GET /documents/doc_123?include_content=true&include_entities=true
        ```
    """
    try:
        logger.info(f"获取文档详情: {document_id}")
        
        # 获取文档基本信息
        document = await document_service.get_document(document_id)
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"文档不存在: {document_id}"
            )
        
        # 根据参数获取额外信息
        if include_content:
            content = await document_service.get_document_content(document_id)
            document["content"] = content
        
        if include_chunks:
            chunks = await document_service.get_document_chunks(document_id)
            document["chunks"] = chunks
        
        if include_entities:
            entities = await document_service.get_document_entities(document_id)
            document["entities"] = entities
        
        if include_relations:
            relations = await document_service.get_document_relations(document_id)
            document["relations"] = relations
        
        logger.info(f"文档详情获取完成: {document_id}")
        
        return document
        
    except Exception as e:
        logger.error(f"获取文档详情失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取文档详情失败: {str(e)}"
        )


@router.put("/{document_id}", response_model=DocumentResponse, tags=["文档管理"])
async def update_document(
    document_id: str,
    request: DocumentUpdateRequest,
    document_service: DocumentService = Depends(get_document_service)
) -> DocumentResponse:
    """
    更新文档信息
    
    Args:
        document_id: 文档ID
        request: 更新请求
        document_service: 文档服务
        
    Returns:
        DocumentResponse: 更新响应
        
    Example:
        ```json
        {
            "title": "新的文档标题",
            "description": "更新的描述",
            "tags": ["机器学习", "深度学习"],
            "category": "技术文档"
        }
        ```
    """
    try:
        logger.info(f"更新文档信息: {document_id}")
        
        # 检查文档是否存在
        document = await document_service.get_document(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"文档不存在: {document_id}"
            )
        
        # 构建更新数据
        update_data = {}
        if request.title is not None:
            update_data["title"] = request.title
        if request.description is not None:
            update_data["description"] = request.description
        if request.tags is not None:
            update_data["tags"] = request.tags
        if request.category is not None:
            update_data["category"] = request.category
        if request.language is not None:
            update_data["language"] = request.language
        
        update_data["updated_at"] = datetime.utcnow().isoformat()
        
        # 执行更新
        updated_document = await document_service.update_document(document_id, update_data)
        
        logger.info(f"文档更新完成: {document_id}")
        
        return DocumentResponse(
            success=True,
            document_id=document_id,
            message="文档信息更新成功",
            data={"document": updated_document},
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"文档更新失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文档更新失败: {str(e)}"
        )


@router.delete("/{document_id}", response_model=DocumentResponse, tags=["文档管理"])
async def delete_document(
    document_id: str,
    delete_files: bool = True,
    delete_graph_data: bool = True,
    document_service: DocumentService = Depends(get_document_service),
    file_storage_service: FileStorageService = Depends(get_file_storage_service)
) -> DocumentResponse:
    """
    删除文档
    
    Args:
        document_id: 文档ID
        delete_files: 是否删除文件
        delete_graph_data: 是否删除图数据
        document_service: 文档服务
        file_storage_service: 文件存储服务
        
    Returns:
        DocumentResponse: 删除响应
        
    Example:
        ```
        DELETE /documents/doc_123?delete_files=true&delete_graph_data=true
        ```
    """
    try:
        logger.info(f"删除文档: {document_id}")
        
        # 检查文档是否存在
        document = await document_service.get_document(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"文档不存在: {document_id}"
            )
        
        # 删除文档数据
        await document_service.delete_document(
            document_id=document_id,
            delete_files=delete_files,
            delete_graph_data=delete_graph_data
        )
        
        logger.info(f"文档删除完成: {document_id}")
        
        return DocumentResponse(
            success=True,
            document_id=document_id,
            message="文档删除成功",
            data={
                "deleted_files": delete_files,
                "deleted_graph_data": delete_graph_data
            },
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"文档删除失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文档删除失败: {str(e)}"
        )


@router.get("/download/{document_id}", tags=["文档下载"])
async def download_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
    file_storage_service: FileStorageService = Depends(get_file_storage_service)
) -> FileResponse:
    """
    下载文档文件
    
    Args:
        document_id: 文档ID
        document_service: 文档服务
        file_storage_service: 文件存储服务
        
    Returns:
        FileResponse: 文件响应
        
    Example:
        ```
        GET /documents/download/doc_123
        ```
    """
    try:
        logger.info(f"下载文档: {document_id}")
        
        # 获取文档信息
        document = await document_service.get_document(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"文档不存在: {document_id}"
            )
        
        # 获取文件路径
        file_path = document.get("file_path")
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="文档文件不存在"
            )
        
        # 返回文件
        return FileResponse(
            path=file_path,
            filename=document.get("filename", "document"),
            media_type=document.get("content_type", "application/octet-stream")
        )
        
    except Exception as e:
        logger.error(f"文档下载失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文档下载失败: {str(e)}"
        )


@router.get("/stats", tags=["统计信息"])
async def get_document_stats(
    document_service: DocumentService = Depends(get_document_service)
) -> Dict[str, Any]:
    """
    获取文档统计信息
    
    Args:
        document_service: 文档服务
        
    Returns:
        Dict[str, Any]: 统计信息
        
    Example:
        ```json
        {
            "total_documents": 1000,
            "by_status": {"completed": 800, "processing": 50, "failed": 150},
            "by_category": {"技术": 400, "学术": 300, "其他": 300},
            "by_language": {"zh": 700, "en": 300}
        }
        ```
    """
    try:
        logger.info("获取文档统计信息")
        
        # 获取统计信息
        stats = await document_service.get_document_stats()
        
        return {
            "total_documents": stats.get("total_documents", 0),
            "by_status": stats.get("by_status", {}),
            "by_category": stats.get("by_category", {}),
            "by_language": stats.get("by_language", {}),
            "by_file_type": stats.get("by_file_type", {}),
            "storage_stats": stats.get("storage_stats", {}),
            "processing_stats": stats.get("processing_stats", {}),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取文档统计信息失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取文档统计信息失败: {str(e)}"
        )


@router.get("/health", tags=["健康检查"])
async def health_check(
    document_service: DocumentService = Depends(get_document_service),
    file_storage_service: FileStorageService = Depends(get_file_storage_service)
) -> Dict[str, Any]:
    """
    文档管理服务健康检查
    
    Args:
        document_service: 文档服务
        file_storage_service: 文件存储服务
        
    Returns:
        Dict[str, Any]: 健康状态
    """
    try:
        logger.info("执行文档管理服务健康检查")
        
        # 检查文档服务
        doc_health = await document_service.health_check()
        
        # 检查文件存储服务
        storage_health = await file_storage_service.health_check()
        
        # 判断整体健康状态
        overall_status = "healthy"
        if (doc_health.get("status") != "healthy" or 
            storage_health.get("status") != "healthy"):
            overall_status = "unhealthy"
        
        health_status = {
            "status": overall_status,
            "services": {
                "document_service": doc_health.get("status", "unknown"),
                "file_storage_service": storage_health.get("status", "unknown")
            },
            "details": {
                "document_service": doc_health,
                "file_storage_service": storage_health
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