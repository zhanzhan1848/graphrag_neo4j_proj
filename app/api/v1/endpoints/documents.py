#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 文档管理 API 端点
=========================

本模块实现了文档管理相关的 API 端点。

端点功能：
- POST /documents/upload - 文档上传
- GET /documents - 文档列表查询
- GET /documents/{document_id} - 获取单个文档信息
- PUT /documents/{document_id} - 更新文档信息
- DELETE /documents/{document_id} - 删除文档
- POST /documents/{document_id}/process - 手动触发文档处理
- GET /documents/{document_id}/status - 获取文档处理状态
- GET /documents/stats - 获取文档统计信息

支持功能：
- 多种文档格式：PDF、TXT、Markdown、HTML、DOCX等
- 文件上传验证和安全检查
- 异步文档处理
- 处理进度跟踪
- 批量操作支持

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import os
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import (
    APIRouter, 
    Depends, 
    HTTPException, 
    UploadFile, 
    File, 
    Form,
    Query,
    BackgroundTasks,
    status
)
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.models.schemas.documents import (
    DocumentCreate,
    DocumentUpdate,
    DocumentResponse,
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentSearchRequest,
    DocumentStatsResponse,
    DocumentStatus
)
from app.models.schemas.base import (
    StatusResponse,
    ErrorResponse,
    PaginationParams
)
from app.services.document_service import DocumentService
from app.services.file_storage_service import FileStorageService
from app.utils.exceptions import (
    DocumentNotFoundError,
    DocumentValidationError,
    FileValidationError,
    FileStorageError
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 创建路由器
router = APIRouter(prefix="/documents", tags=["documents"])

# 支持的文件类型
SUPPORTED_MIME_TYPES = {
    "application/pdf": "pdf",
    "text/plain": "txt",
    "text/markdown": "markdown",
    "text/html": "html",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "doc",
    "application/rtf": "rtf",
    "application/vnd.oasis.opendocument.text": "odt",
    "application/epub+zip": "epub",
    "image/png": "image",
    "image/jpeg": "image",
    "image/gif": "image",
    "image/webp": "image"
}

# 最大文件大小（100MB）
MAX_FILE_SIZE = 100 * 1024 * 1024


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="上传文档",
    description="上传文档文件并创建文档记录，支持多种格式"
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="要上传的文档文件"),
    title: Optional[str] = Form(None, description="文档标题"),
    description: Optional[str] = Form(None, description="文档描述"),
    author: Optional[str] = Form(None, description="文档作者"),
    tags: Optional[str] = Form(None, description="标签，用逗号分隔"),
    categories: Optional[str] = Form(None, description="分类，用逗号分隔"),
    language: str = Form("zh", description="文档语言"),
    is_public: bool = Form(False, description="是否公开"),
    auto_process: bool = Form(True, description="是否自动处理"),
    db: Session = Depends(get_db)
) -> DocumentUploadResponse:
    """
    上传文档文件
    
    Args:
        background_tasks: 后台任务管理器
        file: 上传的文件
        title: 文档标题
        description: 文档描述
        author: 文档作者
        tags: 标签字符串
        categories: 分类字符串
        language: 文档语言
        is_public: 是否公开
        auto_process: 是否自动处理
        db: 数据库会话
        
    Returns:
        文档上传响应
        
    Raises:
        HTTPException: 文件验证失败或上传错误
    """
    try:
        logger.info(f"开始上传文档: {file.filename}")
        
        # 验证文件
        await validate_file_upload(file, MAX_FILE_SIZE, SUPPORTED_MIME_TYPES)
        
        # 生成文档ID
        document_id = uuid.uuid4()
        
        # 确定文档类型
        document_type = SUPPORTED_MIME_TYPES.get(file.content_type, "other")
        
        # 生成文件路径
        file_storage_service = FileStorageService()
        file_path = await file_storage_service.save_uploaded_file(
            file, 
            document_id, 
            document_type
        )
        
        # 处理标签和分类
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else []
        category_list = [cat.strip() for cat in categories.split(",")] if categories else []
        
        # 创建文档记录
        document_data = DocumentCreate(
            title=title or file.filename,
            description=description,
            file_name=file.filename,
            file_path=file_path,
            file_size=file.size,
            mime_type=file.content_type,
            document_type=document_type,
            language=language,
            author=author,
            tags=tag_list,
            categories=category_list,
            is_public=is_public,
            auto_process=auto_process
        )
        
        # 保存到数据库
        document_service = DocumentService(db)
        document = await document_service.create_document(document_data, document_id)
        
        # 如果启用自动处理，添加后台任务
        processing_started = False
        if auto_process:
            processing_service = ProcessingService()
            background_tasks.add_task(
                processing_service.process_document,
                document_id,
                db
            )
            processing_started = True
        
        logger.info(f"文档上传成功: {document_id}")
        
        return DocumentUploadResponse(
            document_id=document_id,
            file_path=file_path,
            status=DocumentStatus.UPLOADED,
            message="文档上传成功",
            processing_started=processing_started
        )
        
    except (FileValidationError, FileStorageError) as e:
        logger.error(f"文件处理失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"文件上传失败: {str(e)}"
        )
    except DocumentValidationError as e:
        logger.error(f"文档验证失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"文档验证失败: {str(e)}"
        )
    except Exception as e:
        logger.error(f"上传文档失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"上传过程中发生错误: {str(e)}"
        )


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    offset: int = Query(0, ge=0, description="偏移量"),
    limit: int = Query(20, ge=1, le=100, description="限制数量"),
    status_filter: Optional[List[DocumentStatus]] = Query(None, description="状态过滤"),
    author: Optional[str] = Query(None, description="作者过滤"),
    tags: Optional[List[str]] = Query(None, description="标签过滤"),
    categories: Optional[List[str]] = Query(None, description="分类过滤"),
    db: Session = Depends(get_db)
) -> DocumentListResponse:
    """
    获取文档列表
    """
    try:
        document_service = DocumentService(db)
        
        # 构建搜索请求
        search_request = DocumentSearchRequest(
            status=status_filter,
            author=author,
            tags=tags,
            categories=categories
        )
        
        documents, total = await document_service.search_documents(
            search_request, offset, limit
        )
        
        return DocumentListResponse(
            documents=[DocumentResponse.from_orm(doc) for doc in documents],
            total=total,
            offset=offset,
            limit=limit
        )
        
    except Exception as e:
        logger.error(f"获取文档列表失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取文档列表失败"
        )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    db: Session = Depends(get_db)
) -> DocumentResponse:
    """
    获取单个文档详情
    """
    try:
        document_service = DocumentService(db)
        document = await document_service.get_document(document_id)
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="文档不存在"
            )
        
        return DocumentResponse.from_orm(document)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取文档失败"
        )


@router.put("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: uuid.UUID,
    document_update: DocumentUpdate,
    db: Session = Depends(get_db)
) -> DocumentResponse:
    """
    更新文档信息
    """
    try:
        document_service = DocumentService(db)
        document = await document_service.update_document(document_id, document_update)
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="文档不存在"
            )
        
        return DocumentResponse.from_orm(document)
        
    except DocumentNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在"
        )
    except DocumentValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"更新文档失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="更新文档失败"
        )


@router.delete("/{document_id}", response_model=StatusResponse)
async def delete_document(
    document_id: uuid.UUID,
    force: bool = Query(False, description="是否强制删除"),
    db: Session = Depends(get_db)
) -> StatusResponse:
    """
    删除文档
    """
    try:
        document_service = DocumentService(db)
        success = await document_service.delete_document(document_id, force)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="文档不存在或删除失败"
            )
        
        return StatusResponse(
            success=True,
            message="文档删除成功"
        )
        
    except DocumentNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在"
        )
    except Exception as e:
        logger.error(f"删除文档失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除文档失败"
        )


@router.get("/{document_id}/status")
async def get_document_status(
    document_id: uuid.UUID,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    获取文档处理状态
    """
    try:
        document_service = DocumentService(db)
        status_info = await document_service.get_document_status(document_id)
        
        if not status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="文档不存在"
            )
        
        return status_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档状态失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取文档状态失败"
        )


@router.get("/stats/overview", response_model=DocumentStatsResponse)
async def get_document_stats(
    db: Session = Depends(get_db)
) -> DocumentStatsResponse:
    """
    获取文档统计信息
    """
    try:
        document_service = DocumentService(db)
        stats = await document_service.get_document_stats()
        
        return DocumentStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"获取文档统计失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取文档统计失败"
        )


@router.post(
    "/batch/upload",
    response_model=List[DocumentUploadResponse],
    summary="批量上传文档",
    description="批量上传多个文档文件"
)
async def batch_upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="要上传的文档文件列表"),
    auto_process: bool = Form(True, description="是否自动处理"),
    db: Session = Depends(get_db)
) -> List[DocumentUploadResponse]:
    """
    批量上传文档
    
    Args:
        background_tasks: 后台任务管理器
        files: 文件列表
        auto_process: 是否自动处理
        db: 数据库会话
        
    Returns:
        上传结果列表
    """
    results = []
    
    for file in files:
        try:
            # 验证文件
            await validate_file_upload(file, MAX_FILE_SIZE, SUPPORTED_MIME_TYPES)
            
            # 生成文档ID
            document_id = uuid.uuid4()
            
            # 确定文档类型
            document_type = SUPPORTED_MIME_TYPES.get(file.content_type, "other")
            
            # 保存文件
            file_service = FileService()
            file_path = await file_service.save_uploaded_file(
                file, 
                document_id, 
                document_type
            )
            
            # 创建文档记录
            document_data = DocumentCreate(
                title=file.filename,
                file_name=file.filename,
                file_path=file_path,
                file_size=file.size,
                mime_type=file.content_type,
                document_type=document_type,
                auto_process=auto_process
            )
            
            # 保存到数据库
            document_service = DocumentService(db)
            document = await document_service.create_document(document_data, document_id)
            
            # 添加处理任务
            processing_started = False
            if auto_process:
                processing_service = ProcessingService()
                background_tasks.add_task(
                    processing_service.process_document,
                    document_id,
                    db
                )
                processing_started = True
            
            results.append(DocumentUploadResponse(
                document_id=document_id,
                file_path=file_path,
                status=DocumentStatus.UPLOADED,
                message="文档上传成功",
                processing_started=processing_started
            ))
            
        except Exception as e:
            results.append(DocumentUploadResponse(
                document_id=uuid.uuid4(),
                file_path="",
                status=DocumentStatus.FAILED,
                message=f"上传失败: {str(e)}",
                processing_started=False
            ))
    
    return results


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    categories: Optional[str] = Form(None),
    is_public: bool = Form(True),
    db: Session = Depends(get_db)
) -> DocumentUploadResponse:
    """
    上传文档
    
    支持的文件类型：PDF, TXT, MD, HTML, DOC, DOCX, XLS, XLSX, PPT, PPTX, 图片等
    """
    try:
        logger.info(f"开始上传文档: {file.filename}")
        
        # 初始化服务
        document_service = DocumentService(db)
        file_storage_service = FileStorageService()
        
        # 保存文件
        file_path, file_info = await file_storage_service.save_uploaded_file(file, "documents")
        
        # 解析标签和分类
        parsed_tags = []
        if tags:
            parsed_tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
        
        parsed_categories = []
        if categories:
            parsed_categories = [cat.strip() for cat in categories.split(',') if cat.strip()]
        
        # 创建文档记录
        document_data = DocumentCreate(
            title=title or file.filename,
            description=description,
            file_name=file.filename,
            file_path=file_path,
            file_size=file_info['size'],
            mime_type=file_info['mime_type'],
            author=author,
            tags=parsed_tags,
            categories=parsed_categories,
            is_public=is_public
        )
        
        document = await document_service.create_document(document_data)
        
        logger.info(f"文档上传成功: {document.id}")
        
        return DocumentUploadResponse(
            document_id=document.id,
            title=document.title,
            file_name=document.file_name,
            file_size=document.file_size,
            mime_type=document.mime_type,
            status=document.status,
            message="文档上传成功"
        )
        
    except (FileValidationError, FileStorageError) as e:
        logger.error(f"文件处理失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DocumentValidationError as e:
        logger.error(f"文档验证失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"上传文档失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="上传文档失败"
        )