#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 文件服务
================

本模块实现了文件操作的核心功能。

服务功能：
- 文件保存和管理
- 文件类型验证
- 文件路径处理
- 文件元数据提取

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import os
import uuid
import hashlib
import mimetypes
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from fastapi import UploadFile

from app.core.config import settings
from app.utils.exceptions import (
    FileStorageError,
    FileValidationError,
    FileNotFoundError
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class FileService:
    """
    文件服务类
    
    提供文件操作和管理的核心功能。
    """
    
    def __init__(self):
        """
        初始化文件服务
        """
        self.storage_path = Path(settings.STORAGE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info("文件服务初始化完成")
    
    async def save_uploaded_file(
        self, 
        file: UploadFile, 
        document_id: uuid.UUID,
        document_type: str = "document"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        保存上传的文件
        
        Args:
            file: 上传的文件
            document_id: 文档ID
            document_type: 文档类型
            
        Returns:
            文件路径和文件信息的元组
            
        Raises:
            FileValidationError: 文件验证失败
            FileStorageError: 文件存储失败
        """
        try:
            # 验证文件
            await self._validate_file(file)
            
            # 生成文件信息
            file_info = await self._generate_file_info(file)
            
            # 生成存储路径
            storage_path = self._generate_storage_path(
                file_info['filename'], 
                document_type,
                str(document_id)
            )
            
            # 保存文件
            await self._save_file_to_disk(file, storage_path)
            
            # 更新文件信息
            file_info.update({
                'storage_path': str(storage_path),
                'relative_path': str(storage_path.relative_to(self.storage_path)),
                'document_id': str(document_id),
                'document_type': document_type
            })
            
            logger.info(f"文件保存成功: {storage_path}")
            return str(storage_path), file_info
            
        except Exception as e:
            logger.error(f"保存文件失败: {str(e)}")
            raise FileStorageError(f"保存文件失败: {str(e)}")
    
    async def _validate_file(self, file: UploadFile) -> None:
        """
        验证文件
        
        Args:
            file: 上传的文件
            
        Raises:
            FileValidationError: 验证失败
        """
        # 检查文件名
        if not file.filename:
            raise FileValidationError("文件名不能为空")
        
        # 检查文件大小
        if hasattr(file, 'size') and file.size:
            if file.size > 100 * 1024 * 1024:  # 100MB
                raise FileValidationError(f"文件大小超过限制: {file.size}")
        
        # 检查文件类型
        allowed_types = {
            'application/pdf',
            'text/plain',
            'text/markdown',
            'text/html',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'image/jpeg',
            'image/png',
            'image/gif'
        }
        
        if file.content_type and file.content_type not in allowed_types:
            raise FileValidationError(f"不支持的文件类型: {file.content_type}")
    
    async def _generate_file_info(self, file: UploadFile) -> Dict[str, Any]:
        """
        生成文件信息
        
        Args:
            file: 上传的文件
            
        Returns:
            文件信息字典
        """
        # 读取文件内容计算哈希
        content = await file.read()
        await file.seek(0)  # 重置文件指针
        
        file_hash = hashlib.sha256(content).hexdigest()
        
        return {
            'filename': file.filename,
            'size': len(content),
            'mime_type': file.content_type,
            'hash': file_hash,
            'extension': Path(file.filename).suffix.lower(),
            'created_at': datetime.utcnow().isoformat()
        }
    
    def _generate_storage_path(
        self, 
        filename: str, 
        document_type: str, 
        document_id: str
    ) -> Path:
        """
        生成存储路径
        
        Args:
            filename: 文件名
            document_type: 文档类型
            document_id: 文档ID
            
        Returns:
            存储路径
        """
        # 使用文档ID的前两位作为子目录
        id_prefix = document_id[:2]
        
        # 生成唯一文件名
        name_parts = Path(filename).stem, document_id[:8], Path(filename).suffix
        unique_filename = '_'.join(filter(None, name_parts))
        
        return self.storage_path / document_type / id_prefix / unique_filename
    
    async def _save_file_to_disk(self, file: UploadFile, storage_path: Path) -> None:
        """
        保存文件到磁盘
        
        Args:
            file: 上传的文件
            storage_path: 存储路径
            
        Raises:
            FileStorageError: 保存失败
        """
        try:
            # 创建目录
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存文件
            content = await file.read()
            storage_path.write_bytes(content)
            
            # 重置文件指针
            await file.seek(0)
            
        except Exception as e:
            logger.error(f"保存文件到磁盘失败: {str(e)}")
            raise FileStorageError(f"保存文件到磁盘失败: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态信息
        """
        try:
            # 检查存储路径
            if not self.storage_path.exists():
                return {
                    "status": "unhealthy",
                    "message": "存储路径不存在",
                    "storage_path": str(self.storage_path)
                }
            
            # 检查写入权限
            test_file = self.storage_path / "health_check.tmp"
            try:
                test_file.write_text("health check")
                test_file.unlink()
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "message": f"存储路径无写入权限: {str(e)}",
                    "storage_path": str(self.storage_path)
                }
            
            return {
                "status": "healthy",
                "message": "文件服务运行正常",
                "storage_path": str(self.storage_path)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"健康检查失败: {str(e)}"
            }