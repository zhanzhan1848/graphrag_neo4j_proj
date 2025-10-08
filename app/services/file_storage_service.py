#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 文件存储服务
==================

本模块实现了文件存储的核心功能。

服务功能：
- 文件上传和存储
- 文件下载和访问
- 文件删除和清理
- 文件类型验证
- 文件大小限制
- 文件路径管理

存储策略：
- 本地文件系统存储
- 文件分类存储
- 文件去重处理
- 文件安全检查

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import os
import hashlib
import shutil
import mimetypes
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, BinaryIO
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
settings = settings


class FileStorageService:
    """
    文件存储服务类
    
    提供文件存储和管理的核心功能。
    """
    
    # 支持的文件类型
    SUPPORTED_MIME_TYPES = {
        'application/pdf': '.pdf',
        'text/plain': '.txt',
        'text/markdown': '.md',
        'text/html': '.html',
        'application/msword': '.doc',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'application/vnd.ms-excel': '.xls',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
        'application/vnd.ms-powerpoint': '.ppt',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
        'image/jpeg': '.jpg',
        'image/png': '.png',
        'image/gif': '.gif',
        'image/bmp': '.bmp',
        'image/tiff': '.tiff',
        'image/webp': '.webp',
        'application/json': '.json',
        'application/xml': '.xml',
        'text/csv': '.csv',
        'application/rtf': '.rtf'
    }
    
    # 最大文件大小 (100MB)
    MAX_FILE_SIZE = 100 * 1024 * 1024
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        初始化文件存储服务
        
        Args:
            storage_path: 存储路径（可选）
        """
        self.storage_path = Path(storage_path or settings.STORAGE_PATH)
        self._ensure_storage_directories()
    
    def _ensure_storage_directories(self) -> None:
        """
        确保存储目录存在
        """
        try:
            # 创建主存储目录
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # 创建子目录
            subdirs = ['documents', 'images', 'temp', 'processed', 'backups']
            for subdir in subdirs:
                (self.storage_path / subdir).mkdir(exist_ok=True)
            
            logger.info(f"存储目录初始化完成: {self.storage_path}")
            
        except Exception as e:
            logger.error(f"创建存储目录失败: {str(e)}")
            raise FileStorageError(f"创建存储目录失败: {str(e)}")
    
    async def save_uploaded_file(
        self, 
        file: UploadFile, 
        category: str = "documents"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        保存上传的文件
        
        Args:
            file: 上传的文件
            category: 文件分类
            
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
                category,
                file_info['hash']
            )
            
            # 检查文件是否已存在（去重）
            if storage_path.exists():
                logger.info(f"文件已存在，跳过存储: {storage_path}")
                return str(storage_path), file_info
            
            # 保存文件
            await self._save_file_to_disk(file, storage_path)
            
            # 更新文件信息
            file_info.update({
                'storage_path': str(storage_path),
                'relative_path': str(storage_path.relative_to(self.storage_path)),
                'category': category,
                'saved_at': datetime.utcnow().isoformat()
            })
            
            logger.info(f"文件保存成功: {storage_path}")
            return str(storage_path), file_info
            
        except (FileValidationError, FileStorageError):
            raise
        except Exception as e:
            logger.error(f"保存文件失败: {str(e)}")
            raise FileStorageError(f"保存文件失败: {str(e)}")
    
    async def save_file_content(
        self,
        content: bytes,
        filename: str,
        mime_type: str,
        category: str = "documents"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        保存文件内容
        
        Args:
            content: 文件内容
            filename: 文件名
            mime_type: MIME类型
            category: 文件分类
            
        Returns:
            文件路径和文件信息的元组
        """
        try:
            # 验证文件类型
            if mime_type not in self.SUPPORTED_MIME_TYPES:
                raise FileValidationError(f"不支持的文件类型: {mime_type}")
            
            # 验证文件大小
            if len(content) > self.MAX_FILE_SIZE:
                raise FileValidationError(f"文件大小超过限制: {len(content)} > {self.MAX_FILE_SIZE}")
            
            # 生成文件信息
            file_hash = hashlib.sha256(content).hexdigest()
            file_info = {
                'filename': filename,
                'size': len(content),
                'mime_type': mime_type,
                'hash': file_hash,
                'extension': self.SUPPORTED_MIME_TYPES.get(mime_type, '')
            }
            
            # 生成存储路径
            storage_path = self._generate_storage_path(filename, category, file_hash)
            
            # 检查文件是否已存在
            if storage_path.exists():
                logger.info(f"文件已存在，跳过存储: {storage_path}")
                return str(storage_path), file_info
            
            # 保存文件
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            storage_path.write_bytes(content)
            
            # 更新文件信息
            file_info.update({
                'storage_path': str(storage_path),
                'relative_path': str(storage_path.relative_to(self.storage_path)),
                'category': category,
                'saved_at': datetime.utcnow().isoformat()
            })
            
            logger.info(f"文件内容保存成功: {storage_path}")
            return str(storage_path), file_info
            
        except (FileValidationError, FileStorageError):
            raise
        except Exception as e:
            logger.error(f"保存文件内容失败: {str(e)}")
            raise FileStorageError(f"保存文件内容失败: {str(e)}")
    
    async def get_file_content(self, file_path: str) -> bytes:
        """
        获取文件内容
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件内容
            
        Raises:
            FileNotFoundError: 文件不存在
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            return path.read_bytes()
            
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"读取文件失败: {str(e)}")
            raise FileStorageError(f"读取文件失败: {str(e)}")
    
    async def delete_file(self, file_path: str) -> bool:
        """
        删除文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否删除成功
        """
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                logger.info(f"文件删除成功: {file_path}")
                return True
            else:
                logger.warning(f"文件不存在: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"删除文件失败: {str(e)}")
            return False
    
    async def move_file(self, source_path: str, target_path: str) -> bool:
        """
        移动文件
        
        Args:
            source_path: 源路径
            target_path: 目标路径
            
        Returns:
            是否移动成功
        """
        try:
            source = Path(source_path)
            target = Path(target_path)
            
            if not source.exists():
                raise FileNotFoundError(f"源文件不存在: {source_path}")
            
            # 确保目标目录存在
            target.parent.mkdir(parents=True, exist_ok=True)
            
            # 移动文件
            shutil.move(str(source), str(target))
            
            logger.info(f"文件移动成功: {source_path} -> {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"移动文件失败: {str(e)}")
            return False
    
    async def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        获取文件信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件信息字典或None
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return None
            
            stat = path.stat()
            mime_type, _ = mimetypes.guess_type(str(path))
            
            # 计算文件哈希
            file_hash = None
            if stat.st_size < 50 * 1024 * 1024:  # 只为小于50MB的文件计算哈希
                content = path.read_bytes()
                file_hash = hashlib.sha256(content).hexdigest()
            
            return {
                'filename': path.name,
                'size': stat.st_size,
                'mime_type': mime_type,
                'extension': path.suffix,
                'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'hash': file_hash,
                'storage_path': str(path),
                'relative_path': str(path.relative_to(self.storage_path)) if path.is_relative_to(self.storage_path) else None
            }
            
        except Exception as e:
            logger.error(f"获取文件信息失败: {str(e)}")
            return None
    
    async def list_files(
        self, 
        category: Optional[str] = None,
        pattern: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        列出文件
        
        Args:
            category: 文件分类
            pattern: 文件名模式
            
        Returns:
            文件信息列表
        """
        try:
            files = []
            search_path = self.storage_path
            
            if category:
                search_path = search_path / category
                if not search_path.exists():
                    return []
            
            # 搜索文件
            if pattern:
                file_paths = search_path.rglob(pattern)
            else:
                file_paths = search_path.rglob('*')
            
            for file_path in file_paths:
                if file_path.is_file():
                    file_info = await self.get_file_info(str(file_path))
                    if file_info:
                        files.append(file_info)
            
            return files
            
        except Exception as e:
            logger.error(f"列出文件失败: {str(e)}")
            return []
    
    async def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """
        清理临时文件
        
        Args:
            max_age_hours: 最大保留时间（小时）
            
        Returns:
            清理的文件数量
        """
        try:
            temp_path = self.storage_path / 'temp'
            if not temp_path.exists():
                return 0
            
            cleaned_count = 0
            cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
            
            for file_path in temp_path.rglob('*'):
                if file_path.is_file():
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        cleaned_count += 1
            
            logger.info(f"清理临时文件完成，删除 {cleaned_count} 个文件")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"清理临时文件失败: {str(e)}")
            return 0
    
    async def _validate_file(self, file: UploadFile) -> None:
        """
        验证上传的文件
        
        Args:
            file: 上传的文件
            
        Raises:
            FileValidationError: 验证失败
        """
        # 检查文件名
        if not file.filename:
            raise FileValidationError("文件名不能为空")
        
        # 检查文件类型
        if file.content_type not in self.SUPPORTED_MIME_TYPES:
            raise FileValidationError(f"不支持的文件类型: {file.content_type}")
        
        # 检查文件大小
        if hasattr(file, 'size') and file.size:
            if file.size > self.MAX_FILE_SIZE:
                raise FileValidationError(f"文件大小超过限制: {file.size} > {self.MAX_FILE_SIZE}")
    
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
            'extension': self.SUPPORTED_MIME_TYPES.get(file.content_type, ''),
            'uploaded_at': datetime.utcnow().isoformat()
        }
    
    def _generate_storage_path(
        self, 
        filename: str, 
        category: str, 
        file_hash: str
    ) -> Path:
        """
        生成存储路径
        
        Args:
            filename: 文件名
            category: 分类
            file_hash: 文件哈希
            
        Returns:
            存储路径
        """
        # 使用哈希的前两位作为子目录，避免单个目录文件过多
        hash_prefix = file_hash[:2]
        hash_suffix = file_hash[2:8]
        
        # 生成唯一文件名
        name_parts = Path(filename).stem, hash_suffix, Path(filename).suffix
        unique_filename = ''.join(name_parts)
        
        return self.storage_path / category / hash_prefix / unique_filename
    
    async def _save_file_to_disk(self, file: UploadFile, storage_path: Path) -> None:
        """
        保存文件到磁盘
        
        Args:
            file: 上传的文件
            storage_path: 存储路径
        """
        try:
            # 确保目录存在
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存文件
            with storage_path.open('wb') as f:
                content = await file.read()
                f.write(content)
            
            # 重置文件指针
            await file.seek(0)
            
        except Exception as e:
            logger.error(f"保存文件到磁盘失败: {str(e)}")
            raise FileStorageError(f"保存文件到磁盘失败: {str(e)}")