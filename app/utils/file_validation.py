#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 文件验证工具
==================

本模块提供文件上传验证相关的工具函数。

功能：
- 文件类型验证
- 文件大小验证
- 文件内容安全检查
- 文件名验证

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import os
import magic
from typing import Dict, Set
from fastapi import UploadFile, HTTPException, status
from pathlib import Path

from app.utils.exceptions import FileValidationError
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def validate_file_upload(
    file: UploadFile, 
    max_file_size: int, 
    supported_mime_types: Dict[str, str]
) -> None:
    """
    验证上传的文件
    
    Args:
        file: 上传的文件对象
        max_file_size: 最大文件大小（字节）
        supported_mime_types: 支持的MIME类型字典
        
    Raises:
        FileValidationError: 文件验证失败
    """
    try:
        logger.debug(f"开始验证文件: {file.filename}")
        
        # 验证文件名
        if not file.filename:
            raise FileValidationError("文件名不能为空")
        
        # 验证文件名长度
        if len(file.filename) > 255:
            raise FileValidationError("文件名过长，不能超过255个字符")
        
        # 验证文件名字符
        invalid_chars = set('<>:"/\\|?*')
        if any(char in file.filename for char in invalid_chars):
            raise FileValidationError("文件名包含非法字符")
        
        # 验证文件扩展名
        file_ext = Path(file.filename).suffix.lower()
        if not file_ext:
            raise FileValidationError("文件必须有扩展名")
        
        # 验证MIME类型
        if file.content_type not in supported_mime_types:
            raise FileValidationError(
                f"不支持的文件类型: {file.content_type}。"
                f"支持的类型: {', '.join(supported_mime_types.keys())}"
            )
        
        # 验证文件大小
        if hasattr(file, 'size') and file.size is not None:
            if file.size > max_file_size:
                size_mb = file.size / (1024 * 1024)
                max_size_mb = max_file_size / (1024 * 1024)
                raise FileValidationError(
                    f"文件大小超过限制: {size_mb:.2f}MB > {max_size_mb:.2f}MB"
                )
            
            if file.size == 0:
                raise FileValidationError("文件不能为空")
        
        # 读取文件头部进行内容验证
        file_content = await file.read(1024)  # 读取前1KB
        await file.seek(0)  # 重置文件指针
        
        # 验证文件内容是否与声明的MIME类型匹配
        if file_content:
            try:
                detected_mime = magic.from_buffer(file_content, mime=True)
                # 对于某些文件类型，检测结果可能不完全匹配，进行宽松验证
                if not _is_mime_type_compatible(file.content_type, detected_mime):
                    logger.warning(
                        f"文件MIME类型不匹配: 声明={file.content_type}, 检测={detected_mime}"
                    )
                    # 对于安全考虑，可以选择拒绝或警告
                    # raise FileValidationError(f"文件内容与声明的类型不匹配")
            except Exception as e:
                logger.warning(f"MIME类型检测失败: {str(e)}")
        
        logger.debug(f"文件验证通过: {file.filename}")
        
    except FileValidationError:
        raise
    except Exception as e:
        logger.error(f"文件验证过程中发生错误: {str(e)}")
        raise FileValidationError(f"文件验证失败: {str(e)}")


def _is_mime_type_compatible(declared_mime: str, detected_mime: str) -> bool:
    """
    检查声明的MIME类型与检测到的MIME类型是否兼容
    
    Args:
        declared_mime: 声明的MIME类型
        detected_mime: 检测到的MIME类型
        
    Returns:
        bool: 是否兼容
    """
    # 完全匹配
    if declared_mime == detected_mime:
        return True
    
    # 定义兼容的MIME类型映射
    compatible_types = {
        'text/plain': ['text/plain', 'application/octet-stream'],
        'text/markdown': ['text/plain', 'text/markdown', 'application/octet-stream'],
        'text/html': ['text/html', 'text/plain', 'application/octet-stream'],
        'application/pdf': ['application/pdf'],
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': [
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/zip'
        ],
        'application/msword': ['application/msword', 'application/octet-stream'],
        'image/jpeg': ['image/jpeg'],
        'image/png': ['image/png'],
        'image/gif': ['image/gif'],
        'image/webp': ['image/webp']
    }
    
    # 检查兼容性
    compatible_list = compatible_types.get(declared_mime, [declared_mime])
    return detected_mime in compatible_list


def validate_file_name(filename: str) -> bool:
    """
    验证文件名是否合法
    
    Args:
        filename: 文件名
        
    Returns:
        bool: 是否合法
    """
    if not filename or len(filename) > 255:
        return False
    
    # 检查非法字符
    invalid_chars = set('<>:"/\\|?*')
    if any(char in filename for char in invalid_chars):
        return False
    
    # 检查是否有扩展名
    if not Path(filename).suffix:
        return False
    
    return True


def get_file_type_from_extension(filename: str) -> str:
    """
    根据文件扩展名获取文件类型
    
    Args:
        filename: 文件名
        
    Returns:
        str: 文件类型
    """
    ext_to_type = {
        '.pdf': 'pdf',
        '.txt': 'text',
        '.md': 'markdown',
        '.html': 'html',
        '.htm': 'html',
        '.docx': 'docx',
        '.doc': 'doc',
        '.rtf': 'rtf',
        '.odt': 'odt',
        '.epub': 'epub',
        '.png': 'image',
        '.jpg': 'image',
        '.jpeg': 'image',
        '.gif': 'image',
        '.webp': 'image'
    }
    
    ext = Path(filename).suffix.lower()
    return ext_to_type.get(ext, 'other')