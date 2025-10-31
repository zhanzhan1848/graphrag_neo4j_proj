#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 文件服务单元测试
========================

测试 FileService 的功能：
1. 文件保存和管理
2. 文件类型验证
3. 文件路径处理
4. 文件元数据提取
5. 文件操作异常处理

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import pytest
import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from fastapi import UploadFile
from io import BytesIO

from app.services.file_service import FileService
from app.utils.exceptions import (
    FileStorageError,
    FileValidationError,
    FileNotFoundError
)


class TestFileService:
    """FileService 基础功能测试"""
    
    @pytest.fixture
    def temp_storage_path(self):
        """创建临时存储路径"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def file_service(self, temp_storage_path):
        """创建文件服务实例"""
        with patch('app.services.file_service.settings') as mock_settings:
            mock_settings.STORAGE_PATH = str(temp_storage_path)
            service = FileService()
            return service
    
    @pytest.fixture
    def sample_upload_file(self):
        """创建示例上传文件"""
        content = b"This is a test file content"
        file_obj = BytesIO(content)
        upload_file = UploadFile(
            filename="test.txt",
            file=file_obj,
            content_type="text/plain"
        )
        return upload_file
    
    def test_file_service_initialization(self, file_service, temp_storage_path):
        """测试文件服务初始化"""
        assert file_service.storage_path == temp_storage_path
        assert temp_storage_path.exists()
    
    def test_validate_file_type_valid(self, file_service):
        """测试有效文件类型验证"""
        valid_types = [
            ("test.txt", "text/plain"),
            ("document.pdf", "application/pdf"),
            ("image.jpg", "image/jpeg"),
            ("data.json", "application/json")
        ]
        
        for filename, content_type in valid_types:
            # 应该不抛出异常
            file_service._validate_file_type(filename, content_type)
    
    def test_validate_file_type_invalid(self, file_service):
        """测试无效文件类型验证"""
        invalid_types = [
            ("malware.exe", "application/x-executable"),
            ("script.bat", "application/x-bat"),
            ("unknown.xyz", "application/octet-stream")
        ]
        
        for filename, content_type in invalid_types:
            with pytest.raises(FileValidationError):
                file_service._validate_file_type(filename, content_type)
    
    def test_validate_file_size_valid(self, file_service):
        """测试有效文件大小验证"""
        valid_sizes = [1024, 1024*1024, 50*1024*1024]  # 1KB, 1MB, 50MB
        
        for size in valid_sizes:
            # 应该不抛出异常
            file_service._validate_file_size(size)
    
    def test_validate_file_size_invalid(self, file_service):
        """测试无效文件大小验证"""
        invalid_sizes = [0, -1, 200*1024*1024]  # 0字节, 负数, 200MB
        
        for size in invalid_sizes:
            with pytest.raises(FileValidationError):
                file_service._validate_file_size(size)
    
    def test_generate_file_path(self, file_service):
        """测试文件路径生成"""
        filename = "test.txt"
        file_path = file_service._generate_file_path(filename)
        
        assert isinstance(file_path, Path)
        assert file_path.suffix == ".txt"
        assert file_path.parent == file_service.storage_path
        assert len(file_path.stem) > len("test")  # 包含UUID
    
    def test_calculate_file_hash(self, file_service):
        """测试文件哈希计算"""
        content = b"test content for hashing"
        file_obj = BytesIO(content)
        
        hash_value = file_service._calculate_file_hash(file_obj)
        
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA256 hex length
        
        # 验证相同内容产生相同哈希
        file_obj2 = BytesIO(content)
        hash_value2 = file_service._calculate_file_hash(file_obj2)
        assert hash_value == hash_value2
    
    def test_extract_file_metadata(self, file_service, sample_upload_file):
        """测试文件元数据提取"""
        metadata = file_service._extract_file_metadata(sample_upload_file)
        
        assert "filename" in metadata
        assert "content_type" in metadata
        assert "size" in metadata
        assert "upload_time" in metadata
        
        assert metadata["filename"] == "test.txt"
        assert metadata["content_type"] == "text/plain"
        assert metadata["size"] > 0
    
    @patch('app.services.file_service.mimetypes.guess_type')
    def test_get_content_type(self, mock_guess_type, file_service):
        """测试内容类型获取"""
        # 测试已知类型
        mock_guess_type.return_value = ("text/plain", None)
        content_type = file_service._get_content_type("test.txt")
        assert content_type == "text/plain"
        
        # 测试未知类型
        mock_guess_type.return_value = (None, None)
        content_type = file_service._get_content_type("unknown.xyz")
        assert content_type == "application/octet-stream"
    
    async def test_save_file_success(self, file_service, sample_upload_file):
        """测试文件保存成功"""
        result = await file_service.save_file(sample_upload_file)
        
        assert "file_path" in result
        assert "file_hash" in result
        assert "metadata" in result
        
        # 验证文件实际保存
        file_path = Path(result["file_path"])
        assert file_path.exists()
        assert file_path.read_bytes() == b"This is a test file content"
    
    async def test_save_file_with_custom_filename(self, file_service, sample_upload_file):
        """测试使用自定义文件名保存"""
        custom_filename = "custom_name.txt"
        result = await file_service.save_file(sample_upload_file, custom_filename)
        
        file_path = Path(result["file_path"])
        assert custom_filename in file_path.name
    
    async def test_save_file_duplicate_handling(self, file_service, sample_upload_file):
        """测试重复文件处理"""
        # 保存第一次
        result1 = await file_service.save_file(sample_upload_file)
        
        # 重置文件指针
        sample_upload_file.file.seek(0)
        
        # 保存第二次（相同内容）
        result2 = await file_service.save_file(sample_upload_file)
        
        # 应该有不同的文件路径但相同的哈希
        assert result1["file_path"] != result2["file_path"]
        assert result1["file_hash"] == result2["file_hash"]
    
    async def test_get_file_info_existing(self, file_service, sample_upload_file):
        """测试获取存在文件的信息"""
        # 先保存文件
        save_result = await file_service.save_file(sample_upload_file)
        file_path = save_result["file_path"]
        
        # 获取文件信息
        file_info = await file_service.get_file_info(file_path)
        
        assert file_info is not None
        assert "path" in file_info
        assert "size" in file_info
        assert "created_time" in file_info
        assert "modified_time" in file_info
        assert "content_type" in file_info
    
    async def test_get_file_info_nonexistent(self, file_service):
        """测试获取不存在文件的信息"""
        nonexistent_path = "/path/to/nonexistent/file.txt"
        
        with pytest.raises(FileNotFoundError):
            await file_service.get_file_info(nonexistent_path)
    
    async def test_delete_file_existing(self, file_service, sample_upload_file):
        """测试删除存在的文件"""
        # 先保存文件
        save_result = await file_service.save_file(sample_upload_file)
        file_path = save_result["file_path"]
        
        # 验证文件存在
        assert Path(file_path).exists()
        
        # 删除文件
        success = await file_service.delete_file(file_path)
        
        assert success is True
        assert not Path(file_path).exists()
    
    async def test_delete_file_nonexistent(self, file_service):
        """测试删除不存在的文件"""
        nonexistent_path = "/path/to/nonexistent/file.txt"
        
        with pytest.raises(FileNotFoundError):
            await file_service.delete_file(nonexistent_path)
    
    async def test_file_exists(self, file_service, sample_upload_file):
        """测试文件存在性检查"""
        # 不存在的文件
        nonexistent_path = "/path/to/nonexistent/file.txt"
        assert not await file_service.file_exists(nonexistent_path)
        
        # 保存文件后检查
        save_result = await file_service.save_file(sample_upload_file)
        file_path = save_result["file_path"]
        assert await file_service.file_exists(file_path)


class TestFileServiceErrorHandling:
    """FileService 错误处理测试"""
    
    @pytest.fixture
    def file_service(self):
        """创建文件服务实例"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('app.services.file_service.settings') as mock_settings:
                mock_settings.STORAGE_PATH = temp_dir
                service = FileService()
                yield service
    
    async def test_save_file_invalid_type(self, file_service):
        """测试保存无效类型文件"""
        invalid_file = UploadFile(
            filename="malware.exe",
            file=BytesIO(b"malicious content"),
            content_type="application/x-executable"
        )
        
        with pytest.raises(FileValidationError):
            await file_service.save_file(invalid_file)
    
    async def test_save_file_too_large(self, file_service):
        """测试保存过大文件"""
        # 创建一个模拟的大文件
        large_content = b"x" * (200 * 1024 * 1024)  # 200MB
        large_file = UploadFile(
            filename="large.txt",
            file=BytesIO(large_content),
            content_type="text/plain"
        )
        
        with pytest.raises(FileValidationError):
            await file_service.save_file(large_file)
    
    async def test_save_file_empty(self, file_service):
        """测试保存空文件"""
        empty_file = UploadFile(
            filename="empty.txt",
            file=BytesIO(b""),
            content_type="text/plain"
        )
        
        with pytest.raises(FileValidationError):
            await file_service.save_file(empty_file)
    
    @patch('pathlib.Path.write_bytes')
    async def test_save_file_storage_error(self, mock_write_bytes, file_service):
        """测试文件存储错误"""
        mock_write_bytes.side_effect = OSError("Disk full")
        
        sample_file = UploadFile(
            filename="test.txt",
            file=BytesIO(b"test content"),
            content_type="text/plain"
        )
        
        with pytest.raises(FileStorageError):
            await file_service.save_file(sample_file)
    
    @patch('pathlib.Path.stat')
    async def test_get_file_info_permission_error(self, mock_stat, file_service):
        """测试文件信息获取权限错误"""
        mock_stat.side_effect = PermissionError("Access denied")
        
        with pytest.raises(FileStorageError):
            await file_service.get_file_info("/some/path/file.txt")


class TestFileServiceIntegration:
    """FileService 集成测试"""
    
    @pytest.fixture
    def file_service(self):
        """创建文件服务实例"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('app.services.file_service.settings') as mock_settings:
                mock_settings.STORAGE_PATH = temp_dir
                service = FileService()
                yield service
    
    async def test_complete_file_lifecycle(self, file_service):
        """测试完整的文件生命周期"""
        # 1. 创建测试文件
        test_content = b"Complete lifecycle test content"
        upload_file = UploadFile(
            filename="lifecycle_test.txt",
            file=BytesIO(test_content),
            content_type="text/plain"
        )
        
        # 2. 保存文件
        save_result = await file_service.save_file(upload_file)
        file_path = save_result["file_path"]
        
        # 3. 验证文件存在
        assert await file_service.file_exists(file_path)
        
        # 4. 获取文件信息
        file_info = await file_service.get_file_info(file_path)
        assert file_info["size"] == len(test_content)
        
        # 5. 删除文件
        success = await file_service.delete_file(file_path)
        assert success is True
        
        # 6. 验证文件不存在
        assert not await file_service.file_exists(file_path)
    
    async def test_multiple_files_handling(self, file_service):
        """测试多文件处理"""
        files_data = [
            ("file1.txt", b"Content of file 1", "text/plain"),
            ("file2.json", b'{"key": "value"}', "application/json"),
            ("file3.md", b"# Markdown Content", "text/markdown")
        ]
        
        saved_files = []
        
        # 保存多个文件
        for filename, content, content_type in files_data:
            upload_file = UploadFile(
                filename=filename,
                file=BytesIO(content),
                content_type=content_type
            )
            result = await file_service.save_file(upload_file)
            saved_files.append(result["file_path"])
        
        # 验证所有文件都存在
        for file_path in saved_files:
            assert await file_service.file_exists(file_path)
        
        # 清理所有文件
        for file_path in saved_files:
            await file_service.delete_file(file_path)
    
    async def test_concurrent_file_operations(self, file_service):
        """测试并发文件操作"""
        import asyncio
        
        async def save_file_task(index):
            content = f"Concurrent test content {index}".encode()
            upload_file = UploadFile(
                filename=f"concurrent_{index}.txt",
                file=BytesIO(content),
                content_type="text/plain"
            )
            return await file_service.save_file(upload_file)
        
        # 并发保存多个文件
        tasks = [save_file_task(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # 验证所有文件都保存成功
        assert len(results) == 5
        for result in results:
            assert "file_path" in result
            assert await file_service.file_exists(result["file_path"])
        
        # 清理文件
        for result in results:
            await file_service.delete_file(result["file_path"])