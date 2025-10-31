#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 文档数据库模型单元测试
=============================

测试 Document 模型的功能：
1. 基本字段和属性
2. 文档状态管理
3. 处理进度跟踪
4. 关系映射
5. 属性方法
6. 业务逻辑方法

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import pytest
import uuid
from datetime import datetime
from sqlalchemy.orm import Session

from app.models.database.documents import Document


class TestDocumentModel:
    """Document 模型基础功能测试"""
    
    def test_document_creation(self):
        """测试文档创建"""
        doc_data = {
            "title": "测试文档",
            "description": "这是一个测试文档",
            "content": "文档内容",
            "file_path": "/test/path/document.pdf",
            "file_type": "pdf",
            "file_size": 1024,
            "hash_value": "abc123def456",
            "language": "zh",
            "author": "测试作者"
        }
        
        document = Document(**doc_data)
        
        # 验证基本字段
        assert document.title == "测试文档"
        assert document.description == "这是一个测试文档"
        assert document.content == "文档内容"
        assert document.file_path == "/test/path/document.pdf"
        assert document.file_type == "pdf"
        assert document.file_size == 1024
        assert document.hash_value == "abc123def456"
        assert document.language == "zh"
        assert document.author == "测试作者"
    
    def test_document_default_values(self):
        """测试文档默认值"""
        document = Document(
            title="最小文档",
            file_path="/test/minimal.txt",
            file_type="txt"
        )
        
        # 验证默认值
        assert document.status == "pending"
        assert document.processing_progress == 0.0
        assert document.language == "zh"
        assert document.chunk_count == 0
        assert document.entity_count == 0
        assert document.relation_count == 0
        assert document.image_count == 0
        assert document.is_public is False
        assert document.keywords == [] or document.keywords is None
    
    def test_document_status_properties(self):
        """测试文档状态属性"""
        document = Document(
            title="状态测试",
            file_path="/test/status.txt",
            file_type="txt"
        )
        
        # 测试初始状态
        assert document.status == "pending"
        assert not document.is_processed
        assert not document.is_processing
        assert not document.has_error
        
        # 测试处理中状态
        document.status = "processing"
        assert not document.is_processed
        assert document.is_processing
        assert not document.has_error
        
        # 测试完成状态
        document.status = "completed"
        assert document.is_processed
        assert not document.is_processing
        assert not document.has_error
        
        # 测试失败状态
        document.status = "failed"
        document.error_message = "处理失败"
        assert not document.is_processed
        assert not document.is_processing
        assert document.has_error
    
    def test_update_processing_progress(self):
        """测试更新处理进度"""
        document = Document(
            title="进度测试",
            file_path="/test/progress.txt",
            file_type="txt"
        )
        
        # 更新进度
        document.update_processing_progress(0.5)
        assert document.processing_progress == 0.5
        assert document.status == "pending"  # 状态不变
        
        # 更新进度和状态
        document.update_processing_progress(0.8, "processing")
        assert document.processing_progress == 0.8
        assert document.status == "processing"
        
        # 测试进度边界值
        document.update_processing_progress(0.0)
        assert document.processing_progress == 0.0
        
        document.update_processing_progress(1.0)
        assert document.processing_progress == 1.0
    
    def test_mark_as_completed(self):
        """测试标记为完成"""
        document = Document(
            title="完成测试",
            file_path="/test/completed.txt",
            file_type="txt",
            status="processing",
            processing_progress=0.8
        )
        
        document.mark_as_completed()
        
        assert document.status == "completed"
        assert document.processing_progress == 1.0
        assert document.error_message is None
    
    def test_mark_as_failed(self):
        """测试标记为失败"""
        document = Document(
            title="失败测试",
            file_path="/test/failed.txt",
            file_type="txt",
            status="processing"
        )
        
        error_msg = "文件格式不支持"
        document.mark_as_failed(error_msg)
        
        assert document.status == "failed"
        assert document.error_message == error_msg
    
    def test_document_repr(self):
        """测试文档字符串表示"""
        document = Document(
            title="表示测试",
            file_path="/test/repr.txt",
            file_type="txt"
        )
        
        repr_str = repr(document)
        assert "Document" in repr_str
        assert "表示测试" in repr_str
    
    def test_document_to_dict(self):
        """测试文档转换为字典"""
        doc_data = {
            "title": "字典测试",
            "description": "测试转换",
            "file_path": "/test/dict.txt",
            "file_type": "txt",
            "file_size": 512,
            "language": "zh",
            "status": "completed",
            "keywords": ["测试", "字典"]
        }
        
        document = Document(**doc_data)
        result = document.to_dict()
        
        # 验证关键字段
        assert result["title"] == "字典测试"
        assert result["description"] == "测试转换"
        assert result["file_type"] == "txt"
        assert result["status"] == "completed"
        assert result["keywords"] == ["测试", "字典"]
    
    def test_document_update_from_dict(self):
        """测试从字典更新文档"""
        document = Document(
            title="原标题",
            file_path="/test/update.txt",
            file_type="txt"
        )
        
        update_data = {
            "title": "新标题",
            "description": "新描述",
            "status": "processing",
            "processing_progress": 0.6
        }
        
        document.update_from_dict(update_data)
        
        assert document.title == "新标题"
        assert document.description == "新描述"
        assert document.status == "processing"
        assert document.processing_progress == 0.6


class TestDocumentValidation:
    """Document 模型验证测试"""
    
    def test_required_fields(self):
        """测试必需字段"""
        # 缺少必需字段应该能创建对象（验证在数据库层）
        document = Document()
        assert document is not None
        
        # 设置必需字段
        document.title = "必需字段测试"
        document.file_path = "/test/required.txt"
        document.file_type = "txt"
        
        assert document.title == "必需字段测试"
        assert document.file_path == "/test/required.txt"
        assert document.file_type == "txt"
    
    def test_file_type_values(self):
        """测试文件类型值"""
        valid_types = ["pdf", "txt", "md", "html", "docx", "doc", "rtf"]
        
        for file_type in valid_types:
            document = Document(
                title=f"测试{file_type}",
                file_path=f"/test/file.{file_type}",
                file_type=file_type
            )
            assert document.file_type == file_type
    
    def test_status_values(self):
        """测试状态值"""
        valid_statuses = ["pending", "processing", "completed", "failed"]
        
        document = Document(
            title="状态测试",
            file_path="/test/status.txt",
            file_type="txt"
        )
        
        for status in valid_statuses:
            document.status = status
            assert document.status == status
    
    def test_language_codes(self):
        """测试语言代码"""
        valid_languages = ["zh", "en", "ja", "ko", "fr", "de", "es"]
        
        for lang in valid_languages:
            document = Document(
                title="语言测试",
                file_path="/test/lang.txt",
                file_type="txt",
                language=lang
            )
            assert document.language == lang
    
    def test_progress_bounds(self):
        """测试进度边界"""
        document = Document(
            title="进度边界测试",
            file_path="/test/bounds.txt",
            file_type="txt"
        )
        
        # 测试有效进度值
        valid_progress = [0.0, 0.25, 0.5, 0.75, 1.0]
        for progress in valid_progress:
            document.update_processing_progress(progress)
            assert document.processing_progress == progress
    
    def test_file_size_validation(self):
        """测试文件大小"""
        document = Document(
            title="大小测试",
            file_path="/test/size.txt",
            file_type="txt"
        )
        
        # 测试各种文件大小
        sizes = [0, 1024, 1024*1024, 1024*1024*100]  # 0B, 1KB, 1MB, 100MB
        for size in sizes:
            document.file_size = size
            assert document.file_size == size


class TestDocumentBusinessLogic:
    """Document 模型业务逻辑测试"""
    
    def test_processing_workflow(self):
        """测试处理工作流"""
        document = Document(
            title="工作流测试",
            file_path="/test/workflow.txt",
            file_type="txt"
        )
        
        # 1. 初始状态
        assert document.status == "pending"
        assert document.processing_progress == 0.0
        
        # 2. 开始处理
        document.update_processing_progress(0.1, "processing")
        assert document.is_processing
        
        # 3. 处理进度更新
        document.update_processing_progress(0.5)
        assert document.processing_progress == 0.5
        
        # 4. 完成处理
        document.mark_as_completed()
        assert document.is_processed
        assert document.processing_progress == 1.0
    
    def test_error_handling_workflow(self):
        """测试错误处理工作流"""
        document = Document(
            title="错误处理测试",
            file_path="/test/error.txt",
            file_type="txt"
        )
        
        # 开始处理
        document.update_processing_progress(0.3, "processing")
        
        # 处理失败
        error_msg = "文件损坏无法解析"
        document.mark_as_failed(error_msg)
        
        assert document.has_error
        assert document.error_message == error_msg
        assert document.status == "failed"
    
    def test_metadata_management(self):
        """测试元数据管理"""
        document = Document(
            title="元数据测试",
            file_path="/test/metadata.txt",
            file_type="txt"
        )
        
        # 设置关键词
        keywords = ["知识图谱", "自然语言处理", "机器学习"]
        document.keywords = keywords
        assert document.keywords == keywords
        
        # 设置额外数据
        extra_data = {
            "source": "academic_paper",
            "conference": "AAAI 2024",
            "doi": "10.1000/test.doi"
        }
        document.extra_data = extra_data
        assert document.extra_data == extra_data
    
    def test_statistics_tracking(self):
        """测试统计信息跟踪"""
        document = Document(
            title="统计测试",
            file_path="/test/stats.txt",
            file_type="txt"
        )
        
        # 更新统计信息
        document.chunk_count = 10
        document.entity_count = 25
        document.relation_count = 15
        document.image_count = 3
        
        assert document.chunk_count == 10
        assert document.entity_count == 25
        assert document.relation_count == 15
        assert document.image_count == 3
    
    def test_quality_assessment(self):
        """测试质量评估"""
        document = Document(
            title="质量测试",
            file_path="/test/quality.txt",
            file_type="txt"
        )
        
        # 设置质量评分
        quality_scores = [0.0, 0.3, 0.7, 0.9, 1.0]
        for score in quality_scores:
            document.quality_score = score
            assert document.quality_score == score


@pytest.mark.unit
@pytest.mark.database
class TestDocumentDatabaseIntegration:
    """Document 模型数据库集成测试"""
    
    def test_document_persistence(self, test_db_session: Session):
        """测试文档持久化"""
        document = Document(
            title="持久化测试",
            file_path="/test/persist.txt",
            file_type="txt",
            hash_value="unique_hash_123"
        )
        
        # 保存到数据库
        test_db_session.add(document)
        test_db_session.commit()
        test_db_session.refresh(document)
        
        # 验证保存成功
        assert document.id is not None
        assert document.created_at is not None
        assert document.updated_at is not None
    
    def test_document_query(self, test_db_session: Session, sample_document: Document):
        """测试文档查询"""
        # 按标题查询
        found = test_db_session.query(Document).filter(
            Document.title == sample_document.title
        ).first()
        
        assert found is not None
        assert found.id == sample_document.id
        assert found.title == sample_document.title
    
    def test_document_update(self, test_db_session: Session, sample_document: Document):
        """测试文档更新"""
        original_title = sample_document.title
        new_title = "更新后的标题"
        
        # 更新标题
        sample_document.title = new_title
        test_db_session.commit()
        
        # 重新查询验证
        updated = test_db_session.query(Document).filter(
            Document.id == sample_document.id
        ).first()
        
        assert updated.title == new_title
        assert updated.title != original_title
    
    def test_document_deletion(self, test_db_session: Session):
        """测试文档删除"""
        document = Document(
            title="删除测试",
            file_path="/test/delete.txt",
            file_type="txt"
        )
        
        test_db_session.add(document)
        test_db_session.commit()
        doc_id = document.id
        
        # 删除文档
        test_db_session.delete(document)
        test_db_session.commit()
        
        # 验证删除成功
        deleted = test_db_session.query(Document).filter(
            Document.id == doc_id
        ).first()
        
        assert deleted is None