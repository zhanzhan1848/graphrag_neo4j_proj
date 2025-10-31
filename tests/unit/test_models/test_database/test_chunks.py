#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 文本块数据库模型单元测试
===============================

测试 Chunk 模型的功能：
1. 基本字段和属性
2. 向量嵌入管理
3. 位置信息处理
4. 文本统计功能
5. 质量评分
6. 关系映射

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import pytest
import uuid
from sqlalchemy.orm import Session

from app.models.database.chunks import Chunk
from app.models.database.documents import Document


class TestChunkModel:
    """Chunk 模型基础功能测试"""
    
    def test_chunk_creation(self):
        """测试文本块创建"""
        doc_id = uuid.uuid4()
        chunk_data = {
            "document_id": doc_id,
            "content": "这是一个测试文本块的内容。",
            "chunk_index": 0,
            "start_pos": 0,
            "end_pos": 15,
            "token_count": 10,
            "char_count": 15,
            "word_count": 8
        }
        
        chunk = Chunk(**chunk_data)
        
        # 验证基本字段
        assert chunk.document_id == doc_id
        assert chunk.content == "这是一个测试文本块的内容。"
        assert chunk.chunk_index == 0
        assert chunk.start_pos == 0
        assert chunk.end_pos == 15
        assert chunk.token_count == 10
        assert chunk.char_count == 15
        assert chunk.word_count == 8
    
    def test_chunk_default_values(self):
        """测试文本块默认值"""
        doc_id = uuid.uuid4()
        chunk = Chunk(
            document_id=doc_id,
            content="最小文本块",
            chunk_index=0
        )
        
        # 验证默认值
        assert chunk.status == "pending"
        assert chunk.chunk_type == "text"
        assert chunk.embedding_model == "text-embedding-ada-002"
        assert chunk.tags == [] or chunk.tags is None
        assert chunk.categories == [] or chunk.categories is None
    
    def test_content_preview_property(self):
        """测试内容预览属性"""
        doc_id = uuid.uuid4()
        
        # 短内容测试
        short_content = "短内容"
        chunk_short = Chunk(
            document_id=doc_id,
            content=short_content,
            chunk_index=0
        )
        assert chunk_short.content_preview == short_content
        
        # 长内容测试
        long_content = "这是一个很长的文本内容，" * 10  # 超过100个字符
        chunk_long = Chunk(
            document_id=doc_id,
            content=long_content,
            chunk_index=1
        )
        preview = chunk_long.content_preview
        assert len(preview) <= 103  # 100 + "..."
        assert preview.endswith("...")
    
    def test_is_embedded_property(self):
        """测试嵌入状态属性"""
        doc_id = uuid.uuid4()
        chunk = Chunk(
            document_id=doc_id,
            content="测试嵌入",
            chunk_index=0
        )
        
        # 初始状态没有嵌入
        assert not chunk.is_embedded
        
        # 设置嵌入后
        embedding = [0.1] * 1536
        chunk.embedding = embedding
        assert chunk.is_embedded
    
    def test_is_processed_property(self):
        """测试处理状态属性"""
        doc_id = uuid.uuid4()
        chunk = Chunk(
            document_id=doc_id,
            content="测试处理状态",
            chunk_index=0
        )
        
        # 初始状态
        assert chunk.status == "pending"
        assert not chunk.is_processed
        
        # 完成状态
        chunk.status = "completed"
        assert chunk.is_processed
    
    def test_update_embedding_method(self):
        """测试更新嵌入方法"""
        doc_id = uuid.uuid4()
        chunk = Chunk(
            document_id=doc_id,
            content="测试嵌入更新",
            chunk_index=0
        )
        
        # 更新嵌入
        embedding = [0.1, 0.2, 0.3] * 512  # 1536维
        model_name = "text-embedding-3-small"
        
        chunk.update_embedding(embedding, model_name)
        
        assert chunk.embedding == embedding
        assert chunk.embedding_model == model_name
        assert chunk.status == "completed"
    
    def test_update_embedding_without_model(self):
        """测试不指定模型名称的嵌入更新"""
        doc_id = uuid.uuid4()
        chunk = Chunk(
            document_id=doc_id,
            content="测试嵌入更新",
            chunk_index=0
        )
        
        original_model = chunk.embedding_model
        embedding = [0.1] * 1536
        
        chunk.update_embedding(embedding)
        
        assert chunk.embedding == embedding
        assert chunk.embedding_model == original_model  # 保持原有模型名
        assert chunk.status == "completed"
    
    def test_chunk_repr(self):
        """测试文本块字符串表示"""
        doc_id = uuid.uuid4()
        chunk = Chunk(
            document_id=doc_id,
            content="测试表示",
            chunk_index=5
        )
        
        repr_str = repr(chunk)
        assert "Chunk" in repr_str
        assert "index=5" in repr_str
    
    def test_chunk_to_dict(self):
        """测试文本块转换为字典"""
        doc_id = uuid.uuid4()
        chunk_data = {
            "document_id": doc_id,
            "content": "字典测试内容",
            "chunk_index": 2,
            "token_count": 15,
            "chunk_type": "header",
            "tags": ["重要", "标题"]
        }
        
        chunk = Chunk(**chunk_data)
        result = chunk.to_dict()
        
        # 验证关键字段
        assert result["content"] == "字典测试内容"
        assert result["chunk_index"] == 2
        assert result["token_count"] == 15
        assert result["chunk_type"] == "header"
        assert result["tags"] == ["重要", "标题"]


class TestChunkValidation:
    """Chunk 模型验证测试"""
    
    def test_required_fields(self):
        """测试必需字段"""
        doc_id = uuid.uuid4()
        
        # 最小必需字段
        chunk = Chunk(
            document_id=doc_id,
            content="必需字段测试",
            chunk_index=0
        )
        
        assert chunk.document_id == doc_id
        assert chunk.content == "必需字段测试"
        assert chunk.chunk_index == 0
    
    def test_chunk_types(self):
        """测试文本块类型"""
        doc_id = uuid.uuid4()
        valid_types = ["text", "title", "header", "table", "list", "code", "quote"]
        
        for chunk_type in valid_types:
            chunk = Chunk(
                document_id=doc_id,
                content=f"测试{chunk_type}类型",
                chunk_index=0,
                chunk_type=chunk_type
            )
            assert chunk.chunk_type == chunk_type
    
    def test_status_values(self):
        """测试状态值"""
        doc_id = uuid.uuid4()
        valid_statuses = ["pending", "processing", "completed", "failed"]
        
        chunk = Chunk(
            document_id=doc_id,
            content="状态测试",
            chunk_index=0
        )
        
        for status in valid_statuses:
            chunk.status = status
            assert chunk.status == status
    
    def test_position_validation(self):
        """测试位置信息验证"""
        doc_id = uuid.uuid4()
        chunk = Chunk(
            document_id=doc_id,
            content="位置测试内容",
            chunk_index=0,
            start_pos=10,
            end_pos=20
        )
        
        assert chunk.start_pos == 10
        assert chunk.end_pos == 20
        # 在实际应用中，应该验证 end_pos > start_pos
    
    def test_embedding_dimensions(self):
        """测试嵌入维度"""
        doc_id = uuid.uuid4()
        chunk = Chunk(
            document_id=doc_id,
            content="嵌入维度测试",
            chunk_index=0
        )
        
        # 测试正确维度
        correct_embedding = [0.1] * 1536
        chunk.embedding = correct_embedding
        assert len(chunk.embedding) == 1536
    
    def test_quality_scores(self):
        """测试质量评分"""
        doc_id = uuid.uuid4()
        chunk = Chunk(
            document_id=doc_id,
            content="质量评分测试",
            chunk_index=0
        )
        
        # 测试有效评分范围
        valid_scores = [0.0, 0.25, 0.5, 0.75, 1.0]
        for score in valid_scores:
            chunk.quality_score = score
            assert chunk.quality_score == score
            
            chunk.coherence_score = score
            assert chunk.coherence_score == score


class TestChunkBusinessLogic:
    """Chunk 模型业务逻辑测试"""
    
    def test_text_statistics_calculation(self):
        """测试文本统计计算"""
        doc_id = uuid.uuid4()
        content = "这是一个测试文本，包含中文和English words。"
        
        chunk = Chunk(
            document_id=doc_id,
            content=content,
            chunk_index=0,
            char_count=len(content),
            word_count=len(content.split()),
            token_count=25  # 估算的token数量
        )
        
        assert chunk.char_count == len(content)
        assert chunk.word_count > 0
        assert chunk.token_count > 0
    
    def test_embedding_workflow(self):
        """测试嵌入工作流"""
        doc_id = uuid.uuid4()
        chunk = Chunk(
            document_id=doc_id,
            content="嵌入工作流测试",
            chunk_index=0
        )
        
        # 1. 初始状态
        assert not chunk.is_embedded
        assert chunk.status == "pending"
        
        # 2. 生成嵌入
        embedding = [0.1] * 1536
        chunk.update_embedding(embedding, "text-embedding-3-small")
        
        # 3. 验证结果
        assert chunk.is_embedded
        assert chunk.is_processed
        assert chunk.embedding_model == "text-embedding-3-small"
    
    def test_categorization_and_tagging(self):
        """测试分类和标签"""
        doc_id = uuid.uuid4()
        chunk = Chunk(
            document_id=doc_id,
            content="分类标签测试",
            chunk_index=0
        )
        
        # 设置分类和标签
        categories = ["技术文档", "API说明"]
        tags = ["重要", "核心概念", "示例"]
        
        chunk.categories = categories
        chunk.tags = tags
        
        assert chunk.categories == categories
        assert chunk.tags == tags
    
    def test_quality_assessment(self):
        """测试质量评估"""
        doc_id = uuid.uuid4()
        chunk = Chunk(
            document_id=doc_id,
            content="质量评估测试内容，这是一个结构良好的文本块。",
            chunk_index=0
        )
        
        # 设置质量评分
        chunk.quality_score = 0.85
        chunk.coherence_score = 0.90
        
        assert chunk.quality_score == 0.85
        assert chunk.coherence_score == 0.90
    
    def test_similarity_calculation_placeholder(self):
        """测试相似度计算（占位符实现）"""
        doc_id = uuid.uuid4()
        chunk = Chunk(
            document_id=doc_id,
            content="相似度测试",
            chunk_index=0
        )
        
        # 设置嵌入
        embedding = [0.1] * 1536
        chunk.embedding = embedding
        
        # 测试相似度计算（当前返回0.0）
        other_embedding = [0.2] * 1536
        similarity = chunk.calculate_similarity(other_embedding)
        
        # 当前实现返回0.0，实际实现应该计算余弦相似度
        assert similarity == 0.0
        
        # 测试空嵌入情况
        chunk.embedding = None
        similarity = chunk.calculate_similarity(other_embedding)
        assert similarity == 0.0


@pytest.mark.unit
@pytest.mark.database
class TestChunkDatabaseIntegration:
    """Chunk 模型数据库集成测试"""
    
    def test_chunk_persistence(self, test_db_session: Session, sample_document: Document):
        """测试文本块持久化"""
        chunk = Chunk(
            document_id=sample_document.id,
            content="持久化测试内容",
            chunk_index=0,
            token_count=10
        )
        
        # 保存到数据库
        test_db_session.add(chunk)
        test_db_session.commit()
        test_db_session.refresh(chunk)
        
        # 验证保存成功
        assert chunk.id is not None
        assert chunk.created_at is not None
        assert chunk.updated_at is not None
    
    def test_chunk_document_relationship(self, test_db_session: Session, sample_document: Document):
        """测试文本块与文档的关系"""
        chunk = Chunk(
            document_id=sample_document.id,
            content="关系测试内容",
            chunk_index=0
        )
        
        test_db_session.add(chunk)
        test_db_session.commit()
        test_db_session.refresh(chunk)
        
        # 验证关系
        assert chunk.document_id == sample_document.id
        # 注意：在测试环境中，关系可能需要显式加载
    
    def test_chunk_query_by_document(self, test_db_session: Session, sample_document: Document):
        """测试按文档查询文本块"""
        # 创建多个文本块
        chunks = []
        for i in range(3):
            chunk = Chunk(
                document_id=sample_document.id,
                content=f"查询测试内容 {i}",
                chunk_index=i
            )
            chunks.append(chunk)
            test_db_session.add(chunk)
        
        test_db_session.commit()
        
        # 查询文档的所有文本块
        found_chunks = test_db_session.query(Chunk).filter(
            Chunk.document_id == sample_document.id
        ).order_by(Chunk.chunk_index).all()
        
        assert len(found_chunks) == 3
        for i, chunk in enumerate(found_chunks):
            assert chunk.chunk_index == i
            assert f"查询测试内容 {i}" in chunk.content
    
    def test_chunk_update(self, test_db_session: Session, sample_document: Document):
        """测试文本块更新"""
        chunk = Chunk(
            document_id=sample_document.id,
            content="原始内容",
            chunk_index=0,
            status="pending"
        )
        
        test_db_session.add(chunk)
        test_db_session.commit()
        
        # 更新状态和嵌入
        embedding = [0.1] * 1536
        chunk.update_embedding(embedding, "new-model")
        test_db_session.commit()
        
        # 重新查询验证
        updated = test_db_session.query(Chunk).filter(
            Chunk.id == chunk.id
        ).first()
        
        assert updated.status == "completed"
        assert updated.embedding_model == "new-model"
        assert updated.is_embedded
    
    def test_chunk_deletion(self, test_db_session: Session, sample_document: Document):
        """测试文本块删除"""
        chunk = Chunk(
            document_id=sample_document.id,
            content="删除测试",
            chunk_index=0
        )
        
        test_db_session.add(chunk)
        test_db_session.commit()
        chunk_id = chunk.id
        
        # 删除文本块
        test_db_session.delete(chunk)
        test_db_session.commit()
        
        # 验证删除成功
        deleted = test_db_session.query(Chunk).filter(
            Chunk.id == chunk_id
        ).first()
        
        assert deleted is None