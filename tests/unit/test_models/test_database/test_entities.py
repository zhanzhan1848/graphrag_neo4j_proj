#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 实体数据库模型单元测试
=============================

测试 Entity 模型的功能：
1. 基本字段和属性
2. 实体类型和分类
3. 别名和同义词管理
4. 向量嵌入
5. 关系管理
6. 验证和链接状态

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import pytest
import uuid
from sqlalchemy.orm import Session

from app.models.database.entities import Entity
from app.models.database.documents import Document
from app.models.database.chunks import Chunk


class TestEntityModel:
    """Entity 模型基础功能测试"""
    
    def test_entity_creation(self):
        """测试实体创建"""
        doc_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        
        entity_data = {
            "document_id": doc_id,
            "chunk_id": chunk_id,
            "canonical_name": "张三",
            "display_name": "张三",
            "entity_type": "PERSON",
            "description": "一个测试人物实体",
            "confidence": 0.95
        }
        
        entity = Entity(**entity_data)
        
        # 验证基本字段
        assert entity.document_id == doc_id
        assert entity.chunk_id == chunk_id
        assert entity.canonical_name == "张三"
        assert entity.display_name == "张三"
        assert entity.entity_type == "PERSON"
        assert entity.description == "一个测试人物实体"
        assert entity.confidence == 0.95
    
    def test_entity_default_values(self):
        """测试实体默认值"""
        doc_id = uuid.uuid4()
        entity = Entity(
            document_id=doc_id,
            canonical_name="测试实体",
            display_name="测试实体",
            entity_type="CONCEPT"
        )
        
        # 验证默认值
        assert entity.confidence == 0.0
        assert entity.mention_count == 1
        assert entity.is_verified is False
        assert entity.is_linked is False
        assert entity.embedding_model == "text-embedding-ada-002"
        assert entity.aliases == [] or entity.aliases is None
        assert entity.synonyms == [] or entity.synonyms is None
        assert entity.categories == [] or entity.categories is None
        assert entity.tags == [] or entity.tags is None
    
    def test_entity_types(self):
        """测试实体类型"""
        doc_id = uuid.uuid4()
        valid_types = [
            "PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", 
            "EVENT", "PRODUCT", "DATE", "MONEY", "PERCENT"
        ]
        
        for entity_type in valid_types:
            entity = Entity(
                document_id=doc_id,
                canonical_name=f"测试{entity_type}",
                display_name=f"测试{entity_type}",
                entity_type=entity_type
            )
            assert entity.entity_type == entity_type
    
    def test_add_alias_method(self):
        """测试添加别名方法"""
        doc_id = uuid.uuid4()
        entity = Entity(
            document_id=doc_id,
            canonical_name="张三",
            display_name="张三",
            entity_type="PERSON"
        )
        
        # 添加别名
        entity.add_alias("小张")
        assert "小张" in entity.aliases
        
        # 添加重复别名（应该不重复）
        entity.add_alias("小张")
        alias_count = entity.aliases.count("小张") if entity.aliases else 0
        assert alias_count == 1
        
        # 添加多个别名
        entity.add_alias("张先生")
        entity.add_alias("老张")
        assert len(entity.aliases) == 3
        assert "张先生" in entity.aliases
        assert "老张" in entity.aliases
    
    def test_entity_properties(self):
        """测试实体属性"""
        doc_id = uuid.uuid4()
        entity = Entity(
            document_id=doc_id,
            canonical_name="苹果公司",
            display_name="苹果公司",
            entity_type="ORGANIZATION"
        )
        
        # 设置属性
        properties = {
            "founded": "1976",
            "headquarters": "Cupertino, California",
            "industry": "Technology",
            "employees": "164000"
        }
        entity.properties = properties
        
        assert entity.properties == properties
        assert entity.properties["founded"] == "1976"
        assert entity.properties["industry"] == "Technology"
    
    def test_entity_repr(self):
        """测试实体字符串表示"""
        doc_id = uuid.uuid4()
        entity = Entity(
            document_id=doc_id,
            canonical_name="测试实体",
            display_name="测试实体",
            entity_type="CONCEPT"
        )
        
        repr_str = repr(entity)
        assert "Entity" in repr_str
        assert "测试实体" in repr_str
        assert "CONCEPT" in repr_str
    
    def test_entity_to_dict(self):
        """测试实体转换为字典"""
        doc_id = uuid.uuid4()
        entity_data = {
            "document_id": doc_id,
            "canonical_name": "北京",
            "display_name": "北京市",
            "entity_type": "LOCATION",
            "description": "中国首都",
            "aliases": ["北京市", "首都"],
            "properties": {"country": "中国", "population": "21000000"}
        }
        
        entity = Entity(**entity_data)
        result = entity.to_dict()
        
        # 验证关键字段
        assert result["canonical_name"] == "北京"
        assert result["display_name"] == "北京市"
        assert result["entity_type"] == "LOCATION"
        assert result["description"] == "中国首都"
        assert result["aliases"] == ["北京市", "首都"]
        assert result["properties"] == {"country": "中国", "population": "21000000"}


class TestEntityValidation:
    """Entity 模型验证测试"""
    
    def test_required_fields(self):
        """测试必需字段"""
        doc_id = uuid.uuid4()
        
        # 最小必需字段
        entity = Entity(
            document_id=doc_id,
            canonical_name="必需字段测试",
            display_name="必需字段测试",
            entity_type="CONCEPT"
        )
        
        assert entity.document_id == doc_id
        assert entity.canonical_name == "必需字段测试"
        assert entity.display_name == "必需字段测试"
        assert entity.entity_type == "CONCEPT"
    
    def test_confidence_range(self):
        """测试置信度范围"""
        doc_id = uuid.uuid4()
        entity = Entity(
            document_id=doc_id,
            canonical_name="置信度测试",
            display_name="置信度测试",
            entity_type="CONCEPT"
        )
        
        # 测试有效置信度值
        valid_confidences = [0.0, 0.25, 0.5, 0.75, 1.0]
        for confidence in valid_confidences:
            entity.confidence = confidence
            assert entity.confidence == confidence
    
    def test_position_information(self):
        """测试位置信息"""
        doc_id = uuid.uuid4()
        entity = Entity(
            document_id=doc_id,
            canonical_name="位置测试",
            display_name="位置测试",
            entity_type="CONCEPT",
            start_pos=10,
            end_pos=20
        )
        
        assert entity.start_pos == 10
        assert entity.end_pos == 20
    
    def test_external_linking(self):
        """测试外部链接"""
        doc_id = uuid.uuid4()
        entity = Entity(
            document_id=doc_id,
            canonical_name="维基百科测试",
            display_name="维基百科测试",
            entity_type="CONCEPT"
        )
        
        # 设置外部链接信息
        entity.external_id = "Q123456"
        entity.external_url = "https://www.wikidata.org/wiki/Q123456"
        entity.is_linked = True
        
        assert entity.external_id == "Q123456"
        assert entity.external_url == "https://www.wikidata.org/wiki/Q123456"
        assert entity.is_linked is True
    
    def test_verification_status(self):
        """测试验证状态"""
        doc_id = uuid.uuid4()
        entity = Entity(
            document_id=doc_id,
            canonical_name="验证测试",
            display_name="验证测试",
            entity_type="CONCEPT"
        )
        
        # 初始状态
        assert entity.is_verified is False
        
        # 设置为已验证
        entity.is_verified = True
        assert entity.is_verified is True


class TestEntityBusinessLogic:
    """Entity 模型业务逻辑测试"""
    
    def test_alias_management(self):
        """测试别名管理"""
        doc_id = uuid.uuid4()
        entity = Entity(
            document_id=doc_id,
            canonical_name="中华人民共和国",
            display_name="中华人民共和国",
            entity_type="LOCATION"
        )
        
        # 添加多个别名
        aliases = ["中国", "China", "PRC", "中华人民共和国"]
        for alias in aliases:
            entity.add_alias(alias)
        
        # 验证别名去重
        unique_aliases = set(entity.aliases) if entity.aliases else set()
        assert len(unique_aliases) == len(aliases)
    
    def test_synonym_management(self):
        """测试同义词管理"""
        doc_id = uuid.uuid4()
        entity = Entity(
            document_id=doc_id,
            canonical_name="人工智能",
            display_name="人工智能",
            entity_type="CONCEPT"
        )
        
        # 设置同义词
        synonyms = ["AI", "机器智能", "智能系统"]
        entity.synonyms = synonyms
        
        assert entity.synonyms == synonyms
        assert "AI" in entity.synonyms
        assert "机器智能" in entity.synonyms
    
    def test_importance_scoring(self):
        """测试重要性评分"""
        doc_id = uuid.uuid4()
        entity = Entity(
            document_id=doc_id,
            canonical_name="重要实体",
            display_name="重要实体",
            entity_type="CONCEPT"
        )
        
        # 设置重要性评分
        entity.importance_score = 0.85
        entity.mention_count = 15
        
        assert entity.importance_score == 0.85
        assert entity.mention_count == 15
    
    def test_categorization_and_tagging(self):
        """测试分类和标签"""
        doc_id = uuid.uuid4()
        entity = Entity(
            document_id=doc_id,
            canonical_name="技术概念",
            display_name="技术概念",
            entity_type="CONCEPT"
        )
        
        # 设置分类和标签
        categories = ["技术", "计算机科学", "人工智能"]
        tags = ["核心概念", "重要", "前沿技术"]
        
        entity.categories = categories
        entity.tags = tags
        
        assert entity.categories == categories
        assert entity.tags == tags
    
    def test_embedding_management(self):
        """测试嵌入管理"""
        doc_id = uuid.uuid4()
        entity = Entity(
            document_id=doc_id,
            canonical_name="嵌入测试",
            display_name="嵌入测试",
            entity_type="CONCEPT"
        )
        
        # 设置嵌入
        embedding = [0.1] * 1536
        entity.embedding = embedding
        entity.embedding_model = "text-embedding-3-large"
        
        assert entity.embedding == embedding
        assert entity.embedding_model == "text-embedding-3-large"
    
    def test_extraction_metadata(self):
        """测试抽取元数据"""
        doc_id = uuid.uuid4()
        entity = Entity(
            document_id=doc_id,
            canonical_name="抽取测试",
            display_name="抽取测试",
            entity_type="CONCEPT"
        )
        
        # 设置抽取信息
        entity.extraction_method = "LLM"
        entity.confidence = 0.92
        entity.start_pos = 100
        entity.end_pos = 110
        
        assert entity.extraction_method == "LLM"
        assert entity.confidence == 0.92
        assert entity.start_pos == 100
        assert entity.end_pos == 110


class TestEntityRelationships:
    """Entity 关系测试"""
    
    def test_relation_count_property(self):
        """测试关系计数属性"""
        doc_id = uuid.uuid4()
        entity = Entity(
            document_id=doc_id,
            canonical_name="关系测试",
            display_name="关系测试",
            entity_type="PERSON"
        )
        
        # 注意：在单元测试中，关系计数可能为0，因为没有实际的关系对象
        # 这个测试主要验证属性存在且可调用
        relation_count = entity.relation_count
        assert isinstance(relation_count, int)
        assert relation_count >= 0
    
    def test_all_relations_property(self):
        """测试所有关系属性"""
        doc_id = uuid.uuid4()
        entity = Entity(
            document_id=doc_id,
            canonical_name="关系测试",
            display_name="关系测试",
            entity_type="PERSON"
        )
        
        # 注意：在单元测试中，关系列表可能为空
        all_relations = entity.all_relations
        assert isinstance(all_relations, list)


@pytest.mark.unit
@pytest.mark.database
class TestEntityDatabaseIntegration:
    """Entity 模型数据库集成测试"""
    
    def test_entity_persistence(self, test_db_session: Session, sample_document: Document):
        """测试实体持久化"""
        entity = Entity(
            document_id=sample_document.id,
            canonical_name="持久化测试",
            display_name="持久化测试",
            entity_type="CONCEPT",
            confidence=0.85
        )
        
        # 保存到数据库
        test_db_session.add(entity)
        test_db_session.commit()
        test_db_session.refresh(entity)
        
        # 验证保存成功
        assert entity.id is not None
        assert entity.created_at is not None
        assert entity.updated_at is not None
    
    def test_entity_document_relationship(self, test_db_session: Session, sample_document: Document):
        """测试实体与文档的关系"""
        entity = Entity(
            document_id=sample_document.id,
            canonical_name="关系测试",
            display_name="关系测试",
            entity_type="CONCEPT"
        )
        
        test_db_session.add(entity)
        test_db_session.commit()
        test_db_session.refresh(entity)
        
        # 验证关系
        assert entity.document_id == sample_document.id
    
    def test_entity_chunk_relationship(self, test_db_session: Session, sample_document: Document):
        """测试实体与文本块的关系"""
        # 创建文本块
        chunk = Chunk(
            document_id=sample_document.id,
            content="实体关系测试内容",
            chunk_index=0
        )
        test_db_session.add(chunk)
        test_db_session.commit()
        test_db_session.refresh(chunk)
        
        # 创建实体
        entity = Entity(
            document_id=sample_document.id,
            chunk_id=chunk.id,
            canonical_name="块关系测试",
            display_name="块关系测试",
            entity_type="CONCEPT"
        )
        
        test_db_session.add(entity)
        test_db_session.commit()
        test_db_session.refresh(entity)
        
        # 验证关系
        assert entity.chunk_id == chunk.id
    
    def test_entity_query_by_type(self, test_db_session: Session, sample_document: Document):
        """测试按类型查询实体"""
        # 创建不同类型的实体
        entity_types = ["PERSON", "ORGANIZATION", "LOCATION"]
        entities = []
        
        for i, entity_type in enumerate(entity_types):
            entity = Entity(
                document_id=sample_document.id,
                canonical_name=f"测试{entity_type}",
                display_name=f"测试{entity_type}",
                entity_type=entity_type
            )
            entities.append(entity)
            test_db_session.add(entity)
        
        test_db_session.commit()
        
        # 查询特定类型的实体
        person_entities = test_db_session.query(Entity).filter(
            Entity.entity_type == "PERSON"
        ).all()
        
        assert len(person_entities) == 1
        assert person_entities[0].entity_type == "PERSON"
    
    def test_entity_update(self, test_db_session: Session, sample_document: Document):
        """测试实体更新"""
        entity = Entity(
            document_id=sample_document.id,
            canonical_name="原始名称",
            display_name="原始名称",
            entity_type="CONCEPT",
            is_verified=False
        )
        
        test_db_session.add(entity)
        test_db_session.commit()
        
        # 更新实体
        entity.display_name = "更新后名称"
        entity.is_verified = True
        entity.add_alias("别名1")
        test_db_session.commit()
        
        # 重新查询验证
        updated = test_db_session.query(Entity).filter(
            Entity.id == entity.id
        ).first()
        
        assert updated.display_name == "更新后名称"
        assert updated.is_verified is True
        assert "别名1" in updated.aliases
    
    def test_entity_deletion(self, test_db_session: Session, sample_document: Document):
        """测试实体删除"""
        entity = Entity(
            document_id=sample_document.id,
            canonical_name="删除测试",
            display_name="删除测试",
            entity_type="CONCEPT"
        )
        
        test_db_session.add(entity)
        test_db_session.commit()
        entity_id = entity.id
        
        # 删除实体
        test_db_session.delete(entity)
        test_db_session.commit()
        
        # 验证删除成功
        deleted = test_db_session.query(Entity).filter(
            Entity.id == entity_id
        ).first()
        
        assert deleted is None