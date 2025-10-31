#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 关系数据库模型单元测试
=============================

测试 Relation 模型的功能：
1. 基本字段和属性
2. 关系类型和方向
3. 置信度和证据
4. 实体关联
5. 验证和状态管理

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import pytest
import uuid
from sqlalchemy.orm import Session

from app.models.database.relations import Relation
from app.models.database.entities import Entity
from app.models.database.documents import Document
from app.models.database.chunks import Chunk


class TestRelationModel:
    """Relation 模型基础功能测试"""
    
    def test_relation_creation(self):
        """测试关系创建"""
        doc_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        
        relation_data = {
            "document_id": doc_id,
            "chunk_id": chunk_id,
            "source_entity_id": source_id,
            "target_entity_id": target_id,
            "relation_type": "WORKS_FOR",
            "description": "张三在苹果公司工作",
            "confidence": 0.92
        }
        
        relation = Relation(**relation_data)
        
        # 验证基本字段
        assert relation.document_id == doc_id
        assert relation.chunk_id == chunk_id
        assert relation.source_entity_id == source_id
        assert relation.target_entity_id == target_id
        assert relation.relation_type == "WORKS_FOR"
        assert relation.description == "张三在苹果公司工作"
        assert relation.confidence == 0.92
    
    def test_relation_default_values(self):
        """测试关系默认值"""
        doc_id = uuid.uuid4()
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        
        relation = Relation(
            document_id=doc_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type="RELATED_TO"
        )
        
        # 验证默认值
        assert relation.confidence == 0.0
        assert relation.is_verified is False
        assert relation.is_bidirectional is False
        assert relation.weight == 1.0
        assert relation.evidence == [] or relation.evidence is None
        assert relation.properties == {} or relation.properties is None
    
    def test_relation_types(self):
        """测试关系类型"""
        doc_id = uuid.uuid4()
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        
        valid_types = [
            "WORKS_FOR", "LOCATED_IN", "PART_OF", "RELATED_TO",
            "CAUSES", "LEADS_TO", "MENTIONS", "CITES", "SIMILAR_TO"
        ]
        
        for relation_type in valid_types:
            relation = Relation(
                document_id=doc_id,
                source_entity_id=source_id,
                target_entity_id=target_id,
                relation_type=relation_type
            )
            assert relation.relation_type == relation_type
    
    def test_bidirectional_relation(self):
        """测试双向关系"""
        doc_id = uuid.uuid4()
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        
        relation = Relation(
            document_id=doc_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type="SIMILAR_TO",
            is_bidirectional=True
        )
        
        assert relation.is_bidirectional is True
    
    def test_relation_weight(self):
        """测试关系权重"""
        doc_id = uuid.uuid4()
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        
        relation = Relation(
            document_id=doc_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type="WORKS_FOR",
            weight=2.5
        )
        
        assert relation.weight == 2.5
    
    def test_relation_evidence(self):
        """测试关系证据"""
        doc_id = uuid.uuid4()
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        
        evidence = [
            "张三在苹果公司的官网上被列为员工",
            "新闻报道提到张三是苹果公司的工程师"
        ]
        
        relation = Relation(
            document_id=doc_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type="WORKS_FOR",
            evidence=evidence
        )
        
        assert relation.evidence == evidence
        assert len(relation.evidence) == 2
    
    def test_relation_properties(self):
        """测试关系属性"""
        doc_id = uuid.uuid4()
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        
        properties = {
            "start_date": "2020-01-01",
            "position": "Senior Engineer",
            "department": "AI Research"
        }
        
        relation = Relation(
            document_id=doc_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type="WORKS_FOR",
            properties=properties
        )
        
        assert relation.properties == properties
        assert relation.properties["position"] == "Senior Engineer"
    
    def test_relation_repr(self):
        """测试关系字符串表示"""
        doc_id = uuid.uuid4()
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        
        relation = Relation(
            document_id=doc_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type="WORKS_FOR",
            description="工作关系"
        )
        
        repr_str = repr(relation)
        assert "Relation" in repr_str
        assert "WORKS_FOR" in repr_str
    
    def test_relation_to_dict(self):
        """测试关系转换为字典"""
        doc_id = uuid.uuid4()
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        
        relation_data = {
            "document_id": doc_id,
            "source_entity_id": source_id,
            "target_entity_id": target_id,
            "relation_type": "LOCATED_IN",
            "description": "北京位于中国",
            "confidence": 0.98,
            "evidence": ["地理常识"],
            "properties": {"country": "中国"}
        }
        
        relation = Relation(**relation_data)
        result = relation.to_dict()
        
        # 验证关键字段
        assert result["relation_type"] == "LOCATED_IN"
        assert result["description"] == "北京位于中国"
        assert result["confidence"] == 0.98
        assert result["evidence"] == ["地理常识"]
        assert result["properties"] == {"country": "中国"}


class TestRelationValidation:
    """Relation 模型验证测试"""
    
    def test_required_fields(self):
        """测试必需字段"""
        doc_id = uuid.uuid4()
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        
        # 最小必需字段
        relation = Relation(
            document_id=doc_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type="RELATED_TO"
        )
        
        assert relation.document_id == doc_id
        assert relation.source_entity_id == source_id
        assert relation.target_entity_id == target_id
        assert relation.relation_type == "RELATED_TO"
    
    def test_confidence_range(self):
        """测试置信度范围"""
        doc_id = uuid.uuid4()
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        
        relation = Relation(
            document_id=doc_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type="RELATED_TO"
        )
        
        # 测试有效置信度值
        valid_confidences = [0.0, 0.25, 0.5, 0.75, 1.0]
        for confidence in valid_confidences:
            relation.confidence = confidence
            assert relation.confidence == confidence
    
    def test_weight_validation(self):
        """测试权重验证"""
        doc_id = uuid.uuid4()
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        
        relation = Relation(
            document_id=doc_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type="RELATED_TO"
        )
        
        # 测试有效权重值
        valid_weights = [0.1, 1.0, 2.0, 5.0, 10.0]
        for weight in valid_weights:
            relation.weight = weight
            assert relation.weight == weight
    
    def test_position_information(self):
        """测试位置信息"""
        doc_id = uuid.uuid4()
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        
        relation = Relation(
            document_id=doc_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type="MENTIONS",
            start_pos=50,
            end_pos=80
        )
        
        assert relation.start_pos == 50
        assert relation.end_pos == 80
    
    def test_verification_status(self):
        """测试验证状态"""
        doc_id = uuid.uuid4()
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        
        relation = Relation(
            document_id=doc_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type="VERIFIED_RELATION"
        )
        
        # 初始状态
        assert relation.is_verified is False
        
        # 设置为已验证
        relation.is_verified = True
        assert relation.is_verified is True


class TestRelationBusinessLogic:
    """Relation 模型业务逻辑测试"""
    
    def test_evidence_management(self):
        """测试证据管理"""
        doc_id = uuid.uuid4()
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        
        relation = Relation(
            document_id=doc_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type="WORKS_FOR"
        )
        
        # 添加证据
        evidence_list = [
            "员工名录显示该关系",
            "公司官网确认",
            "新闻报道提及"
        ]
        
        relation.evidence = evidence_list
        assert relation.evidence == evidence_list
        assert len(relation.evidence) == 3
    
    def test_temporal_properties(self):
        """测试时间属性"""
        doc_id = uuid.uuid4()
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        
        relation = Relation(
            document_id=doc_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type="WORKS_FOR"
        )
        
        # 设置时间属性
        temporal_props = {
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
            "duration": "4 years"
        }
        
        relation.properties = temporal_props
        assert relation.properties["start_date"] == "2020-01-01"
        assert relation.properties["duration"] == "4 years"
    
    def test_relation_strength(self):
        """测试关系强度"""
        doc_id = uuid.uuid4()
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        
        relation = Relation(
            document_id=doc_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type="SIMILAR_TO",
            confidence=0.95,
            weight=3.0
        )
        
        # 关系强度可以通过置信度和权重计算
        strength = relation.confidence * relation.weight
        assert strength == 2.85  # 0.95 * 3.0
    
    def test_bidirectional_logic(self):
        """测试双向关系逻辑"""
        doc_id = uuid.uuid4()
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        
        # 创建双向关系
        relation = Relation(
            document_id=doc_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type="SIMILAR_TO",
            is_bidirectional=True
        )
        
        assert relation.is_bidirectional is True
        
        # 单向关系
        unidirectional = Relation(
            document_id=doc_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type="WORKS_FOR",
            is_bidirectional=False
        )
        
        assert unidirectional.is_bidirectional is False
    
    def test_extraction_metadata(self):
        """测试抽取元数据"""
        doc_id = uuid.uuid4()
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        
        relation = Relation(
            document_id=doc_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type="MENTIONS"
        )
        
        # 设置抽取信息
        relation.extraction_method = "LLM"
        relation.confidence = 0.88
        relation.start_pos = 200
        relation.end_pos = 250
        
        assert relation.extraction_method == "LLM"
        assert relation.confidence == 0.88
        assert relation.start_pos == 200
        assert relation.end_pos == 250
    
    def test_relation_context(self):
        """测试关系上下文"""
        doc_id = uuid.uuid4()
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        
        relation = Relation(
            document_id=doc_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type="LOCATED_IN",
            description="苹果公司总部位于加利福尼亚州库比蒂诺"
        )
        
        # 设置上下文信息
        context_props = {
            "context": "公司信息介绍段落",
            "sentence": "苹果公司总部位于加利福尼亚州库比蒂诺",
            "paragraph_index": 2
        }
        
        relation.properties = context_props
        assert relation.properties["context"] == "公司信息介绍段落"
        assert relation.properties["paragraph_index"] == 2


class TestRelationComplexScenarios:
    """Relation 复杂场景测试"""
    
    def test_multiple_evidence_sources(self):
        """测试多证据源"""
        doc_id = uuid.uuid4()
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        
        relation = Relation(
            document_id=doc_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type="WORKS_FOR"
        )
        
        # 多个证据源
        evidence_sources = [
            "LinkedIn档案显示",
            "公司年报提及",
            "新闻采访确认",
            "同事证实"
        ]
        
        relation.evidence = evidence_sources
        assert len(relation.evidence) == 4
        
        # 高置信度（多证据支持）
        relation.confidence = 0.98
        assert relation.confidence == 0.98
    
    def test_conflicting_relations(self):
        """测试冲突关系处理"""
        doc_id = uuid.uuid4()
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        
        # 创建两个可能冲突的关系
        relation1 = Relation(
            document_id=doc_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type="WORKS_FOR",
            confidence=0.85,
            evidence=["来源A"]
        )
        
        relation2 = Relation(
            document_id=doc_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type="FORMERLY_WORKED_FOR",
            confidence=0.90,
            evidence=["来源B", "来源C"]
        )
        
        # 验证两个关系都可以存在（可能表示时间上的变化）
        assert relation1.relation_type == "WORKS_FOR"
        assert relation2.relation_type == "FORMERLY_WORKED_FOR"
        assert relation2.confidence > relation1.confidence
    
    def test_hierarchical_relations(self):
        """测试层次关系"""
        doc_id = uuid.uuid4()
        parent_id = uuid.uuid4()
        child_id = uuid.uuid4()
        
        relation = Relation(
            document_id=doc_id,
            source_entity_id=child_id,
            target_entity_id=parent_id,
            relation_type="PART_OF",
            description="部门隶属于公司"
        )
        
        # 设置层次属性
        hierarchy_props = {
            "level": 2,
            "hierarchy_type": "organizational",
            "parent_type": "ORGANIZATION",
            "child_type": "DEPARTMENT"
        }
        
        relation.properties = hierarchy_props
        assert relation.properties["level"] == 2
        assert relation.properties["hierarchy_type"] == "organizational"


@pytest.mark.unit
@pytest.mark.database
class TestRelationDatabaseIntegration:
    """Relation 模型数据库集成测试"""
    
    def test_relation_persistence(self, test_db_session: Session, sample_document: Document):
        """测试关系持久化"""
        # 创建两个实体
        entity1 = Entity(
            document_id=sample_document.id,
            canonical_name="实体1",
            display_name="实体1",
            entity_type="PERSON"
        )
        entity2 = Entity(
            document_id=sample_document.id,
            canonical_name="实体2",
            display_name="实体2",
            entity_type="ORGANIZATION"
        )
        
        test_db_session.add_all([entity1, entity2])
        test_db_session.commit()
        test_db_session.refresh(entity1)
        test_db_session.refresh(entity2)
        
        # 创建关系
        relation = Relation(
            document_id=sample_document.id,
            source_entity_id=entity1.id,
            target_entity_id=entity2.id,
            relation_type="WORKS_FOR",
            confidence=0.90
        )
        
        # 保存到数据库
        test_db_session.add(relation)
        test_db_session.commit()
        test_db_session.refresh(relation)
        
        # 验证保存成功
        assert relation.id is not None
        assert relation.created_at is not None
        assert relation.updated_at is not None
    
    def test_relation_entity_relationships(self, test_db_session: Session, sample_document: Document):
        """测试关系与实体的关联"""
        # 创建实体
        source_entity = Entity(
            document_id=sample_document.id,
            canonical_name="源实体",
            display_name="源实体",
            entity_type="PERSON"
        )
        target_entity = Entity(
            document_id=sample_document.id,
            canonical_name="目标实体",
            display_name="目标实体",
            entity_type="ORGANIZATION"
        )
        
        test_db_session.add_all([source_entity, target_entity])
        test_db_session.commit()
        
        # 创建关系
        relation = Relation(
            document_id=sample_document.id,
            source_entity_id=source_entity.id,
            target_entity_id=target_entity.id,
            relation_type="WORKS_FOR"
        )
        
        test_db_session.add(relation)
        test_db_session.commit()
        test_db_session.refresh(relation)
        
        # 验证关系
        assert relation.source_entity_id == source_entity.id
        assert relation.target_entity_id == target_entity.id
    
    def test_relation_chunk_relationship(self, test_db_session: Session, sample_document: Document):
        """测试关系与文本块的关联"""
        # 创建文本块
        chunk = Chunk(
            document_id=sample_document.id,
            content="关系测试内容",
            chunk_index=0
        )
        test_db_session.add(chunk)
        test_db_session.commit()
        test_db_session.refresh(chunk)
        
        # 创建实体
        entity1 = Entity(
            document_id=sample_document.id,
            canonical_name="实体1",
            display_name="实体1",
            entity_type="PERSON"
        )
        entity2 = Entity(
            document_id=sample_document.id,
            canonical_name="实体2",
            display_name="实体2",
            entity_type="ORGANIZATION"
        )
        
        test_db_session.add_all([entity1, entity2])
        test_db_session.commit()
        
        # 创建关系
        relation = Relation(
            document_id=sample_document.id,
            chunk_id=chunk.id,
            source_entity_id=entity1.id,
            target_entity_id=entity2.id,
            relation_type="WORKS_FOR"
        )
        
        test_db_session.add(relation)
        test_db_session.commit()
        test_db_session.refresh(relation)
        
        # 验证关系
        assert relation.chunk_id == chunk.id
    
    def test_relation_query_by_type(self, test_db_session: Session, sample_document: Document):
        """测试按类型查询关系"""
        # 创建实体
        entity1 = Entity(
            document_id=sample_document.id,
            canonical_name="查询实体1",
            display_name="查询实体1",
            entity_type="PERSON"
        )
        entity2 = Entity(
            document_id=sample_document.id,
            canonical_name="查询实体2",
            display_name="查询实体2",
            entity_type="ORGANIZATION"
        )
        
        test_db_session.add_all([entity1, entity2])
        test_db_session.commit()
        
        # 创建不同类型的关系
        relation_types = ["WORKS_FOR", "LOCATED_IN", "PART_OF"]
        relations = []
        
        for relation_type in relation_types:
            relation = Relation(
                document_id=sample_document.id,
                source_entity_id=entity1.id,
                target_entity_id=entity2.id,
                relation_type=relation_type
            )
            relations.append(relation)
            test_db_session.add(relation)
        
        test_db_session.commit()
        
        # 查询特定类型的关系
        work_relations = test_db_session.query(Relation).filter(
            Relation.relation_type == "WORKS_FOR"
        ).all()
        
        assert len(work_relations) == 1
        assert work_relations[0].relation_type == "WORKS_FOR"
    
    def test_relation_update(self, test_db_session: Session, sample_document: Document):
        """测试关系更新"""
        # 创建实体
        entity1 = Entity(
            document_id=sample_document.id,
            canonical_name="更新实体1",
            display_name="更新实体1",
            entity_type="PERSON"
        )
        entity2 = Entity(
            document_id=sample_document.id,
            canonical_name="更新实体2",
            display_name="更新实体2",
            entity_type="ORGANIZATION"
        )
        
        test_db_session.add_all([entity1, entity2])
        test_db_session.commit()
        
        # 创建关系
        relation = Relation(
            document_id=sample_document.id,
            source_entity_id=entity1.id,
            target_entity_id=entity2.id,
            relation_type="WORKS_FOR",
            confidence=0.80,
            is_verified=False
        )
        
        test_db_session.add(relation)
        test_db_session.commit()
        
        # 更新关系
        relation.confidence = 0.95
        relation.is_verified = True
        relation.evidence = ["新证据1", "新证据2"]
        test_db_session.commit()
        
        # 重新查询验证
        updated = test_db_session.query(Relation).filter(
            Relation.id == relation.id
        ).first()
        
        assert updated.confidence == 0.95
        assert updated.is_verified is True
        assert len(updated.evidence) == 2
    
    def test_relation_deletion(self, test_db_session: Session, sample_document: Document):
        """测试关系删除"""
        # 创建实体
        entity1 = Entity(
            document_id=sample_document.id,
            canonical_name="删除实体1",
            display_name="删除实体1",
            entity_type="PERSON"
        )
        entity2 = Entity(
            document_id=sample_document.id,
            canonical_name="删除实体2",
            display_name="删除实体2",
            entity_type="ORGANIZATION"
        )
        
        test_db_session.add_all([entity1, entity2])
        test_db_session.commit()
        
        # 创建关系
        relation = Relation(
            document_id=sample_document.id,
            source_entity_id=entity1.id,
            target_entity_id=entity2.id,
            relation_type="WORKS_FOR"
        )
        
        test_db_session.add(relation)
        test_db_session.commit()
        relation_id = relation.id
        
        # 删除关系
        test_db_session.delete(relation)
        test_db_session.commit()
        
        # 验证删除成功
        deleted = test_db_session.query(Relation).filter(
            Relation.id == relation_id
        ).first()
        
        assert deleted is None