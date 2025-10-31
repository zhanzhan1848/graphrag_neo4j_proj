#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 测试数据和Fixtures
==========================

提供测试用的数据生成器和fixtures：
1. 文档测试数据
2. 实体测试数据
3. 关系测试数据
4. 文本块测试数据
5. 复杂场景数据
6. 性能测试数据

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import pytest
import uuid
import random
import string
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from faker import Faker

from app.models.database.documents import Document, DocumentStatus
from app.models.database.chunks import Chunk, ChunkType
from app.models.database.entities import Entity, EntityType
from app.models.database.relations import Relation, RelationType


# 初始化Faker
fake = Faker(['zh_CN', 'en_US'])


class TestDataGenerator:
    """测试数据生成器"""
    
    @staticmethod
    def generate_random_string(length: int = 10) -> str:
        """生成随机字符串"""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    @staticmethod
    def generate_random_embedding(dimension: int = 768) -> List[float]:
        """生成随机向量嵌入"""
        return [random.uniform(-1.0, 1.0) for _ in range(dimension)]
    
    @staticmethod
    def generate_document_data(**kwargs) -> Dict[str, Any]:
        """生成文档测试数据"""
        default_data = {
            'title': fake.sentence(nb_words=4),
            'file_name': f"{fake.word()}.{random.choice(['txt', 'pdf', 'docx', 'md'])}",
            'file_type': random.choice(['txt', 'pdf', 'docx', 'md']),
            'content': fake.text(max_nb_chars=2000),
            'description': fake.paragraph(nb_sentences=2),
            'language': random.choice(['zh', 'en', 'auto']),
            'file_size': random.randint(1000, 1000000),
            'file_hash': fake.sha256(),
            'mime_type': random.choice(['text/plain', 'application/pdf', 'text/markdown']),
            'tags': [fake.word() for _ in range(random.randint(1, 5))],
            'categories': [fake.word() for _ in range(random.randint(1, 3))],
            'metadata': {
                'author': fake.name(),
                'created_date': fake.date_time().isoformat(),
                'source': fake.url()
            },
            'status': random.choice(list(DocumentStatus)),
            'processing_progress': random.uniform(0.0, 1.0),
            'error_message': None if random.random() > 0.1 else fake.sentence(),
            'quality_score': random.uniform(0.5, 1.0)
        }
        
        # 更新默认数据
        default_data.update(kwargs)
        return default_data
    
    @staticmethod
    def generate_chunk_data(document_id: Optional[uuid.UUID] = None, **kwargs) -> Dict[str, Any]:
        """生成文本块测试数据"""
        default_data = {
            'document_id': document_id or uuid.uuid4(),
            'content': fake.paragraph(nb_sentences=random.randint(2, 8)),
            'chunk_type': random.choice(list(ChunkType)),
            'position': random.randint(0, 100),
            'start_char': random.randint(0, 1000),
            'end_char': random.randint(1001, 2000),
            'token_count': random.randint(10, 500),
            'embedding': TestDataGenerator.generate_random_embedding(),
            'metadata': {
                'section': fake.word(),
                'page': random.randint(1, 100),
                'confidence': random.uniform(0.7, 1.0)
            },
            'tags': [fake.word() for _ in range(random.randint(0, 3))],
            'quality_score': random.uniform(0.6, 1.0),
            'language': random.choice(['zh', 'en']),
            'summary': fake.sentence()
        }
        
        # 确保end_char > start_char
        if default_data['end_char'] <= default_data['start_char']:
            default_data['end_char'] = default_data['start_char'] + random.randint(100, 500)
        
        default_data.update(kwargs)
        return default_data
    
    @staticmethod
    def generate_entity_data(**kwargs) -> Dict[str, Any]:
        """生成实体测试数据"""
        entity_type = random.choice(list(EntityType))
        
        # 根据实体类型生成合适的名称
        if entity_type == EntityType.PERSON:
            canonical_name = fake.name()
            display_name = canonical_name
        elif entity_type == EntityType.ORGANIZATION:
            canonical_name = fake.company()
            display_name = canonical_name
        elif entity_type == EntityType.LOCATION:
            canonical_name = fake.city()
            display_name = canonical_name
        else:
            canonical_name = fake.word().title()
            display_name = canonical_name
        
        default_data = {
            'canonical_name': canonical_name,
            'display_name': display_name,
            'entity_type': entity_type,
            'aliases': [fake.word() for _ in range(random.randint(0, 3))],
            'description': fake.paragraph(nb_sentences=2),
            'properties': {
                'confidence': random.uniform(0.7, 1.0),
                'source': fake.word(),
                'category': fake.word()
            },
            'confidence': random.uniform(0.7, 1.0),
            'embedding': TestDataGenerator.generate_random_embedding(),
            'metadata': {
                'extraction_method': random.choice(['NER', 'manual', 'rule_based']),
                'last_verified': fake.date_time().isoformat()
            },
            'tags': [fake.word() for _ in range(random.randint(0, 3))],
            'external_ids': {
                'wikidata': f"Q{random.randint(1000, 999999)}",
                'dbpedia': fake.url()
            },
            'importance_score': random.uniform(0.1, 1.0)
        }
        
        default_data.update(kwargs)
        return default_data
    
    @staticmethod
    def generate_relation_data(
        source_entity_id: Optional[uuid.UUID] = None,
        target_entity_id: Optional[uuid.UUID] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """生成关系测试数据"""
        default_data = {
            'source_entity_id': source_entity_id or uuid.uuid4(),
            'target_entity_id': target_entity_id or uuid.uuid4(),
            'relation_type': random.choice(list(RelationType)),
            'description': fake.sentence(),
            'confidence': random.uniform(0.7, 1.0),
            'weight': random.uniform(0.1, 1.0),
            'properties': {
                'temporal': fake.date().isoformat(),
                'context': fake.word(),
                'strength': random.uniform(0.1, 1.0)
            },
            'evidence': [fake.sentence() for _ in range(random.randint(1, 3))],
            'metadata': {
                'extraction_method': random.choice(['pattern', 'ml', 'manual']),
                'verified': random.choice([True, False])
            },
            'tags': [fake.word() for _ in range(random.randint(0, 2))],
            'bidirectional': random.choice([True, False]),
            'temporal_info': {
                'start_date': fake.date().isoformat(),
                'end_date': fake.date().isoformat() if random.random() > 0.5 else None
            }
        }
        
        default_data.update(kwargs)
        return default_data


class TestDataFactory:
    """测试数据工厂"""
    
    def __init__(self, db_session):
        self.db_session = db_session
        self.generator = TestDataGenerator()
    
    def create_document(self, **kwargs) -> Document:
        """创建文档对象"""
        data = self.generator.generate_document_data(**kwargs)
        doc = Document(**data)
        self.db_session.add(doc)
        self.db_session.flush()  # 获取ID但不提交
        return doc
    
    def create_chunk(self, document: Optional[Document] = None, **kwargs) -> Chunk:
        """创建文本块对象"""
        if document is None:
            document = self.create_document()
        
        data = self.generator.generate_chunk_data(document_id=document.id, **kwargs)
        chunk = Chunk(**data)
        self.db_session.add(chunk)
        self.db_session.flush()
        return chunk
    
    def create_entity(self, **kwargs) -> Entity:
        """创建实体对象"""
        data = self.generator.generate_entity_data(**kwargs)
        entity = Entity(**data)
        self.db_session.add(entity)
        self.db_session.flush()
        return entity
    
    def create_relation(
        self,
        source_entity: Optional[Entity] = None,
        target_entity: Optional[Entity] = None,
        **kwargs
    ) -> Relation:
        """创建关系对象"""
        if source_entity is None:
            source_entity = self.create_entity()
        if target_entity is None:
            target_entity = self.create_entity()
        
        data = self.generator.generate_relation_data(
            source_entity_id=source_entity.id,
            target_entity_id=target_entity.id,
            **kwargs
        )
        relation = Relation(**data)
        self.db_session.add(relation)
        self.db_session.flush()
        return relation
    
    def create_document_with_chunks(
        self,
        chunk_count: int = 3,
        document_kwargs: Optional[Dict] = None,
        chunk_kwargs: Optional[Dict] = None
    ) -> tuple[Document, List[Chunk]]:
        """创建带有文本块的文档"""
        document_kwargs = document_kwargs or {}
        chunk_kwargs = chunk_kwargs or {}
        
        document = self.create_document(**document_kwargs)
        chunks = []
        
        for i in range(chunk_count):
            chunk_data = {
                'position': i,
                'start_char': i * 100,
                'end_char': (i + 1) * 100 - 1,
                **chunk_kwargs
            }
            chunk = self.create_chunk(document=document, **chunk_data)
            chunks.append(chunk)
        
        return document, chunks
    
    def create_entity_relation_network(
        self,
        entity_count: int = 5,
        relation_count: int = 7
    ) -> tuple[List[Entity], List[Relation]]:
        """创建实体关系网络"""
        entities = []
        for _ in range(entity_count):
            entity = self.create_entity()
            entities.append(entity)
        
        relations = []
        for _ in range(relation_count):
            source = random.choice(entities)
            target = random.choice([e for e in entities if e.id != source.id])
            relation = self.create_relation(source_entity=source, target_entity=target)
            relations.append(relation)
        
        return entities, relations
    
    def create_complete_knowledge_graph(
        self,
        document_count: int = 3,
        chunks_per_document: int = 5,
        entity_count: int = 10,
        relation_count: int = 15
    ) -> Dict[str, Any]:
        """创建完整的知识图谱数据"""
        # 创建文档和块
        documents = []
        all_chunks = []
        
        for _ in range(document_count):
            doc, chunks = self.create_document_with_chunks(chunks_per_document)
            documents.append(doc)
            all_chunks.extend(chunks)
        
        # 创建实体关系网络
        entities, relations = self.create_entity_relation_network(
            entity_count, relation_count
        )
        
        # 建立一些实体与文档/块的关联
        for entity in entities[:len(all_chunks)]:
            chunk = all_chunks[entities.index(entity) % len(all_chunks)]
            # 这里可以添加实体与块的关联逻辑
        
        return {
            'documents': documents,
            'chunks': all_chunks,
            'entities': entities,
            'relations': relations
        }


# Pytest Fixtures
@pytest.fixture
def test_data_generator():
    """测试数据生成器fixture"""
    return TestDataGenerator()


@pytest.fixture
def test_data_factory(db_session):
    """测试数据工厂fixture"""
    return TestDataFactory(db_session)


@pytest.fixture
def sample_document_data():
    """示例文档数据"""
    return {
        'title': '人工智能发展史',
        'file_name': 'ai_history.pdf',
        'file_type': 'pdf',
        'content': '''
        人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，
        它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
        
        人工智能的发展可以追溯到1950年代，当时艾伦·图灵提出了著名的图灵测试。
        1956年，约翰·麦卡锡在达特茅斯会议上首次提出了"人工智能"这个术语。
        
        近年来，深度学习和神经网络的发展推动了AI技术的快速进步，
        特别是在计算机视觉、自然语言处理和语音识别等领域取得了突破性进展。
        ''',
        'description': '介绍人工智能的发展历程和主要里程碑',
        'language': 'zh',
        'tags': ['人工智能', 'AI', '历史', '技术'],
        'categories': ['科技', '历史'],
        'metadata': {
            'author': '张三',
            'publication_date': '2024-01-01',
            'source': 'AI研究院'
        }
    }


@pytest.fixture
def sample_entities_data():
    """示例实体数据"""
    return [
        {
            'canonical_name': '艾伦·图灵',
            'display_name': 'Alan Turing',
            'entity_type': EntityType.PERSON,
            'aliases': ['图灵', 'Turing'],
            'description': '英国数学家、逻辑学家，被称为计算机科学之父',
            'properties': {
                'birth_year': '1912',
                'death_year': '1954',
                'nationality': '英国'
            }
        },
        {
            'canonical_name': '约翰·麦卡锡',
            'display_name': 'John McCarthy',
            'entity_type': EntityType.PERSON,
            'aliases': ['麦卡锡', 'McCarthy'],
            'description': '美国计算机科学家，人工智能之父',
            'properties': {
                'birth_year': '1927',
                'death_year': '2011',
                'nationality': '美国'
            }
        },
        {
            'canonical_name': '达特茅斯会议',
            'display_name': 'Dartmouth Conference',
            'entity_type': EntityType.EVENT,
            'aliases': ['达特茅斯', 'Dartmouth'],
            'description': '1956年举办的人工智能研讨会，标志着AI学科的诞生',
            'properties': {
                'year': '1956',
                'location': '达特茅斯学院'
            }
        },
        {
            'canonical_name': '图灵测试',
            'display_name': 'Turing Test',
            'entity_type': EntityType.CONCEPT,
            'aliases': ['图灵测试', 'Turing Test'],
            'description': '判断机器是否具有智能的测试方法',
            'properties': {
                'proposed_year': '1950',
                'type': '测试方法'
            }
        }
    ]


@pytest.fixture
def sample_relations_data():
    """示例关系数据"""
    return [
        {
            'relation_type': RelationType.PROPOSED,
            'description': '艾伦·图灵提出了图灵测试',
            'properties': {'year': '1950'}
        },
        {
            'relation_type': RelationType.ORGANIZED,
            'description': '约翰·麦卡锡组织了达特茅斯会议',
            'properties': {'year': '1956'}
        },
        {
            'relation_type': RelationType.FOUNDED,
            'description': '达特茅斯会议标志着人工智能学科的创立',
            'properties': {'significance': 'high'}
        }
    ]


@pytest.fixture
def performance_test_data():
    """性能测试数据"""
    generator = TestDataGenerator()
    
    return {
        'documents': [generator.generate_document_data() for _ in range(100)],
        'chunks': [generator.generate_chunk_data() for _ in range(500)],
        'entities': [generator.generate_entity_data() for _ in range(200)],
        'relations': [generator.generate_relation_data() for _ in range(300)]
    }


@pytest.fixture
def multilingual_test_data():
    """多语言测试数据"""
    return {
        'chinese_doc': {
            'title': '中文文档测试',
            'content': '这是一个中文文档的测试内容，包含了各种中文字符和标点符号。',
            'language': 'zh'
        },
        'english_doc': {
            'title': 'English Document Test',
            'content': 'This is a test content for English document with various characters.',
            'language': 'en'
        },
        'mixed_doc': {
            'title': 'Mixed Language Document 混合语言文档',
            'content': 'This document contains both English and 中文内容 for testing purposes.',
            'language': 'auto'
        }
    }


@pytest.fixture
def edge_case_test_data():
    """边界情况测试数据"""
    return {
        'empty_content': {
            'title': '空内容文档',
            'content': '',
            'file_name': 'empty.txt'
        },
        'large_content': {
            'title': '大内容文档',
            'content': 'A' * 10000,  # 10KB内容
            'file_name': 'large.txt'
        },
        'special_chars': {
            'title': '特殊字符文档 !@#$%^&*()_+-=[]{}|;:,.<>?',
            'content': '包含特殊字符的内容：!@#$%^&*()_+-=[]{}|;:,.<>?',
            'file_name': 'special_chars.txt'
        },
        'unicode_content': {
            'title': 'Unicode测试文档 🚀🎉🔥',
            'content': '包含Unicode字符：🚀🎉🔥 和各种符号 ∑∆∏∫',
            'file_name': 'unicode.txt'
        }
    }


@pytest.fixture
def cleanup_test_data():
    """测试数据清理fixture"""
    created_objects = []
    
    def add_for_cleanup(obj):
        created_objects.append(obj)
        return obj
    
    yield add_for_cleanup
    
    # 清理创建的对象
    # 注意：这个fixture需要在具体的测试中实现清理逻辑