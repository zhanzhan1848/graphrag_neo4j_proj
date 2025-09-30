# 测试策略和指南

## 概述

本文档详细说明了GraphRAG知识库系统的测试策略、测试框架和最佳实践。我们采用多层次的测试方法，确保系统的可靠性、性能和安全性。

## 测试架构

```mermaid
graph TB
    subgraph "测试层次"
        A[单元测试<br/>Unit Tests] --> B[集成测试<br/>Integration Tests]
        B --> C[端到端测试<br/>E2E Tests]
        C --> D[性能测试<br/>Performance Tests]
        D --> E[安全测试<br/>Security Tests]
    end
    
    subgraph "测试工具"
        F[pytest<br/>测试框架]
        G[factory-boy<br/>测试数据]
        H[httpx<br/>HTTP客户端]
        I[k6<br/>性能测试]
        J[bandit<br/>安全扫描]
    end
    
    subgraph "测试环境"
        K[本地开发<br/>Local Dev]
        L[CI/CD<br/>自动化]
        M[预发布<br/>Staging]
        N[生产监控<br/>Production]
    end
```

## 测试分类

### 1. 单元测试 (Unit Tests)

#### 测试范围
- 核心业务逻辑
- 数据模型验证
- 工具函数
- 服务层方法

#### 示例: 文档处理器测试

```python
# tests/unit/test_document_processor.py
"""
文档处理器单元测试
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4
from io import BytesIO

from src.core.document_processor import DocumentProcessor
from src.models.database.documents import Document
from src.services.file_service import FileService
from src.services.vector_service import VectorService

class TestDocumentProcessor:
    """文档处理器测试类"""
    
    @pytest.fixture
    def mock_file_service(self):
        """模拟文件服务"""
        service = Mock(spec=FileService)
        service.save_file = AsyncMock(return_value="/path/to/file.pdf")
        service.extract_text = AsyncMock(return_value="测试文档内容")
        return service
    
    @pytest.fixture
    def mock_vector_service(self):
        """模拟向量服务"""
        service = Mock(spec=VectorService)
        service.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])
        return service
    
    @pytest.fixture
    def document_processor(self, mock_file_service, mock_vector_service):
        """文档处理器实例"""
        return DocumentProcessor(
            file_service=mock_file_service,
            vector_service=mock_vector_service
        )
    
    @pytest.mark.asyncio
    async def test_process_pdf_document(self, document_processor):
        """测试PDF文档处理"""
        # 准备测试数据
        file_content = b"fake pdf content"
        file_obj = BytesIO(file_content)
        file_obj.name = "test.pdf"
        
        # 执行处理
        result = await document_processor.process_document(
            file=file_obj,
            title="测试文档",
            metadata={"author": "测试作者"}
        )
        
        # 验证结果
        assert result is not None
        assert result.title == "测试文档"
        assert result.file_type == "pdf"
        assert result.status == "processing"
        
        # 验证服务调用
        document_processor.file_service.save_file.assert_called_once()
        document_processor.file_service.extract_text.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_chunk_text(self, document_processor):
        """测试文本分块"""
        text = "这是一个测试文档。" * 100  # 创建长文本
        
        chunks = await document_processor.chunk_text(
            text=text,
            chunk_size=200,
            overlap=50
        )
        
        assert len(chunks) > 1
        assert all(len(chunk.content) <= 250 for chunk in chunks)  # 包含重叠
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1
    
    @pytest.mark.asyncio
    async def test_process_invalid_file_type(self, document_processor):
        """测试无效文件类型处理"""
        file_obj = BytesIO(b"content")
        file_obj.name = "test.exe"
        
        with pytest.raises(ValueError, match="不支持的文件类型"):
            await document_processor.process_document(file=file_obj)
    
    @pytest.mark.asyncio
    async def test_process_empty_file(self, document_processor):
        """测试空文件处理"""
        file_obj = BytesIO(b"")
        file_obj.name = "test.pdf"
        
        with pytest.raises(ValueError, match="文件为空"):
            await document_processor.process_document(file=file_obj)
```

#### 示例: 知识抽取器测试

```python
# tests/unit/test_knowledge_extractor.py
"""
知识抽取器单元测试
"""

import pytest
from unittest.mock import Mock, AsyncMock
import json

from src.core.knowledge_extractor import KnowledgeExtractor, ExtractedKnowledge
from src.models.database.chunks import Chunk
from src.services.llm_service import LLMService

class TestKnowledgeExtractor:
    """知识抽取器测试类"""
    
    @pytest.fixture
    def mock_llm_service(self):
        """模拟LLM服务"""
        service = Mock(spec=LLMService)
        
        # 模拟实体抽取响应
        entity_response = json.dumps({
            "entities": [
                {
                    "name": "张三",
                    "type": "PERSON",
                    "description": "一个人物",
                    "confidence": 0.95
                },
                {
                    "name": "北京大学",
                    "type": "ORGANIZATION",
                    "description": "教育机构",
                    "confidence": 0.90
                }
            ]
        })
        
        # 模拟关系抽取响应
        relation_response = json.dumps({
            "relations": [
                {
                    "subject": "张三",
                    "predicate": "就读于",
                    "object": "北京大学",
                    "description": "学习关系",
                    "confidence": 0.85
                }
            ]
        })
        
        service.generate_response = AsyncMock(
            side_effect=[entity_response, relation_response]
        )
        
        return service
    
    @pytest.fixture
    def knowledge_extractor(self, mock_llm_service):
        """知识抽取器实例"""
        return KnowledgeExtractor(
            llm_service=mock_llm_service,
            vector_service=Mock(),
            entity_repository=Mock(),
            relation_repository=Mock()
        )
    
    @pytest.mark.asyncio
    async def test_extract_entities(self, knowledge_extractor):
        """测试实体抽取"""
        text = "张三在北京大学学习计算机科学。"
        
        entities = await knowledge_extractor._extract_entities(text)
        
        assert len(entities) == 2
        assert entities[0]["name"] == "张三"
        assert entities[0]["type"] == "PERSON"
        assert entities[1]["name"] == "北京大学"
        assert entities[1]["type"] == "ORGANIZATION"
    
    @pytest.mark.asyncio
    async def test_extract_relations(self, knowledge_extractor):
        """测试关系抽取"""
        text = "张三在北京大学学习计算机科学。"
        entities = [
            {"name": "张三", "type": "PERSON"},
            {"name": "北京大学", "type": "ORGANIZATION"}
        ]
        
        relations = await knowledge_extractor._extract_relations(text, entities)
        
        assert len(relations) == 1
        assert relations[0]["subject"] == "张三"
        assert relations[0]["predicate"] == "就读于"
        assert relations[0]["object"] == "北京大学"
    
    @pytest.mark.asyncio
    async def test_extract_knowledge_integration(self, knowledge_extractor):
        """测试完整知识抽取流程"""
        chunk = Chunk(
            content="张三在北京大学学习计算机科学，他是一个优秀的学生。",
            chunk_index=0
        )
        
        knowledge = await knowledge_extractor.extract_knowledge(chunk)
        
        assert isinstance(knowledge, ExtractedKnowledge)
        assert len(knowledge.entities) >= 1
        assert len(knowledge.relations) >= 0
        assert knowledge.confidence > 0
```

### 2. 集成测试 (Integration Tests)

#### 测试范围
- API端点集成
- 数据库操作
- 外部服务集成
- 消息队列处理

#### 示例: API集成测试

```python
# tests/integration/test_document_api.py
"""
文档API集成测试
"""

import pytest
from httpx import AsyncClient
from fastapi import status
import tempfile
import os

from src.api.app import create_app
from tests.conftest import TestDatabase

class TestDocumentAPI:
    """文档API集成测试"""
    
    @pytest.fixture
    async def client(self):
        """测试客户端"""
        app = create_app()
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_upload_document_success(self, client, test_db):
        """测试成功上传文档"""
        # 创建测试文件
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"This is a test document content.")
            temp_file_path = f.name
        
        try:
            # 上传文档
            with open(temp_file_path, "rb") as f:
                response = await client.post(
                    "/api/v1/documents/upload",
                    files={"file": ("test.txt", f, "text/plain")},
                    data={"title": "测试文档", "async_processing": "false"}
                )
            
            # 验证响应
            assert response.status_code == status.HTTP_201_CREATED
            data = response.json()
            assert data["title"] == "测试文档"
            assert data["file_type"] == "txt"
            assert data["status"] in ["processing", "completed"]
            
            # 验证数据库记录
            document_id = data["id"]
            doc_response = await client.get(f"/api/v1/documents/{document_id}")
            assert doc_response.status_code == status.HTTP_200_OK
            
        finally:
            os.unlink(temp_file_path)
    
    @pytest.mark.asyncio
    async def test_upload_invalid_file_type(self, client):
        """测试上传无效文件类型"""
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
            f.write(b"fake executable")
            temp_file_path = f.name
        
        try:
            with open(temp_file_path, "rb") as f:
                response = await client.post(
                    "/api/v1/documents/upload",
                    files={"file": ("test.exe", f, "application/octet-stream")}
                )
            
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert "不支持的文件类型" in response.json()["detail"]
            
        finally:
            os.unlink(temp_file_path)
    
    @pytest.mark.asyncio
    async def test_list_documents(self, client, test_db, sample_documents):
        """测试文档列表查询"""
        response = await client.get("/api/v1/documents/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert len(data["items"]) <= data["total"]
    
    @pytest.mark.asyncio
    async def test_get_document_chunks(self, client, test_db, sample_document):
        """测试获取文档文本块"""
        document_id = sample_document.id
        
        response = await client.get(f"/api/v1/documents/{document_id}/chunks")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "chunks" in data
        assert "total" in data
        assert data["document_id"] == str(document_id)
```

#### 示例: 数据库集成测试

```python
# tests/integration/test_database_operations.py
"""
数据库操作集成测试
"""

import pytest
from uuid import uuid4

from src.repositories.document_repository import DocumentRepository
from src.repositories.entity_repository import EntityRepository
from src.repositories.relation_repository import RelationRepository
from src.models.database.documents import Document
from src.models.database.entities import Entity
from src.models.database.relations import Relation

class TestDatabaseOperations:
    """数据库操作集成测试"""
    
    @pytest.mark.asyncio
    async def test_document_crud_operations(self, test_db):
        """测试文档CRUD操作"""
        repo = DocumentRepository()
        
        # 创建文档
        document = Document(
            title="测试文档",
            content="这是测试内容",
            file_path="/path/to/test.pdf",
            file_type="pdf",
            status="completed"
        )
        
        created_doc = await repo.create(document)
        assert created_doc.id is not None
        assert created_doc.title == "测试文档"
        
        # 读取文档
        retrieved_doc = await repo.get_by_id(created_doc.id)
        assert retrieved_doc is not None
        assert retrieved_doc.title == "测试文档"
        
        # 更新文档
        updated_doc = await repo.update(
            created_doc.id,
            {"title": "更新后的文档"}
        )
        assert updated_doc.title == "更新后的文档"
        
        # 删除文档
        success = await repo.delete(created_doc.id)
        assert success is True
        
        # 验证删除
        deleted_doc = await repo.get_by_id(created_doc.id)
        assert deleted_doc is None
    
    @pytest.mark.asyncio
    async def test_entity_relationship_operations(self, test_db, sample_document):
        """测试实体关系操作"""
        entity_repo = EntityRepository()
        relation_repo = RelationRepository()
        
        # 创建实体
        person = Entity(
            name="张三",
            entity_type="PERSON",
            description="一个人物",
            confidence=0.95,
            source_document_id=sample_document.id
        )
        
        organization = Entity(
            name="北京大学",
            entity_type="ORGANIZATION",
            description="教育机构",
            confidence=0.90,
            source_document_id=sample_document.id
        )
        
        created_person = await entity_repo.create(person)
        created_org = await entity_repo.create(organization)
        
        # 创建关系
        relation = Relation(
            subject_id=created_person.id,
            predicate="就读于",
            object_id=created_org.id,
            description="学习关系",
            confidence=0.85,
            source_document_id=sample_document.id
        )
        
        created_relation = await relation_repo.create(relation)
        assert created_relation.id is not None
        
        # 查询实体的关系
        person_relations = await relation_repo.get_by_subject_id(created_person.id)
        assert len(person_relations) == 1
        assert person_relations[0].predicate == "就读于"
        
        # 查询关系的实体
        relation_with_entities = await relation_repo.get_with_entities(created_relation.id)
        assert relation_with_entities.subject.name == "张三"
        assert relation_with_entities.object.name == "北京大学"
```

### 3. 端到端测试 (E2E Tests)

#### 测试范围
- 完整业务流程
- 用户场景模拟
- 系统集成验证

#### 示例: 完整文档处理流程测试

```python
# tests/e2e/test_document_processing_workflow.py
"""
文档处理完整流程端到端测试
"""

import pytest
import asyncio
from httpx import AsyncClient
import tempfile
import time

class TestDocumentProcessingWorkflow:
    """文档处理工作流测试"""
    
    @pytest.mark.asyncio
    async def test_complete_document_processing_workflow(self, client):
        """测试完整的文档处理工作流"""
        
        # 1. 上传文档
        test_content = """
        张三是北京大学计算机科学系的教授。他专门研究人工智能和机器学习。
        张三发表了多篇关于深度学习的论文，并且指导了许多研究生。
        北京大学是中国顶尖的高等教育机构之一。
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".txt", delete=False) as f:
            f.write(test_content)
            temp_file_path = f.name
        
        try:
            # 上传文档
            with open(temp_file_path, "rb") as f:
                upload_response = await client.post(
                    "/api/v1/documents/upload",
                    files={"file": ("test_document.txt", f, "text/plain")},
                    data={"title": "测试文档", "async_processing": "true"}
                )
            
            assert upload_response.status_code == 201
            document_data = upload_response.json()
            document_id = document_data["id"]
            
            # 2. 等待文档处理完成
            max_wait_time = 60  # 最大等待60秒
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                status_response = await client.get(f"/api/v1/documents/{document_id}")
                status_data = status_response.json()
                
                if status_data["status"] == "completed":
                    break
                elif status_data["status"] == "failed":
                    pytest.fail(f"文档处理失败: {status_data}")
                
                await asyncio.sleep(2)
            else:
                pytest.fail("文档处理超时")
            
            # 3. 验证文档文本块
            chunks_response = await client.get(f"/api/v1/documents/{document_id}/chunks")
            assert chunks_response.status_code == 200
            chunks_data = chunks_response.json()
            assert len(chunks_data["chunks"]) > 0
            
            # 4. 验证实体抽取
            entities_response = await client.get("/api/v1/entities/", params={"document_id": document_id})
            assert entities_response.status_code == 200
            entities_data = entities_response.json()
            
            # 验证抽取的实体
            entity_names = [entity["name"] for entity in entities_data["items"]]
            assert "张三" in entity_names
            assert "北京大学" in entity_names
            
            # 5. 验证关系抽取
            relations_response = await client.get("/api/v1/relations/", params={"document_id": document_id})
            assert relations_response.status_code == 200
            relations_data = relations_response.json()
            assert len(relations_data["items"]) > 0
            
            # 6. 测试自然语言查询
            query_response = await client.post(
                "/api/v1/query/natural-language",
                json={
                    "question": "张三是谁？",
                    "context_limit": 3,
                    "include_sources": True
                }
            )
            
            assert query_response.status_code == 200
            query_data = query_response.json()
            assert "张三" in query_data["answer"]
            assert len(query_data["sources"]) > 0
            
            # 7. 测试向量搜索
            vector_search_response = await client.post(
                "/api/v1/query/vector-search",
                json={
                    "query": "人工智能研究",
                    "limit": 5,
                    "threshold": 0.7
                }
            )
            
            assert vector_search_response.status_code == 200
            search_data = vector_search_response.json()
            assert len(search_data["results"]) > 0
            
            # 8. 测试图查询
            graph_query_response = await client.post(
                "/api/v1/query/graph-query",
                json={
                    "cypher_query": "MATCH (p:Entity {name: '张三'})-[r]->(o:Entity) RETURN p, r, o",
                    "parameters": {}
                }
            )
            
            assert graph_query_response.status_code == 200
            graph_data = graph_query_response.json()
            assert len(graph_data["nodes"]) > 0
            
        finally:
            import os
            os.unlink(temp_file_path)
```

### 4. 性能测试

#### 使用k6进行负载测试

```javascript
// tests/performance/load_test.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

// 自定义指标
export let errorRate = new Rate('errors');

// 测试配置
export let options = {
    stages: [
        { duration: '2m', target: 10 }, // 预热
        { duration: '5m', target: 50 }, // 正常负载
        { duration: '2m', target: 100 }, // 峰值负载
        { duration: '5m', target: 50 }, // 回落
        { duration: '2m', target: 0 }, // 结束
    ],
    thresholds: {
        http_req_duration: ['p(95)<500'], // 95%的请求响应时间小于500ms
        http_req_failed: ['rate<0.1'], // 错误率小于10%
        errors: ['rate<0.1'],
    },
};

const BASE_URL = 'http://localhost:8000';

export default function () {
    // 测试文档列表API
    let listResponse = http.get(`${BASE_URL}/api/v1/documents/`);
    check(listResponse, {
        '文档列表状态码为200': (r) => r.status === 200,
        '文档列表响应时间<200ms': (r) => r.timings.duration < 200,
    }) || errorRate.add(1);

    // 测试向量搜索API
    let searchPayload = JSON.stringify({
        query: '人工智能',
        limit: 10,
        threshold: 0.7
    });

    let searchResponse = http.post(
        `${BASE_URL}/api/v1/query/vector-search`,
        searchPayload,
        {
            headers: { 'Content-Type': 'application/json' },
        }
    );

    check(searchResponse, {
        '向量搜索状态码为200': (r) => r.status === 200,
        '向量搜索响应时间<1000ms': (r) => r.timings.duration < 1000,
        '向量搜索返回结果': (r) => JSON.parse(r.body).results.length > 0,
    }) || errorRate.add(1);

    // 测试自然语言查询API
    let queryPayload = JSON.stringify({
        question: '什么是机器学习？',
        context_limit: 5,
        include_sources: true
    });

    let queryResponse = http.post(
        `${BASE_URL}/api/v1/query/natural-language`,
        queryPayload,
        {
            headers: { 'Content-Type': 'application/json' },
        }
    );

    check(queryResponse, {
        '自然语言查询状态码为200': (r) => r.status === 200,
        '自然语言查询响应时间<2000ms': (r) => r.timings.duration < 2000,
        '自然语言查询返回答案': (r) => JSON.parse(r.body).answer.length > 0,
    }) || errorRate.add(1);

    sleep(1);
}
```

### 5. 测试配置和工具

#### pytest配置 (`pytest.ini`)

```ini
[tool:pytest]
minversion = 6.0
addopts = 
    -ra
    -q
    --strict-markers
    --strict-config
    --cov=src
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-report=xml
    --cov-fail-under=80
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: 单元测试
    integration: 集成测试
    e2e: 端到端测试
    slow: 慢速测试
    performance: 性能测试
    security: 安全测试
asyncio_mode = auto
```

#### 测试工厂 (`tests/factories.py`)

```python
"""
测试数据工厂
使用factory-boy创建测试数据
"""

import factory
from factory import fuzzy
from datetime import datetime, timezone
from uuid import uuid4

from src.models.database.documents import Document
from src.models.database.entities import Entity
from src.models.database.relations import Relation
from src.models.database.chunks import Chunk

class DocumentFactory(factory.Factory):
    """文档工厂"""
    
    class Meta:
        model = Document
    
    id = factory.LazyFunction(uuid4)
    title = factory.Faker('sentence', nb_words=4, locale='zh_CN')
    content = factory.Faker('text', max_nb_chars=1000, locale='zh_CN')
    file_path = factory.Faker('file_path', depth=3)
    file_type = fuzzy.FuzzyChoice(['pdf', 'txt', 'docx', 'md'])
    file_size = fuzzy.FuzzyInteger(1000, 1000000)
    language = 'zh'
    status = fuzzy.FuzzyChoice(['processing', 'completed', 'failed'])
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    metadata = factory.Dict({
        'author': factory.Faker('name', locale='zh_CN'),
        'keywords': factory.List([
            factory.Faker('word', locale='zh_CN') for _ in range(3)
        ])
    })

class EntityFactory(factory.Factory):
    """实体工厂"""
    
    class Meta:
        model = Entity
    
    id = factory.LazyFunction(uuid4)
    name = factory.Faker('name', locale='zh_CN')
    entity_type = fuzzy.FuzzyChoice(['PERSON', 'ORGANIZATION', 'LOCATION', 'CONCEPT'])
    description = factory.Faker('sentence', locale='zh_CN')
    confidence = fuzzy.FuzzyFloat(0.7, 1.0)
    embedding = factory.List([fuzzy.FuzzyFloat(-1.0, 1.0) for _ in range(384)])
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))

class RelationFactory(factory.Factory):
    """关系工厂"""
    
    class Meta:
        model = Relation
    
    id = factory.LazyFunction(uuid4)
    predicate = fuzzy.FuzzyChoice(['工作于', '学习于', '位于', '属于', '研究'])
    description = factory.Faker('sentence', locale='zh_CN')
    confidence = fuzzy.FuzzyFloat(0.6, 1.0)
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))

class ChunkFactory(factory.Factory):
    """文本块工厂"""
    
    class Meta:
        model = Chunk
    
    id = factory.LazyFunction(uuid4)
    content = factory.Faker('text', max_nb_chars=500, locale='zh_CN')
    chunk_index = factory.Sequence(lambda n: n)
    start_char = factory.LazyAttribute(lambda obj: obj.chunk_index * 400)
    end_char = factory.LazyAttribute(lambda obj: obj.start_char + len(obj.content))
    token_count = factory.LazyAttribute(lambda obj: len(obj.content.split()))
    embedding = factory.List([fuzzy.FuzzyFloat(-1.0, 1.0) for _ in range(384)])
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
```

## 测试执行和CI/CD集成

### 本地测试执行

```bash
# 运行所有测试
pytest

# 运行特定类型的测试
pytest -m unit
pytest -m integration
pytest -m e2e

# 运行特定文件的测试
pytest tests/unit/test_document_processor.py

# 运行测试并生成覆盖率报告
pytest --cov=src --cov-report=html

# 运行性能测试
k6 run tests/performance/load_test.js
```

### CI/CD集成

```yaml
# .github/workflows/test.yml
name: 测试流水线

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_graphrag
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      neo4j:
        image: neo4j:5
        env:
          NEO4J_AUTH: neo4j/testpassword
        options: >-
          --health-cmd "cypher-shell -u neo4j -p testpassword 'RETURN 1'"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v4
    
    - name: 设置Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: 安装依赖
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: 运行单元测试
      run: pytest -m unit --cov=src --cov-report=xml
    
    - name: 运行集成测试
      run: pytest -m integration
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_graphrag
        REDIS_URL: redis://localhost:6379/0
        NEO4J_URI: bolt://localhost:7687
        NEO4J_USER: neo4j
        NEO4J_PASSWORD: testpassword
    
    - name: 上传覆盖率报告
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
```

## 测试最佳实践

### 1. 测试命名规范
- 测试文件: `test_<module_name>.py`
- 测试类: `Test<ClassName>`
- 测试方法: `test_<specific_behavior>`

### 2. 测试结构
- **Arrange**: 准备测试数据和环境
- **Act**: 执行被测试的操作
- **Assert**: 验证结果

### 3. 测试隔离
- 每个测试独立运行
- 使用fixtures管理测试数据
- 清理测试产生的副作用

### 4. 测试覆盖率
- 目标覆盖率: 80%以上
- 重点关注核心业务逻辑
- 定期审查未覆盖的代码

### 5. 性能测试指标
- 响应时间: P95 < 500ms
- 吞吐量: > 100 RPS
- 错误率: < 1%
- 资源使用: CPU < 80%, Memory < 80%

## 总结

这个测试策略文档提供了：

1. **完整的测试架构**: 从单元测试到端到端测试的全覆盖
2. **实用的测试示例**: 具体的测试代码和最佳实践
3. **自动化测试流程**: CI/CD集成和自动化执行
4. **性能测试方案**: 负载测试和性能监控
5. **测试工具配置**: pytest、k6等工具的配置和使用

通过遵循这些测试策略和实践，可以确保GraphRAG系统的质量、可靠性和性能。