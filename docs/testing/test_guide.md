# GraphRAG 测试编写指南

本指南提供了在 GraphRAG 项目中编写高质量测试的详细说明和最佳实践。

## 📚 目录

- [测试原则](#测试原则)
- [测试类型](#测试类型)
- [编写规范](#编写规范)
- [测试数据](#测试数据)
- [模拟和存根](#模拟和存根)
- [异步测试](#异步测试)
- [数据库测试](#数据库测试)
- [API测试](#api测试)
- [性能测试](#性能测试)
- [最佳实践](#最佳实践)

## 🎯 测试原则

### FIRST 原则

- **Fast (快速)**: 测试应该快速执行
- **Independent (独立)**: 测试之间不应相互依赖
- **Repeatable (可重复)**: 测试结果应该一致
- **Self-Validating (自验证)**: 测试应该有明确的通过/失败结果
- **Timely (及时)**: 测试应该及时编写

### 测试金字塔

```
    /\
   /  \     E2E Tests (少量)
  /____\    
 /      \   Integration Tests (适量)
/__________\ Unit Tests (大量)
```

## 🧪 测试类型

### 单元测试 (Unit Tests)

测试单个函数或类的功能，具有以下特征：
- 执行速度快（< 100ms）
- 无外部依赖
- 使用模拟对象
- 高代码覆盖率

**示例：测试实体模型**

```python
# tests/unit/test_models/test_database/test_entities.py

import pytest
from app.models.database.entities import Entity

class TestEntity:
    """实体模型单元测试"""
    
    def test_entity_creation_success(self):
        """测试成功创建实体"""
        # Arrange
        entity_data = {
            "name": "张三",
            "entity_type": "PERSON",
            "confidence": 0.95
        }
        
        # Act
        entity = Entity(**entity_data)
        
        # Assert
        assert entity.name == "张三"
        assert entity.entity_type == "PERSON"
        assert entity.confidence == 0.95
        assert entity.status == "active"  # 默认值
    
    def test_entity_invalid_confidence(self):
        """测试无效置信度"""
        with pytest.raises(ValueError, match="置信度必须在0-1之间"):
            Entity(name="测试", entity_type="PERSON", confidence=1.5)
    
    @pytest.mark.parametrize("entity_type", [
        "PERSON", "ORGANIZATION", "LOCATION", "CONCEPT"
    ])
    def test_valid_entity_types(self, entity_type):
        """测试有效的实体类型"""
        entity = Entity(name="测试", entity_type=entity_type)
        assert entity.entity_type == entity_type
```

### 集成测试 (Integration Tests)

测试组件间的交互，包括：
- 数据库操作
- API端点
- 服务集成
- 外部系统交互

**示例：测试文档服务**

```python
# tests/integration/test_services/test_document_service.py

import pytest
from app.services.document_service import DocumentService
from app.core.database import SessionLocal

@pytest.mark.integration
class TestDocumentService:
    """文档服务集成测试"""
    
    @pytest.fixture
    def db_session(self):
        """数据库会话fixture"""
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()
    
    @pytest.fixture
    def document_service(self, db_session):
        """文档服务fixture"""
        return DocumentService(db_session)
    
    async def test_create_and_process_document(self, document_service):
        """测试创建和处理文档的完整流程"""
        # Arrange
        file_content = b"这是一个测试文档的内容。"
        filename = "test_document.txt"
        
        # Act - 创建文档
        document = await document_service.create_document(
            filename=filename,
            content=file_content,
            file_type="text/plain"
        )
        
        # Assert - 验证文档创建
        assert document.id is not None
        assert document.title == "test_document.txt"
        assert document.status == "uploaded"
        
        # Act - 处理文档
        await document_service.process_document(document.id)
        
        # Assert - 验证处理结果
        processed_doc = await document_service.get_document(document.id)
        assert processed_doc.status == "processed"
        
        # 验证生成的chunks
        chunks = await document_service.get_document_chunks(document.id)
        assert len(chunks) > 0
        assert all(chunk.document_id == document.id for chunk in chunks)
```

## 📝 编写规范

### 命名规范

```python
# 文件命名
test_<module_name>.py

# 类命名
class Test<ClassName>:

# 方法命名
def test_<function_name>_<scenario>():
    """测试描述"""
```

### 测试结构 (AAA模式)

```python
def test_example():
    """测试示例"""
    # Arrange - 准备测试数据和环境
    user = User(name="张三", age=25)
    service = UserService()
    
    # Act - 执行被测试的操作
    result = service.validate_user(user)
    
    # Assert - 验证结果
    assert result.is_valid is True
    assert result.errors == []
```

### 文档字符串

```python
def test_entity_extraction_with_multiple_types():
    """
    测试从文本中提取多种类型的实体
    
    场景：
    - 输入包含人名、地名、机构名的文本
    - 验证能正确识别所有实体类型
    - 验证实体的置信度和位置信息
    
    预期结果：
    - 提取出所有实体
    - 实体类型正确
    - 置信度在合理范围内
    """
    # 测试实现...
```

## 🗃️ 测试数据

### 使用 Fixtures

```python
# tests/conftest.py

import pytest
from app.models.database import Document, Entity, Relation

@pytest.fixture
def sample_document():
    """创建示例文档"""
    return Document(
        title="测试文档",
        content="这是一个包含张三和北京大学的测试文档。",
        file_type="text/plain",
        status="uploaded"
    )

@pytest.fixture
def sample_entities():
    """创建示例实体列表"""
    return [
        Entity(name="张三", entity_type="PERSON", confidence=0.95),
        Entity(name="北京大学", entity_type="ORGANIZATION", confidence=0.90),
        Entity(name="北京", entity_type="LOCATION", confidence=0.85)
    ]

@pytest.fixture
def sample_relation():
    """创建示例关系"""
    return Relation(
        subject_id=1,
        predicate="就读于",
        object_id=2,
        confidence=0.88
    )
```

### 使用 Factory

```python
# tests/fixtures/factories.py

import factory
from app.models.database import Document, Entity

class DocumentFactory(factory.Factory):
    """文档工厂"""
    class Meta:
        model = Document
    
    title = factory.Sequence(lambda n: f"文档{n}")
    content = factory.Faker('text', locale='zh_CN')
    file_type = "text/plain"
    status = "uploaded"

class EntityFactory(factory.Factory):
    """实体工厂"""
    class Meta:
        model = Entity
    
    name = factory.Faker('name', locale='zh_CN')
    entity_type = factory.Iterator(["PERSON", "ORGANIZATION", "LOCATION"])
    confidence = factory.Faker('pyfloat', min_value=0.7, max_value=1.0)

# 使用示例
def test_with_factory():
    """使用工厂创建测试数据"""
    document = DocumentFactory()
    entities = EntityFactory.create_batch(5)
    
    assert len(entities) == 5
    assert all(0.7 <= e.confidence <= 1.0 for e in entities)
```

## 🎭 模拟和存根

### 使用 unittest.mock

```python
from unittest.mock import Mock, patch, MagicMock
import pytest

@patch('app.services.embedding_service.OpenAIClient')
def test_embedding_generation(mock_openai):
    """测试嵌入向量生成（模拟OpenAI API）"""
    # Arrange
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
    mock_openai.return_value.embeddings.create.return_value = mock_response
    
    service = EmbeddingService()
    
    # Act
    result = service.generate_embedding("测试文本")
    
    # Assert
    assert result == [0.1, 0.2, 0.3]
    mock_openai.return_value.embeddings.create.assert_called_once_with(
        model="text-embedding-ada-002",
        input="测试文本"
    )
```

### 使用 pytest-mock

```python
def test_file_upload(mocker):
    """测试文件上传（模拟文件系统）"""
    # Arrange
    mock_save = mocker.patch('app.services.file_service.save_file')
    mock_save.return_value = "/uploads/test_file.pdf"
    
    service = FileService()
    file_data = b"PDF content"
    
    # Act
    result = service.upload_file("test.pdf", file_data)
    
    # Assert
    assert result.path == "/uploads/test_file.pdf"
    mock_save.assert_called_once_with("test.pdf", file_data)
```

## ⚡ 异步测试

### 基本异步测试

```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_document_processing():
    """测试异步文档处理"""
    processor = DocumentProcessor()
    document = create_test_document()
    
    # 测试异步方法
    result = await processor.process_async(document)
    
    assert result.status == "completed"
    assert len(result.chunks) > 0
```

### 异步上下文管理器

```python
@pytest.mark.asyncio
async def test_async_database_transaction():
    """测试异步数据库事务"""
    async with AsyncSessionLocal() as session:
        # 创建测试数据
        document = Document(title="测试", content="内容")
        session.add(document)
        await session.commit()
        
        # 验证数据
        result = await session.get(Document, document.id)
        assert result.title == "测试"
```

### 异步模拟

```python
@pytest.mark.asyncio
async def test_async_api_call(mocker):
    """测试异步API调用"""
    # 模拟异步HTTP客户端
    mock_client = mocker.patch('httpx.AsyncClient')
    mock_response = Mock()
    mock_response.json.return_value = {"status": "success"}
    mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
    
    service = ExternalAPIService()
    result = await service.call_api("test_data")
    
    assert result["status"] == "success"
```

## 🗄️ 数据库测试

### 事务回滚

```python
@pytest.fixture
def db_session():
    """数据库会话fixture（自动回滚）"""
    connection = engine.connect()
    transaction = connection.begin()
    session = SessionLocal(bind=connection)
    
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
```

### 数据库状态验证

```python
@pytest.mark.database
def test_document_cascade_delete(db_session):
    """测试文档级联删除"""
    # Arrange
    document = Document(title="测试文档")
    chunk = Chunk(content="测试内容", document=document)
    entity = Entity(name="测试实体", document=document)
    
    db_session.add_all([document, chunk, entity])
    db_session.commit()
    
    doc_id = document.id
    
    # Act
    db_session.delete(document)
    db_session.commit()
    
    # Assert
    assert db_session.get(Document, doc_id) is None
    assert db_session.query(Chunk).filter_by(document_id=doc_id).count() == 0
    assert db_session.query(Entity).filter_by(document_id=doc_id).count() == 0
```

### Neo4j 测试

```python
@pytest.mark.neo4j
def test_graph_relationship_creation():
    """测试图数据库关系创建"""
    with neo4j_driver.session() as session:
        # 创建测试节点和关系
        result = session.run("""
            CREATE (p:Person {name: $person_name})
            CREATE (o:Organization {name: $org_name})
            CREATE (p)-[:WORKS_FOR]->(o)
            RETURN p, o
        """, person_name="张三", org_name="ABC公司")
        
        record = result.single()
        assert record["p"]["name"] == "张三"
        assert record["o"]["name"] == "ABC公司"
        
        # 验证关系
        relationship_result = session.run("""
            MATCH (p:Person {name: $name})-[r:WORKS_FOR]->(o:Organization)
            RETURN type(r) as rel_type
        """, name="张三")
        
        assert relationship_result.single()["rel_type"] == "WORKS_FOR"
```

## 🌐 API测试

### FastAPI 测试客户端

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_upload_document():
    """测试文档上传API"""
    # Arrange
    test_file = ("test.txt", b"测试文档内容", "text/plain")
    
    # Act
    response = client.post(
        "/api/documents/upload",
        files={"file": test_file}
    )
    
    # Assert
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["title"] == "test.txt"
    assert data["status"] == "uploaded"
```

### 异步API测试

```python
import httpx
import pytest

@pytest.mark.asyncio
async def test_async_api_endpoint():
    """测试异步API端点"""
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/documents/")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
```

### API认证测试

```python
def test_protected_endpoint_without_auth():
    """测试未认证访问受保护端点"""
    response = client.get("/api/admin/users")
    assert response.status_code == 401

def test_protected_endpoint_with_auth():
    """测试认证访问受保护端点"""
    headers = {"Authorization": "Bearer valid_token"}
    response = client.get("/api/admin/users", headers=headers)
    assert response.status_code == 200
```

## 🚀 性能测试

### 基准测试

```python
import time
import pytest

@pytest.mark.performance
def test_entity_extraction_performance():
    """测试实体抽取性能"""
    # Arrange
    large_text = "很长的文本内容..." * 1000
    extractor = EntityExtractor()
    
    # Act
    start_time = time.time()
    entities = extractor.extract(large_text)
    duration = time.time() - start_time
    
    # Assert
    assert duration < 5.0  # 5秒内完成
    assert len(entities) > 0
```

### 内存使用测试

```python
import psutil
import os

@pytest.mark.performance
def test_memory_usage():
    """测试内存使用情况"""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # 执行内存密集型操作
    large_data = process_large_dataset()
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # 验证内存增长在合理范围内（例如：< 100MB）
    assert memory_increase < 100 * 1024 * 1024
```

### 并发测试

```python
import concurrent.futures
import pytest

@pytest.mark.performance
def test_concurrent_document_processing():
    """测试并发文档处理"""
    documents = [create_test_document() for _ in range(10)]
    processor = DocumentProcessor()
    
    # 并发处理文档
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(processor.process, doc) 
            for doc in documents
        ]
        
        results = [future.result() for future in futures]
    
    # 验证所有文档都被正确处理
    assert len(results) == 10
    assert all(result.status == "completed" for result in results)
```

## ✅ 最佳实践

### 1. 测试组织

```python
class TestDocumentService:
    """文档服务测试类"""
    
    class TestCreation:
        """文档创建相关测试"""
        
        def test_create_success(self):
            """测试成功创建"""
            pass
        
        def test_create_invalid_data(self):
            """测试无效数据创建失败"""
            pass
    
    class TestProcessing:
        """文档处理相关测试"""
        
        def test_process_success(self):
            """测试成功处理"""
            pass
        
        def test_process_large_file(self):
            """测试大文件处理"""
            pass
```

### 2. 参数化测试

```python
@pytest.mark.parametrize("file_type,expected_processor", [
    ("application/pdf", PDFProcessor),
    ("text/plain", TextProcessor),
    ("application/docx", DocxProcessor),
])
def test_processor_selection(file_type, expected_processor):
    """测试处理器选择"""
    factory = ProcessorFactory()
    processor = factory.get_processor(file_type)
    assert isinstance(processor, expected_processor)
```

### 3. 测试标记使用

```python
@pytest.mark.unit
def test_fast_unit_test():
    """快速单元测试"""
    pass

@pytest.mark.integration
@pytest.mark.slow
def test_slow_integration_test():
    """慢速集成测试"""
    pass

@pytest.mark.skip(reason="功能尚未实现")
def test_future_feature():
    """未来功能测试"""
    pass

@pytest.mark.xfail(reason="已知问题")
def test_known_issue():
    """已知问题测试"""
    pass
```

### 4. 错误测试

```python
def test_invalid_input_raises_exception():
    """测试无效输入抛出异常"""
    service = DocumentService()
    
    with pytest.raises(ValueError, match="文件不能为空"):
        service.process_document(None)

def test_network_error_handling():
    """测试网络错误处理"""
    with patch('requests.get', side_effect=requests.ConnectionError):
        service = ExternalAPIService()
        
        with pytest.raises(ServiceUnavailableError):
            service.fetch_data()
```

### 5. 清理和资源管理

```python
@pytest.fixture
def temp_file():
    """临时文件fixture"""
    import tempfile
    import os
    
    fd, path = tempfile.mkstemp()
    try:
        yield path
    finally:
        os.close(fd)
        os.unlink(path)

@pytest.fixture
def mock_redis():
    """模拟Redis fixture"""
    import fakeredis
    
    redis_client = fakeredis.FakeRedis()
    with patch('app.core.redis_client.redis_client', redis_client):
        yield redis_client
        redis_client.flushall()
```

### 6. 测试数据隔离

```python
@pytest.fixture(autouse=True)
def isolate_test_data():
    """自动隔离测试数据"""
    # 测试前设置
    test_db_name = f"test_db_{uuid.uuid4().hex[:8]}"
    create_test_database(test_db_name)
    
    yield
    
    # 测试后清理
    drop_test_database(test_db_name)
```

## 🔍 调试技巧

### 1. 使用 pytest 调试器

```bash
# 失败时进入调试器
pytest --pdb

# 使用 IPython 调试器
pytest --pdbcls=IPython.terminal.debugger:Pdb

# 在特定测试中设置断点
def test_debug_example():
    result = some_function()
    import pdb; pdb.set_trace()  # 设置断点
    assert result == expected
```

### 2. 详细输出

```bash
# 显示详细输出
pytest -v -s

# 显示局部变量
pytest --tb=long

# 显示最短回溯
pytest --tb=short
```

### 3. 日志调试

```python
import logging

def test_with_logging(caplog):
    """使用日志进行调试"""
    with caplog.at_level(logging.DEBUG):
        result = complex_function()
        
    # 检查日志输出
    assert "处理开始" in caplog.text
    assert "处理完成" in caplog.text
```

## 📊 测试报告

### 覆盖率报告

```bash
# 生成覆盖率报告
pytest --cov=app --cov-report=html --cov-report=term

# 查看未覆盖的行
pytest --cov=app --cov-report=term-missing

# 设置覆盖率阈值
pytest --cov=app --cov-fail-under=80
```

### HTML测试报告

```bash
# 生成HTML测试报告
pytest --html=report.html --self-contained-html
```

### JUnit XML报告

```bash
# 生成JUnit XML报告（用于CI/CD）
pytest --junitxml=test-results.xml
```

---

**最后更新**: 2024年
**维护者**: GraphRAG Team