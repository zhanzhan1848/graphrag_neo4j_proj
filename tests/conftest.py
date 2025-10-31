#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 测试配置
================

本模块提供 pytest 配置和共享测试夹具。

夹具说明：
- app: FastAPI 应用实例
- client: 测试客户端
- db_session: 数据库会话
- test_user: 测试用户
- sample_document: 示例文档
- mock_services: 模拟服务

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import os
import sys
import pytest
import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Generator, AsyncGenerator, Dict, Any
from unittest.mock import Mock, AsyncMock, patch

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from app.main import create_application
from app.core.config import settings
from app.core.database import get_db, Base
from app.models.database.documents import Document
from app.models.database.chunks import Chunk
from app.models.database.entities import Entity
from app.models.database.relations import Relation
from app.models.schemas.documents import DocumentCreate, DocumentStatus, DocumentType


# 测试数据库配置
TEST_DATABASE_URL = "sqlite:///./test.db"

# 创建测试数据库引擎
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

# 创建测试数据库会话
TestingSessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=test_engine
)


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def db_session() -> Generator[Session, None, None]:
    """
    创建测试数据库会话
    
    每个测试函数都会获得一个独立的数据库会话，
    测试结束后会回滚所有更改。
    """
    # 创建所有表
    Base.metadata.create_all(bind=test_engine)
    
    # 创建会话
    session = TestingSessionLocal()
    
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        # 清理表
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def app():
    """创建 FastAPI 应用实例"""
    # 设置测试环境变量
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["DEBUG"] = "true"
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    
    # 创建应用
    app = create_application()
    
    return app


@pytest.fixture(scope="function")
def client(app, db_session):
    """创建测试客户端"""
    
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_document_data() -> Dict[str, Any]:
    """示例文档数据"""
    return {
        "title": "测试文档",
        "description": "这是一个测试文档",
        "content": "这是测试文档的内容。包含一些示例文本用于测试。",
        "file_name": "test_document.txt",
        "file_size": 1024,
        "mime_type": "text/plain",
        "document_type": DocumentType.TXT,
        "language": "zh",
        "author": "测试作者",
        "tags": ["测试", "示例"],
        "categories": ["测试分类"],
        "is_public": True,
        "auto_process": True,
        "metadata": {
            "source": "test",
            "version": "1.0"
        }
    }


@pytest.fixture
def sample_document(db_session: Session, sample_document_data: Dict[str, Any]) -> Document:
    """创建示例文档"""
    document = Document(
        id=uuid.uuid4(),
        **sample_document_data,
        status=DocumentStatus.UPLOADED
    )
    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)
    return document


@pytest.fixture
def sample_chunk_data() -> Dict[str, Any]:
    """示例文本块数据"""
    return {
        "content": "这是一个测试文本块的内容。",
        "chunk_index": 0,
        "start_char": 0,
        "end_char": 20,
        "token_count": 10,
        "metadata": {
            "page": 1,
            "section": "introduction"
        }
    }


@pytest.fixture
def sample_entity_data() -> Dict[str, Any]:
    """示例实体数据"""
    return {
        "name": "测试实体",
        "entity_type": "PERSON",
        "description": "这是一个测试实体",
        "properties": {
            "confidence": 0.95,
            "source": "test"
        }
    }


@pytest.fixture
def sample_relation_data() -> Dict[str, Any]:
    """示例关系数据"""
    return {
        "relation_type": "WORKS_AT",
        "description": "工作关系",
        "properties": {
            "confidence": 0.9,
            "start_date": "2024-01-01"
        }
    }


@pytest.fixture
def temp_file():
    """创建临时文件"""
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as f:
        f.write("这是一个测试文件的内容。\n包含多行文本用于测试。")
        temp_path = f.name
    
    yield temp_path
    
    # 清理临时文件
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_azure_openai_service():
    """模拟 Azure OpenAI 服务"""
    mock_service = AsyncMock()
    mock_service.generate_text.return_value = "这是生成的文本"
    mock_service.generate_embeddings.return_value = [0.1] * 1536
    mock_service.extract_entities.return_value = [
        {"name": "测试实体", "type": "PERSON", "confidence": 0.95}
    ]
    mock_service.extract_relations.return_value = [
        {"source": "实体1", "target": "实体2", "type": "WORKS_AT", "confidence": 0.9}
    ]
    return mock_service


@pytest.fixture
def mock_file_storage_service():
    """模拟文件存储服务"""
    mock_service = AsyncMock()
    mock_service.save_file.return_value = "/test/path/file.txt"
    mock_service.get_file.return_value = b"test file content"
    mock_service.delete_file.return_value = True
    mock_service.file_exists.return_value = True
    return mock_service


@pytest.fixture
def mock_graph_service():
    """模拟图数据库服务"""
    mock_service = AsyncMock()
    mock_service.create_node.return_value = {"id": "test_node_id"}
    mock_service.create_relationship.return_value = {"id": "test_rel_id"}
    mock_service.query.return_value = [{"node": {"name": "测试节点"}}]
    return mock_service


@pytest.fixture
def mock_embedding_service():
    """模拟嵌入服务"""
    mock_service = AsyncMock()
    mock_service.generate_embeddings.return_value = [0.1] * 1536
    mock_service.search_similar.return_value = [
        {"id": "doc1", "score": 0.95, "content": "相似内容1"},
        {"id": "doc2", "score": 0.85, "content": "相似内容2"}
    ]
    return mock_service


# 测试标记
pytest_plugins = []

# 测试配置
def pytest_configure(config):
    """pytest 配置"""
    config.addinivalue_line(
        "markers", "unit: 标记单元测试"
    )
    config.addinivalue_line(
        "markers", "integration: 标记集成测试"
    )
    config.addinivalue_line(
        "markers", "api: 标记 API 测试"
    )
    config.addinivalue_line(
        "markers", "database: 标记数据库测试"
    )
    config.addinivalue_line(
        "markers", "slow: 标记慢速测试"
    )


# 测试收集配置
def pytest_collection_modifyitems(config, items):
    """修改测试收集"""
    for item in items:
        # 为所有测试添加适当的标记
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        if "api" in str(item.fspath):
            item.add_marker(pytest.mark.api)
        
        if "database" in str(item.fspath) or "db" in str(item.fspath):
            item.add_marker(pytest.mark.database)