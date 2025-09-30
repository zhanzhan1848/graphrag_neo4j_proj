# 项目结构说明

## 概述

本文档详细描述了GraphRAG知识库系统的项目结构、代码组织和各个模块的职责。项目采用模块化设计，便于开发、测试和维护。

## 项目目录结构

```
GraphRAG_NEO_IMG/
├── README.md                           # 项目主说明文档
├── develop_doc.md                      # 开发手册
├── requirements.txt                    # Python依赖
├── pyproject.toml                      # 项目配置
├── .env.example                        # 环境变量示例
├── .gitignore                          # Git忽略文件
├── .pre-commit-config.yaml            # 预提交钩子配置
│
├── docs/                               # 文档目录
│   ├── README.md                       # 文档总览
│   ├── architecture/                   # 架构文档
│   │   ├── system-overview.md          # 系统架构概览
│   │   └── workflow-diagrams.md        # 工作流程图
│   ├── phase0/                         # 阶段0文档
│   │   └── README.md                   # 基础架构搭建
│   ├── phase1/                         # 阶段1文档
│   │   └── README.md                   # MVP功能实现
│   ├── phase2/                         # 阶段2文档
│   │   └── README.md                   # 多模态扩展
│   ├── phase3/                         # 阶段3文档
│   │   └── README.md                   # 企业级部署
│   ├── cicd/                           # CI/CD文档
│   │   └── README.md                   # 持续集成部署
│   └── project-structure/              # 项目结构文档
│       └── README.md                   # 本文档
│
├── src/                                # 源代码目录
│   ├── __init__.py
│   ├── main.py                         # 应用入口
│   ├── config/                         # 配置模块
│   │   ├── __init__.py
│   │   ├── settings.py                 # 应用配置
│   │   ├── database.py                 # 数据库配置
│   │   └── logging.py                  # 日志配置
│   │
│   ├── api/                            # API接口层
│   │   ├── __init__.py
│   │   ├── app.py                      # FastAPI应用
│   │   ├── dependencies.py             # 依赖注入
│   │   ├── middleware.py               # 中间件
│   │   └── routes/                     # 路由模块
│   │       ├── __init__.py
│   │       ├── documents.py            # 文档管理API
│   │       ├── query.py                # 查询API
│   │       ├── entities.py             # 实体API
│   │       ├── relations.py            # 关系API
│   │       ├── images.py               # 图像API
│   │       └── health.py               # 健康检查API
│   │
│   ├── core/                           # 核心业务逻辑
│   │   ├── __init__.py
│   │   ├── document_processor.py       # 文档处理器
│   │   ├── knowledge_extractor.py      # 知识抽取器
│   │   ├── entity_linker.py            # 实体链接器
│   │   ├── relation_extractor.py       # 关系抽取器
│   │   ├── image_processor.py          # 图像处理器
│   │   └── query_engine.py             # 查询引擎
│   │
│   ├── models/                         # 数据模型
│   │   ├── __init__.py
│   │   ├── database/                   # 数据库模型
│   │   │   ├── __init__.py
│   │   │   ├── base.py                 # 基础模型
│   │   │   ├── documents.py            # 文档模型
│   │   │   ├── chunks.py               # 文本块模型
│   │   │   ├── entities.py             # 实体模型
│   │   │   ├── relations.py            # 关系模型
│   │   │   └── images.py               # 图像模型
│   │   │
│   │   ├── schemas/                    # API模式
│   │   │   ├── __init__.py
│   │   │   ├── documents.py            # 文档模式
│   │   │   ├── query.py                # 查询模式
│   │   │   ├── entities.py             # 实体模式
│   │   │   ├── relations.py            # 关系模式
│   │   │   └── images.py               # 图像模式
│   │   │
│   │   └── graph/                      # 图数据库模型
│   │       ├── __init__.py
│   │       ├── nodes.py                # 节点模型
│   │       ├── relationships.py        # 关系模型
│   │       └── queries.py              # 查询模型
│   │
│   ├── services/                       # 服务层
│   │   ├── __init__.py
│   │   ├── document_service.py         # 文档服务
│   │   ├── knowledge_service.py        # 知识服务
│   │   ├── query_service.py            # 查询服务
│   │   ├── entity_service.py           # 实体服务
│   │   ├── relation_service.py         # 关系服务
│   │   ├── image_service.py            # 图像服务
│   │   └── vector_service.py           # 向量服务
│   │
│   ├── repositories/                   # 数据访问层
│   │   ├── __init__.py
│   │   ├── base.py                     # 基础仓库
│   │   ├── document_repository.py      # 文档仓库
│   │   ├── entity_repository.py        # 实体仓库
│   │   ├── relation_repository.py      # 关系仓库
│   │   ├── image_repository.py         # 图像仓库
│   │   └── graph_repository.py         # 图数据库仓库
│   │
│   ├── utils/                          # 工具模块
│   │   ├── __init__.py
│   │   ├── text_processing.py          # 文本处理工具
│   │   ├── file_utils.py               # 文件处理工具
│   │   ├── embedding_utils.py          # 嵌入工具
│   │   ├── graph_utils.py              # 图处理工具
│   │   ├── image_utils.py              # 图像处理工具
│   │   └── validation.py               # 验证工具
│   │
│   └── workers/                        # 异步任务
│       ├── __init__.py
│       ├── celery_app.py               # Celery应用
│       ├── document_tasks.py           # 文档处理任务
│       ├── knowledge_tasks.py          # 知识抽取任务
│       ├── image_tasks.py              # 图像处理任务
│       └── maintenance_tasks.py        # 维护任务
│
├── tests/                              # 测试目录
│   ├── __init__.py
│   ├── conftest.py                     # 测试配置
│   ├── unit/                           # 单元测试
│   │   ├── __init__.py
│   │   ├── test_document_processor.py
│   │   ├── test_knowledge_extractor.py
│   │   ├── test_entity_linker.py
│   │   ├── test_query_engine.py
│   │   └── test_image_processor.py
│   │
│   ├── integration/                    # 集成测试
│   │   ├── __init__.py
│   │   ├── test_api_endpoints.py
│   │   ├── test_database_operations.py
│   │   ├── test_graph_operations.py
│   │   └── test_vector_operations.py
│   │
│   ├── e2e/                           # 端到端测试
│   │   ├── __init__.py
│   │   ├── test_document_workflow.py
│   │   ├── test_query_workflow.py
│   │   └── test_multimodal_workflow.py
│   │
│   ├── performance/                    # 性能测试
│   │   ├── __init__.py
│   │   ├── test_load.py
│   │   └── test_stress.py
│   │
│   └── fixtures/                       # 测试数据
│       ├── documents/
│       ├── images/
│       └── data/
│
├── scripts/                            # 脚本目录
│   ├── setup.sh                        # 环境设置脚本
│   ├── migrate.py                      # 数据迁移脚本
│   ├── seed_data.py                    # 种子数据脚本
│   ├── backup.py                       # 备份脚本
│   └── performance_test.py             # 性能测试脚本
│
├── docker/                             # Docker配置
│   ├── Dockerfile.api                  # API服务镜像
│   ├── Dockerfile.worker               # Worker服务镜像
│   ├── Dockerfile.web                  # Web界面镜像
│   └── docker-compose.yml              # 开发环境编排
│
├── k8s/                               # Kubernetes配置
│   ├── namespace.yaml                  # 命名空间
│   ├── configmap.yaml                  # 配置映射
│   ├── secret.yaml                     # 密钥
│   ├── api-deployment.yaml             # API部署
│   ├── worker-deployment.yaml          # Worker部署
│   ├── service.yaml                    # 服务
│   └── ingress.yaml                    # 入口
│
├── helm/                              # Helm Charts
│   └── graphrag/
│       ├── Chart.yaml                  # Chart定义
│       ├── values.yaml                 # 默认值
│       ├── values-dev.yaml             # 开发环境值
│       ├── values-staging.yaml         # 测试环境值
│       ├── values-production.yaml      # 生产环境值
│       └── templates/                  # 模板文件
│           ├── deployment.yaml
│           ├── service.yaml
│           ├── ingress.yaml
│           └── configmap.yaml
│
├── web/                               # Web前端 (可选)
│   ├── package.json                    # 前端依赖
│   ├── src/                           # 前端源码
│   ├── public/                        # 静态资源
│   └── dist/                          # 构建输出
│
├── migrations/                         # 数据库迁移
│   ├── alembic.ini                     # Alembic配置
│   ├── env.py                         # 迁移环境
│   └── versions/                      # 迁移版本
│
└── .github/                           # GitHub配置
    ├── workflows/                      # GitHub Actions
    │   ├── ci-cd.yml                  # CI/CD工作流
    │   ├── security.yml               # 安全扫描
    │   └── release.yml                # 发布工作流
    └── ISSUE_TEMPLATE/                # Issue模板
        ├── bug_report.md
        └── feature_request.md
```

## 核心模块说明

### 1. API层 (`src/api/`)

负责处理HTTP请求和响应，提供RESTful API接口。

```python
# src/api/app.py - FastAPI应用主文件
"""
GraphRAG API应用
提供文档管理、知识查询、实体关系等API接口
"""

from fastapi import FastAPI, Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from src.api.middleware import LoggingMiddleware, AuthMiddleware
from src.api.routes import (
    documents, query, entities, relations, images, health
)
from src.config.settings import settings

def create_app() -> FastAPI:
    """创建FastAPI应用实例"""
    app = FastAPI(
        title="GraphRAG Knowledge Base API",
        description="基于图数据库的知识库系统API",
        version="1.0.0",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None
    )
    
    # 添加中间件
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(AuthMiddleware)
    
    # 注册路由
    app.include_router(health.router, prefix="/health", tags=["健康检查"])
    app.include_router(documents.router, prefix="/api/v1/documents", tags=["文档管理"])
    app.include_router(query.router, prefix="/api/v1/query", tags=["知识查询"])
    app.include_router(entities.router, prefix="/api/v1/entities", tags=["实体管理"])
    app.include_router(relations.router, prefix="/api/v1/relations", tags=["关系管理"])
    app.include_router(images.router, prefix="/api/v1/images", tags=["图像处理"])
    
    return app

app = create_app()
```

### 2. 核心业务层 (`src/core/`)

包含主要的业务逻辑处理模块。

```python
# src/core/document_processor.py - 文档处理器
"""
文档处理核心模块
负责文档解析、文本分块、向量化等功能
"""

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.models.database.documents import Document
from src.models.database.chunks import Chunk
from src.utils.text_processing import TextSplitter
from src.utils.file_utils import FileHandler
from src.services.vector_service import VectorService

class DocumentProcessor:
    """文档处理器"""
    
    def __init__(
        self,
        text_splitter: TextSplitter,
        file_handler: FileHandler,
        vector_service: VectorService
    ):
        self.text_splitter = text_splitter
        self.file_handler = file_handler
        self.vector_service = vector_service
    
    async def process_document(
        self, 
        file_path: Path, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """
        处理单个文档
        
        Args:
            file_path: 文档文件路径
            metadata: 文档元数据
            
        Returns:
            Document: 处理后的文档对象
        """
        # 1. 提取文本内容
        text_content = await self.file_handler.extract_text(file_path)
        
        # 2. 创建文档记录
        document = Document(
            title=file_path.stem,
            content=text_content,
            file_path=str(file_path),
            file_type=file_path.suffix.lower(),
            metadata=metadata or {}
        )
        
        # 3. 文本分块
        chunks = await self.text_splitter.split_text(
            text_content, 
            document_id=document.id
        )
        
        # 4. 生成向量嵌入
        await self._generate_embeddings(chunks)
        
        # 5. 存储文档和分块
        await self._store_document_and_chunks(document, chunks)
        
        return document
    
    async def _generate_embeddings(self, chunks: List[Chunk]) -> None:
        """为文本块生成向量嵌入"""
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.vector_service.embed_texts(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
    
    async def _store_document_and_chunks(
        self, 
        document: Document, 
        chunks: List[Chunk]
    ) -> None:
        """存储文档和文本块到数据库"""
        # 实现数据库存储逻辑
        pass
```

### 3. 数据模型层 (`src/models/`)

定义数据结构和数据库模型。

```python
# src/models/database/base.py - 基础数据库模型
"""
数据库模型基类
提供通用的字段和方法
"""

import uuid
from datetime import datetime
from typing import Any, Dict

from sqlalchemy import Column, String, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import declared_attr

Base = declarative_base()

class BaseModel(Base):
    """数据库模型基类"""
    __abstract__ = True
    
    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower()
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON, default=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """从字典更新模型属性"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
```

```python
# src/models/database/documents.py - 文档模型
"""
文档数据库模型
"""

from sqlalchemy import Column, String, Text, Integer, Float
from sqlalchemy.orm import relationship

from .base import BaseModel

class Document(BaseModel):
    """文档模型"""
    __tablename__ = "documents"
    
    title = Column(String(500), nullable=False, index=True)
    content = Column(Text)
    file_path = Column(String(1000), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer)
    language = Column(String(10), default="zh")
    status = Column(String(20), default="processing", index=True)
    
    # 关联关系
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    images = relationship("Image", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Document(id={self.id}, title='{self.title}')>"
```

### 4. 服务层 (`src/services/`)

提供业务逻辑的服务接口。

```python
# src/services/document_service.py - 文档服务
"""
文档管理服务
提供文档的增删改查和处理功能
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from pathlib import Path

from src.models.database.documents import Document
from src.repositories.document_repository import DocumentRepository
from src.core.document_processor import DocumentProcessor
from src.workers.document_tasks import process_document_task

class DocumentService:
    """文档服务"""
    
    def __init__(
        self,
        document_repository: DocumentRepository,
        document_processor: DocumentProcessor
    ):
        self.repository = document_repository
        self.processor = document_processor
    
    async def upload_document(
        self,
        file_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
        async_processing: bool = True
    ) -> Document:
        """
        上传并处理文档
        
        Args:
            file_path: 文档文件路径
            metadata: 文档元数据
            async_processing: 是否异步处理
            
        Returns:
            Document: 文档对象
        """
        if async_processing:
            # 异步处理
            document = await self._create_document_record(file_path, metadata)
            process_document_task.delay(str(document.id), str(file_path))
            return document
        else:
            # 同步处理
            return await self.processor.process_document(file_path, metadata)
    
    async def get_document(self, document_id: UUID) -> Optional[Document]:
        """获取文档"""
        return await self.repository.get_by_id(document_id)
    
    async def list_documents(
        self,
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None
    ) -> List[Document]:
        """获取文档列表"""
        return await self.repository.list_documents(
            skip=skip, 
            limit=limit, 
            status=status
        )
    
    async def delete_document(self, document_id: UUID) -> bool:
        """删除文档"""
        return await self.repository.delete(document_id)
    
    async def _create_document_record(
        self,
        file_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """创建文档记录"""
        document = Document(
            title=file_path.stem,
            file_path=str(file_path),
            file_type=file_path.suffix.lower(),
            file_size=file_path.stat().st_size,
            status="pending",
            metadata=metadata or {}
        )
        return await self.repository.create(document)
```

### 5. 数据访问层 (`src/repositories/`)

封装数据库操作逻辑。

```python
# src/repositories/base.py - 基础仓库
"""
数据访问层基类
提供通用的CRUD操作
"""

from typing import List, Optional, Type, TypeVar, Generic
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload

from src.models.database.base import BaseModel

ModelType = TypeVar("ModelType", bound=BaseModel)

class BaseRepository(Generic[ModelType]):
    """基础仓库类"""
    
    def __init__(self, model: Type[ModelType], session: AsyncSession):
        self.model = model
        self.session = session
    
    async def create(self, obj: ModelType) -> ModelType:
        """创建对象"""
        self.session.add(obj)
        await self.session.commit()
        await self.session.refresh(obj)
        return obj
    
    async def get_by_id(self, obj_id: UUID) -> Optional[ModelType]:
        """根据ID获取对象"""
        result = await self.session.execute(
            select(self.model).where(self.model.id == obj_id)
        )
        return result.scalar_one_or_none()
    
    async def list_all(
        self,
        skip: int = 0,
        limit: int = 100
    ) -> List[ModelType]:
        """获取所有对象"""
        result = await self.session.execute(
            select(self.model).offset(skip).limit(limit)
        )
        return result.scalars().all()
    
    async def update(self, obj_id: UUID, **kwargs) -> Optional[ModelType]:
        """更新对象"""
        await self.session.execute(
            update(self.model)
            .where(self.model.id == obj_id)
            .values(**kwargs)
        )
        await self.session.commit()
        return await self.get_by_id(obj_id)
    
    async def delete(self, obj_id: UUID) -> bool:
        """删除对象"""
        result = await self.session.execute(
            delete(self.model).where(self.model.id == obj_id)
        )
        await self.session.commit()
        return result.rowcount > 0
```

### 6. 异步任务层 (`src/workers/`)

处理后台异步任务。

```python
# src/workers/celery_app.py - Celery应用配置
"""
Celery异步任务应用配置
"""

from celery import Celery
from src.config.settings import settings

# 创建Celery应用
celery_app = Celery(
    "graphrag_workers",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "src.workers.document_tasks",
        "src.workers.knowledge_tasks",
        "src.workers.image_tasks",
        "src.workers.maintenance_tasks"
    ]
)

# Celery配置
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30分钟
    task_soft_time_limit=25 * 60,  # 25分钟
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
)

# 任务路由
celery_app.conf.task_routes = {
    "src.workers.document_tasks.*": {"queue": "documents"},
    "src.workers.knowledge_tasks.*": {"queue": "knowledge"},
    "src.workers.image_tasks.*": {"queue": "images"},
    "src.workers.maintenance_tasks.*": {"queue": "maintenance"},
}
```

```python
# src/workers/document_tasks.py - 文档处理任务
"""
文档处理异步任务
"""

import asyncio
from uuid import UUID
from pathlib import Path

from src.workers.celery_app import celery_app
from src.core.document_processor import DocumentProcessor
from src.config.database import get_async_session

@celery_app.task(bind=True, name="process_document")
def process_document_task(self, document_id: str, file_path: str):
    """
    异步处理文档任务
    
    Args:
        document_id: 文档ID
        file_path: 文件路径
    """
    try:
        # 运行异步处理逻辑
        asyncio.run(_process_document_async(UUID(document_id), Path(file_path)))
        
        return {
            "status": "success",
            "document_id": document_id,
            "message": "文档处理完成"
        }
    except Exception as exc:
        # 记录错误并重试
        self.retry(exc=exc, countdown=60, max_retries=3)

async def _process_document_async(document_id: UUID, file_path: Path):
    """异步处理文档的内部方法"""
    async with get_async_session() as session:
        # 初始化处理器
        processor = DocumentProcessor(...)
        
        # 处理文档
        await processor.process_document(file_path)
        
        # 更新文档状态
        await session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(status="completed")
        )
        await session.commit()
```

## 配置管理

### 应用配置 (`src/config/settings.py`)

```python
"""
应用配置管理
支持环境变量和配置文件
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, validator

class Settings(BaseSettings):
    """应用配置类"""
    
    # 基础配置
    APP_NAME: str = "GraphRAG Knowledge Base"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    SECRET_KEY: str
    
    # 数据库配置
    DATABASE_URL: str
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 30
    
    # Redis配置
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Neo4j配置
    NEO4J_URI: str
    NEO4J_USER: str
    NEO4J_PASSWORD: str
    
    # Weaviate配置
    WEAVIATE_URL: str = "http://localhost:8080"
    WEAVIATE_API_KEY: Optional[str] = None
    
    # MinIO配置
    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_BUCKET: str = "graphrag"
    
    # Celery配置
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    
    # API配置
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # 文件处理配置
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_FILE_TYPES: List[str] = [".pdf", ".txt", ".md", ".docx"]
    UPLOAD_DIR: str = "/tmp/uploads"
    
    # AI模型配置
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "gpt-3.5-turbo"
    OPENAI_API_KEY: Optional[str] = None
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @validator("ALLOWED_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# 全局配置实例
settings = Settings()
```

## 测试结构

### 测试配置 (`tests/conftest.py`)

```python
"""
测试配置和夹具
"""

import pytest
import asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.config.settings import Settings
from src.models.database.base import Base
from src.config.database import get_async_session

# 测试配置
test_settings = Settings(
    DATABASE_URL="postgresql+asyncpg://test:test@localhost:5432/test_graphrag",
    REDIS_URL="redis://localhost:6379/1",
    DEBUG=True
)

@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_engine():
    """创建测试数据库引擎"""
    engine = create_async_engine(
        test_settings.DATABASE_URL,
        echo=True,
        future=True
    )
    
    # 创建表
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # 清理
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()

@pytest.fixture
async def test_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """创建测试数据库会话"""
    async_session = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()

@pytest.fixture
def override_get_session(test_session):
    """覆盖数据库会话依赖"""
    async def _override_get_session():
        yield test_session
    
    return _override_get_session
```

### 单元测试示例

```python
# tests/unit/test_document_processor.py
"""
文档处理器单元测试
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from src.core.document_processor import DocumentProcessor
from src.models.database.documents import Document

class TestDocumentProcessor:
    """文档处理器测试类"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """模拟依赖"""
        text_splitter = Mock()
        file_handler = Mock()
        vector_service = Mock()
        
        return text_splitter, file_handler, vector_service
    
    @pytest.fixture
    def document_processor(self, mock_dependencies):
        """创建文档处理器实例"""
        text_splitter, file_handler, vector_service = mock_dependencies
        return DocumentProcessor(text_splitter, file_handler, vector_service)
    
    @pytest.mark.asyncio
    async def test_process_document_success(self, document_processor, mock_dependencies):
        """测试文档处理成功场景"""
        text_splitter, file_handler, vector_service = mock_dependencies
        
        # 设置模拟返回值
        file_handler.extract_text = AsyncMock(return_value="测试文档内容")
        text_splitter.split_text = AsyncMock(return_value=[])
        vector_service.embed_texts = AsyncMock(return_value=[])
        
        # 执行测试
        file_path = Path("test.pdf")
        result = await document_processor.process_document(file_path)
        
        # 验证结果
        assert isinstance(result, Document)
        assert result.title == "test"
        assert result.file_type == ".pdf"
        
        # 验证方法调用
        file_handler.extract_text.assert_called_once_with(file_path)
        text_splitter.split_text.assert_called_once()
```

## 部署配置

### Docker配置

```dockerfile
# docker/Dockerfile.api - API服务镜像
FROM python:3.11-slim as base

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 开发阶段
FROM base as development
COPY . .
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# 生产阶段
FROM base as production

# 创建非root用户
RUN useradd --create-home --shell /bin/bash app

# 复制应用代码
COPY --chown=app:app . .

# 切换到非root用户
USER app

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动应用
CMD ["gunicorn", "src.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### Docker Compose配置

```yaml
# docker-compose.yml - 开发环境编排
version: '3.8'

services:
  # API服务
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
      target: development
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:password@postgres:5432/graphrag
      - REDIS_URL=redis://redis:6379/0
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=password
    volumes:
      - .:/app
      - uploads:/app/uploads
    depends_on:
      - postgres
      - redis
      - neo4j
    networks:
      - graphrag-network

  # Worker服务
  worker:
    build:
      context: .
      dockerfile: docker/Dockerfile.worker
      target: development
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql+asyncpg://postgres:password@postgres:5432/graphrag
    volumes:
      - .:/app
      - uploads:/app/uploads
    depends_on:
      - postgres
      - redis
    networks:
      - graphrag-network

  # PostgreSQL数据库
  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=graphrag
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - graphrag-network

  # Redis缓存
  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - graphrag-network

  # Neo4j图数据库
  neo4j:
    image: neo4j:5.0
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc"]
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
    networks:
      - graphrag-network

  # Weaviate向量数据库
  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "8080:8080"
    environment:
      - QUERY_DEFAULTS_LIMIT=25
      - AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
      - PERSISTENCE_DATA_PATH=/var/lib/weaviate
    volumes:
      - weaviate_data:/var/lib/weaviate
    networks:
      - graphrag-network

volumes:
  postgres_data:
  redis_data:
  neo4j_data:
  weaviate_data:
  uploads:

networks:
  graphrag-network:
    driver: bridge
```

## 开发工具配置

### 预提交钩子 (`.pre-commit-config.yaml`)

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### 项目配置 (`pyproject.toml`)

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "graphrag"
version = "1.0.0"
description = "GraphRAG Knowledge Base System"
authors = [
    {name = "GraphRAG Team", email = "team@graphrag.com"}
]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.11"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
]

dependencies = [
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.22.0",
    "sqlalchemy[asyncio]>=2.0.0",
    "asyncpg>=0.28.0",
    "alembic>=1.11.0",
    "pydantic>=2.0.0",
    "celery>=5.3.0",
    "redis>=4.6.0",
    "neo4j>=5.10.0",
    "weaviate-client>=3.22.0",
    "minio>=7.1.0",
    "sentence-transformers>=2.2.0",
    "openai>=0.27.0",
    "pypdf2>=3.0.0",
    "python-multipart>=0.0.6",
    "python-jose[cryptography]>=3.3.0",
    "passlib[bcrypt]>=1.7.4",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "black>=23.3.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.3.0",
    "pre-commit>=3.3.0",
]

test = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "httpx>=0.24.0",
    "factory-boy>=3.3.0",
]

[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
line_length = 88

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]
```

## 总结

这个项目结构提供了：

1. **清晰的分层架构**: API层、业务层、数据层分离
2. **模块化设计**: 每个功能模块独立，便于维护
3. **完整的测试体系**: 单元测试、集成测试、端到端测试
4. **标准化配置**: 统一的配置管理和环境变量
5. **容器化部署**: Docker和Kubernetes支持
6. **开发工具集成**: 代码格式化、类型检查、预提交钩子
7. **异步任务支持**: Celery后台任务处理
8. **多数据库支持**: PostgreSQL、Neo4j、Weaviate、Redis

这个结构为GraphRAG知识库系统提供了坚实的基础，支持快速开发和扩展。