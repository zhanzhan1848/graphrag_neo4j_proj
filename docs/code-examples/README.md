# 代码示例和骨架

## 概述

本文档提供了GraphRAG知识库系统的核心代码示例和实现骨架，帮助开发者快速理解系统架构和实现细节。

## 核心代码示例

### 1. 应用入口 (`src/main.py`)

```python
"""
GraphRAG知识库系统主入口
启动FastAPI应用和相关服务
"""

import uvicorn
from contextlib import asynccontextmanager

from src.api.app import create_app
from src.config.settings import settings
from src.config.database import init_database
from src.config.logging import setup_logging

@asynccontextmanager
async def lifespan(app):
    """应用生命周期管理"""
    # 启动时初始化
    setup_logging()
    await init_database()
    
    yield
    
    # 关闭时清理资源
    # 这里可以添加清理逻辑

def main():
    """主函数"""
    app = create_app()
    
    if settings.DEBUG:
        # 开发模式
        uvicorn.run(
            "src.main:app",
            host=settings.API_HOST,
            port=settings.API_PORT,
            reload=True,
            log_level=settings.LOG_LEVEL.lower()
        )
    else:
        # 生产模式
        uvicorn.run(
            app,
            host=settings.API_HOST,
            port=settings.API_PORT,
            workers=settings.API_WORKERS,
            log_level=settings.LOG_LEVEL.lower()
        )

if __name__ == "__main__":
    main()
```

### 2. 文档管理API (`src/api/routes/documents.py`)

```python
"""
文档管理API路由
提供文档上传、查询、删除等功能
"""

from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse

from src.models.schemas.documents import (
    DocumentCreate, DocumentResponse, DocumentList, DocumentUpdate
)
from src.services.document_service import DocumentService
from src.api.dependencies import get_document_service
from src.utils.validation import validate_file_type, validate_file_size

router = APIRouter()

@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = None,
    metadata: Optional[dict] = None,
    async_processing: bool = True,
    document_service: DocumentService = Depends(get_document_service)
):
    """
    上传文档
    
    Args:
        file: 上传的文件
        title: 文档标题（可选）
        metadata: 文档元数据（可选）
        async_processing: 是否异步处理
        document_service: 文档服务
        
    Returns:
        DocumentResponse: 文档信息
    """
    # 验证文件
    if not validate_file_type(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="不支持的文件类型"
        )
    
    if not validate_file_size(file.size):
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="文件大小超过限制"
        )
    
    try:
        # 保存文件并处理
        document = await document_service.upload_document(
            file=file,
            title=title,
            metadata=metadata,
            async_processing=async_processing
        )
        
        return DocumentResponse.from_orm(document)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文档上传失败: {str(e)}"
        )

@router.get("/", response_model=DocumentList)
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    status_filter: Optional[str] = None,
    document_service: DocumentService = Depends(get_document_service)
):
    """
    获取文档列表
    
    Args:
        skip: 跳过的记录数
        limit: 返回的记录数限制
        status_filter: 状态过滤
        document_service: 文档服务
        
    Returns:
        DocumentList: 文档列表
    """
    documents = await document_service.list_documents(
        skip=skip,
        limit=limit,
        status=status_filter
    )
    
    total = await document_service.count_documents(status=status_filter)
    
    return DocumentList(
        items=[DocumentResponse.from_orm(doc) for doc in documents],
        total=total,
        skip=skip,
        limit=limit
    )

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,
    document_service: DocumentService = Depends(get_document_service)
):
    """
    获取单个文档
    
    Args:
        document_id: 文档ID
        document_service: 文档服务
        
    Returns:
        DocumentResponse: 文档信息
    """
    document = await document_service.get_document(document_id)
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在"
        )
    
    return DocumentResponse.from_orm(document)

@router.put("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: UUID,
    document_update: DocumentUpdate,
    document_service: DocumentService = Depends(get_document_service)
):
    """
    更新文档信息
    
    Args:
        document_id: 文档ID
        document_update: 更新数据
        document_service: 文档服务
        
    Returns:
        DocumentResponse: 更新后的文档信息
    """
    document = await document_service.update_document(
        document_id, 
        document_update.dict(exclude_unset=True)
    )
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在"
        )
    
    return DocumentResponse.from_orm(document)

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: UUID,
    document_service: DocumentService = Depends(get_document_service)
):
    """
    删除文档
    
    Args:
        document_id: 文档ID
        document_service: 文档服务
    """
    success = await document_service.delete_document(document_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在"
        )

@router.get("/{document_id}/chunks")
async def get_document_chunks(
    document_id: UUID,
    skip: int = 0,
    limit: int = 50,
    document_service: DocumentService = Depends(get_document_service)
):
    """
    获取文档的文本块
    
    Args:
        document_id: 文档ID
        skip: 跳过的记录数
        limit: 返回的记录数限制
        document_service: 文档服务
        
    Returns:
        文档文本块列表
    """
    chunks = await document_service.get_document_chunks(
        document_id, skip=skip, limit=limit
    )
    
    return {
        "document_id": document_id,
        "chunks": chunks,
        "total": len(chunks)
    }

@router.post("/{document_id}/reprocess")
async def reprocess_document(
    document_id: UUID,
    document_service: DocumentService = Depends(get_document_service)
):
    """
    重新处理文档
    
    Args:
        document_id: 文档ID
        document_service: 文档服务
        
    Returns:
        处理状态
    """
    success = await document_service.reprocess_document(document_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在"
        )
    
    return {"message": "文档重新处理已启动", "document_id": document_id}
```

### 3. 知识查询API (`src/api/routes/query.py`)

```python
"""
知识查询API路由
提供自然语言查询、向量检索、图查询等功能
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.models.schemas.query import (
    QueryRequest, QueryResponse, VectorSearchRequest, VectorSearchResponse,
    GraphQueryRequest, GraphQueryResponse
)
from src.services.query_service import QueryService
from src.api.dependencies import get_query_service

router = APIRouter()

class NaturalLanguageQuery(BaseModel):
    """自然语言查询请求"""
    question: str
    context_limit: int = 5
    include_sources: bool = True
    temperature: float = 0.7

class HybridSearchRequest(BaseModel):
    """混合搜索请求"""
    query: str
    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    limit: int = 10
    filters: Optional[Dict[str, Any]] = None

@router.post("/natural-language", response_model=QueryResponse)
async def natural_language_query(
    request: NaturalLanguageQuery,
    query_service: QueryService = Depends(get_query_service)
):
    """
    自然语言查询
    
    Args:
        request: 查询请求
        query_service: 查询服务
        
    Returns:
        QueryResponse: 查询结果
    """
    try:
        result = await query_service.natural_language_query(
            question=request.question,
            context_limit=request.context_limit,
            include_sources=request.include_sources,
            temperature=request.temperature
        )
        
        return QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            confidence=result.get("confidence", 0.0),
            processing_time=result.get("processing_time", 0.0)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"查询处理失败: {str(e)}"
        )

@router.post("/vector-search", response_model=VectorSearchResponse)
async def vector_search(
    request: VectorSearchRequest,
    query_service: QueryService = Depends(get_query_service)
):
    """
    向量相似度搜索
    
    Args:
        request: 搜索请求
        query_service: 查询服务
        
    Returns:
        VectorSearchResponse: 搜索结果
    """
    try:
        results = await query_service.vector_search(
            query=request.query,
            limit=request.limit,
            threshold=request.threshold,
            filters=request.filters
        )
        
        return VectorSearchResponse(
            results=results,
            total=len(results),
            query=request.query
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"向量搜索失败: {str(e)}"
        )

@router.post("/graph-query", response_model=GraphQueryResponse)
async def graph_query(
    request: GraphQueryRequest,
    query_service: QueryService = Depends(get_query_service)
):
    """
    图数据库查询
    
    Args:
        request: 图查询请求
        query_service: 查询服务
        
    Returns:
        GraphQueryResponse: 查询结果
    """
    try:
        result = await query_service.graph_query(
            cypher_query=request.cypher_query,
            parameters=request.parameters
        )
        
        return GraphQueryResponse(
            nodes=result.get("nodes", []),
            relationships=result.get("relationships", []),
            data=result.get("data", [])
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"图查询失败: {str(e)}"
        )

@router.post("/hybrid-search")
async def hybrid_search(
    request: HybridSearchRequest,
    query_service: QueryService = Depends(get_query_service)
):
    """
    混合搜索（向量+关键词）
    
    Args:
        request: 混合搜索请求
        query_service: 查询服务
        
    Returns:
        混合搜索结果
    """
    try:
        results = await query_service.hybrid_search(
            query=request.query,
            vector_weight=request.vector_weight,
            keyword_weight=request.keyword_weight,
            limit=request.limit,
            filters=request.filters
        )
        
        return {
            "results": results,
            "total": len(results),
            "query": request.query,
            "search_type": "hybrid"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"混合搜索失败: {str(e)}"
        )

@router.get("/suggestions")
async def get_query_suggestions(
    query: str,
    limit: int = 5,
    query_service: QueryService = Depends(get_query_service)
):
    """
    获取查询建议
    
    Args:
        query: 查询文本
        limit: 建议数量限制
        query_service: 查询服务
        
    Returns:
        查询建议列表
    """
    try:
        suggestions = await query_service.get_query_suggestions(
            query=query,
            limit=limit
        )
        
        return {
            "suggestions": suggestions,
            "query": query
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取建议失败: {str(e)}"
        )
```

### 4. 知识抽取核心 (`src/core/knowledge_extractor.py`)

```python
"""
知识抽取核心模块
从文本中抽取实体、关系和断言
"""

import asyncio
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from uuid import UUID

from src.models.database.chunks import Chunk
from src.models.database.entities import Entity
from src.models.database.relations import Relation
from src.services.llm_service import LLMService
from src.services.vector_service import VectorService
from src.repositories.entity_repository import EntityRepository
from src.repositories.relation_repository import RelationRepository
from src.utils.text_processing import clean_text, extract_sentences

@dataclass
class ExtractedKnowledge:
    """抽取的知识结构"""
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    claims: List[Dict[str, Any]]
    confidence: float

class KnowledgeExtractor:
    """知识抽取器"""
    
    def __init__(
        self,
        llm_service: LLMService,
        vector_service: VectorService,
        entity_repository: EntityRepository,
        relation_repository: RelationRepository
    ):
        self.llm_service = llm_service
        self.vector_service = vector_service
        self.entity_repository = entity_repository
        self.relation_repository = relation_repository
        
        # 抽取提示模板
        self.entity_extraction_prompt = """
        请从以下文本中抽取所有重要的实体，包括人物、地点、组织、概念等。
        对每个实体，请提供：
        1. 实体名称
        2. 实体类型（PERSON, LOCATION, ORGANIZATION, CONCEPT等）
        3. 实体描述
        4. 置信度（0-1）
        
        文本：{text}
        
        请以JSON格式返回结果：
        {
            "entities": [
                {
                    "name": "实体名称",
                    "type": "实体类型",
                    "description": "实体描述",
                    "confidence": 0.95
                }
            ]
        }
        """
        
        self.relation_extraction_prompt = """
        请从以下文本中抽取实体之间的关系。
        对每个关系，请提供：
        1. 主体实体
        2. 关系类型
        3. 客体实体
        4. 关系描述
        5. 置信度（0-1）
        
        已知实体：{entities}
        文本：{text}
        
        请以JSON格式返回结果：
        {
            "relations": [
                {
                    "subject": "主体实体",
                    "predicate": "关系类型",
                    "object": "客体实体",
                    "description": "关系描述",
                    "confidence": 0.9
                }
            ]
        }
        """
    
    async def extract_knowledge(
        self, 
        chunk: Chunk,
        existing_entities: Optional[List[Entity]] = None
    ) -> ExtractedKnowledge:
        """
        从文本块中抽取知识
        
        Args:
            chunk: 文本块
            existing_entities: 已存在的实体列表
            
        Returns:
            ExtractedKnowledge: 抽取的知识
        """
        # 1. 清理文本
        clean_content = clean_text(chunk.content)
        
        # 2. 抽取实体
        entities = await self._extract_entities(clean_content)
        
        # 3. 实体链接和去重
        linked_entities = await self._link_entities(entities, existing_entities)
        
        # 4. 抽取关系
        relations = await self._extract_relations(clean_content, linked_entities)
        
        # 5. 抽取断言
        claims = await self._extract_claims(clean_content, linked_entities, relations)
        
        # 6. 计算整体置信度
        confidence = self._calculate_confidence(entities, relations, claims)
        
        return ExtractedKnowledge(
            entities=linked_entities,
            relations=relations,
            claims=claims,
            confidence=confidence
        )
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """抽取实体"""
        try:
            prompt = self.entity_extraction_prompt.format(text=text)
            response = await self.llm_service.generate_response(
                prompt=prompt,
                temperature=0.1,
                max_tokens=1000
            )
            
            # 解析JSON响应
            result = json.loads(response)
            return result.get("entities", [])
            
        except Exception as e:
            print(f"实体抽取失败: {e}")
            return []
    
    async def _extract_relations(
        self, 
        text: str, 
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """抽取关系"""
        try:
            entity_names = [entity["name"] for entity in entities]
            prompt = self.relation_extraction_prompt.format(
                text=text,
                entities=", ".join(entity_names)
            )
            
            response = await self.llm_service.generate_response(
                prompt=prompt,
                temperature=0.1,
                max_tokens=1500
            )
            
            # 解析JSON响应
            result = json.loads(response)
            return result.get("relations", [])
            
        except Exception as e:
            print(f"关系抽取失败: {e}")
            return []
    
    async def _extract_claims(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """抽取断言/声明"""
        claims = []
        sentences = extract_sentences(text)
        
        for sentence in sentences:
            # 检查句子是否包含实体
            sentence_entities = []
            for entity in entities:
                if entity["name"].lower() in sentence.lower():
                    sentence_entities.append(entity)
            
            if len(sentence_entities) >= 1:  # 至少包含一个实体
                claim = {
                    "text": sentence,
                    "entities": sentence_entities,
                    "confidence": 0.8,
                    "type": "factual_claim"
                }
                claims.append(claim)
        
        return claims
    
    async def _link_entities(
        self,
        extracted_entities: List[Dict[str, Any]],
        existing_entities: Optional[List[Entity]] = None
    ) -> List[Dict[str, Any]]:
        """实体链接和去重"""
        linked_entities = []
        
        for entity in extracted_entities:
            # 查找相似的已存在实体
            similar_entity = await self._find_similar_entity(
                entity, existing_entities
            )
            
            if similar_entity:
                # 合并实体信息
                merged_entity = self._merge_entities(entity, similar_entity)
                linked_entities.append(merged_entity)
            else:
                # 新实体
                linked_entities.append(entity)
        
        return linked_entities
    
    async def _find_similar_entity(
        self,
        entity: Dict[str, Any],
        existing_entities: Optional[List[Entity]] = None
    ) -> Optional[Dict[str, Any]]:
        """查找相似实体"""
        if not existing_entities:
            return None
        
        # 使用向量相似度查找
        entity_embedding = await self.vector_service.embed_text(entity["name"])
        
        max_similarity = 0.0
        most_similar_entity = None
        
        for existing_entity in existing_entities:
            if existing_entity.embedding:
                similarity = self.vector_service.cosine_similarity(
                    entity_embedding, existing_entity.embedding
                )
                
                if similarity > max_similarity and similarity > 0.85:
                    max_similarity = similarity
                    most_similar_entity = {
                        "id": existing_entity.id,
                        "name": existing_entity.name,
                        "type": existing_entity.entity_type,
                        "description": existing_entity.description,
                        "confidence": existing_entity.confidence
                    }
        
        return most_similar_entity
    
    def _merge_entities(
        self,
        new_entity: Dict[str, Any],
        existing_entity: Dict[str, Any]
    ) -> Dict[str, Any]:
        """合并实体信息"""
        return {
            "id": existing_entity.get("id"),
            "name": existing_entity["name"],  # 保持已存在实体的名称
            "type": existing_entity["type"],
            "description": self._merge_descriptions(
                new_entity.get("description", ""),
                existing_entity.get("description", "")
            ),
            "confidence": max(
                new_entity.get("confidence", 0.0),
                existing_entity.get("confidence", 0.0)
            ),
            "is_existing": True
        }
    
    def _merge_descriptions(self, desc1: str, desc2: str) -> str:
        """合并描述"""
        if not desc1:
            return desc2
        if not desc2:
            return desc1
        
        # 简单合并，实际可以使用更复杂的逻辑
        if desc1 in desc2:
            return desc2
        if desc2 in desc1:
            return desc1
        
        return f"{desc2}; {desc1}"
    
    def _calculate_confidence(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        claims: List[Dict[str, Any]]
    ) -> float:
        """计算整体置信度"""
        all_confidences = []
        
        # 收集所有置信度
        for entity in entities:
            all_confidences.append(entity.get("confidence", 0.0))
        
        for relation in relations:
            all_confidences.append(relation.get("confidence", 0.0))
        
        for claim in claims:
            all_confidences.append(claim.get("confidence", 0.0))
        
        if not all_confidences:
            return 0.0
        
        # 计算加权平均
        return sum(all_confidences) / len(all_confidences)
    
    async def store_knowledge(
        self,
        knowledge: ExtractedKnowledge,
        chunk_id: UUID,
        document_id: UUID
    ) -> Dict[str, int]:
        """
        存储抽取的知识到数据库
        
        Args:
            knowledge: 抽取的知识
            chunk_id: 文本块ID
            document_id: 文档ID
            
        Returns:
            存储统计信息
        """
        stored_entities = 0
        stored_relations = 0
        
        # 存储实体
        entity_mapping = {}
        for entity_data in knowledge.entities:
            if entity_data.get("is_existing"):
                # 已存在的实体，只需要建立关联
                entity_mapping[entity_data["name"]] = entity_data["id"]
            else:
                # 新实体，需要创建
                entity = Entity(
                    name=entity_data["name"],
                    entity_type=entity_data["type"],
                    description=entity_data.get("description", ""),
                    confidence=entity_data.get("confidence", 0.0),
                    source_chunk_id=chunk_id,
                    source_document_id=document_id
                )
                
                # 生成实体嵌入
                entity.embedding = await self.vector_service.embed_text(entity.name)
                
                # 保存到数据库
                saved_entity = await self.entity_repository.create(entity)
                entity_mapping[entity_data["name"]] = saved_entity.id
                stored_entities += 1
        
        # 存储关系
        for relation_data in knowledge.relations:
            subject_id = entity_mapping.get(relation_data["subject"])
            object_id = entity_mapping.get(relation_data["object"])
            
            if subject_id and object_id:
                relation = Relation(
                    subject_id=subject_id,
                    predicate=relation_data["predicate"],
                    object_id=object_id,
                    description=relation_data.get("description", ""),
                    confidence=relation_data.get("confidence", 0.0),
                    source_chunk_id=chunk_id,
                    source_document_id=document_id
                )
                
                await self.relation_repository.create(relation)
                stored_relations += 1
        
        return {
            "entities": stored_entities,
            "relations": stored_relations,
            "claims": len(knowledge.claims)
        }
```

### 5. 图数据库操作 (`src/repositories/graph_repository.py`)

```python
"""
图数据库操作仓库
提供Neo4j图数据库的CRUD操作
"""

from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
import asyncio

from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import Neo4jError

from src.config.settings import settings
from src.models.graph.nodes import EntityNode, DocumentNode, ChunkNode
from src.models.graph.relationships import RelationshipType

class GraphRepository:
    """图数据库仓库"""
    
    def __init__(self):
        self.driver: Optional[AsyncDriver] = None
    
    async def connect(self):
        """连接到Neo4j数据库"""
        self.driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
        
        # 验证连接
        await self.driver.verify_connectivity()
    
    async def close(self):
        """关闭数据库连接"""
        if self.driver:
            await self.driver.close()
    
    async def create_entity_node(
        self,
        entity_id: UUID,
        name: str,
        entity_type: str,
        properties: Dict[str, Any] = None
    ) -> bool:
        """
        创建实体节点
        
        Args:
            entity_id: 实体ID
            name: 实体名称
            entity_type: 实体类型
            properties: 附加属性
            
        Returns:
            bool: 创建是否成功
        """
        query = """
        CREATE (e:Entity {
            id: $entity_id,
            name: $name,
            type: $entity_type,
            created_at: datetime(),
            updated_at: datetime()
        })
        SET e += $properties
        RETURN e
        """
        
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    query,
                    entity_id=str(entity_id),
                    name=name,
                    entity_type=entity_type,
                    properties=properties or {}
                )
                
                return await result.single() is not None
                
        except Neo4jError as e:
            print(f"创建实体节点失败: {e}")
            return False
    
    async def create_relationship(
        self,
        subject_id: UUID,
        predicate: str,
        object_id: UUID,
        properties: Dict[str, Any] = None
    ) -> bool:
        """
        创建关系
        
        Args:
            subject_id: 主体实体ID
            predicate: 关系类型
            object_id: 客体实体ID
            properties: 关系属性
            
        Returns:
            bool: 创建是否成功
        """
        query = """
        MATCH (s:Entity {id: $subject_id})
        MATCH (o:Entity {id: $object_id})
        CREATE (s)-[r:RELATES {
            type: $predicate,
            created_at: datetime(),
            updated_at: datetime()
        }]->(o)
        SET r += $properties
        RETURN r
        """
        
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    query,
                    subject_id=str(subject_id),
                    predicate=predicate,
                    object_id=str(object_id),
                    properties=properties or {}
                )
                
                return await result.single() is not None
                
        except Neo4jError as e:
            print(f"创建关系失败: {e}")
            return False
    
    async def find_entity_by_name(
        self,
        name: str,
        entity_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        根据名称查找实体
        
        Args:
            name: 实体名称
            entity_type: 实体类型（可选）
            
        Returns:
            实体信息或None
        """
        if entity_type:
            query = """
            MATCH (e:Entity {name: $name, type: $entity_type})
            RETURN e
            """
            params = {"name": name, "entity_type": entity_type}
        else:
            query = """
            MATCH (e:Entity {name: $name})
            RETURN e
            """
            params = {"name": name}
        
        try:
            async with self.driver.session() as session:
                result = await session.run(query, **params)
                record = await result.single()
                
                if record:
                    return dict(record["e"])
                return None
                
        except Neo4jError as e:
            print(f"查找实体失败: {e}")
            return None
    
    async def get_entity_neighbors(
        self,
        entity_id: UUID,
        relationship_types: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        获取实体的邻居节点
        
        Args:
            entity_id: 实体ID
            relationship_types: 关系类型过滤
            limit: 结果数量限制
            
        Returns:
            邻居节点列表
        """
        if relationship_types:
            rel_filter = f"WHERE r.type IN {relationship_types}"
        else:
            rel_filter = ""
        
        query = f"""
        MATCH (e:Entity {{id: $entity_id}})-[r:RELATES]-(neighbor:Entity)
        {rel_filter}
        RETURN neighbor, r, type(r) as rel_type
        LIMIT $limit
        """
        
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    query,
                    entity_id=str(entity_id),
                    limit=limit
                )
                
                neighbors = []
                async for record in result:
                    neighbors.append({
                        "entity": dict(record["neighbor"]),
                        "relationship": dict(record["r"]),
                        "relationship_type": record["rel_type"]
                    })
                
                return neighbors
                
        except Neo4jError as e:
            print(f"获取邻居节点失败: {e}")
            return []
    
    async def find_shortest_path(
        self,
        start_entity_id: UUID,
        end_entity_id: UUID,
        max_depth: int = 5
    ) -> Optional[List[Dict[str, Any]]]:
        """
        查找两个实体之间的最短路径
        
        Args:
            start_entity_id: 起始实体ID
            end_entity_id: 结束实体ID
            max_depth: 最大搜索深度
            
        Returns:
            路径信息或None
        """
        query = """
        MATCH path = shortestPath(
            (start:Entity {id: $start_id})-[*1..$max_depth]-(end:Entity {id: $end_id})
        )
        RETURN path, length(path) as path_length
        """
        
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    query,
                    start_id=str(start_entity_id),
                    end_id=str(end_entity_id),
                    max_depth=max_depth
                )
                
                record = await result.single()
                if record:
                    path = record["path"]
                    return {
                        "nodes": [dict(node) for node in path.nodes],
                        "relationships": [dict(rel) for rel in path.relationships],
                        "length": record["path_length"]
                    }
                
                return None
                
        except Neo4jError as e:
            print(f"查找最短路径失败: {e}")
            return None
    
    async def execute_cypher_query(
        self,
        query: str,
        parameters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        执行自定义Cypher查询
        
        Args:
            query: Cypher查询语句
            parameters: 查询参数
            
        Returns:
            查询结果
        """
        try:
            async with self.driver.session() as session:
                result = await session.run(query, **(parameters or {}))
                
                records = []
                async for record in result:
                    records.append(dict(record))
                
                return records
                
        except Neo4jError as e:
            print(f"执行Cypher查询失败: {e}")
            return []
    
    async def get_graph_statistics(self) -> Dict[str, int]:
        """
        获取图数据库统计信息
        
        Returns:
            统计信息字典
        """
        queries = {
            "total_nodes": "MATCH (n) RETURN count(n) as count",
            "total_relationships": "MATCH ()-[r]->() RETURN count(r) as count",
            "entity_nodes": "MATCH (n:Entity) RETURN count(n) as count",
            "document_nodes": "MATCH (n:Document) RETURN count(n) as count",
            "chunk_nodes": "MATCH (n:Chunk) RETURN count(n) as count"
        }
        
        statistics = {}
        
        try:
            async with self.driver.session() as session:
                for stat_name, query in queries.items():
                    result = await session.run(query)
                    record = await result.single()
                    statistics[stat_name] = record["count"] if record else 0
            
            return statistics
            
        except Neo4jError as e:
            print(f"获取统计信息失败: {e}")
            return {}
    
    async def create_indexes(self):
        """创建必要的索引"""
        indexes = [
            "CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX document_id_index IF NOT EXISTS FOR (d:Document) ON (d.id)",
            "CREATE INDEX chunk_id_index IF NOT EXISTS FOR (c:Chunk) ON (c.id)"
        ]
        
        try:
            async with self.driver.session() as session:
                for index_query in indexes:
                    await session.run(index_query)
            
            print("图数据库索引创建完成")
            
        except Neo4jError as e:
            print(f"创建索引失败: {e}")
```

### 6. 环境配置文件

#### `.env.example`

```bash
# 应用配置
APP_NAME=GraphRAG Knowledge Base
VERSION=1.0.0
DEBUG=true
SECRET_KEY=your-secret-key-here

# 数据库配置
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/graphrag
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis配置
REDIS_URL=redis://localhost:6379/0

# Neo4j配置
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Weaviate配置
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=

# MinIO配置
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=graphrag

# Celery配置
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# API配置
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080

# 文件处理配置
MAX_FILE_SIZE=104857600  # 100MB
ALLOWED_FILE_TYPES=.pdf,.txt,.md,.docx,.html
UPLOAD_DIR=/tmp/uploads

# AI模型配置
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gpt-3.5-turbo
OPENAI_API_KEY=your-openai-api-key

# 日志配置
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

#### `requirements.txt`

```txt
# Web框架
fastapi==0.104.1
uvicorn[standard]==0.24.0
gunicorn==21.2.0

# 数据库
sqlalchemy[asyncio]==2.0.23
asyncpg==0.29.0
alembic==1.12.1
psycopg2-binary==2.9.9

# 图数据库
neo4j==5.14.1

# 向量数据库
weaviate-client==3.25.3

# 对象存储
minio==7.2.0

# 缓存和消息队列
redis==5.0.1
celery==5.3.4

# 数据验证和序列化
pydantic==2.5.0
pydantic-settings==2.1.0

# 认证和安全
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6

# AI和机器学习
openai==1.3.7
sentence-transformers==2.2.2
transformers==4.36.2
torch==2.1.1
numpy==1.24.4

# 文档处理
pypdf2==3.0.1
python-docx==1.1.0
python-pptx==0.6.23
openpyxl==3.1.2
beautifulsoup4==4.12.2
markdown==3.5.1

# 图像处理
pillow==10.1.0
opencv-python==4.8.1.78
pytesseract==0.3.10

# 工具库
httpx==0.25.2
aiofiles==23.2.1
python-dotenv==1.0.0
click==8.1.7
rich==13.7.0
typer==0.9.0

# 监控和日志
prometheus-client==0.19.0
structlog==23.2.0

# 测试
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
factory-boy==3.3.0
faker==20.1.0

# 代码质量
black==23.11.0
isort==5.12.0
flake8==6.1.0
mypy==1.7.1
pre-commit==3.6.0

# 开发工具
ipython==8.17.2
jupyter==1.0.0
```

### 7. 启动脚本 (`scripts/setup.sh`)

```bash
#!/bin/bash

# GraphRAG项目环境设置脚本

set -e

echo "🚀 开始设置GraphRAG开发环境..."

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ 需要Python 3.11或更高版本，当前版本: $python_version"
    exit 1
fi

echo "✅ Python版本检查通过: $python_version"

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "📦 创建Python虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source venv/bin/activate

# 升级pip
echo "⬆️ 升级pip..."
pip install --upgrade pip

# 安装依赖
echo "📚 安装Python依赖..."
pip install -r requirements.txt

# 安装开发依赖
echo "🛠️ 安装开发依赖..."
pip install -r requirements-dev.txt

# 设置环境变量
if [ ! -f ".env" ]; then
    echo "⚙️ 创建环境配置文件..."
    cp .env.example .env
    echo "请编辑 .env 文件配置您的环境变量"
fi

# 设置pre-commit钩子
echo "🔗 设置pre-commit钩子..."
pre-commit install

# 创建必要的目录
echo "📁 创建项目目录..."
mkdir -p logs
mkdir -p uploads
mkdir -p data/backups
mkdir -p data/exports

# 检查Docker
if command -v docker &> /dev/null; then
    echo "✅ Docker已安装"
    
    # 检查Docker Compose
    if command -v docker-compose &> /dev/null; then
        echo "✅ Docker Compose已安装"
        
        echo "🐳 启动开发环境服务..."
        docker-compose up -d postgres redis neo4j weaviate
        
        echo "⏳ 等待服务启动..."
        sleep 10
        
    else
        echo "⚠️ Docker Compose未安装，请手动安装后运行 docker-compose up -d"
    fi
else
    echo "⚠️ Docker未安装，请手动安装Docker和Docker Compose"
fi

# 运行数据库迁移
echo "🗄️ 运行数据库迁移..."
alembic upgrade head

# 创建Neo4j索引
echo "📊 创建图数据库索引..."
python -c "
import asyncio
from src.repositories.graph_repository import GraphRepository

async def setup_indexes():
    repo = GraphRepository()
    await repo.connect()
    await repo.create_indexes()
    await repo.close()

asyncio.run(setup_indexes())
"

echo "🎉 环境设置完成！"
echo ""
echo "下一步："
echo "1. 编辑 .env 文件配置您的API密钥"
echo "2. 运行 'python src/main.py' 启动API服务"
echo "3. 访问 http://localhost:8000/docs 查看API文档"
echo ""
echo "开发命令："
echo "- 启动API服务: python src/main.py"
echo "- 启动Worker: celery -A src.workers.celery_app worker --loglevel=info"
echo "- 运行测试: pytest"
echo "- 代码格式化: black src/ && isort src/"
echo "- 类型检查: mypy src/"
```

### 8. 数据库迁移示例 (`migrations/versions/001_initial_schema.py`)

```python
"""初始数据库架构

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    """创建初始数据库架构"""
    
    # 创建documents表
    op.create_table(
        'documents',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('title', sa.String(500), nullable=False, index=True),
        sa.Column('content', sa.Text),
        sa.Column('file_path', sa.String(1000), nullable=False),
        sa.Column('file_type', sa.String(50), nullable=False),
        sa.Column('file_size', sa.Integer),
        sa.Column('language', sa.String(10), default='zh'),
        sa.Column('status', sa.String(20), default='processing', index=True),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime),
        sa.Column('metadata', postgresql.JSON, default=dict)
    )
    
    # 创建chunks表
    op.create_table(
        'chunks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('chunk_index', sa.Integer, nullable=False),
        sa.Column('start_char', sa.Integer),
        sa.Column('end_char', sa.Integer),
        sa.Column('token_count', sa.Integer),
        sa.Column('embedding', postgresql.ARRAY(sa.Float)),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime),
        sa.Column('metadata', postgresql.JSON, default=dict)
    )
    
    # 创建entities表
    op.create_table(
        'entities',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(500), nullable=False, index=True),
        sa.Column('entity_type', sa.String(100), nullable=False, index=True),
        sa.Column('description', sa.Text),
        sa.Column('confidence', sa.Float, default=0.0),
        sa.Column('embedding', postgresql.ARRAY(sa.Float)),
        sa.Column('source_chunk_id', postgresql.UUID(as_uuid=True),
                 sa.ForeignKey('chunks.id', ondelete='SET NULL')),
        sa.Column('source_document_id', postgresql.UUID(as_uuid=True),
                 sa.ForeignKey('documents.id', ondelete='CASCADE')),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime),
        sa.Column('metadata', postgresql.JSON, default=dict)
    )
    
    # 创建relations表
    op.create_table(
        'relations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('subject_id', postgresql.UUID(as_uuid=True),
                 sa.ForeignKey('entities.id', ondelete='CASCADE'), nullable=False),
        sa.Column('predicate', sa.String(200), nullable=False, index=True),
        sa.Column('object_id', postgresql.UUID(as_uuid=True),
                 sa.ForeignKey('entities.id', ondelete='CASCADE'), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('confidence', sa.Float, default=0.0),
        sa.Column('source_chunk_id', postgresql.UUID(as_uuid=True),
                 sa.ForeignKey('chunks.id', ondelete='SET NULL')),
        sa.Column('source_document_id', postgresql.UUID(as_uuid=True),
                 sa.ForeignKey('documents.id', ondelete='CASCADE')),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime),
        sa.Column('metadata', postgresql.JSON, default=dict)
    )
    
    # 创建images表
    op.create_table(
        'images',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('document_id', postgresql.UUID(as_uuid=True),
                 sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('file_path', sa.String(1000), nullable=False),
        sa.Column('file_name', sa.String(500), nullable=False),
        sa.Column('file_size', sa.Integer),
        sa.Column('width', sa.Integer),
        sa.Column('height', sa.Integer),
        sa.Column('format', sa.String(20)),
        sa.Column('extracted_text', sa.Text),
        sa.Column('features', postgresql.ARRAY(sa.Float)),
        sa.Column('status', sa.String(20), default='processing'),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime),
        sa.Column('metadata', postgresql.JSON, default=dict)
    )
    
    # 创建索引
    op.create_index('idx_chunks_document_id', 'chunks', ['document_id'])
    op.create_index('idx_chunks_embedding', 'chunks', ['embedding'], postgresql_using='gin')
    op.create_index('idx_entities_name_type', 'entities', ['name', 'entity_type'])
    op.create_index('idx_entities_embedding', 'entities', ['embedding'], postgresql_using='gin')
    op.create_index('idx_relations_subject_predicate', 'relations', ['subject_id', 'predicate'])
    op.create_index('idx_relations_object_predicate', 'relations', ['object_id', 'predicate'])
    op.create_index('idx_images_document_id', 'images', ['document_id'])

def downgrade() -> None:
    """删除数据库架构"""
    op.drop_table('images')
    op.drop_table('relations')
    op.drop_table('entities')
    op.drop_table('chunks')
    op.drop_table('documents')
```

## 总结

这个代码示例文档提供了：

1. **完整的应用架构**: 从入口点到各个层次的实现
2. **RESTful API设计**: 标准的HTTP接口实现
3. **核心业务逻辑**: 知识抽取和图数据库操作
4. **数据库设计**: 关系型和图数据库的集成
5. **配置管理**: 环境变量和设置管理
6. **部署脚本**: 自动化环境设置
7. **数据库迁移**: 版本化的数据库架构管理

这些代码骨架为GraphRAG系统提供了坚实的基础，开发者可以基于这些示例快速开始项目开发。