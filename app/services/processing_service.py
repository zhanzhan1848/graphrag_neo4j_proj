#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 文档处理服务
==================

本模块实现了文档处理的核心服务。

服务功能：
- 文档内容提取
- 文本分块处理
- 实体关系抽取
- 向量嵌入生成
- 图数据库存储
- 处理状态跟踪

处理流程：
1. 文档解析和内容提取
2. 文本清理和预处理
3. 智能分块
4. 实体和关系抽取
5. 向量嵌入生成
6. 数据存储到数据库

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from app.services.document_service import DocumentService
from app.services.text_service import TextService
from app.services.entity_service import EntityService
from app.services.relation_service import RelationService
from app.services.embedding_service import EmbeddingService
from app.services.graph_service import GraphService
from app.utils.exceptions import (
    DocumentProcessingError,
    TextProcessingError,
    EntityExtractionError,
    RelationExtractionError
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ProcessingService:
    """
    文档处理服务
    
    提供完整的文档处理流程，包括内容提取、分块、实体关系抽取等。
    """
    
    def __init__(self):
        """
        初始化处理服务
        """
        # 初始化各个服务组件
        self.text_service = TextService()
        self.entity_service = EntityService()
        self.relation_service = RelationService()
        self.embedding_service = EmbeddingService()
        self.graph_service = GraphService()
        
        # 处理状态跟踪
        self.processing_status = {}
        
        logger.info("文档处理服务初始化完成")
    
    async def process_document(
        self,
        document_id: uuid.UUID,
        db: Session,
        extract_entities: bool = True,
        extract_relations: bool = True,
        generate_embeddings: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> Dict[str, Any]:
        """
        处理文档的主要方法
        
        Args:
            document_id: 文档ID
            db: 数据库会话
            extract_entities: 是否抽取实体
            extract_relations: 是否抽取关系
            generate_embeddings: 是否生成嵌入
            chunk_size: 分块大小
            chunk_overlap: 分块重叠
            
        Returns:
            Dict[str, Any]: 处理结果
            
        Raises:
            DocumentProcessingError: 处理失败时抛出
        """
        try:
            logger.info(f"开始处理文档: {document_id}")
            
            # 初始化处理状态
            await self._init_processing_status(document_id)
            
            # 1. 获取文档信息
            document_service = DocumentService(db)
            document = await document_service.get_document(document_id)
            
            if not document:
                raise DocumentProcessingError(f"文档不存在: {document_id}")
            
            await self._update_processing_status(
                document_id, 
                "extracting_content", 
                0.1, 
                "正在提取文档内容..."
            )
            
            # 2. 提取文档内容
            content = await self._extract_document_content(document)
            
            if not content or not content.strip():
                raise DocumentProcessingError("文档内容为空")
            
            await self._update_processing_status(
                document_id, 
                "chunking_text", 
                0.2, 
                "正在分块文本..."
            )
            
            # 3. 文本分块
            chunks = await self.text_service.process_text(
                text=content,
                document_id=str(document_id),
                metadata={
                    "title": document.title,
                    "filename": document.file_name,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                },
                generate_embeddings=generate_embeddings
            )
            
            logger.info(f"文档分块完成，生成 {len(chunks)} 个文本块")
            
            # 4. 存储文本块
            await self._store_text_chunks(document_id, chunks, db)
            
            processing_results = {
                "document_id": str(document_id),
                "chunks_count": len(chunks),
                "entities_count": 0,
                "relations_count": 0,
                "processing_time": 0
            }
            
            # 5. 实体抽取
            if extract_entities:
                await self._update_processing_status(
                    document_id, 
                    "extracting_entities", 
                    0.4, 
                    "正在抽取实体..."
                )
                
                entities = await self._extract_entities(chunks, document_id)
                processing_results["entities_count"] = len(entities)
                
                logger.info(f"实体抽取完成，抽取到 {len(entities)} 个实体")
            
            # 6. 关系抽取
            if extract_relations:
                await self._update_processing_status(
                    document_id, 
                    "extracting_relations", 
                    0.6, 
                    "正在抽取关系..."
                )
                
                relations = await self._extract_relations(chunks, document_id)
                processing_results["relations_count"] = len(relations)
                
                logger.info(f"关系抽取完成，抽取到 {len(relations)} 个关系")
            
            # 7. 更新文档状态
            await self._update_processing_status(
                document_id, 
                "updating_database", 
                0.8, 
                "正在更新数据库..."
            )
            
            await document_service.update_document_status(
                document_id, 
                "completed", 
                "文档处理完成"
            )
            
            # 8. 完成处理
            await self._update_processing_status(
                document_id, 
                "completed", 
                1.0, 
                "文档处理完成"
            )
            
            logger.info(f"文档处理完成: {document_id}")
            
            return processing_results
            
        except Exception as e:
            logger.error(f"文档处理失败: {document_id}, 错误: {str(e)}")
            
            # 更新错误状态
            await self._update_processing_status(
                document_id, 
                "failed", 
                0.0, 
                f"处理失败: {str(e)}"
            )
            
            # 更新文档状态
            try:
                document_service = DocumentService(db)
                await document_service.update_document_status(
                    document_id, 
                    "failed", 
                    f"处理失败: {str(e)}"
                )
            except Exception as update_error:
                logger.error(f"更新文档状态失败: {str(update_error)}")
            
            raise DocumentProcessingError(f"文档处理失败: {str(e)}")
    
    async def _init_processing_status(self, document_id: uuid.UUID) -> None:
        """
        初始化处理状态
        
        Args:
            document_id: 文档ID
        """
        self.processing_status[str(document_id)] = {
            "document_id": str(document_id),
            "status": "started",
            "progress": 0.0,
            "current_step": "初始化处理",
            "total_steps": 8,
            "completed_steps": 0,
            "error_message": None,
            "started_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "estimated_completion": None
        }
    
    async def _update_processing_status(
        self, 
        document_id: uuid.UUID, 
        status: str, 
        progress: float, 
        message: str
    ) -> None:
        """
        更新处理状态
        
        Args:
            document_id: 文档ID
            status: 状态
            progress: 进度
            message: 消息
        """
        doc_id_str = str(document_id)
        if doc_id_str in self.processing_status:
            self.processing_status[doc_id_str].update({
                "status": status,
                "progress": progress,
                "current_step": message,
                "completed_steps": int(progress * 8),
                "updated_at": datetime.utcnow().isoformat()
            })
    
    async def _extract_document_content(self, document) -> str:
        """
        提取文档内容
        
        Args:
            document: 文档对象
            
        Returns:
            str: 文档内容
        """
        try:
            # 根据文档类型提取内容
            if hasattr(document, 'content') and document.content:
                return document.content
            
            # 如果没有直接内容，从文件路径读取
            if hasattr(document, 'file_path') and document.file_path:
                # 这里应该根据文件类型调用相应的内容提取器
                # 暂时返回占位符内容
                return f"从文件提取的内容: {document.file_path}"
            
            raise DocumentProcessingError("无法获取文档内容")
            
        except Exception as e:
            logger.error(f"提取文档内容失败: {str(e)}")
            raise DocumentProcessingError(f"提取文档内容失败: {str(e)}")
    
    async def _store_text_chunks(
        self, 
        document_id: uuid.UUID, 
        chunks: List[Dict[str, Any]], 
        db: Session
    ) -> None:
        """
        存储文本块
        
        Args:
            document_id: 文档ID
            chunks: 文本块列表
            db: 数据库会话
        """
        try:
            # 这里应该将文本块存储到数据库
            # 暂时只记录日志
            logger.info(f"存储 {len(chunks)} 个文本块到数据库")
            
        except Exception as e:
            logger.error(f"存储文本块失败: {str(e)}")
            raise DocumentProcessingError(f"存储文本块失败: {str(e)}")
    
    async def _extract_entities(
        self, 
        chunks: List[Dict[str, Any]], 
        document_id: uuid.UUID
    ) -> List[Dict[str, Any]]:
        """
        抽取实体
        
        Args:
            chunks: 文本块列表
            document_id: 文档ID
            
        Returns:
            List[Dict[str, Any]]: 实体列表
        """
        try:
            entities = []
            
            for chunk in chunks:
                chunk_text = chunk.get('text', '')
                if chunk_text:
                    chunk_entities = await self.entity_service.extract_entities(
                        text=chunk_text,
                        document_id=str(document_id),
                        chunk_id=chunk.get('id')
                    )
                    entities.extend(chunk_entities)
            
            return entities
            
        except Exception as e:
            logger.error(f"实体抽取失败: {str(e)}")
            raise EntityExtractionError(f"实体抽取失败: {str(e)}")
    
    async def _extract_relations(
        self, 
        chunks: List[Dict[str, Any]], 
        document_id: uuid.UUID
    ) -> List[Dict[str, Any]]:
        """
        抽取关系
        
        Args:
            chunks: 文本块列表
            document_id: 文档ID
            
        Returns:
            List[Dict[str, Any]]: 关系列表
        """
        try:
            relations = []
            
            for chunk in chunks:
                chunk_text = chunk.get('text', '')
                if chunk_text:
                    chunk_relations = await self.relation_service.extract_relations(
                        text=chunk_text,
                        document_id=str(document_id),
                        chunk_id=chunk.get('id')
                    )
                    relations.extend(chunk_relations)
            
            return relations
            
        except Exception as e:
            logger.error(f"关系抽取失败: {str(e)}")
            raise RelationExtractionError(f"关系抽取失败: {str(e)}")
    
    def get_processing_status(self, document_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """
        获取处理状态
        
        Args:
            document_id: 文档ID
            
        Returns:
            Optional[Dict[str, Any]]: 处理状态
        """
        return self.processing_status.get(str(document_id))
    
    def clear_processing_status(self, document_id: uuid.UUID) -> None:
        """
        清除处理状态
        
        Args:
            document_id: 文档ID
        """
        doc_id_str = str(document_id)
        if doc_id_str in self.processing_status:
            del self.processing_status[doc_id_str]