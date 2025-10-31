#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 文档服务
================

本模块实现了文档管理的核心业务逻辑。

服务功能：
- 文档创建、查询、更新、删除
- 文档搜索和过滤
- 文档状态管理
- 文档统计信息
- 文档处理状态跟踪

业务逻辑：
- 文档生命周期管理
- 文档权限控制
- 文档关联数据管理
- 文档处理流程协调

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc

from app.models.database.documents import Document
from app.models.database.chunks import Chunk
from app.models.database.entities import Entity
from app.models.database.relations import Relation
from app.models.database.images import Image
from app.models.schemas.documents import (
    DocumentCreate,
    DocumentUpdate,
    DocumentSearchRequest,
    DocumentStatus
)
from app.utils.exceptions import (
    DocumentNotFoundError,
    DocumentValidationError,
    DocumentPermissionError
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentService:
    """
    文档服务类
    
    提供文档管理的核心业务逻辑。
    """
    
    def __init__(self, db: Session):
        """
        初始化文档服务
        
        Args:
            db: 数据库会话
        """
        self.db = db
    
    async def create_document(
        self, 
        document_data: DocumentCreate, 
        document_id: Optional[uuid.UUID] = None
    ) -> Document:
        """
        创建文档
        
        Args:
            document_data: 文档创建数据
            document_id: 指定的文档ID（可选）
            
        Returns:
            创建的文档实例
            
        Raises:
            DocumentValidationError: 数据验证失败
        """
        try:
            # 验证数据
            await self._validate_document_data(document_data)
            
            # 创建文档实例
            document = Document(
                id=document_id or uuid.uuid4(),
                title=document_data.title,
                description=document_data.description,
                file_name=document_data.file_name,
                file_path=document_data.file_path,
                file_size=document_data.file_size,
                mime_type=document_data.mime_type,
                document_type=document_data.document_type,
                language=document_data.language,
                source_url=str(document_data.source_url) if document_data.source_url else None,
                author=document_data.author,
                tags=document_data.tags or [],
                categories=document_data.categories or [],
                is_public=document_data.is_public,
                status=DocumentStatus.UPLOADED,
                extra_data=document_data.metadata or {}
            )
            
            # 保存到数据库
            self.db.add(document)
            self.db.commit()
            self.db.refresh(document)
            
            logger.info(f"文档创建成功: {document.id}")
            return document
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"文档创建失败: {str(e)}")
            raise DocumentValidationError(f"文档创建失败: {str(e)}")
    
    async def get_document(self, document_id: uuid.UUID) -> Optional[Document]:
        """
        获取文档
        
        Args:
            document_id: 文档ID
            
        Returns:
            文档实例或None
        """
        try:
            document = self.db.query(Document).filter(
                Document.id == document_id
            ).first()
            
            if document:
                # 更新统计信息
                await self._update_document_stats(document)
            
            return document
            
        except Exception as e:
            logger.error(f"获取文档失败: {str(e)}")
            return None
    
    async def update_document(
        self, 
        document_id: uuid.UUID, 
        document_update: DocumentUpdate
    ) -> Optional[Document]:
        """
        更新文档
        
        Args:
            document_id: 文档ID
            document_update: 更新数据
            
        Returns:
            更新后的文档实例或None
            
        Raises:
            DocumentNotFoundError: 文档不存在
            DocumentValidationError: 数据验证失败
        """
        try:
            document = await self.get_document(document_id)
            if not document:
                raise DocumentNotFoundError(f"文档 {document_id} 不存在")
            
            # 更新字段
            update_data = document_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                if hasattr(document, field):
                    setattr(document, field, value)
            
            # 更新时间戳
            document.updated_at = datetime.utcnow()
            
            # 保存更改
            self.db.commit()
            self.db.refresh(document)
            
            logger.info(f"文档更新成功: {document_id}")
            return document
            
        except DocumentNotFoundError:
            raise
        except Exception as e:
            self.db.rollback()
            logger.error(f"文档更新失败: {str(e)}")
            raise DocumentValidationError(f"文档更新失败: {str(e)}")
    
    async def delete_document(
        self, 
        document_id: uuid.UUID, 
        force: bool = False
    ) -> bool:
        """
        删除文档
        
        Args:
            document_id: 文档ID
            force: 是否强制删除
            
        Returns:
            是否删除成功
            
        Raises:
            DocumentNotFoundError: 文档不存在
            DocumentPermissionError: 权限不足
        """
        try:
            document = await self.get_document(document_id)
            if not document:
                raise DocumentNotFoundError(f"文档 {document_id} 不存在")
            
            # 检查是否可以删除
            if not force and document.status == DocumentStatus.PROCESSING:
                raise DocumentPermissionError("文档正在处理中，无法删除")
            
            # 删除关联数据
            await self._delete_document_relations(document_id)
            
            # 删除文档
            self.db.delete(document)
            self.db.commit()
            
            logger.info(f"文档删除成功: {document_id}")
            return True
            
        except (DocumentNotFoundError, DocumentPermissionError):
            raise
        except Exception as e:
            self.db.rollback()
            logger.error(f"文档删除失败: {str(e)}")
            return False
    
    async def search_documents(
        self,
        search_request: DocumentSearchRequest,
        offset: int = 0,
        limit: int = 20
    ) -> Tuple[List[Document], int]:
        """
        搜索文档
        
        Args:
            search_request: 搜索请求
            offset: 偏移量
            limit: 限制数量
            
        Returns:
            文档列表和总数量的元组
        """
        try:
            query = self.db.query(Document)
            
            # 构建过滤条件
            filters = []
            
            # 关键词搜索
            if search_request.query:
                search_term = f"%{search_request.query}%"
                filters.append(
                    or_(
                        Document.title.ilike(search_term),
                        Document.description.ilike(search_term),
                        Document.author.ilike(search_term)
                    )
                )
            
            # 状态过滤
            if search_request.status:
                filters.append(Document.status.in_(search_request.status))
            
            # 文档类型过滤
            if search_request.document_type:
                filters.append(Document.document_type.in_(search_request.document_type))
            
            # 作者过滤
            if search_request.author:
                filters.append(Document.author.ilike(f"%{search_request.author}%"))
            
            # 标签过滤
            if search_request.tags:
                for tag in search_request.tags:
                    filters.append(Document.tags.contains([tag]))
            
            # 分类过滤
            if search_request.categories:
                for category in search_request.categories:
                    filters.append(Document.categories.contains([category]))
            
            # 语言过滤
            if search_request.language:
                filters.append(Document.language == search_request.language)
            
            # 公开状态过滤
            if search_request.is_public is not None:
                filters.append(Document.is_public == search_request.is_public)
            
            # 时间范围过滤
            if search_request.created_after:
                filters.append(Document.created_at >= search_request.created_after)
            
            if search_request.created_before:
                filters.append(Document.created_at <= search_request.created_before)
            
            # 文件大小过滤
            if search_request.min_file_size:
                filters.append(Document.file_size >= search_request.min_file_size)
            
            if search_request.max_file_size:
                filters.append(Document.file_size <= search_request.max_file_size)
            
            # 应用过滤条件
            if filters:
                query = query.filter(and_(*filters))
            
            # 获取总数
            total = query.count()
            
            # 排序和分页
            documents = query.order_by(desc(Document.created_at)).offset(offset).limit(limit).all()
            
            return documents, total
            
        except Exception as e:
            logger.error(f"文档搜索失败: {str(e)}")
            return [], 0
    
    async def update_document_status(
        self, 
        document_id: uuid.UUID, 
        status: DocumentStatus,
        progress: float = None,
        error_message: str = None
    ) -> bool:
        """
        更新文档状态
        
        Args:
            document_id: 文档ID
            status: 新状态
            progress: 处理进度
            error_message: 错误信息
            
        Returns:
            是否更新成功
        """
        try:
            document = await self.get_document(document_id)
            if not document:
                return False
            
            document.status = status
            if progress is not None:
                document.processing_progress = progress
            if error_message is not None:
                document.error_message = error_message
            
            # 如果完成处理，设置处理完成时间
            if status == DocumentStatus.COMPLETED:
                document.processed_at = datetime.utcnow()
                document.processing_progress = 1.0
            
            document.updated_at = datetime.utcnow()
            
            self.db.commit()
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"更新文档状态失败: {str(e)}")
            return False
    
    async def get_document_status(self, document_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """
        获取文档处理状态
        
        Args:
            document_id: 文档ID
            
        Returns:
            状态信息字典或None
        """
        try:
            document = await self.get_document(document_id)
            if not document:
                return None
            
            # 获取处理统计
            chunks_count = self.db.query(func.count(Chunk.id)).filter(
                Chunk.document_id == document_id
            ).scalar() or 0
            
            entities_count = self.db.query(func.count(Entity.id)).filter(
                Entity.document_id == document_id
            ).scalar() or 0
            
            relations_count = self.db.query(func.count(Relation.id)).filter(
                Relation.document_id == document_id
            ).scalar() or 0
            
            images_count = self.db.query(func.count(Image.id)).filter(
                Image.document_id == document_id
            ).scalar() or 0
            
            return {
                "document_id": str(document_id),
                "status": document.status,
                "progress": document.processing_progress,
                "error_message": document.error_message,
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat(),
                "processed_at": document.processed_at.isoformat() if document.processed_at else None,
                "statistics": {
                    "chunks": chunks_count,
                    "entities": entities_count,
                    "relations": relations_count,
                    "images": images_count
                }
            }
            
        except Exception as e:
            logger.error(f"获取文档状态失败: {str(e)}")
            return None
    
    async def get_document_stats(self) -> Dict[str, Any]:
        """
        获取文档统计信息
        
        Returns:
            统计信息字典
        """
        try:
            # 总文档数
            total_documents = self.db.query(func.count(Document.id)).scalar() or 0
            
            # 各状态文档数量
            status_counts = {}
            for status in DocumentStatus:
                count = self.db.query(func.count(Document.id)).filter(
                    Document.status == status
                ).scalar() or 0
                status_counts[status.value] = count
            
            # 各类型文档数量
            type_counts = {}
            type_results = self.db.query(
                Document.document_type,
                func.count(Document.id)
            ).group_by(Document.document_type).all()
            
            for doc_type, count in type_results:
                type_counts[doc_type or "unknown"] = count
            
            # 文件总大小
            total_file_size = self.db.query(
                func.coalesce(func.sum(Document.file_size), 0)
            ).scalar() or 0
            
            # 总文本块数
            total_chunks = self.db.query(func.count(Chunk.id)).scalar() or 0
            
            # 总实体数
            total_entities = self.db.query(func.count(Entity.id)).scalar() or 0
            
            # 总关系数
            total_relations = self.db.query(func.count(Relation.id)).scalar() or 0
            
            # 最近24小时上传数量
            yesterday = datetime.utcnow() - timedelta(days=1)
            recent_uploads = self.db.query(func.count(Document.id)).filter(
                Document.created_at >= yesterday
            ).scalar() or 0
            
            # 平均处理时间
            avg_processing_time = None
            completed_docs = self.db.query(Document).filter(
                and_(
                    Document.status == DocumentStatus.COMPLETED,
                    Document.processed_at.isnot(None)
                )
            ).all()
            
            if completed_docs:
                processing_times = []
                for doc in completed_docs:
                    if doc.processed_at and doc.created_at:
                        delta = doc.processed_at - doc.created_at
                        processing_times.append(delta.total_seconds())
                
                if processing_times:
                    avg_processing_time = sum(processing_times) / len(processing_times)
            
            return {
                "total_documents": total_documents,
                "status_counts": status_counts,
                "type_counts": type_counts,
                "total_file_size": total_file_size,
                "total_chunks": total_chunks,
                "total_entities": total_entities,
                "total_relations": total_relations,
                "avg_processing_time": avg_processing_time,
                "recent_uploads": recent_uploads
            }
            
        except Exception as e:
            logger.error(f"获取文档统计失败: {str(e)}")
            return {}
    
    async def _validate_document_data(self, document_data: DocumentCreate) -> None:
        """
        验证文档数据
        
        Args:
            document_data: 文档数据
            
        Raises:
            DocumentValidationError: 验证失败
        """
        # 检查标题
        if not document_data.title or not document_data.title.strip():
            raise DocumentValidationError("文档标题不能为空")
        
        # 检查文件路径或内容
        if not document_data.file_path and not document_data.content:
            raise DocumentValidationError("必须提供文件路径或文档内容")
        
        # 检查文件大小
        if document_data.file_size and document_data.file_size < 0:
            raise DocumentValidationError("文件大小不能为负数")
    
    async def _update_document_stats(self, document: Document) -> None:
        """
        更新文档统计信息
        
        Args:
            document: 文档实例
        """
        try:
            # 更新关联数据统计
            document.total_chunks = self.db.query(func.count(Chunk.id)).filter(
                Chunk.document_id == document.id
            ).scalar() or 0
            
            document.total_entities = self.db.query(func.count(Entity.id)).filter(
                Entity.document_id == document.id
            ).scalar() or 0
            
            document.total_relations = self.db.query(func.count(Relation.id)).filter(
                Relation.document_id == document.id
            ).scalar() or 0
            
            document.total_images = self.db.query(func.count(Image.id)).filter(
                Image.document_id == document.id
            ).scalar() or 0
            
        except Exception as e:
            logger.warning(f"更新文档统计失败: {str(e)}")
    
    async def _delete_document_relations(self, document_id: uuid.UUID) -> None:
        """
        删除文档关联数据
        
        Args:
            document_id: 文档ID
        """
        try:
            # 删除图像
            self.db.query(Image).filter(Image.document_id == document_id).delete()
            
            # 删除关系
            self.db.query(Relation).filter(Relation.document_id == document_id).delete()
            
            # 删除实体
            self.db.query(Entity).filter(Entity.document_id == document_id).delete()
            
            # 删除文本块
            self.db.query(Chunk).filter(Chunk.document_id == document_id).delete()
            
        except Exception as e:
            logger.error(f"删除文档关联数据失败: {str(e)}")
            raise

    async def process_document(
        self,
        document_id: uuid.UUID,
        extract_entities: bool = True,
        extract_relations: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> Dict[str, Any]:
        """
        处理文档
        
        Args:
            document_id: 文档ID
            extract_entities: 是否抽取实体
            extract_relations: 是否抽取关系
            chunk_size: 分块大小
            chunk_overlap: 分块重叠
            
        Returns:
            处理结果
        """
        try:
            # 获取文档
            document = await self.get_document(document_id)
            if not document:
                raise DocumentNotFoundError(f"文档不存在: {document_id}")
            
            # 更新状态为处理中
            await self.update_document_status(document_id, DocumentStatus.PROCESSING)
            
            # 这里应该调用处理服务进行实际处理
            # 暂时返回成功状态
            result = {
                "document_id": str(document_id),
                "status": "processing",
                "message": "文档处理已开始"
            }
            
            logger.info(f"文档处理完成: {document_id}")
            return result
            
        except Exception as e:
            logger.error(f"处理文档失败: {str(e)}")
            await self.update_document_status(document_id, DocumentStatus.FAILED, str(e))
            raise

    async def update_document_status(
        self,
        document_id: uuid.UUID,
        status: DocumentStatus,
        error_message: Optional[str] = None
    ) -> None:
        """
        更新文档状态
        
        Args:
            document_id: 文档ID
            status: 新状态
            error_message: 错误信息（可选）
        """
        try:
            document = self.db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise DocumentNotFoundError(f"文档不存在: {document_id}")
            
            document.status = status
            document.error_message = error_message
            document.updated_at = datetime.utcnow()
            
            if status == DocumentStatus.COMPLETED:
                document.processed_at = datetime.utcnow()
            
            self.db.commit()
            logger.info(f"文档状态已更新: {document_id} -> {status}")
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"更新文档状态失败: {str(e)}")
            raise

    async def get_document_content(self, document_id: uuid.UUID) -> Optional[str]:
        """
        获取文档内容
        
        Args:
            document_id: 文档ID
            
        Returns:
            文档内容或None
        """
        try:
            document = await self.get_document(document_id)
            if not document:
                return None
            
            # 从文件中读取内容
            if document.file_path and Path(document.file_path).exists():
                try:
                    return Path(document.file_path).read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    # 如果是二进制文件，返回文件信息
                    return f"二进制文件: {document.file_name}"
            
            return None
            
        except Exception as e:
            logger.error(f"获取文档内容失败: {str(e)}")
            return None

    async def get_document_chunks(self, document_id: uuid.UUID) -> List[Dict[str, Any]]:
        """
        获取文档的文本块
        
        Args:
            document_id: 文档ID
            
        Returns:
            文本块列表
        """
        try:
            chunks = self.db.query(Chunk).filter(
                Chunk.document_id == document_id
            ).order_by(Chunk.chunk_index).all()
            
            return [
                {
                    "id": str(chunk.id),
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "token_count": chunk.token_count,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "created_at": chunk.created_at.isoformat() if chunk.created_at else None
                }
                for chunk in chunks
            ]
            
        except Exception as e:
            logger.error(f"获取文档文本块失败: {str(e)}")
            return []

    async def get_document_entities(self, document_id: uuid.UUID) -> List[Dict[str, Any]]:
        """
        获取文档的实体
        
        Args:
            document_id: 文档ID
            
        Returns:
            实体列表
        """
        try:
            entities = self.db.query(Entity).filter(
                Entity.document_id == document_id
            ).all()
            
            return [
                {
                    "id": str(entity.id),
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "description": entity.description,
                    "properties": entity.properties,
                    "created_at": entity.created_at.isoformat() if entity.created_at else None
                }
                for entity in entities
            ]
            
        except Exception as e:
            logger.error(f"获取文档实体失败: {str(e)}")
            return []

    async def get_document_relations(self, document_id: uuid.UUID) -> List[Dict[str, Any]]:
        """
        获取文档的关系
        
        Args:
            document_id: 文档ID
            
        Returns:
            关系列表
        """
        try:
            relations = self.db.query(Relation).filter(
                Relation.document_id == document_id
            ).all()
            
            return [
                {
                    "id": str(relation.id),
                    "source_entity": relation.source_entity,
                    "target_entity": relation.target_entity,
                    "relation_type": relation.relation_type,
                    "description": relation.description,
                    "properties": relation.properties,
                    "confidence": relation.confidence,
                    "created_at": relation.created_at.isoformat() if relation.created_at else None
                }
                for relation in relations
            ]
            
        except Exception as e:
            logger.error(f"获取文档关系失败: {str(e)}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        检查文档服务的健康状态，包括数据库连接和基本功能。
        
        Returns:
            健康状态信息
        """
        try:
            # 检查数据库连接
            db_status = "healthy"
            db_error = None
            
            try:
                # 执行简单查询测试数据库连接
                self.db.execute("SELECT 1")
                
                # 检查文档表是否可访问
                document_count = self.db.query(func.count(Document.id)).scalar()
                
            except Exception as e:
                db_status = "unhealthy"
                db_error = str(e)
                logger.error(f"数据库健康检查失败: {str(e)}")
            
            # 检查服务功能
            service_status = "healthy"
            service_error = None
            
            try:
                # 测试基本统计功能
                stats = await self.get_document_stats()
                if not isinstance(stats, dict):
                    raise Exception("统计功能异常")
                    
            except Exception as e:
                service_status = "unhealthy"
                service_error = str(e)
                logger.error(f"服务功能检查失败: {str(e)}")
            
            # 整体健康状态
            overall_status = "healthy" if db_status == "healthy" and service_status == "healthy" else "unhealthy"
            
            return {
                "service": "DocumentService",
                "status": overall_status,
                "timestamp": datetime.utcnow().isoformat(),
                "checks": {
                    "database": {
                        "status": db_status,
                        "error": db_error
                    },
                    "service_functions": {
                        "status": service_status,
                        "error": service_error
                    }
                },
                "metadata": {
                    "total_documents": document_count if db_status == "healthy" else None,
                    "service_version": "1.0.0"
                }
            }
            
        except Exception as e:
            logger.error(f"健康检查执行失败: {str(e)}")
            return {
                "service": "DocumentService",
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "checks": {
                    "database": {"status": "unknown"},
                    "service_functions": {"status": "unknown"}
                }
            }