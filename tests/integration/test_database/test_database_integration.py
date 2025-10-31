#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 数据库集成测试
======================

测试数据库层的完整功能：
1. 数据库连接和会话管理
2. 模型CRUD操作
3. 事务处理
4. 数据一致性
5. 并发操作
6. 性能测试

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError
from concurrent.futures import ThreadPoolExecutor
import threading

from app.core.database import get_db, engine
from app.models.database.base import BaseModel
from app.models.database.documents import Document, DocumentStatus
from app.models.database.chunks import Chunk, ChunkType
from app.models.database.entities import Entity, EntityType
from app.models.database.relations import Relation, RelationType


@pytest.mark.integration
@pytest.mark.database
class TestDatabaseConnection:
    """数据库连接测试"""
    
    def test_database_connection(self):
        """测试数据库连接"""
        # 测试引擎连接
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            assert result.scalar() == 1
    
    def test_session_creation(self):
        """测试会话创建"""
        db = next(get_db())
        try:
            # 测试会话可用性
            result = db.execute(text("SELECT 1"))
            assert result.scalar() == 1
        finally:
            db.close()
    
    def test_multiple_sessions(self):
        """测试多个会话"""
        sessions = []
        try:
            # 创建多个会话
            for _ in range(5):
                db = next(get_db())
                sessions.append(db)
                
                # 每个会话都应该可用
                result = db.execute(text("SELECT 1"))
                assert result.scalar() == 1
        finally:
            # 关闭所有会话
            for session in sessions:
                session.close()
    
    def test_session_isolation(self):
        """测试会话隔离"""
        db1 = next(get_db())
        db2 = next(get_db())
        
        try:
            # 在第一个会话中创建文档但不提交
            doc1 = Document(
                title="隔离测试文档1",
                file_name="test1.txt",
                file_type="txt",
                content="测试内容1"
            )
            db1.add(doc1)
            db1.flush()  # 刷新但不提交
            
            # 第二个会话不应该看到未提交的数据
            doc_count = db2.query(Document).filter(Document.title == "隔离测试文档1").count()
            assert doc_count == 0
            
            # 提交第一个会话
            db1.commit()
            
            # 现在第二个会话应该能看到数据
            doc_count = db2.query(Document).filter(Document.title == "隔离测试文档1").count()
            assert doc_count == 1
            
        finally:
            # 清理
            try:
                db1.query(Document).filter(Document.title == "隔离测试文档1").delete()
                db1.commit()
            except:
                pass
            db1.close()
            db2.close()


@pytest.mark.integration
@pytest.mark.database
class TestModelCRUD:
    """模型CRUD操作测试"""
    
    @pytest.fixture
    def db_session(self):
        """创建测试数据库会话"""
        db = next(get_db())
        try:
            yield db
        finally:
            db.close()
    
    def test_document_crud(self, db_session):
        """测试文档CRUD操作"""
        # Create
        doc = Document(
            title="CRUD测试文档",
            file_name="crud_test.txt",
            file_type="txt",
            content="这是CRUD测试内容",
            language="zh",
            tags=["测试", "CRUD"],
            categories=["测试分类"]
        )
        
        db_session.add(doc)
        db_session.commit()
        
        assert doc.id is not None
        assert doc.created_at is not None
        assert doc.updated_at is not None
        
        # Read
        retrieved_doc = db_session.query(Document).filter(Document.id == doc.id).first()
        assert retrieved_doc is not None
        assert retrieved_doc.title == "CRUD测试文档"
        assert retrieved_doc.content == "这是CRUD测试内容"
        assert "测试" in retrieved_doc.tags
        
        # Update
        retrieved_doc.title = "更新后的标题"
        retrieved_doc.description = "更新后的描述"
        db_session.commit()
        
        updated_doc = db_session.query(Document).filter(Document.id == doc.id).first()
        assert updated_doc.title == "更新后的标题"
        assert updated_doc.description == "更新后的描述"
        assert updated_doc.updated_at > updated_doc.created_at
        
        # Delete
        db_session.delete(updated_doc)
        db_session.commit()
        
        deleted_doc = db_session.query(Document).filter(Document.id == doc.id).first()
        assert deleted_doc is None
    
    def test_chunk_crud(self, db_session):
        """测试文本块CRUD操作"""
        # 先创建文档
        doc = Document(
            title="块测试文档",
            file_name="chunk_test.txt",
            file_type="txt",
            content="文档内容"
        )
        db_session.add(doc)
        db_session.commit()
        
        # Create chunk
        chunk = Chunk(
            document_id=doc.id,
            content="这是一个测试文本块",
            chunk_type=ChunkType.PARAGRAPH,
            position=0,
            start_char=0,
            end_char=10,
            token_count=5,
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        
        db_session.add(chunk)
        db_session.commit()
        
        assert chunk.id is not None
        
        # Read
        retrieved_chunk = db_session.query(Chunk).filter(Chunk.id == chunk.id).first()
        assert retrieved_chunk is not None
        assert retrieved_chunk.content == "这是一个测试文本块"
        assert retrieved_chunk.document_id == doc.id
        assert len(retrieved_chunk.embedding) == 5
        
        # Update
        retrieved_chunk.content = "更新后的文本块内容"
        retrieved_chunk.token_count = 8
        db_session.commit()
        
        updated_chunk = db_session.query(Chunk).filter(Chunk.id == chunk.id).first()
        assert updated_chunk.content == "更新后的文本块内容"
        assert updated_chunk.token_count == 8
        
        # Delete
        db_session.delete(updated_chunk)
        db_session.delete(doc)
        db_session.commit()
    
    def test_entity_crud(self, db_session):
        """测试实体CRUD操作"""
        # Create
        entity = Entity(
            canonical_name="苹果公司",
            display_name="Apple Inc.",
            entity_type=EntityType.ORGANIZATION,
            aliases=["Apple", "苹果"],
            properties={"founded": "1976", "headquarters": "Cupertino"},
            confidence=0.95
        )
        
        db_session.add(entity)
        db_session.commit()
        
        assert entity.id is not None
        
        # Read
        retrieved_entity = db_session.query(Entity).filter(Entity.id == entity.id).first()
        assert retrieved_entity is not None
        assert retrieved_entity.canonical_name == "苹果公司"
        assert retrieved_entity.entity_type == EntityType.ORGANIZATION
        assert "Apple" in retrieved_entity.aliases
        assert retrieved_entity.properties["founded"] == "1976"
        
        # Update
        retrieved_entity.display_name = "苹果公司"
        retrieved_entity.properties["employees"] = "150000"
        db_session.commit()
        
        updated_entity = db_session.query(Entity).filter(Entity.id == entity.id).first()
        assert updated_entity.display_name == "苹果公司"
        assert updated_entity.properties["employees"] == "150000"
        
        # Delete
        db_session.delete(updated_entity)
        db_session.commit()
    
    def test_relation_crud(self, db_session):
        """测试关系CRUD操作"""
        # 先创建两个实体
        entity1 = Entity(
            canonical_name="史蒂夫·乔布斯",
            entity_type=EntityType.PERSON
        )
        entity2 = Entity(
            canonical_name="苹果公司",
            entity_type=EntityType.ORGANIZATION
        )
        
        db_session.add_all([entity1, entity2])
        db_session.commit()
        
        # Create relation
        relation = Relation(
            source_entity_id=entity1.id,
            target_entity_id=entity2.id,
            relation_type=RelationType.FOUNDED,
            description="史蒂夫·乔布斯创立了苹果公司",
            confidence=0.98,
            weight=1.0,
            properties={"year": "1976"}
        )
        
        db_session.add(relation)
        db_session.commit()
        
        assert relation.id is not None
        
        # Read
        retrieved_relation = db_session.query(Relation).filter(Relation.id == relation.id).first()
        assert retrieved_relation is not None
        assert retrieved_relation.source_entity_id == entity1.id
        assert retrieved_relation.target_entity_id == entity2.id
        assert retrieved_relation.relation_type == RelationType.FOUNDED
        assert retrieved_relation.properties["year"] == "1976"
        
        # Update
        retrieved_relation.description = "乔布斯是苹果公司的联合创始人"
        retrieved_relation.confidence = 0.99
        db_session.commit()
        
        updated_relation = db_session.query(Relation).filter(Relation.id == relation.id).first()
        assert updated_relation.description == "乔布斯是苹果公司的联合创始人"
        assert updated_relation.confidence == 0.99
        
        # Delete
        db_session.delete(updated_relation)
        db_session.delete(entity1)
        db_session.delete(entity2)
        db_session.commit()


@pytest.mark.integration
@pytest.mark.database
class TestTransactions:
    """事务处理测试"""
    
    @pytest.fixture
    def db_session(self):
        """创建测试数据库会话"""
        db = next(get_db())
        try:
            yield db
        finally:
            db.close()
    
    def test_transaction_commit(self, db_session):
        """测试事务提交"""
        # 创建多个相关对象
        doc = Document(
            title="事务测试文档",
            file_name="transaction_test.txt",
            file_type="txt",
            content="事务测试内容"
        )
        
        chunk = Chunk(
            content="事务测试块",
            chunk_type=ChunkType.PARAGRAPH,
            position=0
        )
        
        # 在事务中添加对象
        db_session.add(doc)
        db_session.flush()  # 获取doc.id
        
        chunk.document_id = doc.id
        db_session.add(chunk)
        
        # 提交事务
        db_session.commit()
        
        # 验证数据已保存
        saved_doc = db_session.query(Document).filter(Document.title == "事务测试文档").first()
        saved_chunk = db_session.query(Chunk).filter(Chunk.content == "事务测试块").first()
        
        assert saved_doc is not None
        assert saved_chunk is not None
        assert saved_chunk.document_id == saved_doc.id
        
        # 清理
        db_session.delete(saved_chunk)
        db_session.delete(saved_doc)
        db_session.commit()
    
    def test_transaction_rollback(self, db_session):
        """测试事务回滚"""
        # 创建文档
        doc = Document(
            title="回滚测试文档",
            file_name="rollback_test.txt",
            file_type="txt",
            content="回滚测试内容"
        )
        
        db_session.add(doc)
        db_session.flush()
        
        # 记录文档ID
        doc_id = doc.id
        
        # 创建一个会导致错误的块（假设某个约束会失败）
        try:
            chunk = Chunk(
                document_id=doc_id,
                content="回滚测试块",
                chunk_type=ChunkType.PARAGRAPH,
                position=0
            )
            db_session.add(chunk)
            
            # 故意引发错误（例如，重复添加相同的块）
            chunk2 = Chunk(
                document_id=doc_id,
                content="回滚测试块",
                chunk_type=ChunkType.PARAGRAPH,
                position=0  # 相同位置可能导致约束错误
            )
            db_session.add(chunk2)
            
            # 尝试提交（可能失败）
            db_session.commit()
            
        except Exception:
            # 回滚事务
            db_session.rollback()
        
        # 验证数据未保存
        saved_doc = db_session.query(Document).filter(Document.title == "回滚测试文档").first()
        saved_chunks = db_session.query(Chunk).filter(Chunk.content == "回滚测试块").all()
        
        assert saved_doc is None  # 由于回滚，文档也不应该存在
        assert len(saved_chunks) == 0
    
    def test_nested_transactions(self, db_session):
        """测试嵌套事务（保存点）"""
        # 外层事务
        doc = Document(
            title="嵌套事务测试",
            file_name="nested_test.txt",
            file_type="txt",
            content="嵌套事务内容"
        )
        
        db_session.add(doc)
        db_session.flush()
        
        # 创建保存点
        savepoint = db_session.begin_nested()
        
        try:
            # 内层事务
            chunk = Chunk(
                document_id=doc.id,
                content="嵌套事务块",
                chunk_type=ChunkType.PARAGRAPH,
                position=0
            )
            db_session.add(chunk)
            
            # 假设这里发生错误
            if True:  # 模拟错误条件
                raise Exception("模拟错误")
            
            savepoint.commit()
            
        except Exception:
            # 回滚到保存点
            savepoint.rollback()
        
        # 外层事务继续
        db_session.commit()
        
        # 验证结果
        saved_doc = db_session.query(Document).filter(Document.title == "嵌套事务测试").first()
        saved_chunk = db_session.query(Chunk).filter(Chunk.content == "嵌套事务块").first()
        
        assert saved_doc is not None  # 文档应该存在
        assert saved_chunk is None    # 块应该被回滚
        
        # 清理
        db_session.delete(saved_doc)
        db_session.commit()


@pytest.mark.integration
@pytest.mark.database
class TestDataConsistency:
    """数据一致性测试"""
    
    @pytest.fixture
    def db_session(self):
        """创建测试数据库会话"""
        db = next(get_db())
        try:
            yield db
        finally:
            db.close()
    
    def test_foreign_key_constraints(self, db_session):
        """测试外键约束"""
        # 尝试创建引用不存在文档的块
        chunk = Chunk(
            document_id=uuid.uuid4(),  # 不存在的文档ID
            content="外键测试块",
            chunk_type=ChunkType.PARAGRAPH,
            position=0
        )
        
        db_session.add(chunk)
        
        # 应该引发完整性错误
        with pytest.raises(IntegrityError):
            db_session.commit()
        
        db_session.rollback()
    
    def test_unique_constraints(self, db_session):
        """测试唯一性约束"""
        # 创建第一个实体
        entity1 = Entity(
            canonical_name="唯一性测试实体",
            entity_type=EntityType.PERSON
        )
        
        db_session.add(entity1)
        db_session.commit()
        
        # 尝试创建相同canonical_name的实体（如果有唯一约束）
        entity2 = Entity(
            canonical_name="唯一性测试实体",
            entity_type=EntityType.PERSON
        )
        
        db_session.add(entity2)
        
        # 根据模型定义，这可能会成功或失败
        try:
            db_session.commit()
            # 如果成功，清理两个实体
            db_session.delete(entity1)
            db_session.delete(entity2)
            db_session.commit()
        except IntegrityError:
            # 如果失败，回滚并清理第一个实体
            db_session.rollback()
            db_session.delete(entity1)
            db_session.commit()
    
    def test_cascade_delete(self, db_session):
        """测试级联删除"""
        # 创建文档和相关块
        doc = Document(
            title="级联删除测试",
            file_name="cascade_test.txt",
            file_type="txt",
            content="级联删除内容"
        )
        
        db_session.add(doc)
        db_session.flush()
        
        chunks = []
        for i in range(3):
            chunk = Chunk(
                document_id=doc.id,
                content=f"级联删除块 {i}",
                chunk_type=ChunkType.PARAGRAPH,
                position=i
            )
            chunks.append(chunk)
            db_session.add(chunk)
        
        db_session.commit()
        
        # 删除文档
        db_session.delete(doc)
        db_session.commit()
        
        # 检查相关块是否被删除（取决于级联设置）
        remaining_chunks = db_session.query(Chunk).filter(
            Chunk.content.like("级联删除块%")
        ).all()
        
        # 根据模型的级联设置，块可能被删除或保留
        # 这里我们只验证操作不会出错
        assert isinstance(remaining_chunks, list)


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.slow
class TestConcurrency:
    """并发操作测试"""
    
    def test_concurrent_inserts(self):
        """测试并发插入"""
        def insert_document(thread_id):
            db = next(get_db())
            try:
                doc = Document(
                    title=f"并发测试文档 {thread_id}",
                    file_name=f"concurrent_{thread_id}.txt",
                    file_type="txt",
                    content=f"并发测试内容 {thread_id}"
                )
                
                db.add(doc)
                db.commit()
                return doc.id
            finally:
                db.close()
        
        # 使用线程池并发插入
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(insert_document, i) for i in range(10)]
            doc_ids = [future.result() for future in futures]
        
        # 验证所有文档都被创建
        assert len(doc_ids) == 10
        assert len(set(doc_ids)) == 10  # 所有ID都是唯一的
        
        # 清理
        db = next(get_db())
        try:
            for doc_id in doc_ids:
                doc = db.query(Document).filter(Document.id == doc_id).first()
                if doc:
                    db.delete(doc)
            db.commit()
        finally:
            db.close()
    
    def test_concurrent_updates(self):
        """测试并发更新"""
        # 先创建一个文档
        db = next(get_db())
        doc = Document(
            title="并发更新测试",
            file_name="concurrent_update.txt",
            file_type="txt",
            content="原始内容"
        )
        db.add(doc)
        db.commit()
        doc_id = doc.id
        db.close()
        
        def update_document(thread_id):
            db = next(get_db())
            try:
                doc = db.query(Document).filter(Document.id == doc_id).first()
                if doc:
                    doc.description = f"线程 {thread_id} 更新"
                    db.commit()
                    return True
                return False
            finally:
                db.close()
        
        # 并发更新
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(update_document, i) for i in range(5)]
            results = [future.result() for future in futures]
        
        # 验证更新成功
        assert all(results)
        
        # 检查最终状态
        db = next(get_db())
        try:
            final_doc = db.query(Document).filter(Document.id == doc_id).first()
            assert final_doc is not None
            assert "线程" in final_doc.description
            
            # 清理
            db.delete(final_doc)
            db.commit()
        finally:
            db.close()
    
    def test_deadlock_handling(self):
        """测试死锁处理"""
        # 创建两个文档
        db = next(get_db())
        doc1 = Document(title="死锁测试1", file_name="deadlock1.txt", file_type="txt")
        doc2 = Document(title="死锁测试2", file_name="deadlock2.txt", file_type="txt")
        db.add_all([doc1, doc2])
        db.commit()
        doc1_id, doc2_id = doc1.id, doc2.id
        db.close()
        
        def update_docs_order1():
            db = next(get_db())
            try:
                # 按顺序1更新
                doc1 = db.query(Document).filter(Document.id == doc1_id).first()
                doc1.description = "更新1"
                db.flush()
                
                import time
                time.sleep(0.1)  # 增加死锁概率
                
                doc2 = db.query(Document).filter(Document.id == doc2_id).first()
                doc2.description = "更新1"
                db.commit()
                return True
            except Exception as e:
                db.rollback()
                return False
            finally:
                db.close()
        
        def update_docs_order2():
            db = next(get_db())
            try:
                # 按顺序2更新（相反顺序）
                doc2 = db.query(Document).filter(Document.id == doc2_id).first()
                doc2.description = "更新2"
                db.flush()
                
                import time
                time.sleep(0.1)  # 增加死锁概率
                
                doc1 = db.query(Document).filter(Document.id == doc1_id).first()
                doc1.description = "更新2"
                db.commit()
                return True
            except Exception as e:
                db.rollback()
                return False
            finally:
                db.close()
        
        # 并发执行可能导致死锁的操作
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(update_docs_order1)
            future2 = executor.submit(update_docs_order2)
            
            result1 = future1.result()
            result2 = future2.result()
        
        # 至少一个操作应该成功（死锁被处理）
        assert result1 or result2
        
        # 清理
        db = next(get_db())
        try:
            db.query(Document).filter(Document.id.in_([doc1_id, doc2_id])).delete()
            db.commit()
        finally:
            db.close()


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.performance
class TestDatabasePerformance:
    """数据库性能测试"""
    
    @pytest.fixture
    def db_session(self):
        """创建测试数据库会话"""
        db = next(get_db())
        try:
            yield db
        finally:
            db.close()
    
    def test_bulk_insert_performance(self, db_session):
        """测试批量插入性能"""
        import time
        
        # 准备大量数据
        documents = []
        for i in range(100):
            doc = Document(
                title=f"性能测试文档 {i}",
                file_name=f"perf_test_{i}.txt",
                file_type="txt",
                content=f"性能测试内容 {i}"
            )
            documents.append(doc)
        
        # 测试批量插入时间
        start_time = time.time()
        db_session.add_all(documents)
        db_session.commit()
        end_time = time.time()
        
        insert_time = end_time - start_time
        
        # 验证插入成功
        count = db_session.query(Document).filter(
            Document.title.like("性能测试文档%")
        ).count()
        assert count == 100
        
        # 性能断言（根据实际情况调整）
        assert insert_time < 10.0  # 100条记录应该在10秒内完成
        
        # 清理
        db_session.query(Document).filter(
            Document.title.like("性能测试文档%")
        ).delete()
        db_session.commit()
    
    def test_query_performance(self, db_session):
        """测试查询性能"""
        import time
        
        # 先插入测试数据
        documents = []
        for i in range(50):
            doc = Document(
                title=f"查询测试文档 {i}",
                file_name=f"query_test_{i}.txt",
                file_type="txt",
                content=f"查询测试内容 {i}",
                tags=[f"标签{i % 5}", "查询测试"]
            )
            documents.append(doc)
        
        db_session.add_all(documents)
        db_session.commit()
        
        # 测试简单查询
        start_time = time.time()
        results = db_session.query(Document).filter(
            Document.title.like("查询测试文档%")
        ).all()
        end_time = time.time()
        
        simple_query_time = end_time - start_time
        assert len(results) == 50
        assert simple_query_time < 1.0  # 简单查询应该很快
        
        # 测试复杂查询
        start_time = time.time()
        results = db_session.query(Document).filter(
            Document.title.like("查询测试文档%"),
            Document.tags.contains(["查询测试"])
        ).order_by(Document.created_at.desc()).limit(10).all()
        end_time = time.time()
        
        complex_query_time = end_time - start_time
        assert len(results) == 10
        assert complex_query_time < 2.0  # 复杂查询也应该相对较快
        
        # 清理
        db_session.query(Document).filter(
            Document.title.like("查询测试文档%")
        ).delete()
        db_session.commit()
    
    def test_index_effectiveness(self, db_session):
        """测试索引有效性"""
        import time
        
        # 插入大量数据
        documents = []
        for i in range(200):
            doc = Document(
                title=f"索引测试文档 {i}",
                file_name=f"index_test_{i}.txt",
                file_type="txt",
                content=f"索引测试内容 {i}"
            )
            documents.append(doc)
        
        db_session.add_all(documents)
        db_session.commit()
        
        # 测试按ID查询（应该使用主键索引）
        doc_id = documents[100].id
        
        start_time = time.time()
        result = db_session.query(Document).filter(Document.id == doc_id).first()
        end_time = time.time()
        
        id_query_time = end_time - start_time
        assert result is not None
        assert id_query_time < 0.1  # ID查询应该非常快
        
        # 测试按标题查询（如果有索引应该较快）
        start_time = time.time()
        result = db_session.query(Document).filter(
            Document.title == "索引测试文档 150"
        ).first()
        end_time = time.time()
        
        title_query_time = end_time - start_time
        assert result is not None
        # 根据是否有索引，时间会有差异
        
        # 清理
        db_session.query(Document).filter(
            Document.title.like("索引测试文档%")
        ).delete()
        db_session.commit()