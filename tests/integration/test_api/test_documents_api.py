#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 文档API集成测试
=======================

测试文档管理API的完整功能：
1. 文档上传和创建
2. 文档查询和搜索
3. 文档更新和删除
4. 文档状态管理
5. 文档处理流程
6. 错误处理和边界情况

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import pytest
import uuid
import json
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from io import BytesIO

from app.main import app
from app.models.database.documents import Document
from app.models.schemas.documents import DocumentStatus


@pytest.mark.integration
@pytest.mark.api
class TestDocumentsAPI:
    """文档API集成测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_text_file(self):
        """创建示例文本文件"""
        content = "这是一个测试文档的内容。\n包含多行文本用于测试文档处理功能。"
        return ("test_document.txt", content.encode(), "text/plain")
    
    @pytest.fixture
    def sample_pdf_file(self):
        """创建示例PDF文件（模拟）"""
        # 这里使用简单的字节内容模拟PDF
        content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        return ("test_document.pdf", content, "application/pdf")
    
    def test_upload_document_success(self, client, sample_text_file):
        """测试成功上传文档"""
        filename, content, content_type = sample_text_file
        
        files = {"file": (filename, BytesIO(content), content_type)}
        data = {
            "title": "测试文档",
            "description": "这是一个测试文档",
            "tags": json.dumps(["测试", "文档"]),
            "categories": json.dumps(["技术文档"]),
            "language": "zh"
        }
        
        response = client.post("/api/v1/documents/upload", files=files, data=data)
        
        assert response.status_code == 201
        result = response.json()
        
        assert "id" in result
        assert result["title"] == "测试文档"
        assert result["description"] == "这是一个测试文档"
        assert result["file_name"] == filename
        assert result["file_type"] == "txt"
        assert result["language"] == "zh"
        assert result["status"] == DocumentStatus.UPLOADED.value
        assert "测试" in result["tags"]
        assert "技术文档" in result["categories"]
    
    def test_upload_document_pdf(self, client, sample_pdf_file):
        """测试上传PDF文档"""
        filename, content, content_type = sample_pdf_file
        
        files = {"file": (filename, BytesIO(content), content_type)}
        data = {"title": "PDF测试文档"}
        
        response = client.post("/api/v1/documents/upload", files=files, data=data)
        
        assert response.status_code == 201
        result = response.json()
        
        assert result["file_type"] == "pdf"
        assert result["file_name"] == filename
    
    def test_upload_document_without_file(self, client):
        """测试不上传文件"""
        data = {"title": "无文件测试"}
        
        response = client.post("/api/v1/documents/upload", data=data)
        
        assert response.status_code == 422  # Validation error
    
    def test_upload_document_invalid_file_type(self, client):
        """测试上传无效文件类型"""
        files = {"file": ("malware.exe", BytesIO(b"malicious"), "application/x-executable")}
        data = {"title": "恶意文件"}
        
        response = client.post("/api/v1/documents/upload", files=files, data=data)
        
        assert response.status_code == 400
        assert "不支持的文件类型" in response.json()["detail"]
    
    def test_upload_document_too_large(self, client):
        """测试上传过大文件"""
        # 创建一个模拟的大文件（实际测试中可能需要调整大小限制）
        large_content = b"x" * (200 * 1024 * 1024)  # 200MB
        files = {"file": ("large.txt", BytesIO(large_content), "text/plain")}
        data = {"title": "大文件"}
        
        response = client.post("/api/v1/documents/upload", files=files, data=data)
        
        assert response.status_code == 400
        assert "文件大小超过限制" in response.json()["detail"]
    
    def test_get_document_by_id(self, client, sample_text_file):
        """测试通过ID获取文档"""
        # 先上传文档
        filename, content, content_type = sample_text_file
        files = {"file": (filename, BytesIO(content), content_type)}
        data = {"title": "获取测试文档"}
        
        upload_response = client.post("/api/v1/documents/upload", files=files, data=data)
        document_id = upload_response.json()["id"]
        
        # 获取文档
        response = client.get(f"/api/v1/documents/{document_id}")
        
        assert response.status_code == 200
        result = response.json()
        
        assert result["id"] == document_id
        assert result["title"] == "获取测试文档"
        assert result["file_name"] == filename
    
    def test_get_document_not_found(self, client):
        """测试获取不存在的文档"""
        non_existent_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/documents/{non_existent_id}")
        
        assert response.status_code == 404
        assert "文档不存在" in response.json()["detail"]
    
    def test_get_document_invalid_id(self, client):
        """测试使用无效ID获取文档"""
        invalid_id = "invalid-uuid"
        
        response = client.get(f"/api/v1/documents/{invalid_id}")
        
        assert response.status_code == 422  # Validation error
    
    def test_list_documents_empty(self, client):
        """测试列出空文档列表"""
        response = client.get("/api/v1/documents/")
        
        assert response.status_code == 200
        result = response.json()
        
        assert "items" in result
        assert "total" in result
        assert "page" in result
        assert "size" in result
        assert isinstance(result["items"], list)
    
    def test_list_documents_with_pagination(self, client, sample_text_file):
        """测试分页列出文档"""
        # 上传多个文档
        filename, content, content_type = sample_text_file
        for i in range(5):
            files = {"file": (f"doc_{i}.txt", BytesIO(content), content_type)}
            data = {"title": f"文档 {i}"}
            client.post("/api/v1/documents/upload", files=files, data=data)
        
        # 测试分页
        response = client.get("/api/v1/documents/?page=1&size=3")
        
        assert response.status_code == 200
        result = response.json()
        
        assert len(result["items"]) <= 3
        assert result["page"] == 1
        assert result["size"] == 3
        assert result["total"] >= 5
    
    def test_search_documents_by_title(self, client, sample_text_file):
        """测试按标题搜索文档"""
        # 上传测试文档
        filename, content, content_type = sample_text_file
        files = {"file": (filename, BytesIO(content), content_type)}
        data = {"title": "搜索测试文档"}
        
        upload_response = client.post("/api/v1/documents/upload", files=files, data=data)
        
        # 搜索文档
        response = client.get("/api/v1/documents/search?query=搜索测试")
        
        assert response.status_code == 200
        result = response.json()
        
        assert len(result["items"]) >= 1
        found_doc = next((doc for doc in result["items"] if doc["title"] == "搜索测试文档"), None)
        assert found_doc is not None
    
    def test_search_documents_by_tags(self, client, sample_text_file):
        """测试按标签搜索文档"""
        # 上传带标签的文档
        filename, content, content_type = sample_text_file
        files = {"file": (filename, BytesIO(content), content_type)}
        data = {
            "title": "标签测试文档",
            "tags": json.dumps(["特殊标签", "测试"])
        }
        
        client.post("/api/v1/documents/upload", files=files, data=data)
        
        # 按标签搜索
        response = client.get("/api/v1/documents/search?tags=特殊标签")
        
        assert response.status_code == 200
        result = response.json()
        
        assert len(result["items"]) >= 1
        found_doc = next((doc for doc in result["items"] if "特殊标签" in doc["tags"]), None)
        assert found_doc is not None
    
    def test_update_document(self, client, sample_text_file):
        """测试更新文档"""
        # 先上传文档
        filename, content, content_type = sample_text_file
        files = {"file": (filename, BytesIO(content), content_type)}
        data = {"title": "原始标题"}
        
        upload_response = client.post("/api/v1/documents/upload", files=files, data=data)
        document_id = upload_response.json()["id"]
        
        # 更新文档
        update_data = {
            "title": "更新后标题",
            "description": "更新后描述",
            "tags": ["更新", "测试"],
            "categories": ["更新分类"]
        }
        
        response = client.put(f"/api/v1/documents/{document_id}", json=update_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert result["title"] == "更新后标题"
        assert result["description"] == "更新后描述"
        assert "更新" in result["tags"]
        assert "更新分类" in result["categories"]
    
    def test_update_document_not_found(self, client):
        """测试更新不存在的文档"""
        non_existent_id = str(uuid.uuid4())
        update_data = {"title": "不存在的文档"}
        
        response = client.put(f"/api/v1/documents/{non_existent_id}", json=update_data)
        
        assert response.status_code == 404
        assert "文档不存在" in response.json()["detail"]
    
    def test_delete_document(self, client, sample_text_file):
        """测试删除文档"""
        # 先上传文档
        filename, content, content_type = sample_text_file
        files = {"file": (filename, BytesIO(content), content_type)}
        data = {"title": "待删除文档"}
        
        upload_response = client.post("/api/v1/documents/upload", files=files, data=data)
        document_id = upload_response.json()["id"]
        
        # 删除文档
        response = client.delete(f"/api/v1/documents/{document_id}")
        
        assert response.status_code == 200
        result = response.json()
        assert result["message"] == "文档删除成功"
        
        # 验证文档已删除
        get_response = client.get(f"/api/v1/documents/{document_id}")
        assert get_response.status_code == 404
    
    def test_delete_document_not_found(self, client):
        """测试删除不存在的文档"""
        non_existent_id = str(uuid.uuid4())
        
        response = client.delete(f"/api/v1/documents/{non_existent_id}")
        
        assert response.status_code == 404
        assert "文档不存在" in response.json()["detail"]
    
    def test_get_document_content(self, client, sample_text_file):
        """测试获取文档内容"""
        # 先上传文档
        filename, content, content_type = sample_text_file
        files = {"file": (filename, BytesIO(content), content_type)}
        data = {"title": "内容测试文档"}
        
        upload_response = client.post("/api/v1/documents/upload", files=files, data=data)
        document_id = upload_response.json()["id"]
        
        # 获取文档内容
        response = client.get(f"/api/v1/documents/{document_id}/content")
        
        assert response.status_code == 200
        result = response.json()
        
        assert "content" in result
        assert "chunks" in result
        # 注意：实际内容可能经过处理，这里主要验证接口可用
    
    def test_get_document_status(self, client, sample_text_file):
        """测试获取文档状态"""
        # 先上传文档
        filename, content, content_type = sample_text_file
        files = {"file": (filename, BytesIO(content), content_type)}
        data = {"title": "状态测试文档"}
        
        upload_response = client.post("/api/v1/documents/upload", files=files, data=data)
        document_id = upload_response.json()["id"]
        
        # 获取文档状态
        response = client.get(f"/api/v1/documents/{document_id}/status")
        
        assert response.status_code == 200
        result = response.json()
        
        assert "status" in result
        assert "processing_progress" in result
        assert "created_at" in result
        assert "updated_at" in result
    
    def test_process_document(self, client, sample_text_file):
        """测试处理文档"""
        # 先上传文档
        filename, content, content_type = sample_text_file
        files = {"file": (filename, BytesIO(content), content_type)}
        data = {"title": "处理测试文档"}
        
        upload_response = client.post("/api/v1/documents/upload", files=files, data=data)
        document_id = upload_response.json()["id"]
        
        # 开始处理文档
        response = client.post(f"/api/v1/documents/{document_id}/process")
        
        assert response.status_code == 200
        result = response.json()
        
        assert "message" in result
        assert "task_id" in result or "status" in result
    
    def test_batch_upload_documents(self, client, sample_text_file):
        """测试批量上传文档"""
        filename, content, content_type = sample_text_file
        
        # 准备多个文件
        files = [
            ("files", (f"batch_doc_1.txt", BytesIO(content), content_type)),
            ("files", (f"batch_doc_2.txt", BytesIO(content), content_type)),
            ("files", (f"batch_doc_3.txt", BytesIO(content), content_type))
        ]
        
        data = {
            "titles": json.dumps(["批量文档1", "批量文档2", "批量文档3"]),
            "tags": json.dumps(["批量", "测试"])
        }
        
        response = client.post("/api/v1/documents/batch-upload", files=files, data=data)
        
        assert response.status_code == 201
        result = response.json()
        
        assert "uploaded_documents" in result
        assert len(result["uploaded_documents"]) == 3
        
        for doc in result["uploaded_documents"]:
            assert "id" in doc
            assert doc["status"] == DocumentStatus.UPLOADED.value


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.slow
class TestDocumentsAPIAdvanced:
    """文档API高级功能测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    def test_document_processing_workflow(self, client, sample_text_file):
        """测试完整的文档处理工作流"""
        filename, content, content_type = sample_text_file
        
        # 1. 上传文档
        files = {"file": (filename, BytesIO(content), content_type)}
        data = {"title": "工作流测试文档"}
        
        upload_response = client.post("/api/v1/documents/upload", files=files, data=data)
        assert upload_response.status_code == 201
        document_id = upload_response.json()["id"]
        
        # 2. 检查初始状态
        status_response = client.get(f"/api/v1/documents/{document_id}/status")
        assert status_response.status_code == 200
        assert status_response.json()["status"] == DocumentStatus.UPLOADED.value
        
        # 3. 开始处理
        process_response = client.post(f"/api/v1/documents/{document_id}/process")
        assert process_response.status_code == 200
        
        # 4. 等待处理完成（在实际测试中可能需要轮询）
        # 这里简化处理，假设处理很快完成
        
        # 5. 检查处理后状态
        final_status_response = client.get(f"/api/v1/documents/{document_id}/status")
        assert final_status_response.status_code == 200
        # 状态可能是 PROCESSING 或 COMPLETED，取决于处理速度
    
    def test_document_search_advanced_filters(self, client, sample_text_file):
        """测试高级搜索过滤器"""
        filename, content, content_type = sample_text_file
        
        # 上传多个不同属性的文档
        test_docs = [
            {
                "title": "技术文档A",
                "tags": ["技术", "AI"],
                "categories": ["研发"],
                "language": "zh"
            },
            {
                "title": "商业文档B", 
                "tags": ["商业", "策略"],
                "categories": ["商务"],
                "language": "en"
            },
            {
                "title": "技术文档C",
                "tags": ["技术", "数据库"],
                "categories": ["研发"],
                "language": "zh"
            }
        ]
        
        uploaded_ids = []
        for doc_data in test_docs:
            files = {"file": (f"{doc_data['title']}.txt", BytesIO(content), content_type)}
            data = {
                "title": doc_data["title"],
                "tags": json.dumps(doc_data["tags"]),
                "categories": json.dumps(doc_data["categories"]),
                "language": doc_data["language"]
            }
            
            response = client.post("/api/v1/documents/upload", files=files, data=data)
            uploaded_ids.append(response.json()["id"])
        
        # 测试按语言过滤
        response = client.get("/api/v1/documents/search?language=zh")
        assert response.status_code == 200
        zh_docs = response.json()["items"]
        assert len([doc for doc in zh_docs if doc["language"] == "zh"]) >= 2
        
        # 测试按分类过滤
        response = client.get("/api/v1/documents/search?categories=研发")
        assert response.status_code == 200
        rd_docs = response.json()["items"]
        assert len([doc for doc in rd_docs if "研发" in doc["categories"]]) >= 2
        
        # 测试组合过滤
        response = client.get("/api/v1/documents/search?language=zh&categories=研发&tags=技术")
        assert response.status_code == 200
        filtered_docs = response.json()["items"]
        assert len(filtered_docs) >= 2
    
    def test_document_statistics(self, client, sample_text_file):
        """测试文档统计信息"""
        filename, content, content_type = sample_text_file
        
        # 上传一些测试文档
        for i in range(3):
            files = {"file": (f"stats_doc_{i}.txt", BytesIO(content), content_type)}
            data = {"title": f"统计测试文档 {i}"}
            client.post("/api/v1/documents/upload", files=files, data=data)
        
        # 获取统计信息
        response = client.get("/api/v1/documents/stats")
        
        assert response.status_code == 200
        stats = response.json()
        
        assert "total_documents" in stats
        assert "documents_by_status" in stats
        assert "documents_by_type" in stats
        assert "documents_by_language" in stats
        assert stats["total_documents"] >= 3
    
    def test_document_export(self, client, sample_text_file):
        """测试文档导出功能"""
        filename, content, content_type = sample_text_file
        
        # 上传文档
        files = {"file": (filename, BytesIO(content), content_type)}
        data = {"title": "导出测试文档"}
        
        upload_response = client.post("/api/v1/documents/upload", files=files, data=data)
        document_id = upload_response.json()["id"]
        
        # 导出文档
        response = client.get(f"/api/v1/documents/{document_id}/export?format=json")
        
        assert response.status_code == 200
        export_data = response.json()
        
        assert "document" in export_data
        assert "chunks" in export_data
        assert "entities" in export_data
        assert "relations" in export_data