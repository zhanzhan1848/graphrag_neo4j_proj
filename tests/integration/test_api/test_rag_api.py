#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG RAG问答API集成测试
==========================

测试RAG（检索增强生成）问答API的完整功能：
1. 基础问答功能
2. 上下文检索
3. 多轮对话
4. 引用和溯源
5. 不同查询模式
6. 错误处理和边界情况

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import pytest
import json
import uuid
from fastapi.testclient import TestClient
from io import BytesIO

from app.main import app


@pytest.mark.integration
@pytest.mark.api
class TestRAGAPI:
    """RAG问答API集成测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_document_id(self, client):
        """创建示例文档并返回ID"""
        content = """
        人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，
        它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
        
        机器学习是人工智能的一个重要分支，它使计算机能够在没有明确编程的情况下学习。
        深度学习是机器学习的一个子集，它基于人工神经网络，特别是深层神经网络。
        
        自然语言处理（NLP）是人工智能和语言学领域的分支学科，
        它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。
        """
        
        files = {"file": ("ai_knowledge.txt", BytesIO(content.encode()), "text/plain")}
        data = {
            "title": "人工智能知识文档",
            "description": "包含AI、ML、DL、NLP相关知识",
            "tags": json.dumps(["AI", "机器学习", "深度学习", "NLP"]),
            "language": "zh"
        }
        
        response = client.post("/api/v1/documents/upload", files=files, data=data)
        assert response.status_code == 201
        
        document_id = response.json()["id"]
        
        # 触发文档处理
        process_response = client.post(f"/api/v1/documents/{document_id}/process")
        assert process_response.status_code == 200
        
        return document_id
    
    def test_basic_rag_query(self, client, sample_document_id):
        """测试基础RAG查询"""
        query_data = {
            "query": "什么是人工智能？",
            "max_results": 5,
            "include_sources": True
        }
        
        response = client.post("/api/v1/rag/query", json=query_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert "answer" in result
        assert "sources" in result
        assert "query_id" in result
        assert "confidence" in result
        
        # 验证答案包含相关内容
        assert len(result["answer"]) > 0
        assert isinstance(result["sources"], list)
        assert result["confidence"] >= 0.0
    
    def test_rag_query_with_context(self, client, sample_document_id):
        """测试带上下文的RAG查询"""
        query_data = {
            "query": "机器学习和深度学习有什么关系？",
            "context": "我想了解AI相关技术的层次关系",
            "max_results": 3,
            "include_sources": True,
            "temperature": 0.7
        }
        
        response = client.post("/api/v1/rag/query", json=query_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert "answer" in result
        assert "sources" in result
        assert len(result["sources"]) <= 3
        
        # 验证答案质量
        answer = result["answer"].lower()
        assert "机器学习" in answer or "深度学习" in answer
    
    def test_rag_query_multiple_documents(self, client, sample_document_id):
        """测试多文档RAG查询"""
        # 上传另一个相关文档
        content2 = """
        计算机视觉是人工智能的一个重要应用领域，
        它使计算机能够从数字图像或视频中获取高层次的理解。
        
        推荐系统是信息过滤系统的一个子集，
        它试图预测用户对物品的"评分"或"偏好"。
        """
        
        files = {"file": ("cv_rec.txt", BytesIO(content2.encode()), "text/plain")}
        data = {"title": "计算机视觉和推荐系统", "language": "zh"}
        
        upload_response = client.post("/api/v1/documents/upload", files=files, data=data)
        document_id2 = upload_response.json()["id"]
        
        # 处理第二个文档
        client.post(f"/api/v1/documents/{document_id2}/process")
        
        # 查询涉及多个文档的问题
        query_data = {
            "query": "人工智能有哪些应用领域？",
            "max_results": 10,
            "include_sources": True
        }
        
        response = client.post("/api/v1/rag/query", json=query_data)
        
        assert response.status_code == 200
        result = response.json()
        
        # 验证结果来自多个文档
        sources = result["sources"]
        document_ids = set(source.get("document_id") for source in sources)
        assert len(document_ids) >= 1  # 至少来自一个文档
    
    def test_rag_query_with_filters(self, client, sample_document_id):
        """测试带过滤器的RAG查询"""
        query_data = {
            "query": "深度学习的定义",
            "filters": {
                "document_ids": [sample_document_id],
                "tags": ["深度学习"],
                "language": "zh"
            },
            "max_results": 5,
            "include_sources": True
        }
        
        response = client.post("/api/v1/rag/query", json=query_data)
        
        assert response.status_code == 200
        result = response.json()
        
        # 验证所有来源都符合过滤条件
        for source in result["sources"]:
            if "document_id" in source:
                assert source["document_id"] == sample_document_id
    
    def test_rag_query_different_modes(self, client, sample_document_id):
        """测试不同的查询模式"""
        base_query = "什么是自然语言处理？"
        
        modes = ["semantic", "keyword", "hybrid"]
        
        for mode in modes:
            query_data = {
                "query": base_query,
                "search_mode": mode,
                "max_results": 3,
                "include_sources": True
            }
            
            response = client.post("/api/v1/rag/query", json=query_data)
            
            assert response.status_code == 200
            result = response.json()
            
            assert "answer" in result
            assert "sources" in result
            assert len(result["sources"]) <= 3
    
    def test_rag_query_empty_query(self, client):
        """测试空查询"""
        query_data = {
            "query": "",
            "max_results": 5
        }
        
        response = client.post("/api/v1/rag/query", json=query_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_rag_query_very_long_query(self, client, sample_document_id):
        """测试超长查询"""
        long_query = "什么是人工智能？" * 100  # 创建一个很长的查询
        
        query_data = {
            "query": long_query,
            "max_results": 3
        }
        
        response = client.post("/api/v1/rag/query", json=query_data)
        
        # 可能返回400（查询太长）或200（截断处理）
        assert response.status_code in [200, 400]
    
    def test_rag_query_no_relevant_documents(self, client):
        """测试查询无相关文档的问题"""
        query_data = {
            "query": "量子计算的基本原理是什么？",  # 假设没有相关文档
            "max_results": 5,
            "include_sources": True
        }
        
        response = client.post("/api/v1/rag/query", json=query_data)
        
        assert response.status_code == 200
        result = response.json()
        
        # 应该返回结果，但可能置信度较低或明确说明没有相关信息
        assert "answer" in result
        assert "sources" in result
    
    def test_rag_conversation_history(self, client, sample_document_id):
        """测试多轮对话历史"""
        # 第一轮对话
        query1_data = {
            "query": "什么是机器学习？",
            "max_results": 3,
            "include_sources": True
        }
        
        response1 = client.post("/api/v1/rag/query", json=query1_data)
        assert response1.status_code == 200
        query_id1 = response1.json()["query_id"]
        
        # 第二轮对话，引用前一轮
        query2_data = {
            "query": "它和深度学习有什么区别？",
            "conversation_history": [
                {
                    "query": "什么是机器学习？",
                    "answer": response1.json()["answer"],
                    "query_id": query_id1
                }
            ],
            "max_results": 3,
            "include_sources": True
        }
        
        response2 = client.post("/api/v1/rag/query", json=query2_data)
        
        assert response2.status_code == 200
        result2 = response2.json()
        
        # 验证第二轮回答考虑了上下文
        assert "answer" in result2
        assert "深度学习" in result2["answer"] or "机器学习" in result2["answer"]
    
    def test_rag_query_with_citations(self, client, sample_document_id):
        """测试带引用的RAG查询"""
        query_data = {
            "query": "人工智能的定义是什么？",
            "include_sources": True,
            "include_citations": True,
            "citation_style": "numbered",
            "max_results": 3
        }
        
        response = client.post("/api/v1/rag/query", json=query_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert "answer" in result
        assert "sources" in result
        assert "citations" in result
        
        # 验证引用格式
        citations = result["citations"]
        assert isinstance(citations, list)
        
        # 答案中应该包含引用标记
        answer = result["answer"]
        if citations:
            # 检查是否有引用标记（如 [1], [2] 等）
            import re
            citation_pattern = r'\[\d+\]'
            assert re.search(citation_pattern, answer) is not None
    
    def test_rag_query_streaming(self, client, sample_document_id):
        """测试流式RAG查询"""
        query_data = {
            "query": "解释一下深度学习的概念",
            "stream": True,
            "max_results": 3
        }
        
        response = client.post("/api/v1/rag/query/stream", json=query_data)
        
        assert response.status_code == 200
        
        # 验证流式响应
        content_type = response.headers.get("content-type", "")
        assert "text/event-stream" in content_type or "application/json" in content_type
    
    def test_rag_query_with_metadata(self, client, sample_document_id):
        """测试返回详细元数据的RAG查询"""
        query_data = {
            "query": "什么是NLP？",
            "include_sources": True,
            "include_metadata": True,
            "max_results": 3
        }
        
        response = client.post("/api/v1/rag/query", json=query_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert "answer" in result
        assert "sources" in result
        assert "metadata" in result
        
        metadata = result["metadata"]
        assert "query_time" in metadata
        assert "retrieval_time" in metadata
        assert "generation_time" in metadata
        assert "total_chunks_searched" in metadata
    
    def test_rag_feedback(self, client, sample_document_id):
        """测试RAG查询反馈"""
        # 先进行查询
        query_data = {
            "query": "人工智能的应用有哪些？",
            "max_results": 3
        }
        
        response = client.post("/api/v1/rag/query", json=query_data)
        assert response.status_code == 200
        query_id = response.json()["query_id"]
        
        # 提交反馈
        feedback_data = {
            "query_id": query_id,
            "rating": 4,
            "feedback": "答案很有帮助，但可以更详细一些",
            "helpful_sources": [0, 1],  # 源索引
            "issues": ["incomplete"]
        }
        
        feedback_response = client.post("/api/v1/rag/feedback", json=feedback_data)
        
        assert feedback_response.status_code == 200
        result = feedback_response.json()
        assert result["message"] == "反馈提交成功"
    
    def test_rag_query_history(self, client, sample_document_id):
        """测试查询历史记录"""
        # 进行几次查询
        queries = [
            "什么是人工智能？",
            "机器学习的定义",
            "深度学习和神经网络"
        ]
        
        query_ids = []
        for query in queries:
            query_data = {"query": query, "max_results": 2}
            response = client.post("/api/v1/rag/query", json=query_data)
            assert response.status_code == 200
            query_ids.append(response.json()["query_id"])
        
        # 获取查询历史
        history_response = client.get("/api/v1/rag/history?limit=10")
        
        assert history_response.status_code == 200
        history = history_response.json()
        
        assert "queries" in history
        assert "total" in history
        assert len(history["queries"]) >= 3
        
        # 验证历史记录包含我们的查询
        historical_queries = [q["query"] for q in history["queries"]]
        for original_query in queries:
            assert any(original_query in hq for hq in historical_queries)


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.slow
class TestRAGAPIAdvanced:
    """RAG API高级功能测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    @pytest.fixture
    def complex_knowledge_base(self, client):
        """创建复杂的知识库"""
        documents = [
            {
                "filename": "ai_basics.txt",
                "content": """
                人工智能基础知识：
                1. 机器学习：让机器从数据中学习
                2. 深度学习：基于神经网络的学习方法
                3. 强化学习：通过奖励机制学习最优策略
                """,
                "title": "AI基础",
                "tags": ["AI", "基础"]
            },
            {
                "filename": "ml_algorithms.txt", 
                "content": """
                机器学习算法分类：
                1. 监督学习：线性回归、决策树、随机森林、SVM
                2. 无监督学习：K-means聚类、PCA降维
                3. 半监督学习：结合标记和未标记数据
                """,
                "title": "机器学习算法",
                "tags": ["机器学习", "算法"]
            },
            {
                "filename": "dl_architectures.txt",
                "content": """
                深度学习架构：
                1. CNN：卷积神经网络，适用于图像处理
                2. RNN：循环神经网络，适用于序列数据
                3. Transformer：注意力机制，适用于NLP任务
                """,
                "title": "深度学习架构",
                "tags": ["深度学习", "架构"]
            }
        ]
        
        document_ids = []
        for doc in documents:
            files = {"file": (doc["filename"], BytesIO(doc["content"].encode()), "text/plain")}
            data = {
                "title": doc["title"],
                "tags": json.dumps(doc["tags"]),
                "language": "zh"
            }
            
            response = client.post("/api/v1/documents/upload", files=files, data=data)
            assert response.status_code == 201
            doc_id = response.json()["id"]
            document_ids.append(doc_id)
            
            # 处理文档
            client.post(f"/api/v1/documents/{doc_id}/process")
        
        return document_ids
    
    def test_complex_multi_hop_reasoning(self, client, complex_knowledge_base):
        """测试复杂的多跳推理查询"""
        query_data = {
            "query": "比较监督学习和深度学习中的CNN，它们分别适用于什么场景？",
            "reasoning_mode": "multi_hop",
            "max_results": 10,
            "include_sources": True,
            "include_reasoning_steps": True
        }
        
        response = client.post("/api/v1/rag/query", json=query_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert "answer" in result
        assert "sources" in result
        assert "reasoning_steps" in result
        
        # 验证推理步骤
        reasoning_steps = result["reasoning_steps"]
        assert len(reasoning_steps) >= 2  # 至少两个推理步骤
        
        # 验证答案涉及多个概念
        answer = result["answer"].lower()
        assert "监督学习" in answer
        assert "cnn" in answer or "卷积" in answer
    
    def test_rag_query_with_graph_context(self, client, complex_knowledge_base):
        """测试结合图谱上下文的RAG查询"""
        query_data = {
            "query": "深度学习和机器学习的关系是什么？",
            "use_graph_context": True,
            "graph_depth": 2,
            "max_results": 5,
            "include_sources": True,
            "include_graph_paths": True
        }
        
        response = client.post("/api/v1/rag/query", json=query_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert "answer" in result
        assert "sources" in result
        assert "graph_paths" in result
        
        # 验证图谱路径信息
        graph_paths = result["graph_paths"]
        assert isinstance(graph_paths, list)
    
    def test_rag_query_performance_analysis(self, client, complex_knowledge_base):
        """测试RAG查询性能分析"""
        query_data = {
            "query": "什么是Transformer架构？",
            "max_results": 5,
            "include_performance_metrics": True,
            "include_sources": True
        }
        
        response = client.post("/api/v1/rag/query", json=query_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert "performance_metrics" in result
        metrics = result["performance_metrics"]
        
        assert "query_processing_time" in metrics
        assert "retrieval_time" in metrics
        assert "generation_time" in metrics
        assert "total_time" in metrics
        assert "chunks_processed" in metrics
        assert "tokens_generated" in metrics
    
    def test_rag_batch_queries(self, client, complex_knowledge_base):
        """测试批量RAG查询"""
        queries = [
            "什么是机器学习？",
            "深度学习有哪些架构？",
            "监督学习包括哪些算法？",
            "CNN适用于什么任务？"
        ]
        
        batch_data = {
            "queries": [{"query": q, "max_results": 3} for q in queries],
            "include_sources": True
        }
        
        response = client.post("/api/v1/rag/batch-query", json=batch_data)
        
        assert response.status_code == 200
        results = response.json()
        
        assert "results" in results
        assert len(results["results"]) == len(queries)
        
        for i, result in enumerate(results["results"]):
            assert "answer" in result
            assert "sources" in result
            assert "query" in result
            assert result["query"] == queries[i]
    
    def test_rag_query_with_custom_prompt(self, client, complex_knowledge_base):
        """测试自定义提示词的RAG查询"""
        query_data = {
            "query": "解释机器学习的概念",
            "custom_prompt_template": "作为一名AI专家，请用简洁明了的方式回答：{query}\n\n基于以下信息：{context}",
            "max_results": 3,
            "include_sources": True
        }
        
        response = client.post("/api/v1/rag/query", json=query_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert "answer" in result
        # 验证答案风格符合自定义提示词
        answer = result["answer"]
        assert len(answer) > 0