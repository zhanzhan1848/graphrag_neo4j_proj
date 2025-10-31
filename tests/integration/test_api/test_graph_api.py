#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 图谱查询API集成测试
============================

测试图谱查询API的完整功能：
1. 实体查询和搜索
2. 关系查询和遍历
3. 图谱可视化数据
4. 路径查找
5. 子图提取
6. 图谱统计分析

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
class TestGraphAPI:
    """图谱查询API集成测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    @pytest.fixture
    def knowledge_graph_data(self, client):
        """创建包含实体和关系的知识图谱数据"""
        content = """
        苹果公司（Apple Inc.）是一家美国跨国科技公司，总部位于加利福尼亚州库比蒂诺。
        史蒂夫·乔布斯（Steve Jobs）是苹果公司的联合创始人之一。
        蒂姆·库克（Tim Cook）是苹果公司现任首席执行官。
        iPhone是苹果公司开发的智能手机产品线。
        iPad是苹果公司开发的平板电脑产品。
        macOS是苹果公司为Mac计算机开发的操作系统。
        
        微软公司（Microsoft Corporation）是一家美国跨国科技公司。
        比尔·盖茨（Bill Gates）是微软公司的联合创始人。
        萨蒂亚·纳德拉（Satya Nadella）是微软公司现任首席执行官。
        Windows是微软公司开发的操作系统。
        Office是微软公司开发的办公软件套件。
        """
        
        files = {"file": ("tech_companies.txt", BytesIO(content.encode()), "text/plain")}
        data = {
            "title": "科技公司知识图谱",
            "description": "包含苹果、微软等公司的实体关系",
            "tags": json.dumps(["科技", "公司", "人物", "产品"]),
            "language": "zh"
        }
        
        response = client.post("/api/v1/documents/upload", files=files, data=data)
        assert response.status_code == 201
        document_id = response.json()["id"]
        
        # 处理文档以提取实体和关系
        process_response = client.post(f"/api/v1/documents/{document_id}/process")
        assert process_response.status_code == 200
        
        return document_id
    
    def test_get_entities_list(self, client, knowledge_graph_data):
        """测试获取实体列表"""
        response = client.get("/api/v1/graph/entities")
        
        assert response.status_code == 200
        result = response.json()
        
        assert "entities" in result
        assert "total" in result
        assert "page" in result
        assert "size" in result
        
        entities = result["entities"]
        assert isinstance(entities, list)
        
        # 验证实体结构
        if entities:
            entity = entities[0]
            assert "id" in entity
            assert "name" in entity
            assert "type" in entity
            assert "properties" in entity
    
    def test_get_entities_with_pagination(self, client, knowledge_graph_data):
        """测试分页获取实体"""
        response = client.get("/api/v1/graph/entities?page=1&size=5")
        
        assert response.status_code == 200
        result = response.json()
        
        assert result["page"] == 1
        assert result["size"] == 5
        assert len(result["entities"]) <= 5
    
    def test_get_entities_by_type(self, client, knowledge_graph_data):
        """测试按类型获取实体"""
        # 先获取所有实体类型
        all_entities_response = client.get("/api/v1/graph/entities")
        all_entities = all_entities_response.json()["entities"]
        
        if all_entities:
            # 选择一个实体类型进行测试
            entity_type = all_entities[0]["type"]
            
            response = client.get(f"/api/v1/graph/entities?type={entity_type}")
            
            assert response.status_code == 200
            result = response.json()
            
            # 验证所有返回的实体都是指定类型
            for entity in result["entities"]:
                assert entity["type"] == entity_type
    
    def test_search_entities(self, client, knowledge_graph_data):
        """测试搜索实体"""
        search_data = {
            "query": "苹果",
            "limit": 10,
            "include_properties": True
        }
        
        response = client.post("/api/v1/graph/entities/search", json=search_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert "entities" in result
        assert "total" in result
        
        entities = result["entities"]
        assert isinstance(entities, list)
        
        # 验证搜索结果相关性
        if entities:
            found_apple = any("苹果" in entity["name"] for entity in entities)
            assert found_apple
    
    def test_get_entity_by_id(self, client, knowledge_graph_data):
        """测试通过ID获取实体详情"""
        # 先获取一个实体ID
        entities_response = client.get("/api/v1/graph/entities?size=1")
        entities = entities_response.json()["entities"]
        
        if entities:
            entity_id = entities[0]["id"]
            
            response = client.get(f"/api/v1/graph/entities/{entity_id}")
            
            assert response.status_code == 200
            result = response.json()
            
            assert result["id"] == entity_id
            assert "name" in result
            assert "type" in result
            assert "properties" in result
            assert "relations" in result
    
    def test_get_entity_not_found(self, client):
        """测试获取不存在的实体"""
        non_existent_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/graph/entities/{non_existent_id}")
        
        assert response.status_code == 404
        assert "实体不存在" in response.json()["detail"]
    
    def test_get_entity_relations(self, client, knowledge_graph_data):
        """测试获取实体的关系"""
        # 先获取一个实体
        entities_response = client.get("/api/v1/graph/entities?size=1")
        entities = entities_response.json()["entities"]
        
        if entities:
            entity_id = entities[0]["id"]
            
            response = client.get(f"/api/v1/graph/entities/{entity_id}/relations")
            
            assert response.status_code == 200
            result = response.json()
            
            assert "relations" in result
            assert "total" in result
            
            relations = result["relations"]
            assert isinstance(relations, list)
            
            # 验证关系结构
            if relations:
                relation = relations[0]
                assert "id" in relation
                assert "type" in relation
                assert "source_entity" in relation
                assert "target_entity" in relation
    
    def test_get_relations_list(self, client, knowledge_graph_data):
        """测试获取关系列表"""
        response = client.get("/api/v1/graph/relations")
        
        assert response.status_code == 200
        result = response.json()
        
        assert "relations" in result
        assert "total" in result
        assert "page" in result
        assert "size" in result
        
        relations = result["relations"]
        assert isinstance(relations, list)
    
    def test_get_relations_by_type(self, client, knowledge_graph_data):
        """测试按类型获取关系"""
        # 先获取所有关系类型
        all_relations_response = client.get("/api/v1/graph/relations")
        all_relations = all_relations_response.json()["relations"]
        
        if all_relations:
            relation_type = all_relations[0]["type"]
            
            response = client.get(f"/api/v1/graph/relations?type={relation_type}")
            
            assert response.status_code == 200
            result = response.json()
            
            # 验证所有返回的关系都是指定类型
            for relation in result["relations"]:
                assert relation["type"] == relation_type
    
    def test_search_relations(self, client, knowledge_graph_data):
        """测试搜索关系"""
        search_data = {
            "query": "创始人",
            "limit": 10,
            "include_entities": True
        }
        
        response = client.post("/api/v1/graph/relations/search", json=search_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert "relations" in result
        assert "total" in result
        
        relations = result["relations"]
        assert isinstance(relations, list)
    
    def test_find_path_between_entities(self, client, knowledge_graph_data):
        """测试查找实体间路径"""
        # 获取两个实体
        entities_response = client.get("/api/v1/graph/entities?size=2")
        entities = entities_response.json()["entities"]
        
        if len(entities) >= 2:
            source_id = entities[0]["id"]
            target_id = entities[1]["id"]
            
            path_data = {
                "source_entity_id": source_id,
                "target_entity_id": target_id,
                "max_depth": 3,
                "algorithm": "shortest_path"
            }
            
            response = client.post("/api/v1/graph/path", json=path_data)
            
            assert response.status_code == 200
            result = response.json()
            
            assert "paths" in result
            assert "total_paths" in result
            
            paths = result["paths"]
            assert isinstance(paths, list)
            
            # 验证路径结构
            if paths:
                path = paths[0]
                assert "nodes" in path
                assert "edges" in path
                assert "length" in path
    
    def test_find_path_no_connection(self, client, knowledge_graph_data):
        """测试查找无连接实体间的路径"""
        # 创建两个不相关的实体ID（假设它们不相连）
        entities_response = client.get("/api/v1/graph/entities")
        entities = entities_response.json()["entities"]
        
        if len(entities) >= 2:
            # 选择可能不相连的实体
            source_id = entities[0]["id"]
            target_id = entities[-1]["id"]  # 选择列表末尾的实体
            
            path_data = {
                "source_entity_id": source_id,
                "target_entity_id": target_id,
                "max_depth": 2
            }
            
            response = client.post("/api/v1/graph/path", json=path_data)
            
            assert response.status_code == 200
            result = response.json()
            
            # 可能没有路径，但应该正常返回
            assert "paths" in result
            assert "total_paths" in result
    
    def test_get_subgraph(self, client, knowledge_graph_data):
        """测试获取子图"""
        # 获取一个实体作为中心节点
        entities_response = client.get("/api/v1/graph/entities?size=1")
        entities = entities_response.json()["entities"]
        
        if entities:
            center_entity_id = entities[0]["id"]
            
            subgraph_data = {
                "center_entity_id": center_entity_id,
                "depth": 2,
                "max_nodes": 20,
                "include_properties": True
            }
            
            response = client.post("/api/v1/graph/subgraph", json=subgraph_data)
            
            assert response.status_code == 200
            result = response.json()
            
            assert "nodes" in result
            assert "edges" in result
            assert "center_node" in result
            
            nodes = result["nodes"]
            edges = result["edges"]
            
            assert isinstance(nodes, list)
            assert isinstance(edges, list)
            
            # 验证中心节点在结果中
            center_found = any(node["id"] == center_entity_id for node in nodes)
            assert center_found
    
    def test_get_graph_statistics(self, client, knowledge_graph_data):
        """测试获取图谱统计信息"""
        response = client.get("/api/v1/graph/stats")
        
        assert response.status_code == 200
        result = response.json()
        
        assert "total_entities" in result
        assert "total_relations" in result
        assert "entity_types" in result
        assert "relation_types" in result
        assert "graph_density" in result
        
        # 验证统计数据类型
        assert isinstance(result["total_entities"], int)
        assert isinstance(result["total_relations"], int)
        assert isinstance(result["entity_types"], dict)
        assert isinstance(result["relation_types"], dict)
        assert isinstance(result["graph_density"], (int, float))
    
    def test_get_graph_visualization_data(self, client, knowledge_graph_data):
        """测试获取图谱可视化数据"""
        viz_data = {
            "layout": "force_directed",
            "max_nodes": 50,
            "include_isolated_nodes": False,
            "node_size_by": "degree",
            "edge_width_by": "weight"
        }
        
        response = client.post("/api/v1/graph/visualization", json=viz_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert "nodes" in result
        assert "edges" in result
        assert "layout_info" in result
        
        nodes = result["nodes"]
        edges = result["edges"]
        
        assert isinstance(nodes, list)
        assert isinstance(edges, list)
        
        # 验证可视化节点结构
        if nodes:
            node = nodes[0]
            assert "id" in node
            assert "label" in node
            assert "x" in node or "position" in node
            assert "y" in node or "position" in node
            assert "size" in node
            assert "color" in node
        
        # 验证可视化边结构
        if edges:
            edge = edges[0]
            assert "source" in edge
            assert "target" in edge
            assert "weight" in edge or "width" in edge
    
    def test_graph_query_cypher(self, client, knowledge_graph_data):
        """测试Cypher查询"""
        cypher_data = {
            "query": "MATCH (n) RETURN n LIMIT 10",
            "parameters": {}
        }
        
        response = client.post("/api/v1/graph/cypher", json=cypher_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert "results" in result
        assert "columns" in result
        assert "execution_time" in result
        
        results = result["results"]
        assert isinstance(results, list)
    
    def test_graph_query_cypher_with_parameters(self, client, knowledge_graph_data):
        """测试带参数的Cypher查询"""
        cypher_data = {
            "query": "MATCH (n) WHERE n.name CONTAINS $name_part RETURN n LIMIT $limit",
            "parameters": {
                "name_part": "苹果",
                "limit": 5
            }
        }
        
        response = client.post("/api/v1/graph/cypher", json=cypher_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert "results" in result
        assert len(result["results"]) <= 5
    
    def test_graph_query_cypher_invalid(self, client):
        """测试无效的Cypher查询"""
        cypher_data = {
            "query": "INVALID CYPHER QUERY",
            "parameters": {}
        }
        
        response = client.post("/api/v1/graph/cypher", json=cypher_data)
        
        assert response.status_code == 400
        assert "查询语法错误" in response.json()["detail"]
    
    def test_get_entity_neighbors(self, client, knowledge_graph_data):
        """测试获取实体邻居"""
        # 获取一个实体
        entities_response = client.get("/api/v1/graph/entities?size=1")
        entities = entities_response.json()["entities"]
        
        if entities:
            entity_id = entities[0]["id"]
            
            response = client.get(f"/api/v1/graph/entities/{entity_id}/neighbors?depth=1&limit=10")
            
            assert response.status_code == 200
            result = response.json()
            
            assert "neighbors" in result
            assert "total" in result
            
            neighbors = result["neighbors"]
            assert isinstance(neighbors, list)
            
            # 验证邻居结构
            if neighbors:
                neighbor = neighbors[0]
                assert "entity" in neighbor
                assert "relation" in neighbor
                assert "distance" in neighbor


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.slow
class TestGraphAPIAdvanced:
    """图谱API高级功能测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    @pytest.fixture
    def complex_graph_data(self, client):
        """创建复杂的图谱数据"""
        content = """
        在人工智能领域，深度学习是机器学习的一个重要分支。
        Geoffrey Hinton被称为深度学习之父，他在多伦多大学工作。
        Yann LeCun是卷积神经网络的先驱，现在在Meta公司工作。
        Yoshua Bengio是蒙特利尔大学的教授，专注于深度学习研究。
        
        TensorFlow是Google开发的深度学习框架。
        PyTorch是Facebook（现Meta）开发的深度学习框架。
        Keras是一个高级神经网络API，现在集成在TensorFlow中。
        
        ImageNet是一个大型视觉数据库，用于视觉对象识别研究。
        BERT是Google开发的预训练语言模型。
        GPT是OpenAI开发的生成式预训练变换器。
        """
        
        files = {"file": ("ai_knowledge.txt", BytesIO(content.encode()), "text/plain")}
        data = {
            "title": "AI领域知识图谱",
            "description": "包含AI领域的人物、机构、技术、产品等实体关系",
            "tags": json.dumps(["AI", "深度学习", "人物", "技术"]),
            "language": "zh"
        }
        
        response = client.post("/api/v1/documents/upload", files=files, data=data)
        assert response.status_code == 201
        document_id = response.json()["id"]
        
        # 处理文档
        process_response = client.post(f"/api/v1/documents/{document_id}/process")
        assert process_response.status_code == 200
        
        return document_id
    
    def test_graph_clustering(self, client, complex_graph_data):
        """测试图谱聚类分析"""
        clustering_data = {
            "algorithm": "louvain",
            "resolution": 1.0,
            "min_cluster_size": 2
        }
        
        response = client.post("/api/v1/graph/clustering", json=clustering_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert "clusters" in result
        assert "modularity" in result
        assert "total_clusters" in result
        
        clusters = result["clusters"]
        assert isinstance(clusters, list)
        
        # 验证聚类结构
        if clusters:
            cluster = clusters[0]
            assert "cluster_id" in cluster
            assert "entities" in cluster
            assert "size" in cluster
    
    def test_graph_centrality_analysis(self, client, complex_graph_data):
        """测试图谱中心性分析"""
        centrality_data = {
            "metrics": ["degree", "betweenness", "closeness", "pagerank"],
            "top_k": 10
        }
        
        response = client.post("/api/v1/graph/centrality", json=centrality_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert "centrality_scores" in result
        
        centrality_scores = result["centrality_scores"]
        for metric in centrality_data["metrics"]:
            assert metric in centrality_scores
            assert isinstance(centrality_scores[metric], list)
            assert len(centrality_scores[metric]) <= centrality_data["top_k"]
    
    def test_graph_similarity_search(self, client, complex_graph_data):
        """测试图谱相似性搜索"""
        # 先获取一个实体
        entities_response = client.get("/api/v1/graph/entities?size=1")
        entities = entities_response.json()["entities"]
        
        if entities:
            reference_entity_id = entities[0]["id"]
            
            similarity_data = {
                "reference_entity_id": reference_entity_id,
                "similarity_metric": "structural",
                "top_k": 5,
                "min_similarity": 0.1
            }
            
            response = client.post("/api/v1/graph/similarity", json=similarity_data)
            
            assert response.status_code == 200
            result = response.json()
            
            assert "similar_entities" in result
            assert "reference_entity" in result
            
            similar_entities = result["similar_entities"]
            assert isinstance(similar_entities, list)
            assert len(similar_entities) <= similarity_data["top_k"]
            
            # 验证相似性分数
            for entity in similar_entities:
                assert "entity" in entity
                assert "similarity_score" in entity
                assert 0 <= entity["similarity_score"] <= 1
    
    def test_graph_temporal_analysis(self, client, complex_graph_data):
        """测试图谱时间分析"""
        temporal_data = {
            "time_range": {
                "start": "2020-01-01",
                "end": "2024-12-31"
            },
            "granularity": "year",
            "metrics": ["entity_growth", "relation_growth", "activity"]
        }
        
        response = client.post("/api/v1/graph/temporal", json=temporal_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert "temporal_metrics" in result
        assert "time_series" in result
        
        temporal_metrics = result["temporal_metrics"]
        for metric in temporal_data["metrics"]:
            assert metric in temporal_metrics
    
    def test_graph_export(self, client, complex_graph_data):
        """测试图谱导出"""
        export_data = {
            "format": "graphml",
            "include_properties": True,
            "filter": {
                "entity_types": ["PERSON", "ORGANIZATION"],
                "relation_types": ["WORKS_AT", "DEVELOPED"]
            }
        }
        
        response = client.post("/api/v1/graph/export", json=export_data)
        
        assert response.status_code == 200
        
        # 验证响应头
        content_type = response.headers.get("content-type")
        assert "application/xml" in content_type or "text/xml" in content_type
        
        # 验证内容不为空
        content = response.content
        assert len(content) > 0
    
    def test_graph_import_validation(self, client):
        """测试图谱导入验证"""
        # 模拟图谱数据
        import_data = {
            "format": "json",
            "data": {
                "nodes": [
                    {"id": "1", "label": "测试实体1", "type": "PERSON"},
                    {"id": "2", "label": "测试实体2", "type": "ORGANIZATION"}
                ],
                "edges": [
                    {"source": "1", "target": "2", "type": "WORKS_AT", "weight": 1.0}
                ]
            },
            "validate_only": True
        }
        
        response = client.post("/api/v1/graph/import", json=import_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert "validation_result" in result
        assert "is_valid" in result["validation_result"]
        assert "errors" in result["validation_result"]
        assert "warnings" in result["validation_result"]
    
    def test_graph_schema_analysis(self, client, complex_graph_data):
        """测试图谱模式分析"""
        response = client.get("/api/v1/graph/schema")
        
        assert response.status_code == 200
        result = response.json()
        
        assert "entity_types" in result
        assert "relation_types" in result
        assert "schema_patterns" in result
        
        entity_types = result["entity_types"]
        relation_types = result["relation_types"]
        
        # 验证模式信息结构
        for entity_type in entity_types:
            assert "type" in entity_type
            assert "count" in entity_type
            assert "properties" in entity_type
        
        for relation_type in relation_types:
            assert "type" in relation_type
            assert "count" in relation_type
            assert "source_types" in relation_type
            assert "target_types" in relation_type