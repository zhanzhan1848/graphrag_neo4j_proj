# GraphRAG 知识库系统使用示例

## 概述

本文档提供了 GraphRAG 知识库系统的详细使用示例，包括常见使用场景、完整的工作流程和最佳实践。通过这些示例，您可以快速上手并充分利用系统的各项功能。

## 目录

1. [快速开始](#快速开始)
2. [文档管理示例](#文档管理示例)
3. [知识抽取示例](#知识抽取示例)
4. [图查询示例](#图查询示例)
5. [RAG 问答示例](#rag-问答示例)
6. [完整工作流程](#完整工作流程)
7. [高级用法](#高级用法)
8. [故障排除](#故障排除)

## 快速开始

### 1. 启动系统

```bash
# 启动所有服务
./scripts/start.sh

# 检查服务状态
curl http://localhost:8000/api/v1/system/health
```

### 2. 上传第一个文档

```bash
# 上传 PDF 文档
curl -X POST "http://localhost:8000/api/v1/document-management/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@research_paper.pdf" \
  -F "metadata={\"category\": \"research\", \"tags\": [\"AI\", \"ML\"]}"
```

### 3. 处理文档

```bash
# 处理上传的文档
curl -X POST "http://localhost:8000/api/v1/document-management/process/doc_123" \
  -H "Content-Type: application/json" \
  -d '{
    "processing_options": {
      "extract_text": true,
      "create_chunks": true,
      "generate_embeddings": true,
      "extract_entities": true
    }
  }'
```

### 4. 进行问答

```bash
# 基于知识库问答
curl -X POST "http://localhost:8000/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "这篇论文的主要贡献是什么？",
    "include_sources": true
  }'
```

## 文档管理示例

### 场景1：批量上传学术论文

```python
import requests
import os
from pathlib import Path

def upload_papers_batch(papers_dir: str, category: str = "research"):
    """批量上传学术论文"""
    
    base_url = "http://localhost:8000/api/v1"
    upload_url = f"{base_url}/document-management/upload"
    
    papers_path = Path(papers_dir)
    pdf_files = list(papers_path.glob("*.pdf"))
    
    print(f"找到 {len(pdf_files)} 个 PDF 文件")
    
    # 批量上传（每次最多5个文件）
    batch_size = 5
    for i in range(0, len(pdf_files), batch_size):
        batch_files = pdf_files[i:i+batch_size]
        
        files = []
        for pdf_file in batch_files:
            files.append(('files', (pdf_file.name, open(pdf_file, 'rb'), 'application/pdf')))
        
        metadata = {
            "category": category,
            "tags": ["academic", "research"],
            "batch_id": f"batch_{i//batch_size + 1}"
        }
        
        data = {
            "metadata": json.dumps(metadata)
        }
        
        try:
            response = requests.post(upload_url, files=files, data=data)
            response.raise_for_status()
            
            result = response.json()
            print(f"批次 {i//batch_size + 1} 上传成功:")
            for file_info in result["data"]["uploaded_files"]:
                print(f"  - {file_info['filename']}: {file_info['id']}")
                
        except requests.exceptions.RequestException as e:
            print(f"批次 {i//batch_size + 1} 上传失败: {e}")
        
        finally:
            # 关闭文件
            for _, (_, file_obj, _) in files:
                file_obj.close()

# 使用示例
upload_papers_batch("./papers", "ai_research")
```

### 场景2：监控文档处理状态

```python
import time
import requests

def monitor_document_processing(document_id: str):
    """监控文档处理状态"""
    
    base_url = "http://localhost:8000/api/v1"
    status_url = f"{base_url}/document-management/documents/{document_id}"
    
    print(f"开始监控文档 {document_id} 的处理状态...")
    
    while True:
        try:
            response = requests.get(status_url)
            response.raise_for_status()
            
            doc_info = response.json()["data"]
            status = doc_info["status"]
            
            print(f"当前状态: {status}")
            
            if status == "processed":
                print("✅ 文档处理完成!")
                print(f"  - 文本块数量: {len(doc_info.get('chunks', []))}")
                print(f"  - 实体数量: {len(doc_info.get('entities', []))}")
                break
            elif status == "failed":
                print("❌ 文档处理失败!")
                print(f"  - 错误信息: {doc_info.get('error_message', 'Unknown error')}")
                break
            elif status in ["uploaded", "processing"]:
                print("⏳ 处理中，等待5秒后重试...")
                time.sleep(5)
            else:
                print(f"⚠️  未知状态: {status}")
                break
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 请求失败: {e}")
            break
        except KeyboardInterrupt:
            print("\n⚠️  监控被用户中断")
            break

# 使用示例
monitor_document_processing("doc_123")
```

### 场景3：文档元数据管理

```python
def update_document_metadata(document_id: str, new_metadata: dict):
    """更新文档元数据"""
    
    base_url = "http://localhost:8000/api/v1"
    update_url = f"{base_url}/document-management/documents/{document_id}"
    
    payload = {
        "metadata": new_metadata,
        "update_mode": "merge"  # 合并模式，保留原有元数据
    }
    
    try:
        response = requests.patch(update_url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        print("✅ 元数据更新成功!")
        print(f"更新后的元数据: {result['data']['metadata']}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 元数据更新失败: {e}")

# 使用示例
new_metadata = {
    "authors": ["张三", "李四"],
    "publication_year": 2024,
    "journal": "AI Research Journal",
    "keywords": ["深度学习", "自然语言处理", "知识图谱"],
    "abstract": "本文提出了一种新的知识图谱构建方法..."
}

update_document_metadata("doc_123", new_metadata)
```

## 知识抽取示例

### 场景1：自定义实体抽取

```python
def extract_custom_entities(text: str, custom_types: list):
    """自定义实体抽取"""
    
    base_url = "http://localhost:8000/api/v1"
    extract_url = f"{base_url}/knowledge/extract-entities"
    
    payload = {
        "text": text,
        "entity_types": custom_types,
        "language": "zh",
        "confidence_threshold": 0.7,
        "custom_patterns": [
            {
                "type": "MODEL_NAME",
                "pattern": r"(GPT-\d+|BERT|Transformer|ResNet-\d+)",
                "description": "AI模型名称"
            },
            {
                "type": "METRIC",
                "pattern": r"(准确率|召回率|F1分数|BLEU分数)",
                "description": "评估指标"
            }
        ]
    }
    
    try:
        response = requests.post(extract_url, json=payload)
        response.raise_for_status()
        
        result = response.json()["data"]
        entities = result["entities"]
        
        print(f"✅ 抽取到 {len(entities)} 个实体:")
        
        # 按类型分组显示
        entities_by_type = {}
        for entity in entities:
            entity_type = entity["type"]
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        
        for entity_type, type_entities in entities_by_type.items():
            print(f"\n{entity_type}:")
            for entity in type_entities:
                print(f"  - {entity['name']} (置信度: {entity['confidence']:.2f})")
                if entity.get('properties'):
                    for key, value in entity['properties'].items():
                        print(f"    {key}: {value}")
        
        return entities
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 实体抽取失败: {e}")
        return []

# 使用示例
text = """
本研究使用GPT-4模型进行自然语言生成任务，在WMT数据集上取得了95%的准确率。
我们还比较了BERT和Transformer模型的性能，发现GPT-4在BLEU分数上表现最佳。
实验在NVIDIA A100 GPU上进行，使用了PyTorch框架。
"""

custom_types = ["MODEL_NAME", "METRIC", "DATASET", "HARDWARE", "FRAMEWORK"]
entities = extract_custom_entities(text, custom_types)
```

### 场景2：关系抽取和验证

```python
def extract_and_validate_relations(text: str, entities: list):
    """抽取关系并进行验证"""
    
    base_url = "http://localhost:8000/api/v1"
    extract_url = f"{base_url}/knowledge/extract-relations"
    
    payload = {
        "text": text,
        "entities": entities,
        "relation_types": [
            "USES", "ACHIEVES", "COMPARES_WITH", "RUNS_ON", 
            "IMPLEMENTED_IN", "EVALUATED_ON", "OUTPERFORMS"
        ],
        "confidence_threshold": 0.6,
        "validate_relations": True,  # 启用关系验证
        "include_evidence": True     # 包含证据文本
    }
    
    try:
        response = requests.post(extract_url, json=payload)
        response.raise_for_status()
        
        result = response.json()["data"]
        relations = result["relations"]
        
        print(f"✅ 抽取到 {len(relations)} 个关系:")
        
        # 按置信度排序
        relations.sort(key=lambda x: x["confidence"], reverse=True)
        
        for i, relation in enumerate(relations, 1):
            print(f"\n{i}. {relation['source_entity']} --[{relation['relation_type']}]--> {relation['target_entity']}")
            print(f"   置信度: {relation['confidence']:.2f}")
            print(f"   证据: \"{relation['evidence']}\"")
            
            if relation.get('properties'):
                print(f"   属性: {relation['properties']}")
        
        # 验证结果统计
        if "validation" in result:
            validation = result["validation"]
            print(f"\n📊 验证统计:")
            print(f"  - 高置信度关系: {validation['high_confidence_count']}")
            print(f"  - 中等置信度关系: {validation['medium_confidence_count']}")
            print(f"  - 低置信度关系: {validation['low_confidence_count']}")
            print(f"  - 可能的错误关系: {validation['potential_errors']}")
        
        return relations
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 关系抽取失败: {e}")
        return []

# 使用示例（接上面的实体抽取结果）
if entities:
    relations = extract_and_validate_relations(text, entities)
```

### 场景3：批量知识抽取和进度监控

```python
def batch_knowledge_extraction(document_ids: list):
    """批量知识抽取"""
    
    base_url = "http://localhost:8000/api/v1"
    batch_url = f"{base_url}/knowledge/extract-batch"
    
    payload = {
        "document_ids": document_ids,
        "extraction_options": {
            "extract_entities": True,
            "extract_relations": True,
            "extract_claims": True,
            "entity_types": ["PERSON", "ORGANIZATION", "CONCEPT", "METHOD", "DATASET"],
            "confidence_threshold": 0.8,
            "parallel_processing": True,
            "max_workers": 4
        }
    }
    
    try:
        # 提交批量任务
        response = requests.post(batch_url, json=payload)
        response.raise_for_status()
        
        result = response.json()["data"]
        batch_id = result["batch_id"]
        
        print(f"✅ 批量任务已提交: {batch_id}")
        print(f"预计处理时间: {result['estimated_time']} 秒")
        
        # 监控进度
        status_url = f"{base_url}/knowledge/batch-status/{batch_id}"
        
        while True:
            time.sleep(10)  # 每10秒检查一次
            
            status_response = requests.get(status_url)
            status_response.raise_for_status()
            
            status_data = status_response.json()["data"]
            
            print(f"进度: {status_data['completed_documents']}/{status_data['total_documents']} "
                  f"({status_data['progress_percentage']:.1f}%)")
            
            if status_data["status"] == "completed":
                print("✅ 批量抽取完成!")
                
                # 显示结果统计
                stats = status_data["statistics"]
                print(f"📊 抽取统计:")
                print(f"  - 总实体数: {stats['total_entities']}")
                print(f"  - 总关系数: {stats['total_relations']}")
                print(f"  - 总断言数: {stats['total_claims']}")
                print(f"  - 处理时间: {stats['processing_time']:.2f} 秒")
                
                break
            elif status_data["status"] == "failed":
                print("❌ 批量抽取失败!")
                print(f"错误信息: {status_data.get('error_message', 'Unknown error')}")
                break
        
        return batch_id
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 批量抽取失败: {e}")
        return None

# 使用示例
document_ids = ["doc_123", "doc_456", "doc_789"]
batch_id = batch_knowledge_extraction(document_ids)
```

## 图查询示例

### 场景1：实体关系探索

```python
def explore_entity_relationships(entity_name: str, max_depth: int = 2):
    """探索实体的关系网络"""
    
    base_url = "http://localhost:8000/api/v1"
    
    # 1. 首先搜索实体
    search_url = f"{base_url}/graph/nodes/search"
    search_params = {
        "query": entity_name,
        "node_type": "Entity",
        "limit": 1
    }
    
    try:
        response = requests.get(search_url, params=search_params)
        response.raise_for_status()
        
        nodes = response.json()["data"]["nodes"]
        if not nodes:
            print(f"❌ 未找到实体: {entity_name}")
            return
        
        entity_node = nodes[0]
        entity_id = entity_node["id"]
        
        print(f"✅ 找到实体: {entity_node['properties']['name']}")
        print(f"实体ID: {entity_id}")
        
        # 2. 获取实体的直接关系
        cypher_url = f"{base_url}/graph/cypher"
        cypher_query = {
            "query": """
            MATCH (e:Entity {id: $entity_id})-[r]-(connected)
            RETURN e, r, connected, type(r) as relation_type
            ORDER BY r.confidence DESC
            LIMIT 20
            """,
            "parameters": {"entity_id": entity_id}
        }
        
        response = requests.post(cypher_url, json=cypher_query)
        response.raise_for_status()
        
        records = response.json()["data"]["records"]
        
        print(f"\n🔗 直接关系 ({len(records)} 个):")
        
        relation_stats = {}
        for record in records:
            relation_type = record["relation_type"]
            connected_entity = record["connected"]["properties"]["name"]
            confidence = record["r"].get("confidence", 0)
            
            print(f"  - {entity_name} --[{relation_type}]--> {connected_entity} (置信度: {confidence:.2f})")
            
            # 统计关系类型
            if relation_type not in relation_stats:
                relation_stats[relation_type] = 0
            relation_stats[relation_type] += 1
        
        print(f"\n📊 关系类型统计:")
        for rel_type, count in sorted(relation_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {rel_type}: {count}")
        
        # 3. 探索更深层的关系（如果需要）
        if max_depth > 1:
            print(f"\n🌐 探索 {max_depth} 层关系网络...")
            
            deep_query = {
                "query": f"""
                MATCH path = (e:Entity {{id: $entity_id}})-[*1..{max_depth}]-(connected)
                WHERE connected.id <> $entity_id
                RETURN path, length(path) as depth
                ORDER BY depth, connected.name
                LIMIT 50
                """,
                "parameters": {"entity_id": entity_id}
            }
            
            response = requests.post(cypher_url, json=deep_query)
            response.raise_for_status()
            
            paths = response.json()["data"]["records"]
            
            depth_stats = {}
            for path_record in paths:
                depth = path_record["depth"]
                if depth not in depth_stats:
                    depth_stats[depth] = 0
                depth_stats[depth] += 1
            
            print(f"发现 {len(paths)} 条路径:")
            for depth, count in sorted(depth_stats.items()):
                print(f"  - {depth} 层: {count} 条路径")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 查询失败: {e}")

# 使用示例
explore_entity_relationships("人工智能", max_depth=3)
```

### 场景2：社区发现和分析

```python
def discover_communities(min_community_size: int = 5):
    """发现和分析图中的社区结构"""
    
    base_url = "http://localhost:8000/api/v1"
    community_url = f"{base_url}/graph/community-detection"
    
    payload = {
        "algorithm": "louvain",
        "min_community_size": min_community_size,
        "resolution": 1.0,
        "include_statistics": True
    }
    
    try:
        response = requests.post(community_url, json=payload)
        response.raise_for_status()
        
        result = response.json()["data"]
        communities = result["communities"]
        
        print(f"✅ 发现 {len(communities)} 个社区:")
        
        # 按社区大小排序
        communities.sort(key=lambda x: x["size"], reverse=True)
        
        for i, community in enumerate(communities, 1):
            print(f"\n🏘️  社区 {i} (ID: {community['id']}):")
            print(f"  - 节点数量: {community['size']}")
            print(f"  - 模块度: {community['modularity']:.3f}")
            print(f"  - 主要实体类型: {', '.join(community['dominant_types'])}")
            
            # 显示核心节点
            if "core_nodes" in community:
                print(f"  - 核心节点:")
                for node in community["core_nodes"][:5]:  # 只显示前5个
                    print(f"    • {node['name']} ({node['type']})")
            
            # 显示主要主题
            if "topics" in community:
                print(f"  - 主要主题: {', '.join(community['topics'][:3])}")
        
        # 显示全局统计
        if "statistics" in result:
            stats = result["statistics"]
            print(f"\n📊 社区统计:")
            print(f"  - 总社区数: {stats['total_communities']}")
            print(f"  - 平均社区大小: {stats['average_community_size']:.1f}")
            print(f"  - 最大社区大小: {stats['max_community_size']}")
            print(f"  - 全局模块度: {stats['global_modularity']:.3f}")
            print(f"  - 覆盖率: {stats['coverage']:.1%}")
        
        return communities
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 社区发现失败: {e}")
        return []

# 使用示例
communities = discover_communities(min_community_size=3)
```

### 场景3：路径分析和推理

```python
def analyze_entity_paths(source_entity: str, target_entity: str):
    """分析两个实体之间的路径"""
    
    base_url = "http://localhost:8000/api/v1"
    
    # 1. 查找实体ID
    def find_entity_id(entity_name: str):
        search_url = f"{base_url}/graph/nodes/search"
        params = {"query": entity_name, "node_type": "Entity", "limit": 1}
        
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        
        nodes = response.json()["data"]["nodes"]
        return nodes[0]["id"] if nodes else None
    
    try:
        source_id = find_entity_id(source_entity)
        target_id = find_entity_id(target_entity)
        
        if not source_id:
            print(f"❌ 未找到源实体: {source_entity}")
            return
        if not target_id:
            print(f"❌ 未找到目标实体: {target_entity}")
            return
        
        print(f"✅ 找到实体:")
        print(f"  - 源实体: {source_entity} ({source_id})")
        print(f"  - 目标实体: {target_entity} ({target_id})")
        
        # 2. 查找路径
        paths_url = f"{base_url}/graph/paths/find"
        paths_payload = {
            "source_node_id": source_id,
            "target_node_id": target_id,
            "max_depth": 4,
            "path_type": "all",  # 查找所有路径
            "limit": 10
        }
        
        response = requests.post(paths_url, json=paths_payload)
        response.raise_for_status()
        
        result = response.json()["data"]
        paths = result["paths"]
        
        if not paths:
            print(f"❌ 未找到 {source_entity} 和 {target_entity} 之间的路径")
            return
        
        print(f"\n🛤️  找到 {len(paths)} 条路径:")
        
        # 按路径长度和权重排序
        paths.sort(key=lambda x: (x["length"], -x.get("weight", 0)))
        
        for i, path in enumerate(paths, 1):
            print(f"\n路径 {i} (长度: {path['length']}, 权重: {path.get('weight', 0):.2f}):")
            
            nodes = path["nodes"]
            relationships = path["relationships"]
            
            # 构建路径字符串
            path_str = nodes[0]["name"]
            for j, rel in enumerate(relationships):
                path_str += f" --[{rel['type']}]--> {nodes[j+1]['name']}"
            
            print(f"  {path_str}")
            
            # 显示路径中的关键信息
            if "semantic_similarity" in path:
                print(f"  语义相似度: {path['semantic_similarity']:.3f}")
            
            if "confidence_score" in path:
                print(f"  置信度分数: {path['confidence_score']:.3f}")
        
        # 3. 路径分析
        print(f"\n📈 路径分析:")
        
        # 最短路径
        shortest_path = min(paths, key=lambda x: x["length"])
        print(f"  - 最短路径长度: {shortest_path['length']}")
        
        # 最高权重路径
        if any("weight" in p for p in paths):
            highest_weight_path = max(paths, key=lambda x: x.get("weight", 0))
            print(f"  - 最高权重路径: {highest_weight_path.get('weight', 0):.2f}")
        
        # 关系类型统计
        relation_types = []
        for path in paths:
            for rel in path["relationships"]:
                relation_types.append(rel["type"])
        
        from collections import Counter
        relation_counts = Counter(relation_types)
        
        print(f"  - 常见关系类型:")
        for rel_type, count in relation_counts.most_common(3):
            print(f"    • {rel_type}: {count} 次")
        
        return paths
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 路径分析失败: {e}")
        return []

# 使用示例
paths = analyze_entity_paths("人工智能", "深度学习")
```

## RAG 问答示例

### 场景1：多轮对话问答

```python
class ConversationManager:
    """对话管理器"""
    
    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url
        self.conversation_id = None
        self.conversation_history = []
    
    def start_conversation(self):
        """开始新对话"""
        import uuid
        self.conversation_id = str(uuid.uuid4())
        self.conversation_history = []
        print(f"✅ 开始新对话: {self.conversation_id}")
    
    def ask(self, question: str, include_sources: bool = True):
        """提问"""
        if not self.conversation_id:
            self.start_conversation()
        
        conversation_url = f"{self.base_url}/rag/conversation"
        
        payload = {
            "message": question,
            "conversation_id": self.conversation_id,
            "context_window": 5,
            "maintain_context": True,
            "include_sources": include_sources,
            "response_format": "detailed"
        }
        
        try:
            response = requests.post(conversation_url, json=payload)
            response.raise_for_status()
            
            result = response.json()["data"]
            answer = result["response"]
            sources = result.get("sources", [])
            
            # 添加到对话历史
            self.conversation_history.append({
                "role": "user",
                "content": question,
                "timestamp": time.time()
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "timestamp": time.time()
            })
            
            print(f"\n🤖 助手: {answer}")
            
            if sources and include_sources:
                print(f"\n📚 参考来源:")
                for i, source in enumerate(sources[:3], 1):  # 只显示前3个来源
                    print(f"  {i}. {source['document_title']} (相关度: {source['relevance_score']:.2f})")
                    if source.get('page'):
                        print(f"     页码: {source['page']}")
            
            return answer, sources
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 问答失败: {e}")
            return None, []
    
    def get_conversation_summary(self):
        """获取对话摘要"""
        if not self.conversation_history:
            print("📝 对话历史为空")
            return
        
        print(f"\n📝 对话摘要 (对话ID: {self.conversation_id}):")
        print(f"总轮次: {len([msg for msg in self.conversation_history if msg['role'] == 'user'])}")
        
        for i, message in enumerate(self.conversation_history):
            if message["role"] == "user":
                print(f"\n👤 用户: {message['content']}")
            else:
                print(f"🤖 助手: {message['content'][:100]}...")
                if message.get('sources'):
                    print(f"   📚 引用了 {len(message['sources'])} 个来源")

# 使用示例
def demo_conversation():
    """演示多轮对话"""
    
    chat = ConversationManager()
    
    # 第一轮：基础问题
    print("=" * 50)
    print("第一轮：基础问题")
    print("=" * 50)
    chat.ask("什么是人工智能？")
    
    # 第二轮：深入问题
    print("\n" + "=" * 50)
    print("第二轮：深入问题")
    print("=" * 50)
    chat.ask("人工智能有哪些主要的应用领域？")
    
    # 第三轮：关联问题
    print("\n" + "=" * 50)
    print("第三轮：关联问题")
    print("=" * 50)
    chat.ask("深度学习在这些应用中起到什么作用？")
    
    # 第四轮：具体问题
    print("\n" + "=" * 50)
    print("第四轮：具体问题")
    print("=" * 50)
    chat.ask("能给我一些深度学习在计算机视觉中的具体例子吗？")
    
    # 显示对话摘要
    chat.get_conversation_summary()

# 运行演示
demo_conversation()
```

### 场景2：多模态查询

```python
def multimodal_query_example():
    """多模态查询示例"""
    
    base_url = "http://localhost:8000/api/v1"
    multimodal_url = f"{base_url}/rag/multimodal-query"
    
    # 准备图像和文本查询
    image_path = "diagram.png"  # 假设有一个技术图表
    text_query = "这个图表展示的是什么技术架构？请详细解释各个组件的作用。"
    
    try:
        # 准备文件和数据
        files = {
            'image': ('diagram.png', open(image_path, 'rb'), 'image/png')
        }
        
        data = {
            'query': text_query,
            'query_options': json.dumps({
                'include_sources': True,
                'analysis_depth': 'detailed',
                'extract_text_from_image': True,
                'identify_objects': True,
                'describe_relationships': True
            })
        }
        
        response = requests.post(multimodal_url, files=files, data=data)
        response.raise_for_status()
        
        result = response.json()["data"]
        
        print("🖼️  多模态查询结果:")
        print(f"📝 文本查询: {text_query}")
        print(f"🤖 AI 回答: {result['response']}")
        
        # 图像分析结果
        if "image_analysis" in result:
            analysis = result["image_analysis"]
            print(f"\n🔍 图像分析:")
            
            if "detected_objects" in analysis:
                print(f"  - 检测到的对象: {', '.join(analysis['detected_objects'])}")
            
            if "extracted_text" in analysis:
                print(f"  - 提取的文本: {analysis['extracted_text']}")
            
            if "scene_description" in analysis:
                print(f"  - 场景描述: {analysis['scene_description']}")
        
        # 相关文档
        if "sources" in result:
            print(f"\n📚 相关文档:")
            for source in result["sources"][:3]:
                print(f"  - {source['document_title']} (相关度: {source['relevance_score']:.2f})")
        
    except FileNotFoundError:
        print(f"❌ 图像文件不存在: {image_path}")
    except requests.exceptions.RequestException as e:
        print(f"❌ 多模态查询失败: {e}")
    finally:
        if 'files' in locals():
            files['image'][1].close()

# 使用示例
multimodal_query_example()
```

### 场景3：批量问答和结果分析

```python
def batch_qa_analysis(questions: list):
    """批量问答和结果分析"""
    
    base_url = "http://localhost:8000/api/v1"
    batch_url = f"{base_url}/rag/batch-query"
    
    payload = {
        "queries": [
            {
                "id": f"q_{i}",
                "query": question,
                "query_type": "qa",
                "include_sources": True
            }
            for i, question in enumerate(questions, 1)
        ],
        "processing_options": {
            "parallel_processing": True,
            "max_workers": 3,
            "timeout_per_query": 30
        }
    }
    
    try:
        # 提交批量查询
        response = requests.post(batch_url, json=payload)
        response.raise_for_status()
        
        result = response.json()["data"]
        batch_id = result["batch_id"]
        
        print(f"✅ 批量查询已提交: {batch_id}")
        print(f"查询数量: {len(questions)}")
        
        # 等待结果
        status_url = f"{base_url}/rag/batch-status/{batch_id}"
        
        while True:
            time.sleep(5)
            
            status_response = requests.get(status_url)
            status_response.raise_for_status()
            
            status_data = status_response.json()["data"]
            
            if status_data["status"] == "completed":
                break
            elif status_data["status"] == "failed":
                print("❌ 批量查询失败")
                return
            
            print(f"进度: {status_data['completed_queries']}/{status_data['total_queries']}")
        
        # 获取结果
        results_url = f"{base_url}/rag/batch-results/{batch_id}"
        results_response = requests.get(results_url)
        results_response.raise_for_status()
        
        results = results_response.json()["data"]["results"]
        
        print(f"\n✅ 批量查询完成，共 {len(results)} 个结果:")
        
        # 分析结果
        total_confidence = 0
        total_sources = 0
        response_lengths = []
        
        for i, result in enumerate(results, 1):
            query_result = result["result"]
            
            print(f"\n📝 问题 {i}: {result['query']}")
            print(f"🤖 回答: {query_result['answer'][:200]}...")
            print(f"📊 置信度: {query_result['confidence']:.2f}")
            print(f"📚 来源数量: {len(query_result.get('sources', []))}")
            
            # 统计数据
            total_confidence += query_result['confidence']
            total_sources += len(query_result.get('sources', []))
            response_lengths.append(len(query_result['answer']))
        
        # 显示统计信息
        print(f"\n📊 批量查询统计:")
        print(f"  - 平均置信度: {total_confidence / len(results):.2f}")
        print(f"  - 平均来源数量: {total_sources / len(results):.1f}")
        print(f"  - 平均回答长度: {sum(response_lengths) / len(response_lengths):.0f} 字符")
        print(f"  - 最长回答: {max(response_lengths)} 字符")
        print(f"  - 最短回答: {min(response_lengths)} 字符")
        
        return results
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 批量查询失败: {e}")
        return []

# 使用示例
questions = [
    "什么是人工智能？",
    "机器学习和深度学习有什么区别？",
    "自然语言处理的主要任务有哪些？",
    "计算机视觉在实际应用中有哪些挑战？",
    "强化学习的基本原理是什么？"
]

results = batch_qa_analysis(questions)
```

## 完整工作流程

### 场景：构建学术论文知识库

```python
import os
import time
import requests
from pathlib import Path

class AcademicKnowledgeBase:
    """学术论文知识库构建器"""
    
    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def build_knowledge_base(self, papers_directory: str):
        """构建完整的学术论文知识库"""
        
        print("🚀 开始构建学术论文知识库")
        print("=" * 50)
        
        # 步骤1：上传论文
        print("\n📤 步骤1：上传论文")
        document_ids = self._upload_papers(papers_directory)
        
        if not document_ids:
            print("❌ 没有成功上传的论文")
            return
        
        # 步骤2：处理文档
        print("\n⚙️  步骤2：处理文档")
        self._process_documents(document_ids)
        
        # 步骤3：知识抽取
        print("\n🧠 步骤3：知识抽取")
        self._extract_knowledge(document_ids)
        
        # 步骤4：构建知识图谱
        print("\n🕸️  步骤4：构建知识图谱")
        self._build_knowledge_graph()
        
        # 步骤5：验证和测试
        print("\n✅ 步骤5：验证和测试")
        self._validate_knowledge_base()
        
        print("\n🎉 知识库构建完成！")
        
    def _upload_papers(self, papers_directory: str):
        """上传论文"""
        
        papers_path = Path(papers_directory)
        pdf_files = list(papers_path.glob("*.pdf"))
        
        print(f"找到 {len(pdf_files)} 个PDF文件")
        
        document_ids = []
        batch_size = 5
        
        for i in range(0, len(pdf_files), batch_size):
            batch_files = pdf_files[i:i+batch_size]
            
            files = []
            for pdf_file in batch_files:
                files.append(('files', (pdf_file.name, open(pdf_file, 'rb'), 'application/pdf')))
            
            metadata = {
                "category": "academic_paper",
                "tags": ["research", "academic"],
                "batch_id": f"batch_{i//batch_size + 1}"
            }
            
            data = {"metadata": json.dumps(metadata)}
            
            try:
                upload_url = f"{self.base_url}/document-management/upload"
                response = self.session.post(upload_url, files=files, data=data)
                response.raise_for_status()
                
                result = response.json()["data"]
                batch_doc_ids = [file_info["id"] for file_info in result["uploaded_files"]]
                document_ids.extend(batch_doc_ids)
                
                print(f"  ✅ 批次 {i//batch_size + 1}: 上传 {len(batch_doc_ids)} 个文件")
                
            except Exception as e:
                print(f"  ❌ 批次 {i//batch_size + 1} 上传失败: {e}")
            
            finally:
                for _, (_, file_obj, _) in files:
                    file_obj.close()
        
        print(f"📊 总计上传 {len(document_ids)} 个文档")
        return document_ids
    
    def _process_documents(self, document_ids: list):
        """处理文档"""
        
        processing_options = {
            "extract_text": True,
            "create_chunks": True,
            "generate_embeddings": True,
            "extract_entities": False,  # 稍后批量处理
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
        
        processed_count = 0
        
        for doc_id in document_ids:
            try:
                process_url = f"{self.base_url}/document-management/process/{doc_id}"
                payload = {"processing_options": processing_options}
                
                response = self.session.post(process_url, json=payload)
                response.raise_for_status()
                
                # 等待处理完成
                self._wait_for_processing(doc_id)
                processed_count += 1
                
                print(f"  ✅ 处理完成: {doc_id} ({processed_count}/{len(document_ids)})")
                
            except Exception as e:
                print(f"  ❌ 处理失败: {doc_id} - {e}")
        
        print(f"📊 成功处理 {processed_count} 个文档")
    
    def _wait_for_processing(self, document_id: str, timeout: int = 300):
        """等待文档处理完成"""
        
        start_time = time.time()
        status_url = f"{self.base_url}/document-management/documents/{document_id}"
        
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(status_url)
                response.raise_for_status()
                
                doc_info = response.json()["data"]
                status = doc_info["status"]
                
                if status == "processed":
                    return True
                elif status == "failed":
                    raise Exception(f"文档处理失败: {doc_info.get('error_message', 'Unknown error')}")
                
                time.sleep(5)
                
            except Exception as e:
                raise Exception(f"检查处理状态失败: {e}")
        
        raise Exception("文档处理超时")
    
    def _extract_knowledge(self, document_ids: list):
        """批量知识抽取"""
        
        batch_url = f"{self.base_url}/knowledge/extract-batch"
        
        payload = {
            "document_ids": document_ids,
            "extraction_options": {
                "extract_entities": True,
                "extract_relations": True,
                "extract_claims": True,
                "entity_types": [
                    "PERSON", "ORGANIZATION", "CONCEPT", "METHOD", 
                    "DATASET", "METRIC", "TECHNOLOGY", "LOCATION"
                ],
                "confidence_threshold": 0.7,
                "parallel_processing": True,
                "max_workers": 4
            }
        }
        
        try:
            response = self.session.post(batch_url, json=payload)
            response.raise_for_status()
            
            result = response.json()["data"]
            batch_id = result["batch_id"]
            
            print(f"  📋 批量抽取任务ID: {batch_id}")
            
            # 监控进度
            self._monitor_batch_extraction(batch_id)
            
        except Exception as e:
            print(f"  ❌ 批量知识抽取失败: {e}")
    
    def _monitor_batch_extraction(self, batch_id: str):
        """监控批量抽取进度"""
        
        status_url = f"{self.base_url}/knowledge/batch-status/{batch_id}"
        
        while True:
            try:
                response = self.session.get(status_url)
                response.raise_for_status()
                
                status_data = response.json()["data"]
                
                progress = status_data["progress_percentage"]
                print(f"  📊 抽取进度: {progress:.1f}%")
                
                if status_data["status"] == "completed":
                    stats = status_data["statistics"]
                    print(f"  ✅ 抽取完成:")
                    print(f"    - 实体: {stats['total_entities']}")
                    print(f"    - 关系: {stats['total_relations']}")
                    print(f"    - 断言: {stats['total_claims']}")
                    break
                elif status_data["status"] == "failed":
                    print(f"  ❌ 抽取失败: {status_data.get('error_message', 'Unknown error')}")
                    break
                
                time.sleep(10)
                
            except Exception as e:
                print(f"  ❌ 监控抽取进度失败: {e}")
                break
    
    def _build_knowledge_graph(self):
        """构建知识图谱"""
        
        # 社区发现
        print("  🏘️  发现社区结构...")
        community_url = f"{self.base_url}/graph/community-detection"
        
        try:
            payload = {
                "algorithm": "louvain",
                "min_community_size": 3,
                "resolution": 1.0
            }
            
            response = self.session.post(community_url, json=payload)
            response.raise_for_status()
            
            communities = response.json()["data"]["communities"]
            print(f"    发现 {len(communities)} 个社区")
            
        except Exception as e:
            print(f"    ❌ 社区发现失败: {e}")
        
        # 中心性分析
        print("  📊 计算节点中心性...")
        centrality_url = f"{self.base_url}/graph/centrality-analysis"
        
        try:
            payload = {
                "algorithms": ["betweenness", "closeness", "pagerank"],
                "node_types": ["Entity"]
            }
            
            response = self.session.post(centrality_url, json=payload)
            response.raise_for_status()
            
            print("    ✅ 中心性分析完成")
            
        except Exception as e:
            print(f"    ❌ 中心性分析失败: {e}")
    
    def _validate_knowledge_base(self):
        """验证知识库"""
        
        # 测试查询
        test_queries = [
            "知识库中有多少篇论文？",
            "主要的研究主题有哪些？",
            "最重要的研究者是谁？",
            "深度学习相关的论文有哪些？"
        ]
        
        print("  🧪 测试查询:")
        
        for query in test_queries:
            try:
                query_url = f"{self.base_url}/rag/query"
                payload = {
                    "query": query,
                    "include_sources": True
                }
                
                response = self.session.post(query_url, json=payload)
                response.raise_for_status()
                
                result = response.json()["data"]
                answer = result["answer"]
                confidence = result["confidence"]
                
                print(f"    ❓ {query}")
                print(f"    🤖 {answer[:100]}... (置信度: {confidence:.2f})")
                
            except Exception as e:
                print(f"    ❌ 查询失败: {query} - {e}")
        
        # 统计信息
        print("\n  📊 知识库统计:")
        try:
            stats_url = f"{self.base_url}/system/statistics"
            response = self.session.get(stats_url)
            response.raise_for_status()
            
            stats = response.json()["data"]
            print(f"    - 文档数量: {stats.get('documents', 0)}")
            print(f"    - 实体数量: {stats.get('entities', 0)}")
            print(f"    - 关系数量: {stats.get('relationships', 0)}")
            print(f"    - 文本块数量: {stats.get('chunks', 0)}")
            
        except Exception as e:
            print(f"    ❌ 获取统计信息失败: {e}")

# 使用示例
def build_academic_kb():
    """构建学术知识库示例"""
    
    kb_builder = AcademicKnowledgeBase()
    
    # 指定论文目录
    papers_directory = "./academic_papers"
    
    # 确保目录存在
    if not os.path.exists(papers_directory):
        print(f"❌ 论文目录不存在: {papers_directory}")
        return
    
    # 构建知识库
    kb_builder.build_knowledge_base(papers_directory)

# 运行示例
if __name__ == "__main__":
    build_academic_kb()
```

## 高级用法

### 自定义处理流水线

```python
class CustomProcessingPipeline:
    """自定义处理流水线"""
    
    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def create_custom_pipeline(self, pipeline_config: dict):
        """创建自定义处理流水线"""
        
        pipeline_url = f"{self.base_url}/processing/create-pipeline"
        
        try:
            response = self.session.post(pipeline_url, json=pipeline_config)
            response.raise_for_status()
            
            result = response.json()["data"]
            pipeline_id = result["pipeline_id"]
            
            print(f"✅ 创建自定义流水线: {pipeline_id}")
            return pipeline_id
            
        except Exception as e:
            print(f"❌ 创建流水线失败: {e}")
            return None

# 使用示例
pipeline_config = {
    "name": "Academic Paper Processing",
    "description": "专门处理学术论文的流水线",
    "steps": [
        {
            "name": "text_extraction",
            "type": "text_extractor",
            "config": {
                "extract_tables": True,
                "extract_figures": True,
                "preserve_formatting": True
            }
        },
        {
            "name": "academic_chunking",
            "type": "chunker",
            "config": {
                "strategy": "semantic_sections",
                "respect_sections": True,
                "min_chunk_size": 500,
                "max_chunk_size": 1500
            }
        },
        {
            "name": "academic_entity_extraction",
            "type": "entity_extractor",
            "config": {
                "models": ["academic_ner", "general_ner"],
                "entity_types": [
                    "AUTHOR", "INSTITUTION", "PUBLICATION", 
                    "METHOD", "DATASET", "METRIC", "CONCEPT"
                ],
                "confidence_threshold": 0.8
            }
        }
    ]
}

pipeline = CustomProcessingPipeline()
pipeline_id = pipeline.create_custom_pipeline(pipeline_config)
```

## 故障排除

### 常见问题和解决方案

```python
def diagnose_system_issues():
    """诊断系统问题"""
    
    base_url = "http://localhost:8000/api/v1"
    
    print("🔍 系统诊断开始...")
    
    # 1. 检查系统健康状态
    print("\n1. 检查系统健康状态")
    try:
        health_response = requests.get(f"{base_url}/system/health", timeout=10)
        health_response.raise_for_status()
        
        health_data = health_response.json()
        print(f"  ✅ 系统状态: {health_data['status']}")
        
        services = health_data.get('services', {})
        for service, status in services.items():
            status_icon = "✅" if status == "healthy" else "❌"
            print(f"  {status_icon} {service}: {status}")
            
    except requests.exceptions.Timeout:
        print("  ❌ 系统响应超时")
    except requests.exceptions.ConnectionError:
        print("  ❌ 无法连接到系统")
    except Exception as e:
        print(f"  ❌ 健康检查失败: {e}")
    
    # 2. 检查数据库连接
    print("\n2. 检查数据库连接")
    try:
        db_response = requests.get(f"{base_url}/system/database-status", timeout=5)
        db_response.raise_for_status()
        
        db_data = db_response.json()["data"]
        
        for db_name, db_info in db_data.items():
            status_icon = "✅" if db_info["status"] == "connected" else "❌"
            print(f"  {status_icon} {db_name}: {db_info['status']}")
            
            if db_info["status"] != "connected":
                print(f"    错误: {db_info.get('error', 'Unknown error')}")
                
    except Exception as e:
        print(f"  ❌ 数据库状态检查失败: {e}")
    
    # 3. 检查存储空间
    print("\n3. 检查存储空间")
    try:
        storage_response = requests.get(f"{base_url}/system/storage-info", timeout=5)
        storage_response.raise_for_status()
        
        storage_data = storage_response.json()["data"]
        
        for storage_type, info in storage_data.items():
            used_percent = (info["used"] / info["total"]) * 100
            status_icon = "⚠️" if used_percent > 80 else "✅"
            
            print(f"  {status_icon} {storage_type}: {used_percent:.1f}% 已使用")
            print(f"    总空间: {info['total'] / (1024**3):.1f} GB")
            print(f"    已使用: {info['used'] / (1024**3):.1f} GB")
            
    except Exception as e:
        print(f"  ❌ 存储信息检查失败: {e}")
    
    # 4. 检查最近的错误日志
    print("\n4. 检查最近的错误")
    try:
        logs_response = requests.get(f"{base_url}/system/recent-errors", timeout=5)
        logs_response.raise_for_status()
        
        errors = logs_response.json()["data"]["errors"]
        
        if errors:
            print(f"  ⚠️  发现 {len(errors)} 个最近的错误:")
            for error in errors[-5:]:  # 只显示最近5个错误
                print(f"    - {error['timestamp']}: {error['message']}")
        else:
            print("  ✅ 没有发现最近的错误")
            
    except Exception as e:
        print(f"  ❌ 错误日志检查失败: {e}")
    
    print("\n🔍 系统诊断完成")

# 运行诊断
diagnose_system_issues()
```

### 性能优化建议

```python
def performance_optimization_tips():
    """性能优化建议"""
    
    print("🚀 性能优化建议:")
    print("=" * 40)
    
    print("\n📤 文档上传优化:")
    print("  - 使用批量上传，每批5-10个文件")
    print("  - 压缩大文件后再上传")
    print("  - 避免在高峰期上传大量文件")
    
    print("\n🧠 知识抽取优化:")
    print("  - 根据文档类型选择合适的抽取策略")
    print("  - 设置合理的置信度阈值（0.7-0.8）")
    print("  - 使用并行处理加速批量抽取")
    
    print("\n🕸️  图查询优化:")
    print("  - 使用索引优化查询性能")
    print("  - 限制查询深度（建议不超过4层）")
    print("  - 缓存常用查询结果")
    
    print("\n💬 RAG 问答优化:")
    print("  - 使用对话上下文提高回答质量")
    print("  - 设置合适的检索数量（5-10个相关文档）")
    print("  - 根据问题类型选择检索策略")
    
    print("\n🔧 系统配置优化:")
    print("  - 根据硬件配置调整并发数")
    print("  - 定期清理临时文件和缓存")
    print("  - 监控内存使用情况")

# 运行优化建议
performance_optimization_tips()
```

## 总结

本文档提供了 GraphRAG 知识库系统的详细使用示例，涵盖了从基础操作到高级用法的各种场景。通过这些示例，您可以：

1. **快速上手**：通过简单的示例了解系统的基本功能
2. **深入学习**：通过复杂的工作流程掌握系统的高级特性
3. **实际应用**：参考完整的项目示例构建自己的知识库
4. **问题解决**：使用故障排除指南解决常见问题
5. **性能优化**：根据优化建议提升系统性能

### 最佳实践总结

1. **文档管理**：
   - 批量上传文件以提高效率
   - 为文档添加详细的元数据
   - 定期监控文档处理状态

2. **知识抽取**：
   - 根据文档类型选择合适的抽取策略
   - 设置合理的置信度阈值
   - 使用并行处理加速批量操作

3. **图查询**：
   - 合理设计查询深度
   - 利用索引优化查询性能
   - 缓存常用查询结果

4. **RAG 问答**：
   - 维护对话上下文以提高回答质量
   - 根据问题类型选择检索策略
   - 包含来源信息以提高可信度

5. **系统维护**：
   - 定期进行系统健康检查
   - 监控资源使用情况
   - 及时处理错误和异常

### 下一步

- 查看 [API 文档](API_DOCUMENTATION.md) 了解详细的接口说明
- 参考 [部署指南](DEPLOYMENT_GUIDE.md) 进行生产环境部署
- 阅读 [开发指南](DEVELOPMENT_GUIDE.md) 了解系统架构和扩展方法

如有问题或需要帮助，请参考故障排除部分或联系技术支持。