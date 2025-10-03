# 多学科知识库系统使用示例

## 目录
1. [系统启动](#系统启动)
2. [文档导入](#文档导入)
3. [知识抽取](#知识抽取)
4. [概念查询](#概念查询)
5. [关系探索](#关系探索)
6. [RAG问答](#rag问答)
7. [跨学科分析](#跨学科分析)
8. [系统维护](#系统维护)

## 系统启动

### 1. 环境准备
```bash
# 启动数据库服务
docker-compose up -d neo4j postgresql

# 安装Python依赖
pip install -r requirements.txt

# 设置环境变量
export OPENAI_API_KEY="your-api-key"
export NEO4J_PASSWORD="your-password"
export POSTGRES_PASSWORD="your-password"
```

### 2. 初始化数据库
```python
from knowledge_base import KnowledgeBase

# 初始化系统
kb = KnowledgeBase(config_path="config/system_config.json")

# 创建数据库结构
kb.initialize_databases()

# 创建学科层次结构
kb.create_discipline_hierarchy()
```

## 文档导入

### 1. 单个文档导入
```python
# 导入PDF文档
document_id = kb.import_document(
    file_path="papers/attention_is_all_you_need.pdf",
    title="Attention Is All You Need",
    authors=["Vaswani, Ashish", "Shazeer, Noam"],
    venue="NIPS 2017",
    discipline="computer_science",
    subfield="artificial_intelligence"
)

print(f"文档导入成功，ID: {document_id}")
```

### 2. 批量文档导入
```python
# 批量导入目录下的所有PDF
import os

papers_dir = "papers/computer_science/"
for filename in os.listdir(papers_dir):
    if filename.endswith('.pdf'):
        try:
            doc_id = kb.import_document(
                file_path=os.path.join(papers_dir, filename),
                auto_extract_metadata=True,  # 自动提取标题、作者等
                discipline="computer_science"
            )
            print(f"导入成功: {filename} -> {doc_id}")
        except Exception as e:
            print(f"导入失败: {filename} -> {e}")
```

### 3. 从URL导入
```python
# 从arXiv导入论文
arxiv_url = "https://arxiv.org/abs/1706.03762"
doc_id = kb.import_from_url(
    url=arxiv_url,
    discipline="computer_science",
    subfield="artificial_intelligence"
)
```

## 知识抽取

### 1. 实体抽取示例
```python
# 对特定文档进行实体抽取
entities = kb.extract_entities(document_id, chunk_ids=None)

# 查看抽取结果
for entity in entities[:10]:
    print(f"实体: {entity['name']}")
    print(f"类型: {entity['type']}")
    print(f"置信度: {entity['confidence']}")
    print(f"上下文: {entity['context'][:100]}...")
    print("---")
```

### 2. 关系抽取示例
```python
# 抽取概念间关系
relationships = kb.extract_relationships(document_id)

for rel in relationships[:10]:
    print(f"{rel['source']} --[{rel['type']}]--> {rel['target']}")
    print(f"置信度: {rel['confidence']}")
    print(f"证据: {rel['evidence'][:100]}...")
    print("---")
```

### 3. 引用关系抽取
```python
# 抽取文献引用关系
citations = kb.extract_citations(document_id)

for citation in citations:
    print(f"引用: {citation['cited_work']}")
    print(f"上下文: {citation['context']}")
    print(f"引用类型: {citation['citation_type']}")  # 支持性、对比性等
```

## 概念查询

### 1. 基础概念查询
```python
# 查找特定概念
concept = kb.find_concept("transformer")
print(f"概念: {concept['name']}")
print(f"定义: {concept['definition']}")
print(f"学科: {concept['disciplines']}")
print(f"相关文档数: {concept['document_count']}")
```

### 2. 概念层次查询
```python
# 查询概念的上下位关系
hierarchy = kb.get_concept_hierarchy("neural network")

print("上位概念:")
for parent in hierarchy['parents']:
    print(f"  - {parent['name']} ({parent['relationship']})")

print("下位概念:")
for child in hierarchy['children']:
    print(f"  - {child['name']} ({child['relationship']})")
```

### 3. 相似概念查询
```python
# 查找相似概念
similar_concepts = kb.find_similar_concepts(
    concept_name="deep learning",
    similarity_threshold=0.8,
    max_results=10
)

for concept in similar_concepts:
    print(f"{concept['name']} (相似度: {concept['similarity']:.3f})")
```

## 关系探索

### 1. 概念关系网络
```python
# 构建概念关系网络
network = kb.build_concept_network(
    center_concept="attention mechanism",
    max_hops=2,
    min_relationship_strength=0.7
)

# 可视化网络（需要安装networkx和matplotlib）
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
for node in network['nodes']:
    G.add_node(node['id'], label=node['name'])

for edge in network['edges']:
    G.add_edge(edge['source'], edge['target'], 
               weight=edge['strength'], 
               label=edge['type'])

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=1000, font_size=8)
plt.title("Attention Mechanism 概念网络")
plt.show()
```

### 2. 跨文档关系追踪
```python
# 追踪概念在不同文档中的演化
concept_evolution = kb.trace_concept_evolution(
    concept_name="transformer",
    time_range=("2017-01-01", "2023-12-31")
)

for milestone in concept_evolution:
    print(f"时间: {milestone['date']}")
    print(f"文档: {milestone['document_title']}")
    print(f"发展: {milestone['development']}")
    print("---")
```

## RAG问答

### 1. 基础问答
```python
# 简单问答
question = "什么是Transformer架构？"
answer = kb.answer_question(question)

print(f"问题: {question}")
print(f"答案: {answer['response']}")
print(f"置信度: {answer['confidence']}")

print("参考文献:")
for source in answer['sources']:
    print(f"  - {source['title']} (相关度: {source['relevance']:.3f})")
```

### 2. 多轮对话
```python
# 创建对话会话
session = kb.create_chat_session()

# 第一轮
response1 = session.ask("解释一下注意力机制的工作原理")
print(f"回答1: {response1['response']}")

# 第二轮（基于上下文）
response2 = session.ask("它与传统的RNN相比有什么优势？")
print(f"回答2: {response2['response']}")

# 查看对话历史
print("对话历史:")
for turn in session.get_history():
    print(f"Q: {turn['question']}")
    print(f"A: {turn['answer'][:100]}...")
```

### 3. 专业领域问答
```python
# 限定学科范围的问答
answer = kb.answer_question(
    question="深度学习在计算机视觉中的最新进展是什么？",
    disciplines=["computer_science"],
    subfields=["artificial_intelligence", "computer_graphics"],
    time_range=("2022-01-01", "2024-01-01")
)

print(f"答案: {answer['response']}")
print(f"涉及概念: {answer['mentioned_concepts']}")
```

## 跨学科分析

### 1. 学科交叉概念发现
```python
# 发现跨学科概念
cross_disciplinary = kb.find_cross_disciplinary_concepts(
    disciplines=["computer_science", "biology", "mathematics"],
    min_disciplines=2
)

for concept in cross_disciplinary:
    print(f"概念: {concept['name']}")
    print(f"涉及学科: {concept['disciplines']}")
    print(f"交叉强度: {concept['cross_strength']:.3f}")
    print("---")
```

### 2. 学科影响分析
```python
# 分析学科间的影响关系
influence_map = kb.analyze_disciplinary_influence(
    source_discipline="computer_science",
    target_disciplines=["biology", "physics", "chemistry"],
    time_window="2020-2024"
)

for target, influence in influence_map.items():
    print(f"计算机科学 -> {target}")
    print(f"影响强度: {influence['strength']:.3f}")
    print(f"主要概念: {influence['key_concepts'][:5]}")
    print("---")
```

### 3. 知识传播路径
```python
# 追踪概念在学科间的传播
propagation = kb.trace_concept_propagation(
    concept="machine learning",
    origin_discipline="computer_science",
    max_hops=3
)

print("概念传播路径:")
for path in propagation['paths']:
    disciplines = " -> ".join([step['discipline'] for step in path])
    print(f"路径: {disciplines}")
    print(f"传播时间: {path[-1]['first_appearance']}")
```

## 系统维护

### 1. 数据质量检查
```python
# 检查数据完整性
quality_report = kb.check_data_quality()

print("数据质量报告:")
print(f"孤立概念数: {quality_report['orphaned_concepts']}")
print(f"缺失嵌入向量: {quality_report['missing_embeddings']}")
print(f"低置信度关系: {quality_report['low_confidence_relations']}")
print(f"重复概念候选: {quality_report['duplicate_candidates']}")
```

### 2. 概念合并
```python
# 合并重复概念
merge_candidates = kb.find_merge_candidates(
    similarity_threshold=0.9
)

for candidate in merge_candidates:
    print(f"候选合并: {candidate['concept1']} <-> {candidate['concept2']}")
    print(f"相似度: {candidate['similarity']:.3f}")
    
    # 手动确认后合并
    if input("是否合并? (y/n): ").lower() == 'y':
        kb.merge_concepts(candidate['concept1'], candidate['concept2'])
        print("合并完成")
```

### 3. 性能优化
```python
# 重建索引
kb.rebuild_indexes()

# 更新统计信息
kb.update_statistics()

# 清理缓存
kb.clear_cache()

# 性能报告
perf_report = kb.get_performance_report()
print(f"平均查询时间: {perf_report['avg_query_time']:.3f}s")
print(f"缓存命中率: {perf_report['cache_hit_rate']:.2%}")
```

### 4. 数据导出
```python
# 导出知识图谱
kb.export_knowledge_graph(
    format="graphml",
    output_path="exports/knowledge_graph.graphml",
    include_embeddings=False
)

# 导出概念词典
kb.export_concept_dictionary(
    output_path="exports/concept_dictionary.json",
    include_definitions=True,
    include_relationships=True
)

# 导出引用网络
kb.export_citation_network(
    output_path="exports/citation_network.json",
    disciplines=["computer_science"]
)
```

## API使用示例

### 1. REST API调用
```python
import requests

# 文档上传
files = {'file': open('paper.pdf', 'rb')}
data = {
    'title': 'Sample Paper',
    'discipline': 'computer_science',
    'subfield': 'artificial_intelligence'
}

response = requests.post('http://localhost:8000/api/documents', 
                        files=files, data=data)
document_id = response.json()['document_id']

# 概念查询
response = requests.get(f'http://localhost:8000/api/concepts/search',
                       params={'query': 'transformer', 'limit': 10})
concepts = response.json()['concepts']

# RAG问答
response = requests.post('http://localhost:8000/api/chat',
                        json={'question': '什么是注意力机制？'})
answer = response.json()['answer']
```

### 2. WebSocket实时更新
```python
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    if data['type'] == 'processing_update':
        print(f"处理进度: {data['progress']}%")
    elif data['type'] == 'extraction_complete':
        print(f"抽取完成: {data['entities_count']} 个实体")

ws = websocket.WebSocketApp("ws://localhost:8000/ws",
                           on_message=on_message)
ws.run_forever()
```

## 高级用法

### 1. 自定义实体类型
```python
# 添加新的实体类型
kb.add_entity_type(
    name="research_method",
    description="研究方法或技术",
    extraction_patterns=[
        r"(\w+)\s+(method|technique|approach|algorithm)",
        r"using\s+(\w+)\s+to"
    ]
)

# 重新训练实体识别模型
kb.retrain_ner_model(include_new_types=True)
```

### 2. 自定义关系类型
```python
# 定义新的关系类型
kb.add_relationship_type(
    name="IMPROVES_UPON",
    description="一个概念改进了另一个概念",
    extraction_prompt="识别文本中表示改进关系的概念对",
    confidence_threshold=0.8
)
```

### 3. 领域特定配置
```python
# 为特定领域创建专门配置
biology_config = kb.create_domain_config(
    discipline="biology",
    entity_types=["gene", "protein", "pathway", "disease"],
    relationship_types=["REGULATES", "INTERACTS_WITH", "CAUSES"],
    extraction_models=["biobert", "pubmedbert"]
)

# 使用领域配置处理文档
kb.process_document(doc_id, config=biology_config)
```

这些示例展示了多学科知识库系统的主要功能和使用方法。系统支持灵活的配置和扩展，可以根据具体需求进行定制化开发。