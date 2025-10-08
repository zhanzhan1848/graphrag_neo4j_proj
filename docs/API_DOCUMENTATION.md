# GraphRAG 知识库系统 API 文档

## 概述

GraphRAG 知识库系统提供了一套完整的 RESTful API，用于文档管理、知识抽取、图查询和 RAG 问答等功能。本文档详细介绍了所有可用的 API 端点、请求格式、响应格式和使用示例。

## 基础信息

- **基础 URL**: `http://localhost:8000`
- **API 版本**: `v1`
- **API 前缀**: `/api/v1`
- **文档地址**: `http://localhost:8000/docs`
- **OpenAPI 规范**: `http://localhost:8000/openapi.json`

## 认证

目前系统支持以下认证方式：

### Bearer Token 认证

```http
Authorization: Bearer <your-token>
```

### API Key 认证

```http
X-API-Key: <your-api-key>
```

## 通用响应格式

### 成功响应

```json
{
  "success": true,
  "data": {
    // 具体数据
  },
  "message": "操作成功",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### 错误响应

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "错误描述",
    "details": {}
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## API 端点

### 1. 系统管理 (`/api/v1/system`)

#### 1.1 健康检查

**端点**: `GET /api/v1/system/health`

**描述**: 检查系统健康状态

**请求示例**:
```bash
curl -X GET "http://localhost:8000/api/v1/system/health"
```

**响应示例**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "services": {
    "database": "healthy",
    "neo4j": "healthy",
    "redis": "healthy",
    "vector_store": "healthy"
  },
  "version": "1.0.0"
}
```

#### 1.2 系统信息

**端点**: `GET /api/v1/system/info`

**描述**: 获取系统信息

**响应示例**:
```json
{
  "name": "GraphRAG Knowledge Base",
  "version": "1.0.0",
  "environment": "development",
  "uptime": 3600,
  "statistics": {
    "documents": 150,
    "entities": 1200,
    "relationships": 3500,
    "chunks": 5000
  }
}
```

### 2. 文档管理 (`/api/v1/document-management`)

#### 2.1 上传文档

**端点**: `POST /api/v1/document-management/upload`

**描述**: 上传单个或多个文档

**请求格式**: `multipart/form-data`

**请求参数**:
- `files`: 文件列表（必需）
- `metadata`: 文档元数据（可选，JSON 字符串）

**请求示例**:
```bash
curl -X POST "http://localhost:8000/api/v1/document-management/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document1.pdf" \
  -F "files=@document2.txt" \
  -F "metadata={\"category\": \"research\", \"tags\": [\"AI\", \"ML\"]}"
```

**响应示例**:
```json
{
  "success": true,
  "data": {
    "uploaded_files": [
      {
        "id": "doc_123",
        "filename": "document1.pdf",
        "size": 1024000,
        "status": "uploaded",
        "processing_id": "proc_456"
      }
    ],
    "total_files": 2,
    "total_size": 2048000
  }
}
```

#### 2.2 处理文档

**端点**: `POST /api/v1/document-management/process/{document_id}`

**描述**: 处理已上传的文档（文本提取、分块、向量化等）

**路径参数**:
- `document_id`: 文档 ID

**请求体**:
```json
{
  "processing_options": {
    "extract_text": true,
    "create_chunks": true,
    "generate_embeddings": true,
    "extract_entities": true,
    "chunk_size": 1000,
    "chunk_overlap": 200
  }
}
```

**请求示例**:
```bash
curl -X POST "http://localhost:8000/api/v1/document-management/process/doc_123" \
  -H "Content-Type: application/json" \
  -d '{
    "processing_options": {
      "extract_text": true,
      "create_chunks": true,
      "generate_embeddings": true
    }
  }'
```

#### 2.3 获取文档列表

**端点**: `GET /api/v1/document-management/documents`

**描述**: 获取文档列表

**查询参数**:
- `page`: 页码（默认: 1）
- `size`: 每页大小（默认: 20）
- `status`: 文档状态过滤
- `category`: 分类过滤
- `search`: 搜索关键词

**请求示例**:
```bash
curl -X GET "http://localhost:8000/api/v1/document-management/documents?page=1&size=10&status=processed"
```

**响应示例**:
```json
{
  "success": true,
  "data": {
    "documents": [
      {
        "id": "doc_123",
        "title": "AI Research Paper",
        "filename": "ai_research.pdf",
        "status": "processed",
        "size": 1024000,
        "created_at": "2024-01-01T00:00:00Z",
        "processed_at": "2024-01-01T00:05:00Z",
        "metadata": {
          "category": "research",
          "tags": ["AI", "ML"]
        }
      }
    ],
    "total": 150,
    "page": 1,
    "size": 10,
    "pages": 15
  }
}
```

#### 2.4 获取文档详情

**端点**: `GET /api/v1/document-management/documents/{document_id}`

**描述**: 获取特定文档的详细信息

**响应示例**:
```json
{
  "success": true,
  "data": {
    "id": "doc_123",
    "title": "AI Research Paper",
    "filename": "ai_research.pdf",
    "status": "processed",
    "content": "文档内容...",
    "chunks": [
      {
        "id": "chunk_456",
        "content": "文本块内容...",
        "position": 0,
        "embedding": [0.1, 0.2, ...]
      }
    ],
    "entities": [
      {
        "id": "entity_789",
        "name": "人工智能",
        "type": "concept",
        "mentions": 15
      }
    ]
  }
}
```

### 3. 知识抽取 (`/api/v1/knowledge`)

#### 3.1 实体抽取

**端点**: `POST /api/v1/knowledge/extract-entities`

**描述**: 从文本中抽取实体

**请求体**:
```json
{
  "text": "苹果公司的CEO蒂姆·库克在加利福尼亚州库比蒂诺发表了演讲。",
  "entity_types": ["PERSON", "ORGANIZATION", "LOCATION"],
  "language": "zh",
  "confidence_threshold": 0.8
}
```

**请求示例**:
```bash
curl -X POST "http://localhost:8000/api/v1/knowledge/extract-entities" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "苹果公司的CEO蒂姆·库克在加利福尼亚州库比蒂诺发表了演讲。",
    "entity_types": ["PERSON", "ORGANIZATION", "LOCATION"]
  }'
```

**响应示例**:
```json
{
  "success": true,
  "data": {
    "entities": [
      {
        "id": "entity_001",
        "name": "苹果公司",
        "type": "ORGANIZATION",
        "start": 0,
        "end": 3,
        "confidence": 0.95,
        "properties": {
          "industry": "technology",
          "founded": "1976"
        }
      },
      {
        "id": "entity_002",
        "name": "蒂姆·库克",
        "type": "PERSON",
        "start": 7,
        "end": 11,
        "confidence": 0.98,
        "properties": {
          "role": "CEO",
          "company": "苹果公司"
        }
      }
    ],
    "processing_time": 0.5
  }
}
```

#### 3.2 关系抽取

**端点**: `POST /api/v1/knowledge/extract-relations`

**描述**: 从文本中抽取实体间关系

**请求体**:
```json
{
  "text": "苹果公司的CEO蒂姆·库克在加利福尼亚州库比蒂诺发表了演讲。",
  "entities": [
    {
      "name": "苹果公司",
      "type": "ORGANIZATION",
      "start": 0,
      "end": 3
    },
    {
      "name": "蒂姆·库克",
      "type": "PERSON",
      "start": 7,
      "end": 11
    }
  ],
  "relation_types": ["WORKS_FOR", "LOCATED_IN", "CEO_OF"]
}
```

**响应示例**:
```json
{
  "success": true,
  "data": {
    "relations": [
      {
        "id": "rel_001",
        "source_entity": "蒂姆·库克",
        "target_entity": "苹果公司",
        "relation_type": "CEO_OF",
        "confidence": 0.92,
        "evidence": "苹果公司的CEO蒂姆·库克",
        "properties": {
          "start_date": "2011-08-24"
        }
      }
    ],
    "processing_time": 0.3
  }
}
```

#### 3.3 批量知识抽取

**端点**: `POST /api/v1/knowledge/extract-batch`

**描述**: 批量处理多个文档的知识抽取

**请求体**:
```json
{
  "document_ids": ["doc_123", "doc_456", "doc_789"],
  "extraction_options": {
    "extract_entities": true,
    "extract_relations": true,
    "extract_claims": true,
    "entity_types": ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT"],
    "confidence_threshold": 0.8
  }
}
```

**响应示例**:
```json
{
  "success": true,
  "data": {
    "batch_id": "batch_001",
    "status": "processing",
    "total_documents": 3,
    "estimated_time": 300,
    "progress_url": "/api/v1/knowledge/batch-status/batch_001"
  }
}
```

### 4. 图查询 (`/api/v1/graph`)

#### 4.1 节点搜索

**端点**: `GET /api/v1/graph/nodes/search`

**描述**: 搜索图中的节点

**查询参数**:
- `query`: 搜索关键词
- `node_type`: 节点类型过滤
- `limit`: 结果数量限制（默认: 20）
- `properties`: 返回的属性列表

**请求示例**:
```bash
curl -X GET "http://localhost:8000/api/v1/graph/nodes/search?query=人工智能&node_type=Entity&limit=10"
```

**响应示例**:
```json
{
  "success": true,
  "data": {
    "nodes": [
      {
        "id": "entity_001",
        "labels": ["Entity", "Concept"],
        "properties": {
          "name": "人工智能",
          "type": "concept",
          "description": "模拟人类智能的技术",
          "created_at": "2024-01-01T00:00:00Z"
        },
        "relationships_count": 25
      }
    ],
    "total": 5,
    "query_time": 0.1
  }
}
```

#### 4.2 关系搜索

**端点**: `GET /api/v1/graph/relationships/search`

**描述**: 搜索图中的关系

**查询参数**:
- `source_node`: 源节点 ID
- `target_node`: 目标节点 ID
- `relation_type`: 关系类型
- `limit`: 结果数量限制

**请求示例**:
```bash
curl -X GET "http://localhost:8000/api/v1/graph/relationships/search?source_node=entity_001&relation_type=RELATED_TO"
```

#### 4.3 路径查找

**端点**: `POST /api/v1/graph/paths/find`

**描述**: 查找两个节点之间的路径

**请求体**:
```json
{
  "source_node_id": "entity_001",
  "target_node_id": "entity_002",
  "max_depth": 3,
  "path_type": "shortest",
  "relationship_types": ["RELATED_TO", "PART_OF"]
}
```

**响应示例**:
```json
{
  "success": true,
  "data": {
    "paths": [
      {
        "nodes": [
          {
            "id": "entity_001",
            "name": "人工智能"
          },
          {
            "id": "entity_003",
            "name": "机器学习"
          },
          {
            "id": "entity_002",
            "name": "深度学习"
          }
        ],
        "relationships": [
          {
            "type": "PART_OF",
            "properties": {
              "confidence": 0.9
            }
          },
          {
            "type": "RELATED_TO",
            "properties": {
              "confidence": 0.85
            }
          }
        ],
        "length": 2,
        "weight": 1.75
      }
    ],
    "total_paths": 1
  }
}
```

#### 4.4 自定义 Cypher 查询

**端点**: `POST /api/v1/graph/cypher`

**描述**: 执行自定义 Cypher 查询

**请求体**:
```json
{
  "query": "MATCH (e:Entity)-[r:RELATED_TO]->(e2:Entity) WHERE e.name CONTAINS $name RETURN e, r, e2 LIMIT 10",
  "parameters": {
    "name": "人工智能"
  }
}
```

**响应示例**:
```json
{
  "success": true,
  "data": {
    "records": [
      {
        "e": {
          "id": "entity_001",
          "name": "人工智能",
          "type": "concept"
        },
        "r": {
          "type": "RELATED_TO",
          "confidence": 0.9
        },
        "e2": {
          "id": "entity_002",
          "name": "机器学习",
          "type": "concept"
        }
      }
    ],
    "summary": {
      "query_type": "r",
      "counters": {
        "nodes_created": 0,
        "relationships_created": 0
      },
      "result_available_after": 5,
      "result_consumed_after": 10
    }
  }
}
```

### 5. RAG 问答 (`/api/v1/rag`)

#### 5.1 自然语言问答

**端点**: `POST /api/v1/rag/query`

**描述**: 基于知识库进行自然语言问答

**请求体**:
```json
{
  "query": "什么是人工智能？它有哪些应用领域？",
  "query_type": "qa",
  "retrieval_strategy": "hybrid",
  "generation_mode": "balanced",
  "max_results": 5,
  "include_sources": true,
  "language": "zh"
}
```

**请求示例**:
```bash
curl -X POST "http://localhost:8000/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是人工智能？它有哪些应用领域？",
    "query_type": "qa",
    "include_sources": true
  }'
```

**响应示例**:
```json
{
  "success": true,
  "data": {
    "query_id": "query_001",
    "answer": "人工智能（Artificial Intelligence，AI）是指由机器展现出的智能行为，它模拟人类的认知功能，如学习、推理和问题解决。主要应用领域包括：\n\n1. 自然语言处理：机器翻译、语音识别、文本分析\n2. 计算机视觉：图像识别、人脸识别、自动驾驶\n3. 机器学习：数据挖掘、预测分析、推荐系统\n4. 机器人技术：工业自动化、服务机器人\n5. 医疗健康：疾病诊断、药物发现、个性化治疗",
    "confidence": 0.92,
    "sources": [
      {
        "document_id": "doc_123",
        "document_title": "人工智能概论",
        "chunk_id": "chunk_456",
        "content": "人工智能是计算机科学的一个分支...",
        "relevance_score": 0.95,
        "page": 15
      }
    ],
    "retrieved_entities": [
      {
        "id": "entity_001",
        "name": "人工智能",
        "type": "concept",
        "relevance": 0.98
      }
    ],
    "processing_time": 1.2,
    "tokens_used": {
      "input": 25,
      "output": 150
    }
  }
}
```

#### 5.2 多模态查询

**端点**: `POST /api/v1/rag/multimodal-query`

**描述**: 支持文本和图像的多模态查询

**请求格式**: `multipart/form-data`

**请求参数**:
- `query`: 文本查询（可选）
- `image`: 图像文件（可选）
- `query_options`: 查询选项（JSON 字符串）

**请求示例**:
```bash
curl -X POST "http://localhost:8000/api/v1/rag/multimodal-query" \
  -H "Content-Type: multipart/form-data" \
  -F "query=这张图片显示的是什么技术？" \
  -F "image=@diagram.png" \
  -F "query_options={\"include_sources\": true}"
```

#### 5.3 对话查询

**端点**: `POST /api/v1/rag/conversation`

**描述**: 支持上下文的对话式查询

**请求体**:
```json
{
  "message": "深度学习和机器学习有什么区别？",
  "conversation_id": "conv_001",
  "context_window": 5,
  "maintain_context": true
}
```

**响应示例**:
```json
{
  "success": true,
  "data": {
    "conversation_id": "conv_001",
    "message_id": "msg_002",
    "response": "深度学习是机器学习的一个子集，主要区别在于：\n\n1. 网络结构：深度学习使用多层神经网络...",
    "context": [
      {
        "role": "user",
        "content": "什么是人工智能？",
        "timestamp": "2024-01-01T00:00:00Z"
      },
      {
        "role": "assistant",
        "content": "人工智能是...",
        "timestamp": "2024-01-01T00:01:00Z"
      }
    ],
    "sources": [...],
    "processing_time": 1.5
  }
}
```

## 错误代码

| 错误代码 | HTTP状态码 | 描述 |
|---------|-----------|------|
| `INVALID_REQUEST` | 400 | 请求格式错误 |
| `UNAUTHORIZED` | 401 | 未授权访问 |
| `FORBIDDEN` | 403 | 禁止访问 |
| `NOT_FOUND` | 404 | 资源不存在 |
| `METHOD_NOT_ALLOWED` | 405 | 方法不允许 |
| `VALIDATION_ERROR` | 422 | 数据验证错误 |
| `RATE_LIMIT_EXCEEDED` | 429 | 请求频率超限 |
| `INTERNAL_ERROR` | 500 | 内部服务器错误 |
| `SERVICE_UNAVAILABLE` | 503 | 服务不可用 |

## 使用限制

### 请求频率限制

- 普通用户：100 请求/分钟
- 高级用户：1000 请求/分钟
- 企业用户：10000 请求/分钟

### 文件上传限制

- 单个文件最大：100MB
- 批量上传最大：500MB
- 支持格式：PDF, TXT, DOCX, HTML, MD, PNG, JPG

### 查询限制

- 文本查询最大长度：10000 字符
- 批量查询最大数量：100 个
- 图查询最大深度：5 层
- 结果返回最大数量：1000 条

## SDK 和客户端库

### Python SDK

```bash
pip install graphrag-client
```

```python
from graphrag_client import GraphRAGClient

client = GraphRAGClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# 上传文档
result = client.upload_document("document.pdf")

# 查询知识库
answer = client.query("什么是人工智能？")
```

### JavaScript SDK

```bash
npm install @graphrag/client
```

```javascript
import { GraphRAGClient } from '@graphrag/client';

const client = new GraphRAGClient({
  baseURL: 'http://localhost:8000',
  apiKey: 'your-api-key'
});

// 查询知识库
const result = await client.query({
  query: '什么是人工智能？',
  includeSources: true
});
```

## 最佳实践

### 1. 文档上传

- 使用批量上传提高效率
- 为文档添加有意义的元数据
- 定期清理不需要的文档

### 2. 知识抽取

- 根据文档类型选择合适的抽取策略
- 设置合理的置信度阈值
- 定期更新实体和关系

### 3. 图查询

- 使用索引优化查询性能
- 限制查询深度避免性能问题
- 缓存常用查询结果

### 4. RAG 问答

- 使用混合检索策略获得更好效果
- 根据问题类型选择合适的生成模式
- 启用源引用提高可信度

## 故障排除

### 常见问题

1. **连接超时**
   - 检查网络连接
   - 确认服务是否正常运行
   - 增加请求超时时间

2. **认证失败**
   - 检查 API 密钥是否正确
   - 确认密钥是否过期
   - 验证权限设置

3. **查询结果为空**
   - 检查查询语法
   - 确认数据是否已正确导入
   - 调整查询参数

4. **处理速度慢**
   - 检查系统资源使用情况
   - 优化查询条件
   - 考虑使用缓存

### 日志和监控

- 查看应用日志：`logs/app.log`
- 监控系统状态：`/api/v1/system/health`
- 性能指标：`/api/v1/system/metrics`

## 更新日志

### v1.0.0 (2024-01-01)
- 初始版本发布
- 支持文档管理、知识抽取、图查询和 RAG 问答
- 提供完整的 RESTful API

## 联系支持

如有问题或建议，请联系：

- 邮箱：support@graphrag.com
- 文档：https://docs.graphrag.com
- GitHub：https://github.com/graphrag/graphrag