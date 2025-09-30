# GraphRAG 系统 API 文档

## 概述

GraphRAG系统提供RESTful API接口，支持文档管理、知识查询、实体关系操作等功能。本文档详细描述了所有可用的API端点、请求格式、响应格式和使用示例。

## 目录

1. [API概览](#1-api概览)
2. [认证和授权](#2-认证和授权)
3. [文档管理API](#3-文档管理api)
4. [知识查询API](#4-知识查询api)
5. [实体管理API](#5-实体管理api)
6. [关系管理API](#6-关系管理api)
7. [图查询API](#7-图查询api)
8. [系统管理API](#8-系统管理api)
9. [错误处理](#9-错误处理)
10. [SDK和客户端](#10-sdk和客户端)

## 1. API概览

### 1.1 基础信息

- **Base URL**: `http://localhost:8000/api/v1`
- **协议**: HTTP/HTTPS
- **数据格式**: JSON
- **字符编码**: UTF-8
- **API版本**: v1

### 1.2 API架构图

```mermaid
graph TB
    A[客户端应用] --> B[API网关]
    B --> C[认证中间件]
    C --> D[路由分发]
    
    D --> E[文档管理API]
    D --> F[知识查询API]
    D --> G[实体管理API]
    D --> H[关系管理API]
    D --> I[图查询API]
    D --> J[系统管理API]
    
    E --> K[文档服务]
    F --> L[查询服务]
    G --> M[实体服务]
    H --> N[关系服务]
    I --> O[图数据库]
    J --> P[系统服务]
    
    K --> Q[(PostgreSQL)]
    L --> Q
    L --> R[(Neo4j)]
    M --> Q
    M --> R
    N --> Q
    N --> R
    O --> R
    P --> S[(Redis)]
```

### 1.3 通用响应格式

所有API响应都遵循统一的格式：

```json
{
  "success": true,
  "data": {},
  "message": "操作成功",
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "req_123456789"
}
```

错误响应格式：

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "请求参数验证失败",
    "details": {
      "field": "filename",
      "reason": "文件名不能为空"
    }
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "req_123456789"
}
```

## 2. 认证和授权

### 2.1 JWT认证

系统使用JWT (JSON Web Token) 进行用户认证。

#### 获取访问令牌

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "password123"
}
```

响应：

```json
{
  "success": true,
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "Bearer",
    "expires_in": 3600,
    "user": {
      "id": "user_123",
      "username": "user@example.com",
      "roles": ["user"]
    }
  }
}
```

#### 使用访问令牌

在所有需要认证的请求中添加Authorization头：

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### 刷新令牌

```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### 2.2 API密钥认证

对于服务间调用，支持API密钥认证：

```http
X-API-Key: your-api-key-here
```

## 3. 文档管理API

### 3.1 上传文档

上传单个文档进行处理。

```http
POST /api/v1/documents/upload
Authorization: Bearer {token}
Content-Type: multipart/form-data

file: [文件内容]
metadata: {
  "title": "文档标题",
  "description": "文档描述",
  "tags": ["标签1", "标签2"],
  "source": "来源信息"
}
```

响应：

```json
{
  "success": true,
  "data": {
    "document_id": "doc_123456",
    "filename": "example.pdf",
    "file_type": "pdf",
    "file_size": 1024000,
    "status": "processing",
    "upload_time": "2024-01-01T00:00:00Z",
    "metadata": {
      "title": "文档标题",
      "description": "文档描述",
      "tags": ["标签1", "标签2"],
      "source": "来源信息"
    }
  }
}
```

### 3.2 批量上传文档

```http
POST /api/v1/documents/batch-upload
Authorization: Bearer {token}
Content-Type: multipart/form-data

files: [文件1, 文件2, ...]
metadata: {
  "batch_name": "批次名称",
  "description": "批次描述"
}
```

### 3.3 获取文档列表

```http
GET /api/v1/documents?page=1&size=20&status=completed&file_type=pdf
Authorization: Bearer {token}
```

查询参数：
- `page`: 页码（默认1）
- `size`: 每页数量（默认20，最大100）
- `status`: 文档状态（processing, completed, failed）
- `file_type`: 文件类型（pdf, txt, docx等）
- `search`: 搜索关键词
- `tags`: 标签过滤
- `date_from`: 开始日期
- `date_to`: 结束日期

响应：

```json
{
  "success": true,
  "data": {
    "documents": [
      {
        "id": "doc_123456",
        "filename": "example.pdf",
        "file_type": "pdf",
        "file_size": 1024000,
        "status": "completed",
        "upload_time": "2024-01-01T00:00:00Z",
        "processing_time": "2024-01-01T00:05:00Z",
        "metadata": {
          "title": "文档标题",
          "page_count": 10,
          "entity_count": 25,
          "relation_count": 15
        }
      }
    ],
    "pagination": {
      "page": 1,
      "size": 20,
      "total": 100,
      "pages": 5
    }
  }
}
```

### 3.4 获取文档详情

```http
GET /api/v1/documents/{document_id}
Authorization: Bearer {token}
```

响应：

```json
{
  "success": true,
  "data": {
    "id": "doc_123456",
    "filename": "example.pdf",
    "file_type": "pdf",
    "file_size": 1024000,
    "status": "completed",
    "upload_time": "2024-01-01T00:00:00Z",
    "processing_time": "2024-01-01T00:05:00Z",
    "metadata": {
      "title": "文档标题",
      "description": "文档描述",
      "page_count": 10,
      "word_count": 5000,
      "entity_count": 25,
      "relation_count": 15,
      "tags": ["标签1", "标签2"]
    },
    "processing_stats": {
      "chunks_created": 50,
      "entities_extracted": 25,
      "relations_extracted": 15,
      "processing_duration": 300
    }
  }
}
```

### 3.5 删除文档

```http
DELETE /api/v1/documents/{document_id}
Authorization: Bearer {token}
```

### 3.6 重新处理文档

```http
POST /api/v1/documents/{document_id}/reprocess
Authorization: Bearer {token}
Content-Type: application/json

{
  "force": true,
  "options": {
    "extract_entities": true,
    "extract_relations": true,
    "update_embeddings": true
  }
}
```

## 4. 知识查询API

### 4.1 自然语言查询

使用自然语言查询知识库。

```http
POST /api/v1/query/natural
Authorization: Bearer {token}
Content-Type: application/json

{
  "query": "什么是机器学习？",
  "options": {
    "max_results": 10,
    "include_sources": true,
    "similarity_threshold": 0.7,
    "search_type": "hybrid"
  }
}
```

响应：

```json
{
  "success": true,
  "data": {
    "query": "什么是机器学习？",
    "results": [
      {
        "content": "机器学习是人工智能的一个分支...",
        "score": 0.95,
        "source": {
          "document_id": "doc_123456",
          "chunk_id": "chunk_789",
          "page": 1,
          "position": {
            "start": 100,
            "end": 500
          }
        },
        "entities": [
          {
            "name": "机器学习",
            "type": "CONCEPT",
            "confidence": 0.9
          }
        ]
      }
    ],
    "total_results": 25,
    "processing_time": 0.15,
    "search_metadata": {
      "vector_search_results": 15,
      "keyword_search_results": 10,
      "graph_search_results": 5
    }
  }
}
```

### 4.2 语义搜索

基于向量相似度的语义搜索。

```http
POST /api/v1/query/semantic
Authorization: Bearer {token}
Content-Type: application/json

{
  "query": "深度学习算法",
  "options": {
    "max_results": 20,
    "similarity_threshold": 0.8,
    "document_filters": {
      "document_ids": ["doc_123", "doc_456"],
      "file_types": ["pdf", "txt"],
      "tags": ["AI", "机器学习"]
    }
  }
}
```

### 4.3 关键词搜索

传统的关键词搜索。

```http
GET /api/v1/query/keyword?q=机器学习&highlight=true&page=1&size=10
Authorization: Bearer {token}
```

### 4.4 混合搜索

结合语义搜索和关键词搜索。

```http
POST /api/v1/query/hybrid
Authorization: Bearer {token}
Content-Type: application/json

{
  "query": "神经网络训练方法",
  "options": {
    "semantic_weight": 0.7,
    "keyword_weight": 0.3,
    "max_results": 15,
    "rerank": true
  }
}
```

## 5. 实体管理API

### 5.1 获取实体列表

```http
GET /api/v1/entities?type=PERSON&page=1&size=20&search=张三
Authorization: Bearer {token}
```

查询参数：
- `type`: 实体类型（PERSON, ORGANIZATION, CONCEPT等）
- `search`: 搜索关键词
- `document_id`: 文档ID过滤
- `confidence_min`: 最小置信度
- `page`, `size`: 分页参数

响应：

```json
{
  "success": true,
  "data": {
    "entities": [
      {
        "id": "entity_123",
        "name": "张三",
        "type": "PERSON",
        "confidence": 0.95,
        "properties": {
          "title": "教授",
          "organization": "清华大学",
          "field": "计算机科学"
        },
        "mentions": [
          {
            "document_id": "doc_123",
            "chunk_id": "chunk_456",
            "position": {
              "start": 100,
              "end": 102
            },
            "context": "...张三教授在机器学习领域..."
          }
        ],
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z"
      }
    ],
    "pagination": {
      "page": 1,
      "size": 20,
      "total": 150,
      "pages": 8
    },
    "statistics": {
      "total_entities": 150,
      "by_type": {
        "PERSON": 50,
        "ORGANIZATION": 30,
        "CONCEPT": 70
      }
    }
  }
}
```

### 5.2 获取实体详情

```http
GET /api/v1/entities/{entity_id}
Authorization: Bearer {token}
```

### 5.3 创建实体

```http
POST /api/v1/entities
Authorization: Bearer {token}
Content-Type: application/json

{
  "name": "人工智能",
  "type": "CONCEPT",
  "properties": {
    "description": "模拟人类智能的技术",
    "category": "技术领域"
  },
  "aliases": ["AI", "Artificial Intelligence"]
}
```

### 5.4 更新实体

```http
PUT /api/v1/entities/{entity_id}
Authorization: Bearer {token}
Content-Type: application/json

{
  "properties": {
    "description": "更新后的描述",
    "category": "新分类"
  }
}
```

### 5.5 删除实体

```http
DELETE /api/v1/entities/{entity_id}
Authorization: Bearer {token}
```

### 5.6 合并实体

```http
POST /api/v1/entities/{entity_id}/merge
Authorization: Bearer {token}
Content-Type: application/json

{
  "target_entity_id": "entity_456",
  "merge_strategy": "keep_both_properties"
}
```

## 6. 关系管理API

### 6.1 获取关系列表

```http
GET /api/v1/relations?type=WORKS_AT&source_entity=entity_123&page=1&size=20
Authorization: Bearer {token}
```

查询参数：
- `type`: 关系类型
- `source_entity`: 源实体ID
- `target_entity`: 目标实体ID
- `confidence_min`: 最小置信度
- `document_id`: 文档ID过滤

响应：

```json
{
  "success": true,
  "data": {
    "relations": [
      {
        "id": "rel_123",
        "type": "WORKS_AT",
        "source_entity": {
          "id": "entity_123",
          "name": "张三",
          "type": "PERSON"
        },
        "target_entity": {
          "id": "entity_456",
          "name": "清华大学",
          "type": "ORGANIZATION"
        },
        "confidence": 0.92,
        "properties": {
          "position": "教授",
          "department": "计算机系",
          "start_date": "2020-01-01"
        },
        "evidence": [
          {
            "document_id": "doc_123",
            "chunk_id": "chunk_456",
            "text": "张三教授在清华大学计算机系工作",
            "position": {
              "start": 50,
              "end": 65
            }
          }
        ],
        "created_at": "2024-01-01T00:00:00Z"
      }
    ],
    "pagination": {
      "page": 1,
      "size": 20,
      "total": 80,
      "pages": 4
    }
  }
}
```

### 6.2 创建关系

```http
POST /api/v1/relations
Authorization: Bearer {token}
Content-Type: application/json

{
  "type": "COLLABORATES_WITH",
  "source_entity_id": "entity_123",
  "target_entity_id": "entity_456",
  "confidence": 0.9,
  "properties": {
    "project": "AI研究项目",
    "duration": "2年"
  },
  "evidence": [
    {
      "document_id": "doc_789",
      "chunk_id": "chunk_101",
      "text": "张三和李四在AI项目上合作"
    }
  ]
}
```

### 6.3 更新关系

```http
PUT /api/v1/relations/{relation_id}
Authorization: Bearer {token}
Content-Type: application/json

{
  "confidence": 0.95,
  "properties": {
    "status": "active",
    "last_updated": "2024-01-01"
  }
}
```

### 6.4 删除关系

```http
DELETE /api/v1/relations/{relation_id}
Authorization: Bearer {token}
```

## 7. 图查询API

### 7.1 路径查询

查找两个实体之间的路径。

```http
POST /api/v1/graph/path
Authorization: Bearer {token}
Content-Type: application/json

{
  "source_entity_id": "entity_123",
  "target_entity_id": "entity_456",
  "max_depth": 3,
  "relation_types": ["WORKS_AT", "COLLABORATES_WITH"],
  "limit": 10
}
```

响应：

```json
{
  "success": true,
  "data": {
    "paths": [
      {
        "length": 2,
        "nodes": [
          {
            "id": "entity_123",
            "name": "张三",
            "type": "PERSON"
          },
          {
            "id": "entity_789",
            "name": "AI实验室",
            "type": "ORGANIZATION"
          },
          {
            "id": "entity_456",
            "name": "李四",
            "type": "PERSON"
          }
        ],
        "relationships": [
          {
            "id": "rel_123",
            "type": "WORKS_AT",
            "confidence": 0.9
          },
          {
            "id": "rel_456",
            "type": "WORKS_AT",
            "confidence": 0.85
          }
        ]
      }
    ],
    "total_paths": 5
  }
}
```

### 7.2 邻居查询

获取实体的邻居节点。

```http
GET /api/v1/graph/neighbors/{entity_id}?depth=2&limit=20&relation_types=WORKS_AT,COLLABORATES_WITH
Authorization: Bearer {token}
```

### 7.3 子图查询

获取指定实体周围的子图。

```http
POST /api/v1/graph/subgraph
Authorization: Bearer {token}
Content-Type: application/json

{
  "center_entities": ["entity_123", "entity_456"],
  "max_depth": 2,
  "max_nodes": 50,
  "include_relation_types": ["WORKS_AT", "COLLABORATES_WITH"],
  "exclude_entity_types": ["LOCATION"]
}
```

### 7.4 图统计

获取图的统计信息。

```http
GET /api/v1/graph/statistics
Authorization: Bearer {token}
```

响应：

```json
{
  "success": true,
  "data": {
    "nodes": {
      "total": 10000,
      "by_type": {
        "PERSON": 3000,
        "ORGANIZATION": 1500,
        "CONCEPT": 5500
      }
    },
    "relationships": {
      "total": 25000,
      "by_type": {
        "WORKS_AT": 5000,
        "COLLABORATES_WITH": 8000,
        "RELATED_TO": 12000
      }
    },
    "graph_metrics": {
      "density": 0.0025,
      "average_degree": 5.0,
      "clustering_coefficient": 0.3,
      "connected_components": 1
    }
  }
}
```

### 7.5 Cypher查询

执行自定义Cypher查询（需要管理员权限）。

```http
POST /api/v1/graph/cypher
Authorization: Bearer {token}
Content-Type: application/json

{
  "query": "MATCH (p:PERSON)-[:WORKS_AT]->(o:ORGANIZATION) WHERE o.name = '清华大学' RETURN p.name, p.title LIMIT 10",
  "parameters": {}
}
```

## 8. 系统管理API

### 8.1 健康检查

```http
GET /api/v1/health
```

响应：

```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "timestamp": "2024-01-01T00:00:00Z",
    "version": "1.0.0",
    "components": {
      "database": {
        "status": "healthy",
        "response_time": 0.05
      },
      "redis": {
        "status": "healthy",
        "response_time": 0.01
      },
      "neo4j": {
        "status": "healthy",
        "response_time": 0.03
      }
    }
  }
}
```

### 8.2 系统信息

```http
GET /api/v1/system/info
Authorization: Bearer {token}
```

### 8.3 系统指标

```http
GET /api/v1/system/metrics
Authorization: Bearer {token}
```

响应：

```json
{
  "success": true,
  "data": {
    "system": {
      "cpu_usage": 45.2,
      "memory_usage": 68.5,
      "disk_usage": 32.1
    },
    "application": {
      "active_connections": 25,
      "requests_per_minute": 150,
      "average_response_time": 0.25
    },
    "database": {
      "postgresql": {
        "active_connections": 15,
        "query_rate": 50
      },
      "neo4j": {
        "active_transactions": 5,
        "query_rate": 20
      },
      "redis": {
        "memory_usage": 45.2,
        "hit_rate": 95.5
      }
    }
  }
}
```

### 8.4 任务状态

```http
GET /api/v1/system/tasks?status=running&page=1&size=20
Authorization: Bearer {token}
```

### 8.5 日志查询

```http
GET /api/v1/system/logs?level=ERROR&start_time=2024-01-01T00:00:00Z&end_time=2024-01-01T23:59:59Z&limit=100
Authorization: Bearer {token}
```

## 9. 错误处理

### 9.1 错误代码

| 错误代码 | HTTP状态码 | 描述 |
|----------|------------|------|
| VALIDATION_ERROR | 400 | 请求参数验证失败 |
| UNAUTHORIZED | 401 | 未授权访问 |
| FORBIDDEN | 403 | 权限不足 |
| NOT_FOUND | 404 | 资源不存在 |
| CONFLICT | 409 | 资源冲突 |
| RATE_LIMIT_EXCEEDED | 429 | 请求频率超限 |
| INTERNAL_ERROR | 500 | 服务器内部错误 |
| SERVICE_UNAVAILABLE | 503 | 服务不可用 |

### 9.2 错误响应示例

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "请求参数验证失败",
    "details": {
      "field": "query",
      "reason": "查询内容不能为空",
      "value": ""
    }
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "req_123456789"
}
```

### 9.3 错误处理最佳实践

1. **检查HTTP状态码**：首先检查HTTP状态码判断请求是否成功
2. **解析错误信息**：从响应体中获取详细的错误信息
3. **重试机制**：对于临时性错误（5xx），实现指数退避重试
4. **日志记录**：记录错误信息和request_id用于问题追踪

## 10. SDK和客户端

### 10.1 Python SDK

```python
# 安装
pip install graphrag-client

# 使用示例
from graphrag_client import GraphRAGClient

client = GraphRAGClient(
    base_url="http://localhost:8000/api/v1",
    api_key="your-api-key"
)

# 上传文档
with open("document.pdf", "rb") as f:
    result = client.documents.upload(
        file=f,
        metadata={
            "title": "测试文档",
            "tags": ["AI", "机器学习"]
        }
    )

# 查询知识
results = client.query.natural(
    query="什么是深度学习？",
    max_results=10
)

# 获取实体
entities = client.entities.list(
    type="PERSON",
    search="张三"
)
```

### 10.2 JavaScript SDK

```javascript
// 安装
npm install @graphrag/client

// 使用示例
import { GraphRAGClient } from '@graphrag/client';

const client = new GraphRAGClient({
  baseURL: 'http://localhost:8000/api/v1',
  apiKey: 'your-api-key'
});

// 上传文档
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('metadata', JSON.stringify({
  title: '测试文档',
  tags: ['AI', '机器学习']
}));

const result = await client.documents.upload(formData);

// 查询知识
const results = await client.query.natural({
  query: '什么是深度学习？',
  maxResults: 10
});
```

### 10.3 cURL示例

```bash
# 登录获取token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user@example.com", "password": "password123"}'

# 上传文档
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@document.pdf" \
  -F 'metadata={"title": "测试文档"}'

# 查询知识
curl -X POST http://localhost:8000/api/v1/query/natural \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是机器学习？", "options": {"max_results": 10}}'

# 获取实体列表
curl -X GET "http://localhost:8000/api/v1/entities?type=PERSON&page=1&size=20" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 10.4 Postman集合

我们提供了完整的Postman集合，包含所有API端点的示例请求：

```json
{
  "info": {
    "name": "GraphRAG API Collection",
    "description": "GraphRAG系统完整API集合",
    "version": "1.0.0"
  },
  "auth": {
    "type": "bearer",
    "bearer": [
      {
        "key": "token",
        "value": "{{access_token}}",
        "type": "string"
      }
    ]
  },
  "variable": [
    {
      "key": "base_url",
      "value": "http://localhost:8000/api/v1"
    },
    {
      "key": "access_token",
      "value": ""
    }
  ]
}
```

## 附录

### A. 数据模型

#### 文档模型

```json
{
  "id": "string",
  "filename": "string",
  "file_type": "string",
  "file_size": "integer",
  "status": "enum[processing, completed, failed]",
  "upload_time": "datetime",
  "processing_time": "datetime",
  "metadata": {
    "title": "string",
    "description": "string",
    "tags": ["string"],
    "source": "string",
    "page_count": "integer",
    "word_count": "integer"
  }
}
```

#### 实体模型

```json
{
  "id": "string",
  "name": "string",
  "type": "enum[PERSON, ORGANIZATION, CONCEPT, LOCATION, EVENT]",
  "confidence": "float",
  "properties": "object",
  "aliases": ["string"],
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

#### 关系模型

```json
{
  "id": "string",
  "type": "string",
  "source_entity_id": "string",
  "target_entity_id": "string",
  "confidence": "float",
  "properties": "object",
  "evidence": [
    {
      "document_id": "string",
      "chunk_id": "string",
      "text": "string",
      "position": {
        "start": "integer",
        "end": "integer"
      }
    }
  ],
  "created_at": "datetime"
}
```

### B. 限制和配额

| 项目 | 限制 |
|------|------|
| 文件大小 | 100MB |
| 批量上传文件数 | 50个 |
| API请求频率 | 1000次/分钟 |
| 查询结果数量 | 1000条 |
| Cypher查询超时 | 30秒 |
| 并发连接数 | 100个 |

### C. 版本历史

| 版本 | 发布日期 | 主要变更 |
|------|----------|----------|
| v1.0.0 | 2024-01-01 | 初始版本发布 |
| v1.1.0 | 2024-02-01 | 添加图查询API |
| v1.2.0 | 2024-03-01 | 支持批量操作 |

---

**联系我们**

如有API使用问题，请联系：
- 邮箱：api-support@graphrag.com
- 文档：https://docs.graphrag.com
- GitHub：https://github.com/graphrag/api