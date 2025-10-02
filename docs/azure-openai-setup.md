# Azure OpenAI 集成配置指南

## 概述

本文档介绍如何在 GraphRAG 系统中配置和使用 Azure OpenAI 服务，包括环境配置、API 使用和最佳实践。

## 配置步骤

### 1. Azure OpenAI 服务准备

在开始配置之前，您需要：

1. **创建 Azure OpenAI 资源**
   - 登录 Azure 门户
   - 创建 Azure OpenAI 服务实例
   - 获取端点 URL 和 API 密钥

2. **部署模型**
   - 部署 GPT-4 或 GPT-3.5-turbo 模型用于文本生成
   - 部署 text-embedding-ada-002 模型用于向量嵌入
   - 记录部署名称（deployment name）

### 2. 环境变量配置

复制 `.env.example` 文件为 `.env`，并配置以下 Azure OpenAI 相关参数：

```bash
# Azure OpenAI 配置
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# LLM 模型配置
AZURE_OPENAI_LLM_MODEL=gpt-4
AZURE_OPENAI_LLM_DEPLOYMENT_NAME=gpt-4-deployment

# 嵌入模型配置
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002-deployment

# GraphRAG 特定配置
GRAPHRAG_LLM_MAX_TOKENS=4000
GRAPHRAG_LLM_TEMPERATURE=0.1
GRAPHRAG_CHUNK_SIZE=1000
GRAPHRAG_CHUNK_OVERLAP=200
```

### 3. 依赖安装

系统已自动更新了 `requirements.txt`，包含以下 Azure OpenAI 相关依赖：

```
openai==1.3.0                    # OpenAI Python SDK (支持 Azure OpenAI)
azure-identity==1.15.0           # Azure 身份验证
azure-core==1.29.5              # Azure 核心库
tiktoken==0.5.2                 # 令牌计算
```

安装依赖：

```bash
pip install -r requirements.txt
```

## 功能特性

### 1. Azure OpenAI 服务模块

**文件位置**: `app/services/azure_openai_service.py`

**主要功能**:
- 统一的 Azure OpenAI 客户端接口
- 异步文本生成和对话
- 批量向量嵌入生成
- 令牌计算和成本控制
- 错误处理和重试机制
- 健康检查

**使用示例**:
```python
from app.services.azure_openai_service import get_azure_openai_service

# 获取服务实例
azure_openai = await get_azure_openai_service()

# 生成文本
response = await azure_openai.generate_text(
    prompt="解释什么是知识图谱",
    temperature=0.7
)

# 生成嵌入
embedding = await azure_openai.generate_single_embedding("知识图谱")
```

### 2. GraphRAG 服务模块

**文件位置**: `app/services/graphrag_service.py`

**主要功能**:
- 文档处理和分块
- 实体和关系抽取
- 断言（claims）识别
- 知识图谱查询
- RAG 问答

**使用示例**:
```python
from app.services.graphrag_service import get_graphrag_service

# 获取服务实例
graphrag = await get_graphrag_service()

# 处理文档
result = await graphrag.process_document("doc_001", document_text)

# 知识图谱查询
answer = await graphrag.query_knowledge_graph("什么是人工智能？")
```

### 3. API 端点

**文件位置**: `app/api/v1/endpoints/graphrag.py`

**可用端点**:

#### 文档处理
- `POST /api/v1/graphrag/process-document` - 处理文档文本
- `POST /api/v1/graphrag/upload-document` - 上传并处理文档文件

#### 知识抽取
- `POST /api/v1/graphrag/extract-entities` - 实体抽取
- `POST /api/v1/graphrag/extract-relationships` - 关系抽取

#### 查询和问答
- `POST /api/v1/graphrag/query-knowledge-graph` - 知识图谱查询
- `POST /api/v1/graphrag/rag-query` - RAG 问答

#### 系统信息
- `GET /api/v1/graphrag/health` - 健康检查
- `GET /api/v1/graphrag/models` - 模型配置信息

## API 使用示例

### 1. 文档处理

```bash
curl -X POST "http://localhost:8000/api/v1/graphrag/process-document" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
    "document_id": "ai_intro_001",
    "metadata": {"source": "教科书", "chapter": "第一章"}
  }'
```

### 2. 知识图谱查询

```bash
curl -X POST "http://localhost:8000/api/v1/graphrag/query-knowledge-graph" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是人工智能？",
    "max_results": 10
  }'
```

### 3. RAG 问答

```bash
curl -X POST "http://localhost:8000/api/v1/graphrag/rag-query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "人工智能有哪些应用领域？",
    "temperature": 0.7
  }'
```

## 最佳实践

### 1. 成本控制

- **令牌限制**: 设置合理的 `GRAPHRAG_LLM_MAX_TOKENS` 值
- **批量处理**: 使用批量嵌入生成减少 API 调用次数
- **缓存策略**: 对相同内容的嵌入进行缓存
- **监控使用**: 定期检查 Azure OpenAI 使用情况和成本

### 2. 性能优化

- **异步处理**: 所有 API 调用都使用异步方式
- **并发控制**: 避免过多并发请求导致速率限制
- **分块策略**: 合理设置文档分块大小和重叠
- **模型选择**: 根据任务需求选择合适的模型

### 3. 错误处理

- **重试机制**: 对临时失败进行自动重试
- **降级策略**: 在服务不可用时提供备选方案
- **日志记录**: 详细记录错误信息用于调试
- **健康检查**: 定期检查服务状态

### 4. 安全考虑

- **密钥管理**: 使用环境变量存储敏感信息
- **访问控制**: 限制 API 访问权限
- **数据隐私**: 确保敏感数据不被记录或传输
- **审计日志**: 记录重要操作用于审计

## 故障排除

### 常见问题

1. **认证失败**
   - 检查 `AZURE_OPENAI_API_KEY` 是否正确
   - 确认 `AZURE_OPENAI_ENDPOINT` 格式正确
   - 验证 API 版本是否支持

2. **模型不可用**
   - 确认模型已在 Azure OpenAI 中部署
   - 检查部署名称是否正确
   - 验证模型配额是否充足

3. **速率限制**
   - 减少并发请求数量
   - 增加请求间隔
   - 升级 Azure OpenAI 服务层级

4. **响应格式错误**
   - 检查提示词格式
   - 调整温度参数
   - 验证 JSON 解析逻辑

### 调试方法

1. **启用详细日志**
   ```bash
   export LOG_LEVEL=DEBUG
   ```

2. **健康检查**
   ```bash
   curl http://localhost:8000/api/v1/graphrag/health
   ```

3. **模型信息查询**
   ```bash
   curl http://localhost:8000/api/v1/graphrag/models
   ```

## 下一步

1. **数据库集成**: 将抽取的知识存储到 Neo4j 和 PostgreSQL
2. **向量检索**: 集成 Weaviate 进行语义检索
3. **可视化**: 添加知识图谱可视化功能
4. **批量处理**: 支持大规模文档批量处理
5. **监控告警**: 添加系统监控和告警机制

## 参考资料

- [Azure OpenAI 官方文档](https://docs.microsoft.com/azure/cognitive-services/openai/)
- [OpenAI Python SDK 文档](https://github.com/openai/openai-python)
- [GraphRAG 论文](https://arxiv.org/abs/2404.16130)
- [FastAPI 文档](https://fastapi.tiangolo.com/)