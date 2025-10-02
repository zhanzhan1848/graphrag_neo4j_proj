# Microsoft GraphRAG 集成指南

## 概述

本文档详细说明如何在 GraphRAG 知识库系统中集成 Microsoft GraphRAG 库，实现更强大的知识图谱构建和检索增强生成功能。

## 当前状态分析

### 项目现状
当前项目是一个自建的 GraphRAG 系统，具有以下特点：
- 使用 FastAPI 构建 API 服务
- 采用 Neo4j 作为图数据库
- 使用 PostgreSQL 存储文档元数据
- 集成 Weaviate 进行向量检索
- 支持多种文档格式处理

### Microsoft GraphRAG 简介
Microsoft GraphRAG 是微软研究院开发的模块化图基础检索增强生成系统，主要特点：
- 从非结构化文本中提取有意义的结构化数据
- 使用 LLM 的能力进行知识图谱构建
- 支持复杂的图查询和推理
- 提供完整的数据处理管道

## 集成方案

### 方案一：Python 库直接集成（推荐）

#### 优势
- 集成简单，开发效率高
- 可以复用现有的数据库和存储架构
- 便于定制和扩展
- 与现有 FastAPI 服务无缝集成

#### 实施步骤

1. **安装依赖**
   ```bash
   pip install graphrag==0.3.0 tiktoken==0.5.2 openai==1.3.0
   ```

2. **配置 GraphRAG**
   创建 GraphRAG 配置文件：
   ```yaml
   # config/graphrag_config.yaml
   llm:
     api_key: ${OPENAI_API_KEY}
     model: gpt-4-turbo-preview
     max_tokens: 4000
     temperature: 0.1
   
   embeddings:
     llm:
       api_key: ${OPENAI_API_KEY}
       model: text-embedding-ada-002
   
   chunks:
     size: 1200
     overlap: 100
   
   entity_extraction:
     max_gleanings: 1
   
   community_reports:
     max_length: 2000
   ```

3. **创建 GraphRAG 服务模块**
   ```python
   # app/services/graphrag_service.py
   from graphrag.query.structured_search import GlobalSearch, LocalSearch
   from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
   ```

4. **集成到现有 API**
   在现有的 FastAPI 路由中添加 GraphRAG 功能

### 方案二：独立部署集成

#### 优势
- 服务解耦，便于独立扩展
- 可以利用 GraphRAG 的完整功能
- 便于版本管理和更新

#### 实施步骤

1. **创建独立的 GraphRAG 服务**
   ```dockerfile
   # Dockerfile.graphrag
   FROM python:3.11-slim
   
   WORKDIR /app
   COPY requirements-graphrag.txt .
   RUN pip install -r requirements-graphrag.txt
   
   COPY graphrag_service/ .
   CMD ["python", "main.py"]
   ```

2. **更新 Docker Compose**
   ```yaml
   # docker-compose.yml 添加服务
   graphrag-service:
     build:
       context: .
       dockerfile: Dockerfile.graphrag
     container_name: graphrag_service
     environment:
       - OPENAI_API_KEY=${OPENAI_API_KEY}
     networks:
       - graphrag_network
   ```

## 推荐实施方案

基于您的项目现状，我推荐采用 **方案一：Python 库直接集成**，原因如下：

1. **现有架构兼容性好**：可以复用现有的 Neo4j、PostgreSQL 和 Weaviate
2. **开发效率高**：无需额外的服务部署和管理
3. **定制化程度高**：可以根据需求调整 GraphRAG 的处理流程
4. **维护成本低**：统一的代码库和部署流程

## 具体集成步骤

### 第一步：环境准备
1. 更新 requirements.txt（已完成）
2. 配置 OpenAI API Key
3. 创建 GraphRAG 配置文件

### 第二步：创建 GraphRAG 服务层
1. 实现文档处理管道
2. 集成实体和关系抽取
3. 实现图查询接口

### 第三步：API 集成
1. 添加 GraphRAG 相关的 API 端点
2. 集成到现有的文档处理流程
3. 实现混合检索（向量 + 图）

### 第四步：测试和优化
1. 单元测试和集成测试
2. 性能优化
3. 文档更新

## 注意事项

1. **API 成本**：Microsoft GraphRAG 依赖 OpenAI API，需要考虑成本控制
2. **数据隐私**：如果处理敏感数据，需要考虑本地化部署
3. **性能优化**：大规模文档处理时需要考虑批处理和缓存策略
4. **版本兼容性**：定期更新 GraphRAG 库版本，注意兼容性问题

## 下一步行动

1. 配置 OpenAI API Key
2. 创建 GraphRAG 服务模块
3. 实现基础的文档处理管道
4. 添加相关的 API 端点
5. 进行测试和验证

通过这种集成方式，您可以在现有系统的基础上，充分利用 Microsoft GraphRAG 的强大功能，提升知识图谱的构建质量和检索效果。