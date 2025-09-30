# Weaviate 问题排查指南

## 问题描述

在启动 Weaviate 服务时，可能会遇到以下警告信息：

```
{"level":"warning","msg":"Multiple vector spaces are present, GraphQL Explore and REST API list objects endpoint module include params has been disabled as a result.","time":"2025-09-30T06:39:20Z"}
```

## 问题分析

这个警告出现的原因是：

1. **多向量空间冲突**：当启用多个向量化模块时（如 `text2vec-openai`, `text2vec-cohere`, `text2vec-huggingface` 等），Weaviate 会创建多个向量空间，导致某些 API 功能被禁用。

2. **版本兼容性**：使用的 Weaviate 版本 `1.22.4` 相对较旧，新版本对多向量空间的处理更加优化。

## 解决方案

### 1. 升级 Weaviate 版本

将 Weaviate 版本从 `1.22.4` 升级到 `1.33.0`：

```yaml
weaviate:
  image: cr.weaviate.io/semitechnologies/weaviate:1.33.0
```

### 2. 优化模块配置

减少启用的向量化模块，只保留必要的模块：

```yaml
environment:
  # 原配置（会导致警告）
  # ENABLE_MODULES: 'text2vec-openai,text2vec-cohere,text2vec-huggingface,ref2vec-centroid,generative-openai,qna-openai'
  
  # 优化后配置
  ENABLE_MODULES: 'text2vec-openai,generative-openai'
```

### 3. 添加性能优化配置

```yaml
environment:
  LIMIT_RESOURCES: 'false'
  GOMAXPROCS: '4'
  AUTO_SCHEMA_ENABLED: 'true'
  LOG_LEVEL: 'info'
```

### 4. 优化健康检查配置

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8080/v1/.well-known/ready"]
  interval: 15s
  timeout: 10s
  retries: 10
  start_period: 30s
```

## 当前状态

经过配置优化后，Weaviate 服务现在：

1. **版本**：已升级到 1.33.0
2. **启动状态**：服务正常启动并运行
3. **API 可用性**：REST API 端点正常响应
4. **警告状态**：多向量空间警告仍然存在，但不影响核心功能

## 警告说明

虽然仍有多向量空间警告，但这不会影响系统的核心功能：

- ✅ REST API 正常工作
- ✅ 向量搜索功能可用
- ✅ 数据存储和检索正常
- ⚠️ GraphQL Explore 功能受限
- ⚠️ 某些 REST API 列表端点的模块参数被禁用

## 配置说明

### 保留的模块

- **text2vec-openai**：OpenAI 文本向量化模块，用于生成文本嵌入
- **generative-openai**：OpenAI 生成模块，用于 RAG 功能

### 移除的模块

- **text2vec-cohere**：Cohere 向量化模块（可根据需要重新启用）
- **text2vec-huggingface**：HuggingFace 向量化模块（可根据需要重新启用）
- **ref2vec-centroid**：引用向量化模块
- **qna-openai**：问答模块（已被 generative-openai 替代）

## 验证修复

1. 重启 Weaviate 服务：
   ```bash
   docker-compose down weaviate
   docker-compose up -d weaviate
   ```

2. 检查服务状态：
   ```bash
   docker-compose ps weaviate
   ```

3. 验证 API 可用性：
   ```bash
   curl http://localhost:8080/v1/.well-known/ready
   curl http://localhost:8080/v1/meta
   ```

## 注意事项

1. **数据兼容性**：升级版本前建议备份现有数据
2. **模块依赖**：如果应用代码依赖特定模块，需要相应调整
3. **性能影响**：减少模块数量可能会提升启动速度和运行性能
4. **功能限制**：多向量空间警告会限制某些 GraphQL 和 REST API 功能

## 相关链接

- [Weaviate 官方文档](https://weaviate.io/developers/weaviate)
- [Weaviate GitHub 仓库](https://github.com/weaviate/weaviate)
- [向量化模块文档](https://weaviate.io/developers/weaviate/modules)