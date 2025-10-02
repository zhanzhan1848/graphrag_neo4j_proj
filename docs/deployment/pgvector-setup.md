# PostgreSQL + pgvector 配置说明

## 概述

本项目使用 PostgreSQL 数据库配合 pgvector 扩展来提供向量存储和检索功能，支持文档嵌入的相似度搜索。

## 配置详情

### Docker 镜像

我们使用官方的 `pgvector/pgvector:pg15` 镜像，该镜像基于 PostgreSQL 15 并预装了 pgvector 扩展。

```yaml
postgres:
  image: pgvector/pgvector:pg15  # 包含 pgvector 扩展的 PostgreSQL 15
  container_name: graphrag_postgres
  # ... 其他配置
```

### 自动初始化

系统会在数据库启动时自动执行初始化脚本 `init-scripts/postgres/01-init-pgvector.sql`，该脚本会：

1. **启用 pgvector 扩展**
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

2. **创建向量嵌入表**
   ```sql
   CREATE TABLE document_embeddings (
       id SERIAL PRIMARY KEY,
       document_id INTEGER NOT NULL,
       chunk_id INTEGER NOT NULL,
       embedding vector(1536),  -- 支持 OpenAI 嵌入维度
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
       updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

3. **创建向量索引**
   ```sql
   CREATE INDEX idx_document_embeddings_vector 
   ON document_embeddings USING hnsw (embedding vector_cosine_ops);
   ```

## 向量操作示例

### 插入向量数据

```sql
INSERT INTO document_embeddings (document_id, chunk_id, embedding) 
VALUES (1, 1, '[0.1, 0.2, 0.3, ...]'::vector);
```

### 向量相似度搜索

```sql
-- 余弦相似度搜索
SELECT document_id, chunk_id, 1 - (embedding <=> '[0.1, 0.2, 0.3, ...]'::vector) AS similarity
FROM document_embeddings
ORDER BY embedding <=> '[0.1, 0.2, 0.3, ...]'::vector
LIMIT 10;

-- 欧几里得距离搜索
SELECT document_id, chunk_id, embedding <-> '[0.1, 0.2, 0.3, ...]'::vector AS distance
FROM document_embeddings
ORDER BY embedding <-> '[0.1, 0.2, 0.3, ...]'::vector
LIMIT 10;
```

## 性能优化

### 索引类型

- **HNSW 索引**: 适用于高维向量的近似最近邻搜索
- **IVFFlat 索引**: 适用于较低维度的向量搜索

### 索引参数调优

```sql
-- HNSW 索引参数
CREATE INDEX ON document_embeddings USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- IVFFlat 索引参数
CREATE INDEX ON document_embeddings USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);
```

## 支持的向量维度

- OpenAI text-embedding-ada-002: 1536 维
- OpenAI text-embedding-3-small: 1536 维
- OpenAI text-embedding-3-large: 3072 维
- 其他模型: 根据具体模型调整

## 故障排除

### 常见问题

1. **扩展未安装**
   ```sql
   -- 检查扩展状态
   SELECT * FROM pg_extension WHERE extname = 'vector';
   ```

2. **向量维度不匹配**
   - 确保插入的向量维度与表定义一致
   - 检查嵌入模型的输出维度

3. **索引性能问题**
   - 调整索引参数
   - 考虑使用不同的距离函数

### 监控查询

```sql
-- 查看向量表统计信息
SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del 
FROM pg_stat_user_tables 
WHERE tablename = 'document_embeddings';

-- 查看索引使用情况
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes 
WHERE tablename = 'document_embeddings';
```

## 相关链接

- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [pgvector 文档](https://github.com/pgvector/pgvector#readme)
- [PostgreSQL 向量操作符](https://github.com/pgvector/pgvector#operators)