-- PostgreSQL 初始化脚本 - 启用 pgvector 扩展
-- 文件: 01-init-pgvector.sql
-- 作用: 在数据库初始化时自动创建 pgvector 扩展，支持向量存储和检索
-- 创建时间: 2024

-- 连接到默认数据库
\c graphrag;

-- 创建 pgvector 扩展（如果不存在）
-- pgvector 扩展提供向量数据类型和相关函数，支持向量相似度搜索
CREATE EXTENSION IF NOT EXISTS vector;

-- 验证扩展是否成功安装
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- 创建示例向量表结构（可选，用于测试）
-- 这个表将用于存储文档的向量嵌入
CREATE TABLE IF NOT EXISTS document_embeddings (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL,
    chunk_id INTEGER NOT NULL,
    embedding vector(1536),  -- OpenAI text-embedding-ada-002 的维度
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 为向量字段创建索引以提高查询性能
-- 使用 HNSW 索引进行高效的近似最近邻搜索
CREATE INDEX IF NOT EXISTS idx_document_embeddings_vector 
ON document_embeddings USING hnsw (embedding vector_cosine_ops);

-- 创建更新时间戳的触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 为 document_embeddings 表创建更新触发器
CREATE TRIGGER update_document_embeddings_updated_at 
    BEFORE UPDATE ON document_embeddings 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 输出成功信息
\echo 'pgvector extension initialized successfully!'
\echo 'Created document_embeddings table with vector index'