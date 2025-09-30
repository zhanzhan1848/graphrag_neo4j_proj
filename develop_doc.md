1 概览（要点）
本地部署：使用 docker-compose 编排：服务包括 FastAPI 后端、Postgres（带 pgvector）、Neo4j、Weaviate（向量/多模态存储，可选）、Redis（队列）、MinIO（对象存储，保存原始文件/图片）、可选后台 worker（Celery/RQ）和管理工具。
数据分层：Postgres 负责文档/chunk/元数据与事务；Neo4j 负责实体/关系图；Weaviate（或 pgvector）负责向量检索（文本/图像 embedding）；MinIO 存二进制大文件。
接口职责：FastAPI 提供 CRUD（文档/实体/关系/chunk），导入任务写入 Postgres 并触发 worker 做解析/嵌入/抽取；Neo4j 在抽取阶段写入（MERGE & provenance）；查询接口仅做读（或者写权限通过 RBAC 控制）。
关键保证：新增知识自动关联旧知识 → 使用 entity canonicalization + embedding 相似度查找 + fuzzy-match （合并或建立 new link）+ 人工/阈值策略。
2 Docker Compose（本地一键运行示例）
下面是一个可直接作为起点的 docker-compose.yml（示例），以及一个用于 Postgres 安装 pgvector 的 Dockerfile 的提示。把这两文件放在项目根目录（./compose）：
docker-compose.yml
version: "3.8"
services:
  postgres:
    build: ./postgres
    image: myproject-postgres:15
    environment:
      POSTGRES_USER: rguser
      POSTGRES_PASSWORD: rgpass
      POSTGRES_DB: graphrag
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  neo4j:
    image: neo4j:5.8
    environment:
      NEO4J_AUTH: neo4j/neo4jpass
      NEO4J_dbms_memory_pagecache_size: 1G
      NEO4J_dbms_connectors_default_listen_address: 0.0.0.0
    volumes:
      - neo4jdata:/data
    ports:
      - "7474:7474"
      - "7687:7687"

  weaviate:
    image: semitechnologies/weaviate:latest
    environment:
      QUERY_DEFAULTS_LIMIT: 20
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: "true"
      PERSISTENCE_DATA_PATH: /var/lib/weaviate
      DEFAULT_VECTORIZER_MODULE: "none"
    ports:
      - "8080:8080"
    volumes:
      - weaviate_data:/var/lib/weaviate

  minio:
    image: minio/minio
    command: server /data
    environment:
      MINIO_ROOT_USER: minio
      MINIO_ROOT_PASSWORD: minio123
    ports:
      - "9000:9000"
    volumes:
      - minio_data:/data

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  backend:
    build: ./app
    depends_on:
      - postgres
      - neo4j
      - weaviate
      - minio
      - redis
    environment:
      DATABASE_URL: postgresql://rguser:rgpass@postgres:5432/graphrag
      NEO4J_URI: bolt://neo4j:7687
      NEO4J_USER: neo4j
      NEO4J_PASSWORD: neo4jpass
      MINIO_ENDPOINT: minio:9000
      REDIS_URL: redis://redis:6379/0
      WEAVIATE_URL: http://weaviate:8080
    ports:
      - "8000:8000"
    volumes:
      - ./app:/app

volumes:
  pgdata:
  neo4jdata:
  weaviate_data:
  minio_data:
说明：
weaviate 在本地可作为多模态向量 DB（我们这里把 vectorizer 设为 none，意味着你会在外部生成 embeddings，然后导入 Weaviate）。也可以替换为 Milvus/Redis/pgvector 等。
postgres 我建议 build 一个带 pgvector 的镜像（下文给出 Dockerfile 说明）。Neo4j 默认端口开放做本地开发。
backend 指向你的 FastAPI app，worker 可与 backend 同镜像但不同命令。
Postgres Dockerfile（./postgres/Dockerfile）示例（伪）：
FROM postgres:15
RUN apt-get update && apt-get install -y build-essential git \
  && apt-get install -y postgresql-server-dev-15 \
  && git clone https://github.com/pgvector/pgvector.git /pgvector \
  && cd /pgvector && make && make install \
  && rm -rf /pgvector && apt-get clean
（注意：在企业环境可用官方镜像或 prebuilt pgvector 镜像以简化构建）
3 数据库 Schema（Postgres） & 索引（适合 CRUD + 向量检索）
主要表：documents, chunks, entities, relations, entity_aliases, ingest_jobs
SQL 示例（Postgres + pgvector）：
-- extension (执行一次)
CREATE EXTENSION IF NOT EXISTS "pgvector";

CREATE TABLE documents (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  title text,
  source text,
  authors jsonb,
  doc_type text,
  metadata jsonb,
  uploaded_by text,
  created_at timestamptz DEFAULT now()
);

CREATE TABLE chunks (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id uuid REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index int,
  text text,
  token_count int,
  embedding vector(1536), -- 根据 embedding 模型维度调整
  metadata jsonb,
  created_at timestamptz DEFAULT now(),
  UNIQUE (document_id, chunk_index)
);

-- 实体表（轻量，图存主结构在 Neo4j）
CREATE TABLE entities (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  canonical text, -- canonical name
  display text,
  types text[],   -- ['Method','Metric',...]
  alias_count int DEFAULT 0,
  metadata jsonb,
  created_at timestamptz DEFAULT now(),
  UNIQUE (canonical)
);

CREATE TABLE entity_aliases (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  entity_id uuid REFERENCES entities(id) ON DELETE CASCADE,
  alias text,
  source text
);

CREATE TABLE relations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  subject_entity uuid REFERENCES entities(id),
  object_entity uuid REFERENCES entities(id),
  predicate text,
  confidence float,
  evidence jsonb, -- [{doc_id, chunk_id, excerpt, score}]
  created_at timestamptz DEFAULT now()
);

-- 提高检索性能
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_chunks_text ON chunks USING gin (to_tsvector('english', text));
CREATE INDEX idx_entities_canonical ON entities (canonical);
备注：
chunks.embedding 的维度依赖你所选 embedding 模型（在示例用 1536）。如果你使用 Weaviate 存 embedding，则可以省掉 embedding 字段或保留备份。
relations.evidence 用 JSON 存多条证据，便于 CRUD 与审计。
4 Neo4j 图模型与索引（用于图查询/结构化关系）
节点 Label：Document, Chunk（或 Evidence），Entity, Claim
关系 type：MENTIONS, RELATED_TO, SUPPORTS, CITES, WROTE, HAS_CHUNK
推荐 Index / Constraints：
CREATE CONSTRAINT entity_canonical_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.canonical IS UNIQUE;
CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.id);
CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.id);
示例合并（upsert）实体与关系（Cypher）：
// upsert entity
MERGE (e:Entity {canonical: $canonical})
ON CREATE SET e.display = $display, e.types = $types, e.created_at = datetime()
RETURN e

// link chunk -> entity
MATCH (c:Chunk {id:$chunk_id}), (e:Entity {canonical:$canonical})
MERGE (c)-[r:MENTIONS]->(e)
ON CREATE SET r.confidence = $conf, r.excerpt = $excerpt, r.source_doc = $doc_id
RETURN r

// create relationship between entities with provenance
MATCH (a:Entity {canonical:$a}), (b:Entity {canonical:$b})
MERGE (a)-[rel:RELATED_TO {predicate:$predicate}]->(b)
SET rel.confidence = coalesce(rel.confidence, 0) + $conf,
    rel.sources = coalesce(rel.sources, []) + [$evidence]
RETURN rel
要点：sources 字段可以累计多个证据（chunk id + doc id + excerpt + timestamp）。
5 保证“新增知识能关联旧知识”的策略（核心）
这个需求是核心中的核心。方案由多个层级组成（结合规则 + 向量 + 人工回退）：
实体归一化（Canonicalization）
文本 normalize（小写、去标点、全角半角转换、数字归一）。
使用 alias 表维护手工/自动别名映射（例如“ResNet-50”↔“resnet50”）。
新实体尝试与 entities.canonical 精确匹配或 alias 表匹配。
向量相似度比对（Embedding-based linking）
对新实体或候选名称生成 embedding，查询 Weaviate / pgvector top-K（例如 top-5）。
若 top1 相似度 > T_high（比如 0.92），直接认为是相同实体，自动合并（upsert）。
若 T_low < top1 <= T_high（例如 0.75 - 0.92），标记为疑似合并，产生一个 pending link（待人工审核或自动使用 business rule 合并）。
若 top1 <= T_low，认为是新实体，创建新节点。
上下文+图信号增强
如果新实体与现有某个实体在多个 chunk 中共同被提到（co-mention），或有相同的引用（citation overlap），增加合并置信度。
使用 Neo4j 做邻域查询（如果新实体的 neighbor pattern 与一个已知实体的 neighbor pattern高度重合，也倾向合并）。
业务规则 & 人工回退
对高风险 predicate（例如“contradicts”/“negates”），不自动合并，必须人工确认。
提供管理界面（或 API）以查看 pending link 并批准/拒绝。
事务 & 并发
Upsert 在 Postgres + Neo4j 两端采用事务化策略：先在 Postgres 写证据/metadata，再在 Neo4j MERGE。使用唯一 ID（UUID）跨 DB 关联。
对并发 ingest，使用乐观锁（例如 PostgreSQL updated_at + compare-and-swap）或队列串行化同一文档任务。
6 FastAPI 路由设计（CRUD 为主）与样板代码
主要路由（示例）：
POST /api/v1/documents 上传文档（返回 doc_id + job_id）
GET /api/v1/documents/{doc_id}
DELETE /api/v1/documents/{doc_id}
POST /api/v1/entities 创建/Upsert 实体（body 允许 auto_link=true）
GET /api/v1/entities/{entity_id} / GET /api/v1/entities?canonical=...
POST /api/v1/relations 创建关系（支持 evidence 参数）
GET /api/v1/relations/{id}
POST /api/v1/ingest/{job_id}/confirm（人工确认 pending 执行）
GET /api/v1/provenance/{entity_or_relation_id} 返回证据链
FastAPI 简要样板（关键部分）：
from fastapi import FastAPI, UploadFile, BackgroundTasks
from pydantic import BaseModel
import uuid
app = FastAPI()

class UpsertEntity(BaseModel):
    canonical: str
    display: str | None = None
    types: list[str] = []
    auto_link: bool = True

@app.post("/api/v1/entities")
async def upsert_entity(payload: UpsertEntity):
    # 1. normalize canonical
    canonical = normalize_name(payload.canonical)
    # 2. try exact upsert in Postgres (sqlalchemy)
    entity = upsert_entity_db(canonical, payload.display, payload.types)
    # 3. if auto_link True -> call linking pipeline (embedding similarity -> merge)
    if payload.auto_link:
        link_result = await linking_pipeline(entity)
    return {"entity": entity, "link": link_result}
linking_pipeline（伪）会：
产生 embedding（调用本地 embed server 或外部）
查询 Weaviate/pgvector
根据阈值决定 merge/mark-pending/create-new
如果 merge -> 调用 Neo4j MERGE Cypher，将 evidence + metadata 写入 relations/nodes
7 证据（Provenance）设计（必须）
每个实体、关系都必须可回溯到原始文档 chunk。
数据设计：
在 Postgres relations.evidence 存 {doc_id, chunk_id, excerpt, score, extractor, time}
在 Neo4j rel.sources 存数组（相同信息）或建立 (:Claim)-[:SUPPORTED_BY]->(:Chunk) 的真实节点关系
API 返回时始终带上证据列表，LLM 可通过这些证据直接展示页码、段落
8 图片 / 多模态的后续扩展（Weaviate, DINO, DEIMv2, VLM 等）
规划分两层：数据存储与处理管线、检索/关联策略。
A. 存储与处理
原始图片存 MinIO（或本地 FS），保存 object_url、metadata（拍摄时间、alt text、OCR 文本等）到 Postgres documents/chunks（或者单独images表）。
图片预处理 pipeline：
图像特征提取（DINO / DEIMv2 / VLM 等）生成视觉嵌入。
可选：运行 OCR -> 产生 text chunk（便于与文本结合）。
将视觉 embedding 导入 Weaviate（每个对象带 modality: "image" 标签与 metadata）。
Weaviate 支持多模态 schema（vector + metadata），并能做 cross-modal search（text query -> image result）——设置好 class/props 即可。
B. 图像与现有知识图的关联
将图片所检测到的对象（objects）或场景描述作为实体（Entity）写入系统，例如 Entity{canonical:"Cat", types:["Object"]}，并关联到图片 chunk 作为 evidence。
使用 cross-modal retrieval：文本查询生成 text-embedding -> weaviate cross-modal -> 返回 top images -> 再把 image 所对应的 entities 连到 Neo4j。
对于专业领域（医疗/工程），需要定制视觉模型和抽取器（DEIMv2 等）。
C. 扩展架构注意：
为不同模态维护独立的向量 namespace/class（e.g., TextChunk, ImageObject），但对外提供统一检索 API（后端做融合/排序）。
存储每种模态的 extractor/version 以便回溯与 model-upgrade 重跑。
9 同步/一致性 & 迁移策略
双写策略：当 worker 完成抽取时同时写 Postgres（chunks/evidence）与 Neo4j（实体/关系）。若写 Neo4j 失败，标记该 job 为 incomplete 并入 retry 队列（幂等）。
幂等性：所有写操作使用 document_id + chunk_index + extractor_version 做幂等 key。
重跑策略：当你换 embedding/model，只需把 embedding/extractor_version 更新并提供重跑脚本去重建 Weaviate / Neo4j（使用 batch workers）。
数据迁移：提供 migration scripts（SQL + Cypher）。所有变更需版本化（schema_migrations 表）。
10 权限/安全/备份/运维
Auth：API Key 或 OAuth2，结合 FastAPI 的 Depends 实现 RBAC。
审计：记录 who did what（上传/approve/delete），写入 audit_log 表。
备份：
Postgres：定期 pg_dump 到物理卷或 S3（MinIO）。
Neo4j：使用内置导出或 snapshot。
Weaviate 数据：备份 data dir 或导出格式。
MinIO 对象保留策略/生命周期。
监控：Prometheus + Grafana（容器监控、队列深度、ingest 成功率）。
证书/加密：在生产环境使用 TLS（nginx 反向代理），以及为敏感数据做加密 at-rest（DB-level 或 disk).
11 测试、验证与质量控制
单元测试：FastAPI 路由 + DB mock（pytest）。
集成测试：使用 docker-compose 启动最小 stack（postgres + neo4j + backend），跑 end-to-end ingest -> graph creation -> query。
数据质量验证：
抽取精度统计（precision/recall）对标注数据集。
定期运行“duplicate detection”脚本检出疑似重复实体/关系。
性能测试：并发 ingest + 查询压力测试（locust 或 k6）。
12 示例工作流（从上传到关系构建）
用户 POST /api/v1/documents 上传 PDF（或提供 URL） → 返回 doc_id, job_id。
后端存文件到 MinIO，写 documents 表，入队 ingest job。
Worker:
提取文本（或 OCR） -> 分 chunk -> 存 chunks。
调用 embedding 服务（text embedding） -> 存 Weaviate 或 chunks.embedding。
对每个 chunk 调用 extractor（NER + RE 或 LLM-based extractor） -> 产生实体/关系候选。
对实体候选执行 linking_pipeline（embedding similarity + alias match + threshold） -> merge/create entity 并写入 Postgres & Neo4j。
将关系写入 Postgres.relations 并建立 Neo4j edges，附 evidence。
完成后 job status -> completed。若有 pending links，通知管理员或进入 review UI。
13 代码/脚本样板（更多细节）
我把关键代码骨架放在下面，供复制粘贴与快速启动。
A. Postgres upsert entity（SQL）
INSERT INTO entities (canonical, display, types, metadata)
VALUES ($1, $2, $3, $4)
ON CONFLICT (canonical) DO UPDATE
  SET display = EXCLUDED.display,
      types = array(SELECT DISTINCT unnest(coalesce(entities.types, '{}') || EXCLUDED.types)),
      metadata = jsonb_strip_nulls(entities.metadata || EXCLUDED.metadata)
RETURNING *;
B. Neo4j upsert relation（python-driver）
from neo4j import GraphDatabase
def upsert_relation(tx, a, b, predicate, evidence, conf):
    query = """
    MERGE (x:Entity {canonical:$a})
    MERGE (y:Entity {canonical:$b})
    MERGE (x)-[r:RELATED_TO {predicate:$predicate}]->(y)
    SET r.confidence = coalesce(r.confidence, 0) + $conf,
        r.sources = coalesce(r.sources, []) + [$evidence],
        r.updated_at = datetime()
    RETURN r
    """
    tx.run(query, a=a, b=b, predicate=predicate, evidence=evidence, conf=conf)
14 运维 & 扩展路线（时间线/建议功能）
阶段 0（今天就能跑）：用上面 compose 启动，搭建 FastAPI skeleton，支持文本 PDF ingest -> chunk -> store -> 手动创建 entity/relations（无自动抽取）。
阶段 1（MVP）：加自动 chunking + embedding（外部 embedding 服务）+ LLM-based extractor（外部调用）+ Neo4j 写入 + 自动 linking（阈值）。
阶段 2（多模态）：引入 image pipeline（DINO/DEIMv2/VLM），Weaviate multi-class schema，cross-modal retrieval。
阶段 3（规模化/企业）：迁移向量 DB（Milvus/Pinecone）、Neo4j 集群、K8s 部署 + CI/CD + 完整监控、RBAC、审计扩展。
15 风险清单（与缓解）
数据不一致（Postgres/Neo4j）：使用幂等 key + retry + periodic consistency checker（脚本对齐两端）。
自动合并导致误合并：采用保守阈值 + pending 阶段 + 人工审批。
向量/模型漂移：强制记录 extractor/model version，提供一键重跑历史数据的工具。
16 交付包建议（我可以现在直接生成）
如果你希望我把本设计转成可直接运行的 PoC 项目结构（包含：
docker-compose.yml（上面的已可用）、
一个最小 app/ FastAPI 实现（包含上面的 CRUD 路由与 worker skeleton）、
postgres/Dockerfile（安装 pgvector）、
初始化 SQL 脚本（创建表 + constraints）、
一个简单的 ingest worker（同步模式，方便 PoC）、
README（如何运行与测试）），