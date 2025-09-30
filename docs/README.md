# GraphRAG系统文档中心

欢迎来到GraphRAG知识图谱系统的文档中心！本文档体系为您提供了完整的系统使用、开发和维护指南。

## 📚 文档结构

### 🚀 快速开始
- **[快速开始指南](./quick-start/README.md)** - 系统安装、配置和基本使用
- **[用户指南](./user-guide/README.md)** - 详细的功能使用说明和最佳实践

### 🔧 开发文档
- **[开发者指南](./development/README.md)** - 完整的开发环境搭建和代码开发指南
- **[API文档](./api/README.md)** - RESTful API接口详细说明和SDK示例

### 🏗️ 架构设计
- **[系统架构](./architecture/README.md)** - 整体架构设计和技术选型说明
- **[数据库设计](./database/README.md)** - PostgreSQL和Neo4j数据模型设计

### 🔒 安全与运维
- **[安全指南](./security/README.md)** - 系统安全配置和最佳实践
- **[部署指南](./deployment/README.md)** - Docker和Kubernetes部署配置
- **[监控指南](./monitoring/README.md)** - 系统监控、日志和告警配置

### 📊 性能与优化
- **[性能优化](./performance/README.md)** - 系统性能监控、分析和优化指南

## 🎯 文档特色

### 📖 全面覆盖
- **用户视角**: 从快速开始到高级功能的完整使用指南
- **开发视角**: 从环境搭建到代码贡献的完整开发流程
- **运维视角**: 从部署配置到监控维护的完整运维方案

### 🛠️ 实用性强
- **代码示例**: 每个功能都提供详细的代码示例
- **配置模板**: 提供生产就绪的配置文件模板
- **最佳实践**: 基于实际项目经验的最佳实践建议

### 🔄 持续更新
- **版本同步**: 文档与系统版本保持同步更新
- **社区贡献**: 欢迎社区贡献和反馈改进
- **问题跟踪**: 及时响应和解决文档相关问题

## 🚀 快速导航

### 新用户推荐路径
1. 📖 [快速开始](./quick-start/README.md) - 了解系统基本概念和快速上手
2. 👤 [用户指南](./user-guide/README.md) - 学习详细功能使用
3. 🔧 [API文档](./api/README.md) - 集成和自定义开发

### 开发者推荐路径
1. 🏗️ [系统架构](./architecture/README.md) - 理解系统整体设计
2. 🔧 [开发者指南](./development/README.md) - 搭建开发环境
3. 💾 [数据库设计](./database/README.md) - 了解数据模型
4. 📊 [性能优化](./performance/README.md) - 性能调优指南

### 运维人员推荐路径
1. 🚀 [部署指南](./deployment/README.md) - 系统部署配置
2. 🔒 [安全指南](./security/README.md) - 安全配置和防护
3. 📊 [监控指南](./monitoring/README.md) - 监控和告警设置

## 📋 系统概述

GraphRAG是一个基于知识图谱的检索增强生成系统，主要功能包括：

### 🔍 核心功能
- **文档管理**: 支持PDF、TXT、Markdown等多种格式文档的上传和管理
- **知识抽取**: 自动从文档中抽取实体、关系和断言
- **图谱构建**: 基于Neo4j构建知识图谱，支持复杂关系查询
- **语义检索**: 基于向量嵌入的语义相似度检索
- **智能问答**: 结合知识图谱和RAG技术的智能问答

### 🏗️ 技术架构
- **后端**: Python + FastAPI + Celery
- **前端**: React + TypeScript + Ant Design
- **数据库**: PostgreSQL + Neo4j + Redis
- **部署**: Docker + Kubernetes
- **监控**: Prometheus + Grafana + ELK Stack

### 🎯 应用场景
- **学术研究**: 论文知识管理和研究辅助
- **企业知识库**: 内部文档管理和知识检索
- **智能客服**: 基于知识图谱的智能问答
- **内容分析**: 大规模文档的知识发现

## 核心特性

- **多格式文档接入**：支持PDF/TXT/Markdown/HTML/图片（OCR）等格式
- **智能文本处理**：自动分块、去重与存储
- **向量嵌入生成**：支持文本和图像的语义检索
- **知识图谱构建**：自动实体抽取、关系抽取、引用识别
- **可追溯性保证**：每个关系/断言都能回溯到源文档位置
- **多模态支持**：文本、图像的统一检索和关联

## 技术架构

### 数据分层架构
- **Postgres**：文档/chunk/元数据与事务管理
- **Neo4j**：实体/关系图存储
- **Weaviate**：向量检索（文本/图像embedding）
- **MinIO**：对象存储（原始文件/图片）
- **Redis**：任务队列管理

### 服务组件
- **FastAPI后端**：提供RESTful API接口
- **后台Worker**：处理文档解析、嵌入生成、实体抽取
- **管理工具**：监控、管理界面

## 快速开始

### 环境要求
- Docker & Docker Compose
- Python 3.9+
- 至少8GB内存

### 一键启动
```bash
# 克隆项目
git clone <repository-url>
cd GraphRAG_NEO_IMG

# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps
```

### 访问地址
- **API文档**: http://localhost:8000/docs
- **Neo4j浏览器**: http://localhost:7474
- **MinIO控制台**: http://localhost:9000

## 开发阶段

本项目采用分阶段开发模式：

- **[阶段0 - 基础架构](./phase0/README.md)**: 基础服务搭建和API框架
- **[阶段1 - MVP功能](./phase1/README.md)**: 核心功能实现
- **[阶段2 - 多模态扩展](./phase2/README.md)**: 图像处理和多模态检索
- **[阶段3 - 企业级部署](./phase3/README.md)**: 生产环境部署和监控

## 文档结构

```
docs/
├── README.md                 # 项目总览
├── architecture/             # 架构设计文档
├── api/                     # API文档
├── deployment/              # 部署文档
├── development/             # 开发指南
├── phase0/                  # 阶段0文档
├── phase1/                  # 阶段1文档
├── phase2/                  # 阶段2文档
└── phase3/                  # 阶段3文档
```

## 贡献指南

1. 遵循项目规则和编码规范
2. 每次只完成一个功能点
3. 添加完整的函数级和文件级注释
4. 提交前运行测试用例

## 许可证

[MIT License](../LICENSE)