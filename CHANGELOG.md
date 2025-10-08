# 变更日志

本文档记录了 GraphRAG 项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

# 变更日志

本文档记录了 GraphRAG 项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

### 修复
- 🔧 **Neo4j 会话管理** - 修复 AsyncSession 资源泄漏问题
  - 修复了 `GraphService` 中 12 个方法的会话管理问题
  - 将错误的 `async with` 语法替换为手动会话管理
  - 为所有会话调用添加了 `finally` 块确保正确关闭
  - 消除了 `ResourceWarning: Unclosed AsyncSession` 警告
  - 涉及的类和方法：
    - `GraphConnectionManager.health_check()`
    - `GraphNodeManager`: `create_node()`, `get_node()`, `update_node()`, `delete_node()`, `find_nodes()`
    - `GraphRelationshipManager`: `create_relationship()`, `get_relationship()`, `update_relationship()`, `delete_relationship()`, `find_relationships()`
    - `GraphService.execute_cypher()`

### 文档更新
- 📚 **故障排查指南** - 更新了服务连接问题修复文档
  - 添加了 Neo4j 会话资源泄漏问题的详细修复方案
  - 提供了会话管理的最佳实践和错误模式说明
  - 增加了资源泄漏监控和预防措施

### 新增
- 完整的项目文档体系
- 详细的安装和部署指南
- 贡献者指南和开发规范

### 变更
- 优化了文档结构和导航
- 改进了 README 的可读性

## [1.0.0] - 2024-12-19

### 新增
- 🎉 **首次发布** - GraphRAG 知识图谱系统正式发布
- 📄 **文档处理** - 支持 PDF、TXT、Markdown、HTML 等多种格式
- 🧠 **知识抽取** - 自动实体识别、关系抽取和引用识别
- 🔍 **智能检索** - 基于向量嵌入的语义检索
- 📊 **知识图谱** - Neo4j 图数据库存储和查询
- 🚀 **RESTful API** - 完整的 API 接口和文档
- 🐳 **容器化部署** - Docker Compose 一键部署
- 📈 **监控系统** - Prometheus + Grafana 监控方案
- 🔒 **安全机制** - JWT 认证和权限控制

### 技术栈
- **后端**: Python 3.9+ + FastAPI + Celery
- **前端**: React + TypeScript + Ant Design
- **数据库**: PostgreSQL + Neo4j + Redis + Weaviate
- **部署**: Docker + Docker Compose
- **监控**: Prometheus + Grafana + ELK Stack

## [0.3.0] - 2024-12-15

### 新增
- 🔍 **语义检索** - 集成 Weaviate 向量数据库
- 📊 **监控面板** - Grafana 仪表板和告警
- 🔒 **安全增强** - API 密钥认证和权限管理
- 📝 **API 文档** - Swagger/OpenAPI 自动生成文档

### 改进
- ⚡ **性能优化** - 查询响应时间提升 50%
- 🛠️ **错误处理** - 更完善的异常处理机制
- 📋 **日志系统** - 结构化日志和日志聚合

### 修复
- 🐛 修复文档上传时的内存泄漏问题
- 🐛 解决 Neo4j 连接池配置问题
- 🐛 修复前端路由跳转异常

## [0.2.0] - 2024-12-10

### 新增
- 🧠 **知识抽取** - 实体识别和关系抽取功能
- 📊 **图谱构建** - Neo4j 知识图谱存储
- 🔄 **异步处理** - Celery 任务队列
- 💾 **数据持久化** - PostgreSQL 元数据存储

### 改进
- 📄 **文档解析** - 支持更多文档格式
- 🎨 **用户界面** - React 前端界面优化
- 🔧 **配置管理** - 环境变量配置

### 修复
- 🐛 修复 PDF 解析中文乱码问题
- 🐛 解决 Docker 容器启动顺序问题

## [0.1.0] - 2024-12-05

### 新增
- 🏗️ **基础架构** - 项目初始化和基础框架
- 📄 **文档上传** - 基本的文档上传功能
- 🐳 **Docker 支持** - 容器化部署配置
- 🔧 **开发环境** - 开发工具和配置

### 技术债务
- 需要添加更多测试用例
- 需要完善错误处理机制
- 需要优化数据库查询性能

---

## 版本说明

### 版本号格式

我们使用 [语义化版本](https://semver.org/lang/zh-CN/) 格式：`主版本号.次版本号.修订号`

- **主版本号**: 不兼容的 API 修改
- **次版本号**: 向下兼容的功能性新增
- **修订号**: 向下兼容的问题修正

### 变更类型

- **新增** (`Added`) - 新功能
- **变更** (`Changed`) - 对现有功能的变更
- **弃用** (`Deprecated`) - 即将移除的功能
- **移除** (`Removed`) - 已移除的功能
- **修复** (`Fixed`) - 问题修复
- **安全** (`Security`) - 安全相关的修复

### 发布周期

- **主版本**: 每年 1-2 次，包含重大功能更新
- **次版本**: 每月 1-2 次，包含新功能和改进
- **修订版本**: 根据需要发布，主要是 bug 修复

### 支持政策

- **当前版本**: 完全支持，包含新功能和 bug 修复
- **前一个主版本**: 仅提供安全更新和关键 bug 修复
- **更早版本**: 不再提供官方支持

---

## 迁移指南

### 从 0.x 升级到 1.0

#### 重大变更

1. **API 端点变更**
   ```bash
   # 旧版本
   POST /upload
   GET /documents
   
   # 新版本
   POST /api/documents/upload
   GET /api/documents/
   ```

2. **配置文件格式**
   ```yaml
   # 旧版本 (config.yaml)
   database:
     host: localhost
     port: 5432
   
   # 新版本 (.env)
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   ```

3. **Docker Compose 变更**
   ```bash
   # 更新 docker-compose.yml
   docker-compose down
   git pull origin main
   docker-compose up -d
   ```

#### 迁移步骤

1. **备份数据**
   ```bash
   # 备份 PostgreSQL
   docker-compose exec postgres pg_dump -U graphrag graphrag > backup.sql
   
   # 备份 Neo4j
   docker-compose exec neo4j neo4j-admin dump --database=neo4j --to=/backups/neo4j.dump
   ```

2. **更新配置**
   ```bash
   # 复制新的配置模板
   cp .env.example .env
   # 根据旧配置更新 .env 文件
   ```

3. **重新部署**
   ```bash
   docker-compose down
   docker-compose pull
   docker-compose up -d
   ```

4. **验证升级**
   ```bash
   # 检查服务状态
   curl http://localhost:8000/health
   
   # 验证 API 功能
   curl http://localhost:8000/api/documents/
   ```

---

## 贡献者

感谢所有为 GraphRAG 项目做出贡献的开发者：

### 核心团队
- [@maintainer1](https://github.com/maintainer1) - 项目负责人
- [@maintainer2](https://github.com/maintainer2) - 技术负责人
- [@maintainer3](https://github.com/maintainer3) - 文档负责人

### 贡献者
- [@contributor1](https://github.com/contributor1) - 知识抽取模块
- [@contributor2](https://github.com/contributor2) - 前端界面开发
- [@contributor3](https://github.com/contributor3) - 部署脚本优化
- [@contributor4](https://github.com/contributor4) - 文档翻译

### 特别感谢
- 所有提交 Issue 和 PR 的社区成员
- 参与测试和反馈的早期用户
- 提供技术建议和支持的专家们

---

## 路线图

### 短期目标 (3个月)

- [ ] **多模态支持** - 图像 OCR 和多媒体处理
- [ ] **高级查询** - 复杂图查询和推理
- [ ] **性能优化** - 查询性能和并发处理
- [ ] **用户界面** - 图谱可视化和交互

### 中期目标 (6个月)

- [ ] **企业功能** - 多租户和权限管理
- [ ] **集成能力** - 第三方系统集成
- [ ] **AI 增强** - 更智能的知识抽取
- [ ] **移动支持** - 移动端应用

### 长期目标 (12个月)

- [ ] **云原生** - Kubernetes 和微服务
- [ ] **国际化** - 多语言支持
- [ ] **生态系统** - 插件和扩展机制
- [ ] **商业版本** - 企业级功能和支持

---

**持续更新中...** 📝

如有问题或建议，请访问我们的 [GitHub Issues](https://github.com/your-org/GraphRAG_NEO_IMG/issues)。