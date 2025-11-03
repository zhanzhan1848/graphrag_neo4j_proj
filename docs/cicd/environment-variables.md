# CI/CD 环境变量配置

## 概述

本文档描述了GraphRAG项目在CI/CD流程中如何处理环境变量，以及如何将GitHub Actions的环境变量转换为.env文件。

## 环境变量来源

### GitHub Actions 环境变量

项目使用以下GitHub Actions环境变量：

1. **Repository Variables (vars)**:
   - `AZURE_OPENAI_ENDPOINT`: Azure OpenAI服务端点

2. **Repository Secrets (secrets)**:
   - `AZURE_OPENAI_API_KEY`: Azure OpenAI API密钥
   - `GITHUB_TOKEN`: GitHub访问令牌（自动提供）

### 测试环境变量

在CI/CD流程中，还会设置以下测试专用的环境变量：

- `DATABASE_URL`: PostgreSQL测试数据库连接URL
- `NEO4J_URI`: Neo4j测试数据库连接URI
- `NEO4J_USER`: Neo4j用户名
- `NEO4J_PASSWORD`: Neo4j密码
- `REDIS_URL`: Redis连接URL
- `TESTING`: 测试模式标记

## .env文件生成

### 生成脚本

项目使用 `scripts/generate_env_file.sh` 脚本将环境变量写入.env文件。该脚本：

1. 读取GitHub Actions环境变量
2. 生成包含所有必要配置的.env文件
3. 确保敏感信息的安全处理

### 生成时机

.env文件在以下CI/CD作业中生成：

1. **unit-tests**: 单元测试前生成
2. **integration-tests**: 集成测试环境设置时生成
3. **performance-tests**: 性能测试前生成

### 生成的配置项

生成的.env文件包含：

```bash
# Azure OpenAI配置
AZURE_OPENAI_ENDPOINT=<从GitHub变量读取>
AZURE_OPENAI_API_KEY=<从GitHub密钥读取>
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# 模型配置
AZURE_OPENAI_LLM_MODEL=gpt-4-turbo
AZURE_OPENAI_LLM_DEPLOYMENT_NAME=gpt-4-turbo
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002

# 应用配置
APP_ENV=testing
DEBUG=true
LOG_LEVEL=DEBUG

# GraphRAG配置
GRAPHRAG_LLM_MAX_TOKENS=4000
GRAPHRAG_LLM_TEMPERATURE=0.1
GRAPHRAG_CHUNK_SIZE=1200
GRAPHRAG_CHUNK_OVERLAP=100
```

## CI/CD流程修改

### 修改内容

1. **添加.env文件生成步骤**:
   - 在需要环境变量的作业中添加生成步骤
   - 确保脚本有执行权限

2. **简化环境变量传递**:
   - 减少在每个步骤中重复设置环境变量
   - 通过.env文件统一管理配置

3. **保持安全性**:
   - 敏感信息仍通过GitHub Secrets传递
   - .env文件在CI/CD运行时动态生成

### 作业流程

```yaml
# 示例：单元测试作业
- name: Generate .env file
  env:
    AZURE_OPENAI_ENDPOINT: ${{ vars.AZURE_OPENAI_ENDPOINT }}
    AZURE_OPENAI_API_KEY: ${{ secrets.AZURE_OPENAI_API_KEY }}
  run: |
    chmod +x scripts/generate_env_file.sh
    ./scripts/generate_env_file.sh

- name: Run unit tests
  run: |
    uv run pytest tests/unit/ --cov=app
```

## 优势

1. **统一配置管理**: 所有配置通过.env文件统一管理
2. **减少重复**: 避免在多个步骤中重复设置相同的环境变量
3. **易于维护**: 配置变更只需修改生成脚本
4. **安全性**: 敏感信息通过GitHub Secrets安全传递
5. **兼容性**: 与Pydantic Settings的.env文件读取机制完全兼容

## 故障排除

### 常见问题

1. **环境变量未设置**:
   - 检查GitHub仓库的Variables和Secrets配置
   - 确认环境变量名称拼写正确

2. **脚本执行权限**:
   - 确保脚本有执行权限：`chmod +x scripts/generate_env_file.sh`

3. **.env文件未生成**:
   - 检查脚本执行日志
   - 确认所有必要的环境变量都已设置

### 调试方法

1. **查看生成的.env文件内容**:
   ```bash
   cat .env | grep -v "_KEY" | grep -v "_PASSWORD"
   ```

2. **检查环境变量**:
   ```bash
   echo "AZURE_OPENAI_ENDPOINT: $AZURE_OPENAI_ENDPOINT"
   echo "AZURE_OPENAI_API_KEY: [HIDDEN]"
   ```

## 相关文件

- `scripts/generate_env_file.sh`: 环境变量生成脚本
- `.github/workflows/ci.yml`: CI/CD流程配置
- `app/core/config.py`: 应用配置管理
- `.env.example`: 环境变量示例文件