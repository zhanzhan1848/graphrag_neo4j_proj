#!/bin/bash
# 脚本名称: generate_env_file.sh
# 描述: 将GitHub Actions环境变量写入.env文件
# 作者: GraphRAG Team
# 创建时间: 2024

set -e

# 输出信息函数
info() {
  echo "[INFO] $1"
}

warn() {
  echo "[WARN] $1"
}

error() {
  echo "[ERROR] $1"
  exit 1
}

# 检查是否在GitHub Actions环境中运行
if [ -z "$GITHUB_ACTIONS" ]; then
  warn "此脚本设计用于GitHub Actions环境中运行"
  warn "在本地环境中运行可能无法获取所有必要的环境变量"
fi

# 创建或清空.env文件
ENV_FILE=".env"
info "创建 $ENV_FILE 文件..."
> $ENV_FILE

# 添加文件头部注释
cat << EOF >> $ENV_FILE
# GraphRAG 知识库系统 - 环境变量配置文件
# 此文件由CI/CD流程自动生成
# 生成时间: $(date)
# 注意: 请勿手动修改此文件，它会在每次CI/CD运行时被重新生成

EOF

# 添加Azure OpenAI相关配置
info "添加Azure OpenAI配置..."
if [ -n "$AZURE_OPENAI_ENDPOINT" ]; then
  echo "AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT" >> $ENV_FILE
else
  warn "AZURE_OPENAI_ENDPOINT 环境变量未设置"
fi

if [ -n "$AZURE_OPENAI_API_KEY" ]; then
  echo "AZURE_OPENAI_API_KEY=$AZURE_OPENAI_API_KEY" >> $ENV_FILE
else
  warn "AZURE_OPENAI_API_KEY 环境变量未设置"
fi

# 添加Azure OpenAI API版本
echo "AZURE_OPENAI_API_VERSION=2024-02-15-preview" >> $ENV_FILE

# 添加Azure OpenAI模型配置
echo "AZURE_OPENAI_LLM_MODEL=gpt-4-turbo" >> $ENV_FILE
echo "AZURE_OPENAI_LLM_DEPLOYMENT_NAME=gpt-4-turbo" >> $ENV_FILE
echo "AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-ada-002" >> $ENV_FILE
echo "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002" >> $ENV_FILE

# 添加数据库配置（测试环境）
info "添加数据库配置..."
if [ -n "$DATABASE_URL" ]; then
  echo "DATABASE_URL=$DATABASE_URL" >> $ENV_FILE
fi

if [ -n "$NEO4J_URI" ]; then
  echo "NEO4J_URI=$NEO4J_URI" >> $ENV_FILE
fi

if [ -n "$NEO4J_USER" ]; then
  echo "NEO4J_USER=$NEO4J_USER" >> $ENV_FILE
fi

if [ -n "$NEO4J_PASSWORD" ]; then
  echo "NEO4J_PASSWORD=$NEO4J_PASSWORD" >> $ENV_FILE
fi

if [ -n "$REDIS_URL" ]; then
  echo "REDIS_URL=$REDIS_URL" >> $ENV_FILE
fi

# 添加测试标记
if [ -n "$TESTING" ]; then
  echo "TESTING=$TESTING" >> $ENV_FILE
fi

# 添加应用程序配置
echo "APP_ENV=testing" >> $ENV_FILE
echo "DEBUG=true" >> $ENV_FILE
echo "LOG_LEVEL=DEBUG" >> $ENV_FILE

# 添加GraphRAG特定配置
echo "GRAPHRAG_LLM_MAX_TOKENS=4000" >> $ENV_FILE
echo "GRAPHRAG_LLM_TEMPERATURE=0.1" >> $ENV_FILE
echo "GRAPHRAG_CHUNK_SIZE=1200" >> $ENV_FILE
echo "GRAPHRAG_CHUNK_OVERLAP=100" >> $ENV_FILE

info "$ENV_FILE 文件生成完成"
cat $ENV_FILE | grep -v "_KEY" | grep -v "_PASSWORD" # 显示生成的文件内容（不显示敏感信息）