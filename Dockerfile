# GraphRAG API 服务 Docker 镜像
# 基于 Python 3.11 构建的轻量级镜像，使用 uv 进行快速依赖管理

# 使用官方 Python 3.11 slim 镜像作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache

# 安装系统依赖和 uv
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# 复制项目配置文件
COPY pyproject.toml .

# 使用 uv 安装依赖（比 pip 快 10-100 倍）
RUN uv pip install --system --no-cache -r pyproject.toml

# 复制应用代码
COPY app/ ./app/

# 创建非 root 用户
RUN useradd --create-home --shell /bin/bash appuser

# 创建必要的目录并设置权限
RUN mkdir -p /app/logs /app/data /app/uploads && \
    chown -R appuser:appuser /app

COPY .env.example .env

# 切换到非 root 用户
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]