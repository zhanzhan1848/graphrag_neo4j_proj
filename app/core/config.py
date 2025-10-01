#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 系统配置管理模块
=======================

本模块负责管理 GraphRAG 系统的所有配置项，包括：
1. 环境变量读取和验证
2. 数据库连接配置
3. 日志系统配置
4. API 服务配置
5. 外部服务配置

使用 Pydantic Settings 进行配置管理，支持从环境变量、.env 文件等多种方式读取配置。

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import os
from typing import List, Optional, Union
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """
    应用配置类
    
    使用 Pydantic BaseSettings 自动从环境变量读取配置，
    支持类型验证和默认值设置。
    """
    
    # ==================== 基础应用配置 ====================
    APP_NAME: str = Field(default="GraphRAG Knowledge Base API", description="应用名称")
    VERSION: str = Field(default="1.0.0", description="应用版本")
    DESCRIPTION: str = Field(
        default="GraphRAG 知识库系统 - 基于图数据库和向量检索的智能知识管理平台",
        description="应用描述"
    )
    ENVIRONMENT: str = Field(default="development", description="运行环境")
    DEBUG: bool = Field(default=True, description="调试模式")
    
    # ==================== 服务器配置 ====================
    HOST: str = Field(default="0.0.0.0", description="服务器监听地址")
    PORT: int = Field(default=8000, description="服务器监听端口")
    WORKERS: int = Field(default=1, description="工作进程数量")
    
    # ==================== API 配置 ====================
    API_V1_STR: str = Field(default="/api/v1", description="API v1 路径前缀")
    ALLOWED_HOSTS: List[str] = Field(
        default=["*"], 
        description="允许的跨域主机列表"
    )
    
    # ==================== 安全配置 ====================
    SECRET_KEY: str = Field(
        default="your-secret-key-change-in-production",
        description="应用密钥，生产环境必须修改"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="访问令牌过期时间（分钟）")
    
    # ==================== 数据库配置 ====================
    # PostgreSQL 配置
    POSTGRES_HOST: str = Field(default="localhost", description="PostgreSQL 主机地址")
    POSTGRES_PORT: int = Field(default=5432, description="PostgreSQL 端口")
    POSTGRES_USER: str = Field(default="graphrag", description="PostgreSQL 用户名")
    POSTGRES_PASSWORD: str = Field(default="graphrag123", description="PostgreSQL 密码")
    POSTGRES_DB: str = Field(default="graphrag", description="PostgreSQL 数据库名")
    
    # Neo4j 配置
    NEO4J_HOST: str = Field(default="localhost", description="Neo4j 主机地址")
    NEO4J_PORT: int = Field(default=7688, description="Neo4j Bolt 端口")
    NEO4J_USER: str = Field(default="neo4j", description="Neo4j 用户名")
    NEO4J_PASSWORD: str = Field(default="neo4j123", description="Neo4j 密码")
    NEO4J_DATABASE: str = Field(default="graphrag", description="Neo4j 数据库名")
    
    # Redis 配置
    REDIS_HOST: str = Field(default="localhost", description="Redis 主机地址")
    REDIS_PORT: int = Field(default=6379, description="Redis 端口")
    REDIS_PASSWORD: str = Field(default="redis123", description="Redis 密码")
    REDIS_DB: int = Field(default=0, description="Redis 数据库编号")
    
    # Weaviate 配置
    WEAVIATE_HOST: str = Field(default="localhost", description="Weaviate 主机地址")
    WEAVIATE_PORT: int = Field(default=8080, description="Weaviate 端口")
    WEAVIATE_GRPC_PORT: int = Field(default=50051, description="Weaviate gRPC 端口")
    
    # MinIO 配置
    MINIO_HOST: str = Field(default="localhost", description="MinIO 主机地址")
    MINIO_PORT: int = Field(default=9000, description="MinIO 端口")
    MINIO_ACCESS_KEY: str = Field(default="minioadmin", description="MinIO 访问密钥")
    MINIO_SECRET_KEY: str = Field(default="minioadmin123", description="MinIO 秘密密钥")
    MINIO_BUCKET: str = Field(default="graphrag", description="MinIO 存储桶名称")
    
    # ==================== 日志配置 ====================
    LOG_LEVEL: str = Field(default="INFO", description="日志级别")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="日志格式"
    )
    LOG_FILE_PATH: str = Field(default="/app/logs/app.log", description="日志文件路径")
    LOG_MAX_SIZE: int = Field(default=10 * 1024 * 1024, description="单个日志文件最大大小（字节）")
    LOG_BACKUP_COUNT: int = Field(default=5, description="日志文件备份数量")
    LOG_ROTATION: str = Field(default="midnight", description="日志轮转时间")
    
    # ==================== 文件处理配置 ====================
    UPLOAD_MAX_SIZE: int = Field(default=100 * 1024 * 1024, description="上传文件最大大小（字节）")
    ALLOWED_FILE_TYPES: List[str] = Field(
        default=[".pdf", ".txt", ".md", ".docx", ".html", ".json"],
        description="允许上传的文件类型"
    )
    TEMP_DIR: str = Field(default="temp", description="临时文件目录")
    
    # ==================== 处理配置 ====================
    CHUNK_SIZE: int = Field(default=1000, description="文本分块大小")
    CHUNK_OVERLAP: int = Field(default=200, description="文本分块重叠大小")
    MAX_CONCURRENT_TASKS: int = Field(default=5, description="最大并发任务数")
    
    # ==================== 外部服务配置 ====================
    # LLM 服务配置
    LLM_API_BASE: Optional[str] = Field(default=None, description="LLM API 基础地址")
    LLM_API_KEY: Optional[str] = Field(default=None, description="LLM API 密钥")
    LLM_MODEL: str = Field(default="gpt-3.5-turbo", description="LLM 模型名称")
    
    # 嵌入服务配置
    EMBEDDING_API_BASE: Optional[str] = Field(default=None, description="嵌入服务 API 基础地址")
    EMBEDDING_API_KEY: Optional[str] = Field(default=None, description="嵌入服务 API 密钥")
    EMBEDDING_MODEL: str = Field(default="text-embedding-ada-002", description="嵌入模型名称")
    EMBEDDING_DIMENSION: int = Field(default=1536, description="嵌入向量维度")
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        """验证运行环境配置"""
        allowed_envs = ["development", "testing", "staging", "production"]
        if v not in allowed_envs:
            raise ValueError(f"环境配置必须是以下之一: {allowed_envs}")
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """验证日志级别配置"""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"日志级别必须是以下之一: {allowed_levels}")
        return v.upper()
    
    @validator("ALLOWED_HOSTS", pre=True)
    def validate_allowed_hosts(cls, v):
        """验证允许的主机列表"""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v
    
    @validator("ALLOWED_FILE_TYPES", pre=True)
    def validate_allowed_file_types(cls, v):
        """验证允许的文件类型列表"""
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(",")]
        return v
    
    @property
    def postgres_url(self) -> str:
        """获取 PostgreSQL 连接 URL"""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )
    
    @property
    def neo4j_url(self) -> str:
        """获取 Neo4j 连接 URL"""
        return f"bolt://{self.NEO4J_HOST}:{self.NEO4J_PORT}"
    
    @property
    def redis_url(self) -> str:
        """获取 Redis 连接 URL"""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    @property
    def weaviate_url(self) -> str:
        """获取 Weaviate 连接 URL"""
        return f"http://{self.WEAVIATE_HOST}:{self.WEAVIATE_PORT}"
    
    @property
    def minio_url(self) -> str:
        """获取 MinIO 连接 URL"""
        return f"{self.MINIO_HOST}:{self.MINIO_PORT}"
    
    def ensure_directories(self):
        """确保必要的目录存在"""
        directories = [
            Path(self.LOG_FILE_PATH).parent,
            Path(self.TEMP_DIR),
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    class Config:
        """Pydantic 配置"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# 创建全局配置实例
settings = Settings()

# 确保必要的目录存在
settings.ensure_directories()