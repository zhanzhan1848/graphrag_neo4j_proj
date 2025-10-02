#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Azure OpenAI 服务模块
==================

本模块提供 Azure OpenAI 服务的统一接口，支持：
1. LLM 文本生成和对话
2. 文本嵌入生成
3. GraphRAG 集成支持
4. 错误处理和重试机制

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletion
from openai.types import CreateEmbeddingResponse
import tiktoken

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class AzureOpenAIService:
    """
    Azure OpenAI 服务类
    
    提供统一的 Azure OpenAI 接口，包括 LLM 和嵌入服务。
    支持异步操作和错误处理。
    """
    
    def __init__(self):
        """初始化 Azure OpenAI 服务"""
        self.client = None
        self.encoding = None
        self._initialize_client()
        self._initialize_tokenizer()
    
    def _initialize_client(self):
        """初始化 Azure OpenAI 客户端"""
        try:
            if not settings.AZURE_OPENAI_ENDPOINT or not settings.AZURE_OPENAI_API_KEY:
                raise ValueError("Azure OpenAI 配置不完整，请检查 AZURE_OPENAI_ENDPOINT 和 AZURE_OPENAI_API_KEY")
            
            self.client = AsyncAzureOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION
            )
            logger.info("Azure OpenAI 客户端初始化成功")
            
        except Exception as e:
            logger.error(f"Azure OpenAI 客户端初始化失败: {e}")
            raise
    
    def _initialize_tokenizer(self):
        """初始化分词器"""
        try:
            # 使用 tiktoken 进行分词
            self.encoding = tiktoken.encoding_for_model("gpt-4")
            logger.info("分词器初始化成功")
        except Exception as e:
            logger.warning(f"分词器初始化失败，使用默认配置: {e}")
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """
        计算文本的令牌数量
        
        Args:
            text: 输入文本
            
        Returns:
            令牌数量
        """
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.warning(f"令牌计算失败: {e}")
            # 简单估算：1个令牌约等于4个字符
            return len(text) // 4
    
    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> ChatCompletion:
        """
        生成聊天完成
        
        Args:
            messages: 对话消息列表
            model: 模型名称（可选，默认使用配置中的模型）
            max_tokens: 最大令牌数（可选）
            temperature: 温度参数（可选）
            **kwargs: 其他参数
            
        Returns:
            聊天完成响应
        """
        try:
            # 使用配置中的默认值
            model = model or settings.AZURE_OPENAI_LLM_DEPLOYMENT_NAME
            max_tokens = max_tokens or settings.GRAPHRAG_LLM_MAX_TOKENS
            temperature = temperature or settings.GRAPHRAG_LLM_TEMPERATURE
            
            logger.debug(f"发送聊天请求: model={model}, max_tokens={max_tokens}, temperature={temperature}")
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            logger.debug(f"聊天请求成功，使用令牌: {response.usage.total_tokens}")
            return response
            
        except Exception as e:
            logger.error(f"聊天完成请求失败: {e}")
            raise
    
    async def generate_text(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        生成文本（简化接口）
        
        Args:
            prompt: 用户提示
            system_message: 系统消息（可选）
            **kwargs: 其他参数
            
        Returns:
            生成的文本
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        response = await self.generate_completion(messages, **kwargs)
        return response.choices[0].message.content
    
    async def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """
        生成文本嵌入
        
        Args:
            texts: 文本或文本列表
            model: 嵌入模型名称（可选）
            
        Returns:
            嵌入向量列表
        """
        try:
            # 确保输入是列表
            if isinstance(texts, str):
                texts = [texts]
            
            model = model or settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
            
            logger.debug(f"生成嵌入: model={model}, texts_count={len(texts)}")
            
            response = await self.client.embeddings.create(
                model=model,
                input=texts
            )
            
            embeddings = [data.embedding for data in response.data]
            logger.debug(f"嵌入生成成功，向量数量: {len(embeddings)}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"嵌入生成失败: {e}")
            raise
    
    async def generate_single_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        生成单个文本的嵌入（便捷方法）
        
        Args:
            text: 输入文本
            model: 嵌入模型名称（可选）
            
        Returns:
            嵌入向量
        """
        embeddings = await self.generate_embeddings([text], model)
        return embeddings[0]
    
    async def batch_generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 100,
        model: Optional[str] = None
    ) -> List[List[float]]:
        """
        批量生成嵌入（支持大量文本）
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            model: 嵌入模型名称（可选）
            
        Returns:
            嵌入向量列表
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = await self.generate_embeddings(batch_texts, model)
            all_embeddings.extend(batch_embeddings)
            
            # 添加延迟以避免速率限制
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        return all_embeddings
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型配置信息
        
        Returns:
            模型配置字典
        """
        return {
            "llm_model": settings.AZURE_OPENAI_LLM_MODEL,
            "llm_deployment": settings.AZURE_OPENAI_LLM_DEPLOYMENT_NAME,
            "embedding_model": settings.AZURE_OPENAI_EMBEDDING_MODEL,
            "embedding_deployment": settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
            "api_version": settings.AZURE_OPENAI_API_VERSION,
            "max_tokens": settings.GRAPHRAG_LLM_MAX_TOKENS,
            "temperature": settings.GRAPHRAG_LLM_TEMPERATURE
        }
    
    async def health_check(self) -> bool:
        """
        健康检查
        
        Returns:
            服务是否正常
        """
        try:
            # 发送简单的测试请求
            await self.generate_text("Hello", max_tokens=1)
            return True
        except Exception as e:
            logger.error(f"Azure OpenAI 健康检查失败: {e}")
            return False


# 全局服务实例
azure_openai_service = AzureOpenAIService()


async def get_azure_openai_service() -> AzureOpenAIService:
    """
    获取 Azure OpenAI 服务实例（依赖注入）
    
    Returns:
        Azure OpenAI 服务实例
    """
    return azure_openai_service