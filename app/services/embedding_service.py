#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 向量嵌入服务
===================

本模块实现了向量嵌入的核心功能。

服务功能：
- 文本向量化
- 批量嵌入生成
- 向量相似度计算
- 嵌入缓存管理
- 多模型支持
- 向量维度管理

支持的模型：
- OpenAI Embeddings
- Azure OpenAI Embeddings
- 本地嵌入模型
- Sentence Transformers

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import asyncio
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from app.services.azure_openai_service import get_azure_openai_service, AzureOpenAIService
import tiktoken

from app.core.config import settings
from app.utils.exceptions import (
    VectorError,
    VectorEmbeddingError,
    VectorDimensionError,
    ExternalServiceError,
    EmbeddingServiceError
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = settings


@dataclass
class EmbeddingResult:
    """嵌入结果"""
    text: str
    embedding: List[float]
    model: str
    dimensions: int
    token_count: int
    processing_time: float
    created_at: str
    text_hash: str


@dataclass
class BatchEmbeddingResult:
    """批量嵌入结果"""
    results: List[EmbeddingResult]
    total_texts: int
    successful_count: int
    failed_count: int
    total_tokens: int
    total_processing_time: float
    model: str
    batch_id: str


class EmbeddingCache:
    """
    嵌入缓存管理器
    
    缓存已生成的嵌入向量，避免重复计算。
    """
    
    def __init__(self, max_size: int = 10000, ttl_hours: int = 24):
        """
        初始化缓存
        
        Args:
            max_size: 最大缓存条目数
            ttl_hours: 缓存生存时间（小时）
        """
        self.max_size = max_size
        self.ttl_hours = ttl_hours
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, datetime] = {}
    
    def get(self, text: str, model: str) -> Optional[List[float]]:
        """
        获取缓存的嵌入
        
        Args:
            text: 文本
            model: 模型名称
            
        Returns:
            嵌入向量或None
        """
        cache_key = self._generate_cache_key(text, model)
        
        if cache_key in self.cache:
            # 检查是否过期
            if self._is_expired(cache_key):
                self._remove(cache_key)
                return None
            
            # 更新访问时间
            self.access_times[cache_key] = datetime.utcnow()
            return self.cache[cache_key]["embedding"]
        
        return None
    
    def put(self, text: str, model: str, embedding: List[float]) -> None:
        """
        存储嵌入到缓存
        
        Args:
            text: 文本
            model: 模型名称
            embedding: 嵌入向量
        """
        cache_key = self._generate_cache_key(text, model)
        
        # 如果缓存已满，移除最旧的条目
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[cache_key] = {
            "embedding": embedding,
            "created_at": datetime.utcnow(),
            "text_length": len(text)
        }
        self.access_times[cache_key] = datetime.utcnow()
    
    def _generate_cache_key(self, text: str, model: str) -> str:
        """生成缓存键"""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return f"{model}:{text_hash}"
    
    def _is_expired(self, cache_key: str) -> bool:
        """检查缓存是否过期"""
        if cache_key not in self.cache:
            return True
        
        created_at = self.cache[cache_key]["created_at"]
        expiry_time = created_at + timedelta(hours=self.ttl_hours)
        return datetime.utcnow() > expiry_time
    
    def _remove(self, cache_key: str) -> None:
        """移除缓存条目"""
        self.cache.pop(cache_key, None)
        self.access_times.pop(cache_key, None)
    
    def _evict_oldest(self) -> None:
        """移除最旧的缓存条目"""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove(oldest_key)
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": 0.0,  # 需要在使用中统计
            "total_memory": sum(
                len(item["embedding"]) * 4 + item["text_length"] 
                for item in self.cache.values()
            )
        }


class AzureOpenAIEmbeddingProvider:
    """
    Azure OpenAI 嵌入提供者
    
    使用 Azure OpenAI API 生成文本嵌入。
    """
    
    def __init__(
        self,
        azure_openai_service: Optional[AzureOpenAIService] = None,
        model: Optional[str] = None,
        max_tokens: int = 8192,
        batch_size: int = 100
    ):
        """
        初始化 Azure OpenAI 嵌入提供者
        
        Args:
            azure_openai_service: Azure OpenAI 服务
            model: 嵌入模型名称
            max_tokens: 最大令牌数
            batch_size: 批处理大小
        """
        # 直接使用传入的服务实例或创建新实例，避免异步调用
        if azure_openai_service is not None:
            self.azure_openai_service = azure_openai_service
        else:
            # 直接创建实例而不是调用异步函数
            from app.services.azure_openai_service import AzureOpenAIService
            self.azure_openai_service = AzureOpenAIService()
        
        self.model = model or settings.AZURE_OPENAI_EMBEDDING_MODEL
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        
        # 初始化分词器
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    async def embed_text(self, text: str) -> EmbeddingResult:
        """
        生成单个文本的嵌入
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入结果
            
        Raises:
            EmbeddingServiceError: 嵌入生成失败
        """
        try:
            start_time = datetime.utcnow()
            
            # 检查文本长度
            token_count = len(self.tokenizer.encode(text))
            if token_count > self.max_tokens:
                # 截断文本
                tokens = self.tokenizer.encode(text)[:self.max_tokens]
                text = self.tokenizer.decode(tokens)
                token_count = self.max_tokens
                logger.warning(f"文本被截断到 {self.max_tokens} 个令牌")
            
            # 调用 Azure OpenAI API
            embedding = await self.azure_openai_service.generate_single_embedding(text)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # 生成文本哈希
            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            
            return EmbeddingResult(
                text=text,
                embedding=embedding,
                model=self.model,
                dimensions=len(embedding),
                token_count=token_count,
                processing_time=processing_time,
                created_at=datetime.utcnow().isoformat(),
                text_hash=text_hash
            )
            
        except Exception as e:
            if "API" in str(e):
                logger.error(f"Azure OpenAI API 错误: {str(e)}")
                raise EmbeddingServiceError(f"Azure OpenAI API 错误: {str(e)}")
            else:
                logger.error(f"嵌入生成失败: {str(e)}")
                raise EmbeddingServiceError(f"嵌入生成失败: {str(e)}")
    
    async def embed_texts(self, texts: List[str]) -> BatchEmbeddingResult:
        """
        批量生成文本嵌入
        
        Args:
            texts: 文本列表
            
        Returns:
            批量嵌入结果
        """
        try:
            start_time = datetime.utcnow()
            batch_id = hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:8]
            
            results = []
            failed_count = 0
            total_tokens = 0
            
            # 分批处理
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                try:
                    # 处理批次
                    batch_results = await self._process_batch(batch_texts)
                    results.extend(batch_results)
                    total_tokens += sum(r.token_count for r in batch_results)
                    
                except Exception as e:
                    logger.error(f"批次处理失败: {str(e)}")
                    failed_count += len(batch_texts)
                    
                    # 尝试单独处理失败的文本
                    for text in batch_texts:
                        try:
                            result = await self.embed_text(text)
                            results.append(result)
                            total_tokens += result.token_count
                        except Exception:
                            failed_count += 1
            
            total_processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return BatchEmbeddingResult(
                results=results,
                total_texts=len(texts),
                successful_count=len(results),
                failed_count=failed_count,
                total_tokens=total_tokens,
                total_processing_time=total_processing_time,
                model=self.model,
                batch_id=batch_id
            )
            
        except Exception as e:
            logger.error(f"批量嵌入失败: {str(e)}")
            raise EmbeddingServiceError(f"批量嵌入失败: {str(e)}")
    
    async def _process_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """处理单个批次"""
        # 预处理文本
        processed_texts = []
        token_counts = []
        
        for text in texts:
            token_count = len(self.tokenizer.encode(text))
            if token_count > self.max_tokens:
                tokens = self.tokenizer.encode(text)[:self.max_tokens]
                text = self.tokenizer.decode(tokens)
                token_count = self.max_tokens
            
            processed_texts.append(text)
            token_counts.append(token_count)
        
        # 调用 Azure OpenAI API 批量生成嵌入
        embeddings = await self.azure_openai_service.batch_generate_embeddings(processed_texts)
        
        # 构建结果
        results = []
        for i, (text, embedding, token_count) in enumerate(zip(processed_texts, embeddings, token_counts)):
            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            
            result = EmbeddingResult(
                text=text,
                embedding=embedding,
                model=self.model,
                dimensions=len(embedding),
                token_count=token_count,
                processing_time=0.0,  # 批处理时无法准确计算单个文本的处理时间
                created_at=datetime.utcnow().isoformat(),
                text_hash=text_hash
            )
            results.append(result)
        
        return results


class EmbeddingService:
    """
    向量嵌入服务
    
    提供统一的文本向量化接口。
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "text-embedding-3-small",
        cache_enabled: bool = True,
        cache_size: int = 10000
    ):
        """
        初始化嵌入服务
        
        Args:
            provider: 嵌入提供者
            model: 嵌入模型
            cache_enabled: 是否启用缓存
            cache_size: 缓存大小
        """
        self.provider_name = provider
        self.model = model
        self.cache_enabled = cache_enabled
        
        # 初始化缓存
        if cache_enabled:
            self.cache = EmbeddingCache(max_size=cache_size)
        else:
            self.cache = None
        
        # 初始化提供者
        self.provider = self._create_provider(provider, model)
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_tokens": 0,
            "total_processing_time": 0.0
        }
    
    def _create_provider(self, provider: str, model: str):
        """创建嵌入提供者"""
        if provider.lower() == "openai" or provider.lower() == "azure_openai":
            return AzureOpenAIEmbeddingProvider(model=model)
        else:
            raise ValueError(f"不支持的嵌入提供者: {provider}")
    
    async def embed_text(self, text: str) -> List[float]:
        """
        生成文本嵌入
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
            
        Raises:
            VectorEmbeddingError: 嵌入生成失败
        """
        try:
            self.stats["total_requests"] += 1
            
            # 检查缓存
            if self.cache_enabled and self.cache:
                cached_embedding = self.cache.get(text, self.model)
                if cached_embedding is not None:
                    self.stats["cache_hits"] += 1
                    logger.debug("使用缓存的嵌入")
                    return cached_embedding
                else:
                    self.stats["cache_misses"] += 1
            
            # 生成嵌入
            result = await self.provider.embed_text(text)
            
            # 更新统计
            self.stats["total_tokens"] += result.token_count
            self.stats["total_processing_time"] += result.processing_time
            
            # 存储到缓存
            if self.cache_enabled and self.cache:
                self.cache.put(text, self.model, result.embedding)
            
            logger.debug(f"生成嵌入: {result.dimensions} 维, {result.token_count} 令牌")
            return result.embedding
            
        except Exception as e:
            logger.error(f"文本嵌入失败: {str(e)}")
            raise VectorEmbeddingError(f"文本嵌入失败: {str(e)}")
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文本嵌入
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        try:
            if not texts:
                return []
            
            # 检查缓存
            embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            if self.cache_enabled and self.cache:
                for i, text in enumerate(texts):
                    cached_embedding = self.cache.get(text, self.model)
                    if cached_embedding is not None:
                        embeddings.append((i, cached_embedding))
                        self.stats["cache_hits"] += 1
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
                        self.stats["cache_misses"] += 1
            else:
                uncached_texts = texts
                uncached_indices = list(range(len(texts)))
            
            # 生成未缓存的嵌入
            if uncached_texts:
                batch_result = await self.provider.embed_texts(uncached_texts)
                
                # 更新统计
                self.stats["total_requests"] += len(uncached_texts)
                self.stats["total_tokens"] += batch_result.total_tokens
                self.stats["total_processing_time"] += batch_result.total_processing_time
                
                # 添加到结果和缓存
                for i, result in enumerate(batch_result.results):
                    original_index = uncached_indices[i]
                    embeddings.append((original_index, result.embedding))
                    
                    # 存储到缓存
                    if self.cache_enabled and self.cache:
                        self.cache.put(result.text, self.model, result.embedding)
            
            # 按原始顺序排序
            embeddings.sort(key=lambda x: x[0])
            result_embeddings = [emb for _, emb in embeddings]
            
            logger.info(f"批量嵌入完成: {len(result_embeddings)} 个向量")
            return result_embeddings
            
        except Exception as e:
            logger.error(f"批量嵌入失败: {str(e)}")
            raise VectorEmbeddingError(f"批量嵌入失败: {str(e)}")
    
    async def calculate_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """
        计算嵌入相似度
        
        Args:
            embedding1: 第一个嵌入向量
            embedding2: 第二个嵌入向量
            
        Returns:
            余弦相似度
            
        Raises:
            VectorDimensionError: 向量维度不匹配
        """
        try:
            if len(embedding1) != len(embedding2):
                raise VectorDimensionError(
                    f"向量维度不匹配: {len(embedding1)} vs {len(embedding2)}"
                )
            
            # 转换为 numpy 数组
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # 计算余弦相似度
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"相似度计算失败: {str(e)}")
            raise VectorError(f"相似度计算失败: {str(e)}")
    
    async def find_similar_texts(
        self, 
        query_embedding: List[float], 
        candidate_embeddings: List[Tuple[str, List[float]]],
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        查找相似文本
        
        Args:
            query_embedding: 查询嵌入
            candidate_embeddings: 候选嵌入列表 (文本, 嵌入)
            top_k: 返回前K个结果
            threshold: 相似度阈值
            
        Returns:
            相似文本列表 (文本, 相似度)
        """
        try:
            similarities = []
            
            for text, embedding in candidate_embeddings:
                similarity = await self.calculate_similarity(query_embedding, embedding)
                if similarity >= threshold:
                    similarities.append((text, similarity))
            
            # 按相似度排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"相似文本查找失败: {str(e)}")
            raise VectorError(f"相似文本查找失败: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        stats = self.stats.copy()
        
        # 添加缓存统计
        if self.cache:
            cache_stats = self.cache.get_stats()
            stats.update({
                "cache_size": cache_stats["size"],
                "cache_max_size": cache_stats["max_size"],
                "cache_memory": cache_stats["total_memory"]
            })
        
        # 计算命中率
        total_requests = stats["cache_hits"] + stats["cache_misses"]
        if total_requests > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / total_requests
        else:
            stats["cache_hit_rate"] = 0.0
        
        return stats
    
    def clear_cache(self) -> None:
        """清空缓存"""
        if self.cache:
            self.cache.clear()
            logger.info("嵌入缓存已清空")
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 测试嵌入生成
            test_text = "这是一个测试文本"
            start_time = datetime.utcnow()
            
            embedding = await self.embed_text(test_text)
            
            response_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "status": "healthy",
                "provider": self.provider_name,
                "model": self.model,
                "embedding_dimensions": len(embedding),
                "response_time": response_time,
                "cache_enabled": self.cache_enabled,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"健康检查失败: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }