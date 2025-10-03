#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 服务模块
================

本模块提供 Microsoft GraphRAG 的集成服务，结合 Azure OpenAI 和现有的知识图谱系统：
1. 文档处理和分块
2. 实体和关系抽取
3. 知识图谱构建
4. RAG 查询和推理
5. 与 Neo4j、PostgreSQL 的集成

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import asyncio
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger
from app.services.azure_openai_service import get_azure_openai_service
# from app.models.document import Document, DocumentChunk

logger = get_logger(__name__)


class GraphRAGService:
    """
    GraphRAG 服务类
    
    集成 Microsoft GraphRAG 功能，提供文档处理、实体抽取、
    关系构建和 RAG 查询等核心功能。
    """
    
    def __init__(self):
        """初始化 GraphRAG 服务"""
        self.azure_openai = None
        self.chunk_size = settings.GRAPHRAG_CHUNK_SIZE
        self.chunk_overlap = settings.GRAPHRAG_CHUNK_OVERLAP

    @classmethod
    async def create(cls):
        """工厂方法：异步创建并初始化实例"""
        self = cls()
        await self._initialize_service()
        return self
    
    async def _initialize_service(self):
        """初始化服务依赖"""
        try:
            self.azure_openai = await get_azure_openai_service()
            logger.info("GraphRAG 服务初始化成功")
        except Exception as e:
            logger.error(f"GraphRAG 服务初始化失败: {e}")
            raise
    
    def chunk_text(self, text: str, chunk_id_prefix: str = "") -> List[Dict[str, Any]]:
        """
        文本分块处理
        
        Args:
            text: 输入文本
            chunk_id_prefix: 分块ID前缀
            
        Returns:
            分块结果列表
        """
        try:
            chunks = []
            text_length = len(text)
            
            # 简单的滑动窗口分块
            start = 0
            chunk_index = 0
            
            while start < text_length:
                end = min(start + self.chunk_size, text_length)
                chunk_text = text[start:end]
                
                # 尝试在句号处分割，避免截断句子
                if end < text_length and '。' in chunk_text[-50:]:
                    last_period = chunk_text.rfind('。')
                    if last_period > len(chunk_text) * 0.7:  # 确保不会过度缩短
                        end = start + last_period + 1
                        chunk_text = text[start:end]
                
                chunk_id = f"{chunk_id_prefix}chunk_{chunk_index}"
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text.strip(),
                    "start_pos": start,
                    "end_pos": end,
                    "length": len(chunk_text),
                    "chunk_index": chunk_index
                })
                
                start = end - self.chunk_overlap
                chunk_index += 1
                
                # 防止无限循环
                if start >= end:
                    break
            
            logger.info(f"文本分块完成: {len(chunks)} 个分块")
            return chunks
            
        except Exception as e:
            logger.error(f"文本分块失败: {e}")
            raise
    
    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        实体抽取
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表
        """
        try:
            system_message = """你是一个专业的实体抽取专家。请从给定的文本中抽取所有重要的实体。

对于每个实体，请提供以下信息：
- name: 实体名称
- type: 实体类型（如：人物、组织、地点、概念、事件等）
- description: 实体的简短描述
- importance: 重要性评分（1-10）

请以JSON格式返回结果，格式如下：
{
  "entities": [
    {
      "name": "实体名称",
      "type": "实体类型",
      "description": "实体描述",
      "importance": 8
    }
  ]
}"""
            
            user_message = f"请从以下文本中抽取实体：\n\n{text}"
            
            response = await self.azure_openai.generate_text(
                prompt=user_message,
                system_message=system_message,
                temperature=0.1
            )
            
            # 解析JSON响应
            try:
                result = json.loads(response)
                entities = result.get("entities", [])
                logger.debug(f"抽取到 {len(entities)} 个实体")
                return entities
            except json.JSONDecodeError:
                logger.warning("实体抽取响应格式错误，返回空列表")
                return []
                
        except Exception as e:
            logger.error(f"实体抽取失败: {e}")
            return []
    
    async def extract_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        关系抽取
        
        Args:
            text: 输入文本
            entities: 已抽取的实体列表
            
        Returns:
            关系列表
        """
        try:
            if not entities:
                return []
            
            entity_names = [entity["name"] for entity in entities]
            
            system_message = """你是一个专业的关系抽取专家。请从给定的文本中抽取实体之间的关系。

对于每个关系，请提供以下信息：
- source: 源实体名称
- target: 目标实体名称
- relationship: 关系类型
- description: 关系描述
- strength: 关系强度（1-10）

请以JSON格式返回结果，格式如下：
{
  "relationships": [
    {
      "source": "源实体",
      "target": "目标实体",
      "relationship": "关系类型",
      "description": "关系描述",
      "strength": 8
    }
  ]
}"""
            
            user_message = f"""请从以下文本中抽取关系，重点关注这些实体之间的关系：
实体列表：{', '.join(entity_names)}

文本内容：
{text}"""
            
            response = await self.azure_openai.generate_text(
                prompt=user_message,
                system_message=system_message,
                temperature=0.1
            )
            
            # 解析JSON响应
            try:
                result = json.loads(response)
                relationships = result.get("relationships", [])
                logger.debug(f"抽取到 {len(relationships)} 个关系")
                return relationships
            except json.JSONDecodeError:
                logger.warning("关系抽取响应格式错误，返回空列表")
                return []
                
        except Exception as e:
            logger.error(f"关系抽取失败: {e}")
            return []
    
    async def extract_claims(self, text: str) -> List[Dict[str, Any]]:
        """
        断言（claims）抽取
        
        Args:
            text: 输入文本
            
        Returns:
            断言列表
        """
        try:
            system_message = """你是一个专业的断言抽取专家。请从给定的文本中抽取所有重要的断言或声明。

对于每个断言，请提供以下信息：
- claim: 断言内容
- subject: 断言主体
- predicate: 断言谓词
- object: 断言客体
- confidence: 置信度（1-10）
- evidence: 支持证据

请以JSON格式返回结果，格式如下：
{
  "claims": [
    {
      "claim": "断言内容",
      "subject": "主体",
      "predicate": "谓词",
      "object": "客体",
      "confidence": 8,
      "evidence": "支持证据"
    }
  ]
}"""
            
            user_message = f"请从以下文本中抽取断言：\n\n{text}"
            
            response = await self.azure_openai.generate_text(
                prompt=user_message,
                system_message=system_message,
                temperature=0.1
            )
            
            # 解析JSON响应
            try:
                result = json.loads(response)
                claims = result.get("claims", [])
                logger.debug(f"抽取到 {len(claims)} 个断言")
                return claims
            except json.JSONDecodeError:
                logger.warning("断言抽取响应格式错误，返回空列表")
                return []
                
        except Exception as e:
            logger.error(f"断言抽取失败: {e}")
            return []
    
    async def process_document(self, document_id: str, text: str) -> Dict[str, Any]:
        """
        处理单个文档
        
        Args:
            document_id: 文档ID
            text: 文档文本
            
        Returns:
            处理结果
        """
        try:
            logger.info(f"开始处理文档: {document_id}")
            
            # 1. 文本分块
            chunks = self.chunk_text(text, f"{document_id}_")
            
            # 2. 为每个分块生成嵌入
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = await self.azure_openai.batch_generate_embeddings(chunk_texts)
            
            # 3. 为分块添加嵌入
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding
            
            # 4. 对整个文档进行实体抽取
            entities = await self.extract_entities(text)
            
            # 5. 关系抽取
            relationships = await self.extract_relationships(text, entities)
            
            # 6. 断言抽取
            claims = await self.extract_claims(text)
            
            result = {
                "document_id": document_id,
                "chunks": chunks,
                "entities": entities,
                "relationships": relationships,
                "claims": claims,
                "processed_at": datetime.utcnow().isoformat(),
                "stats": {
                    "chunk_count": len(chunks),
                    "entity_count": len(entities),
                    "relationship_count": len(relationships),
                    "claim_count": len(claims)
                }
            }
            
            logger.info(f"文档处理完成: {document_id}, 统计: {result['stats']}")
            return result
            
        except Exception as e:
            logger.error(f"文档处理失败: {document_id}, 错误: {e}")
            raise
    
    async def query_knowledge_graph(
        self,
        query: str,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        知识图谱查询
        
        Args:
            query: 查询问题
            max_results: 最大结果数
            
        Returns:
            查询结果
        """
        try:
            logger.info(f"执行知识图谱查询: {query}")
            
            # 1. 生成查询嵌入
            query_embedding = await self.azure_openai.generate_single_embedding(query)
            
            # 2. 这里应该与 Neo4j 和 PostgreSQL 集成进行相似性搜索
            # 暂时返回模拟结果
            
            # 3. 使用 LLM 生成回答
            system_message = """你是一个知识图谱问答专家。基于提供的上下文信息，请回答用户的问题。

要求：
1. 回答要准确、简洁
2. 如果信息不足，请说明
3. 提供相关的实体和关系信息
4. 标注信息来源"""
            
            # 这里应该包含从知识图谱检索到的相关信息
            context = "暂时没有具体的上下文信息，需要与数据库集成。"
            
            user_message = f"""问题：{query}

上下文信息：
{context}

请基于上下文信息回答问题。"""
            
            answer = await self.azure_openai.generate_text(
                prompt=user_message,
                system_message=system_message
            )
            
            result = {
                "query": query,
                "answer": answer,
                "context": context,
                "query_embedding": query_embedding,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"知识图谱查询完成: {query}")
            return result
            
        except Exception as e:
            logger.error(f"知识图谱查询失败: {query}, 错误: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态
        """
        try:
            # 检查 Azure OpenAI 服务
            azure_openai_healthy = await self.azure_openai.health_check()
            
            return {
                "status": "healthy" if azure_openai_healthy else "unhealthy",
                "azure_openai": azure_openai_healthy,
                "config": {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"GraphRAG 健康检查失败: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# 全局服务实例
graphrag_service = GraphRAGService()


async def get_graphrag_service() -> GraphRAGService:
    """
    获取 GraphRAG 服务实例（依赖注入）
    
    Returns:
        GraphRAG 服务实例
    """
    return await graphrag_service.create()
