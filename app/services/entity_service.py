#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 实体抽取服务
===================

本模块实现了实体抽取的核心功能。

服务功能：
- 命名实体识别（NER）
- 实体类型分类
- 实体标准化
- 实体链接
- 实体去重
- 实体验证
- 实体评分

支持的实体类型：
- PERSON: 人物
- ORGANIZATION: 组织机构
- LOCATION: 地点
- CONCEPT: 概念
- EVENT: 事件
- PRODUCT: 产品
- DATE: 日期
- MONEY: 金额
- MISC: 其他

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import asyncio
import json
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict, Counter
import spacy
from spacy import displacy
from app.services.azure_openai_service import get_azure_openai_service, AzureOpenAIService

from app.core.config import settings
from app.models.database.entities import Entity
# from app.models.schemas.entities import EntityCreate, EntityUpdate
from app.utils.exceptions import (
    EntityError,
    EntityExtractionError,
    EntityValidationError,
    EntityLinkingError,
    ExternalServiceError
)
from app.utils.logger import get_logger
from app.services.embedding_service import EmbeddingService

logger = get_logger(__name__)
settings = settings


@dataclass
class ExtractedEntity:
    """抽取的实体"""
    name: str
    entity_type: str
    description: Optional[str] = None
    confidence: float = 0.0
    start_pos: int = 0
    end_pos: int = 0
    context: Optional[str] = None
    attributes: Dict[str, Any] = None
    canonical_name: Optional[str] = None
    entity_id: Optional[str] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


@dataclass
class EntityExtractionResult:
    """实体抽取结果"""
    entities: List[ExtractedEntity]
    text: str
    total_entities: int
    unique_entities: int
    entity_types: Dict[str, int]
    confidence_stats: Dict[str, float]
    processing_time: float
    model_used: str
    extraction_method: str
    created_at: str


class EntityTypeClassifier:
    """
    实体类型分类器
    
    根据实体名称和上下文确定实体类型。
    """
    
    def __init__(self):
        """初始化分类器"""
        # 实体类型模式
        self.type_patterns = {
            "PERSON": [
                r"^(Dr|Prof|Mr|Ms|Mrs|Miss)\.?\s+\w+",
                r"\w+\s+(博士|教授|先生|女士|老师)",
                r"^[A-Z][a-z]+\s+[A-Z][a-z]+$",  # 英文姓名
                r"^[\u4e00-\u9fa5]{2,4}$"  # 中文姓名
            ],
            "ORGANIZATION": [
                r"(公司|集团|企业|机构|组织|协会|学会|研究所|大学|学院|部门|委员会)",
                r"(Company|Corp|Inc|Ltd|University|Institute|Department|Association)",
                r"(Co\.|Inc\.|Ltd\.|Corp\.)",
                r"^[A-Z][A-Za-z\s&]+(?:Inc|Corp|Ltd|Co)\.?$"
            ],
            "LOCATION": [
                r"(省|市|县|区|镇|村|街道|路|号|楼)",
                r"(Province|City|County|District|Street|Road|Avenue)",
                r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Province|City|County|State))$"
            ],
            "CONCEPT": [
                r"(理论|方法|技术|算法|模型|系统|框架|概念|原理|机制)",
                r"(theory|method|technique|algorithm|model|system|framework|concept|principle)",
                r"(学习|训练|优化|分析|处理|识别|检测|预测)",
                r"(learning|training|optimization|analysis|processing|recognition|detection|prediction)"
            ],
            "EVENT": [
                r"(会议|研讨会|论坛|峰会|展览|比赛|活动|项目|计划)",
                r"(conference|workshop|forum|summit|exhibition|competition|event|project|program)",
                r"(发布|启动|开始|结束|完成|实施)",
                r"(release|launch|start|end|complete|implement)"
            ],
            "PRODUCT": [
                r"(产品|软件|系统|平台|工具|设备|装置|器件)",
                r"(product|software|system|platform|tool|device|equipment|apparatus)",
                r"(版本|型号|规格|标准)",
                r"(version|model|specification|standard)"
            ],
            "DATE": [
                r"\d{4}年\d{1,2}月\d{1,2}日",
                r"\d{1,2}/\d{1,2}/\d{4}",
                r"\d{4}-\d{2}-\d{2}",
                r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}"
            ],
            "MONEY": [
                r"[¥$€£]\d+(?:,\d{3})*(?:\.\d{2})?",
                r"\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:元|美元|欧元|英镑|万元|亿元)",
                r"\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|CNY|RMB)"
            ]
        }
        
        # 上下文关键词
        self.context_keywords = {
            "PERSON": ["说", "认为", "表示", "指出", "提到", "研究", "发现", "开发", "设计"],
            "ORGANIZATION": ["发布", "宣布", "推出", "开发", "研究", "生产", "提供", "服务"],
            "LOCATION": ["位于", "在", "来自", "前往", "到达", "离开", "建立", "设立"],
            "CONCEPT": ["基于", "使用", "应用", "实现", "采用", "提出", "改进", "优化"],
            "EVENT": ["举办", "参加", "组织", "召开", "进行", "开展", "实施", "执行"],
            "PRODUCT": ["使用", "购买", "销售", "开发", "生产", "制造", "设计", "改进"]
        }
    
    def classify_entity_type(
        self, 
        entity_name: str, 
        context: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        分类实体类型
        
        Args:
            entity_name: 实体名称
            context: 上下文
            
        Returns:
            (实体类型, 置信度)
        """
        scores = defaultdict(float)
        
        # 1. 基于模式匹配
        for entity_type, patterns in self.type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, entity_name, re.IGNORECASE):
                    scores[entity_type] += 0.8
        
        # 2. 基于上下文关键词
        if context:
            context_lower = context.lower()
            for entity_type, keywords in self.context_keywords.items():
                for keyword in keywords:
                    if keyword in context_lower:
                        scores[entity_type] += 0.3
        
        # 3. 基于实体名称特征
        if entity_name:
            # 长度特征
            if len(entity_name) <= 4 and re.match(r'^[\u4e00-\u9fa5]+$', entity_name):
                scores["PERSON"] += 0.2
            
            # 大写字母特征
            if entity_name.isupper():
                scores["ORGANIZATION"] += 0.3
                scores["PRODUCT"] += 0.2
            
            # 数字特征
            if re.search(r'\d', entity_name):
                scores["PRODUCT"] += 0.2
                scores["DATE"] += 0.1
        
        # 4. 选择最高分的类型
        if scores:
            best_type = max(scores.keys(), key=lambda k: scores[k])
            confidence = min(scores[best_type], 1.0)
            return best_type, confidence
        else:
            return "MISC", 0.1


class EntityNormalizer:
    """
    实体标准化器
    
    将实体名称标准化为规范形式。
    """
    
    def __init__(self):
        """初始化标准化器"""
        # 标准化规则
        self.normalization_rules = {
            # 人名标准化
            "PERSON": [
                (r"^(Dr|Prof|Mr|Ms|Mrs|Miss)\.?\s+", ""),  # 移除称谓
                (r"\s+(博士|教授|先生|女士|老师)$", ""),  # 移除中文称谓
            ],
            # 组织名标准化
            "ORGANIZATION": [
                (r"\s*(Co\.|Inc\.|Ltd\.|Corp\.)$", lambda m: m.group(1).replace(".", "")),
                (r"(公司|集团|企业)$", ""),  # 可选：移除通用后缀
            ],
            # 地点标准化
            "LOCATION": [
                (r"^(中国|美国|英国|法国|德国|日本)\s*", ""),  # 移除国家前缀（可选）
            ],
            # 概念标准化
            "CONCEPT": [
                (r"(技术|方法|算法|模型)$", ""),  # 移除通用后缀（可选）
            ]
        }
        
        # 别名映射
        self.alias_mapping = {
            "AI": "人工智能",
            "ML": "机器学习",
            "DL": "深度学习",
            "NLP": "自然语言处理",
            "CV": "计算机视觉",
            "USA": "美国",
            "UK": "英国",
            "PRC": "中国"
        }
    
    def normalize_entity_name(
        self, 
        entity_name: str, 
        entity_type: str = "MISC"
    ) -> str:
        """
        标准化实体名称
        
        Args:
            entity_name: 原始实体名称
            entity_type: 实体类型
            
        Returns:
            标准化后的实体名称
        """
        if not entity_name:
            return entity_name
        
        normalized = entity_name.strip()
        
        # 1. 应用别名映射
        if normalized in self.alias_mapping:
            normalized = self.alias_mapping[normalized]
        
        # 2. 应用类型特定的标准化规则
        if entity_type in self.normalization_rules:
            for pattern, replacement in self.normalization_rules[entity_type]:
                if callable(replacement):
                    normalized = re.sub(pattern, replacement, normalized)
                else:
                    normalized = re.sub(pattern, replacement, normalized)
        
        # 3. 通用清理
        normalized = re.sub(r'\s+', ' ', normalized)  # 合并多个空格
        normalized = normalized.strip()
        
        # 4. 大小写标准化
        if entity_type == "PERSON":
            # 人名首字母大写
            normalized = ' '.join(word.capitalize() for word in normalized.split())
        elif entity_type == "LOCATION":
            # 地名首字母大写
            normalized = ' '.join(word.capitalize() for word in normalized.split())
        
        return normalized


class LLMEntityExtractor:
    """
    基于大语言模型的实体抽取器
    
    使用 Azure OpenAI GPT 模型进行实体抽取。
    """
    
    def __init__(
        self,
        azure_openai_service: Optional[AzureOpenAIService] = None,
        model: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.1
    ):
        """
        初始化 LLM 实体抽取器
        
        Args:
            azure_openai_service: Azure OpenAI 服务
            model: 使用的模型
            max_tokens: 最大令牌数
            temperature: 温度参数
        """
        self.azure_openai_service = azure_openai_service or get_azure_openai_service()
        self.model = model or settings.AZURE_OPENAI_CHAT_MODEL
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # 实体抽取提示模板
        self.extraction_prompt = """你是一个专业的命名实体识别专家。请从给定的文本中抽取所有重要的实体。

对于每个实体，请提供以下信息：
- name: 实体名称（保持原文形式）
- type: 实体类型，必须是以下之一：PERSON（人物）、ORGANIZATION（组织）、LOCATION（地点）、CONCEPT（概念）、EVENT（事件）、PRODUCT（产品）、DATE（日期）、MONEY（金额）、MISC（其他）
- description: 实体的简短描述（可选）
- confidence: 置信度（0.0-1.0）
- start_pos: 在文本中的起始位置
- end_pos: 在文本中的结束位置

请以JSON格式返回结果，格式如下：
{
  "entities": [
    {
      "name": "实体名称",
      "type": "实体类型",
      "description": "实体描述",
      "confidence": 0.95,
      "start_pos": 10,
      "end_pos": 15
    }
  ]
}

注意事项：
1. 只抽取重要的、有意义的实体
2. 避免抽取过于通用的词汇
3. 确保实体类型准确
4. 置信度要合理评估

文本内容：
{text}"""
    
    async def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """
        使用 LLM 抽取实体
        
        Args:
            text: 输入文本
            
        Returns:
            抽取的实体列表
            
        Raises:
            EntityExtractionError: 抽取失败
        """
        try:
            # 构建提示
            prompt = self.extraction_prompt.format(text=text)
            
            # 调用 Azure OpenAI API
            response = await self.azure_openai_service.generate_text(
                prompt=prompt,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            try:
                result = json.loads(response)
                entities_data = result.get("entities", [])
                
                entities = []
                for entity_data in entities_data:
                    entity = ExtractedEntity(
                        name=entity_data.get("name", ""),
                        entity_type=entity_data.get("type", "MISC"),
                        description=entity_data.get("description"),
                        confidence=float(entity_data.get("confidence", 0.0)),
                        start_pos=int(entity_data.get("start_pos", 0)),
                        end_pos=int(entity_data.get("end_pos", 0)),
                        context=self._extract_context(text, entity_data.get("start_pos", 0), entity_data.get("end_pos", 0))
                    )
                    entities.append(entity)
                
                logger.debug(f"LLM 抽取到 {len(entities)} 个实体")
                return entities
                
            except json.JSONDecodeError as e:
                logger.error(f"LLM 响应解析失败: {str(e)}")
                logger.debug(f"原始响应: {response}")
                return []
                
        except Exception as e:
            if "API" in str(e):
                logger.error(f"Azure OpenAI API 错误: {str(e)}")
                raise EntityExtractionError(f"LLM 实体抽取失败: {str(e)}")
            else:
                logger.error(f"实体抽取失败: {str(e)}")
                raise EntityExtractionError(f"实体抽取失败: {str(e)}")
    
    def _extract_context(self, text: str, start_pos: int, end_pos: int, window: int = 50) -> str:
        """提取实体上下文"""
        try:
            context_start = max(0, start_pos - window)
            context_end = min(len(text), end_pos + window)
            return text[context_start:context_end]
        except Exception:
            return ""


class EntityLinker:
    """
    实体链接器
    
    将抽取的实体链接到知识库中的现有实体。
    """
    
    def __init__(self, embedding_service: EmbeddingService):
        """
        初始化实体链接器
        
        Args:
            embedding_service: 嵌入服务
        """
        self.embedding_service = embedding_service
        self.similarity_threshold = 0.85
    
    async def link_entities(
        self,
        entities: List[ExtractedEntity],
        existing_entities: List[Entity]
    ) -> List[ExtractedEntity]:
        """
        链接实体到现有实体
        
        Args:
            entities: 待链接的实体
            existing_entities: 现有实体
            
        Returns:
            链接后的实体列表
        """
        try:
            if not existing_entities:
                return entities
            
            # 构建现有实体的嵌入索引
            existing_names = [entity.name for entity in existing_entities]
            existing_embeddings = await self.embedding_service.embed_texts(existing_names)
            
            linked_entities = []
            
            for entity in entities:
                # 1. 精确匹配
                exact_match = self._find_exact_match(entity, existing_entities)
                if exact_match:
                    entity.entity_id = exact_match.id
                    entity.canonical_name = exact_match.name
                    linked_entities.append(entity)
                    continue
                
                # 2. 语义相似度匹配
                entity_embedding = await self.embedding_service.embed_text(entity.name)
                
                best_match = None
                best_similarity = 0.0
                
                for i, existing_embedding in enumerate(existing_embeddings):
                    similarity = await self.embedding_service.calculate_similarity(
                        entity_embedding, existing_embedding
                    )
                    
                    if similarity > best_similarity and similarity >= self.similarity_threshold:
                        best_similarity = similarity
                        best_match = existing_entities[i]
                
                if best_match:
                    entity.entity_id = best_match.id
                    entity.canonical_name = best_match.name
                    entity.confidence = min(entity.confidence, best_similarity)
                
                linked_entities.append(entity)
            
            logger.debug(f"实体链接完成: {len([e for e in linked_entities if e.entity_id])} 个实体被链接")
            return linked_entities
            
        except Exception as e:
            logger.error(f"实体链接失败: {str(e)}")
            raise EntityLinkingError(f"实体链接失败: {str(e)}")
    
    def _find_exact_match(self, entity: ExtractedEntity, existing_entities: List[Entity]) -> Optional[Entity]:
        """查找精确匹配的实体"""
        entity_name_lower = entity.name.lower()
        
        for existing_entity in existing_entities:
            if existing_entity.name.lower() == entity_name_lower:
                return existing_entity
            
            # 检查别名
            if hasattr(existing_entity, 'aliases') and existing_entity.aliases:
                for alias in existing_entity.aliases:
                    if alias.lower() == entity_name_lower:
                        return existing_entity
        
        return None


class EntityService:
    """
    实体抽取服务
    
    提供完整的实体抽取、标准化、链接和管理功能。
    """
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        azure_openai_service: Optional[AzureOpenAIService] = None,
        use_llm: bool = True,
        llm_model: Optional[str] = None
    ):
        """
        初始化实体服务
        
        Args:
            embedding_service: 嵌入服务
            azure_openai_service: Azure OpenAI 服务
            use_llm: 是否使用 LLM
            llm_model: LLM 模型名称
        """
        # 初始化组件
        self.type_classifier = EntityTypeClassifier()
        self.normalizer = EntityNormalizer()
        
        # 初始化嵌入服务
        self.embedding_service = embedding_service or EmbeddingService()
        
        # 初始化实体链接器
        self.entity_linker = EntityLinker(self.embedding_service)
        
        # 初始化 LLM 抽取器
        if use_llm:
            self.llm_extractor = LLMEntityExtractor(
                azure_openai_service=azure_openai_service,
                model=llm_model
            )
        else:
            self.llm_extractor = None
        
        # 配置参数
        self.use_llm = use_llm
        self.min_confidence = 0.3
        self.max_entities_per_text = 100
        
        # 统计信息
        self.stats = {
            "total_extractions": 0,
            "total_entities": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "entity_types": defaultdict(int),
            "average_confidence": 0.0
        }
        
        logger.info(f"实体服务初始化完成 - LLM: {use_llm}, 模型: {llm_model if use_llm else 'N/A'}")
    
    async def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
        existing_entities: Optional[List[Entity]] = None,
        normalize: bool = True,
        link_entities: bool = True
    ) -> EntityExtractionResult:
        """
        从文本中抽取实体
        
        Args:
            text: 输入文本
            entity_types: 指定的实体类型
            existing_entities: 现有实体（用于链接）
            normalize: 是否标准化实体名称
            link_entities: 是否进行实体链接
            
        Returns:
            实体抽取结果
            
        Raises:
            EntityExtractionError: 抽取失败
        """
        try:
            start_time = datetime.utcnow()
            self.stats["total_extractions"] += 1
            
            logger.info(f"开始实体抽取 - 文本长度: {len(text)}")
            
            # 1. 使用 LLM 抽取实体
            if self.use_llm and self.llm_extractor:
                entities = await self.llm_extractor.extract_entities(text)
                extraction_method = "LLM"
            else:
                # 备用方法：基于规则的抽取
                entities = await self._rule_based_extraction(text)
                extraction_method = "Rule-based"
            
            logger.debug(f"初步抽取到 {len(entities)} 个实体")
            
            # 2. 过滤和验证实体
            entities = self._filter_entities(entities, entity_types)
            logger.debug(f"过滤后剩余 {len(entities)} 个实体")
            
            # 3. 实体标准化
            if normalize:
                entities = self._normalize_entities(entities)
                logger.debug("实体标准化完成")
            
            # 4. 实体去重
            entities = self._deduplicate_entities(entities)
            logger.debug(f"去重后剩余 {len(entities)} 个实体")
            
            # 5. 实体链接
            if link_entities and existing_entities:
                entities = await self.entity_linker.link_entities(entities, existing_entities)
                logger.debug("实体链接完成")
            
            # 6. 计算统计信息
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            entity_types_count = Counter(entity.entity_type for entity in entities)
            confidence_scores = [entity.confidence for entity in entities if entity.confidence > 0]
            
            confidence_stats = {
                "mean": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0,
                "min": min(confidence_scores) if confidence_scores else 0.0,
                "max": max(confidence_scores) if confidence_scores else 0.0
            }
            
            # 7. 更新统计
            self.stats["successful_extractions"] += 1
            self.stats["total_entities"] += len(entities)
            for entity_type, count in entity_types_count.items():
                self.stats["entity_types"][entity_type] += count
            
            # 8. 构建结果
            result = EntityExtractionResult(
                entities=entities,
                text=text,
                total_entities=len(entities),
                unique_entities=len(set(entity.name for entity in entities)),
                entity_types=dict(entity_types_count),
                confidence_stats=confidence_stats,
                processing_time=processing_time,
                model_used=self.llm_extractor.model if self.llm_extractor else "Rule-based",
                extraction_method=extraction_method,
                created_at=datetime.utcnow().isoformat()
            )
            
            logger.info(f"实体抽取完成 - 抽取 {len(entities)} 个实体，耗时 {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.stats["failed_extractions"] += 1
            logger.error(f"实体抽取失败: {str(e)}")
            raise EntityExtractionError(f"实体抽取失败: {str(e)}")
    
    async def _rule_based_extraction(self, text: str) -> List[ExtractedEntity]:
        """基于规则的实体抽取（备用方法）"""
        entities = []
        
        # 简单的基于模式的抽取
        patterns = {
            "PERSON": r"[\u4e00-\u9fa5]{2,4}(?=说|认为|表示|指出)",
            "ORGANIZATION": r"[\u4e00-\u9fa5]+(?:公司|集团|企业|机构|组织|大学|学院)",
            "LOCATION": r"[\u4e00-\u9fa5]+(?:省|市|县|区|镇|村)",
            "CONCEPT": r"[\u4e00-\u9fa5]+(?:理论|方法|技术|算法|模型|系统)"
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entity = ExtractedEntity(
                    name=match.group(),
                    entity_type=entity_type,
                    confidence=0.6,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    context=text[max(0, match.start()-20):match.end()+20]
                )
                entities.append(entity)
        
        return entities
    
    def _filter_entities(
        self, 
        entities: List[ExtractedEntity], 
        allowed_types: Optional[List[str]] = None
    ) -> List[ExtractedEntity]:
        """过滤实体"""
        filtered = []
        
        for entity in entities:
            # 过滤置信度过低的实体
            if entity.confidence < self.min_confidence:
                continue
            
            # 过滤指定类型
            if allowed_types and entity.entity_type not in allowed_types:
                continue
            
            # 过滤过短的实体名称
            if len(entity.name.strip()) < 2:
                continue
            
            # 过滤纯数字或符号
            if re.match(r'^[\d\s\W]+$', entity.name):
                continue
            
            filtered.append(entity)
        
        return filtered[:self.max_entities_per_text]
    
    def _normalize_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """标准化实体"""
        for entity in entities:
            # 标准化实体名称
            normalized_name = self.normalizer.normalize_entity_name(
                entity.name, entity.entity_type
            )
            
            if normalized_name != entity.name:
                entity.canonical_name = normalized_name
            
            # 重新分类实体类型（如果需要）
            new_type, confidence = self.type_classifier.classify_entity_type(
                entity.name, entity.context
            )
            
            if confidence > entity.confidence:
                entity.entity_type = new_type
                entity.confidence = confidence
        
        return entities
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """实体去重"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            # 使用标准化名称或原名称作为去重键
            key = entity.canonical_name or entity.name
            key = key.lower().strip()
            
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    async def validate_entity(self, entity: ExtractedEntity) -> bool:
        """验证实体的有效性"""
        try:
            # 基本验证
            if not entity.name or len(entity.name.strip()) < 2:
                return False
            
            # 类型验证
            valid_types = ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "EVENT", "PRODUCT", "DATE", "MONEY", "MISC"]
            if entity.entity_type not in valid_types:
                return False
            
            # 置信度验证
            if entity.confidence < 0.0 or entity.confidence > 1.0:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"实体验证失败: {str(e)}")
            return False
    
    async def batch_extract_entities(
        self,
        texts: List[str],
        **kwargs
    ) -> List[EntityExtractionResult]:
        """批量抽取实体"""
        try:
            results = []
            
            for i, text in enumerate(texts):
                logger.info(f"处理第 {i+1}/{len(texts)} 个文本")
                
                try:
                    result = await self.extract_entities(text, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"第 {i+1} 个文本处理失败: {str(e)}")
                    # 创建空结果
                    empty_result = EntityExtractionResult(
                        entities=[],
                        text=text,
                        total_entities=0,
                        unique_entities=0,
                        entity_types={},
                        confidence_stats={"mean": 0.0, "min": 0.0, "max": 0.0},
                        processing_time=0.0,
                        model_used="Error",
                        extraction_method="Error",
                        created_at=datetime.utcnow().isoformat()
                    )
                    results.append(empty_result)
            
            logger.info(f"批量实体抽取完成 - 处理 {len(texts)} 个文本")
            return results
            
        except Exception as e:
            logger.error(f"批量实体抽取失败: {str(e)}")
            raise EntityExtractionError(f"批量实体抽取失败: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        stats = self.stats.copy()
        
        # 计算成功率
        if stats["total_extractions"] > 0:
            stats["success_rate"] = stats["successful_extractions"] / stats["total_extractions"]
        else:
            stats["success_rate"] = 0.0
        
        # 计算平均实体数
        if stats["successful_extractions"] > 0:
            stats["average_entities_per_text"] = stats["total_entities"] / stats["successful_extractions"]
        else:
            stats["average_entities_per_text"] = 0.0
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            health_status = {
                "status": "healthy",
                "components": {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # 检查 LLM 抽取器
            if self.llm_extractor:
                try:
                    test_text = "张三是一位来自北京大学的研究员，专注于人工智能技术的研究。"
                    entities = await self.llm_extractor.extract_entities(test_text)
                    health_status["components"]["llm_extractor"] = {
                        "status": "healthy",
                        "test_entities": len(entities)
                    }
                except Exception as e:
                    health_status["components"]["llm_extractor"] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    health_status["status"] = "degraded"
            
            # 检查嵌入服务
            if self.embedding_service:
                embedding_health = await self.embedding_service.health_check()
                health_status["components"]["embedding_service"] = embedding_health
                if embedding_health["status"] != "healthy":
                    health_status["status"] = "degraded"
            
            # 检查其他组件
            health_status["components"]["type_classifier"] = {"status": "healthy"}
            health_status["components"]["normalizer"] = {"status": "healthy"}
            health_status["components"]["entity_linker"] = {"status": "healthy"}
            
            return health_status
            
        except Exception as e:
            logger.error(f"健康检查失败: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }