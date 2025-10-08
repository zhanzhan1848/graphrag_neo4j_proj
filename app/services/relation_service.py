#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 关系抽取服务
===================

本模块实现了关系抽取的核心功能。

服务功能：
- 实体关系识别
- 关系类型分类
- 关系强度评估
- 关系验证
- 关系去重
- 关系标准化
- 关系链接

支持的关系类型：
- IS_A: 是一个/属于关系
- PART_OF: 部分关系
- WORKS_AT: 工作于
- LOCATED_IN: 位于
- COLLABORATES_WITH: 合作关系
- DEVELOPS: 开发关系
- USES: 使用关系
- IMPROVES: 改进关系
- EXTENDS: 扩展关系
- CONTRADICTS: 矛盾关系
- SUPPORTS: 支持关系
- CITES: 引用关系
- MENTIONS: 提及关系
- RELATED_TO: 相关关系

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
from app.services.azure_openai_service import get_azure_openai_service, AzureOpenAIService

from app.core.config import settings
from app.models.database.relations import Relation
from app.models.database.entities import Entity
# from app.models.schemas.relations import RelationCreate, RelationUpdate
from app.services.entity_service import ExtractedEntity
from app.utils.exceptions import (
    RelationError,
    RelationExtractionError,
    RelationValidationError,
    RelationLinkingError,
    ExternalServiceError
)
from app.utils.logger import get_logger
from app.services.embedding_service import EmbeddingService

logger = get_logger(__name__)
settings = settings


@dataclass
class ExtractedRelation:
    """抽取的关系"""
    source_entity: str
    target_entity: str
    relation_type: str
    description: Optional[str] = None
    confidence: float = 0.0
    strength: float = 0.0
    evidence_text: Optional[str] = None
    start_pos: int = 0
    end_pos: int = 0
    context: Optional[str] = None
    properties: Dict[str, Any] = None
    relation_id: Optional[str] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class RelationExtractionResult:
    """关系抽取结果"""
    relations: List[ExtractedRelation]
    text: str
    entities: List[str]
    total_relations: int
    unique_relations: int
    relation_types: Dict[str, int]
    confidence_stats: Dict[str, float]
    strength_stats: Dict[str, float]
    processing_time: float
    model_used: str
    extraction_method: str
    created_at: str


class RelationTypeClassifier:
    """
    关系类型分类器
    
    根据实体类型和上下文确定关系类型。
    """
    
    def __init__(self):
        """初始化分类器"""
        # 关系类型模式
        self.relation_patterns = {
            "IS_A": [
                r"是一个|是一种|属于|归属于",
                r"is a|is an|belongs to|is part of",
                r"类型|种类|分类",
                r"type|kind|category"
            ],
            "PART_OF": [
                r"的一部分|包含|组成|构成",
                r"part of|component of|consists of|comprises",
                r"部分|组件|成分",
                r"component|element|part"
            ],
            "WORKS_AT": [
                r"工作于|就职于|任职于|供职于",
                r"works at|employed by|works for",
                r"员工|职员|研究员|教授",
                r"employee|staff|researcher|professor"
            ],
            "LOCATED_IN": [
                r"位于|坐落于|在.*地区|在.*城市",
                r"located in|situated in|based in",
                r"地址|位置|所在地",
                r"address|location|based"
            ],
            "COLLABORATES_WITH": [
                r"合作|协作|配合|共同",
                r"collaborates with|works with|partners with",
                r"合作伙伴|协作者|团队成员",
                r"partner|collaborator|teammate"
            ],
            "DEVELOPS": [
                r"开发|研发|创建|构建|设计",
                r"develops|creates|builds|designs",
                r"开发者|创造者|设计师",
                r"developer|creator|designer"
            ],
            "USES": [
                r"使用|采用|应用|利用",
                r"uses|utilizes|applies|employs",
                r"工具|方法|技术|系统",
                r"tool|method|technique|system"
            ],
            "IMPROVES": [
                r"改进|改善|优化|提升",
                r"improves|enhances|optimizes|upgrades",
                r"改进版|优化版|升级版",
                r"improved|enhanced|optimized"
            ],
            "EXTENDS": [
                r"扩展|延伸|拓展|继承",
                r"extends|expands|inherits from",
                r"扩展版|增强版|派生",
                r"extension|enhancement|derived"
            ],
            "CONTRADICTS": [
                r"矛盾|冲突|相反|对立",
                r"contradicts|conflicts with|opposes",
                r"相反|对比|不同",
                r"opposite|contrary|different"
            ],
            "SUPPORTS": [
                r"支持|证实|验证|证明",
                r"supports|validates|confirms|proves",
                r"证据|支撑|依据",
                r"evidence|support|basis"
            ],
            "CITES": [
                r"引用|参考|援引|提及",
                r"cites|references|mentions|refers to",
                r"参考文献|引文|出处",
                r"reference|citation|source"
            ],
            "MENTIONS": [
                r"提到|提及|涉及|谈到",
                r"mentions|refers to|discusses|talks about",
                r"讨论|描述|说明",
                r"discusses|describes|explains"
            ]
        }
        
        # 实体类型组合的默认关系
        self.entity_type_relations = {
            ("PERSON", "ORGANIZATION"): ["WORKS_AT", "COLLABORATES_WITH"],
            ("PERSON", "PERSON"): ["COLLABORATES_WITH", "MENTIONS"],
            ("ORGANIZATION", "LOCATION"): ["LOCATED_IN"],
            ("CONCEPT", "CONCEPT"): ["IS_A", "PART_OF", "USES", "IMPROVES", "EXTENDS"],
            ("PRODUCT", "ORGANIZATION"): ["DEVELOPS"],
            ("DOCUMENT", "DOCUMENT"): ["CITES", "MENTIONS"],
            ("EVENT", "LOCATION"): ["LOCATED_IN"],
            ("EVENT", "PERSON"): ["MENTIONS"],
        }
    
    def classify_relation_type(
        self,
        source_entity: str,
        target_entity: str,
        context: str,
        source_type: Optional[str] = None,
        target_type: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        分类关系类型
        
        Args:
            source_entity: 源实体
            target_entity: 目标实体
            context: 上下文
            source_type: 源实体类型
            target_type: 目标实体类型
            
        Returns:
            (关系类型, 置信度)
        """
        scores = defaultdict(float)
        context_lower = context.lower()
        
        # 1. 基于模式匹配
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, context_lower):
                    scores[relation_type] += 0.8
        
        # 2. 基于实体类型组合
        if source_type and target_type:
            type_pair = (source_type, target_type)
            if type_pair in self.entity_type_relations:
                for relation_type in self.entity_type_relations[type_pair]:
                    scores[relation_type] += 0.6
            
            # 反向关系
            reverse_pair = (target_type, source_type)
            if reverse_pair in self.entity_type_relations:
                for relation_type in self.entity_type_relations[reverse_pair]:
                    scores[relation_type] += 0.4
        
        # 3. 基于实体名称特征
        source_lower = source_entity.lower()
        target_lower = target_entity.lower()
        
        # 检查是否为同一实体的不同表述
        if source_lower == target_lower or source_lower in target_lower or target_lower in source_lower:
            scores["IS_A"] += 0.3
        
        # 4. 选择最高分的关系类型
        if scores:
            best_type = max(scores.keys(), key=lambda k: scores[k])
            confidence = min(scores[best_type], 1.0)
            return best_type, confidence
        else:
            return "RELATED_TO", 0.1


class RelationValidator:
    """
    关系验证器
    
    验证抽取的关系是否合理。
    """
    
    def __init__(self):
        """初始化验证器"""
        # 不合理的关系组合
        self.invalid_combinations = {
            ("PERSON", "CONCEPT", "LOCATED_IN"),
            ("CONCEPT", "PERSON", "WORKS_AT"),
            ("DATE", "LOCATION", "DEVELOPS"),
        }
        
        # 关系强度阈值
        self.min_strength = 0.1
        self.max_strength = 1.0
    
    def validate_relation(self, relation: ExtractedRelation) -> Tuple[bool, str]:
        """
        验证关系
        
        Args:
            relation: 待验证的关系
            
        Returns:
            (是否有效, 错误信息)
        """
        # 1. 基本验证
        if not relation.source_entity or not relation.target_entity:
            return False, "源实体或目标实体为空"
        
        if relation.source_entity == relation.target_entity:
            return False, "源实体和目标实体相同"
        
        # 2. 置信度验证
        if relation.confidence < 0.0 or relation.confidence > 1.0:
            return False, "置信度超出范围 [0, 1]"
        
        # 3. 强度验证
        if relation.strength < self.min_strength or relation.strength > self.max_strength:
            return False, f"关系强度超出范围 [{self.min_strength}, {self.max_strength}]"
        
        # 4. 关系类型验证
        valid_types = [
            "IS_A", "PART_OF", "WORKS_AT", "LOCATED_IN", "COLLABORATES_WITH",
            "DEVELOPS", "USES", "IMPROVES", "EXTENDS", "CONTRADICTS",
            "SUPPORTS", "CITES", "MENTIONS", "RELATED_TO"
        ]
        
        if relation.relation_type not in valid_types:
            return False, f"无效的关系类型: {relation.relation_type}"
        
        return True, ""


class LLMRelationExtractor:
    """
    基于大语言模型的关系抽取器
    
    使用 Azure OpenAI GPT 模型进行关系抽取。
    """
    
    def __init__(
        self,
        azure_openai_service: Optional[AzureOpenAIService] = None,
        model: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.1
    ):
        """
        初始化 LLM 关系抽取器
        
        Args:
            azure_openai_service: Azure OpenAI 服务实例
            model: 使用的模型（可选，使用服务默认模型）
            max_tokens: 最大令牌数
            temperature: 温度参数
        """
        self.azure_openai_service = azure_openai_service
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # 关系抽取提示模板
        self.extraction_prompt = """你是一个专业的关系抽取专家。请从给定的文本中抽取实体之间的关系。

对于每个关系，请提供以下信息：
- source_entity: 源实体名称
- target_entity: 目标实体名称
- relation_type: 关系类型，必须是以下之一：
  * IS_A: 是一个/属于关系
  * PART_OF: 部分关系
  * WORKS_AT: 工作于
  * LOCATED_IN: 位于
  * COLLABORATES_WITH: 合作关系
  * DEVELOPS: 开发关系
  * USES: 使用关系
  * IMPROVES: 改进关系
  * EXTENDS: 扩展关系
  * CONTRADICTS: 矛盾关系
  * SUPPORTS: 支持关系
  * CITES: 引用关系
  * MENTIONS: 提及关系
  * RELATED_TO: 相关关系
- description: 关系的简短描述
- confidence: 置信度（0.0-1.0）
- strength: 关系强度（0.0-1.0）
- evidence_text: 支持该关系的证据文本

请以JSON格式返回结果，格式如下：
{
  "relations": [
    {
      "source_entity": "源实体名称",
      "target_entity": "目标实体名称",
      "relation_type": "关系类型",
      "description": "关系描述",
      "confidence": 0.95,
      "strength": 0.8,
      "evidence_text": "支持证据"
    }
  ]
}

注意事项：
1. 只抽取明确的、有意义的关系
2. 确保关系类型准确
3. 置信度和强度要合理评估
4. 提供充分的证据文本

已知实体列表：
{entities}

文本内容：
{text}"""
    
    async def extract_relations(
        self, 
        text: str, 
        entities: List[Union[str, ExtractedEntity]]
    ) -> List[ExtractedRelation]:
        """
        使用 LLM 抽取关系
        
        Args:
            text: 输入文本
            entities: 实体列表
            
        Returns:
            抽取的关系列表
            
        Raises:
            RelationExtractionError: 抽取失败
        """
        try:
            # 处理实体列表
            entity_names = []
            for entity in entities:
                if isinstance(entity, str):
                    entity_names.append(entity)
                elif isinstance(entity, ExtractedEntity):
                    entity_names.append(entity.name)
                else:
                    entity_names.append(str(entity))
            
            if len(entity_names) < 2:
                logger.debug("实体数量不足，无法抽取关系")
                return []
            
            # 构建提示
            entities_str = ", ".join(entity_names)
            prompt = self.extraction_prompt.format(
                entities=entities_str,
                text=text
            )
            
            # 获取 Azure OpenAI 服务
            if not self.azure_openai_service:
                self.azure_openai_service = await get_azure_openai_service()
            
            # 调用 Azure OpenAI API
            response = await self.azure_openai_service.generate_text(
                prompt=prompt,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # 解析响应
            content = response
            
            try:
                result = json.loads(content)
                relations_data = result.get("relations", [])
                
                relations = []
                for relation_data in relations_data:
                    relation = ExtractedRelation(
                        source_entity=relation_data.get("source_entity", ""),
                        target_entity=relation_data.get("target_entity", ""),
                        relation_type=relation_data.get("relation_type", "RELATED_TO"),
                        description=relation_data.get("description"),
                        confidence=float(relation_data.get("confidence", 0.0)),
                        strength=float(relation_data.get("strength", 0.0)),
                        evidence_text=relation_data.get("evidence_text"),
                        context=self._extract_context(text, relation_data.get("source_entity", ""), relation_data.get("target_entity", ""))
                    )
                    relations.append(relation)
                
                logger.debug(f"LLM 抽取到 {len(relations)} 个关系")
                return relations
                
            except json.JSONDecodeError as e:
                logger.error(f"LLM 响应解析失败: {str(e)}")
                logger.debug(f"原始响应: {content}")
                return []
                
        except Exception as e:
            if "API" in str(e):
                logger.error(f"Azure OpenAI API 错误: {str(e)}")
                raise RelationExtractionError(f"LLM 关系抽取失败: {str(e)}")
            else:
                logger.error(f"关系抽取失败: {str(e)}")
                raise RelationExtractionError(f"关系抽取失败: {str(e)}")
    
    def _extract_context(self, text: str, source_entity: str, target_entity: str, window: int = 100) -> str:
        """提取关系上下文"""
        try:
            # 查找实体在文本中的位置
            source_pos = text.lower().find(source_entity.lower())
            target_pos = text.lower().find(target_entity.lower())
            
            if source_pos == -1 or target_pos == -1:
                return text[:200]  # 返回文本开头
            
            # 确定上下文范围
            start_pos = min(source_pos, target_pos)
            end_pos = max(source_pos + len(source_entity), target_pos + len(target_entity))
            
            context_start = max(0, start_pos - window)
            context_end = min(len(text), end_pos + window)
            
            return text[context_start:context_end]
        except Exception:
            return text[:200]


class RelationLinker:
    """
    关系链接器
    
    将抽取的关系链接到知识库中的现有关系。
    """
    
    def __init__(self, embedding_service: EmbeddingService):
        """
        初始化关系链接器
        
        Args:
            embedding_service: 嵌入服务
        """
        self.embedding_service = embedding_service
        self.similarity_threshold = 0.8
    
    async def link_relations(
        self,
        relations: List[ExtractedRelation],
        existing_relations: List[Relation]
    ) -> List[ExtractedRelation]:
        """
        链接关系到现有关系
        
        Args:
            relations: 待链接的关系
            existing_relations: 现有关系
            
        Returns:
            链接后的关系列表
        """
        try:
            if not existing_relations:
                return relations
            
            linked_relations = []
            
            for relation in relations:
                # 1. 精确匹配
                exact_match = self._find_exact_match(relation, existing_relations)
                if exact_match:
                    relation.relation_id = str(exact_match.id)
                    linked_relations.append(relation)
                    continue
                
                # 2. 语义相似度匹配
                best_match = await self._find_semantic_match(relation, existing_relations)
                if best_match:
                    relation.relation_id = str(best_match.id)
                    relation.confidence = min(relation.confidence, 0.9)  # 降低置信度
                
                linked_relations.append(relation)
            
            logger.debug(f"关系链接完成: {len([r for r in linked_relations if r.relation_id])} 个关系被链接")
            return linked_relations
            
        except Exception as e:
            logger.error(f"关系链接失败: {str(e)}")
            raise RelationLinkingError(f"关系链接失败: {str(e)}")
    
    def _find_exact_match(self, relation: ExtractedRelation, existing_relations: List[Relation]) -> Optional[Relation]:
        """查找精确匹配的关系"""
        for existing_relation in existing_relations:
            # 检查关系类型和实体匹配
            if (existing_relation.relation_type == relation.relation_type and
                hasattr(existing_relation, 'source_entity') and
                hasattr(existing_relation, 'target_entity')):
                
                source_match = (existing_relation.source_entity.name.lower() == relation.source_entity.lower())
                target_match = (existing_relation.target_entity.name.lower() == relation.target_entity.lower())
                
                if source_match and target_match:
                    return existing_relation
        
        return None
    
    async def _find_semantic_match(self, relation: ExtractedRelation, existing_relations: List[Relation]) -> Optional[Relation]:
        """查找语义相似的关系"""
        try:
            # 构建关系描述
            relation_desc = f"{relation.source_entity} {relation.relation_type} {relation.target_entity}"
            relation_embedding = await self.embedding_service.embed_text(relation_desc)
            
            best_match = None
            best_similarity = 0.0
            
            for existing_relation in existing_relations:
                if hasattr(existing_relation, 'source_entity') and hasattr(existing_relation, 'target_entity'):
                    existing_desc = f"{existing_relation.source_entity.name} {existing_relation.relation_type} {existing_relation.target_entity.name}"
                    existing_embedding = await self.embedding_service.embed_text(existing_desc)
                    
                    similarity = await self.embedding_service.calculate_similarity(
                        relation_embedding, existing_embedding
                    )
                    
                    if similarity > best_similarity and similarity >= self.similarity_threshold:
                        best_similarity = similarity
                        best_match = existing_relation
            
            return best_match
            
        except Exception as e:
            logger.error(f"语义匹配失败: {str(e)}")
            return None


class RelationService:
    """
    关系抽取服务
    
    提供完整的关系抽取、验证、链接和管理功能。
    """
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        azure_openai_service: Optional[AzureOpenAIService] = None,
        use_llm: bool = True,
        llm_model: Optional[str] = None
    ):
        """
        初始化关系服务
        
        Args:
            embedding_service: 嵌入服务
            azure_openai_service: Azure OpenAI 服务
            use_llm: 是否使用 LLM
            llm_model: LLM 模型名称
        """
        # 初始化组件
        self.type_classifier = RelationTypeClassifier()
        self.validator = RelationValidator()
        
        # 初始化嵌入服务
        self.embedding_service = embedding_service or EmbeddingService()
        
        # 初始化关系链接器
        self.relation_linker = RelationLinker(self.embedding_service)
        
        # 初始化 LLM 抽取器
        if use_llm:
            self.llm_extractor = LLMRelationExtractor(
                azure_openai_service=azure_openai_service,
                model=llm_model
            )
        else:
            self.llm_extractor = None
        
        # 配置参数
        self.use_llm = use_llm
        self.min_confidence = 0.3
        self.max_relations_per_text = 50
        
        # 统计信息
        self.stats = {
            "total_extractions": 0,
            "total_relations": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "relation_types": defaultdict(int),
            "average_confidence": 0.0,
            "average_strength": 0.0
        }
        
        logger.info(f"关系服务初始化完成 - LLM: {use_llm}, 模型: {llm_model if use_llm else 'N/A'}")
    
    async def extract_relations(
        self,
        text: str,
        entities: List[Union[str, ExtractedEntity]],
        relation_types: Optional[List[str]] = None,
        existing_relations: Optional[List[Relation]] = None,
        link_relations: bool = True
    ) -> RelationExtractionResult:
        """
        从文本中抽取关系
        
        Args:
            text: 输入文本
            entities: 实体列表
            relation_types: 指定的关系类型
            existing_relations: 现有关系（用于链接）
            link_relations: 是否进行关系链接
            
        Returns:
            关系抽取结果
            
        Raises:
            RelationExtractionError: 抽取失败
        """
        try:
            start_time = datetime.utcnow()
            self.stats["total_extractions"] += 1
            
            logger.info(f"开始关系抽取 - 文本长度: {len(text)}, 实体数量: {len(entities)}")
            
            if len(entities) < 2:
                logger.warning("实体数量不足，无法抽取关系")
                return self._create_empty_result(text, entities, start_time)
            
            # 1. 使用 LLM 抽取关系
            if self.use_llm and self.llm_extractor:
                relations = await self.llm_extractor.extract_relations(text, entities)
                extraction_method = "LLM"
            else:
                # 备用方法：基于规则的抽取
                relations = await self._rule_based_extraction(text, entities)
                extraction_method = "Rule-based"
            
            logger.debug(f"初步抽取到 {len(relations)} 个关系")
            
            # 2. 过滤和验证关系
            relations = self._filter_relations(relations, relation_types)
            logger.debug(f"过滤后剩余 {len(relations)} 个关系")
            
            # 3. 关系去重
            relations = self._deduplicate_relations(relations)
            logger.debug(f"去重后剩余 {len(relations)} 个关系")
            
            # 4. 关系链接
            if link_relations and existing_relations:
                relations = await self.relation_linker.link_relations(relations, existing_relations)
                logger.debug("关系链接完成")
            
            # 5. 计算统计信息
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            relation_types_count = Counter(relation.relation_type for relation in relations)
            confidence_scores = [relation.confidence for relation in relations if relation.confidence > 0]
            strength_scores = [relation.strength for relation in relations if relation.strength > 0]
            
            confidence_stats = {
                "mean": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0,
                "min": min(confidence_scores) if confidence_scores else 0.0,
                "max": max(confidence_scores) if confidence_scores else 0.0
            }
            
            strength_stats = {
                "mean": sum(strength_scores) / len(strength_scores) if strength_scores else 0.0,
                "min": min(strength_scores) if strength_scores else 0.0,
                "max": max(strength_scores) if strength_scores else 0.0
            }
            
            # 6. 更新统计
            self.stats["successful_extractions"] += 1
            self.stats["total_relations"] += len(relations)
            for relation_type, count in relation_types_count.items():
                self.stats["relation_types"][relation_type] += count
            
            # 7. 构建结果
            entity_names = [str(entity) if isinstance(entity, str) else entity.name for entity in entities]
            
            result = RelationExtractionResult(
                relations=relations,
                text=text,
                entities=entity_names,
                total_relations=len(relations),
                unique_relations=len(set((r.source_entity, r.target_entity, r.relation_type) for r in relations)),
                relation_types=dict(relation_types_count),
                confidence_stats=confidence_stats,
                strength_stats=strength_stats,
                processing_time=processing_time,
                model_used=self.llm_extractor.model if self.llm_extractor else "Rule-based",
                extraction_method=extraction_method,
                created_at=datetime.utcnow().isoformat()
            )
            
            logger.info(f"关系抽取完成 - 抽取 {len(relations)} 个关系，耗时 {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.stats["failed_extractions"] += 1
            logger.error(f"关系抽取失败: {str(e)}")
            raise RelationExtractionError(f"关系抽取失败: {str(e)}")
    
    async def _rule_based_extraction(
        self, 
        text: str, 
        entities: List[Union[str, ExtractedEntity]]
    ) -> List[ExtractedRelation]:
        """基于规则的关系抽取（备用方法）"""
        relations = []
        
        # 转换实体为字符串列表
        entity_names = []
        for entity in entities:
            if isinstance(entity, str):
                entity_names.append(entity)
            elif isinstance(entity, ExtractedEntity):
                entity_names.append(entity.name)
            else:
                entity_names.append(str(entity))
        
        # 简单的基于模式的关系抽取
        for i, source in enumerate(entity_names):
            for j, target in enumerate(entity_names):
                if i != j:
                    # 查找实体在文本中的共现
                    source_pos = text.lower().find(source.lower())
                    target_pos = text.lower().find(target.lower())
                    
                    if source_pos != -1 and target_pos != -1:
                        # 提取上下文
                        start_pos = min(source_pos, target_pos)
                        end_pos = max(source_pos + len(source), target_pos + len(target))
                        context = text[max(0, start_pos-50):end_pos+50]
                        
                        # 分类关系类型
                        relation_type, confidence = self.type_classifier.classify_relation_type(
                            source, target, context
                        )
                        
                        if confidence >= self.min_confidence:
                            relation = ExtractedRelation(
                                source_entity=source,
                                target_entity=target,
                                relation_type=relation_type,
                                confidence=confidence,
                                strength=confidence * 0.8,  # 强度略低于置信度
                                context=context,
                                evidence_text=context
                            )
                            relations.append(relation)
        
        return relations
    
    def _filter_relations(
        self, 
        relations: List[ExtractedRelation], 
        allowed_types: Optional[List[str]] = None
    ) -> List[ExtractedRelation]:
        """过滤关系"""
        filtered = []
        
        for relation in relations:
            # 验证关系
            is_valid, error_msg = self.validator.validate_relation(relation)
            if not is_valid:
                logger.debug(f"关系验证失败: {error_msg}")
                continue
            
            # 过滤置信度过低的关系
            if relation.confidence < self.min_confidence:
                continue
            
            # 过滤指定类型
            if allowed_types and relation.relation_type not in allowed_types:
                continue
            
            filtered.append(relation)
        
        return filtered[:self.max_relations_per_text]
    
    def _deduplicate_relations(self, relations: List[ExtractedRelation]) -> List[ExtractedRelation]:
        """关系去重"""
        seen = set()
        unique_relations = []
        
        for relation in relations:
            # 使用关系三元组作为去重键
            key = (
                relation.source_entity.lower().strip(),
                relation.target_entity.lower().strip(),
                relation.relation_type
            )
            
            if key not in seen:
                seen.add(key)
                unique_relations.append(relation)
            else:
                # 如果已存在相同关系，保留置信度更高的
                for i, existing in enumerate(unique_relations):
                    existing_key = (
                        existing.source_entity.lower().strip(),
                        existing.target_entity.lower().strip(),
                        existing.relation_type
                    )
                    if existing_key == key and relation.confidence > existing.confidence:
                        unique_relations[i] = relation
                        break
        
        return unique_relations
    
    def _create_empty_result(
        self, 
        text: str, 
        entities: List[Union[str, ExtractedEntity]], 
        start_time: datetime
    ) -> RelationExtractionResult:
        """创建空的抽取结果"""
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        entity_names = [str(entity) if isinstance(entity, str) else entity.name for entity in entities]
        
        return RelationExtractionResult(
            relations=[],
            text=text,
            entities=entity_names,
            total_relations=0,
            unique_relations=0,
            relation_types={},
            confidence_stats={"mean": 0.0, "min": 0.0, "max": 0.0},
            strength_stats={"mean": 0.0, "min": 0.0, "max": 0.0},
            processing_time=processing_time,
            model_used="N/A",
            extraction_method="Empty",
            created_at=datetime.utcnow().isoformat()
        )
    
    async def batch_extract_relations(
        self,
        texts: List[str],
        entities_list: List[List[Union[str, ExtractedEntity]]],
        **kwargs
    ) -> List[RelationExtractionResult]:
        """批量抽取关系"""
        try:
            if len(texts) != len(entities_list):
                raise RelationExtractionError("文本数量与实体列表数量不匹配")
            
            results = []
            
            for i, (text, entities) in enumerate(zip(texts, entities_list)):
                logger.info(f"处理第 {i+1}/{len(texts)} 个文本")
                
                try:
                    result = await self.extract_relations(text, entities, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"第 {i+1} 个文本处理失败: {str(e)}")
                    # 创建空结果
                    empty_result = self._create_empty_result(text, entities, datetime.utcnow())
                    results.append(empty_result)
            
            logger.info(f"批量关系抽取完成 - 处理 {len(texts)} 个文本")
            return results
            
        except Exception as e:
            logger.error(f"批量关系抽取失败: {str(e)}")
            raise RelationExtractionError(f"批量关系抽取失败: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        stats = self.stats.copy()
        
        # 计算成功率
        if stats["total_extractions"] > 0:
            stats["success_rate"] = stats["successful_extractions"] / stats["total_extractions"]
        else:
            stats["success_rate"] = 0.0
        
        # 计算平均关系数
        if stats["successful_extractions"] > 0:
            stats["average_relations_per_text"] = stats["total_relations"] / stats["successful_extractions"]
        else:
            stats["average_relations_per_text"] = 0.0
        
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
                    test_text = "张三在北京大学工作，他与李四合作开发了一个新的人工智能系统。"
                    test_entities = ["张三", "北京大学", "李四", "人工智能系统"]
                    relations = await self.llm_extractor.extract_relations(test_text, test_entities)
                    health_status["components"]["llm_extractor"] = {
                        "status": "healthy",
                        "test_relations": len(relations)
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
            health_status["components"]["validator"] = {"status": "healthy"}
            health_status["components"]["relation_linker"] = {"status": "healthy"}
            
            return health_status
            
        except Exception as e:
            logger.error(f"健康检查失败: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }