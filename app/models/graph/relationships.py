#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 图关系模型
==================

本模块定义了 GraphRAG 系统的图关系模型（Neo4j）。

关系类型说明：
- ContainsRelationship: 包含关系，表示容器与内容的关系
- MentionsRelationship: 提及关系，表示文本中提及实体的关系
- RelatesToRelationship: 相关关系，表示实体间的一般关联
- PartOfRelationship: 部分关系，表示整体与部分的关系
- SimilarToRelationship: 相似关系，表示实体间的相似性
- LocatedInRelationship: 位于关系，表示地理位置关系
- WorksForRelationship: 工作关系，表示人员与组织的雇佣关系
- ParticipatesInRelationship: 参与关系，表示参与事件或活动
- InfluencesRelationship: 影响关系，表示影响或被影响
- DependsOnRelationship: 依赖关系，表示依赖或被依赖

字段说明：
- weight: 关系权重或强度
- confidence: 关系置信度
- evidence: 支持关系的证据
- source: 关系来源
- created_at/updated_at: 创建和更新时间

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from neomodel import (
    StructuredRel,
    StringProperty,
    DateTimeProperty,
    JSONProperty,
    FloatProperty,
    IntegerProperty,
    BooleanProperty,
    ArrayProperty
)

from .base import BaseRelationship


class ContainsRelationship(BaseRelationship):
    """
    包含关系
    
    表示容器与内容的关系，如文档包含文本块、组织包含部门等。
    """
    
    # 关系特定属性
    container_type = StringProperty()
    content_type = StringProperty()
    
    # 位置信息
    position = IntegerProperty()
    order_index = IntegerProperty()
    
    # 包含程度
    containment_type = StringProperty(default='full')  # full, partial, reference
    coverage_ratio = FloatProperty(default=1.0)
    
    # 统计信息
    size_ratio = FloatProperty()
    importance_ratio = FloatProperty()


class MentionsRelationship(BaseRelationship):
    """
    提及关系
    
    表示文本中提及实体的关系。
    """
    
    # 提及信息
    mention_text = StringProperty()
    mention_type = StringProperty()  # direct, indirect, implicit
    
    # 位置信息
    start_position = IntegerProperty()
    end_position = IntegerProperty()
    sentence_index = IntegerProperty()
    
    # 上下文信息
    context_before = StringProperty()
    context_after = StringProperty()
    context_window = IntegerProperty(default=50)
    
    # 提及质量
    mention_quality = FloatProperty(default=0.0)
    disambiguation_score = FloatProperty(default=0.0)
    
    # 频次统计
    mention_count = IntegerProperty(default=1)
    first_mention = BooleanProperty(default=False)


class RelatesToRelationship(BaseRelationship):
    """
    相关关系
    
    表示实体间的一般关联关系。
    """
    
    # 关系类型
    relation_type = StringProperty(required=True)
    relation_subtype = StringProperty()
    
    # 关系方向
    is_directed = BooleanProperty(default=False)
    direction = StringProperty()  # forward, backward, bidirectional
    
    # 关系强度
    strength = FloatProperty(default=0.5)
    frequency = IntegerProperty(default=1)
    
    # 语义信息
    semantic_type = StringProperty()
    domain = StringProperty()
    
    # 时间信息
    temporal_scope = StringProperty()
    is_temporal = BooleanProperty(default=False)


class PartOfRelationship(BaseRelationship):
    """
    部分关系
    
    表示整体与部分的关系。
    """
    
    # 部分类型
    part_type = StringProperty(required=True)
    whole_type = StringProperty(required=True)
    
    # 层次信息
    hierarchy_level = IntegerProperty(default=1)
    is_direct_part = BooleanProperty(default=True)
    
    # 重要性
    importance_to_whole = FloatProperty(default=0.5)
    size_ratio = FloatProperty()
    
    # 功能信息
    functional_role = StringProperty()
    is_essential = BooleanProperty(default=False)


class SimilarToRelationship(BaseRelationship):
    """
    相似关系
    
    表示实体间的相似性关系。
    """
    
    # 相似性类型
    similarity_type = StringProperty(required=True)
    similarity_dimension = StringProperty()
    
    # 相似度分数
    similarity_score = FloatProperty(required=True)
    cosine_similarity = FloatProperty()
    jaccard_similarity = FloatProperty()
    
    # 相似性特征
    common_features = ArrayProperty(StringProperty(), default=[])
    different_features = ArrayProperty(StringProperty(), default=[])
    
    # 计算信息
    algorithm_used = StringProperty()
    feature_vector_size = IntegerProperty()
    
    # 验证信息
    is_verified = BooleanProperty(default=False)
    human_validated = BooleanProperty(default=False)


class LocatedInRelationship(BaseRelationship):
    """
    位于关系
    
    表示地理位置关系。
    """
    
    # 位置类型
    location_type = StringProperty(required=True)
    precision_level = StringProperty()  # exact, approximate, region
    
    # 地理信息
    distance = FloatProperty()
    direction = StringProperty()
    
    # 行政层级
    administrative_level = StringProperty()
    jurisdiction = StringProperty()
    
    # 时间范围
    start_date = StringProperty()
    end_date = StringProperty()
    is_current = BooleanProperty(default=True)
    
    # 验证信息
    is_verified = BooleanProperty(default=False)
    verification_source = StringProperty()


class WorksForRelationship(BaseRelationship):
    """
    工作关系
    
    表示人员与组织的雇佣关系。
    """
    
    # 职位信息
    position = StringProperty()
    job_title = StringProperty()
    department = StringProperty()
    
    # 雇佣类型
    employment_type = StringProperty()  # full-time, part-time, contract, etc.
    employment_status = StringProperty()  # active, inactive, terminated
    
    # 时间信息
    start_date = StringProperty()
    end_date = StringProperty()
    duration_months = IntegerProperty()
    
    # 薪资信息（如果可用）
    salary_range = StringProperty()
    compensation_type = StringProperty()
    
    # 职责信息
    responsibilities = ArrayProperty(StringProperty(), default=[])
    reporting_to = StringProperty()
    
    # 验证信息
    is_verified = BooleanProperty(default=False)
    verification_source = StringProperty()


class ParticipatesInRelationship(BaseRelationship):
    """
    参与关系
    
    表示参与事件或活动的关系。
    """
    
    # 参与类型
    participation_type = StringProperty(required=True)
    role = StringProperty()
    
    # 参与程度
    involvement_level = StringProperty()  # primary, secondary, observer
    contribution_type = StringProperty()
    
    # 时间信息
    participation_start = StringProperty()
    participation_end = StringProperty()
    duration = StringProperty()
    
    # 结果信息
    outcome = StringProperty()
    impact_level = StringProperty()
    
    # 验证信息
    is_confirmed = BooleanProperty(default=False)
    confirmation_source = StringProperty()


class InfluencesRelationship(BaseRelationship):
    """
    影响关系
    
    表示影响或被影响的关系。
    """
    
    # 影响类型
    influence_type = StringProperty(required=True)
    influence_mechanism = StringProperty()
    
    # 影响强度
    influence_strength = FloatProperty(required=True)
    influence_direction = StringProperty()  # positive, negative, neutral
    
    # 影响范围
    scope = StringProperty()
    affected_aspects = ArrayProperty(StringProperty(), default=[])
    
    # 时间信息
    influence_start = StringProperty()
    influence_duration = StringProperty()
    is_ongoing = BooleanProperty(default=True)
    
    # 证据信息
    evidence_strength = FloatProperty(default=0.0)
    supporting_evidence = ArrayProperty(StringProperty(), default=[])
    
    # 量化指标
    measurable_impact = FloatProperty()
    impact_metrics = JSONProperty(default=dict)


class DependsOnRelationship(BaseRelationship):
    """
    依赖关系
    
    表示依赖或被依赖的关系。
    """
    
    # 依赖类型
    dependency_type = StringProperty(required=True)
    dependency_nature = StringProperty()  # functional, structural, temporal
    
    # 依赖强度
    dependency_strength = FloatProperty(required=True)
    criticality_level = StringProperty()  # critical, important, optional
    
    # 依赖条件
    conditions = ArrayProperty(StringProperty(), default=[])
    prerequisites = ArrayProperty(StringProperty(), default=[])
    
    # 失效影响
    failure_impact = StringProperty()
    cascade_potential = FloatProperty(default=0.0)
    
    # 时间特性
    is_temporal = BooleanProperty(default=False)
    temporal_order = StringProperty()  # before, after, during, concurrent
    
    # 可替代性
    is_replaceable = BooleanProperty(default=True)
    alternatives = ArrayProperty(StringProperty(), default=[])
    
    # 验证信息
    is_validated = BooleanProperty(default=False)
    validation_method = StringProperty()


class CooccursWithRelationship(BaseRelationship):
    """
    共现关系
    
    表示实体在文档或上下文中的共现关系。
    """
    
    # 共现类型
    cooccurrence_type = StringProperty(required=True)
    context_type = StringProperty()  # sentence, paragraph, document, section
    
    # 共现统计
    cooccurrence_count = IntegerProperty(required=True)
    total_occurrences = IntegerProperty()
    cooccurrence_ratio = FloatProperty()
    
    # 距离信息
    average_distance = FloatProperty()
    min_distance = IntegerProperty()
    max_distance = IntegerProperty()
    
    # 上下文信息
    common_contexts = ArrayProperty(StringProperty(), default=[])
    context_similarity = FloatProperty(default=0.0)
    
    # 统计显著性
    statistical_significance = FloatProperty()
    p_value = FloatProperty()
    
    # 时间分布
    temporal_distribution = JSONProperty(default=dict)
    is_consistent = BooleanProperty(default=True)


class CausesRelationship(BaseRelationship):
    """
    因果关系
    
    表示因果关系。
    """
    
    # 因果类型
    causation_type = StringProperty(required=True)
    causal_mechanism = StringProperty()
    
    # 因果强度
    causal_strength = FloatProperty(required=True)
    certainty_level = FloatProperty(default=0.5)
    
    # 时间信息
    temporal_lag = StringProperty()
    causal_direction = StringProperty()  # direct, indirect, bidirectional
    
    # 条件信息
    necessary_conditions = ArrayProperty(StringProperty(), default=[])
    sufficient_conditions = ArrayProperty(StringProperty(), default=[])
    
    # 证据支持
    evidence_type = StringProperty()
    evidence_quality = FloatProperty(default=0.0)
    
    # 验证信息
    is_established = BooleanProperty(default=False)
    research_support = StringProperty()


class CompetesWithRelationship(BaseRelationship):
    """
    竞争关系
    
    表示竞争关系。
    """
    
    # 竞争类型
    competition_type = StringProperty(required=True)
    competition_domain = StringProperty()
    
    # 竞争强度
    competition_intensity = FloatProperty(required=True)
    market_overlap = FloatProperty()
    
    # 竞争维度
    competition_dimensions = ArrayProperty(StringProperty(), default=[])
    competitive_advantages = ArrayProperty(StringProperty(), default=[])
    
    # 市场信息
    market_share_a = FloatProperty()
    market_share_b = FloatProperty()
    market_size = StringProperty()
    
    # 时间信息
    competition_start = StringProperty()
    competition_duration = StringProperty()
    is_ongoing = BooleanProperty(default=True)
    
    # 结果信息
    competitive_outcome = StringProperty()
    winner = StringProperty()


class CollaboratesWithRelationship(BaseRelationship):
    """
    协作关系
    
    表示协作或合作关系。
    """
    
    # 协作类型
    collaboration_type = StringProperty(required=True)
    collaboration_nature = StringProperty()
    
    # 协作范围
    collaboration_scope = StringProperty()
    shared_goals = ArrayProperty(StringProperty(), default=[])
    
    # 协作强度
    collaboration_intensity = FloatProperty(default=0.5)
    resource_sharing_level = FloatProperty()
    
    # 角色分工
    role_distribution = JSONProperty(default=dict)
    responsibilities = ArrayProperty(StringProperty(), default=[])
    
    # 时间信息
    collaboration_start = StringProperty()
    collaboration_duration = StringProperty()
    is_ongoing = BooleanProperty(default=True)
    
    # 成果信息
    joint_outcomes = ArrayProperty(StringProperty(), default=[])
    success_metrics = JSONProperty(default=dict)
    
    # 协议信息
    formal_agreement = BooleanProperty(default=False)
    agreement_type = StringProperty()