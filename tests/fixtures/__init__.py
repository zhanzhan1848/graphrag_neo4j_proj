#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 测试Fixtures包
=======================

提供测试用的fixtures和数据生成器：
- 测试数据生成器
- 数据库fixtures
- API测试fixtures
- 性能测试数据
- 边界情况数据

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

from .test_data import (
    TestDataGenerator,
    TestDataFactory,
    test_data_generator,
    test_data_factory,
    sample_document_data,
    sample_entities_data,
    sample_relations_data,
    performance_test_data,
    multilingual_test_data,
    edge_case_test_data,
    cleanup_test_data
)

__all__ = [
    'TestDataGenerator',
    'TestDataFactory',
    'test_data_generator',
    'test_data_factory',
    'sample_document_data',
    'sample_entities_data',
    'sample_relations_data',
    'performance_test_data',
    'multilingual_test_data',
    'edge_case_test_data',
    'cleanup_test_data'
]