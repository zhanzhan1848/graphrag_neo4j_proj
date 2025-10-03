# 多学科论文知识库设计文档

## 1. 系统概述

本系统基于Neo4j图数据库和PostgreSQL关系数据库，构建一个支持多学科论文和文献的知识库系统。系统采用TBox（本体层）+ ABox（实例层）的设计模式，支持动态扩展和概念收敛。

### 1.1 核心特性

- **多学科支持**：计算机科学、物理学、数学、生物学、化学、工程学等
- **动态扩展**：运行时新增概念和关系
- **概念收敛**：别名和同义词自动归并
- **可追溯性**：每个关系和断言都能回溯到源文档
- **向量检索**：基于embedding的语义相似度检索

### 1.2 技术架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   文档输入层     │    │   处理引擎层     │    │   存储层        │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ PDF/TXT/MD/HTML │───▶│ 文本分块        │───▶│ PostgreSQL      │
│ 图片OCR         │    │ 实体抽取        │    │ (文档/向量)     │
│ 批量导入API     │    │ 关系抽取        │    │                 │
└─────────────────┘    │ 概念收敛        │    │ Neo4j           │
                       │ 向量生成        │    │ (知识图谱)      │
                       └─────────────────┘    └─────────────────┘
```

## 2. 多学科本体设计

### 2.1 学科分类体系

基于国际标准学科分类，支持以下主要学科：

#### 2.1.1 计算机科学 (Computer Science)
- **图形学** (Computer Graphics)
  - 渲染 (Rendering)
  - 光照 (Lighting) 
  - 几何处理 (Geometry Processing)
  - 加速结构 (Acceleration Structures)
- **人工智能** (Artificial Intelligence)
  - 机器学习 (Machine Learning)
  - 深度学习 (Deep Learning)
  - 自然语言处理 (Natural Language Processing)
  - 计算机视觉 (Computer Vision)
- **系统与网络** (Systems and Networks)
  - 操作系统 (Operating Systems)
  - 分布式系统 (Distributed Systems)
  - 网络协议 (Network Protocols)

#### 2.1.2 物理学 (Physics)
- **理论物理** (Theoretical Physics)
  - 量子力学 (Quantum Mechanics)
  - 相对论 (Relativity)
  - 统计力学 (Statistical Mechanics)
- **应用物理** (Applied Physics)
  - 凝聚态物理 (Condensed Matter Physics)
  - 光学 (Optics)
  - 声学 (Acoustics)

#### 2.1.3 数学 (Mathematics)
- **纯数学** (Pure Mathematics)
  - 代数 (Algebra)
  - 几何 (Geometry)
  - 分析 (Analysis)
- **应用数学** (Applied Mathematics)
  - 数值分析 (Numerical Analysis)
  - 优化理论 (Optimization Theory)
  - 概率统计 (Probability and Statistics)

#### 2.1.4 生物学 (Biology)
- **分子生物学** (Molecular Biology)
  - 基因组学 (Genomics)
  - 蛋白质组学 (Proteomics)
- **生态学** (Ecology)
  - 环境生物学 (Environmental Biology)
  - 保护生物学 (Conservation Biology)

#### 2.1.5 化学 (Chemistry)
- **有机化学** (Organic Chemistry)
- **无机化学** (Inorganic Chemistry)
- **物理化学** (Physical Chemistry)
- **分析化学** (Analytical Chemistry)

#### 2.1.6 工程学 (Engineering)
- **机械工程** (Mechanical Engineering)
- **电气工程** (Electrical Engineering)
- **土木工程** (Civil Engineering)
- **化学工程** (Chemical Engineering)

### 2.2 节点类型定义

#### 2.2.1 核心节点类型
```
:Discipline {
  id: String,           // 学科唯一标识
  name: String,         // 学科名称
  name_en: String,      // 英文名称
  name_zh: String,      // 中文名称
  description: String,  // 学科描述
  level: Integer,       // 学科层级 (1=一级学科, 2=二级学科, etc.)
  created_at: DateTime,
  updated_at: DateTime
}

:Subfield {
  id: String,
  name: String,
  name_en: String,
  name_zh: String,
  description: String,
  level: Integer,
  parent_discipline: String,
  created_at: DateTime,
  updated_at: DateTime
}

:Concept {
  id: String,
  preferredLabel: String,    // 首选标签
  altLabels: [String],       // 别名列表
  description: String,
  canonical: Boolean,        // 是否为规范概念
  confidence: Float,         // 置信度
  source: String,           // 来源
  embedding_id: String,     // 对应的向量ID
  created_at: DateTime,
  updated_at: DateTime
}

:Document {
  id: String,
  title: String,
  authors: [String],
  year: Integer,
  venue: String,           // 期刊/会议
  doi: String,
  url: String,
  abstract: String,
  keywords: [String],
  language: String,
  document_type: String,   // paper, book, thesis, etc.
  file_path: String,
  created_at: DateTime,
  updated_at: DateTime
}

:Segment {
  id: String,
  text: String,
  start_char: Integer,
  end_char: Integer,
  page: Integer,
  section: String,         // 章节信息
  document_id: String,
  embedding_id: String,    // 对应的向量ID
  language: String,
  created_at: DateTime
}

:Author {
  id: String,
  name: String,
  orcid: String,
  affiliation: String,
  email: String,
  h_index: Integer,
  created_at: DateTime,
  updated_at: DateTime
}

:Venue {
  id: String,
  name: String,
  type: String,           // journal, conference, workshop
  impact_factor: Float,
  h5_index: Integer,
  publisher: String,
  created_at: DateTime
}
```

### 2.3 关系类型定义

#### 2.3.1 层次关系
```
(:Subfield)-[:BELONGS_TO]->(:Discipline)
(:Concept)-[:IS_A]->(:Subfield)
(:Concept)-[:IS_A]->(:Concept)  // 概念层次
```

#### 2.3.2 文档关系
```
(:Document)-[:HAS_SEGMENT]->(:Segment)
(:Document)-[:AUTHORED_BY]->(:Author)
(:Document)-[:PUBLISHED_IN]->(:Venue)
(:Document)-[:CITES {context: String, page: Integer}]->(:Document)
```

#### 2.3.3 概念关系
```
(:Segment)-[:MENTIONS {score: Float, position: Integer, provenance: String}]->(:Concept)
(:Concept)-[:SAME_AS {confidence: Float, method: String}]->(:Concept)
(:Concept)-[:RELATED_TO {relationType: String, weight: Float, evidence: String}]->(:Concept)
```

#### 2.3.4 跨学科关系
```
(:Concept)-[:APPLIES_TO]->(:Concept)      // 应用关系
(:Concept)-[:IMPROVES]->(:Concept)        // 改进关系
(:Concept)-[:EXTENDS]->(:Concept)         // 扩展关系
(:Concept)-[:CONTRADICTS]->(:Concept)     // 矛盾关系
(:Concept)-[:SUPPORTS]->(:Concept)        // 支持关系
```

## 3. 概念收敛策略

### 3.1 别名识别方法

1. **字符串相似度**：编辑距离、Jaccard相似度
2. **向量相似度**：基于embedding的余弦相似度
3. **上下文相似度**：共现模式分析
4. **专家规则**：领域特定的同义词规则

### 3.2 收敛流程

```
输入概念 → 预处理 → 候选匹配 → 相似度计算 → 阈值判断 → 人工确认 → 合并/创建
```

### 3.3 质量控制

- **置信度评分**：每个合并操作都有置信度分数
- **人工审核**：低置信度的合并需要人工确认
- **版本控制**：保留合并历史，支持回滚
- **冲突解决**：处理多个候选概念的冲突情况

## 4. 数据导入流程

### 4.1 文档处理流程

```
1. 文档上传 → 格式识别 → 文本提取
2. 文本分块 → 去重 → 存储到PostgreSQL
3. 向量生成 → 存储到PostgreSQL向量表
4. 实体抽取 → 关系抽取 → 概念收敛
5. 知识图谱构建 → 存储到Neo4j
6. 索引构建 → 质量检查 → 完成
```

### 4.2 批量导入支持

- **并行处理**：支持多文档并行处理
- **增量更新**：支持增量导入和更新
- **错误恢复**：处理失败时的回滚机制
- **进度跟踪**：实时显示处理进度

## 5. 查询和检索

### 5.1 查询类型

1. **关键词查询**：基于文本匹配
2. **语义查询**：基于向量相似度
3. **图查询**：基于图结构遍历
4. **混合查询**：结合多种查询方式

### 5.2 RAG集成

- **检索增强**：结合向量检索和图查询
- **上下文扩展**：利用图结构扩展相关概念
- **证据链**：提供完整的推理证据链
- **多跳推理**：支持多步推理查询

## 6. 系统扩展性

### 6.1 新学科添加

- **模板化**：基于模板快速添加新学科
- **本体映射**：自动映射到现有本体结构
- **关系发现**：自动发现跨学科关系

### 6.2 性能优化

- **索引策略**：针对不同查询模式的索引优化
- **缓存机制**：热点数据缓存
- **分片策略**：大规模数据的分片存储
- **查询优化**：Cypher查询优化

## 7. 质量保证

### 7.1 数据质量

- **一致性检查**：定期检查数据一致性
- **完整性验证**：确保关系的完整性
- **准确性评估**：基于专家知识的准确性评估

### 7.2 系统监控

- **性能监控**：查询响应时间、吞吐量
- **错误监控**：异常检测和报警
- **使用统计**：用户行为分析

## 8. 未来扩展

### 8.1 高级功能

- **时间维度**：支持知识的时间演化
- **不确定性**：处理不确定和模糊知识
- **多模态**：支持图像、视频等多模态数据
- **协作编辑**：支持多用户协作编辑

### 8.2 AI增强

- **自动标注**：基于大模型的自动标注
- **关系发现**：自动发现隐含关系
- **知识推理**：基于图神经网络的推理
- **质量评估**：AI辅助的质量评估