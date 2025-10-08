#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 文本处理服务
===================

本模块实现了文本处理的核心功能。

服务功能：
- 文本分块（chunking）
- 文本去重和标准化
- 文本清理和预处理
- 语言检测
- 文本质量评估
- 文本统计分析

处理策略：
- 智能分块算法
- 语义边界识别
- 重复内容检测
- 文本标准化处理

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import re
import hashlib
import unicodedata
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import langdetect
from langdetect.lang_detect_exception import LangDetectException

from app.utils.exceptions import (
    TextProcessingError,
    TextChunkingError,
    TextCleaningError,
    TextNormalizationError,
    LanguageDetectionError
)
from app.utils.logger import get_logger
from app.services.embedding_service import EmbeddingService

logger = get_logger(__name__)


@dataclass
class ChunkMetadata:
    """文本块元数据"""
    chunk_id: str
    start_pos: int
    end_pos: int
    length: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    language: Optional[str] = None
    quality_score: float = 0.0
    chunk_type: str = "text"
    section_title: Optional[str] = None
    is_duplicate: bool = False
    similarity_hash: Optional[str] = None


@dataclass
class TextStats:
    """文本统计信息"""
    total_length: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    line_count: int
    language: Optional[str] = None
    encoding: str = "utf-8"
    has_special_chars: bool = False
    readability_score: float = 0.0


class TextSplitter:
    """
    文本分块器
    
    实现智能文本分块算法，考虑语义边界和文档结构。
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        respect_sentence_boundary: bool = True,
        respect_paragraph_boundary: bool = True
    ):
        """
        初始化文本分块器
        
        Args:
            chunk_size: 目标分块大小（字符数）
            chunk_overlap: 分块重叠大小
            min_chunk_size: 最小分块大小
            max_chunk_size: 最大分块大小
            respect_sentence_boundary: 是否尊重句子边界
            respect_paragraph_boundary: 是否尊重段落边界
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.respect_sentence_boundary = respect_sentence_boundary
        self.respect_paragraph_boundary = respect_paragraph_boundary
        
        # 句子分割正则表达式
        self.sentence_pattern = re.compile(r'[.!?。！？]+\s*')
        # 段落分割正则表达式
        self.paragraph_pattern = re.compile(r'\n\s*\n')
    
    def split_text(self, text: str, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        分割文本为块
        
        Args:
            text: 输入文本
            document_id: 文档ID
            
        Returns:
            文本块列表
            
        Raises:
            TextChunkingError: 分块失败
        """
        try:
            if not text or not text.strip():
                return []
            
            # 预处理文本
            cleaned_text = self._preprocess_text(text)
            
            # 检测语言
            language = self._detect_language(cleaned_text)
            
            # 根据策略分块
            if self.respect_paragraph_boundary:
                chunks = self._split_by_paragraphs(cleaned_text)
            else:
                chunks = self._split_by_sentences(cleaned_text)
            
            # 生成块元数据
            chunk_list = []
            for i, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) < self.min_chunk_size:
                    continue
                
                chunk_id = f"{document_id}_chunk_{i}" if document_id else f"chunk_{i}"
                metadata = self._generate_chunk_metadata(
                    chunk_id, chunk_text, i, language
                )
                
                chunk_list.append({
                    "id": chunk_id,
                    "text": chunk_text.strip(),
                    "metadata": metadata.__dict__,
                    "document_id": document_id
                })
            
            logger.info(f"文本分块完成: {len(chunk_list)} 个块")
            return chunk_list
            
        except Exception as e:
            logger.error(f"文本分块失败: {str(e)}")
            raise TextChunkingError(f"文本分块失败: {str(e)}")
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 标准化空白字符
        text = re.sub(r'\s+', ' ', text)
        # 移除多余的换行符
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        return text.strip()
    
    def _detect_language(self, text: str) -> Optional[str]:
        """检测文本语言"""
        try:
            if len(text) < 50:  # 文本太短，无法准确检测
                return None
            return langdetect.detect(text)
        except LangDetectException:
            return None
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """按段落分割文本"""
        paragraphs = self.paragraph_pattern.split(text)
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # 如果当前块加上新段落超过最大大小，保存当前块
            if (len(current_chunk) + len(paragraph) > self.max_chunk_size and 
                len(current_chunk) >= self.min_chunk_size):
                chunks.append(current_chunk)
                # 保留重叠部分
                current_chunk = self._get_overlap_text(current_chunk) + paragraph
            else:
                current_chunk += ("\n\n" if current_chunk else "") + paragraph
            
            # 如果当前块达到目标大小，尝试在句子边界分割
            if len(current_chunk) >= self.chunk_size:
                split_point = self._find_split_point(current_chunk)
                if split_point > self.min_chunk_size:
                    chunks.append(current_chunk[:split_point])
                    current_chunk = self._get_overlap_text(current_chunk[:split_point]) + current_chunk[split_point:]
        
        # 添加最后一个块
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """按句子分割文本"""
        sentences = self.sentence_pattern.split(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 如果当前块加上新句子超过最大大小，保存当前块
            if (len(current_chunk) + len(sentence) > self.max_chunk_size and 
                len(current_chunk) >= self.min_chunk_size):
                chunks.append(current_chunk)
                current_chunk = self._get_overlap_text(current_chunk) + sentence
            else:
                current_chunk += (" " if current_chunk else "") + sentence
            
            # 如果当前块达到目标大小，保存并开始新块
            if len(current_chunk) >= self.chunk_size:
                chunks.append(current_chunk)
                current_chunk = self._get_overlap_text(current_chunk)
        
        # 添加最后一个块
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _find_split_point(self, text: str) -> int:
        """找到最佳分割点"""
        if not self.respect_sentence_boundary:
            return self.chunk_size
        
        # 在目标大小附近寻找句子边界
        search_start = max(self.chunk_size - 200, self.min_chunk_size)
        search_end = min(self.chunk_size + 200, len(text))
        
        # 寻找句子结束标点
        for i in range(search_end - 1, search_start - 1, -1):
            if text[i] in '.!?。！？':
                return i + 1
        
        # 如果没找到句子边界，返回目标大小
        return self.chunk_size
    
    def _get_overlap_text(self, text: str) -> str:
        """获取重叠文本"""
        if len(text) <= self.chunk_overlap:
            return text
        
        # 从末尾开始寻找句子边界
        overlap_text = text[-self.chunk_overlap:]
        if self.respect_sentence_boundary:
            # 寻找句子开始位置
            for i, char in enumerate(overlap_text):
                if char in '.!?。！？' and i < len(overlap_text) - 1:
                    return overlap_text[i + 1:].strip()
        
        return overlap_text
    
    def _generate_chunk_metadata(
        self, 
        chunk_id: str, 
        text: str, 
        index: int, 
        language: Optional[str]
    ) -> ChunkMetadata:
        """生成块元数据"""
        word_count = len(text.split())
        sentence_count = len(self.sentence_pattern.split(text))
        paragraph_count = len(self.paragraph_pattern.split(text))
        
        # 计算质量评分
        quality_score = self._calculate_quality_score(text)
        
        # 生成相似性哈希
        similarity_hash = self._generate_similarity_hash(text)
        
        return ChunkMetadata(
            chunk_id=chunk_id,
            start_pos=0,  # 需要在上层计算
            end_pos=len(text),
            length=len(text),
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            language=language,
            quality_score=quality_score,
            similarity_hash=similarity_hash
        )
    
    def _calculate_quality_score(self, text: str) -> float:
        """计算文本质量评分"""
        score = 1.0
        
        # 长度评分
        if len(text) < self.min_chunk_size:
            score *= 0.5
        elif len(text) > self.max_chunk_size:
            score *= 0.8
        
        # 内容质量评分
        word_count = len(text.split())
        if word_count < 10:
            score *= 0.6
        
        # 特殊字符比例
        special_char_ratio = len(re.findall(r'[^\w\s\u4e00-\u9fff]', text)) / len(text)
        if special_char_ratio > 0.3:
            score *= 0.7
        
        return min(score, 1.0)
    
    def _generate_similarity_hash(self, text: str) -> str:
        """生成相似性哈希"""
        # 标准化文本
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        # 生成MD5哈希
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()


class TextDeduplicator:
    """
    文本去重器
    
    检测和处理重复文本内容。
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        初始化去重器
        
        Args:
            similarity_threshold: 相似度阈值
        """
        self.similarity_threshold = similarity_threshold
        self.seen_hashes: Set[str] = set()
        self.content_cache: Dict[str, str] = {}
    
    def is_duplicate(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        检查文本是否重复
        
        Args:
            text: 输入文本
            
        Returns:
            (是否重复, 相似文本哈希)
        """
        try:
            # 生成文本哈希
            text_hash = self._generate_hash(text)
            
            # 精确匹配检查
            if text_hash in self.seen_hashes:
                return True, text_hash
            
            # 相似度检查
            for existing_hash in self.seen_hashes:
                if existing_hash in self.content_cache:
                    similarity = self._calculate_similarity(
                        text, self.content_cache[existing_hash]
                    )
                    if similarity >= self.similarity_threshold:
                        return True, existing_hash
            
            # 记录新文本
            self.seen_hashes.add(text_hash)
            self.content_cache[text_hash] = text
            
            return False, None
            
        except Exception as e:
            logger.error(f"去重检查失败: {str(e)}")
            return False, None
    
    def _generate_hash(self, text: str) -> str:
        """生成文本哈希"""
        # 标准化文本
        normalized = self._normalize_text(text)
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def _normalize_text(self, text: str) -> str:
        """标准化文本"""
        # 转换为小写
        text = text.lower()
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        # 移除标点符号
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        return text.strip()
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        # 简单的Jaccard相似度
        words1 = set(self._normalize_text(text1).split())
        words2 = set(self._normalize_text(text2).split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def clear_cache(self):
        """清空缓存"""
        self.seen_hashes.clear()
        self.content_cache.clear()


class TextCleaner:
    """
    文本清理器
    
    清理和标准化文本内容。
    """
    
    def __init__(self):
        """初始化文本清理器"""
        # 常见的噪声模式
        self.noise_patterns = [
            r'\s*\n\s*\n\s*\n+',  # 多余的空行
            r'[^\S\n]+',  # 多余的空白字符（保留换行）
            r'[\u200b-\u200d\ufeff]',  # 零宽字符
            r'[\u2000-\u206f]',  # 通用标点符号
        ]
    
    def clean_text(self, text: str) -> str:
        """
        清理文本
        
        Args:
            text: 输入文本
            
        Returns:
            清理后的文本
            
        Raises:
            TextCleaningError: 清理失败
        """
        try:
            if not text:
                return ""
            
            # Unicode标准化
            text = unicodedata.normalize('NFKC', text)
            
            # 移除噪声
            for pattern in self.noise_patterns:
                text = re.sub(pattern, ' ', text)
            
            # 标准化空白字符
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n\s*\n', '\n\n', text)
            
            # 移除首尾空白
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"文本清理失败: {str(e)}")
            raise TextCleaningError(f"文本清理失败: {str(e)}")
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """
        提取文本元数据
        
        Args:
            text: 输入文本
            
        Returns:
            元数据字典
        """
        try:
            stats = self._calculate_stats(text)
            
            return {
                "length": stats.total_length,
                "word_count": stats.word_count,
                "sentence_count": stats.sentence_count,
                "paragraph_count": stats.paragraph_count,
                "line_count": stats.line_count,
                "language": stats.language,
                "encoding": stats.encoding,
                "has_special_chars": stats.has_special_chars,
                "readability_score": stats.readability_score
            }
            
        except Exception as e:
            logger.error(f"元数据提取失败: {str(e)}")
            return {}
    
    def _calculate_stats(self, text: str) -> TextStats:
        """计算文本统计信息"""
        # 基本统计
        total_length = len(text)
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?。！？]+', text))
        paragraph_count = len(re.split(r'\n\s*\n', text))
        line_count = len(text.split('\n'))
        
        # 语言检测
        language = None
        try:
            if len(text) > 50:
                language = langdetect.detect(text)
        except LangDetectException:
            pass
        
        # 特殊字符检测
        has_special_chars = bool(re.search(r'[^\w\s\u4e00-\u9fff]', text))
        
        # 简单的可读性评分
        readability_score = self._calculate_readability(text, word_count, sentence_count)
        
        return TextStats(
            total_length=total_length,
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            line_count=line_count,
            language=language,
            has_special_chars=has_special_chars,
            readability_score=readability_score
        )
    
    def _calculate_readability(self, text: str, word_count: int, sentence_count: int) -> float:
        """计算可读性评分"""
        if sentence_count == 0:
            return 0.0
        
        avg_sentence_length = word_count / sentence_count
        
        # 简化的可读性评分
        if avg_sentence_length < 10:
            return 0.9
        elif avg_sentence_length < 20:
            return 0.7
        elif avg_sentence_length < 30:
            return 0.5
        else:
            return 0.3


class TextService:
    """
    文本处理服务
    
    提供文本分块、去重、清理和向量化的统一接口。
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        enable_deduplication: bool = True,
        similarity_threshold: float = 0.85,
        enable_cleaning: bool = True,
        enable_embedding: bool = True,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        初始化文本处理服务
        
        Args:
            chunk_size: 分块大小
            chunk_overlap: 分块重叠
            min_chunk_size: 最小分块大小
            enable_deduplication: 是否启用去重
            similarity_threshold: 相似度阈值
            enable_cleaning: 是否启用清理
            enable_embedding: 是否启用向量化
            embedding_model: 嵌入模型
        """
        # 初始化组件
        self.splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size
        )
        
        self.deduplicator = TextDeduplicator(
            similarity_threshold=similarity_threshold
        ) if enable_deduplication else None
        
        self.cleaner = TextCleaner() if enable_cleaning else None
        
        # 初始化嵌入服务
        self.embedding_service = EmbeddingService(
            model=embedding_model
        ) if enable_embedding else None
        
        # 配置参数
        self.enable_deduplication = enable_deduplication
        self.enable_cleaning = enable_cleaning
        self.enable_embedding = enable_embedding
        
        logger.info(f"文本处理服务初始化完成 - 分块: {chunk_size}, 去重: {enable_deduplication}, 清理: {enable_cleaning}, 向量化: {enable_embedding}")
    
    async def process_text(
        self,
        text: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        generate_embeddings: bool = True
    ) -> List[Dict[str, Any]]:
        """
        处理文本（完整流程）
        
        Args:
            text: 输入文本
            document_id: 文档ID
            metadata: 元数据
            generate_embeddings: 是否生成嵌入
            
        Returns:
            处理后的文本块列表
            
        Raises:
            TextProcessingError: 文本处理失败
        """
        try:
            logger.info(f"开始处理文本 - 长度: {len(text)}, 文档ID: {document_id}")
            
            # 1. 文本清理
            if self.enable_cleaning and self.cleaner:
                text = await self.cleaner.clean_text(text)
                logger.debug("文本清理完成")
            
            # 2. 文本分块
            chunks = await self.splitter.split_text(text, metadata or {})
            logger.info(f"文本分块完成 - 生成 {len(chunks)} 个块")
            
            # 3. 文本去重
            if self.enable_deduplication and self.deduplicator and len(chunks) > 1:
                original_count = len(chunks)
                chunks = await self.deduplicator.deduplicate_chunks(chunks)
                logger.info(f"文本去重完成 - 从 {original_count} 个块减少到 {len(chunks)} 个块")
            
            # 4. 生成嵌入向量
            if generate_embeddings and self.enable_embedding and self.embedding_service:
                chunk_texts = [chunk.content for chunk in chunks]
                embeddings = await self.embedding_service.embed_texts(chunk_texts)
                
                # 将嵌入添加到块中
                for chunk, embedding in zip(chunks, embeddings):
                    chunk.embedding = embedding
                
                logger.info(f"嵌入生成完成 - {len(embeddings)} 个向量")
            
            # 5. 构建结果
            results = []
            for i, chunk in enumerate(chunks):
                result = {
                    "content": chunk.content,
                    "extra_data": chunk.metadata,
                    "chunk_index": i,
                    "document_id": document_id,
                    "token_count": chunk.token_count,
                    "char_count": len(chunk.content),
                    "language": chunk.language,
                    "quality_score": chunk.quality_score,
                    "content_hash": chunk.content_hash,
                    "created_at": datetime.utcnow().isoformat()
                }
                
                # 添加嵌入（如果存在）
                if hasattr(chunk, 'embedding') and chunk.embedding:
                    result["embedding"] = chunk.embedding
                    result["embedding_model"] = self.embedding_service.model if self.embedding_service else None
                
                results.append(result)
            
            logger.info(f"文本处理完成 - 输出 {len(results)} 个处理后的块")
            return results
            
        except Exception as e:
            logger.error(f"文本处理失败: {str(e)}")
            raise TextProcessingError(f"文本处理失败: {str(e)}")
    
    async def split_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        仅进行文本分块
        
        Args:
            text: 输入文本
            metadata: 元数据
            
        Returns:
            文本块列表
        """
        try:
            chunks = await self.splitter.split_text(text, metadata or {})
            
            results = []
            for i, chunk in enumerate(chunks):
                results.append({
                    "content": chunk.content,
                    "extra_data": chunk.metadata,
                    "chunk_index": i,
                    "token_count": chunk.token_count,
                    "char_count": len(chunk.content),
                    "language": chunk.language,
                    "quality_score": chunk.quality_score,
                    "content_hash": chunk.content_hash
                })
            
            return results
            
        except Exception as e:
            logger.error(f"文本分块失败: {str(e)}")
            raise TextChunkingError(f"文本分块失败: {str(e)}")
    
    async def clean_text(self, text: str) -> str:
        """
        仅进行文本清理
        
        Args:
            text: 输入文本
            
        Returns:
            清理后的文本
        """
        if not self.enable_cleaning or not self.cleaner:
            return text
        
        return await self.cleaner.clean_text(text)
    
    async def deduplicate_texts(self, texts: List[str]) -> List[str]:
        """
        仅进行文本去重
        
        Args:
            texts: 文本列表
            
        Returns:
            去重后的文本列表
        """
        if not self.enable_deduplication or not self.deduplicator:
            return texts
        
        # 创建临时块对象
        chunks = []
        for i, text in enumerate(texts):
            chunk = TextChunk(
                content=text,
                start_pos=0,
                end_pos=len(text),
                metadata={"index": i},
                token_count=len(text.split()),
                language="unknown",
                quality_score=1.0,
                content_hash=hashlib.sha256(text.encode('utf-8')).hexdigest()
            )
            chunks.append(chunk)
        
        # 去重
        deduplicated_chunks = await self.deduplicator.deduplicate_chunks(chunks)
        
        return [chunk.content for chunk in deduplicated_chunks]
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        仅生成嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        if not self.enable_embedding or not self.embedding_service:
            raise TextProcessingError("嵌入服务未启用")
        
        return await self.embedding_service.embed_texts(texts)
    
    async def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            相似度分数
        """
        if not self.enable_embedding or not self.embedding_service:
            # 使用简单的字符串相似度
            if self.deduplicator:
                return self.deduplicator._calculate_similarity(text1, text2)
            else:
                return 0.0
        
        # 使用嵌入相似度
        embeddings = await self.embedding_service.embed_texts([text1, text2])
        return await self.embedding_service.calculate_similarity(embeddings[0], embeddings[1])
    
    async def find_similar_chunks(
        self,
        query_text: str,
        candidate_chunks: List[Dict[str, Any]],
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        查找相似文本块
        
        Args:
            query_text: 查询文本
            candidate_chunks: 候选文本块
            top_k: 返回前K个结果
            threshold: 相似度阈值
            
        Returns:
            相似文本块列表 (块, 相似度)
        """
        if not self.enable_embedding or not self.embedding_service:
            raise TextProcessingError("嵌入服务未启用")
        
        try:
            # 生成查询嵌入
            query_embedding = await self.embedding_service.embed_text(query_text)
            
            # 准备候选嵌入
            candidate_embeddings = []
            for chunk in candidate_chunks:
                if "embedding" in chunk and chunk["embedding"]:
                    candidate_embeddings.append((chunk, chunk["embedding"]))
                else:
                    # 如果没有嵌入，现场生成
                    embedding = await self.embedding_service.embed_text(chunk["content"])
                    candidate_embeddings.append((chunk, embedding))
            
            # 计算相似度
            similarities = []
            for chunk, embedding in candidate_embeddings:
                similarity = await self.embedding_service.calculate_similarity(
                    query_embedding, embedding
                )
                if similarity >= threshold:
                    similarities.append((chunk, similarity))
            
            # 排序并返回前K个
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"相似块查找失败: {str(e)}")
            raise TextProcessingError(f"相似块查找失败: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        stats = {
            "splitter_stats": self.splitter.get_stats() if self.splitter else {},
            "deduplicator_stats": self.deduplicator.get_stats() if self.deduplicator else {},
            "cleaner_stats": self.cleaner.get_stats() if self.cleaner else {},
            "embedding_stats": self.embedding_service.get_stats() if self.embedding_service else {},
            "configuration": {
                "enable_deduplication": self.enable_deduplication,
                "enable_cleaning": self.enable_cleaning,
                "enable_embedding": self.enable_embedding
            }
        }
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            health_status = {
                "status": "healthy",
                "components": {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # 检查分块器
            try:
                test_text = "这是一个测试文本，用于检查文本分块功能是否正常工作。"
                chunks = await self.splitter.split_text(test_text, {})
                health_status["components"]["splitter"] = {
                    "status": "healthy",
                    "test_chunks": len(chunks)
                }
            except Exception as e:
                health_status["components"]["splitter"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
            
            # 检查去重器
            if self.deduplicator:
                health_status["components"]["deduplicator"] = {"status": "healthy"}
            
            # 检查清理器
            if self.cleaner:
                health_status["components"]["cleaner"] = {"status": "healthy"}
            
            # 检查嵌入服务
            if self.embedding_service:
                embedding_health = await self.embedding_service.health_check()
                health_status["components"]["embedding"] = embedding_health
                if embedding_health["status"] != "healthy":
                    health_status["status"] = "degraded"
            
            return health_status
            
        except Exception as e:
            logger.error(f"健康检查失败: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def clear_cache(self):
        """清空缓存"""
        self.deduplicator.clear_cache()