#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 知识检索引擎
支持 bge-m3 向量检索 + 关键词匹配混合检索
为三个实验提供领域知识外挂
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """检索结果"""
    content: str
    source: str
    score: float
    metadata: Dict[str, Any]


class RAGEngine:
    """
    轻量级 RAG 引擎
    优先使用 bge-m3（Ollama本地）做向量检索，失败时回退到关键词匹配
    """

    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host
        self._embed_model = "bge-m3:latest"
        self._cache: Dict[str, List[Dict]] = {}  # 知识库缓存
        self._vectors: Dict[str, np.ndarray] = {}  # 向量缓存

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """调用 Ollama bge-m3 获取文本嵌入向量"""
        try:
            import requests
            resp = requests.post(
                f"{self.ollama_host}/api/embeddings",
                json={"model": self._embed_model, "prompt": text},
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            vec = np.array(data.get("embedding", []), dtype=np.float32)
            if len(vec) == 0:
                return None
            # L2 归一化
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
        except Exception as e:
            print(f"[RAG] embedding 获取失败: {e}")
            return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度（假设已归一化）"""
        return float(np.dot(a, b))

    def load_knowledge_base(self, kb_path: str, kb_name: str):
        """
        加载知识库文件并预计算向量
        kb_path: JSON/JSONL 文件路径
        kb_name: 知识库标识名
        """
        records = []
        if kb_path.endswith('.jsonl'):
            with open(kb_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        else:
            with open(kb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict) and 'entries' in data:
                    records = data['entries']

        self._cache[kb_name] = records

        # 预计算向量
        vectors = []
        for rec in records:
            text = rec.get('text', rec.get('content', json.dumps(rec, ensure_ascii=False)))
            vec = self._get_embedding(text)
            if vec is not None:
                vectors.append(vec)
            else:
                vectors.append(np.zeros(1024, dtype=np.float32))  # bge-m3 默认1024维

        self._vectors[kb_name] = np.stack(vectors)
        print(f"[RAG] 知识库 '{kb_name}' 加载完成: {len(records)} 条记录")

    def vector_search(
        self,
        query: str,
        kb_name: str,
        top_k: int = 3
    ) -> List[RetrievalResult]:
        """向量检索"""
        if kb_name not in self._cache:
            raise ValueError(f"知识库 '{kb_name}' 未加载")

        query_vec = self._get_embedding(query)
        if query_vec is None:
            return []

        doc_vectors = self._vectors[kb_name]
        similarities = np.dot(doc_vectors, query_vec)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            rec = self._cache[kb_name][idx]
            results.append(RetrievalResult(
                content=rec.get('text', rec.get('content', '')),
                source=rec.get('source', kb_name),
                score=float(similarities[idx]),
                metadata=rec.get('metadata', {})
            ))
        return results

    def keyword_search(
        self,
        query: str,
        kb_name: str,
        top_k: int = 3
    ) -> List[RetrievalResult]:
        """关键词匹配检索（回退方案）"""
        if kb_name not in self._cache:
            raise ValueError(f"知识库 '{kb_name}' 未加载")

        query_words = set(query.lower().split())
        scored = []

        for rec in self._cache[kb_name]:
            text = rec.get('text', rec.get('content', '')).lower()
            match_count = sum(1 for w in query_words if w in text)
            score = match_count / max(len(query_words), 1)
            scored.append((score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, rec in scored[:top_k]:
            results.append(RetrievalResult(
                content=rec.get('text', rec.get('content', '')),
                source=rec.get('source', kb_name),
                score=score,
                metadata=rec.get('metadata', {})
            ))
        return results

    def retrieve(
        self,
        query: str,
        kb_name: str,
        top_k: int = 3,
        use_vector: bool = True
    ) -> List[RetrievalResult]:
        """
        统一检索接口：优先向量检索，失败时回退关键词匹配
        返回格式化后的知识片段文本
        """
        if use_vector:
            results = self.vector_search(query, kb_name, top_k)
            if results and results[0].score > 0.3:
                return results

        return self.keyword_search(query, kb_name, top_k)

    def format_context(self, results: List[RetrievalResult]) -> str:
        """将检索结果格式化为 Prompt 可用的上下文文本"""
        if not results:
            return ""
        lines = ["【相关参考资料】"]
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] 来源: {r.source}")
            lines.append(r.content)
            lines.append("")
        return "\n".join(lines)

    def retrieve_and_format(self, query: str, kb_name: str, top_k: int = 3) -> str:
        """一站式检索并格式化"""
        results = self.retrieve(query, kb_name, top_k)
        return self.format_context(results)

    # ============ SALAD: FSSR 过滤式单条RAG ============

    def retrieve_and_format_salad(
        self,
        query: str,
        kb_name: str,
        top_k: int = 1,
        similarity_threshold: float = 0.7,
        max_chars: int = 120,
    ) -> str:
        """
        SALAD FSSR: Filtered Single-Shot RAG
        - 只返回最高相关性的1条结果
        - 低于阈值时返回空字符串（回退到无RAG）
        - 对结果做摘要压缩（只保留前max_chars字符）
        """
        results = self.retrieve(query, kb_name, top_k=top_k, use_vector=True)
        if not results:
            return ""

        best = results[0]
        if best.score < similarity_threshold:
            return ""

        # 摘要压缩：取前max_chars字符，优先保留包含数字/阈值的句子
        content = best.content.strip()
        if len(content) > max_chars:
            # 尝试在句子边界截断
            truncated = content[:max_chars]
            last_period = max(truncated.rfind('。'), truncated.rfind('.'), truncated.rfind(';'))
            if last_period > max_chars * 0.5:
                content = truncated[:last_period + 1]
            else:
                content = truncated + "..."

        return content


# ============ 便捷函数 ============

def get_defect_knowledge(query: str, engine: Optional[RAGEngine] = None) -> str:
    """获取设备缺陷分类相关知识"""
    if engine is None:
        engine = RAGEngine()
    kb_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'knowledge_base', 'defect_standard.json')
    kb_path = os.path.abspath(kb_path)
    if os.path.exists(kb_path) and 'defect' not in engine._cache:
        engine.load_knowledge_base(kb_path, 'defect')
    if 'defect' in engine._cache:
        return engine.retrieve_and_format(query, 'defect', top_k=3)
    return ""


def get_dq_knowledge(query: str, engine: Optional[RAGEngine] = None) -> str:
    """获取数据质量规则相关知识"""
    if engine is None:
        engine = RAGEngine()
    kb_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'knowledge_base', 'dq_rules.json')
    kb_path = os.path.abspath(kb_path)
    if os.path.exists(kb_path) and 'dq_rules' not in engine._cache:
        engine.load_knowledge_base(kb_path, 'dq_rules')
    if 'dq_rules' in engine._cache:
        return engine.retrieve_and_format(query, 'dq_rules', top_k=3)
    return ""


def get_dispatch_knowledge(query: str, engine: Optional[RAGEngine] = None) -> str:
    """获取调度规程相关知识"""
    if engine is None:
        engine = RAGEngine()
    kb_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'knowledge_base', 'dispatch_regulations.json')
    kb_path = os.path.abspath(kb_path)
    if os.path.exists(kb_path) and 'dispatch' not in engine._cache:
        engine.load_knowledge_base(kb_path, 'dispatch')
    if 'dispatch' in engine._cache:
        return engine.retrieve_and_format(query, 'dispatch', top_k=3)
    return ""
