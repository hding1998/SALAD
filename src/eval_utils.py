#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估工具集（重构版）
支持实验 A/B/C 的评分函数、指标计算和结果汇总
"""

import re
import json
import jieba
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter, defaultdict


# ========================
# 通用工具
# ========================

def save_json(data: Dict, filepath: str):
    """保存 JSON 文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath: str) -> Dict:
    """加载 JSON 文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(filepath: str) -> List[Dict]:
    """加载 JSON Lines 文件"""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_first_choice(text: str, choices: str = "ABCDEF") -> Optional[str]:
    """从文本中提取第一个出现的选项字母"""
    text_upper = text.upper()
    for ch in choices:
        if ch in text_upper:
            return ch
    return None


# ========================
# 实验 A：缺陷分类评估
# ========================

def parse_defect_answer(model_answer: str) -> Dict[str, str]:
    """
    从模型回答中解析缺陷类型和严重程度
    支持多种输出格式：JSON、键值对、纯文本
    """
    result = {"defect_type": "", "severity": ""}
    text = model_answer.strip()

    # 尝试提取 JSON
    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            if "defect_type" in data:
                result["defect_type"] = str(data["defect_type"]).strip()
            if "severity" in data:
                result["severity"] = str(data["severity"]).strip()
            if result["defect_type"] and result["severity"]:
                return result
    except Exception:
        pass

    # 关键词匹配
    text_lower = text.lower()

    # 缺陷类型
    type_keywords = {
        "过热": ["过热", "温度高", "温升", "发热"],
        "绝缘": ["绝缘", "放电", "击穿", "介损", "绝缘电阻"],
        "机械": ["机械", "卡涩", "变形", "破损", "断裂", "松动"],
        "油务": ["油务", "油位", "油色", "漏油", "渗油", "油色谱"],
        # "二次回路" 类别当前数据集中无样本，暂不参与评分
    }
    for dtype, kws in type_keywords.items():
        if any(kw in text_lower for kw in kws):
            result["defect_type"] = dtype
            break

    # 严重程度
    if any(w in text_lower for w in ["危急", "紧急", "critical", "危险"]):
        result["severity"] = "危急"
    elif any(w in text_lower for w in ["严重", "重要", "serious", "major"]):
        result["severity"] = "严重"
    elif any(w in text_lower for w in ["一般", "普通", "轻微", "general", "minor"]):
        result["severity"] = "一般"

    return result


def grade_defect_classification(model_answer: str, correct: Dict[str, str]) -> Dict[str, float]:
    """
    缺陷分类评分
    返回：{type_score, severity_score, overall_score}
    """
    parsed = parse_defect_answer(model_answer)
    type_score = 1.0 if parsed["defect_type"] == correct.get("defect_type", "") else 0.0
    severity_score = 1.0 if parsed["severity"] == correct.get("severity", "") else 0.0
    overall_score = (type_score + severity_score) / 2.0
    return {
        "type_score": type_score,
        "severity_score": severity_score,
        "overall_score": overall_score,
        "parsed": parsed,
    }


def summarize_defect_results(results: List[Dict]) -> Dict:
    """汇总实验 A 缺陷分类结果"""
    total = len(results)
    if total == 0:
        return {}

    type_correct = sum(1 for r in results if r.get("type_score", 0) == 1.0)
    severity_correct = sum(1 for r in results if r.get("severity_score", 0) == 1.0)
    overall_correct = sum(1 for r in results if r.get("overall_score", 0) == 1.0)

    # 按设备类别统计
    by_device = defaultdict(list)
    for r in results:
        dev = r.get("device_type", "unknown")
        by_device[dev].append(r["overall_score"])

    # 按缺陷类型统计
    by_defect = defaultdict(list)
    for r in results:
        dt = r.get("correct_defect_type", "unknown")
        by_defect[dt].append(r["overall_score"])

    return {
        "total": total,
        "type_accuracy": type_correct / total,
        "severity_accuracy": severity_correct / total,
        "overall_accuracy": overall_correct / total,
        "by_device": {k: sum(v) / len(v) for k, v in by_device.items()},
        "by_defect_type": {k: sum(v) / len(v) for k, v in by_defect.items()},
    }


# ========================
# 实验 B：规则配置评估
# ========================

def parse_json_safely(text: str) -> Optional[Dict]:
    """安全地从文本中提取 JSON（SALAD: 使用 json_repair 增强鲁棒性）"""
    # 先找 ```json 代码块
    match = re.search(r'```json\s*(.*?)```', text, re.DOTALL)
    candidate = match.group(1).strip() if match else text

    # 尝试标准解析
    try:
        return json.loads(candidate)
    except Exception:
        pass

    # 再找普通 JSON
    match = re.search(r'\{.*\}', candidate, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    # SALAD: 使用 json_repair 兜底
    try:
        import json_repair
        repaired = json_repair.repair_json(match.group(0) if match else candidate, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
    except Exception:
        pass

    return None


def parse_keyvalue(text: str) -> Dict[str, str]:
    """
    SALAD: 从键值对格式文本中提取字段
    匹配格式：字段名：[值] 或 字段名：值
    """
    result = {}
    # 匹配 字段名：[值] 或 字段名：值 或 字段名=值
    pattern = r'^(?:\s*[-*]*\s*)([^\[\]:：=\n]+)[\s]*[：:=][\s]*\[?([^\]\n]+)\]?'
    for line in text.split('\n'):
        m = re.match(pattern, line.strip())
        if m:
            key = m.group(1).strip()
            val = m.group(2).strip()
            # 清理常见前缀后缀
            key = key.lstrip('【').rstrip('】').strip()
            val = val.strip('"').strip("'")
            if key and val:
                result[key] = val
    return result


def grade_rule_config(model_answer: str, correct: Dict) -> Dict[str, Any]:
    """
    数据质量规则配置评分
    返回：{json_valid, field_coverage, rule_correct, overall_score}
    """
    parsed = parse_json_safely(model_answer)
    if parsed is None:
        return {"json_valid": 0.0, "field_coverage": 0.0, "rule_correct": 0.0, "field_match": 0.0, "overall_score": 0.0, "parsed": None}

    # 检查必需字段
    required_fields = ["rule_type", "field", "severity"]
    field_coverage = sum(1 for f in required_fields if f in parsed and parsed[f]) / len(required_fields)

    # 规则类型正确性
    rule_correct = 1.0 if parsed.get("rule_type") == correct.get("rule_type") else 0.0

    # 字段匹配
    field_match = 1.0 if parsed.get("field") == correct.get("field") else 0.0

    overall = (json_valid := 1.0) * 0.2 + field_coverage * 0.3 + rule_correct * 0.3 + field_match * 0.2
    return {
        "json_valid": 1.0,
        "field_coverage": field_coverage,
        "rule_correct": rule_correct,
        "field_match": field_match,
        "overall_score": overall,
        "parsed": parsed,
    }


def summarize_rule_results(results: List[Dict]) -> Dict:
    """汇总实验 B 规则配置结果"""
    total = len(results)
    if total == 0:
        return {}

    json_valid_rate = sum(1 for r in results if r.get("json_valid", 0) == 1.0) / total
    avg_coverage = sum(r.get("field_coverage", 0) for r in results) / total
    avg_rule_correct = sum(r.get("rule_correct", 0) for r in results) / total
    avg_overall = sum(r.get("overall_score", 0) for r in results) / total

    return {
        "total": total,
        "json_valid_rate": json_valid_rate,
        "avg_field_coverage": avg_coverage,
        "avg_rule_correct": avg_rule_correct,
        "avg_overall_score": avg_overall,
    }


# ========================
# 实验 C：指令解析评估
# ========================

def grade_dispatch_parsing(model_answer: str, correct: Dict) -> Dict[str, Any]:
    """
    调度指令结构化解析评分
    核心字段(operation_object, operation_type)与ground truth精确匹配
    额外字段(prerequisites, affected_devices, safety_measures)按存在性奖励
    """
    parsed = parse_json_safely(model_answer)
    if parsed is None:
        return {"json_valid": 0.0, "obj_score": 0.0, "type_score": 0.0, "bonus_score": 0.0, "overall_score": 0.0, "parsed": None}

    # 核心字段：必须与 ground truth 精确匹配
    obj_score = 1.0 if str(parsed.get("operation_object", "")).strip() == correct.get("operation_object") else 0.0
    type_score = 1.0 if str(parsed.get("operation_type", "")).strip() == correct.get("operation_type") else 0.0

    # 额外字段：存在即奖励（每字段最多0.05，上限0.15）
    bonus_fields = ["prerequisites", "affected_devices", "safety_measures"]
    bonus_score = min(sum(0.05 for f in bonus_fields if f in parsed and parsed[f]), 0.15)

    overall = 0.2 + obj_score * 0.4 + type_score * 0.4 + bonus_score

    return {
        "json_valid": 1.0,
        "obj_score": obj_score,
        "type_score": type_score,
        "bonus_score": bonus_score,
        "overall_score": overall,
        "parsed": parsed,
    }


def summarize_dispatch_results(results: List[Dict]) -> Dict:
    """汇总实验 C 指令解析结果"""
    total = len(results)
    if total == 0:
        return {}

    json_valid_rate = sum(1 for r in results if r.get("json_valid", 0) == 1.0) / total
    avg_obj = sum(r.get("obj_score", 0) for r in results) / total
    avg_type = sum(r.get("type_score", 0) for r in results) / total
    avg_bonus = sum(r.get("bonus_score", 0) for r in results) / total
    avg_overall = sum(r.get("overall_score", 0) for r in results) / total

    return {
        "total": total,
        "json_valid_rate": json_valid_rate,
        "avg_obj_score": avg_obj,
        "avg_type_score": avg_type,
        "avg_bonus_score": avg_bonus,
        "avg_overall_score": avg_overall,
    }


# ========================
# 通用实验汇总
# ========================

def build_comparison_table(all_results: Dict[str, List[Dict]]) -> pd.DataFrame:
    """
    构建跨实验的对比表
    all_results: {'exp_a': [...], 'exp_b': [...], 'exp_c': [...]}
    返回 DataFrame
    """
    import pandas as pd
    rows = []
    for exp_name, results in all_results.items():
        for r in results:
            rows.append({
                "experiment": exp_name,
                "model": r.get("model", ""),
                "enhance_level": r.get("enhance_level", 0),
                "overall_score": r.get("overall_score", 0),
            })
    return pd.DataFrame(rows)
