#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 B 执行脚本（重构版）：电力数据质量规则配置与异常标注
支持四级增强对比，输出结构化JSON规则配置

用法：
    python src/run_exp_b.py --model qwen2.5-coder:1.5b --level 3
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_caller import ModelCaller
from src.rag_engine import get_dq_knowledge, RAGEngine
from src.eval_utils import (
    grade_rule_config,
    summarize_rule_results,
    save_json,
    load_jsonl,
)


LEVEL_PROMPTS = {
    0: {
        "name": "裸模型（Zero-Shot）",
        "system": "你是电力数据治理专家。",
        "template": """请为以下数据场景配置数据质量检测规则。

【场景描述】
{scenario_description}

【字段统计特征】
{statistics}

请输出JSON格式的规则配置：
{{"rule_type": "规则类型", "field": "目标字段", "threshold": "阈值", "severity": "一般/严重/危急", "description": "规则说明"}}

只输出JSON，不要额外解释。""",
    },
    1: {
        "name": "+ CoT/Few-Shot",
        "system": "你是电力数据治理专家。",
        "template": """请为以下数据场景配置数据质量检测规则。

【示例1】
场景：变压器油温监测，历史均值65℃，标准差8℃，最大值98℃，最小值22℃
规则：{{"rule_type": "threshold", "field": "oil_temperature", "lower_bound": 20, "upper_bound": 85, "severity": "一般", "description": "油温正常范围20-85℃"}}

【示例2】
场景：线路有功功率，15分钟内多次出现从80MW突降至5MW再恢复
规则：{{"rule_type": "rate_of_change", "field": "active_power", "max_decrease_rate": 50, "window": "15min", "severity": "严重", "description": "15分钟内功率下降超过50%为异常"}}

【待配置场景】
{scenario_description}

【字段统计特征】
{statistics}

请参照示例格式，先分析数据特征，再输出JSON规则配置。""",
    },
    2: {
        "name": "+ RAG 知识外挂",
        "system": "你是电力数据治理专家，请参照数据质量规则标准进行配置。",
        "template": """{rag_context}

【场景描述】
{scenario_description}

【字段统计特征】
{statistics}

请根据上述规则模板和场景特征，选择最合适的规则类型并配置参数，输出JSON格式规则。

规则类型可选：threshold / rate_of_change / consistency / missing_value / flatline / statistical_outlier / pattern

输出：""",
    },
    3: {
        "name": "+ RAG + 专用模板",
        "system": "你是资深电力数据治理工程师，严格依据数据质量规则模板库进行规则配置。",
        "template": """{rag_context}

【配置任务】为以下数据场景设计数据质量检测规则。

【场景描述】
{scenario_description}

【字段统计特征】
{statistics}

【配置流程】
Step 1：分析数据类型和分布特征（连续/离散、正态/偏态、有无周期规律）
Step 2：识别潜在的数据质量问题（越限、突变、缺失、冻结、不一致等）
Step 3：从规则模板库中选择最匹配的1-2条规则类型
Step 4：根据统计特征设定具体阈值参数
Step 5：给出规则严重度等级和说明

【输出格式】
{{"rule_type": "规则类型英文名",
  "field": "字段名",
  "threshold": "具体阈值",
  "severity": "一般/严重/危急",
  "description": "规则描述",
  "rationale": "配置理由"}}

输出：""",
    },
}


def run_single_experiment(
    model_name: str,
    level: int,
    dataset_path: str,
    output_dir: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> Dict:
    """执行单组实验 B"""
    cfg = LEVEL_PROMPTS[level]
    print(f"\n{'='*60}")
    print(f"实验 B: model={model_name}, level={level} ({cfg['name']})")
    print(f"{'='*60}")

    caller = ModelCaller()
    scenarios = load_jsonl(dataset_path)

    # 预加载RAG
    rag_engine = None
    if level >= 2:
        rag_engine = RAGEngine()
        kb_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'knowledge_base', 'dq_rules.json')
        kb_path = os.path.abspath(kb_path)
        if os.path.exists(kb_path):
            rag_engine.load_knowledge_base(kb_path, 'dq_rules')

    results = []
    total_cost = 0.0

    for i, sc in enumerate(scenarios):
        desc = sc.get('description', '')
        stats = sc.get('statistics', '')

        rag_context = ""
        if level >= 2 and rag_engine and 'dq_rules' in rag_engine._cache:
            query = f"{desc} {stats}"
            rag_context = rag_engine.retrieve_and_format(query, 'dq_rules', top_k=2)

        prompt = cfg["template"].format(
            scenario_description=desc,
            statistics=json.dumps(stats, ensure_ascii=False) if isinstance(stats, dict) else stats,
            rag_context=rag_context,
        )

        try:
            resp = caller.call(
                model_name=model_name,
                prompt=prompt,
                temperature=temperature,
                system_prompt=cfg["system"],
                max_tokens=max_tokens,
            )
            model_answer = resp['content']
            total_cost += resp.get('cost_usd', 0)
        except Exception as e:
            print(f"  [ERROR] 场景 {sc.get('id')} 调用失败: {e}")
            model_answer = ""

        eval_result = grade_rule_config(model_answer, {
            "rule_type": sc.get('correct_rule_type', ''),
            "field": sc.get('correct_field', ''),
        })
        eval_result['model'] = model_name
        eval_result['level'] = level
        eval_result['level_name'] = cfg['name']
        eval_result['scenario_id'] = sc.get('id', '')
        eval_result['model_answer'] = model_answer
        results.append(eval_result)

        if (i + 1) % 10 == 0:
            valid_rate = sum(1 for r in results if r.get('json_valid', 0) == 1.0) / len(results)
            avg_score = sum(r['overall_score'] for r in results) / len(results)
            print(f"  进度: {i+1}/{len(scenarios)}, JSON有效={valid_rate:.1%}, 均分={avg_score:.2f}")

        if caller.MODEL_CONFIGS[model_name]['type'] == 'api':
            time.sleep(0.5)

    summary = summarize_rule_results(results)
    summary['model'] = model_name
    summary['level'] = level
    summary['level_name'] = cfg['name']
    summary['total_cost_usd'] = round(total_cost, 6)
    summary['timestamp'] = datetime.now().isoformat()

    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.replace('/', '_').replace(':', '_')
    base_name = f"{safe_name}_L{level}"

    save_json(results, os.path.join(output_dir, f"{base_name}.json"))
    save_json(summary, os.path.join(output_dir, f"{base_name}_metrics.json"))

    print(f"\n实验完成: JSON有效={summary['json_valid_rate']:.1%}, 均分={summary['avg_overall_score']:.2f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="实验 B：电力数据质量规则配置")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--level', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--dataset', type=str, default='data/power_ts/dq_scenarios_50.jsonl')
    parser.add_argument('--output', type=str, default='output/exp_b_results')
    parser.add_argument('--temperature', type=float, default=0.3)
    args = parser.parse_args()

    run_single_experiment(
        model_name=args.model,
        level=args.level,
        dataset_path=args.dataset,
        output_dir=args.output,
        temperature=args.temperature,
    )


if __name__ == '__main__':
    main()
