#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 A 执行脚本（重构版）：电力设备缺陷智能分类与定级
支持四级增强对比：裸模型 / +CoT / +RAG / +RAG+专用模板

用法：
    python src/run_exp_a.py --model qwen2.5:1.5b --level 3
    python src/run_exp_a.py --config configs/exp_a_fast.yaml
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_caller import ModelCaller
from src.rag_engine import get_defect_knowledge, RAGEngine
from src.eval_utils import (
    grade_defect_classification,
    summarize_defect_results,
    save_json,
    load_jsonl,
)


# ============ Prompt 模板 ============

LEVEL_PROMPTS = {
    0: {
        "name": "裸模型（Zero-Shot）",
        "system": "你是电力设备运维专家，请根据设备状态描述判断缺陷类型和严重程度。",
        "template": """请分析以下设备状态描述，判断缺陷类型和严重程度。

设备状态描述：
{description}

要求：
1. 缺陷类型只能从以下类别中选择：过热、绝缘、机械、油务
2. 严重程度只能从以下等级中选择：一般、严重、危急
3. 以JSON格式输出：{{"defect_type": "类型", "severity": "等级", "reason": "简要判断理由"}}

输出：""",
    },
    1: {
        "name": "+ CoT/Few-Shot",
        "system": "你是电力设备运维专家，请根据设备状态描述判断缺陷类型和严重程度。",
        "template": """请分析以下设备状态描述，逐步推理后判断缺陷类型和严重程度。

【示例1】
描述：#2主变顶层油温102℃，环境温度25℃，冷却器运转正常。
分析：油温超过105℃才为危急，102℃处于95-105℃严重缺陷区间；冷却器正常排除冷却故障；温升77K超过55K阈值。
结论：{{"defect_type": "过热", "severity": "严重", "reason": "顶层油温102℃处于严重缺陷区间"}}

【示例2】
描述：110kV线路#15塔A相绝缘子伞裙脱落3片，芯棒未外露。
分析：复合绝缘子伞裙脱落2片及以上为严重缺陷；芯棒未外露排除危急；未提及放电或击穿迹象。
结论：{{"defect_type": "绝缘", "severity": "严重", "reason": "复合绝缘子伞裙脱落3片"}}

【待分析描述】
{description}

要求：
1. 缺陷类型：过热、绝缘、机械、油务
2. 严重程度：一般、严重、危急
3. 先分析关键现象，再给出结论
4. 以JSON格式输出

输出：""",
    },
    2: {
        "name": "+ RAG 知识外挂",
        "system": "你是电力设备运维专家，请结合缺陷分类标准分析设备状态。",
        "template": """{rag_context}

请根据上述缺陷分类标准和以下设备状态描述，判断缺陷类型和严重程度。

设备状态描述：
{description}

要求：
1. 缺陷类型：过热、绝缘、机械、油务
2. 严重程度：一般、严重、危急
3. 参照标准中的具体数值阈值进行判定
4. 以JSON格式输出：{{"defect_type": "类型", "severity": "等级", "reason": "判定依据"}}

输出：""",
    },
    3: {
        "name": "+ RAG + 专用模板",
        "system": "你是资深电力设备状态评价工程师，严格依据Q/GDW 1906缺陷分类标准进行判定。",
        "template": """{rag_context}

【任务】对以下设备状态进行缺陷分类与定级。

【判定流程】
Step 1：识别设备类型和关键异常现象
Step 2：对照标准中的定量阈值（温度/油色谱/绝缘电阻等）
Step 3：若定量指标未明确，按定性描述和工程经验判定
Step 4：给出最终缺陷类型和严重程度等级

【设备状态描述】
{description}

【输出格式】
{{"defect_type": "从{{过热/绝缘/机械/油务}}中选择",
  "severity": "从{{一般/严重/危急}}中选择",
  "reason": "引用标准具体条款作为判定依据",
  "confidence": "高/中/低"}}

输出：""",
    },
}


def build_prompt(description: str, level: int, rag_context: str = "") -> Tuple[str, str]:
    """根据增强等级构建 prompt"""
    cfg = LEVEL_PROMPTS[level]
    prompt = cfg["template"].format(description=description, rag_context=rag_context)
    return prompt, cfg["system"]


def run_single_experiment(
    model_name: str,
    level: int,
    dataset_path: str,
    output_dir: str,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> Dict:
    """执行单组实验"""
    cfg = LEVEL_PROMPTS[level]
    print(f"\n{'='*60}")
    print(f"实验 A: model={model_name}, level={level} ({cfg['name']}), temp={temperature}")
    print(f"{'='*60}")

    caller = ModelCaller()
    questions = load_jsonl(dataset_path)

    # 预加载RAG（Level 2/3需要）
    rag_engine = None
    if level >= 2:
        rag_engine = RAGEngine()
        kb_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'knowledge_base', 'defect_standard.json')
        kb_path = os.path.abspath(kb_path)
        if os.path.exists(kb_path):
            rag_engine.load_knowledge_base(kb_path, 'defect')

    results = []
    total_cost = 0.0
    total_tokens = 0

    for i, q in enumerate(questions):
        desc = q.get('description', q.get('question', ''))

        # 构建RAG上下文
        rag_context = ""
        if level >= 2 and rag_engine and 'defect' in rag_engine._cache:
            rag_context = rag_engine.retrieve_and_format(desc, 'defect', top_k=2)

        prompt, system = build_prompt(desc, level, rag_context)

        try:
            resp = caller.call(
                model_name=model_name,
                prompt=prompt,
                temperature=temperature,
                system_prompt=system,
                max_tokens=max_tokens,
            )
            model_answer = resp['content']
            total_cost += resp.get('cost_usd', 0)
            total_tokens += resp.get('total_tokens', 0)
        except Exception as e:
            print(f"  [ERROR] 样本 {q.get('id')} 调用失败: {e}")
            model_answer = ""

        eval_result = grade_defect_classification(model_answer, {
            "defect_type": q.get('defect_type', ''),
            "severity": q.get('severity', '')
        })
        eval_result['model'] = model_name
        eval_result['level'] = level
        eval_result['level_name'] = cfg['name']
        eval_result['sample_id'] = q.get('id', '')
        eval_result['device_type'] = q.get('device_type', '')
        eval_result['correct_defect_type'] = q.get('defect_type', '')
        eval_result['correct_severity'] = q.get('severity', '')
        eval_result['model_answer'] = model_answer
        eval_result['description'] = desc
        results.append(eval_result)

        if (i + 1) % 20 == 0:
            acc_so_far = sum(r['overall_score'] for r in results) / len(results)
            print(f"  进度: {i+1}/{len(questions)}, 当前准确率: {acc_so_far:.2%}")

        # API 限流保护
        if caller.MODEL_CONFIGS[model_name]['type'] == 'api':
            time.sleep(0.5)

    summary = summarize_defect_results(results)
    summary['model'] = model_name
    summary['level'] = level
    summary['level_name'] = cfg['name']
    summary['temperature'] = temperature
    summary['total_cost_usd'] = round(total_cost, 6)
    summary['total_tokens'] = total_tokens
    summary['timestamp'] = datetime.now().isoformat()

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.replace('/', '_').replace(':', '_')
    base_name = f"{safe_name}_L{level}"

    raw_path = os.path.join(output_dir, f"{base_name}.json")
    save_json(results, raw_path)

    metric_path = os.path.join(output_dir, f"{base_name}_metrics.json")
    save_json(summary, metric_path)

    print(f"\n实验完成: 总体准确率={summary['overall_accuracy']:.2%}, 类型准确率={summary['type_accuracy']:.2%}, 严重度准确率={summary['severity_accuracy']:.2%}")
    print(f"结果保存: {raw_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="实验 A：电力设备缺陷智能分类")
    parser.add_argument('--model', type=str, help='模型名称')
    parser.add_argument('--level', type=int, default=0, choices=[0, 1, 2, 3], help='增强等级')
    parser.add_argument('--dataset', type=str, default='data/power_qa/defect_classification_150.jsonl', help='数据集路径')
    parser.add_argument('--output', type=str, default='output/exp_a_results', help='输出目录')
    parser.add_argument('--config', type=str, help='YAML 配置文件（批量实验）')
    parser.add_argument('--temperature', type=float, default=0.3)

    args = parser.parse_args()

    if args.config:
        import yaml
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        models = config.get('models', [])
        levels = config.get('levels', [0, 1, 2, 3])
        dataset = config.get('dataset', args.dataset)
        output = config.get('output_dir', args.output)

        all_summaries = []
        for model in models:
            for level in levels:
                try:
                    summary = run_single_experiment(
                        model_name=model,
                        level=level,
                        dataset_path=dataset,
                        output_dir=output,
                        temperature=args.temperature,
                    )
                    all_summaries.append(summary)
                except Exception as e:
                    print(f"[FAIL] model={model}, level={level}: {e}")

        summary_path = os.path.join(output, 'summary_report.json')
        save_json(all_summaries, summary_path)
        print(f"\n全部实验汇总已保存: {summary_path}")
    else:
        if not args.model:
            print("错误：请指定 --model 或 --config")
            parser.print_help()
            sys.exit(1)
        run_single_experiment(
            model_name=args.model,
            level=args.level,
            dataset_path=args.dataset,
            output_dir=args.output,
            temperature=args.temperature,
        )


if __name__ == '__main__':
    main()
