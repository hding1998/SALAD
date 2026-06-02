#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 C 执行脚本（重构版）：电力调度指令结构化解析与影响范围推理
支持四级增强对比，输出结构化JSON

用法：
    python src/run_exp_c.py --model deepseek-r1:1.5b --level 3
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_caller import ModelCaller
from src.rag_engine import get_dispatch_knowledge, RAGEngine
from src.eval_utils import (
    grade_dispatch_parsing,
    summarize_dispatch_results,
    save_json,
    load_jsonl,
)


LEVEL_PROMPTS = {
    0: {
        "name": "裸模型（Zero-Shot）",
        "system": "你是电网调度运行专家。",
        "template": """请将以下调度指令解析为结构化信息。

调度指令：
{command}

请提取以下字段并以JSON输出：
- operation_object: 操作对象（设备名称）
- operation_type: 操作类型（停电/送电/检修/投运/调整等）
- prerequisites: 前置条件
- affected_devices: 可能影响的设备列表
- safety_measures: 需要采取的安全措施

输出JSON：""",
    },
    1: {
        "name": "+ CoT/Few-Shot",
        "system": "你是电网调度运行专家。",
        "template": """请将以下调度指令解析为结构化信息。

【示例】
指令："将220kV甲线由运行转检修，需退出线路保护及重合闸，对侧同步配合"
解析：
{{"operation_object": "220kV甲线",
  "operation_type": "停电（运行转检修）",
  "prerequisites": ["确认负荷已转移", "对侧配合操作"],
  "affected_devices": ["220kV甲线断路器", "线路保护装置", "重合闸装置"],
  "safety_measures": ["退出线路保护跳闸出口压板", "退出重合闸功能", "对侧同步操作"]}}

【待解析指令】
{command}

请参照示例格式，逐步分析后输出JSON。""",
    },
    2: {
        "name": "+ RAG 知识外挂",
        "system": "你是电网调度运行专家，请结合调度规程进行解析。",
        "template": """{rag_context}

【调度指令】
{command}

请结合上述调度规程和操作规范，将指令解析为结构化JSON：
{{"operation_object": "",
  "operation_type": "",
  "prerequisites": [],
  "affected_devices": [],
  "safety_measures": []}}

输出：""",
    },
    3: {
        "name": "+ RAG + 专用模板",
        "system": "你是资深电网调度运行工程师，严格依据调度规程解析操作指令。",
        "template": """{rag_context}

【指令解析任务】
原始指令：{command}

【解析流程】
Step 1：识别操作对象（设备双重名称）
Step 2：判定操作类型及目标状态（运行/热备用/冷备用/检修）
Step 3：检索规程中的操作顺序要求
Step 4：识别受影响的关联设备（保护、安控、对侧等）
Step 5：列出必须执行的安全措施（压板退出、功能闭锁等）
Step 6：评估潜在风险点

【输出格式】
{{"operation_object": "操作对象",
  "operation_type": "操作类型",
  "target_state": "目标状态",
  "prerequisites": ["前置条件1", "前置条件2"],
  "affected_devices": ["影响设备1", "影响设备2"],
  "safety_measures": ["安全措施1", "安全措施2"],
  "risk_notes": "风险说明",
  "regulation_ref": "依据规程条款"}}

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
    """执行单组实验 C"""
    cfg = LEVEL_PROMPTS[level]
    print(f"\n{'='*60}")
    print(f"实验 C: model={model_name}, level={level} ({cfg['name']})")
    print(f"{'='*60}")

    caller = ModelCaller()
    commands = load_jsonl(dataset_path)

    # 预加载RAG
    rag_engine = None
    if level >= 2:
        rag_engine = RAGEngine()
        kb_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'knowledge_base', 'dispatch_regulations.json')
        kb_path = os.path.abspath(kb_path)
        if os.path.exists(kb_path):
            rag_engine.load_knowledge_base(kb_path, 'dispatch')

    results = []
    total_cost = 0.0

    for i, cmd in enumerate(commands):
        text = cmd.get('command', '')

        rag_context = ""
        if level >= 2 and rag_engine and 'dispatch' in rag_engine._cache:
            rag_context = rag_engine.retrieve_and_format(text, 'dispatch', top_k=2)

        prompt = cfg["template"].format(command=text, rag_context=rag_context)

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
            print(f"  [ERROR] 指令 {cmd.get('id')} 调用失败: {e}")
            model_answer = ""

        eval_result = grade_dispatch_parsing(model_answer, cmd.get('correct', {}))
        eval_result['model'] = model_name
        eval_result['level'] = level
        eval_result['level_name'] = cfg['name']
        eval_result['cmd_id'] = cmd.get('id', '')
        eval_result['command'] = text
        eval_result['model_answer'] = model_answer
        results.append(eval_result)

        if (i + 1) % 10 == 0:
            valid_rate = sum(1 for r in results if r.get('json_valid', 0) == 1.0) / len(results)
            avg_score = sum(r['overall_score'] for r in results) / len(results)
            print(f"  进度: {i+1}/{len(commands)}, JSON有效={valid_rate:.1%}, 均分={avg_score:.2f}")

        if caller.MODEL_CONFIGS[model_name]['type'] == 'api':
            time.sleep(0.5)

    summary = summarize_dispatch_results(results)
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
    parser = argparse.ArgumentParser(description="实验 C：电力调度指令结构化解析")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--level', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--dataset', type=str, default='data/power_dispatch/dispatch_cmds_80.jsonl')
    parser.add_argument('--output', type=str, default='output/exp_c_results')
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
