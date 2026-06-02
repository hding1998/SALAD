#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载与预处理工具
支持实验 A/B/C 的数据准备和验证
"""

import os
import sys
import json
import argparse
from typing import List, Dict
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def validate_qa_dataset(filepath: str) -> Dict:
    """验证实验 A 题库格式和统计分布"""
    print(f"验证题库: {filepath}")

    if not os.path.exists(filepath):
        print(f"[ERROR] 文件不存在: {filepath}")
        return {}

    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    print(f"总题数: {total}")

    # 类型分布
    type_dist = Counter(r.get('type', 'unknown') for r in records)
    print(f"题型分布: {dict(type_dist)}")

    # 知识领域分布
    cat_dist = Counter(r.get('category', 'unknown') for r in records)
    print(f"领域分布: {dict(cat_dist)}")

    # 难度分布
    diff_dist = Counter(r.get('difficulty', 'unknown') for r in records)
    print(f"难度分布: {dict(diff_dist)}")

    # ID 唯一性
    ids = [r.get('id') for r in records]
    if len(ids) != len(set(ids)):
        print("[WARN] 存在重复 ID")
    else:
        print("ID 唯一性: OK")

    # 检查必填字段
    required_fields = {
        'single_choice': ['id', 'type', 'question', 'options', 'answer', 'difficulty', 'category'],
        'multiple_choice': ['id', 'type', 'question', 'options', 'answer', 'difficulty', 'category'],
        'judgment': ['id', 'type', 'question', 'answer', 'difficulty', 'category'],
        'short_answer': ['id', 'type', 'question', 'answer', 'difficulty', 'category', 'keywords'],
    }

    missing_count = 0
    for r in records:
        qtype = r.get('type', '')
        fields = required_fields.get(qtype, [])
        for f in fields:
            if f not in r:
                missing_count += 1
                print(f"[MISSING] 题目 {r.get('id')} 缺少字段: {f}")

    if missing_count == 0:
        print("字段完整性: OK")
    else:
        print(f"字段缺失: {missing_count} 处")

    return {
        'total': total,
        'by_type': dict(type_dist),
        'by_category': dict(cat_dist),
        'by_difficulty': dict(diff_dist),
    }


def build_qa_dataset(output_path: str, num_questions: int = 100):
    """
    生成示例题库（若用户无现成题库，可作为模板使用）
    注意：实际题库应由领域专家手动编制或从公开教材改编
    """
    print(f"生成示例题库: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sample_questions = [
        {
            "id": "A001",
            "type": "single_choice",
            "question": "在电力系统潮流计算中，PQ节点是指",
            "options": ["有功功率和电压幅值给定的节点", "有功功率和无功功率给定的节点", "电压幅值和相角给定的节点", "无功功率和电压相角给定的节点"],
            "answer": "B",
            "difficulty": "medium",
            "category": "系统分析",
            "score_criteria": "选对得10分，选错不得分"
        },
        {
            "id": "A002",
            "type": "single_choice",
            "question": "变压器空载试验主要用于测定",
            "options": ["铜损", "铁损", "短路阻抗", "绕组电阻"],
            "answer": "B",
            "difficulty": "easy",
            "category": "系统分析",
            "score_criteria": "选对得10分，选错不得分"
        },
        {
            "id": "A041",
            "type": "multiple_choice",
            "question": "下列哪些措施可以提高电力系统暂态稳定性",
            "options": ["快速切除故障", "采用自动重合闸", "安装静止无功补偿器", "减小发电机出力", "使用强行励磁"],
            "answer": ["A", "B", "C", "E"],
            "difficulty": "hard",
            "category": "系统分析",
            "score_criteria": "每选项2.5分，漏选无错选得5分，全对得10分"
        },
        {
            "id": "A061",
            "type": "judgment",
            "question": "变压器空载运行时，一次侧电流全部为励磁电流。",
            "answer": "正确",
            "difficulty": "easy",
            "category": "系统分析",
            "score_criteria": "判断正确得10分"
        },
        {
            "id": "A081",
            "type": "short_answer",
            "question": "简述距离保护的基本原理及影响其测量精度的主要因素。",
            "answer": "距离保护通过测量故障点到保护安装处的阻抗来判断故障位置...",
            "difficulty": "medium",
            "category": "继电保护",
            "keywords": ["阻抗测量", "故障距离", "过渡电阻", "互感器误差", "系统振荡"],
            "score_criteria": {"总分": 10, "阻抗测量原理": 4, "影响因素": 6}
        },
    ]

    with open(output_path, 'w', encoding='utf-8') as f:
        for q in sample_questions:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')

    print(f"已生成 {len(sample_questions)} 道示例题目")
    print(f"请根据实际需求扩展至 {num_questions} 题")


def validate_ts_data(uci_path: str = None, ett_path: str = None):
    """验证实验 B 时序数据"""
    import pandas as pd

    if uci_path and os.path.exists(uci_path):
        print(f"\n验证 UCI 数据: {uci_path}")
        df = pd.read_csv(uci_path)
        print(f"  形状: {df.shape}")
        print(f"  列名: {list(df.columns)}")
        print(f"  时间范围: {df.index.min()} ~ {df.index.max()}")
        print(f"  缺失值: {df.isnull().sum().sum()}")

    if ett_path and os.path.exists(ett_path):
        print(f"\n验证 ETT 数据: {ett_path}")
        df = pd.read_csv(ett_path)
        print(f"  形状: {df.shape}")
        print(f"  列名: {list(df.columns)}")
        print(f"  缺失值: {df.isnull().sum().sum()}")


def validate_power_system(output_dir: str):
    """验证实验 C 系统数据"""
    try:
        import pandapower as pp
        from pandapower.networks import case14, case30, case57, case118
    except ImportError:
        print("[SKIP] pandapower 未安装")
        return

    os.makedirs(output_dir, exist_ok=True)
    systems = {
        'case14': case14(),
        'case30': case30(),
        'case57': case57(),
        'case118': case118(),
    }

    for name, net in systems.items():
        pp.runpp(net)
        print(f"\n{name}: {len(net.bus)}节点, {len(net.line)}线路, {len(net.gen)}发电机")
        print(f"  收敛: {net.converged}, 电压范围: {net.res_bus.vm_pu.min():.4f} ~ {net.res_bus.vm_pu.max():.4f}")

        # 保存文本描述
        from src.run_exp_c import export_system_text
        text = export_system_text(net, name)
        path = os.path.join(output_dir, f"{name}_description.txt")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"  描述文件已保存: {path}")


def main():
    parser = argparse.ArgumentParser(description="数据加载与验证工具")
    parser.add_argument('--task', type=str, required=True,
                        choices=['validate_qa', 'build_qa', 'validate_ts', 'validate_sys'])
    parser.add_argument('--input', type=str, help='输入文件路径')
    parser.add_argument('--output', type=str, help='输出文件路径/目录')
    parser.add_argument('--uci', type=str, help='UCI 数据路径')
    parser.add_argument('--ett', type=str, help='ETT 数据路径')

    args = parser.parse_args()

    if args.task == 'validate_qa':
        validate_qa_dataset(args.input or 'data/power_qa/power_qa_100.jsonl')
    elif args.task == 'build_qa':
        build_qa_dataset(args.output or 'data/power_qa/power_qa_100.jsonl')
    elif args.task == 'validate_ts':
        validate_ts_data(args.uci, args.ett)
    elif args.task == 'validate_sys':
        validate_power_system(args.output or 'data/power_system/')


if __name__ == '__main__':
    main()
