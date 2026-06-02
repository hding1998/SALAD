#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果分析与可视化入口
集成 paper_viz 生成论文级图表
"""

import os
import sys
import json
import glob
import argparse
from collections import defaultdict
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paper_viz import generate_all_plots


def load_exp_metrics(results_dir: str) -> List[Dict]:
    """加载某目录下所有 metrics 文件"""
    pattern = os.path.join(results_dir, "*_metrics.json")
    files = glob.glob(pattern)
    summaries = []
    for f in files:
        with open(f, 'r', encoding='utf-8') as fp:
            summaries.append(json.load(fp))
    return summaries


def print_summary_table(summaries: List[Dict], title: str):
    """打印文本汇总表"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(f"{'模型':<25} {'等级':<18} {'准确率':>10} {'成本($)':>10} {'Token':>10}")
    print("-" * 80)
    for s in sorted(summaries, key=lambda x: (x.get('model', ''), x.get('level', 0))):
        acc = s.get('overall_accuracy', s.get('avg_overall_score', 0))
        cost = s.get('total_cost_usd', 0)
        tokens = s.get('total_tokens', 0)
        level_name = s.get('level_name', f"L{s.get('level', 0)}")
        print(f"{s.get('model', ''):<25} {level_name:<18} {acc:>10.2%} {cost:>10.4f} {tokens:>10}")


def generate_text_report(exp_a_dir: str, exp_b_dir: str, exp_c_dir: str, output_path: str):
    """生成综合分析文本报告"""
    lines = []
    lines.append("# 边端侧小模型电力场景验证实验 — 综合分析报告")
    lines.append(f"\n生成时间: {__import__('datetime').datetime.now().isoformat()}\n")

    for exp, label, d in [('a', '实验A-缺陷分类', exp_a_dir), ('b', '实验B-规则配置', exp_b_dir), ('c', '实验C-指令解析', exp_c_dir)]:
        if not os.path.exists(d):
            continue
        summaries = load_exp_metrics(d)
        lines.append(f"\n## {label}\n")
        lines.append(f"总实验组数: {len(summaries)}\n")

        # 找出最佳小模型和最佳大模型
        local_best = None
        cloud_best = None
        for s in summaries:
            is_cloud = any(x in s.get('model', '') for x in ['pro', 'kimi', 'minimax'])
            acc = s.get('overall_accuracy', s.get('avg_overall_score', 0))
            if is_cloud:
                if cloud_best is None or acc > cloud_best[1]:
                    cloud_best = (s, acc)
            else:
                if local_best is None or acc > local_best[1]:
                    local_best = (s, acc)

        if local_best:
            s, acc = local_best
            lines.append(f"**最佳本地小模型**: {s['model']} ({s.get('level_name', '')}) — 准确率 {acc:.2%}\n")
        if cloud_best:
            s, acc = cloud_best
            lines.append(f"**云端大模型基准**: {s['model']} — 准确率 {acc:.2%}\n")
        if local_best and cloud_best:
            gap = cloud_best[1] - local_best[1]
            lines.append(f"**差距**: {gap:.2%}\n")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n文本报告已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="结果分析与可视化")
    parser.add_argument('--exp_a', type=str, default='output/exp_a_results')
    parser.add_argument('--exp_b', type=str, default='output/exp_b_results')
    parser.add_argument('--exp_c', type=str, default='output/exp_c_results')
    parser.add_argument('--output', type=str, default='output/reports')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 打印各实验汇总
    for exp_dir, title in [(args.exp_a, '实验A'), (args.exp_b, '实验B'), (args.exp_c, '实验C')]:
        if os.path.exists(exp_dir):
            summaries = load_exp_metrics(exp_dir)
            print_summary_table(summaries, title)

    # 生成文本报告
    report_path = os.path.join(args.output, 'comprehensive_report.md')
    generate_text_report(args.exp_a, args.exp_b, args.exp_c, report_path)

    # 生成论文级图表
    print("\n正在生成论文级可视化图表...")
    generate_all_plots('output', args.output)


if __name__ == '__main__':
    main()
