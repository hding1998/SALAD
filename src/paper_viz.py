#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文级可视化图表生成
输出高分辨率、适合论文直接插入的图表
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from typing import List, Dict

# 论文风格设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

COLORS = {
    'local_small': '#3498db',   # 本地小模型
    'local_enhanced': '#2ecc71', # 增强后小模型
    'cloud': '#e74c3c',         # 云端大模型
    'level0': '#95a5a6',
    'level1': '#f39c12',
    'level2': '#9b59b6',
    'level3': '#2ecc71',
}


def load_metrics(results_dir: str, exp: str) -> pd.DataFrame:
    """加载某实验的所有 metrics 文件"""
    pattern = os.path.join(results_dir, f"exp_{exp}_results", "*_metrics.json")
    files = glob.glob(pattern)
    rows = []
    for f in files:
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        row = {
            'model': data.get('model', ''),
            'level': data.get('level', 0),
            'level_name': data.get('level_name', ''),
            'overall_accuracy': data.get('overall_accuracy', data.get('avg_overall_score', 0)),
            'type_accuracy': data.get('type_accuracy', 0),
            'json_valid_rate': data.get('json_valid_rate', 0),
            'cost_usd': data.get('total_cost_usd', 0),
            'latency_ms': data.get('avg_latency_ms', 0),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_waterfall(df: pd.DataFrame, output_path: str, title: str = "增强效果瀑布图"):
    """
    增益瀑布图：展示从 Level 0 到 Level 3 的逐步提升
    """
    # 筛选本地小模型
    local_models = ['qwen2.5:1.5b', 'deepseek-r1:1.5b', 'qwen2.5-coder:1.5b', 'smollm:1.7b', 'gemma2:2b']
    df_local = df[df['model'].isin(local_models)].copy()
    if df_local.empty:
        return

    fig, axes = plt.subplots(1, len(local_models), figsize=(16, 5), sharey=True)
    if len(local_models) == 1:
        axes = [axes]

    levels = [0, 1, 2, 3]
    level_names = ['裸模型', '+CoT', '+RAG', '+RAG+模板']
    colors = [COLORS['level0'], COLORS['level1'], COLORS['level2'], COLORS['level3']]

    for ax, model in zip(axes, local_models):
        df_m = df_local[df_local['model'] == model].sort_values('level')
        accs = [df_m[df_m['level'] == l]['overall_accuracy'].mean() for l in levels]
        accs = [a if not np.isnan(a) else 0 for a in accs]

        # 瀑布图计算
        bottoms = [0] + accs[:-1]
        heights = [accs[0]] + [accs[i] - accs[i-1] for i in range(1, len(accs))]
        bar_colors = [colors[0]] + [colors[i] if heights[i] >= 0 else '#e74c3c' for i in range(1, len(heights))]

        bars = ax.bar(level_names, heights, bottom=bottoms, color=bar_colors, edgecolor='white', linewidth=1.5)

        # 添加数值标签
        for i, (bar, acc) in enumerate(zip(bars, accs)):
            ax.text(bar.get_x() + bar.get_width()/2, acc + 0.01, f'{acc:.1%}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylim(0, 1.15)
        ax.set_title(model.replace(':', '\n'), fontsize=11)
        ax.set_ylabel('准确率' if model == local_models[0] else '')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] 瀑布图已保存: {output_path}")


def plot_radar(df: pd.DataFrame, output_path: str):
    """
    雷达图：多维度能力对比
    维度：准确率、JSON合规率、类型准确率、成本效率、响应速度
    """
    from math import pi

    categories = ['准确率', 'JSON合规率', '类型准确率', '成本效率', '响应速度']
    N = len(categories)

    # 计算各模型均值
    models = df['model'].unique()
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    for model in models:
        df_m = df[df['model'] == model]
        acc = df_m['overall_accuracy'].mean()
        json_v = df_m['json_valid_rate'].mean() if 'json_valid_rate' in df_m else acc
        type_acc = df_m['type_accuracy'].mean() if 'type_accuracy' in df_m else acc
        cost_eff = 1.0 - min(df_m['cost_usd'].mean() / 1.0, 1.0)  # 成本越低越好
        speed = 1.0  # 本地模型速度优势，简化处理

        values = [acc, json_v, type_acc, cost_eff, speed]
        values += values[:1]

        color = COLORS['cloud'] if any(x in model for x in ['pro', 'kimi', 'minimax']) else COLORS['local_small']
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('模型能力雷达图', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] 雷达图已保存: {output_path}")


def plot_accuracy_vs_cost(df: pd.DataFrame, output_path: str):
    """
    准确率-成本散点图（气泡大小=样本数）
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    for model in df['model'].unique():
        df_m = df[df['model'] == model]
        x = df_m['cost_usd'].mean() + 0.0001  # 避免0值取对数问题
        y = df_m['overall_accuracy'].mean()
        size = len(df_m) * 30

        is_cloud = any(x in model for x in ['pro', 'kimi', 'minimax'])
        color = COLORS['cloud'] if is_cloud else COLORS['local_small']
        marker = 's' if is_cloud else 'o'

        ax.scatter(x, y, s=size, c=color, alpha=0.7, edgecolors='black', linewidth=1.5, marker=marker, label=model)
        ax.annotate(model.replace(':', ''), (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

    ax.set_xscale('log')
    ax.set_xlabel('平均单次推理成本 (USD, 对数轴)', fontsize=12)
    ax.set_ylabel('平均准确率', fontsize=12)
    ax.set_title('准确率-成本效益分析', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] 散点图已保存: {output_path}")


def plot_heatmap(df: pd.DataFrame, output_path: str):
    """
    热力图：模型 × 增强等级
    """
    pivot = df.pivot_table(values='overall_accuracy', index='model', columns='level', aggfunc='mean')
    pivot.columns = ['L0裸模型', 'L1+CoT', 'L2+RAG', 'L3+RAG+模板']

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlGn', vmin=0, vmax=1,
                linewidths=1, linecolor='white', cbar_kws={'label': '准确率'}, ax=ax)
    ax.set_title('模型×增强等级 能力热力图', fontsize=16, fontweight='bold')
    ax.set_xlabel('增强等级', fontsize=12)
    ax.set_ylabel('模型', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] 热力图已保存: {output_path}")


def plot_size_vs_accuracy(df: pd.DataFrame, output_path: str):
    """
    模型规模-准确率关系图（对数回归）
    """
    # 模型规模映射（参数规模，单位：B）
    size_map = {
        'qwen2.5:1.5b': 1.5, 'deepseek-r1:1.5b': 1.5, 'qwen2.5-coder:1.5b': 1.5,
        'smollm:1.7b': 1.7, 'gemma2:2b': 2.0,
        'deepseek-v4-pro': 1600, 'kimi-k2.6': 1000, 'minimax-m2.7': 500,
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    df_plot = df.copy()
    df_plot['param_size'] = df_plot['model'].map(size_map)
    df_plot = df_plot.dropna(subset=['param_size'])

    for level in sorted(df_plot['level'].unique()):
        df_l = df_plot[df_plot['level'] == level]
        x = df_l['param_size'].values
        y = df_l['overall_accuracy'].values
        label = f'L{level}'
        ax.scatter(x, y, s=80, alpha=0.7, label=label)

    ax.set_xscale('log')
    ax.set_xlabel('模型参数规模 (B, 对数轴)', fontsize=12)
    ax.set_ylabel('平均准确率', fontsize=12)
    ax.set_title('模型规模-准确率关系（对数坐标）', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(title='增强等级', loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] 规模-准确率图已保存: {output_path}")


def generate_latex_table(df: pd.DataFrame, output_path: str):
    """生成 LaTeX 表格代码"""
    summary = df.groupby(['model', 'level']).agg({
        'overall_accuracy': 'mean',
        'cost_usd': 'sum',
    }).reset_index()

    latex = r"""\begin{table}[htbp]
\centering
\caption{模型性能对比结果}
\label{tab:results}
\begin{tabular}{lcccc}
\hline
\textbf{模型} & \textbf{等级} & \textbf{准确率} & \textbf{总成本(USD)} \\
\hline
"""
    for _, row in summary.iterrows():
        latex += f"{row['model']} & L{row['level']} & {row['overall_accuracy']:.1%} & {row['cost_usd']:.4f} \\\\\n"

    latex += r"""\hline
\end{tabular}
\end{table}
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)
    print(f"[VIZ] LaTeX表格已保存: {output_path}")


def generate_all_plots(results_base_dir: str, output_dir: str):
    """一键生成所有图表"""
    os.makedirs(output_dir, exist_ok=True)

    for exp in ['a', 'b', 'c']:
        df = load_metrics(results_base_dir, exp)
        if df.empty:
            print(f"[WARN] 实验 {exp.upper()} 无数据，跳过可视化")
            continue

        print(f"\n=== 生成实验 {exp.upper()} 可视化 ===")
        plot_waterfall(df, os.path.join(output_dir, f'exp_{exp}_waterfall.png'), title=f'实验{exp.upper()}: 增强效果瀑布图')
        plot_heatmap(df, os.path.join(output_dir, f'exp_{exp}_heatmap.png'))
        plot_accuracy_vs_cost(df, os.path.join(output_dir, f'exp_{exp}_accuracy_cost.png'))
        plot_radar(df, os.path.join(output_dir, f'exp_{exp}_radar.png'))
        plot_size_vs_accuracy(df, os.path.join(output_dir, f'exp_{exp}_size_accuracy.png'))
        generate_latex_table(df, os.path.join(output_dir, f'exp_{exp}_table.tex'))

    print(f"\n全部可视化已保存至: {output_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='output', help='实验结果根目录')
    parser.add_argument('--output', type=str, default='output/reports', help='图表输出目录')
    args = parser.parse_args()
    generate_all_plots(args.input, args.output)
