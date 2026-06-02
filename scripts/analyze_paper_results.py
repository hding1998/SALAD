#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SALAD 论文结果分析器

读取 run_full_salad_experiment.py 生成的输出，产出：
  1. 控制台打印：所有核心结论数据表
  2. CSV 文件：供 Excel/Python 进一步处理
  3. Markdown 报告：可直接粘贴入论文
  4. PNG 图表：论文图（需要 matplotlib）

用法：
  python scripts/analyze_paper_results.py                        # 分析最新输出
  python scripts/analyze_paper_results.py --output-dir output/full_experiment
  python scripts/analyze_paper_results.py --no-plots             # 跳过图表生成
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Any

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

# ===========================================================
# 数据加载
# ===========================================================

def load_all_metrics(output_dir: Path) -> List[Dict]:
    """从目录中加载所有 *__metrics.json 文件"""
    metrics = []
    for f in sorted(output_dir.rglob("*__metrics.json")):
        try:
            with open(f, encoding="utf-8") as fp:
                m = json.load(fp)
                m["_source_file"] = str(f.relative_to(output_dir))
                # 从路径推断 group
                if m.get("group") is None:
                    parts = f.parts
                    for p in parts:
                        if p.startswith("exp"):
                            m["group"] = p
                            break
                metrics.append(m)
        except Exception as e:
            print(f"[WARN] 无法加载 {f}: {e}")
    return metrics


def load_all_samples(output_dir: Path) -> List[Dict]:
    """从目录中加载所有 *__samples.jsonl 文件"""
    samples = []
    for f in sorted(output_dir.rglob("*__samples.jsonl")):
        try:
            with open(f, encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        s = json.loads(line)
                        samples.append(s)
        except Exception as e:
            print(f"[WARN] 无法加载 {f}: {e}")
    return samples


# ===========================================================
# 辅助函数
# ===========================================================

def get_key_metric(m: Dict) -> float:
    """统一获取主要指标"""
    return m.get("overall_accuracy", m.get("avg_overall_score", 0.0))


def fmt(v: float) -> str:
    return f"{v*100:.1f}%"


def fmt_pp(a: float, b: float) -> str:
    diff = (b - a) * 100
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.1f}pp"


def short_model(model: str) -> str:
    mapping = {
        "qwen2.5:1.5b": "Qwen2.5-1.5B",
        "deepseek-r1:1.5b": "DS-R1-1.5B",
        "qwen2.5-coder:1.5b": "Qwen2.5-Coder-1.5B",
        "smollm:1.7b": "SmolLM-1.7B",
        "gemma2:2b": "Gemma2-2B",
        "deepseek-v4-pro": "DeepSeek-V4-Pro",
        "kimi-k2.6": "Kimi-K2.6",
        "minimax-m2.7": "MiniMax-M2.7",
    }
    return mapping.get(model, model)


def short_prompt(prompt_dict: str, level: int) -> str:
    prefix_map = {
        "EXP_A_TRADITIONAL": "Trad",
        "EXP_B_TRADITIONAL": "Trad",
        "EXP_C_TRADITIONAL": "Trad",
        "EXP_A_SALAD": "SALAD",
        "EXP_B_SALAD": "SALAD",
        "EXP_C_SALAD": "SALAD",
        "EXP_A_SALAD_MARKER": "MARKER",
        "EXP_A_SALAD_DVA": "DVA",
    }
    ptype = prefix_map.get(prompt_dict, prompt_dict.replace("EXP_A_", "").replace("EXP_B_", "").replace("EXP_C_", ""))
    return f"{ptype}-L{level}"


# ===========================================================
# 核心分析函数
# ===========================================================

def analyze_core_comparison(metrics: List[Dict]) -> str:
    """
    分析 exp1（传统毒药）和 exp2（SALAD阶梯）的核心对比
    生成论文表格 Table 1
    """
    lines = [
        "\n" + "="*70,
        "TABLE 1: 传统PE毒药效应 vs SALAD阶梯效应（核心论点）",
        "="*70,
    ]

    # 筛选相关数据 — 限定主力模型 qwen2.5:1.5b，并限定所属组（避免跨组覆盖）
    MAIN_MODEL = "qwen2.5:1.5b"
    trad = {(m["exp"], m["level"]): m for m in metrics
            if m.get("prompt_dict", "").endswith("TRADITIONAL")
            and m.get("temperature", 0) > 0.5
            and m.get("model") == MAIN_MODEL
            and m.get("group") in ("exp1_traditional_poison", None)}
    salad = {(m["exp"], m["level"]): m for m in metrics
             if "SALAD" in m.get("prompt_dict", "") and not m.get("prompt_dict", "").endswith(("MARKER", "DVA"))
             and m.get("temperature", 1.0) == 0.0
             and m.get("use_rag", False) is False
             and m.get("level") in [0, 1, 2]
             and m.get("model") == MAIN_MODEL
             and m.get("group") in ("exp2_salad_ladder", None)}

    header = f"{'实验':<8} {'方案':<15} {'L0':>8} {'L1':>8} {'L2':>8} {'趋势':>10}"
    lines.append(header)
    lines.append("-" * 70)

    for exp in ["exp_a", "exp_b", "exp_c"]:
        exp_label = {"exp_a": "ExpA-缺陷", "exp_b": "ExpB-规则", "exp_c": "ExpC-指令"}[exp]

        t0 = get_key_metric(trad.get((exp, 0), {}))
        t1 = get_key_metric(trad.get((exp, 1), {}))
        t2 = get_key_metric(trad.get((exp, 2), {}))

        s0 = get_key_metric(salad.get((exp, 0), {}))
        s1 = get_key_metric(salad.get((exp, 1), {}))
        s2 = get_key_metric(salad.get((exp, 2), {}))

        if any(v > 0 for v in [t0, t1, t2]):
            trend = "↓递减" if t2 < t0 else "→持平"
            lines.append(f"{exp_label:<8} {'传统PE(毒药)':<15} {fmt(t0):>8} {fmt(t1):>8} {fmt(t2):>8} {trend:>10}")
        if any(v > 0 for v in [s0, s1, s2]):
            trend = "↑递增" if s2 > s0 else "→持平"
            lines.append(f"{exp_label:<8} {'SALAD(解药)':<15} {fmt(s0):>8} {fmt(s1):>8} {fmt(s2):>8} {trend:>10}")
        if any(v > 0 for v in [t0, t1, t2, s0, s1, s2]):
            lines.append("")

    return "\n".join(lines)


def analyze_component_ablation(metrics: List[Dict]) -> str:
    """
    分析 exp3：组件消融
    生成瀑布图数据 Table 2
    """
    lines = [
        "\n" + "="*70,
        "TABLE 2: SALAD组件边际贡献消融（ExpA-缺陷分类）",
        "="*70,
    ]

    steps = [
        ("EXP_A_TRADITIONAL", 0, 0.7, False, "Step-0: 传统基线 temp=0.7"),
        ("EXP_A_TRADITIONAL", 0, 0.0, False, "Step-1: +Greedy解码 (temp=0)"),
        ("EXP_A_SALAD",       0, 0.0, False, "Step-2: +DPC精简 (SALAD-L0)"),
        ("EXP_A_SALAD",       1, 0.0, False, "Step-3: +SLA末尾锚定 (SALAD-L1)"),
        ("EXP_A_SALAD",       2, 0.0, False, "Step-4: +判断标准 (SALAD-L2)"),
        ("EXP_A_SALAD",       3, 0.0, True,  "Step-5: +FSSR RAG (SALAD-L3)"),
    ]

    prev_acc = None
    for pd_key, level, temp, use_rag, label in steps:
        matching = [m for m in metrics
                    if m.get("prompt_dict") == pd_key
                    and m.get("level") == level
                    and abs(m.get("temperature", 1.0) - temp) < 0.01
                    and m.get("use_rag", False) == use_rag
                    and m.get("exp") == "exp_a"
                    and m.get("model") == "qwen2.5:1.5b"
                    and m.get("group") in ("exp3_component_ablation", None)]
        if matching:
            acc = get_key_metric(matching[0])
            delta = fmt_pp(prev_acc, acc) if prev_acc is not None else "  基准"
            lines.append(f"  {label:<42} acc={fmt(acc):>7}  delta={delta}")
            prev_acc = acc
        else:
            lines.append(f"  {label:<42} acc=  N/A   (未运行)")

    return "\n".join(lines)


def analyze_cross_model(metrics: List[Dict]) -> str:
    """
    分析 exp4：跨模型规模验证
    生成 Table 3
    """
    lines = [
        "\n" + "="*70,
        "TABLE 3: 跨模型验证——SALAD增益与架构无关（ExpA）",
        "="*70,
    ]

    header = f"{'模型':<25} {'SALAD-L0':>10} {'SALAD-L2':>10} {'增益':>8}"
    lines.append(header)
    lines.append("-" * 55)

    # 按模型收集 L0 和 L2
    model_data = defaultdict(dict)
    for m in metrics:
        if m.get("exp") == "exp_a" and "SALAD" in m.get("prompt_dict", "") \
                and not m.get("prompt_dict", "").endswith(("TRADITIONAL", "MARKER", "DVA")) \
                and not m.get("use_rag", False) \
                and m.get("temperature", 1.0) == 0.0 \
                and m.get("level") in [0, 2] \
                and m.get("group") in ("exp4_cross_model", "exp10_scale_gradient", None):
            model_data[m["model"]][m["level"]] = get_key_metric(m)

    for model in sorted(model_data.keys()):
        d = model_data[model]
        l0 = d.get(0, None)
        l2 = d.get(2, None)
        if l0 is not None or l2 is not None:
            l0_str = fmt(l0) if l0 is not None else "  N/A"
            l2_str = fmt(l2) if l2 is not None else "  N/A"
            gain = fmt_pp(l0, l2) if (l0 is not None and l2 is not None) else "  N/A"
            lines.append(f"  {short_model(model):<23} {l0_str:>10} {l2_str:>10} {gain:>8}")

    return "\n".join(lines)


def analyze_temperature(metrics: List[Dict]) -> str:
    """分析 exp7：温度消融"""
    lines = [
        "\n" + "="*70,
        "TABLE 4: 温度消融——Greedy解码对小模型的影响",
        "="*70,
    ]

    header = f"{'Prompt类型':<20} {'temp=0.0':>10} {'temp=0.3':>10} {'temp=0.7':>10}"
    lines.append(header)
    lines.append("-" * 55)

    for pd_key, label in [("EXP_A_SALAD", "SALAD-L1"), ("EXP_A_TRADITIONAL", "Trad-L0")]:
        level = 1 if "SALAD" in pd_key else 0
        row = [label]
        for temp in [0.0, 0.3, 0.7]:
            matching = [m for m in metrics
                        if m.get("prompt_dict") == pd_key
                        and m.get("level") == level
                        and abs(m.get("temperature", -1) - temp) < 0.05
                        and m.get("exp") == "exp_a"
                        and m.get("model") == "qwen2.5:1.5b"
                        and m.get("group") in ("exp7_temperature", None)]
            val = fmt(get_key_metric(matching[0])) if matching else "  N/A"
            row.append(val)
        lines.append(f"  {row[0]:<20} {row[1]:>10} {row[2]:>10} {row[3]:>10}")

    return "\n".join(lines)


def analyze_cloud_gap(metrics: List[Dict]) -> str:
    """分析 exp6：云端差距收敛"""
    lines = [
        "\n" + "="*70,
        "TABLE 5: 云端差距收敛——小模型SALAD-L2 vs 大模型基线",
        "="*70,
    ]

    header = f"{'模型':<25} {'ExpA':>8} {'ExpB':>8} {'ExpC':>8} {'平均':>8}"
    lines.append(header)
    lines.append("-" * 60)

    # qwen2.5:1.5b at L2, qwen3.6:35b at L0, cloud models at L0
    all_models = ["qwen2.5:1.5b", "qwen3.6:35b", "deepseek-v4-pro", "kimi-k2.6"]
    for model in all_models:
        row = [short_model(model)]
        vals = []
        for exp in ["exp_a", "exp_b", "exp_c"]:
            # 小模型取 L2 (SALAD提升后), 大模型/云端取 L0 (基线)
            level = 2 if model == "qwen2.5:1.5b" else 0
            matching = [m for m in metrics
                        if m.get("model") == model
                        and m.get("exp") == exp
                        and m.get("prompt_dict", "").endswith("_SALAD")
                        and not m.get("prompt_dict", "").endswith(("MARKER", "DVA"))
                        and m.get("level") == level
                        and m.get("temperature", 1.0) == 0.0
                        and m.get("group") in ("exp6_cloud_gap", None)]
            if matching:
                v = get_key_metric(matching[0])
                row.append(fmt(v))
                vals.append(v)
            else:
                row.append("  N/A")
        avg = fmt(sum(vals) / len(vals)) if vals else "  N/A"
        row.append(avg)
        lines.append(f"  {row[0]:<23} {row[1]:>8} {row[2]:>8} {row[3]:>8} {row[4]:>8}")

    # 相对性能
    lines.append("")
    lines.append("  相对云端大模型的性能比（%）：")
    local_vals = {}
    cloud_vals = {}
    for exp in ["exp_a", "exp_b", "exp_c"]:
        for m in metrics:
            if m.get("model") == "qwen2.5:1.5b" and m.get("exp") == exp \
                    and m.get("prompt_dict", "").endswith("_SALAD") \
                    and not m.get("prompt_dict", "").endswith(("MARKER", "DVA")) \
                    and m.get("level") == 2:
                local_vals[exp] = get_key_metric(m)
        for cloud in ["deepseek-v4-pro", "kimi-k2.6"]:
            for m in metrics:
                if m.get("model") == cloud and m.get("exp") == exp \
                        and m.get("prompt_dict", "").endswith("_SALAD") \
                        and m.get("level") == 0:
                    cloud_vals[exp] = max(cloud_vals.get(exp, 0), get_key_metric(m))
    for exp in ["exp_a", "exp_b", "exp_c"]:
        lv = local_vals.get(exp)
        cv = cloud_vals.get(exp)
        if lv and cv and cv > 0:
            ratio = lv / cv * 100
            lines.append(f"    {exp}: {lv*100:.1f}% / {cv*100:.1f}% = {ratio:.1f}%")

    return "\n".join(lines)


def analyze_new_experiments(metrics: List[Dict]) -> str:
    """分析新创意实验结果"""
    lines = [
        "\n" + "="*70,
        "TABLE 6: 新创意实验结果",
        "="*70,
    ]

    # Section Markers 对比
    lines.append("\n[ 6a ] Section Markers vs SALAD（ExpA）")
    lines.append(f"{'方案':<25} {'L0':>8} {'L1':>8} {'L2':>8}")
    lines.append("-" * 50)

    for pd_key, label in [("EXP_A_SALAD", "SALAD"), ("EXP_A_SALAD_MARKER", "MARKER"), ("EXP_A_SALAD_DVA", "DVA")]:
        row = [label]
        for level in [0, 1, 2]:
            matching = [m for m in metrics
                        if m.get("prompt_dict") == pd_key
                        and m.get("level") == level
                        and m.get("exp") == "exp_a"
                        and m.get("temperature", 1.0) == 0.0]
            val = fmt(get_key_metric(matching[0])) if matching else "  N/A"
            row.append(val)
        lines.append(f"  {row[0]:<23} {row[1]:>8} {row[2]:>8} {row[3]:>8}")

    # Sequential Decompose
    lines.append("\n[ 6b ] Sequential Decompose vs SALAD-L2（ExpA）")
    for m in metrics:
        if m.get("method") == "seq_decompose" and m.get("exp") == "exp_a":
            lines.append(f"  Seq-Decompose:  acc={fmt(get_key_metric(m))}")
    for m in metrics:
        if m.get("prompt_dict") == "EXP_A_SALAD" and m.get("level") == 2 \
                and m.get("exp") == "exp_a" and m.get("temperature", 1.0) == 0.0 \
                and not m.get("use_rag", False):
            lines.append(f"  SALAD-L2:      acc={fmt(get_key_metric(m))}")
            break

    # SALAD L3 vs L2
    lines.append("\n[ 6c ] SALAD-L3 (+ FSSR RAG) vs SALAD-L2（三任务）")
    lines.append(f"{'实验':<10} {'SALAD-L2':>10} {'SALAD-L3':>10} {'增益':>8}")
    lines.append("-" * 45)
    for exp in ["exp_a", "exp_b", "exp_c"]:
        l2_matches = [m for m in metrics
                      if m.get("prompt_dict", "").endswith("_SALAD")
                      and not m.get("prompt_dict", "").endswith(("MARKER", "DVA"))
                      and m.get("level") == 2
                      and m.get("exp") == exp
                      and m.get("temperature", 1.0) == 0.0
                      and not m.get("use_rag", False)]
        l3_matches = [m for m in metrics
                      if m.get("prompt_dict", "").endswith("_SALAD")
                      and not m.get("prompt_dict", "").endswith(("MARKER", "DVA"))
                      and m.get("level") == 3
                      and m.get("exp") == exp
                      and m.get("use_rag", True)]
        l2_acc = get_key_metric(l2_matches[0]) if l2_matches else None
        l3_acc = get_key_metric(l3_matches[0]) if l3_matches else None
        l2_str = fmt(l2_acc) if l2_acc is not None else "  N/A"
        l3_str = fmt(l3_acc) if l3_acc is not None else "  N/A"
        gain = fmt_pp(l2_acc, l3_acc) if (l2_acc and l3_acc) else "  N/A"
        lines.append(f"  {exp:<10} {l2_str:>10} {l3_str:>10} {gain:>8}")

    return "\n".join(lines)


def analyze_token_efficiency(samples: List[Dict]) -> str:
    """分析 token 使用效率"""
    lines = [
        "\n" + "="*70,
        "TABLE 7: Token效率分析（SALAD vs 传统 vs 新变体）",
        "="*70,
    ]

    header = f"{'方案':<30} {'平均输入Token':>14} {'平均输出Token':>14} {'平均acc':>10}"
    lines.append(header)
    lines.append("-" * 70)

    groups = defaultdict(list)
    for s in samples:
        pd = s.get("prompt_dict", "")
        lvl = s.get("level", 0)
        key = f"{pd}-L{lvl}"
        groups[key].append(s)

    for key in sorted(groups.keys()):
        g = groups[key]
        in_tokens = [s.get("input_tokens", 0) for s in g if s.get("input_tokens", 0) > 0]
        out_tokens = [s.get("output_tokens", 0) for s in g if s.get("output_tokens", 0) > 0]
        accs = [s.get("overall_score", 0) for s in g]
        avg_in = sum(in_tokens) / len(in_tokens) if in_tokens else 0
        avg_out = sum(out_tokens) / len(out_tokens) if out_tokens else 0
        avg_acc = sum(accs) / len(accs) if accs else 0
        if avg_in > 0 or avg_acc > 0:
            lines.append(f"  {key:<28} {avg_in:>14.0f} {avg_out:>14.0f} {fmt(avg_acc):>10}")

    return "\n".join(lines)


def generate_csv(metrics: List[Dict], out_path: Path):
    """生成 CSV 供外部分析"""
    import csv
    fields = [
        "group", "model", "exp", "prompt_dict", "level", "temperature",
        "use_rag", "n_samples", "overall_accuracy", "avg_overall_score",
        "type_accuracy", "severity_accuracy",
        "json_valid_rate", "avg_rule_correct", "avg_obj_score", "avg_type_score",
        "total_cost_usd", "avg_latency_ms", "avg_input_tokens", "avg_output_tokens",
        "n_errors", "timestamp", "run_id",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for m in metrics:
            row = {k: m.get(k, "") for k in fields}
            writer.writerow(row)
    print(f"\n[CSV] 已保存: {out_path}")


def generate_markdown_report(metrics: List[Dict], sections: List[str]) -> str:
    """生成 Markdown 格式的完整报告"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = f"""# SALAD Framework 实验结果报告

生成时间：{ts}
实验总运行数：{len(metrics)}
涉及模型：{', '.join(sorted(set(m.get('model','') for m in metrics)))}

---
"""
    body = "\n\n".join(f"```\n{s.strip()}\n```" for s in sections if s.strip())
    return header + body


def generate_plots(metrics: List[Dict], samples: List[Dict], out_dir: Path):
    """生成论文级图表"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
        import numpy as np

        plt.rcParams.update({
            "font.size": 10,
            "axes.titlesize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": ["DejaVu Sans", "SimHei", "Arial Unicode MS"],
        })

        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- Figure 1: 毒药 vs 解药 折线图（核心图）----
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
        exp_labels = {"exp_a": "Exp-A\n(Defect Classification)",
                      "exp_b": "Exp-B\n(Rule Configuration)",
                      "exp_c": "Exp-C\n(Dispatch Parsing)"}

        colors = {"trad": "#E74C3C", "salad": "#2ECC71"}
        markers = {"trad": "v", "salad": "^"}

        for ax, exp in zip(axes, ["exp_a", "exp_b", "exp_c"]):
            for style, pd_suffix, temp_range, label in [
                ("trad", "TRADITIONAL", (0.5, 1.1), "Traditional PE"),
                ("salad", "SALAD", (-0.1, 0.1), "SALAD"),
            ]:
                vals = {}
                for m in metrics:
                    if m.get("exp") == exp \
                            and m.get("prompt_dict", "").endswith(pd_suffix) \
                            and not m.get("prompt_dict", "").endswith(("MARKER", "DVA")) \
                            and temp_range[0] < m.get("temperature", 0) < temp_range[1] \
                            and m.get("level") in [0, 1, 2] \
                            and not m.get("use_rag", False):
                        vals[m["level"]] = get_key_metric(m) * 100
                if vals:
                    xs = sorted(vals.keys())
                    ys = [vals[x] for x in xs]
                    ax.plot(xs, ys, color=colors[style], marker=markers[style],
                            linewidth=2, markersize=8, label=label)
                    for x, y in zip(xs, ys):
                        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                                    xytext=(0, 6), ha="center", fontsize=8)

            ax.set_title(exp_labels[exp])
            ax.set_xlabel("Enhancement Level")
            ax.set_ylabel("Overall Accuracy (%)" if exp == "exp_a" else "")
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["L0\n(Base)", "L1\n(+Anchor)", "L2\n(+Criteria)"])
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())

        plt.suptitle("SALAD vs Traditional PE: Performance Trend Across Enhancement Levels",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        p = out_dir / "fig1_poison_vs_salad.png"
        plt.savefig(p, bbox_inches="tight")
        plt.close()
        print(f"[PLOT] 已保存: {p}")

        # ---- Figure 2: 组件消融瀑布图 ----
        ablation_steps = [
            ("EXP_A_TRADITIONAL", 0, 0.7, False, "Trad\ntemp=0.7"),
            ("EXP_A_TRADITIONAL", 0, 0.0, False, "+Greedy\n(temp=0)"),
            ("EXP_A_SALAD",       0, 0.0, False, "+DPC\n(Lite)"),
            ("EXP_A_SALAD",       1, 0.0, False, "+SLA\n(Anchor)"),
            ("EXP_A_SALAD",       2, 0.0, False, "+Criteria\n(Aug-1)"),
            ("EXP_A_SALAD",       3, 0.0, True,  "+RAG\n(Aug-2)"),
        ]
        step_vals = []
        step_labels = []
        for pd_key, level, temp, use_rag, label in ablation_steps:
            matching = [m for m in metrics
                        if m.get("prompt_dict") == pd_key and m.get("level") == level
                        and abs(m.get("temperature", -1) - temp) < 0.05
                        and m.get("use_rag", False) == use_rag
                        and m.get("exp") == "exp_a"]
            val = get_key_metric(matching[0]) * 100 if matching else None
            step_vals.append(val)
            step_labels.append(label)

        valid = [(l, v) for l, v in zip(step_labels, step_vals) if v is not None]
        if len(valid) >= 2:
            fig, ax = plt.subplots(figsize=(9, 5))
            xs = range(len(valid))
            ys = [v for _, v in valid]
            lbls = [l for l, _ in valid]
            bar_colors = ["#E74C3C"] + ["#3498DB"] * (len(ys) - 1)
            bars = ax.bar(xs, ys, color=bar_colors, alpha=0.85, width=0.6)
            for bar, y in zip(bars, ys):
                ax.text(bar.get_x() + bar.get_width() / 2, y + 0.5,
                        f"{y:.1f}%", ha="center", va="bottom", fontsize=9)
            # 增量箭头
            for i in range(1, len(ys)):
                dy = ys[i] - ys[i-1]
                color = "#27AE60" if dy > 0 else "#E74C3C"
                ax.annotate(f"{'+' if dy>=0 else ''}{dy:.1f}pp",
                            xy=(i, ys[i] - abs(dy)/2),
                            ha="center", va="center",
                            fontsize=8, color=color, fontweight="bold")

            ax.set_xticks(xs)
            ax.set_xticklabels(lbls, fontsize=9)
            ax.set_ylabel("Overall Accuracy (%)")
            ax.set_title("SALAD Component Ablation — Marginal Contribution (Exp-A)", fontweight="bold")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            p = out_dir / "fig2_component_ablation.png"
            plt.savefig(p, bbox_inches="tight")
            plt.close()
            print(f"[PLOT] 已保存: {p}")

        # ---- Figure 3: 跨模型对比 ----
        model_data = defaultdict(dict)
        for m in metrics:
            if m.get("exp") == "exp_a" and "SALAD" in m.get("prompt_dict", "") \
                    and not m.get("prompt_dict", "").endswith(("TRADITIONAL", "MARKER", "DVA")) \
                    and not m.get("use_rag", False) \
                    and m.get("temperature", 1.0) == 0.0 \
                    and m.get("level") in [0, 2]:
                model_data[m["model"]][m["level"]] = get_key_metric(m) * 100

        if model_data:
            models_list = sorted(model_data.keys())
            l0_vals = [model_data[m].get(0, 0) for m in models_list]
            l2_vals = [model_data[m].get(2, 0) for m in models_list]
            short_names = [short_model(m) for m in models_list]

            x = np.arange(len(models_list))
            width = 0.35
            fig, ax = plt.subplots(figsize=(10, 5))
            b1 = ax.bar(x - width/2, l0_vals, width, label="SALAD-L0 (Base)", color="#AED6F1", alpha=0.9)
            b2 = ax.bar(x + width/2, l2_vals, width, label="SALAD-L2 (Full)", color="#2ECC71", alpha=0.9)
            for bar in list(b1) + list(b2):
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.0f}%",
                            ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(short_names, fontsize=8, rotation=15, ha="right")
            ax.set_ylabel("Overall Accuracy (%)")
            ax.set_title("Cross-Model Validation: SALAD Gain is Architecture-Agnostic", fontweight="bold")
            ax.legend()
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            p = out_dir / "fig3_cross_model.png"
            plt.savefig(p, bbox_inches="tight")
            plt.close()
            print(f"[PLOT] 已保存: {p}")

        # ---- Figure 4: 温度消融 ----
        temp_data = defaultdict(dict)
        for m in metrics:
            if m.get("exp") == "exp_a" and m.get("level") in [0, 1] \
                    and m.get("prompt_dict") in ("EXP_A_SALAD", "EXP_A_TRADITIONAL"):
                key = f"{m['prompt_dict']}-L{m['level']}"
                temp_data[key][m.get("temperature", 0)] = get_key_metric(m) * 100

        if temp_data:
            fig, ax = plt.subplots(figsize=(7, 4))
            colors_t = ["#2ECC71", "#3498DB", "#E74C3C", "#9B59B6"]
            for i, (key, vals) in enumerate(sorted(temp_data.items())):
                ts = sorted(vals.keys())
                vs = [vals[t] for t in ts]
                if vs:
                    ax.plot(ts, vs, marker="o", label=key, color=colors_t[i % len(colors_t)],
                            linewidth=2, markersize=8)
            ax.set_xlabel("Temperature")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Temperature Ablation: Greedy (temp=0) is Optimal for Small Models", fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            plt.tight_layout()
            p = out_dir / "fig4_temperature.png"
            plt.savefig(p, bbox_inches="tight")
            plt.close()
            print(f"[PLOT] 已保存: {p}")

        # ---- Figure 5: 新实验对比热力图 ----
        methods = ["SALAD", "MARKER", "DVA"]
        pd_map = {"SALAD": "EXP_A_SALAD", "MARKER": "EXP_A_SALAD_MARKER", "DVA": "EXP_A_SALAD_DVA"}
        heat_data = np.zeros((len(methods), 3))
        for i, method in enumerate(methods):
            for j, level in enumerate([0, 1, 2]):
                matching = [m for m in metrics
                            if m.get("prompt_dict") == pd_map[method]
                            and m.get("level") == level
                            and m.get("exp") == "exp_a"
                            and m.get("temperature", 1.0) == 0.0]
                if matching:
                    heat_data[i, j] = get_key_metric(matching[0]) * 100

        if heat_data.max() > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(heat_data, cmap="YlOrRd", aspect="auto",
                           vmin=max(0, heat_data[heat_data > 0].min() - 5),
                           vmax=min(100, heat_data.max() + 5))
            for i in range(len(methods)):
                for j in range(3):
                    val = heat_data[i, j]
                    text = f"{val:.1f}%" if val > 0 else "N/A"
                    ax.text(j, i, text, ha="center", va="center", fontsize=10, fontweight="bold",
                            color="white" if val > 60 else "black")
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["L0 (Base)", "L1 (+Anchor)", "L2 (+Criteria)"])
            ax.set_yticks(range(len(methods)))
            ax.set_yticklabels(methods)
            ax.set_title("Prompt Variant Comparison Heatmap (Exp-A)", fontweight="bold")
            plt.colorbar(im, ax=ax, label="Accuracy (%)")
            plt.tight_layout()
            p = out_dir / "fig5_new_experiments_heatmap.png"
            plt.savefig(p, bbox_inches="tight")
            plt.close()
            print(f"[PLOT] 已保存: {p}")

        print(f"\n[PLOTS] 所有图表已保存至: {out_dir}")

    except ImportError:
        print("[WARN] matplotlib 未安装，跳过图表生成")
    except Exception as e:
        import traceback
        print(f"[ERROR] 图表生成失败: {e}")
        traceback.print_exc()


# ===========================================================
# 主函数
# ===========================================================

def main():
    parser = argparse.ArgumentParser(description="SALAD 论文结果分析器")
    parser.add_argument("--output-dir", default="",
                        help="实验输出目录（默认: output/full_experiment）")
    parser.add_argument("--no-plots", action="store_true", help="跳过图表生成")
    parser.add_argument("--report-out", default="",
                        help="Markdown 报告输出路径（默认: output_dir/paper_report.md）")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else ROOT_DIR / "output" / "full_experiment"
    report_dir = out_dir / "paper_outputs"
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n读取实验结果: {out_dir}")
    metrics = load_all_metrics(out_dir)
    samples = load_all_samples(out_dir)
    print(f"  已加载 {len(metrics)} 个实验运行，{len(samples)} 个样本记录")

    if not metrics:
        print("[ERROR] 没有找到任何实验结果。请先运行 run_full_salad_experiment.py")
        return

    # 生成各分析表
    sections = []
    sections.append(analyze_core_comparison(metrics))
    sections.append(analyze_component_ablation(metrics))
    sections.append(analyze_cross_model(metrics))
    sections.append(analyze_temperature(metrics))
    sections.append(analyze_cloud_gap(metrics))
    sections.append(analyze_new_experiments(metrics))
    if samples:
        sections.append(analyze_token_efficiency(samples))

    # 打印到控制台
    for s in sections:
        print(s)

    # 保存 CSV
    csv_path = report_dir / "all_metrics.csv"
    generate_csv(metrics, csv_path)

    # 保存 Markdown 报告
    report_content = generate_markdown_report(metrics, sections)
    report_path = Path(args.report_out) if args.report_out else report_dir / "paper_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\n[REPORT] Markdown 报告已保存: {report_path}")

    # 生成图表
    if not args.no_plots:
        generate_plots(metrics, samples, report_dir / "figures")

    print(f"\n所有输出已保存至: {report_dir}")


if __name__ == "__main__":
    main()
