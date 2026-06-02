#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick results viewer: prints all experiment metrics in a table."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output" / "full_experiment"

def acc(data):
    return data.get("overall_accuracy", data.get("avg_overall_score", -1))

groups = [
    ("exp1_traditional_poison", "传统PE毒药效应"),
    ("exp2_salad_ladder", "SALAD阶梯效应"),
    ("exp3_component_ablation", "组件消融"),
    ("exp4_cross_model", "跨模型验证"),
    ("exp5_cross_task", "跨任务泛化"),
    ("exp7_temperature", "温度消融"),
    ("exp13_salad_l3_rag", "SALAD-L3 RAG增益"),
    ("exp14_section_markers", "Section Markers对比"),
    ("exp15_sequential_decompose", "顺序分解"),
    ("exp16_domain_vocab_anchor", "领域词汇锚定"),
    ("exp10_scale_gradient", "规模梯度"),
    ("exp6_cloud_gap", "云端差距收敛"),
]

for gname, gdesc in groups:
    gdir = OUT / gname
    if not gdir.exists():
        print(f"\n[{gname}] MISSING")
        continue
    files = sorted(gdir.glob("*__metrics.json"))
    if not files:
        print(f"\n[{gname}] NO RESULTS")
        continue
    print(f"\n=== {gname}: {gdesc} ===")
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            a = acc(data)
            model = data.get("model", "?")
            exp = data.get("exp", "?")
            pname = data.get("prompt_name", data.get("prompt_dict", "?"))
            lvl = data.get("level", "?")
            temp = data.get("temperature", "?")
            print(f"  {model:<25} {exp:<6} {pname:<28} L{lvl} t={temp} => {a:.3f}")
        except Exception as e:
            print(f"  [ERROR] {f.name}: {e}")
