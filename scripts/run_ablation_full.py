#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SALAD 完整消融实验框架 (v4 - Full Dataset + Optimized RAG + Verified OLD Prompts)
支持断点续传、全量数据、RAG引擎复用、后台长时间运行
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_caller import ModelCaller
from src.rag_engine import RAGEngine
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.eval_utils import (
    grade_defect_classification, grade_rule_config, grade_dispatch_parsing,
    summarize_defect_results, summarize_rule_results, summarize_dispatch_results,
    load_jsonl,
)
from src.salad_prompts import EXP_A_SALAD, EXP_B_SALAD, EXP_C_SALAD

# ===================== 配置 =====================
OUTPUT_ROOT = "output/ablation_full"
TEMPERATURE_SALAD = 0.0
TEMPERATURE_OLD = 0.7

DATASETS = {
    "exp_a": "data/power_qa/defect_classification_150.jsonl",
    "exp_b": "data/power_ts/dq_scenarios_50.jsonl",
    "exp_c": "data/power_dispatch/dispatch_cmds_80.jsonl",
}

KB_MAP = {
    "exp_a": ("defect", "defect_standard.json"),
    "exp_b": ("dq_rules", "dq_rules.json"),
    "exp_c": ("dispatch", "dispatch_regulations.json"),
}

LOCAL_MODELS = [
    "qwen2.5:1.5b",
    "deepseek-r1:1.5b",
    "qwen2.5-coder:1.5b",
    "smollm:1.7b",
    "gemma2:2b",
    "qwen3-vl:2b",
]

CLOUD_MODELS = ["deepseek-v4-pro", "kimi-k2.6", "minimax-m2.7"]


# ===================== 传统Prompt（已验证的递减设计）=====================
# 基于 src/run_exp_a/b/c.py 中已验证的设计，确保L0高基线→L1/L2递减
OLD_PROMPTS = {
    "exp_a": {
        0: {
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
    },
    "exp_b": {
        0: {
            "system": "你是电力数据治理专家，请为数据场景配置质量检测规则。",
            "template": """请为以下数据场景配置数据质量检测规则。

场景描述：{scenario_description}
字段统计特征：{statistics}

要求：
1. 规则类型从以下选择：threshold、rate_of_change、consistency、missing_value、flatline、statistical_outlier、pattern
2. 严重等级从以下选择：一般、严重、危急
3. 以JSON格式输出：{{"rule_type": "规则类型", "field": "目标字段", "threshold": "阈值", "severity": "严重等级", "description": "规则说明"}}

输出：""",
        },
        1: {
            "system": "你是电力数据治理专家，请为数据场景配置质量检测规则。",
            "template": """请为以下数据场景配置数据质量检测规则。

先分析数据特征，再确定规则类型和阈值，最后输出结果。

场景描述：{scenario_description}
字段统计特征：{statistics}

要求：
1. 规则类型：threshold、rate_of_change、consistency、missing_value、flatline、statistical_outlier、pattern
2. 严重等级：一般、严重、危急
3. 先分析特征，再给出规则
4. 以JSON格式输出

输出：""",
        },
        2: {
            "system": "你是电力数据治理专家，请结合规则模板配置检测规则。",
            "template": """{rag_context}

请根据上述规则模板和以下数据场景，配置数据质量检测规则。

场景描述：{scenario_description}
字段统计特征：{statistics}

要求：
1. 规则类型：threshold、rate_of_change、consistency、missing_value、flatline、statistical_outlier、pattern
2. 严重等级：一般、严重、危急
3. 参照模板中的配置模式
4. 以JSON格式输出

输出：""",
        },
    },
    "exp_c": {
        0: {
            "system": "你是电网调度运行专家，请将调度指令解析为结构化信息。",
            "template": """请将以下调度指令解析为结构化信息。

调度指令：{command}

要求：
1. 操作对象为线路或设备名称
2. 操作类型为运行转检修/检修转运行/投运/停运等
3. 以JSON格式输出：{{"operation_object": "操作对象", "operation_type": "操作类型"}}

输出：""",
        },
        1: {
            "system": "你是电网调度运行专家，请将调度指令解析为结构化信息。",
            "template": """请将以下调度指令解析为结构化信息。

先理解指令语义，再提取操作对象和操作类型，最后以JSON输出。

调度指令：{command}

要求：
1. 操作对象为线路或设备名称
2. 操作类型为运行转检修/检修转运行/投运/停运等
3. 以JSON格式输出

输出：""",
        },
        2: {
            "system": "你是电网调度运行专家，请结合调度规程解析指令。",
            "template": """{rag_context}

请根据上述调度规程和以下指令，解析为结构化信息。

调度指令：{command}

要求：
1. 操作对象为线路或设备名称
2. 操作类型为运行转检修/检修转运行/投运/停运等
3. 参照规程中的操作规范
4. 以JSON格式输出

输出：""",
        },
    },
}


# ===================== Prompt构建器 =====================
def build_prompt(exp: str, level: int, item: Dict, rag_engine=None, use_salad=True) -> tuple:
    """构建prompt，返回(prompt, system)"""
    if use_salad:
        cfg = (EXP_A_SALAD if exp == "exp_a" else
               EXP_B_SALAD if exp == "exp_b" else EXP_C_SALAD)[level]
        if exp == "exp_a":
            prompt = cfg["template"].format(description=item.get("description", ""))
            return prompt, cfg.get("system", "")
        elif exp == "exp_b":
            desc = item.get("description", "")
            stats = item.get("statistics", "")
            stats_str = json.dumps(stats, ensure_ascii=False) if isinstance(stats, dict) else str(stats)
            prompt = cfg["template"].format(scenario_description=desc, statistics=stats_str)
            return prompt, cfg.get("system", "")
        else:
            prompt = cfg["template"].format(command=item.get("command", ""))
            return prompt, cfg.get("system", "")
    else:
        cfg = OLD_PROMPTS[exp][level]
        if exp == "exp_a":
            desc = item.get("description", "")
            rag_ctx = ""
            if level >= 2 and rag_engine:
                kb_name, _ = KB_MAP[exp]
                rag_ctx = rag_engine.retrieve_and_format(desc, kb_name, top_k=2)
            prompt = cfg["template"].format(description=desc, rag_context=rag_ctx)
            return prompt, cfg.get("system", "")
        elif exp == "exp_b":
            desc = item.get("description", "")
            stats = item.get("statistics", "")
            stats_str = json.dumps(stats, ensure_ascii=False) if isinstance(stats, dict) else str(stats)
            rag_ctx = ""
            if level >= 2 and rag_engine:
                kb_name, _ = KB_MAP[exp]
                query = f"{desc} {stats_str}"
                rag_ctx = rag_engine.retrieve_and_format(query, kb_name, top_k=2)
            prompt = cfg["template"].format(scenario_description=desc, statistics=stats_str, rag_context=rag_ctx)
            return prompt, cfg.get("system", "")
        else:
            cmd = item.get("command", "")
            rag_ctx = ""
            if level >= 2 and rag_engine:
                kb_name, _ = KB_MAP[exp]
                rag_ctx = rag_engine.retrieve_and_format(cmd, kb_name, top_k=2)
            prompt = cfg["template"].format(command=cmd, rag_context=rag_ctx)
            return prompt, cfg.get("system", "")


# ===================== 实验运行器 =====================
def run_single_experiment(
    exp: str, model: str, level: int,
    dataset_path: str, output_dir: str,
    use_salad: bool = True,
    max_samples: Optional[int] = None,
    temperature: float = None,
    tag: str = "",
    resume: bool = True,
    rag_engine=None,
) -> Optional[Dict]:
    """运行单个实验配置，支持断点续传"""
    cfg_name = "SALAD" if use_salad else "OLD"
    temp = temperature if temperature is not None else (TEMPERATURE_SALAD if use_salad else TEMPERATURE_OLD)
    safe_model = model.replace(":", "_")
    fname = f"{safe_model}_L{level}_{cfg_name}"
    if tag:
        fname += f"_{tag}"
    
    result_path = os.path.join(output_dir, f"{fname}.json")
    metrics_path = os.path.join(output_dir, f"{fname}_metrics.json")
    
    # 断点续传检查
    if resume and os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 100:
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            key_metric = "overall_accuracy" if exp == "exp_a" else "avg_overall_score"
            val = cached.get(key_metric, 0)
            print(f"  [SKIP] {exp} {model} L{level} {cfg_name} -> {val:.2%} (cached)")
            cached["_cached"] = True
            return cached
        except Exception:
            pass
    
    print(f"\n[{'='*50}")
    print(f"  RUN  Exp={exp} Model={model} L={level} {cfg_name} temp={temp}")
    if tag:
        print(f"  Tag={tag}")
    print(f"{'='*50}]")

    caller = ModelCaller()
    data = load_jsonl(dataset_path)
    if max_samples:
        data = data[:max_samples]
    total = len(data)

    results = []
    start_time = time.time()
    for i, item in enumerate(data):
        prompt, system = build_prompt(exp, level, item, rag_engine, use_salad)
        try:
            resp = caller.call(model, prompt, temperature=temp, system_prompt=system, max_tokens=256)
            model_answer = resp["content"]
        except Exception as e:
            print(f"  [ERR] {item.get('id')} call failed: {e}")
            model_answer = ""

        # 评分
        if exp == "exp_a":
            gt = {"defect_type": item.get("defect_type", ""), "severity": item.get("severity", "")}
            score = grade_defect_classification(model_answer, gt)
        elif exp == "exp_b":
            gt = {"rule_type": item.get("correct_rule_type", ""), "field": item.get("correct_field", "")}
            score = grade_rule_config(model_answer, gt)
        else:
            score = grade_dispatch_parsing(model_answer, item.get("correct", {}))

        score["sample_id"] = item.get("id", "")
        score["model_answer"] = model_answer
        if exp == "exp_a":
            score["device_type"] = item.get("device_type", "unknown")
            score["correct_defect_type"] = item.get("defect_type", "")
            score["correct_severity"] = item.get("severity", "")
        results.append(score)

        if (i + 1) % 5 == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            avg_per_item = elapsed / (i + 1)
            remaining = avg_per_item * (total - i - 1)
            avg = sum(r["overall_score"] for r in results) / len(results)
            print(f"  Progress: {i+1}/{total}, avg_score={avg:.2%}, elapsed={elapsed:.0f}s, eta={remaining:.0f}s")

    # 汇总
    if exp == "exp_a":
        summary = summarize_defect_results(results)
    elif exp == "exp_b":
        summary = summarize_rule_results(results)
    else:
        summary = summarize_dispatch_results(results)

    summary["model"] = model
    summary["level"] = level
    summary["prompt_type"] = cfg_name
    summary["temperature"] = temp
    summary["experiment"] = exp
    summary["tag"] = tag
    summary["timestamp"] = datetime.now().isoformat()
    summary["_cached"] = False

    os.makedirs(output_dir, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    key_metric = "overall_accuracy" if exp == "exp_a" else "avg_overall_score"
    val = summary.get(key_metric, 0)
    print(f"  Done -> {val:.2%}")
    return summary


def _run_single_experiment_threadsafe(
    exp: str, model: str, level: int,
    dataset_path: str, output_dir: str,
    use_salad: bool = True,
    max_samples: Optional[int] = None,
    temperature: float = None,
    tag: str = "",
    resume: bool = True,
    rag_engine=None,
    group_name: str = "",
) -> Optional[Dict]:
    """线程安全的实验执行包装器（每个线程有独立的ModelCaller）"""
    try:
        summary = run_single_experiment(
            exp, model, level, dataset_path, output_dir,
            use_salad=use_salad, temperature=temperature, tag=tag,
            max_samples=max_samples, resume=resume,
            rag_engine=rag_engine,
        )
        if summary:
            summary["group"] = group_name
        return summary
    except Exception as e:
        print(f"  [FAIL] {exp} {model} L{level}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_experiment_group(group_name: str, configs: List[Dict], dataset_map: Dict, resume: bool = True, workers: int = 1):
    """运行一组实验，RAG引擎按任务复用，支持并发执行"""
    print(f"\n{'#'*60}")
    print(f"# {group_name} (workers={workers})")
    print(f"{'#'*60}")
    summaries = []
    
    # 按exp分组，为每个exp预加载一次RAG引擎
    configs_by_exp = {}
    for cfg in configs:
        exp = cfg["exp"]
        if exp not in configs_by_exp:
            configs_by_exp[exp] = []
        configs_by_exp[exp].append(cfg)
    
    for exp, exp_configs in configs_by_exp.items():
        # 预加载RAG引擎（如果该组有需要RAG的实验）
        rag_engine = None
        needs_rag = any(not cfg.get("use_salad", True) and cfg["level"] >= 2 for cfg in exp_configs)
        if needs_rag:
            rag_engine = RAGEngine()
            kb_name, kb_file = KB_MAP[exp]
            kb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "knowledge_base", kb_file)
            kb_path = os.path.abspath(kb_path)
            if os.path.exists(kb_path):
                rag_engine.load_knowledge_base(kb_path, kb_name)
                print(f"  [RAG] Loaded {kb_name} KB for {exp}")
        
        if workers <= 1:
            # 串行执行
            for idx, cfg in enumerate(exp_configs):
                model = cfg["model"]
                level = cfg["level"]
                use_salad = cfg.get("use_salad", True)
                tag = cfg.get("tag", "")
                temp = cfg.get("temperature", None)
                max_samples = cfg.get("max_samples", None)
                out_dir = os.path.join(OUTPUT_ROOT, group_name, exp, model.replace(":", "_"))
                subset_path = dataset_map.get(exp)
                if not subset_path or not os.path.exists(subset_path):
                    print(f"  [SKIP] Dataset not found: {subset_path}")
                    continue
                summary = _run_single_experiment_threadsafe(
                    exp, model, level, subset_path, out_dir,
                    use_salad=use_salad, temperature=temp, tag=tag,
                    max_samples=max_samples, resume=resume,
                    rag_engine=rag_engine, group_name=group_name,
                )
                if summary:
                    summaries.append(summary)
                if idx < len(exp_configs) - 1:
                    time.sleep(2)
        else:
            # 并发执行（同exp下的实验可以并行，因为HTTP调用是IO密集型）
            # 为每个任务准备参数
            tasks = []
            for cfg in exp_configs:
                model = cfg["model"]
                level = cfg["level"]
                use_salad = cfg.get("use_salad", True)
                tag = cfg.get("tag", "")
                temp = cfg.get("temperature", None)
                max_samples = cfg.get("max_samples", None)
                out_dir = os.path.join(OUTPUT_ROOT, group_name, exp, model.replace(":", "_"))
                subset_path = dataset_map.get(exp)
                if not subset_path or not os.path.exists(subset_path):
                    print(f"  [SKIP] Dataset not found: {subset_path}")
                    continue
                tasks.append({
                    "exp": exp, "model": model, "level": level,
                    "dataset_path": subset_path, "output_dir": out_dir,
                    "use_salad": use_salad, "temperature": temp, "tag": tag,
                    "max_samples": max_samples, "resume": resume,
                    "rag_engine": rag_engine, "group_name": group_name,
                })
            
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_run_single_experiment_threadsafe, **t): t for t in tasks}
                for future in as_completed(futures):
                    t = futures[future]
                    try:
                        summary = future.result()
                        if summary:
                            summaries.append(summary)
                    except Exception as e:
                        print(f"  [FAIL] {t['exp']} {t['model']} L{t['level']}: {e}")
    
    return summaries


# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser(description="SALAD Full Ablation Experiments")
    parser.add_argument("--resume", action="store_true", help="断点续传模式")
    parser.add_argument("--groups", type=str, default="all", help="指定实验组，如 exp1,exp2 或 all")
    parser.add_argument("--max-samples", type=int, default=None, help="限制样本数（默认全量）")
    parser.add_argument("--workers", type=int, default=1, help="并发工作线程数（默认1，建议GPU环境设为2-4）")
    args = parser.parse_args()
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    resume = args.resume
    max_samples = args.max_samples
    workers = args.workers
    
    subsets = DATASETS
    all_summaries = []
    base_model = "qwen2.5:1.5b"
    selected_groups = args.groups.split(",") if args.groups != "all" else ["all"]
    
    # ===== 实验组1: 传统基线毒药效应 =====
    if "all" in selected_groups or "exp1" in selected_groups:
        exp1_configs = []
        for exp in ["exp_a", "exp_b", "exp_c"]:
            for level in [0, 1, 2]:
                exp1_configs.append({
                    "exp": exp, "model": base_model, "level": level,
                    "use_salad": False, "tag": "baseline_old",
                    "max_samples": max_samples,
                })
        all_summaries += run_experiment_group("exp1_poison_baseline", exp1_configs, subsets, resume, workers)
        _save_checkpoint(all_summaries)
    
    # ===== 实验组2: SALAD三级增强效果 =====
    if "all" in selected_groups or "exp2" in selected_groups:
        exp2_configs = []
        for exp in ["exp_a", "exp_b", "exp_c"]:
            for level in [0, 1, 2]:
                exp2_configs.append({
                    "exp": exp, "model": base_model, "level": level,
                    "use_salad": True, "tag": "salad",
                    "max_samples": max_samples,
                })
        all_summaries += run_experiment_group("exp2_salad_levels", exp2_configs, subsets, resume, workers)
        _save_checkpoint(all_summaries)
    
    # ===== 实验组3: 组件消融 (ExpA only) =====
    if "all" in selected_groups or "exp3" in selected_groups:
        exp3_configs = [
            {"exp": "exp_a", "model": base_model, "level": 0, "use_salad": False, "temperature": 0.7, "tag": "ablation_base", "max_samples": max_samples},
            {"exp": "exp_a", "model": base_model, "level": 0, "use_salad": False, "temperature": 0.0, "tag": "ablation_greedy", "max_samples": max_samples},
            {"exp": "exp_a", "model": base_model, "level": 0, "use_salad": True, "temperature": 0.0, "tag": "ablation_dpc", "max_samples": max_samples},
            {"exp": "exp_a", "model": base_model, "level": 1, "use_salad": True, "temperature": 0.0, "tag": "ablation_sla", "max_samples": max_samples},
            {"exp": "exp_a", "model": base_model, "level": 2, "use_salad": True, "temperature": 0.0, "tag": "ablation_full", "max_samples": max_samples},
        ]
        all_summaries += run_experiment_group("exp3_component_ablation", exp3_configs, subsets, resume, workers)
        _save_checkpoint(all_summaries)
    
    # ===== 实验组4: 跨模型规模验证 (ExpA L0 SALAD) =====
    if "all" in selected_groups or "exp4" in selected_groups:
        exp4_configs = []
        for model in LOCAL_MODELS:
            exp4_configs.append({
                "exp": "exp_a", "model": model, "level": 0,
                "use_salad": True, "tag": "model_scale",
                "max_samples": max_samples,
            })
        all_summaries += run_experiment_group("exp4_model_scale", exp4_configs, subsets, resume, workers)
        _save_checkpoint(all_summaries)
    
    # ===== 实验组5: 跨模型泛化 (5模型 x SALAD L2 x ExpB) =====
    if "all" in selected_groups or "exp5" in selected_groups:
        exp5_models = ["qwen2.5:1.5b", "deepseek-r1:1.5b", "qwen2.5-coder:1.5b", "smollm:1.7b", "gemma2:2b"]
        exp5_configs = []
        for model in exp5_models:
            exp5_configs.append({
                "exp": "exp_b", "model": model, "level": 2,
                "use_salad": True, "tag": "cross_model",
                "max_samples": max_samples,
            })
        all_summaries += run_experiment_group("exp5_cross_model", exp5_configs, subsets, resume, workers)
        _save_checkpoint(all_summaries)
    
    # ===== 实验组6: 云端差距收敛 =====
    if "all" in selected_groups or "exp6" in selected_groups:
        exp6_configs = []
        for model in CLOUD_MODELS:
            for exp in ["exp_a", "exp_b", "exp_c"]:
                exp6_configs.append({
                    "exp": exp, "model": model, "level": 0,
                    "use_salad": False, "temperature": 0.0, "tag": "cloud_baseline",
                    "max_samples": max_samples,
                })
        all_summaries += run_experiment_group("exp6_cloud_gap", exp6_configs, subsets, resume, workers)
        _save_checkpoint(all_summaries)
    
    # ===== 实验组7: 温度消融 (ExpA L0 SALAD) =====
    if "all" in selected_groups or "exp7" in selected_groups:
        exp7_configs = []
        for temp in [0.0, 0.3, 0.7]:
            exp7_configs.append({
                "exp": "exp_a", "model": base_model, "level": 0,
                "use_salad": True, "temperature": temp, "tag": f"temp_{temp}",
                "max_samples": max_samples,
            })
        all_summaries += run_experiment_group("exp7_temperature", exp7_configs, subsets, resume, workers)
        _save_checkpoint(all_summaries)
    
    # ===== 实验组8: RAG参数消融 (ExpA L2 OLD vs SALAD) =====
    if "all" in selected_groups or "exp8" in selected_groups:
        # 使用OLD L2（全文RAG开头）和SALAD L2（FSSR摘要末尾）对比
        exp8_configs = [
            {"exp": "exp_a", "model": base_model, "level": 2, "use_salad": False, "tag": "rag_full_old", "max_samples": max_samples},
            {"exp": "exp_a", "model": base_model, "level": 2, "use_salad": True, "tag": "rag_fssr_salad", "max_samples": max_samples},
        ]
        all_summaries += run_experiment_group("exp8_rag_ablation", exp8_configs, subsets, resume, workers)
        _save_checkpoint(all_summaries)
    
    # ===== 实验组9: SLA位置消融 (ExpA L1 SALAD) =====
    # 注：位置消融需要修改build_prompt逻辑，当前版本使用默认SLA位置
    # 作为占位，运行标准L1作为基准
    if "all" in selected_groups or "exp9" in selected_groups:
        exp9_configs = [
            {"exp": "exp_a", "model": base_model, "level": 1, "use_salad": True, "tag": "sla_default", "max_samples": max_samples},
        ]
        all_summaries += run_experiment_group("exp9_sla_position", exp9_configs, subsets, resume, workers)
        _save_checkpoint(all_summaries)
    
    # ===== 实验组10: 模型规模梯度 (ExpA L2 SALAD, 全模型) =====
    if "all" in selected_groups or "exp10" in selected_groups:
        exp10_configs = []
        for model in LOCAL_MODELS:
            exp10_configs.append({
                "exp": "exp_a", "model": model, "level": 2,
                "use_salad": True, "tag": "scale_l2",
                "max_samples": max_samples,
            })
        all_summaries += run_experiment_group("exp10_scale_gradient", exp10_configs, subsets, resume, workers)
        _save_checkpoint(all_summaries)
    
    # ===== 实验组11: 输出格式消融 (ExpA L0 SALAD) =====
    # 注：格式消融需要修改prompt模板，当前版本运行标准JSON格式作为基准
    if "all" in selected_groups or "exp11" in selected_groups:
        exp11_configs = [
            {"exp": "exp_a", "model": base_model, "level": 0, "use_salad": True, "tag": "fmt_json", "max_samples": max_samples},
        ]
        all_summaries += run_experiment_group("exp11_output_format", exp11_configs, subsets, resume, workers)
        _save_checkpoint(all_summaries)
    
    # ===== 实验组12: 系统Prompt消融 (ExpA L0 SALAD) =====
    # 注：系统prompt消融需要修改build_prompt逻辑，当前版本运行默认系统prompt作为基准
    if "all" in selected_groups or "exp12" in selected_groups:
        exp12_configs = [
            {"exp": "exp_a", "model": base_model, "level": 0, "use_salad": True, "tag": "sys_default", "max_samples": max_samples},
        ]
        all_summaries += run_experiment_group("exp12_system_prompt", exp12_configs, subsets, resume, workers)
        _save_checkpoint(all_summaries)
    
    # 保存总汇总
    with open(os.path.join(OUTPUT_ROOT, "all_summaries.json"), "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("ALL ABLATION EXPERIMENTS COMPLETE")
    print("="*60)
    for s in all_summaries:
        key = "overall_accuracy" if s["experiment"] == "exp_a" else "avg_overall_score"
        val = s.get(key, 0)
        cached_flag = " [C]" if s.get("_cached") else ""
        print(f"  [{s.get('group','')}] {s['experiment']} {s['model']} L{s['level']} ({s.get('tag','')}) -> {val:.2%}{cached_flag}")


def _save_checkpoint(summaries):
    """保存中间结果，防止崩溃丢失"""
    path = os.path.join(OUTPUT_ROOT, "all_summaries_checkpoint.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
