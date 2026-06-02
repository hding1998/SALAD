#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SALAD Framework — 全量消融实验主运行器 v2（扩充数据集版）

数据集版本：
  ExpA: defect_classification_200.jsonl  (197 条)
  ExpB: dq_scenarios_120.jsonl           (120 条)
  ExpC: dispatch_cmds_200.jsonl          (192 条)

实验矩阵（15组）：
  核心对比:
    exp1  传统PE毒药效应（CoT+RAG 递减）
    exp2  SALAD阶梯效应（L0→L2 递增）
    exp3  组件消融（量化各组件边际贡献）
  模型泛化:
    exp4  跨模型规模验证（多架构 × ExpA L0/L2）
    exp5  跨任务泛化（双模型 × 三任务 L2）
    exp6  云端差距收敛（云端 vs 本地SALAD）
  参数消融:
    exp7  温度消融（0/0.3/0.7 × ExpA L1）
  新创意实验:
    exp10 规模梯度（0.8B→1.5B→3B→4B→7B→9B→14B→35B 8点曲线）
    exp13 SALAD-L3 完整堆叠（L2 + FSSR RAG）
    exp14 Section Markers（[ROLE][TASK][INPUT][RULE][FORMAT]）
    exp15 Sequential Decompose（两步推理：类型→严重度）
    exp16 Domain-Vocabulary Anchor（领域词汇对照表）

运行方式：
  python scripts/run_full_salad_experiment_v2.py                    # 全量运行
  python scripts/run_full_salad_experiment_v2.py --groups exp1,exp2 # 指定组
  python scripts/run_full_salad_experiment_v2.py --max-samples 10   # 快速验证
  python scripts/run_full_salad_experiment_v2.py --resume           # 断点续传
  python scripts/run_full_salad_experiment_v2.py --models qwen2.5:1.5b
  python scripts/run_full_salad_experiment_v2.py --stats            # 运行完毕后输出统计检验
"""

import os
import sys
import json
import time
import argparse
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from salad_prompts import (
    EXP_A_SALAD, EXP_B_SALAD, EXP_C_SALAD,
    EXP_A_TRADITIONAL, EXP_B_TRADITIONAL, EXP_C_TRADITIONAL,
    EXP_A_SALAD_MARKER, EXP_A_SALAD_SEQ, EXP_A_SALAD_DVA,
)
from model_caller import ModelCaller
from eval_utils import (
    load_jsonl,
    grade_defect_classification, summarize_defect_results,
    grade_rule_config, summarize_rule_results,
    grade_dispatch_parsing, summarize_dispatch_results,
)
from rag_engine import RAGEngine

# ===========================================================
# 路径常量（v2：扩充数据集）
# ===========================================================

DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output" / "v2_experiment"

DATASETS = {
    "exp_a": DATA_DIR / "power_qa" / "defect_classification_200.jsonl",
    "exp_b": DATA_DIR / "power_ts" / "dq_scenarios_120.jsonl",
    "exp_c": DATA_DIR / "power_dispatch" / "dispatch_cmds_200.jsonl",
}

KNOWLEDGE_BASES = {
    "exp_a": (DATA_DIR / "knowledge_base" / "defect_standard.json", "defect"),
    "exp_b": (DATA_DIR / "knowledge_base" / "dq_rules.json", "dq_rules"),
    "exp_c": (DATA_DIR / "knowledge_base" / "dispatch_regulations.json", "dispatch"),
}

PROMPT_REGISTRY = {
    "EXP_A_SALAD": EXP_A_SALAD,
    "EXP_B_SALAD": EXP_B_SALAD,
    "EXP_C_SALAD": EXP_C_SALAD,
    "EXP_A_TRADITIONAL": EXP_A_TRADITIONAL,
    "EXP_B_TRADITIONAL": EXP_B_TRADITIONAL,
    "EXP_C_TRADITIONAL": EXP_C_TRADITIONAL,
    "EXP_A_SALAD_MARKER": EXP_A_SALAD_MARKER,
    "EXP_A_SALAD_DVA": EXP_A_SALAD_DVA,
}

# ===========================================================
# 模型列表（v2：扩充7B/9B/14B，移除不兼容的 glm-4.7-flash）
# ===========================================================

LOCAL_MODELS_PRIMARY = ["qwen2.5:1.5b"]

# 跨模型验证的全量模型池（按规模升序）
LOCAL_MODELS_ALL = [
    "qwen3.5:0.8b",        # 0.8B 超微
    "qwen2.5:1.5b",        # 1.5B 主力实验模型
    "deepseek-r1:1.5b",    # 1.5B 推理增强
    "qwen2.5-coder:1.5b",  # 1.5B 代码优化
    "smollm:1.7b",         # 1.7B 轻量
    "gemma2:2b",           # 2B 国际对照
    "qwen2.5:3b",          # 3B 规模梯度
    "llama3.2:3b",         # 3B Llama 架构
    "phi4-mini:3.8b",      # 4B Microsoft Phi
    "deepseek-r1:7b",      # 7B 推理增强
    "llama3.1:8b",         # 8B Llama Meta
    "qwen3.5:9b",          # 9B Qwen 规模梯度
    "qwen2.5:14b",         # 14B 规模梯度
    "deepseek-r1:14b",     # 14B 推理增强
]

# 规模梯度实验专用模型（8点曲线：覆盖 0.8B→35B）
SCALE_GRADIENT_MODELS = [
    ("qwen3.5:0.8b",    "0.8B"),
    ("qwen2.5:1.5b",    "1.5B"),
    ("qwen2.5:3b",      "3B"),
    ("phi4-mini:3.8b",  "4B"),
    ("deepseek-r1:7b",  "7B"),
    ("qwen3.5:9b",      "9B"),
    ("qwen2.5:14b",     "14B"),
    ("qwen3.6:35b",     "35B"),
]

LOCAL_LARGE_MODELS = [
    "qwen3.6:35b",
    "qwen3.5:35b",
]

CLOUD_MODELS = ["deepseek-v4-pro", "kimi-k2.6"]

# ===========================================================
# 实验组定义
# ===========================================================

EXPERIMENT_GROUPS = {

    # ===== 核心对比实验 =====

    "exp1_traditional_poison": {
        "description": "传统PE毒药效应：CoT+RAG 随层级递增导致小模型性能递减",
        "runs": [
            ("qwen2.5:1.5b", "exp_a", "EXP_A_TRADITIONAL", 0, 0.7, False),
            ("qwen2.5:1.5b", "exp_a", "EXP_A_TRADITIONAL", 1, 0.7, False),
            ("qwen2.5:1.5b", "exp_a", "EXP_A_TRADITIONAL", 2, 0.7, True),
            ("qwen2.5:1.5b", "exp_b", "EXP_B_TRADITIONAL", 0, 0.7, False),
            ("qwen2.5:1.5b", "exp_b", "EXP_B_TRADITIONAL", 1, 0.7, False),
            ("qwen2.5:1.5b", "exp_b", "EXP_B_TRADITIONAL", 2, 0.7, True),
            ("qwen2.5:1.5b", "exp_c", "EXP_C_TRADITIONAL", 0, 0.7, False),
            ("qwen2.5:1.5b", "exp_c", "EXP_C_TRADITIONAL", 1, 0.7, False),
            ("qwen2.5:1.5b", "exp_c", "EXP_C_TRADITIONAL", 2, 0.7, True),
        ],
    },

    "exp2_salad_ladder": {
        "description": "SALAD阶梯效应：L0→L1→L2 三级递增证明SALAD有效性",
        "runs": [
            ("qwen2.5:1.5b", "exp_a", "EXP_A_SALAD", 0, 0.0, False),
            ("qwen2.5:1.5b", "exp_a", "EXP_A_SALAD", 1, 0.0, False),
            ("qwen2.5:1.5b", "exp_a", "EXP_A_SALAD", 2, 0.0, False),
            ("qwen2.5:1.5b", "exp_b", "EXP_B_SALAD", 0, 0.0, False),
            ("qwen2.5:1.5b", "exp_b", "EXP_B_SALAD", 1, 0.0, False),
            ("qwen2.5:1.5b", "exp_b", "EXP_B_SALAD", 2, 0.0, False),
            ("qwen2.5:1.5b", "exp_c", "EXP_C_SALAD", 0, 0.0, False),
            ("qwen2.5:1.5b", "exp_c", "EXP_C_SALAD", 1, 0.0, False),
            ("qwen2.5:1.5b", "exp_c", "EXP_C_SALAD", 2, 0.0, False),
        ],
    },

    "exp3_component_ablation": {
        "description": "组件消融：逐步拆解SALAD各组件的边际贡献",
        "runs": [
            # Step 0: 传统基线 temp=0.7（起点）
            ("qwen2.5:1.5b", "exp_a", "EXP_A_TRADITIONAL", 0, 0.7, False),
            # Step 1: 传统基线 temp=0.0（+Greedy解码）
            ("qwen2.5:1.5b", "exp_a", "EXP_A_TRADITIONAL", 0, 0.0, False),
            # Step 2: SALAD L0 极简（+DPC精简，无约束）
            ("qwen2.5:1.5b", "exp_a", "EXP_A_SALAD", 0, 0.0, False),
            # Step 3: SALAD L1（+SLA末尾锚定）
            ("qwen2.5:1.5b", "exp_a", "EXP_A_SALAD", 1, 0.0, False),
            # Step 4: SALAD L2（+判断标准前置）
            ("qwen2.5:1.5b", "exp_a", "EXP_A_SALAD", 2, 0.0, False),
            # Step 5: SALAD L3（+FSSR RAG 增强）
            ("qwen2.5:1.5b", "exp_a", "EXP_A_SALAD", 3, 0.0, True),
        ],
    },

    "exp4_cross_model": {
        "description": "跨模型验证：SALAD增益与模型架构无关（含1.5B-14B全量模型）",
        "runs": [
            *[
                run
                for model in LOCAL_MODELS_ALL
                for run in [
                    (model, "exp_a", "EXP_A_SALAD", 0, 0.0, False),
                    (model, "exp_a", "EXP_A_SALAD", 2, 0.0, False),
                ]
            ]
        ],
    },

    "exp5_cross_task": {
        "description": "跨任务泛化：SALAD在三类电力任务上均有效",
        "runs": [
            *[
                run
                for model in ["qwen2.5:1.5b", "deepseek-r1:1.5b"]
                for run in [
                    (model, "exp_a", "EXP_A_SALAD", 2, 0.0, False),
                    (model, "exp_b", "EXP_B_SALAD", 2, 0.0, False),
                    (model, "exp_c", "EXP_C_SALAD", 2, 0.0, False),
                ]
            ]
        ],
    },

    "exp6_cloud_gap": {
        "description": "差距收敛：1.5B SALAD-L2 vs 本地35B vs 云端大模型",
        "runs": [
            ("qwen2.5:1.5b", "exp_a", "EXP_A_SALAD", 2, 0.0, False),
            ("qwen2.5:1.5b", "exp_b", "EXP_B_SALAD", 2, 0.0, False),
            ("qwen2.5:1.5b", "exp_c", "EXP_C_SALAD", 2, 0.0, False),
            ("qwen3.6:35b", "exp_a", "EXP_A_SALAD", 0, 0.0, False),
            ("qwen3.6:35b", "exp_b", "EXP_B_SALAD", 0, 0.0, False),
            ("qwen3.6:35b", "exp_c", "EXP_C_SALAD", 0, 0.0, False),
            ("deepseek-v4-pro", "exp_a", "EXP_A_SALAD", 0, 0.0, False),
            ("deepseek-v4-pro", "exp_b", "EXP_B_SALAD", 0, 0.0, False),
            ("deepseek-v4-pro", "exp_c", "EXP_C_SALAD", 0, 0.0, False),
            ("kimi-k2.6", "exp_a", "EXP_A_SALAD", 0, 0.0, False),
            ("kimi-k2.6", "exp_b", "EXP_B_SALAD", 0, 0.0, False),
            ("kimi-k2.6", "exp_c", "EXP_C_SALAD", 0, 0.0, False),
        ],
    },

    "exp7_temperature": {
        "description": "温度消融：Greedy解码(0)对小模型稳定性的影响",
        "runs": [
            ("qwen2.5:1.5b", "exp_a", "EXP_A_SALAD", 1, 0.0, False),
            ("qwen2.5:1.5b", "exp_a", "EXP_A_SALAD", 1, 0.3, False),
            ("qwen2.5:1.5b", "exp_a", "EXP_A_SALAD", 1, 0.7, False),
            ("qwen2.5:1.5b", "exp_a", "EXP_A_TRADITIONAL", 0, 0.0, False),
            ("qwen2.5:1.5b", "exp_a", "EXP_A_TRADITIONAL", 0, 0.3, False),
            ("qwen2.5:1.5b", "exp_a", "EXP_A_TRADITIONAL", 0, 0.7, False),
        ],
    },

    "exp10_scale_gradient": {
        "description": "规模梯度实验：0.8B→1.5B→3B→4B→7B→9B→14B→35B 完整8点SALAD效果曲线",
        "runs": [
            *[
                run
                for model, _label in SCALE_GRADIENT_MODELS
                for run in [
                    (model, "exp_a", "EXP_A_SALAD", 0, 0.0, False),
                    (model, "exp_a", "EXP_A_SALAD", 2, 0.0, False),
                ]
            ]
        ],
    },

    "exp13_salad_l3_rag": {
        "description": "新实验：SALAD-L3完整堆叠（Criteria + FSSR RAG），验证Augmentation增益",
        "runs": [
            ("qwen2.5:1.5b", "exp_a", "EXP_A_SALAD", 2, 0.0, False),
            ("qwen2.5:1.5b", "exp_a", "EXP_A_SALAD", 3, 0.0, True),
            ("qwen2.5:1.5b", "exp_b", "EXP_B_SALAD", 2, 0.0, False),
            ("qwen2.5:1.5b", "exp_b", "EXP_B_SALAD", 3, 0.0, True),
            ("qwen2.5:1.5b", "exp_c", "EXP_C_SALAD", 2, 0.0, False),
            ("qwen2.5:1.5b", "exp_c", "EXP_C_SALAD", 3, 0.0, True),
        ],
    },

    "exp14_section_markers": {
        "description": "新实验：Section Markers [ROLE][TASK][INPUT][RULE] 与 SALAD 的对比",
        "runs": [
            ("qwen2.5:1.5b", "exp_a", "EXP_A_SALAD", 0, 0.0, False),
            ("qwen2.5:1.5b", "exp_a", "EXP_A_SALAD", 1, 0.0, False),
            ("qwen2.5:1.5b", "exp_a", "EXP_A_SALAD", 2, 0.0, False),
            ("qwen2.5:1.5b", "exp_a", "EXP_A_SALAD_MARKER", 0, 0.0, False),
            ("qwen2.5:1.5b", "exp_a", "EXP_A_SALAD_MARKER", 1, 0.0, False),
            ("qwen2.5:1.5b", "exp_a", "EXP_A_SALAD_MARKER", 2, 0.0, False),
        ],
    },

    "exp15_sequential_decompose": {
        "description": "新实验：两步顺序分解（先类型后严重度）降低认知负荷",
        "runs": [
            ("qwen2.5:1.5b", "exp_a", "EXP_A_SALAD", 2, 0.0, False),
        ],
        "extra_seq": [
            ("qwen2.5:1.5b", "exp_a", 0.0),
        ],
    },

    "exp16_domain_vocab_anchor": {
        "description": "新实验：Domain-Vocabulary Anchor，领域词汇对照表辅助理解",
        "runs": [
            ("qwen2.5:1.5b", "exp_a", "EXP_A_SALAD", 0, 0.0, False),
            ("qwen2.5:1.5b", "exp_a", "EXP_A_SALAD", 2, 0.0, False),
            ("qwen2.5:1.5b", "exp_a", "EXP_A_SALAD_DVA", 0, 0.0, False),
            ("qwen2.5:1.5b", "exp_a", "EXP_A_SALAD_DVA", 1, 0.0, False),
            ("qwen2.5:1.5b", "exp_a", "EXP_A_SALAD_DVA", 2, 0.0, False),
        ],
    },
}


# ===========================================================
# 工具函数
# ===========================================================

def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("salad_exp_v2")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def safe_model_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_").replace(".", "_")


def get_run_id(model: str, exp: str, prompt_dict: str, level, temperature: float) -> str:
    temp_str = str(temperature).replace(".", "p")
    lvl_str = str(level)
    return f"{safe_model_name(model)}__{exp}__{prompt_dict.lower()}_L{lvl_str}__t{temp_str}"


def load_data(exp_name: str, max_samples: int = -1) -> List[Dict]:
    path = DATASETS[exp_name]
    if not path.exists():
        raise FileNotFoundError(f"数据集不存在: {path}")
    data = load_jsonl(str(path))
    if max_samples > 0:
        data = data[:max_samples]
    return data


def build_prompt(
    sample: Dict,
    exp_name: str,
    prompt_dict: Dict,
    level: int,
    rag_engine: Optional[RAGEngine],
    use_rag: bool,
) -> Tuple[str, str]:
    config = prompt_dict[level]
    system = config["system"]
    template = config["template"]

    fmt_kwargs = {}
    if exp_name == "exp_a":
        fmt_kwargs["description"] = sample.get("description", "")
    elif exp_name == "exp_b":
        fmt_kwargs["scenario_description"] = sample.get("description", "")
        stats = sample.get("statistics", {})
        fmt_kwargs["statistics"] = json.dumps(stats, ensure_ascii=False)
    elif exp_name == "exp_c":
        fmt_kwargs["command"] = sample.get("command", "")

    rag_context = ""
    if use_rag and rag_engine is not None:
        kb_path, kb_name = KNOWLEDGE_BASES[exp_name]
        query = fmt_kwargs.get("description", fmt_kwargs.get("command", fmt_kwargs.get("scenario_description", "")))
        rag_context = rag_engine.retrieve_and_format_salad(
            query, kb_name,
            top_k=1,
            similarity_threshold=0.65,
            max_chars=150,
        )
    fmt_kwargs["rag_context"] = rag_context if rag_context else "（无相关标准条款）"

    user_prompt = template.format(**fmt_kwargs)
    return system, user_prompt


def grade_sample(model_answer: str, sample: Dict, exp_name: str) -> Dict:
    if exp_name == "exp_a":
        correct = {
            "defect_type": sample.get("defect_type", ""),
            "severity": sample.get("severity", ""),
        }
        return grade_defect_classification(model_answer, correct)
    elif exp_name == "exp_b":
        correct = {
            "rule_type": sample.get("correct_rule_type", ""),
            "field": sample.get("correct_field", ""),
        }
        return grade_rule_config(model_answer, correct)
    elif exp_name == "exp_c":
        correct = sample.get("correct", {})
        return grade_dispatch_parsing(model_answer, correct)
    return {}


def summarize_results(results: List[Dict], exp_name: str) -> Dict:
    if exp_name == "exp_a":
        return summarize_defect_results(results)
    elif exp_name == "exp_b":
        return summarize_rule_results(results)
    elif exp_name == "exp_c":
        return summarize_dispatch_results(results)
    return {}


def save_results(results: List[Dict], metrics: Dict, out_dir: Path, run_id: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / f"{run_id}__samples.jsonl"
    with open(samples_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    metrics_path = out_dir / f"{run_id}__metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return samples_path, metrics_path


def check_run_done(out_dir: Path, run_id: str) -> bool:
    return (out_dir / f"{run_id}__metrics.json").exists()


def setup_rag_engines() -> Dict[str, RAGEngine]:
    engines = {}
    for exp_name, (kb_path, kb_name) in KNOWLEDGE_BASES.items():
        engine = RAGEngine()
        if kb_path.exists():
            try:
                engine.load_knowledge_base(str(kb_path), kb_name)
                engines[exp_name] = engine
            except Exception as e:
                print(f"[WARNING] 知识库加载失败 {exp_name}: {e}")
    return engines


# ===========================================================
# 顺序分解实验（特殊处理）
# ===========================================================

def run_sequential_decompose(
    model: str,
    exp_name: str,
    temperature: float,
    data: List[Dict],
    caller: ModelCaller,
    out_dir: Path,
    logger: logging.Logger,
    resume: bool = False,
    api_delay: float = 0.5,
) -> Dict:
    run_id = f"{safe_model_name(model)}__{exp_name}__salad_seq__t0p0"

    if resume and check_run_done(out_dir, run_id):
        logger.info(f"  [SKIP] {run_id} 已完成，断点续传跳过")
        with open(out_dir / f"{run_id}__metrics.json", encoding="utf-8") as f:
            return json.load(f)

    step1_cfg = EXP_A_SALAD_SEQ["step1"]
    step2_cfg = EXP_A_SALAD_SEQ["step2"]
    results = []
    total = len(data)

    logger.info(f"  [SEQ] {model} / {exp_name} / Sequential / temp={temperature} / N={total}")

    for i, sample in enumerate(data):
        t0 = time.time()
        sample_id = sample.get("id", f"S{i:03d}")

        try:
            step1_prompt = step1_cfg["template"].format(
                description=sample.get("description", "")
            )
            r1 = caller.call(
                model_name=model,
                prompt=step1_prompt,
                temperature=temperature,
                system_prompt=step1_cfg["system"],
                max_tokens=64,
            )
            from eval_utils import parse_defect_answer
            parsed1 = parse_defect_answer(r1["content"])
            pred_type = parsed1.get("defect_type", "过热")

            is_api = model in ("deepseek-v4-pro", "kimi-k2.6", "minimax-m2.7")
            if is_api:
                time.sleep(api_delay)

            step2_prompt = step2_cfg["template"].format(
                description=sample.get("description", ""),
                defect_type=pred_type,
            )
            r2 = caller.call(
                model_name=model,
                prompt=step2_prompt,
                temperature=temperature,
                system_prompt=step2_cfg["system"],
                max_tokens=64,
            )
            if is_api:
                time.sleep(api_delay)

            combined_answer = json.dumps(
                {"defect_type": pred_type, "severity": parse_defect_answer(r2["content"]).get("severity", "")},
                ensure_ascii=False,
            )

            correct = {
                "defect_type": sample.get("defect_type", ""),
                "severity": sample.get("severity", ""),
            }
            score = grade_defect_classification(combined_answer, correct)
            latency = (time.time() - t0) * 1000

            rec = {
                **score,
                "sample_id": sample_id,
                "model": model,
                "exp": exp_name,
                "method": "seq_decompose",
                "step1_answer": r1["content"][:200],
                "step2_answer": r2["content"][:200],
                "pred_type_step1": pred_type,
                "device_type": sample.get("device_type", ""),
                "correct_defect_type": correct["defect_type"],
                "correct_severity": correct["severity"],
                "latency_ms": latency,
                "total_tokens": r1["total_tokens"] + r2["total_tokens"],
                "cost_usd": r1["cost_usd"] + r2["cost_usd"],
            }
            results.append(rec)

        except Exception as e:
            logger.warning(f"  [ERROR] sample {sample_id}: {e}")
            results.append({
                "sample_id": sample_id,
                "model": model,
                "method": "seq_decompose",
                "error": str(e),
                "type_score": 0.0,
                "severity_score": 0.0,
                "overall_score": 0.0,
            })

        if (i + 1) % 10 == 0 or (i + 1) == total:
            done = [r for r in results if "error" not in r]
            avg = sum(r["overall_score"] for r in done) / len(done) if done else 0
            logger.info(f"    [{i+1}/{total}] avg_score={avg:.3f}")

    metrics = summarize_results(results, exp_name)
    metrics.update({
        "model": model,
        "exp": exp_name,
        "method": "seq_decompose",
        "run_id": run_id,
        "temperature": temperature,
        "n_samples": total,
        "timestamp": datetime.now().isoformat(),
        "total_cost_usd": sum(r.get("cost_usd", 0) for r in results),
        "avg_latency_ms": sum(r.get("latency_ms", 0) for r in results) / max(len(results), 1),
    })

    save_results(results, metrics, out_dir, run_id)
    logger.info(f"  [SEQ DONE] overall_accuracy={metrics.get('overall_accuracy', 0):.3f}")
    return metrics


# ===========================================================
# 单次运行
# ===========================================================

def run_single(
    model: str,
    exp_name: str,
    prompt_dict_key: str,
    level: int,
    temperature: float,
    use_rag: bool,
    data: List[Dict],
    caller: ModelCaller,
    rag_engines: Dict[str, RAGEngine],
    out_dir: Path,
    logger: logging.Logger,
    resume: bool = False,
    api_delay: float = 0.5,
) -> Dict:

    prompt_dict = PROMPT_REGISTRY.get(prompt_dict_key)
    if prompt_dict is None or level not in prompt_dict:
        logger.error(f"  [ERROR] prompt_dict_key={prompt_dict_key} level={level} 不存在")
        return {}

    run_id = get_run_id(model, exp_name, prompt_dict_key, level, temperature)

    if resume and check_run_done(out_dir, run_id):
        logger.info(f"  [SKIP] {run_id} 已完成，断点续传跳过")
        with open(out_dir / f"{run_id}__metrics.json", encoding="utf-8") as f:
            return json.load(f)

    rag_engine = rag_engines.get(exp_name) if use_rag else None
    results = []
    total = len(data)
    prompt_name = prompt_dict[level].get("name", f"L{level}")

    logger.info(f"  [RUN] {model} / {exp_name} / {prompt_dict_key} / L{level}({prompt_name}) / "
                f"temp={temperature} / rag={use_rag} / N={total}")

    is_api = model in ("deepseek-v4-pro", "kimi-k2.6", "minimax-m2.7")

    for i, sample in enumerate(data):
        t0 = time.time()
        sample_id = sample.get("id", f"S{i:03d}")

        try:
            system, user_prompt = build_prompt(
                sample, exp_name, prompt_dict, level, rag_engine, use_rag
            )
            response = caller.call(
                model_name=model,
                prompt=user_prompt,
                temperature=temperature,
                system_prompt=system,
                max_tokens=512,
            )
            model_answer = response["content"]
            score = grade_sample(model_answer, sample, exp_name)
            latency = (time.time() - t0) * 1000

            rec = {
                **score,
                "sample_id": sample_id,
                "model": model,
                "exp": exp_name,
                "prompt_dict": prompt_dict_key,
                "level": level,
                "prompt_name": prompt_name,
                "temperature": temperature,
                "use_rag": use_rag,
                "model_answer": model_answer[:500],
                "input_tokens": response["input_tokens"],
                "output_tokens": response["output_tokens"],
                "total_tokens": response["total_tokens"],
                "latency_ms": latency,
                "cost_usd": response["cost_usd"],
            }

            if exp_name == "exp_a":
                rec.update({
                    "device_type": sample.get("device_type", ""),
                    "correct_defect_type": sample.get("defect_type", ""),
                    "correct_severity": sample.get("severity", ""),
                    "description": sample.get("description", "")[:200],
                })
            elif exp_name == "exp_b":
                rec.update({
                    "correct_rule_type": sample.get("correct_rule_type", ""),
                    "correct_field": sample.get("correct_field", ""),
                })
            elif exp_name == "exp_c":
                correct = sample.get("correct", {})
                rec.update({
                    "correct_obj": correct.get("operation_object", ""),
                    "correct_type": correct.get("operation_type", ""),
                    "command": sample.get("command", "")[:200],
                })

            results.append(rec)

            if is_api:
                time.sleep(api_delay)

        except Exception as e:
            logger.warning(f"    [ERROR] sample {sample_id}: {e}")
            logger.debug(traceback.format_exc())
            results.append({
                "sample_id": sample_id,
                "model": model,
                "exp": exp_name,
                "prompt_dict": prompt_dict_key,
                "level": level,
                "error": str(e),
                "overall_score": 0.0,
                "type_score": 0.0,
                "severity_score": 0.0,
                "json_valid": 0.0,
                "obj_score": 0.0,
            })

        if (i + 1) % 20 == 0 or (i + 1) == total:
            valid = [r for r in results if "error" not in r]
            avg = sum(r.get("overall_score", 0) for r in valid) / max(len(valid), 1)
            logger.info(f"    [{i+1}/{total}] avg_score={avg:.3f}  cost=${sum(r.get('cost_usd',0) for r in results):.4f}")

    metrics = summarize_results(results, exp_name)
    metrics.update({
        "model": model,
        "exp": exp_name,
        "prompt_dict": prompt_dict_key,
        "level": level,
        "prompt_name": prompt_name,
        "temperature": temperature,
        "use_rag": use_rag,
        "run_id": run_id,
        "n_samples": total,
        "n_errors": sum(1 for r in results if "error" in r),
        "timestamp": datetime.now().isoformat(),
        "total_cost_usd": sum(r.get("cost_usd", 0) for r in results),
        "avg_latency_ms": sum(r.get("latency_ms", 0) for r in results) / max(len(results), 1),
        "avg_input_tokens": sum(r.get("input_tokens", 0) for r in results) / max(len(results), 1),
        "avg_output_tokens": sum(r.get("output_tokens", 0) for r in results) / max(len(results), 1),
    })

    _, metrics_path = save_results(results, metrics, out_dir, run_id)
    key_metric = metrics.get("overall_accuracy", metrics.get("avg_overall_score", 0))
    logger.info(f"  [DONE] overall={key_metric:.3f}  saved → {metrics_path.name}")
    return metrics


# ===========================================================
# 模型可用性检查
# ===========================================================

def check_model_available(model: str, caller: ModelCaller) -> bool:
    try:
        is_api = model in ("deepseek-v4-pro", "kimi-k2.6", "minimax-m2.7")
        if not is_api:
            import requests
            resp = requests.get(f"{caller.ollama_host}/api/tags", timeout=5)
            if resp.status_code == 200:
                tags = resp.json().get("models", [])
                available = [t.get("name", "") for t in tags]
                return any(model in m or m.startswith(model.split(":")[0]) for m in available)
            return False
        result = caller.call(model, "Hi", max_tokens=5, temperature=0.0, timeout=15)
        return bool(result.get("content") or result.get("total_tokens", 0) > 0)
    except Exception:
        return False


# ===========================================================
# 实验组运行器
# ===========================================================

def run_group(
    group_name: str,
    group_config: Dict,
    caller: ModelCaller,
    rag_engines: Dict[str, RAGEngine],
    data_cache: Dict[str, List[Dict]],
    out_root: Path,
    logger: logging.Logger,
    resume: bool,
    max_samples: int,
    available_models: set,
    api_delay: float,
) -> List[Dict]:

    group_dir = out_root / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    logger.info(f"\n{'='*60}")
    logger.info(f"[GROUP] {group_name}")
    logger.info(f"  {group_config['description']}")
    logger.info(f"{'='*60}")

    for run_spec in group_config.get("runs", []):
        model, exp_name, prompt_dict_key, level, temperature, use_rag = run_spec

        if model not in available_models:
            logger.warning(f"  [SKIP] 模型不可用: {model}")
            continue

        if exp_name not in data_cache:
            try:
                data_cache[exp_name] = load_data(exp_name, max_samples)
                logger.info(f"  [DATA] 已加载 {exp_name}: {len(data_cache[exp_name])} 条")
            except FileNotFoundError as e:
                logger.error(f"  [ERROR] {e}")
                continue

        m = run_single(
            model=model,
            exp_name=exp_name,
            prompt_dict_key=prompt_dict_key,
            level=level,
            temperature=temperature,
            use_rag=use_rag,
            data=data_cache[exp_name],
            caller=caller,
            rag_engines=rag_engines,
            out_dir=group_dir,
            logger=logger,
            resume=resume,
            api_delay=api_delay,
        )
        if m:
            m["group"] = group_name
            all_metrics.append(m)

    for seq_spec in group_config.get("extra_seq", []):
        model, exp_name, temperature = seq_spec

        if model not in available_models:
            logger.warning(f"  [SKIP] 模型不可用: {model}")
            continue

        if exp_name not in data_cache:
            try:
                data_cache[exp_name] = load_data(exp_name, max_samples)
            except FileNotFoundError as e:
                logger.error(f"  [ERROR] {e}")
                continue

        m = run_sequential_decompose(
            model=model,
            exp_name=exp_name,
            temperature=temperature,
            data=data_cache[exp_name],
            caller=caller,
            out_dir=group_dir,
            logger=logger,
            resume=resume,
            api_delay=api_delay,
        )
        if m:
            m["group"] = group_name
            all_metrics.append(m)

    group_summary_path = group_dir / "_group_summary.json"
    with open(group_summary_path, "w", encoding="utf-8") as f:
        json.dump({"group": group_name, "metrics": all_metrics}, f, ensure_ascii=False, indent=2)
    logger.info(f"[GROUP DONE] {group_name}  runs={len(all_metrics)}  → {group_summary_path}")

    return all_metrics


# ===========================================================
# 统计检验汇总（实验完成后可选运行）
# ===========================================================

def run_statistical_summary(out_root: Path, logger: logging.Logger):
    """对 exp2 的 L0/L1/L2 样本文件运行 McNemar 和 Bootstrap CI 检验"""
    try:
        from statistical_tests import (
            bootstrap_ci, mcnemar_from_accuracy
        )
    except ImportError:
        logger.warning("[STATS] statistical_tests 模块未找到，跳过统计检验")
        return

    logger.info("\n===== 统计检验 (McNemar + Bootstrap CI) =====")
    exp2_dir = out_root / "exp2_salad_ladder"
    if not exp2_dir.exists():
        logger.warning(f"[STATS] exp2 目录不存在: {exp2_dir}")
        return

    # 收集 exp_a 三个级别的样本文件
    import glob as globmod
    for exp_name in ["exp_a", "exp_b", "exp_c"]:
        samples_files = {}
        for level in [0, 1, 2]:
            pattern = str(exp2_dir / f"*__{exp_name}__exp_*_salad_L{level}__*__samples.jsonl")
            matches = globmod.glob(pattern)
            if not matches:
                pattern = str(exp2_dir / f"*{exp_name}*L{level}*samples.jsonl")
                matches = globmod.glob(pattern)
            if matches:
                samples_files[level] = matches[0]

        if len(samples_files) < 2:
            logger.info(f"[STATS] {exp_name}: 样本文件不足，跳过")
            continue

        levels_data = {}
        for level, fpath in samples_files.items():
            scores = []
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line.strip())
                        scores.append(r.get("overall_score", 0.0))
                    except Exception:
                        pass
            if scores:
                levels_data[level] = scores

        for level, scores in sorted(levels_data.items()):
            n = len(scores)
            acc = sum(scores) / n if n else 0
            _, ci_lo, ci_hi = bootstrap_ci(scores)
            logger.info(f"  {exp_name} L{level}: acc={acc:.3f}  95%CI=[{ci_lo:.3f}, {ci_hi:.3f}]  N={n}")

        # McNemar 检验 L0 vs L2（若两者均存在）
        if 0 in levels_data and 2 in levels_data:
            n = min(len(levels_data[0]), len(levels_data[2]))
            acc0 = sum(levels_data[0][:n]) / n
            acc2 = sum(levels_data[2][:n]) / n
            result = mcnemar_from_accuracy(acc0, acc2, n)
            logger.info(f"  {exp_name} McNemar L0 vs L2: Z={result['z_stat']:.3f}  p={result['p_value']:.4f}  "
                        f"{'*SIGNIFICANT*' if result['significant'] else 'not significant'}")

    # 保存统计报告
    stats_path = out_root / "statistical_summary.md"
    logger.info(f"[STATS] 详细报告见: {stats_path}")


# ===========================================================
# 主函数
# ===========================================================

def main():
    parser = argparse.ArgumentParser(description="SALAD 全量消融实验运行器 v2（扩充数据集）")
    parser.add_argument("--groups", default="all",
                        help="要运行的实验组，逗号分隔（all 表示全部），例: exp1,exp2,exp10")
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="每个数据集最多使用多少条样本（-1=全量）")
    parser.add_argument("--resume", action="store_true",
                        help="断点续传，跳过已完成的 run_id")
    parser.add_argument("--models", default="",
                        help="只使用这些模型（逗号分隔）")
    parser.add_argument("--output-dir", default="",
                        help="输出根目录（默认: output/v2_experiment）")
    parser.add_argument("--api-delay", type=float, default=0.8,
                        help="云端API调用间隔秒数")
    parser.add_argument("--skip-rag-init", action="store_true",
                        help="跳过RAG引擎初始化（L3实验会降级为无RAG）")
    parser.add_argument("--stats", action="store_true",
                        help="实验完成后运行 McNemar + Bootstrap CI 统计检验")
    args = parser.parse_args()

    out_root = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = out_root / f"run_{ts}.log"
    logger = setup_logging(log_file)
    logger.info(f"SALAD 全量实验 v2 启动  output={out_root}  log={log_file}")
    logger.info(f"数据集: ExpA={DATASETS['exp_a'].name}  "
                f"ExpB={DATASETS['exp_b'].name}  "
                f"ExpC={DATASETS['exp_c'].name}")
    logger.info(f"参数: groups={args.groups} max_samples={args.max_samples} "
                f"resume={args.resume} api_delay={args.api_delay} stats={args.stats}")

    caller = ModelCaller()

    if args.groups.strip().lower() == "all":
        selected_groups = list(EXPERIMENT_GROUPS.keys())
    else:
        selected_groups = [g.strip() for g in args.groups.split(",")]
        invalid = [g for g in selected_groups if g not in EXPERIMENT_GROUPS]
        if invalid:
            logger.error(f"未知实验组: {invalid}")
            logger.info(f"可用实验组: {list(EXPERIMENT_GROUPS.keys())}")
            sys.exit(1)

    logger.info(f"将运行实验组: {selected_groups}")

    override_models = set(m.strip() for m in args.models.split(",") if m.strip())
    all_models_needed = set()
    for gname in selected_groups:
        cfg = EXPERIMENT_GROUPS[gname]
        for run_spec in cfg.get("runs", []):
            all_models_needed.add(run_spec[0])
        for seq_spec in cfg.get("extra_seq", []):
            all_models_needed.add(seq_spec[0])

    if override_models:
        all_models_needed = all_models_needed & override_models

    logger.info(f"正在检查模型可用性: {sorted(all_models_needed)}")
    available_models = set()
    for model in sorted(all_models_needed):
        if check_model_available(model, caller):
            available_models.add(model)
            logger.info(f"  [OK] {model}")
        else:
            logger.warning(f"  [X] {model} (不可用，将跳过)")

    if not available_models:
        logger.error("没有可用的模型，请检查 Ollama 或 API Key 配置")
        sys.exit(1)

    rag_engines = {}
    if not args.skip_rag_init:
        logger.info("正在初始化 RAG 引擎（加载知识库向量索引）...")
        rag_engines = setup_rag_engines()
        logger.info(f"RAG 引擎已加载: {list(rag_engines.keys())}")
    else:
        logger.info("跳过 RAG 引擎初始化（--skip-rag-init）")

    data_cache: Dict[str, List[Dict]] = {}

    master_metrics = []
    for group_name in selected_groups:
        if group_name not in EXPERIMENT_GROUPS:
            logger.warning(f"跳过未知实验组: {group_name}")
            continue
        group_config = EXPERIMENT_GROUPS[group_name]
        group_metrics = run_group(
            group_name=group_name,
            group_config=group_config,
            caller=caller,
            rag_engines=rag_engines,
            data_cache=data_cache,
            out_root=out_root,
            logger=logger,
            resume=args.resume,
            max_samples=args.max_samples,
            available_models=available_models,
            api_delay=args.api_delay,
        )
        master_metrics.extend(group_metrics)

    master_path = out_root / f"master_summary_{ts}.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_timestamp": ts,
            "version": "v2",
            "datasets": {k: str(v) for k, v in DATASETS.items()},
            "groups_run": selected_groups,
            "total_runs": len(master_metrics),
            "available_models": sorted(available_models),
            "metrics": master_metrics,
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"全量实验 v2 完成！总计 {len(master_metrics)} 个运行")
    logger.info(f"汇总报告: {master_path}")
    logger.info(f"日志文件: {log_file}")
    logger.info(f"{'='*60}")

    logger.info("\n===== 实验结果速览 =====")
    for m in sorted(master_metrics, key=lambda x: (x.get("group", ""), x.get("exp", ""), x.get("level", 0))):
        acc = m.get("overall_accuracy", m.get("avg_overall_score", 0))
        logger.info(
            f"  {m.get('group',''):<30} {m.get('model',''):<25} "
            f"{m.get('exp',''):<8} L{m.get('level','-'):<3} "
            f"t={m.get('temperature',0):.1f}  acc={acc:.3f}"
        )

    if args.stats:
        run_statistical_summary(out_root, logger)


if __name__ == "__main__":
    main()
