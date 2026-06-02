#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCI论文级统计显著性检验工具集

实现McNemar配对检验、Bootstrap置信区间、Cohen's Kappa、
Macro-F1/Precision/Recall、MCC（Matthews相关系数）等指标。

无需scipy也可运行：内置正态CDF近似。
"""

import math
import random
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any


# ========================================================
# 内置正态CDF（无需scipy）
# ========================================================

def _norm_cdf(x: float) -> float:
    """标准正态CDF近似，精度满足统计检验需求"""
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422820 * math.exp(-x * x / 2.0)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.7814779 + t * (-1.8212560 + t * 1.3302744))))
    if x > 0:
        return 1.0 - p
    return p


def _chi2_sf(x: float, df: int = 1) -> float:
    """卡方分布生存函数（df=1时用于McNemar检验）"""
    if df == 1:
        return 2.0 * (1.0 - _norm_cdf(math.sqrt(x)))
    # df>1 的近似（Wilson-Hilferty）
    k = df
    if x <= 0:
        return 1.0
    z = ((x / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))
    return 1.0 - _norm_cdf(z)


# ========================================================
# 1. McNemar配对检验
# ========================================================

def mcnemar_test(
    y_true: List[Any],
    model_a_preds: List[Any],
    model_b_preds: List[Any],
    alpha: float = 0.05,
) -> Dict:
    """
    McNemar配对检验：比较两模型在同一数据集上的差异显著性。

    Args:
        y_true: 真实标签列表
        model_a_preds: 模型A的预测列表
        model_b_preds: 模型B的预测列表
        alpha: 显著性水平（默认0.05）

    Returns:
        {
            'b': int,  # A对B错的样本数
            'c': int,  # A错B对的样本数
            'chi2': float,  # 卡方统计量
            'p_value': float,
            'is_significant': bool,
            'interpretation': str,
        }
    """
    n = len(y_true)
    assert len(model_a_preds) == n and len(model_b_preds) == n, "长度不一致"

    b = 0  # A对、B错
    c = 0  # A错、B对

    for gt, pa, pb in zip(y_true, model_a_preds, model_b_preds):
        a_correct = (pa == gt)
        b_correct = (pb == gt)
        if a_correct and not b_correct:
            b += 1
        elif not a_correct and b_correct:
            c += 1

    if b + c == 0:
        return {
            'b': b, 'c': c, 'chi2': 0.0, 'p_value': 1.0,
            'is_significant': False,
            'interpretation': f"两模型预测完全一致，无法区分（b={b}, c={c}）"
        }

    # 使用连续性校正（Yates' correction）
    chi2 = max(0, (abs(b - c) - 1) ** 2) / (b + c)
    p_value = _chi2_sf(chi2, df=1)
    is_sig = p_value < alpha

    interp = (
        f"A vs B：b={b}, c={c}, χ²={chi2:.4f}, p={p_value:.4f} "
        f"({'显著' if is_sig else '不显著'}, α={alpha})"
    )

    return {
        'b': b, 'c': c, 'chi2': round(chi2, 4),
        'p_value': round(p_value, 4),
        'is_significant': is_sig,
        'interpretation': interp,
    }


def mcnemar_from_accuracy(
    acc_a: float, acc_b: float, n: int, alpha: float = 0.05
) -> Dict:
    """
    仅知道准确率和样本量时的配对Z检验近似（适用于无样本级标签时）。

    Z = (acc_a - acc_b) / sqrt((acc_a + acc_b - (acc_a - acc_b)^2) / n)
    """
    diff = acc_a - acc_b
    denom_sq = (acc_a + acc_b - diff ** 2) / n
    if denom_sq <= 0:
        return {'z': 0.0, 'p_value': 1.0, 'is_significant': False,
                'interpretation': "方差为零，无法检验"}
    z = diff / math.sqrt(denom_sq)
    p_value = 2.0 * (1.0 - _norm_cdf(abs(z)))
    is_sig = p_value < alpha
    return {
        'z': round(z, 4), 'p_value': round(p_value, 4),
        'is_significant': is_sig,
        'interpretation': (
            f"Z={z:.4f}, p={p_value:.4f} "
            f"({'显著' if is_sig else '不显著'}, α={alpha}, N={n})"
        )
    }


# ========================================================
# 2. Bootstrap 置信区间
# ========================================================

def bootstrap_ci(
    scores: List[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap重采样置信区间。

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = random.Random(seed)
    n = len(scores)
    if n == 0:
        return 0.0, 0.0, 0.0

    point = sum(scores) / n
    boot_means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(scores) for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    alpha = 1.0 - ci
    lo_idx = int(n_bootstrap * alpha / 2)
    hi_idx = int(n_bootstrap * (1 - alpha / 2))
    return point, boot_means[lo_idx], boot_means[min(hi_idx, len(boot_means) - 1)]


def bootstrap_ci_for_accuracy(
    y_true: List[Any],
    y_pred: List[Any],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """从标签列表计算准确率的Bootstrap CI"""
    scores = [1.0 if t == p else 0.0 for t, p in zip(y_true, y_pred)]
    return bootstrap_ci(scores, n_bootstrap=n_bootstrap, ci=ci, seed=seed)


# ========================================================
# 3. Cohen's Kappa
# ========================================================

def cohens_kappa(y_true: List[Any], y_pred: List[Any]) -> Dict:
    """
    Cohen's Kappa一致性系数（多分类）。

    Returns:
        {'kappa': float, 'interpretation': str}
    """
    n = len(y_true)
    if n == 0:
        return {'kappa': 0.0, 'interpretation': "空列表"}

    classes = sorted(set(y_true) | set(y_pred))
    k = len(classes)
    cls_idx = {c: i for i, c in enumerate(classes)}

    # 构建混淆矩阵
    cm = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        i = cls_idx.get(t, 0)
        j = cls_idx.get(p, 0)
        cm[i][j] += 1

    po = sum(cm[i][i] for i in range(k)) / n  # 观测一致率

    row_sums = [sum(cm[i]) for i in range(k)]
    col_sums = [sum(cm[i][j] for i in range(k)) for j in range(k)]
    pe = sum(row_sums[i] * col_sums[i] for i in range(k)) / (n * n)  # 期望一致率

    if pe >= 1.0:
        kappa = 1.0
    else:
        kappa = (po - pe) / (1.0 - pe)

    kappa = round(kappa, 4)

    if kappa < 0:
        interp = "几乎没有一致（< 0）"
    elif kappa < 0.2:
        interp = "轻微一致（0.01–0.20）"
    elif kappa < 0.4:
        interp = "较差一致（0.21–0.40）"
    elif kappa < 0.6:
        interp = "中等一致（0.41–0.60）"
    elif kappa < 0.8:
        interp = "良好一致（0.61–0.80）"
    else:
        interp = "几乎完全一致（0.81–1.00）"

    return {'kappa': kappa, 'interpretation': interp}


# ========================================================
# 4. Macro / Weighted F1, Precision, Recall
# ========================================================

def macro_f1_precision_recall(
    y_true: List[Any],
    y_pred: List[Any],
    labels: Optional[List[Any]] = None,
) -> Dict:
    """
    计算宏平均、加权平均和每类 F1/Precision/Recall。

    Returns:
        {
            'macro': {'f1': ..., 'precision': ..., 'recall': ...},
            'weighted': {'f1': ..., 'precision': ..., 'recall': ...},
            'per_class': {label: {'f1': ..., 'precision': ..., 'recall': ..., 'support': ...}},
        }
    """
    if labels is None:
        labels = sorted(set(y_true))

    n = len(y_true)
    per_class = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = tp + fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[str(label)] = {
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1': round(f1, 4),
            'support': support,
        }

    active = [v for v in per_class.values() if v['support'] > 0]
    if not active:
        return {'macro': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'weighted': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'per_class': per_class}

    # Macro（等权平均）
    macro = {
        'precision': round(sum(v['precision'] for v in active) / len(active), 4),
        'recall': round(sum(v['recall'] for v in active) / len(active), 4),
        'f1': round(sum(v['f1'] for v in active) / len(active), 4),
    }

    # Weighted（按支持度加权）
    total_support = sum(v['support'] for v in active)
    weighted = {
        'precision': round(sum(v['precision'] * v['support'] for v in active) / total_support, 4),
        'recall': round(sum(v['recall'] * v['support'] for v in active) / total_support, 4),
        'f1': round(sum(v['f1'] * v['support'] for v in active) / total_support, 4),
    }

    return {'macro': macro, 'weighted': weighted, 'per_class': per_class}


# ========================================================
# 5. Matthews Correlation Coefficient (MCC)
# ========================================================

def matthews_corrcoef_multiclass(y_true: List[Any], y_pred: List[Any]) -> float:
    """
    多分类MCC（Matthews相关系数）。
    范围 [-1, 1]，对类别不均衡场景比准确率更稳健。
    """
    classes = sorted(set(y_true) | set(y_pred))
    k = len(classes)
    if k < 2:
        return 1.0

    cls_idx = {c: i for i, c in enumerate(classes)}
    n = len(y_true)

    cm = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        cm[cls_idx.get(t, 0)][cls_idx.get(p, 0)] += 1

    # 按公式计算 MCC for multi-class
    t_k = [sum(cm[i]) for i in range(k)]   # 每类真实数量
    p_k = [sum(cm[i][j] for i in range(k)) for j in range(k)]  # 每类预测数量
    c_k = [cm[i][i] for i in range(k)]     # 正确预测数量

    c = sum(c_k)
    s = n

    s2 = s * s
    sum_tk_sq = sum(t * t for t in t_k)
    sum_pk_sq = sum(p * p for p in p_k)
    sum_tk_pk = sum(t * p for t, p in zip(t_k, p_k))

    numerator = c * s - sum_tk_pk
    denominator = math.sqrt((s2 - sum_tk_sq) * (s2 - sum_pk_sq))

    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


# ========================================================
# 6. 配对Z检验（基于准确率和样本量）
# ========================================================

def paired_t_test_with_ci(
    acc_a: float,
    acc_b: float,
    n_samples: int,
    alpha: float = 0.05,
) -> Dict:
    """
    配对Z检验（适用于已知准确率、样本量的场景）。
    """
    diff = acc_a - acc_b
    se = math.sqrt((acc_a * (1 - acc_a) + acc_b * (1 - acc_b)) / n_samples)
    if se == 0:
        return {'z': 0.0, 'p_value': 1.0, 'is_significant': False,
                'diff': diff, 'ci_95': (diff, diff),
                'interpretation': "标准误为零"}

    z = diff / se
    p_value = 2.0 * (1.0 - _norm_cdf(abs(z)))
    z_crit = 1.96  # 95% CI
    ci = (diff - z_crit * se, diff + z_crit * se)
    is_sig = p_value < alpha

    return {
        'diff_pp': round(diff * 100, 2),
        'z': round(z, 4),
        'p_value': round(p_value, 4),
        'is_significant': is_sig,
        'ci_95': (round(ci[0] * 100, 2), round(ci[1] * 100, 2)),
        'interpretation': (
            f"Δ={diff*100:+.1f}pp, Z={z:.3f}, p={p_value:.4f}, "
            f"95%CI=[{ci[0]*100:+.1f}%, {ci[1]*100:+.1f}%] "
            f"({'显著' if is_sig else '不显著'})"
        )
    }


# ========================================================
# 7. 综合统计报告生成
# ========================================================

def generate_statistical_report(
    y_true: List[Any],
    model_results: Dict[str, List[Any]],   # {'模型A': [pred,...], '模型B': [pred,...]}
    labels: Optional[List[Any]] = None,
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
) -> str:
    """
    生成 Markdown 格式统计报告，可直接用于论文补充材料。

    Args:
        y_true: 真实标签
        model_results: 模型名 -> 预测列表
        labels: 标签集合（None则自动推断）
        alpha: 显著性水平
        n_bootstrap: Bootstrap迭代次数
    """
    if labels is None:
        labels = sorted(set(y_true))

    n = len(y_true)
    lines = [
        "# 统计显著性检验报告",
        "",
        f"样本量：N = {n}，显著性水平：α = {alpha}，Bootstrap迭代：{n_bootstrap}次",
        "",
    ]

    # === 各模型基础指标 ===
    lines += ["## 1. 各模型基础指标", "", "| 模型 | 准确率 | 95% CI | Macro-F1 | κ | MCC |", "|------|--------|--------|----------|---|-----|"]
    for name, preds in model_results.items():
        acc, ci_lo, ci_hi = bootstrap_ci_for_accuracy(y_true, preds, n_bootstrap=n_bootstrap)
        f1 = macro_f1_precision_recall(y_true, preds, labels)['macro']['f1']
        kappa = cohens_kappa(y_true, preds)['kappa']
        mcc = matthews_corrcoef_multiclass(y_true, preds)
        lines.append(
            f"| {name} | {acc*100:.1f}% | [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%] "
            f"| {f1:.3f} | {kappa:.3f} | {mcc:.3f} |"
        )
    lines.append("")

    # === McNemar配对检验矩阵 ===
    model_names = list(model_results.keys())
    if len(model_names) >= 2:
        lines += ["## 2. McNemar 配对检验矩阵", ""]
        lines.append("| 对比 (A vs B) | b | c | χ² | p值 | 显著性 |")
        lines.append("|---|---|---|---|---|---|")
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                na, nb = model_names[i], model_names[j]
                result = mcnemar_test(y_true, model_results[na], model_results[nb], alpha)
                sig = "✅" if result['is_significant'] else "❌"
                lines.append(
                    f"| {na} vs {nb} | {result['b']} | {result['c']} "
                    f"| {result['chi2']:.3f} | {result['p_value']:.4f} | {sig} |"
                )
        lines.append("")

    # === 每类F1 ===
    lines += ["## 3. 每类 F1（详细）", ""]
    lines.append("| 类别 | 支持度 | " + " | ".join(model_results.keys()) + " |")
    lines.append("|------|--------|" + "--|" * len(model_results))
    all_f1 = {name: macro_f1_precision_recall(y_true, preds, labels) for name, preds in model_results.items()}
    for label in labels:
        row = [f"| {label} | {all_f1[model_names[0]]['per_class'].get(str(label), {}).get('support', 0)} |"]
        for name in model_names:
            f1 = all_f1[name]['per_class'].get(str(label), {}).get('f1', 0.0)
            row.append(f" {f1:.3f} |")
        lines.append("".join(row))
    lines.append("")

    return "\n".join(lines)


# ========================================================
# 8. 从 metrics/samples 文件计算统计指标
# ========================================================

def stats_from_samples_file(
    samples_file: str,
    exp_name: str = "exp_a",
    n_bootstrap: int = 5000,
) -> Dict:
    """
    从 *__samples.jsonl 文件读取预测结果，计算完整统计指标。
    """
    results = []
    with open(samples_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    if exp_name == "exp_a":
        y_true_type = [r.get("correct_defect_type", "") for r in results]
        y_pred_type = [r.get("parsed", {}).get("defect_type", r.get("model_answer", "")[:20]) for r in results]
        y_scores = [r.get("overall_score", 0.0) for r in results]

        acc, ci_lo, ci_hi = bootstrap_ci(y_scores, n_bootstrap=n_bootstrap)
        f1_metrics = macro_f1_precision_recall(y_true_type, y_pred_type)
        kappa = cohens_kappa(y_true_type, y_pred_type)
        mcc = matthews_corrcoef_multiclass(y_true_type, y_pred_type)

        return {
            'accuracy': round(acc, 4),
            'ci_95': (round(ci_lo, 4), round(ci_hi, 4)),
            'macro_f1': f1_metrics['macro']['f1'],
            'macro_precision': f1_metrics['macro']['precision'],
            'macro_recall': f1_metrics['macro']['recall'],
            'kappa': kappa['kappa'],
            'kappa_interp': kappa['interpretation'],
            'mcc': mcc,
            'n_samples': len(results),
        }

    # ExpB / ExpC：仅用overall_score做bootstrap
    y_scores = [r.get("overall_score", 0.0) for r in results]
    acc, ci_lo, ci_hi = bootstrap_ci(y_scores, n_bootstrap=n_bootstrap)
    return {
        'accuracy': round(acc, 4),
        'ci_95': (round(ci_lo, 4), round(ci_hi, 4)),
        'n_samples': len(results),
    }


# ========================================================
# 9. 配对 t 检验（样本级分数）
# ========================================================

def paired_t_test_scores(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05,
) -> Dict:
    """配对 t 检验（样本级）：d_i = score_a_i - score_b_i，大样本(n>30)用正态近似 p 值"""
    n = min(len(scores_a), len(scores_b))
    diffs = [scores_a[i] - scores_b[i] for i in range(n)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var_d / n) if var_d > 0 else 0.0
    if se == 0:
        return {
            't': 0.0, 'df': n - 1, 'p_value': 1.0,
            'mean_diff': 0.0, 'mean_diff_pp': 0.0,
            'ci_95': (0.0, 0.0), 'is_significant': False,
            'n': n, 'cohens_d': 0.0,
            'interpretation': "两组分数完全相同，无法区分",
        }
    t = mean_d / se
    p_value = 2.0 * (1.0 - _norm_cdf(abs(t)))
    is_sig = p_value < alpha
    ci_half = 1.96 * se
    std_d = math.sqrt(var_d)
    cohens_d = mean_d / std_d if std_d > 0 else 0.0
    sig_stars = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < alpha else "ns"))
    return {
        't': round(t, 4),
        'df': n - 1,
        'p_value': round(p_value, 6),
        'mean_diff': round(mean_d, 4),
        'mean_diff_pp': round(mean_d * 100, 2),
        'ci_95': (round(mean_d - ci_half, 4), round(mean_d + ci_half, 4)),
        'is_significant': is_sig,
        'cohens_d': round(cohens_d, 4),
        'n': n,
        'sig_stars': sig_stars,
        'interpretation': (
            f"t={t:.4f}, df={n-1}, p={p_value:.4f}, "
            f"mean_diff={mean_d*100:.2f}pp ({sig_stars}, α={alpha})"
        ),
    }


if __name__ == "__main__":
    # 快速自测
    y_true = ["过热", "绝缘", "机械", "油务", "过热", "绝缘", "机械", "油务"] * 5
    y_pred_a = ["过热", "绝缘", "机械", "油务"] * 10
    y_pred_b = ["过热", "绝缘", "过热", "绝缘"] * 10

    print("=== McNemar Test ===")
    print(mcnemar_test(y_true, y_pred_a, y_pred_b))

    print("\n=== Bootstrap CI ===")
    scores = [1.0 if t == p else 0.0 for t, p in zip(y_true, y_pred_a)]
    print(bootstrap_ci(scores))

    print("\n=== Cohen's Kappa ===")
    print(cohens_kappa(y_true, y_pred_a))

    print("\n=== Macro F1 ===")
    print(macro_f1_precision_recall(y_true, y_pred_a, labels=["过热", "绝缘", "机械", "油务"]))

    print("\n=== MCC ===")
    print(matthews_corrcoef_multiclass(y_true, y_pred_a))

    print("\n=== Report ===")
    report = generate_statistical_report(
        y_true,
        {"Model-A": y_pred_a, "Model-B": y_pred_b},
        labels=["过热", "绝缘", "机械", "油务"],
    )
    print(report[:500])
