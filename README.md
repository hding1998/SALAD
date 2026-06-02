# SALAD: Tail-Anchored Prompt Architecture for Trustworthy Edge NLP in Power System CPS

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Core claim**: For resource-constrained edge environments, prompt architecture—not parameter count—is the decisive reliability lever. Traditional prompt engineering (CoT / RAG stacking) is actively harmful for ≤2B models (−24.9 pp), while SALAD's tail semantic anchoring achieves +51.8 pp improvement (Cohen's *d* = −1.510) with deterministic, variance-free inference.

---

## Overview

This repository contains the complete implementation, datasets, and experimental pipelines for the paper:

**"SALAD: Tail-Anchored Prompt Architecture for Trustworthy Edge NLP in Power System Cyber-Physical Systems"**  
*Submitted to Energy and AI*

SALAD (Structured Anchor Lite Augmentation Design) is a four-level prompt framework specifically architected for ≤2B parameter language models deployed in air-gapped substation environments. It enables deterministic structured-output NLP for three critical power-system tasks—equipment defect classification, data-quality rule configuration, and dispatch instruction parsing—without GPUs, network connectivity, or per-query cloud costs.

---

## Repository Structure

```
.
├── data/                          # Datasets and knowledge bases
│   ├── knowledge_base/            # Structured domain knowledge for RAG
│   │   ├── defect_standard.json   # 15 defect classification rules (Q/GDW 1906-2013)
│   │   ├── dq_rules.json          # 10 data-quality rule templates
│   │   └── dispatch_regulations.json  # 12 dispatch regulation snippets
│   ├── power_qa/                  # ExpA: Defect classification
│   │   └── defect_classification_200.jsonl   # N = 197 samples
│   ├── power_ts/                  # ExpB: Data quality rules
│   │   └── dq_scenarios_120.jsonl            # N = 120 samples
│   ├── power_dispatch/            # ExpC: Dispatch parsing
│   │   └── dispatch_cmds_200.jsonl           # N = 192 samples
│   └── sources/                   # Raw open-source data for reproducibility
│       ├── uci/
│       │   └── household_power_consumption.txt   # 2,075,259 rows
│       └── ETT/
│           ├── ETTh1.csv          # Transformer temperature (hourly)
│           ├── ETTh2.csv
│           ├── ETTm1.csv          # Transformer temperature (15-min)
│           └── ETTm2.csv
├── src/                           # Core source code
│   ├── salad_prompts.py           # SALAD L0–L3 prompt templates
│   ├── model_caller.py            # Unified local/cloud model caller (Ollama + APIs)
│   ├── rag_engine.py              # FSSR (Filtered Single-Shot RAG) engine
│   ├── eval_utils.py              # Scoring utilities (JSON / field / keyword)
│   ├── statistical_tests.py       # Paired t-tests, Bootstrap CIs, Cohen's d
│   ├── data_loader.py             # Dataset loaders
│   ├── analyze_results.py         # Result analysis and aggregation
│   ├── paper_viz.py               # Publication-quality figure generation
│   ├── run_exp_a.py               # ExpA runner (traditional + SALAD)
│   ├── run_exp_b.py               # ExpB runner
│   └── run_exp_c.py               # ExpC runner
├── scripts/                       # Experiment execution scripts
│   ├── run_ablation_full.py       # Full 12-group ablation pipeline
│   ├── run_full_salad_experiment.py   # Main cross-model / cross-task experiment
│   ├── run_exp17_significance.py  # Statistical significance matrix
│   ├── analyze_paper_results.py   # Generate all paper figures
│   └── verify/
│       └── verify_prompts.py      # Validate prompt construction (no model call)
├── configs/
│   └── exp_a_fast.yaml            # Fast-test configuration
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
└── CITATION.cff                   # Citation metadata
```

---

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/STATEGRID-SZ/SALAD-EdgeNLP.git
cd SALAD-EdgeNLP

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama (for local model inference)
# https://ollama.com/download
ollama pull qwen2.5:1.5b
```

### 2. Verify Prompt Construction (No API Calls)

```bash
python scripts/verify/verify_prompts.py
```

### 3. Run a Single Experiment

```bash
# ExpA: Defect classification with SALAD-L2 on qwen2.5:1.5b
python src/run_exp_a.py --model qwen2.5:1.5b --level L2 --samples 50

# ExpB: Data quality rules
python src/run_exp_b.py --model qwen2.5:1.5b --level L2 --samples 50

# ExpC: Dispatch parsing
python src/run_exp_c.py --model qwen2.5:1.5b --level L2 --samples 50
```

### 4. Full Ablation Pipeline

```bash
# Run all 12 ablation groups
python scripts/run_ablation_full.py --groups all --max-samples -1

# Or run selectively
python scripts/run_ablation_full.py --groups exp1,exp2    # Core comparisons
python scripts/run_ablation_full.py --groups exp3          # Component ablation
```

### 5. Generate Paper Figures

```bash
python scripts/analyze_paper_results.py --output output/reports
```

---

## Datasets

All datasets are synthetically constructed from publicly available industry standards and open-source sensor data. Each JSONL record includes provenance metadata (`source`, `verified`, `source_url` where applicable).

| Task | File | N | Source |
|------|------|---|--------|
| ExpA Defect Classification | `data/power_qa/defect_classification_200.jsonl` | 197 | Q/GDW 1904.1-2013, Q/GDW 1906-2013, DL/T 722, IEC 60599 |
| ExpB Data Quality Rules | `data/power_ts/dq_scenarios_120.jsonl` | 120 | UCI Household Power Consumption, ETT dataset |
| ExpC Dispatch Parsing | `data/power_dispatch/dispatch_cmds_200.jsonl` | 192 | DL/T 961-2020 |

### Original Open Data

- **UCI Household Power Consumption** (Hebrail & Berard, 2012): `data/sources/uci/household_power_consumption.txt` — 2,075,259 rows, 9 fields, CC BY 4.0.
- **ETT (Electricity Transformer Temperature)** (Zhou et al., 2021): `data/sources/ETT/ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv` — 17,420–69,680 rows.

---

## Key Results (Reproducible)

| Metric | Value | Command |
|--------|-------|---------|
| Traditional PE Poison (ExpA L0→L1) | **−24.9 pp** | `scripts/run_ablation_full.py --groups exp1` |
| SALAD Ladder (ExpA L0→L1) | **+51.8 pp** | `scripts/run_ablation_full.py --groups exp2` |
| Effect size (Cohen's *d*) | **−1.510** | `scripts/run_exp17_significance.py` |
| 1.5B edge vs. 7B competitor | **50.3% vs. 25.4%** | `scripts/run_full_salad_experiment.py` |
| Inference latency (CPU) | **78 ms/sample** | `src/model_caller.py` |
| Deployment footprint | **~1 GB** (Q4_K_M quantized) | Ollama runtime |

---

## Decoding Protocol

All local experiments use **greedy decoding** (`temperature = 0.0`, `top_p = 1.0`) via Ollama v0.4.x with Q4_K_M quantization. This guarantees:
- **Zero inter-query variance** (identical input → identical output)
- **Deterministic structured outputs** for safety-critical grid automation
- **CPU-only execution** (no GPU required)

---

## Requirements

- Python ≥ 3.10
- Ollama ≥ 0.4.x
- 16 GB RAM (for 1.5B–3B models)
- No GPU required

See `requirements.txt` for Python package dependencies.

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{ding2026salad,
  title={SALAD: Tail-Anchored Prompt Architecture for Trustworthy Edge NLP in Power System Cyber-Physical Systems},
  author={Zhuang, Zheyin and Xue, Lujun and Wang, Zhenyu and Ding, Han and Jiang, Tong and Bai, Rui and Feng, Renjun},
  journal={Energy and AI},
  year={2026},
  publisher={Elsevier}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The datasets derived from open sources (UCI, ETT) retain their original licenses. Industry standards are cited for academic reproducibility; full standard texts must be obtained from the respective publishers.

---

## Contact

For questions regarding the code or data, please open an issue or contact the corresponding author: **Han Ding** (<hding1998@foxmail.com>).

---

*This work was conducted at State Grid Suzhou Power Supply Company, Suzhou, China.*
