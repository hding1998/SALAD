#!/usr/bin/env python3
import json, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

for model_prefix in ["deepseek", "kimi"]:
    for f in sorted((ROOT / "output/full_experiment/exp6_cloud_gap").glob(f"{model_prefix}*__samples.jsonl")):
        print(f"\n=== {f.name} ===")
        count = 0
        for line in open(f, encoding="utf-8"):
            s = json.loads(line.strip())
            ans = s.get("model_answer", "")[:120].replace("\n", " ")
            score = s.get("overall_score", "?")
            sid = s.get("sample_id", "?")
            print(f"  [{sid}] score={score} ans={repr(ans)}")
            count += 1
            if count >= 3:
                break
