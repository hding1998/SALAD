#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
EXP = ROOT / "output/full_experiment"

for fname in sorted(EXP.rglob("qwen3_6_35b*__samples.jsonl")):
    print(f"\n=== {fname.parent.name}/{fname.name} ===")
    count = 0
    for line in open(fname, encoding="utf-8"):
        s = json.loads(line.strip())
        ans = s.get("model_answer", "")[:200].replace("\n", " ")
        score = s.get("overall_score", "?")
        correct_type = s.get("correct_defect_type") or s.get("correct_rule_type") or s.get("correct_obj", "")
        correct_sev = s.get("correct_severity") or s.get("correct_type", "")
        print(f"  score={score} | correct: {correct_type}/{correct_sev}")
        print(f"  ans: {repr(ans)}")
        count += 1
        if count >= 3:
            break
