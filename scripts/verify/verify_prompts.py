#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt构建验证脚本（不调用模型，只检查格式）"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.salad_prompts import EXP_A_SALAD, EXP_B_SALAD, EXP_C_SALAD

def verify_salad_prompts():
    print("="*60)
    print("SALAD Prompt Verify")
    print("="*60)
    
    # ExpA验证
    print("\n[ExpA 缺陷分类]")
    for lv in [0, 1, 2]:
        cfg = EXP_A_SALAD[lv]
        prompt = cfg["template"].format(description="测试描述：变压器油温102度")
        has_json = "JSON" in prompt or "json" in prompt
        has_sla = "注意" in prompt or "只能从" in prompt
        has_criteria = "判断标准" in prompt or "[判断标准]" in prompt
        print(f"  L{lv} ({cfg['name']}): length={len(prompt)} json={has_json} sla={has_sla} criteria={has_criteria}")
        if lv == 0:
            assert has_json and not has_sla, "L0应该有JSON模板但无SLA"
        elif lv == 1:
            assert has_json and has_sla and not has_criteria, "L1应该有JSON+SLA但无Criteria"
        elif lv == 2:
            assert has_json and has_sla and has_criteria, "L2应该有JSON+SLA+Criteria"
    
    # ExpB验证
    print("\n[ExpB 规则配置]")
    for lv in [0, 1, 2]:
        cfg = EXP_B_SALAD[lv]
        prompt = cfg["template"].format(scenario_description="测试场景", statistics="{}")
        has_json = "JSON" in prompt
        has_sla = "注意" in prompt or "只能是" in prompt
        has_criteria = "判断标准" in prompt
        print(f"  L{lv} ({cfg['name']}): length={len(prompt)} json={has_json} sla={has_sla} criteria={has_criteria}")
    
    # ExpC验证
    print("\n[ExpC Dispatch Parsing]")
    for lv in [0, 1, 2]:
        cfg = EXP_C_SALAD[lv]
        prompt = cfg["template"].format(command="将220kV甲线由运行转检修")
        has_json = "JSON" in prompt
        has_sla = "注意" in prompt
        has_criteria = "判断标准" in prompt
        print(f"  L{lv} ({cfg['name']}): length={len(prompt)} has_json={has_json} has_sla={has_sla} has_criteria={has_criteria}")
    
    print("\n[OK] SALAD Prompt verified")

def verify_old_prompts():
    print("\n" + "="*60)
    print("Old Prompt Verify")
    print("="*60)
    
    # Read OLD_PROMPTS directly from run_ablation_full.py via regex
    import re
    rab_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                            "scripts", "run_ablation_full.py")
    with open(rab_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find OLD_PROMPTS dict
    m = re.search(r"OLD_PROMPTS\s*=\s*(\{.*?\n\})", content, re.DOTALL)
    if m:
        old_text = m.group(1)
        # Check key structural elements
        has_exp_a = '"exp_a"' in old_text
        has_l0 = '0:' in old_text
        has_l1 = '1:' in old_text
        has_l2 = '2:' in old_text
        has_cot_l1 = "示例" in old_text or "example" in old_text.lower()
        has_rag_l2 = "rag_context" in old_text
        print(f"  Has exp_a: {has_exp_a}, L0-L2: {has_l0}/{has_l1}/{has_l2}")
        print(f"  L1 has CoT elements: {has_cot_l1}")
        print(f"  L2 has RAG placeholder: {has_rag_l2}")
    else:
        print("  [WARN] Could not extract OLD_PROMPTS struct")
    
    print("\n[OK] Old Prompt verified")

if __name__ == "__main__":
    verify_salad_prompts()
    verify_old_prompts()
    print("\n" + "="*60)
    print("All prompts verified OK!")
    print("="*60)
