#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SALAD Framework Prompts (v4 - Full Stack)
S - Structured  A - Anchor  L - Lite  A - Augmentation  D - Design

包含：
  1. SALAD 三级递增 (L0/L1/L2) + L3 完整堆叠 (L2 + FSSR RAG)
  2. 传统 PE 基线 (L0/L1/L2)，用于"毒药效应"对比
  3. SALAD-MARKER 变体 (Section Markers：[ROLE][TASK][INPUT][RULE][FORMAT])
  4. SALAD-SEQ 变体 (Sequential Decompose：两步推理)
"""

# ===========================================================
# ExpA：缺陷分类
# ===========================================================

EXP_A_SALAD = {
    0: {
        "name": "SALAD-L0-Base",
        "system": "你是电力设备运维专家。",
        "template": """请分析以下设备状态描述，判断缺陷类型和严重程度。

设备状态描述：
{description}

以JSON格式输出：{{"defect_type": "类型", "severity": "等级", "reason": "简要判断理由"}}

直接输出JSON，不要分析过程。""",
    },
    1: {
        "name": "SALAD-L1-Anchor",
        "system": "你是电力设备运维专家。",
        "template": """请分析以下设备状态描述，判断缺陷类型和严重程度。

设备状态描述：
{description}

以JSON格式输出：{{"defect_type": "类型", "severity": "等级", "reason": "简要判断理由"}}

注意：缺陷类型只能从 过热/绝缘/机械/油务 中选择
严重程度只能从 一般/严重/危急 中选择

直接输出JSON，不要分析过程。""",
    },
    2: {
        "name": "SALAD-L2-Criteria",
        "system": "你是电力设备运维专家。",
        "template": """请分析以下设备状态描述，判断缺陷类型和严重程度。

设备状态描述：
{description}

以JSON格式输出：{{"defect_type": "类型", "severity": "等级", "reason": "简要判断理由"}}

注意：缺陷类型只能从 过热/绝缘/机械/油务 中选择
严重程度只能从 一般/严重/危急 中选择

[判断标准]
- 过热：温度异常、温升超标、发热、烧损
- 绝缘：放电、击穿、介损、绝缘电阻异常、破损开裂（绝缘部件）
- 机械：变形、破损、卡涩、松动、不到位、倾斜、位移
- 油务：油位异常、油色变化、漏油、渗油、油色谱异常、储油柜问题

严重程度：
- 危急：设备不能继续运行，需立即停电处理
- 严重：设备可短时运行，需尽快安排处理
- 一般：设备可继续运行，按计划检修处理

直接输出JSON，不要分析过程。""",
    },
    3: {
        "name": "SALAD-L3-RAG",
        "system": "你是电力设备运维专家。",
        "template": """请分析以下设备状态描述，判断缺陷类型和严重程度。

设备状态描述：
{description}

以JSON格式输出：{{"defect_type": "类型", "severity": "等级", "reason": "简要判断理由"}}

注意：缺陷类型只能从 过热/绝缘/机械/油务 中选择
严重程度只能从 一般/严重/危急 中选择

[判断标准]
- 过热：温度异常、温升超标、发热、烧损
- 绝缘：放电、击穿、介损、绝缘电阻异常、破损开裂（绝缘部件）
- 机械：变形、破损、卡涩、松动、不到位、倾斜、位移
- 油务：油位异常、油色变化、漏油、渗油、油色谱异常、储油柜问题

严重程度：
- 危急：设备不能继续运行，需立即停电处理
- 严重：设备可短时运行，需尽快安排处理
- 一般：设备可继续运行，按计划检修处理

[参考标准]
{rag_context}

直接输出JSON，不要分析过程。""",
    },
}


# ===========================================================
# ExpB：数据质量规则配置
# ===========================================================

EXP_B_SALAD = {
    0: {
        "name": "SALAD-L0-Base",
        "system": "你是电力数据治理专家。",
        "template": """请为以下数据场景配置数据质量检测规则。

场景描述：{scenario_description}
字段统计特征：{statistics}

输出JSON：{{"rule_type": "规则类型", "field": "目标字段", "threshold": "阈值", "severity": "严重等级"}}

直接输出JSON，不要分析过程。""",
    },
    1: {
        "name": "SALAD-L1-Anchor",
        "system": "你是电力数据治理专家。",
        "template": """请为以下数据场景配置数据质量检测规则。

场景描述：{scenario_description}
字段统计特征：{statistics}

输出JSON：{{"rule_type": "规则类型", "field": "目标字段", "threshold": "阈值", "severity": "严重等级"}}

注意：规则类型只能是 threshold/rate_of_change/consistency/missing_value/flatline/statistical_outlier/pattern
严重等级只能是 一般/严重/危急

直接输出JSON，不要分析过程。""",
    },
    2: {
        "name": "SALAD-L2-Criteria",
        "system": "你是电力数据治理专家。",
        "template": """请为以下数据场景配置数据质量检测规则。

场景描述：{scenario_description}
字段统计特征：{statistics}

输出JSON：{{"rule_type": "规则类型", "field": "目标字段", "threshold": "阈值", "severity": "严重等级"}}

注意：规则类型只能是 threshold/rate_of_change/consistency/missing_value/flatline/statistical_outlier/pattern
严重等级只能是 一般/严重/危急

[判断标准]
- threshold：数值超出固定阈值范围
- rate_of_change：单位时间变化率超出限制
- consistency：多字段/多系统数据不一致
- missing_value：缺失值比例超出限制
- flatline：数值长时间保持不变（数据冻结）
- statistical_outlier：统计离群点（如超出均值±3σ）
- pattern：时序模式异常（如周期性消失）

严重等级：
- 危急：严重偏离，系统告警，影响生产运行
- 严重：明显偏离，影响数据分析和决策
- 一般：轻微偏离，暂不影响正常业务

直接输出JSON，不要分析过程。""",
    },
    3: {
        "name": "SALAD-L3-RAG",
        "system": "你是电力数据治理专家。",
        "template": """请为以下数据场景配置数据质量检测规则。

场景描述：{scenario_description}
字段统计特征：{statistics}

输出JSON：{{"rule_type": "规则类型", "field": "目标字段", "threshold": "阈值", "severity": "严重等级"}}

注意：规则类型只能是 threshold/rate_of_change/consistency/missing_value/flatline/statistical_outlier/pattern
严重等级只能是 一般/严重/危急

[判断标准]
- threshold：数值超出固定阈值范围
- rate_of_change：单位时间变化率超出限制
- consistency：多字段/多系统数据不一致
- missing_value：缺失值比例超出限制
- flatline：数值长时间保持不变（数据冻结）
- statistical_outlier：统计离群点（如超出均值±3σ）
- pattern：时序模式异常（如周期性消失）

严重等级：
- 危急：严重偏离，系统告警，影响生产运行
- 严重：明显偏离，影响数据分析和决策
- 一般：轻微偏离，暂不影响正常业务

[参考规则库]
{rag_context}

直接输出JSON，不要分析过程。""",
    },
}


# ===========================================================
# ExpC：调度指令解析
# ===========================================================

EXP_C_SALAD = {
    0: {
        "name": "SALAD-L0-Base",
        "system": "你是电网调度运行专家。",
        "template": """请将以下调度指令解析为结构化信息。

调度指令：{command}

输出JSON：{{"operation_object": "操作对象", "operation_type": "操作类型"}}

直接输出JSON，不要分析过程。""",
    },
    1: {
        "name": "SALAD-L1-Anchor",
        "system": "你是电网调度运行专家。",
        "template": """请将以下调度指令解析为结构化信息。

调度指令：{command}

输出JSON：{{"operation_object": "操作对象", "operation_type": "操作类型"}}

注意：操作对象是指令中涉及的设备名称（含电压等级/编号）
操作类型是调度动作，常见有：运行转检修/检修转运行/运行转热备用/运行转冷备用/投运/停运/投入/退出/调整/巡视/检查

直接输出JSON，不要分析过程。""",
    },
    2: {
        "name": "SALAD-L2-Criteria",
        "system": "你是电网调度运行专家。",
        "template": """请将以下调度指令解析为结构化信息。

调度指令：{command}

输出JSON：{{"operation_object": "操作对象", "operation_type": "操作类型"}}

注意：操作对象是指令中涉及的设备名称（含电压等级/编号）
操作类型是调度动作，常见有：运行转检修/检修转运行/运行转热备用/运行转冷备用/投运/停运/投入/退出/调整/巡视/检查

[判断标准]
操作对象：
  - 提取设备全名，保留电压等级（如"220kV"）和编号（如"甲线"、"#1"）
  - 多设备时取主要操作对象
操作类型：
  - 状态转换：从X转Y（运行/热备用/冷备用/检修 互转）
  - 保护操作：投入/退出/启用/停用
  - 调整操作：调节/整定/试验
  - 检查操作：巡视/检查/测量

直接输出JSON，不要分析过程。""",
    },
    3: {
        "name": "SALAD-L3-RAG",
        "system": "你是电网调度运行专家。",
        "template": """请将以下调度指令解析为结构化信息。

调度指令：{command}

输出JSON：{{"operation_object": "操作对象", "operation_type": "操作类型"}}

注意：操作对象是指令中涉及的设备名称（含电压等级/编号）
操作类型是调度动作，常见有：运行转检修/检修转运行/运行转热备用/运行转冷备用/投运/停运/投入/退出/调整/巡视/检查

[判断标准]
操作对象：
  - 提取设备全名，保留电压等级（如"220kV"）和编号（如"甲线"、"#1"）
  - 多设备时取主要操作对象
操作类型：
  - 状态转换：从X转Y（运行/热备用/冷备用/检修 互转）
  - 保护操作：投入/退出/启用/停用
  - 调整操作：调节/整定/试验
  - 检查操作：巡视/检查/测量

[参考规程]
{rag_context}

直接输出JSON，不要分析过程。""",
    },
}


# ===========================================================
# 传统 PE 基线（"毒药"对照组）
# 传统做法：L0基础约束 → L1加CoT+Few-shot → L2加全文RAG
# 对1.5B小模型产生"递减"效应
# ===========================================================

EXP_A_TRADITIONAL = {
    0: {
        "name": "Trad-L0-Base",
        "system": "你是电力设备运维专家，负责缺陷分析与判断。",
        "template": """请分析以下电力设备缺陷描述，判断缺陷类型和严重程度。

设备状态描述：
{description}

请以JSON格式输出结果：
{{"defect_type": "类型", "severity": "等级", "reason": "判断理由"}}

缺陷类型候选：过热、绝缘、机械、油务
严重程度候选：一般、严重、危急""",
    },
    1: {
        "name": "Trad-L1-CoT+Examples",
        "system": "你是资深电力设备运维专家，具有丰富的设备缺陷分析经验，熟悉各类电力设备运行规律。",
        "template": """请分析以下电力设备缺陷描述，给出专业判断。

【参考案例】
案例1：
描述：主变压器绕组测温装置显示温度105°C，超出额定运行温度15°C，冷却系统正常运行但仍持续偏高
分析：绕组温度长期超标说明散热异常或过负荷，属于过热类缺陷。温度超额定值15°C，影响绝缘老化速率，需要尽快查明原因并处理，但设备尚可短时运行。
判断：{{"defect_type": "过热", "severity": "严重", "reason": "绕组温度超额定值15°C，散热异常影响绝缘寿命"}}

案例2：
描述：220kV断路器灭弧室绝缘拉杆表面有放电痕迹，局部放电测量超标，绝缘电阻下降明显
分析：放电痕迹和绝缘电阻下降说明绝缘劣化，属于绝缘类缺陷。局放超标意味着击穿风险增大，设备不能继续安全运行，需立即处理。
判断：{{"defect_type": "绝缘", "severity": "危急", "reason": "绝缘拉杆放电痕迹明显，局放超标，存在击穿风险"}}

【待分析案例】
设备状态描述：
{description}

请按以下步骤分析：
第一步：识别描述中的关键故障特征和异常参数
第二步：对比参考案例，判断最符合的缺陷类别
第三步：评估故障对设备安全运行的影响程度
第四步：给出最终判断，说明判断依据

以JSON格式输出最终结果：
{{"defect_type": "类型", "severity": "等级", "reason": "详细分析依据"}}

缺陷类型只能是：过热、绝缘、机械、油务
严重程度只能是：一般、严重、危急""",
    },
    2: {
        "name": "Trad-L2-CoT+RAG",
        "system": "你是资深电力设备运维专家，具有丰富的设备缺陷分析经验，熟悉国家和行业标准规范。",
        "template": """请结合参考资料分析以下电力设备缺陷描述，给出专业判断。

【参考标准资料】
{rag_context}

【参考案例】
案例1：
描述：主变压器绕组温度105°C，超额定值15°C，冷却系统正常
分析：过热类缺陷，绕组温度超标影响绝缘老化速率，需尽快处理
判断：{{"defect_type": "过热", "severity": "严重", "reason": "温度超标"}}

案例2：
描述：断路器绝缘拉杆放电痕迹，局放超标，绝缘电阻下降
分析：绝缘类缺陷，绝缘劣化有击穿风险，需立即停电处理
判断：{{"defect_type": "绝缘", "severity": "危急", "reason": "绝缘劣化有击穿风险"}}

【待分析案例】
设备状态描述：
{description}

请按以下步骤分析：
第一步：查阅参考标准资料，确定相关技术指标
第二步：识别关键故障特征，对照标准判断缺陷类别
第三步：参考案例经验，评估严重程度
第四步：综合判断，给出结论

以JSON格式输出：
{{"defect_type": "类型", "severity": "等级", "reason": "分析依据"}}

缺陷类型只能是：过热、绝缘、机械、油务
严重程度只能是：一般、严重、危急""",
    },
}

EXP_B_TRADITIONAL = {
    0: {
        "name": "Trad-L0-Base",
        "system": "你是电力数据治理专家，负责数据质量管理工作。",
        "template": """请为以下数据质量场景配置合适的检测规则。

场景描述：{scenario_description}
字段统计特征：{statistics}

请以JSON格式输出配置结果：
{{"rule_type": "规则类型", "field": "目标字段", "threshold": "阈值", "severity": "严重等级"}}

规则类型候选：threshold、rate_of_change、consistency、missing_value、flatline、statistical_outlier、pattern
严重等级候选：一般、严重、危急""",
    },
    1: {
        "name": "Trad-L1-CoT+Examples",
        "system": "你是资深电力数据治理专家，具有丰富的电力监测数据质量管理经验。",
        "template": """请为以下数据质量场景配置最合适的检测规则。

【配置案例】
案例1：
场景：智能电表有功功率读数出现瞬间极端高值（正常均值1.09kW，但偶有读数超过10kW），占比约0.3%
统计特征：mean=1.09, std=1.06, max=19.56, outlier_ratio=0.003
分析：存在极端偏离均值的异常点，这是统计离群点问题，需要设置基于均值±3σ的统计异常规则
配置：{{"rule_type": "statistical_outlier", "field": "Global_active_power", "threshold": "mean+3*std", "severity": "严重"}}

案例2：
场景：变压器油温监测数据出现连续48小时读数完全不变（正常情况下每小时温度有±0.5°C波动）
统计特征：normal_std=0.8, stuck_duration=48h, current_std_window=0.0
分析：传感器读数长时间不变，远小于正常波动范围，属于数据冻结（flatline）问题，可能是传感器故障
配置：{{"rule_type": "flatline", "field": "OT", "threshold": "std<0.01_for_24h", "severity": "危急"}}

案例3：
场景：变电站电压测量数据中发现约5%的记录缺失或为空值
统计特征：missing_ratio=0.05, field=Voltage
分析：5%的缺失率对电力质量评估有影响，需要配置缺失值检测规则
配置：{{"rule_type": "missing_value", "field": "Voltage", "threshold": "missing_ratio>0.03", "severity": "严重"}}

【待配置场景】
场景描述：{scenario_description}
字段统计特征：{statistics}

请按以下步骤分析：
第一步：理解场景描述，确定主要数据质量问题类型
第二步：分析统计特征中的异常指标（如极端值、变化率、缺失率等）
第三步：对比参考案例，选择最合适的规则类型
第四步：根据统计特征设置合理的检测阈值
第五步：评估数据质量问题对业务的影响程度，确定严重等级

以JSON格式输出：{{"rule_type": "规则类型", "field": "目标字段", "threshold": "阈值", "severity": "严重等级"}}
规则类型只能是：threshold/rate_of_change/consistency/missing_value/flatline/statistical_outlier/pattern
严重等级只能是：一般/严重/危急""",
    },
    2: {
        "name": "Trad-L2-CoT+RAG",
        "system": "你是资深电力数据治理专家，熟悉各类数据质量标准和检测规则配置方法。",
        "template": """请结合参考规则库为以下数据质量场景配置最合适的检测规则。

【参考规则库】
{rag_context}

【配置案例】
案例1：极端高值 → {{"rule_type": "statistical_outlier", "threshold": "mean+3*std", "severity": "严重"}}
案例2：数据冻结 → {{"rule_type": "flatline", "threshold": "std<0.01_for_24h", "severity": "危急"}}
案例3：缺失值 → {{"rule_type": "missing_value", "threshold": "missing_ratio>0.03", "severity": "严重"}}

【待配置场景】
场景描述：{scenario_description}
字段统计特征：{statistics}

请参考规则库和上述案例，分析场景特征，选择最合适的规则类型，设置合理阈值。

以JSON格式输出：{{"rule_type": "规则类型", "field": "目标字段", "threshold": "阈值", "severity": "严重等级"}}
规则类型只能是：threshold/rate_of_change/consistency/missing_value/flatline/statistical_outlier/pattern
严重等级只能是：一般/严重/危急""",
    },
}

EXP_C_TRADITIONAL = {
    0: {
        "name": "Trad-L0-Base",
        "system": "你是电网调度运行专家，熟悉调度指令规范用语。",
        "template": """请将以下调度指令解析为标准结构化信息。

调度指令：{command}

请以JSON格式提取关键信息：
{{"operation_object": "操作对象（设备名称，含电压等级）", "operation_type": "操作类型（执行的操作动作）"}}

常见操作类型：运行转检修、检修转运行、运行转热备用、运行转冷备用、投运、停运、投入、退出、调整""",
    },
    1: {
        "name": "Trad-L1-CoT+Examples",
        "system": "你是资深电网调度运行专家，熟悉DL/T 961调度规范用语及电力系统操作规程。",
        "template": """请将以下调度指令解析为标准结构化信息。

【解析案例】
案例1：
指令："将220kV甲线由运行转检修，做好安全措施，停止向用户供电"
分析：指令的主体操作对象是"220kV甲线"（线路名称，含电压等级），执行的操作是状态转换"运行转检修"
解析：{{"operation_object": "220kV甲线", "operation_type": "运行转检修"}}

案例2：
指令："#1主变高压侧过流保护装置由退出改为投入运行状态"
分析：操作对象是"#1主变高压侧保护装置"（含变压器编号和保护位置），执行的是"投入"操作
解析：{{"operation_object": "#1主变高压侧保护装置", "operation_type": "投入"}}

案例3：
指令："对110kV乙站进行全面安全巡视检查，重点检查设备外观和接线情况"
分析：操作对象是"110kV乙站"（变电站），执行的是"巡视"操作
解析：{{"operation_object": "110kV乙站", "operation_type": "巡视"}}

案例4：
指令："将35kV丙线电流保护定值调整为10A，时限0.5s"
分析：操作对象是"35kV丙线电流保护"，执行的是定值"调整"操作
解析：{{"operation_object": "35kV丙线电流保护", "operation_type": "调整"}}

【待解析指令】
调度指令：{command}

请按以下步骤分析：
第一步：识别指令中的操作目标设备（线路、变压器、保护装置、母线等）
第二步：提取设备完整名称（含电压等级和编号/名称）
第三步：识别执行的操作动作类型
第四步：规范化表达操作类型

以JSON格式输出：{{"operation_object": "操作对象", "operation_type": "操作类型"}}""",
    },
    2: {
        "name": "Trad-L2-CoT+RAG",
        "system": "你是资深电网调度运行专家，熟悉DL/T 961-2020《电网调度规范用语》及操作规程。",
        "template": """请结合参考规程将以下调度指令解析为标准结构化信息。

【参考规程】
{rag_context}

【解析案例】
案例1：指令"220kV甲线由运行转检修" → {{"operation_object": "220kV甲线", "operation_type": "运行转检修"}}
案例2：指令"#1主变保护由退出改为投入" → {{"operation_object": "#1主变高压侧保护装置", "operation_type": "投入"}}
案例3：指令"110kV乙站全面安全巡视" → {{"operation_object": "110kV乙站", "operation_type": "巡视"}}

【待解析指令】
调度指令：{command}

请参考上述规程和案例，识别操作对象和操作类型，提取结构化信息。

以JSON格式输出：{{"operation_object": "操作对象", "operation_type": "操作类型"}}""",
    },
}


# ===========================================================
# 新创意实验1：Section Markers（SALAD-MARKER）
# 使用 [ROLE][TASK][INPUT][RULE][CRITERIA][FORMAT] 显式区块标记
# 假设：强制的视觉分隔符帮助注意力有限的小模型"定位"关键信息
# ===========================================================

EXP_A_SALAD_MARKER = {
    0: {
        "name": "Marker-L0-Base",
        "system": "你是电力设备运维专家。",
        "template": """[ROLE] 电力设备运维专家
[TASK] 判断设备缺陷类型和严重程度
[INPUT] {description}
[FORMAT] {{"defect_type": "类型", "severity": "等级"}}""",
    },
    1: {
        "name": "Marker-L1-Anchor",
        "system": "你是电力设备运维专家。",
        "template": """[ROLE] 电力设备运维专家
[TASK] 判断设备缺陷类型和严重程度
[INPUT] {description}
[RULE] defect_type ∈ {{过热,绝缘,机械,油务}} | severity ∈ {{一般,严重,危急}}
[FORMAT] {{"defect_type": "类型", "severity": "等级"}}""",
    },
    2: {
        "name": "Marker-L2-Criteria",
        "system": "你是电力设备运维专家。",
        "template": """[ROLE] 电力设备运维专家
[TASK] 判断设备缺陷类型和严重程度
[INPUT] {description}
[RULE] defect_type ∈ {{过热,绝缘,机械,油务}} | severity ∈ {{一般,严重,危急}}
[CRITERIA]
过热=温度异常|温升超标|发热|烧损
绝缘=放电|击穿|介损|绝缘电阻异常
机械=变形|卡涩|松动|破损|不到位
油务=漏油|渗油|油色谱异常|油位异常
危急=立即停电 | 严重=尽快处理 | 一般=按计划
[FORMAT] {{"defect_type": "类型", "severity": "等级"}}""",
    },
}


# ===========================================================
# 新创意实验2：Sequential Decompose（SALAD-SEQ）
# 两步推理：先分类缺陷类型，再根据类型判断严重度
# 假设：拆分双标签任务降低1.5B模型的认知负荷
# 使用方式：先调用 step1 获取 defect_type，再用 step2 获取 severity
# ===========================================================

EXP_A_SALAD_SEQ = {
    "step1": {
        "name": "Seq-Step1-Type",
        "system": "你是电力设备运维专家。",
        "template": """[TASK] 判断设备缺陷类型
[INPUT] {description}
[RULE] 只能选1个: 过热/绝缘/机械/油务
[CRITERIA] 过热=温度|发热 绝缘=放电|介损 机械=变形|卡涩|松动 油务=漏油|油色谱
[FORMAT] {{"defect_type": "类型"}}""",
    },
    "step2": {
        "name": "Seq-Step2-Severity",
        "system": "你是电力设备运维专家。",
        "template": """[TASK] 判断缺陷严重程度
[INPUT] 设备描述：{description}
[CONTEXT] 已识别缺陷类型：{defect_type}
[RULE] 只能选1个: 一般/严重/危急
[CRITERIA] 危急=需立即停电 | 严重=可短时运行需尽快处理 | 一般=可继续运行按计划检修
[FORMAT] {{"severity": "等级"}}""",
    },
}


# ===========================================================
# 新创意实验3：Domain-Vocabulary Anchor（SALAD-DVA）
# 在Anchor层注入领域标准词汇映射表
# 假设：帮助模型将模糊描述与标准术语对齐
# ===========================================================

EXP_A_SALAD_DVA = {
    0: {
        "name": "DVA-L0-Base",
        "system": "你是电力设备运维专家。",
        "template": """请分析以下设备状态描述，判断缺陷类型和严重程度。

设备状态描述：
{description}

以JSON格式输出：{{"defect_type": "类型", "severity": "等级"}}

直接输出JSON，不要分析过程。""",
    },
    1: {
        "name": "DVA-L1-VocabAnchor",
        "system": "你是电力设备运维专家。",
        "template": """请分析以下设备状态描述，判断缺陷类型和严重程度。

设备状态描述：
{description}

[领域词汇对照表]
温度高/温升/发热/烧损 → 过热
放电/击穿/介损/绝缘电阻 → 绝缘
变形/卡涩/松动/破损/倾斜 → 机械
漏油/渗油/油色/油位/油色谱 → 油务

[严重度词汇]
不能运行/立即停电/紧急 → 危急
尽快/短时/需处理 → 严重
正常运行/计划检修/轻微 → 一般

以JSON格式输出：{{"defect_type": "类型", "severity": "等级"}}
缺陷类型只能从 过热/绝缘/机械/油务 中选择
严重程度只能从 一般/严重/危急 中选择

直接输出JSON，不要分析过程。""",
    },
    2: {
        "name": "DVA-L2-VocabAnchor+Criteria",
        "system": "你是电力设备运维专家。",
        "template": """请分析以下设备状态描述，判断缺陷类型和严重程度。

设备状态描述：
{description}

[领域词汇对照表]
温度高/温升/发热/烧损 → 过热
放电/击穿/介损/绝缘电阻 → 绝缘
变形/卡涩/松动/破损/倾斜 → 机械
漏油/渗油/油色/油位/油色谱 → 油务

[判断标准]
- 过热：温度异常、温升超标、发热、烧损
- 绝缘：放电、击穿、介损、绝缘电阻异常、破损开裂（绝缘部件）
- 机械：变形、破损、卡涩、松动、不到位、倾斜、位移
- 油务：油位异常、油色变化、漏油、渗油、油色谱异常、储油柜问题

严重程度：
- 危急：设备不能继续运行，需立即停电处理
- 严重：设备可短时运行，需尽快安排处理
- 一般：设备可继续运行，按计划检修处理

以JSON格式输出：{{"defect_type": "类型", "severity": "等级"}}
缺陷类型只能从 过热/绝缘/机械/油务 中选择
严重程度只能从 一般/严重/危急 中选择

直接输出JSON，不要分析过程。""",
    },
}
