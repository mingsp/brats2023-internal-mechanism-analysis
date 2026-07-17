# 像素转移因果追踪实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 对验证集选定的 `up1→up2` 跳接路径完成必要性、恢复性、区域特异性、剂量关系和无跳接负控制验证。

**Architecture:** 纯函数模块定义持续纠正集合、空间映射、平移破坏和局部恢复；执行脚本加载冻结权重与正式观察器，在测试病例上生成患者级反事实结果；统计脚本执行三种子配对检验和结论门控；绘图脚本只生成一张因果主图。

**Tech Stack:** Python、PyTorch、NumPy、Pandas、SciPy、Matplotlib、PyTest。

---

### Task 1: 锁定配置和协议状态

**Files:**
- Create: `configs/experiments/v6_pixel_transition_causal_tracing.yaml`
- Create: `scripts/lock_v6_causal_protocol.py`
- Test: `tests/unit/test_causal_protocol_gate.py`

- [x] 写入候选特异门控失败测试：要求三个验证集种子均满足方向、置信区间、Holm 和 `d>=0.8`。
- [x] 运行测试，确认因门控函数缺失而失败。
- [x] 实现门控并生成正式 `results/v6_pixel_transition_causal/v6_protocol_lock_v2.json`；该锁在首次干预前仅修正观察器种子身份，并保留被替代锁哈希。
- [x] 验证候选固定为 `up1->up2`，全局双指标门控仍记录为失败，禁止覆盖。

### Task 2: 实现干预纯函数

**Files:**
- Create: `src/pptt/interventions/causal_tracing.py`
- Test: `tests/unit/test_causal_tracing.py`

- [x] 为持续纠正掩码、目标切片选择、最大池化映射、平移破坏、局部恢复、保留率和恢复比例写失败测试。
- [x] 运行测试，确认失败原因是接口不存在。
- [x] 实现最小纯函数；验证形状、剂量范围、空掩码和 `TE<=0`。
- [x] 运行单元测试并确认全部通过。

### Task 3: 实现模型级反事实执行器

**Files:**
- Create: `scripts/run_v6_pixel_transition_causal.py`
- Create: `src/pptt/interventions/causal_runtime.py`
- Test: `tests/integration/test_causal_trace_smoke.py`

- [x] 写一个合成 U-Net 前向测试，验证 `clean→corrupt→restore` 的钩子顺序和钩子清理。
- [x] 增加批量输入变换和模块输入捕获，不修改原 V4 协议。
- [x] 加载冻结模型、真实观察器和正式轨迹；对每个病例按固定规则选择切片。
- [x] 运行干净、破坏、四级目标恢复和匹配区域恢复，原子写入患者结果。
- [x] 对无跳接模型运行同名路径负控制并记录 logit/状态误差。

### Task 4: 运行统计和结论门控

**Files:**
- Create: `scripts/summarize_v6_causal.py`
- Test: `tests/unit/test_causal_conclusion_gate.py`

- [x] 为必要性、恢复性、区域特异性、剂量关系和负控制门控写失败测试。
- [x] 聚合患者级结果，执行 10,000 次配对自助法、Wilcoxon、效应量和 Holm 校正。
- [x] 输出 `causal_patient_statistics.parquet`、`causal_dose_summary.parquet` 和 `causal_conclusion_gate.json`。
- [x] 保留固定路径、剂量和患者规则，并由结论门控统一判定。

### Task 5: 生成因果结果图和审计

**Files:**
- Create: `scripts/make_v6_causal_figure.py`
- Modify: `docs/final_experiment_audit_cn.md`

- [x] 固定同一患者、同一切片展示 clean、corrupt、matched-target restore 和 matched-control restore。
- [x] 绘制三种子剂量曲线以及必要性、恢复性和区域特异性森林图。
- [x] 输出中英文 600 dpi PNG/PDF 和来源 JSON。
- [x] 运行完整测试、语法检查、定义性缺失值检查、图像尺寸检查和钩子残留检查。
