# PPTT 主文图重构与 TransUNet 机制复现 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用三张真实数据主图呈现 PPTT 的方法必要性、过程发现与干预忠实性，并在 TransUNet 上完成验证集发现、测试前锁定和测试集反事实复现。

**Architecture:** 图件层只消费正式结果，不重新计算或筛选实验结论。TransUNet 复现层把四条预声明 decoder 输入路径映射为源节点到接收节点的输出锚定过程候选，在验证集上最多选择一条路径；内容寻址协议锁生成后，测试集运行器只执行锁定的空间损坏、目标恢复、匹配对照和剂量条件。

**Tech Stack:** Python 3.10、PyTorch、NumPy、SciPy、pandas、Matplotlib、PyArrow、PyYAML、pytest。

---

## 文件职责

- `src/pptt/lineage/direct_paths.py`：定义任意有序源节点到接收节点的输出锚定持续纠正/破坏队列。
- `src/pptt/interventions/transunet_paths.py`：声明并核验四条 TransUNet decoder 输入变量，解析嵌套模块路径。
- `src/pptt/statistics/mechanism_replication.py`：计算验证集路径统计、唯一候选选择和测试集结论门控。
- `configs/experiments/v8_transunet_mechanism_replication.yaml`：冻结候选空间、覆盖门槛、空间算子、剂量、匹配和统计规则。
- `scripts/select_v8_transunet_candidate.py`：从三种子验证集轨迹生成患者表、路径统计和唯一候选。
- `scripts/lock_v8_transunet_protocol.py`：在任何测试干预前冻结代码、数据、患者、权重、观察器、候选和规则哈希。
- `scripts/run_v8_transunet_mechanism.py`：按种子执行正式测试集反事实干预并原子保存患者结果和进度。
- `scripts/summarize_v8_transunet_mechanism.py`：生成患者统计、剂量汇总、结论门控和研究者诊断报告。
- `scripts/monitor_v8_progress.py`：只读显示当前阶段、每种子进度、失败和动态 ETA。
- `scripts/server/run_v8_transunet_pipeline.sh`：按依赖顺序运行验证、锁定、三种子并行测试和汇总。
- `src/pptt/visualization/{method_overview,pixel_fate,causal,v7_coverage}.py`：分别实现图 1、图 2、图 3 和 V7 补充覆盖图。
- `scripts/make_{pptt_method_figure,network_pixel_fate_figure,v6_causal_figure,v7_coverage_figure}.py`：从正式资产生成中英文 PNG/PDF/JSON。

### Task 1: 固定非相邻过程候选的数学对象

**Files:**
- Create: `src/pptt/lineage/direct_paths.py`
- Create: `tests/unit/test_direct_path_cohorts.py`

- [x] **Step 1: 写失败测试，覆盖持续纠正、持续破坏、可靠性和源/接收节点合法性**

```python
def test_direct_path_cohorts_are_output_anchored_and_reliable():
    cohorts = output_anchored_direct_path_cohorts(
        states, truth, reliable, final_state,
        source_index=1, receiver_index=4, truth_classes=(1, 2, 3),
    )
    assert cohorts.correction.dtype == np.bool_
    assert cohorts.damage.dtype == np.bool_
    assert not np.any(cohorts.correction & cohorts.damage)
    assert np.all(final_state[cohorts.correction] == truth[cohorts.correction])
    assert np.all(final_state[cohorts.damage] != truth[cohorts.damage])
```

- [x] **Step 2: 确认测试因模块不存在而失败**

Run: `python -m pytest tests/unit/test_direct_path_cohorts.py -q`

Expected: `ModuleNotFoundError: pptt.lineage.direct_paths`

- [x] **Step 3: 实现不可重叠的直接路径队列与类别平衡净恢复率**

```python
@dataclass(frozen=True)
class DirectPathCohorts:
    correction: np.ndarray
    damage: np.ndarray

def output_anchored_direct_path_cohorts(
    states, truth, reliable, final_model_state, *,
    source_index: int, receiver_index: int, truth_classes: Sequence[int],
) -> DirectPathCohorts: ...

def class_balanced_direct_net_recovery(
    cohorts: DirectPathCohorts, truth, *, truth_classes: Sequence[int],
) -> float: ...
```

约束：`0 <= source_index < receiver_index < K`；可靠性要求覆盖 `source_index:` 的全部后缀；纠正要求源节点错误、接收节点起始持续正确且原模型最终正确；破坏采用对称定义。

- [x] **Step 4: 运行单元测试与现有 lineage 测试**

Run: `python -m pytest tests/unit/test_direct_path_cohorts.py tests/unit/test_lineage_cohorts.py tests/unit/test_lineage_depths.py -q`

Expected: all passed.

- [x] **Step 5: 提交**

```bash
git add src/pptt/lineage/direct_paths.py tests/unit/test_direct_path_cohorts.py
git commit -m "feat: define output-anchored direct path cohorts"
```

### Task 2: 固定 TransUNet 的四条可干预 decoder 输入路径

**Files:**
- Create: `src/pptt/interventions/transunet_paths.py`
- Create: `tests/unit/test_transunet_paths.py`
- Modify: `tests/integration/test_activation_restore.py`

- [ ] **Step 1: 写失败测试，检查拓扑顺序、唯一张量输入和非法路径拒绝**

```python
def test_transunet_decoder_paths_match_real_forward_arguments():
    paths = declared_transunet_decoder_paths()
    assert [path.path_id for path in paths] == [
        "bottleneck_to_up1", "skip_down3_to_up1",
        "skip_down2_to_up2", "skip_down1_to_up3",
    ]
    assert [(p.receiver_node, p.argument_index) for p in paths] == [
        ("up1", 0), ("up1", 1), ("up2", 1), ("up3", 1),
    ]
```

- [ ] **Step 2: 运行并确认失败**

Run: `python -m pytest tests/unit/test_transunet_paths.py -q`

Expected: import failure.

- [ ] **Step 3: 实现声明、点路径解析和运行时输入一致性审计**

```python
@dataclass(frozen=True)
class InterventionPath:
    path_id: str
    source_node: str
    receiver_node: str
    source_index: int
    receiver_index: int
    source_module_path: str
    receiver_module_path: str
    argument_index: int
    topology_order: int

def resolve_module(root: nn.Module, dotted_path: str) -> nn.Module: ...
def declared_transunet_decoder_paths() -> tuple[InterventionPath, ...]: ...
def audit_path_activation_identity(adapter, path, image) -> dict[str, object]: ...
```

身份审计要求源模块输出与接收模块对应输入在形状、数据指针之外的逐元素数值上完全一致；四条路径均须通过，不能依据实验结果修改路径表。

- [ ] **Step 4: 验证空间循环平移、完整恢复和钩子清理**

Run: `python -m pytest tests/unit/test_transunet_paths.py tests/integration/test_activation_restore.py tests/integration/test_transunet_trace.py -q`

Expected: all passed.

- [ ] **Step 5: 提交**

```bash
git add src/pptt/interventions/transunet_paths.py tests/unit/test_transunet_paths.py tests/integration/test_activation_restore.py
git commit -m "feat: register auditable TransUNet intervention paths"
```

### Task 3: 实现验证集候选统计与唯一选择器

**Files:**
- Create: `src/pptt/statistics/mechanism_replication.py`
- Create: `tests/unit/test_mechanism_candidate_selection.py`
- Create: `configs/experiments/v8_transunet_mechanism_replication.yaml`
- Modify: `scripts/run_v5_transunet.py`
- Create: `scripts/select_v8_transunet_candidate.py`

- [ ] **Step 1: 写失败测试，锁定“最多一条候选”和无候选停止语义**

```python
def test_selector_uses_cross_seed_minimum_effect_then_topology():
    selected = select_registered_candidate(rows, required_seeds=(42, 123, 3407), ...)
    assert selected["status"] == "REGISTERED_TRANSUNET_CANDIDATE"
    assert selected["candidate"]["path_id"] == "skip_down2_to_up2"

def test_selector_returns_explicit_no_candidate_status():
    selected = select_registered_candidate(nonpassing_rows, ...)
    assert selected == {"status": "NO_REGISTERED_TRANSUNET_CANDIDATE", ...}
```

- [ ] **Step 2: 运行并确认失败**

Run: `python -m pytest tests/unit/test_mechanism_candidate_selection.py -q`

Expected: import failure.

- [ ] **Step 3: 实现患者级净恢复、全局 Holm 校正和确定性选择规则**

```python
def summarize_validation_candidates(patient_rows, *, required_seeds,
                                    bootstrap_iterations, bootstrap_seed): ...

def select_registered_candidate(statistics_rows, coverage_rows, *,
                                required_seeds, minimum_supported_seeds,
                                minimum_patients_per_seed,
                                minimum_target_pixels_per_seed,
                                alpha, minimum_standardized_effect): ...
```

四条路径、三种子的 12 个检验共同执行 Holm 校正。候选要求三种子均值方向为正，至少两个种子同时满足覆盖、`CI_low > 0`、`Holm p < 0.05` 和预注册效应量；排序为最小跨种子效应降序、平均效应降序、拓扑序升序。

- [ ] **Step 4: 写入 V8 配置并让既有 TransUNet 轨迹脚本支持通用 split 配置**

配置固定：验证集 125 人/种子，测试集 250 人/种子，四条路径，目标切片最少 32 个像素、特征位置和对照位置各 8 个，空间平移 `(2, 2)`，剂量 `[0, .25, .5, .75, 1]`，三模型种子和三观察器种子，bootstrap 10000，`alpha=.05`，最低标准化效应 `0.8`，最低支持种子数 `2`。`run_v5_transunet.py` 仅去除 V5 专属错误文案，使同一正式轨迹生成器可由 V8 验证集配置调用。

- [ ] **Step 5: 实现只读取验证轨迹的选择脚本**

输出：`validation_candidate_patient_scores.parquet`、`validation_candidate_statistics.parquet`、`validation_candidate_coverage.parquet`、`candidate_registration.json`。脚本不得读取测试结果目录。

- [ ] **Step 6: 验证测试与 CLI 帮助**

Run: `python -m pytest tests/unit/test_mechanism_candidate_selection.py tests/unit/test_transfer_segments.py -q`

Run: `python scripts/select_v8_transunet_candidate.py --help`

Expected: tests pass; help exits 0.

- [ ] **Step 7: 提交**

```bash
git add src/pptt/statistics/mechanism_replication.py tests/unit/test_mechanism_candidate_selection.py configs/experiments/v8_transunet_mechanism_replication.yaml scripts/run_v5_transunet.py scripts/select_v8_transunet_candidate.py
git commit -m "feat: select one TransUNet mechanism candidate on validation data"
```

### Task 4: 实现内容寻址协议锁

**Files:**
- Create: `scripts/lock_v8_transunet_protocol.py`
- Create: `tests/unit/test_v8_protocol_lock.py`

- [ ] **Step 1: 写失败测试，覆盖首次锁定、幂等重放、差异拒绝和无候选停止**

```python
def test_protocol_lock_is_content_addressed_and_precedes_test_outputs(tmp_path): ...
def test_protocol_lock_rejects_changed_candidate_or_existing_patient_output(tmp_path): ...
def test_no_candidate_writes_terminal_status_without_authorizing_test(tmp_path): ...
```

- [ ] **Step 2: 运行并确认失败**

Run: `python -m pytest tests/unit/test_v8_protocol_lock.py -q`

- [ ] **Step 3: 实现锁文件**

锁文件必须包含：候选与选择统计、配置哈希、候选脚本/路径声明/队列定义/运行器源码哈希、验证资产哈希、三个模型权重哈希、观察器目录清单哈希、验证/测试有序患者 ID 及哈希、结构变量、算子、剂量、匹配、统计门槛和 `LOCKED_BEFORE_FIRST_TEST_INTERVENTION` 状态。若候选状态为 `NO_REGISTERED_TRANSUNET_CANDIDATE`，写终止状态且 `test_intervention_authorized=false`。

- [ ] **Step 4: 运行测试和隐私扫描**

Run: `python -m pytest tests/unit/test_v8_protocol_lock.py -q`

Run: `rg -n "connect\.westc|oyhz|password|SSH_PASSWORD" scripts configs src docs/superpowers/plans/2026-07-20-pptt-figures-transunet-mechanism-replication-plan-cn.md`

Expected: tests pass; privacy scan has no credential matches.

- [ ] **Step 5: 提交**

```bash
git add scripts/lock_v8_transunet_protocol.py tests/unit/test_v8_protocol_lock.py
git commit -m "feat: lock TransUNet mechanism replication before test intervention"
```

### Task 5: 实现 TransUNet 测试集反事实运行器与结论门控

**Files:**
- Create: `src/pptt/statistics/causal_replication.py`
- Create: `scripts/run_v8_transunet_mechanism.py`
- Create: `scripts/summarize_v8_transunet_mechanism.py`
- Create: `tests/unit/test_v8_causal_gate.py`
- Create: `tests/integration/test_v8_transunet_mechanism_smoke.py`

- [ ] **Step 1: 写失败测试，覆盖四条件、剂量、匹配对照和门控**

```python
def test_replication_gate_requires_necessity_restoration_specificity_dose_and_operator(): ...
def test_failed_specificity_cannot_authorize_mechanism_claim(): ...
def test_smoke_restores_only_registered_receiver_argument(): ...
```

- [ ] **Step 2: 运行并确认失败**

Run: `python -m pytest tests/unit/test_v8_causal_gate.py tests/integration/test_v8_transunet_mechanism_smoke.py -q`

- [ ] **Step 3: 实现按种子运行器**

每名测试患者只使用锁定路径：从正式测试轨迹构造直接路径持续纠正集合；选择目标像素最多且满足门槛的固定切片；映射至候选输入张量；运行 `Clean`、`Corrupt`、四个非零目标恢复剂量、等量匹配目标和等量匹配对照；主要结局为原模型最终输出在固定目标集合上的真实类别保持率。患者 JSON 与种子状态 JSON 必须原子写入，`--resume` 只能跳过内容哈希匹配的已完成患者。

- [ ] **Step 4: 实现算子与锚点审计**

逐患者记录：clean 与正式轨迹状态误差、完整恢复最大绝对误差、空间循环平移逐通道多重集合误差、运行前后钩子数、源输出与接收输入一致性。任一硬审计失败使种子状态为 `FAILED_AUDIT`，不得进入正式统计。

- [ ] **Step 5: 实现患者统计、剂量汇总和结论门控**

必要性=`Q_clean-Q_corrupt`；恢复性=`Q_target-Q_corrupt`；特异性=`Q_matched_target-Q_matched_control`。每个端点按种子进行患者配对 bootstrap、Wilcoxon、Cohen's dz，并对 9 个端点检验全局 Holm 校正。至少两个种子通过三端点、剂量单调和算子审计才允许 `INTERVENTIONALLY_FAITHFUL_TRANSUNET_REPLICATION`；否则写明具体失败门。

- [ ] **Step 6: 运行测试与 smoke**

Run: `python -m pytest tests/unit/test_v8_causal_gate.py tests/integration/test_v8_transunet_mechanism_smoke.py tests/integration/test_activation_restore.py -q`

Expected: all passed; smoke outputs are explicitly marked `SMOKE_NOT_FOR_CONCLUSION`.

- [ ] **Step 7: 提交**

```bash
git add src/pptt/statistics/causal_replication.py scripts/run_v8_transunet_mechanism.py scripts/summarize_v8_transunet_mechanism.py tests/unit/test_v8_causal_gate.py tests/integration/test_v8_transunet_mechanism_smoke.py
git commit -m "feat: validate TransUNet process candidates by locked intervention"
```

### Task 6: 重构三张主图和 V7 补充覆盖图

**Files:**
- Modify: `src/pptt/visualization/method_overview.py`
- Modify: `src/pptt/visualization/pixel_fate.py`
- Modify: `src/pptt/visualization/causal.py`
- Create: `src/pptt/visualization/v7_coverage.py`
- Modify: `scripts/make_pptt_method_figure.py`
- Modify: `scripts/make_network_pixel_fate_figure.py`
- Modify: `scripts/make_v6_causal_figure.py`
- Create: `scripts/make_v7_coverage_figure.py`
- Modify: `tests/unit/test_method_overview_visualization.py`
- Modify: `tests/unit/test_causal_visualization.py`
- Modify: `tests/visual/test_figure_contracts.py`

- [ ] **Step 1: 写失败的图件语义契约**

契约检查：图 1 八节点同病例同切片且只含三类输出符号；图 2 精确包含 750 点、`<=2 pp` 灰带、`86/750`、`0.163`、三项效应和 `3.80x`，两模型均为八节点事件图；图 3 固定病例、四条件、三种子剂量与三端点；V7 灰色空值与 `INSUFFICIENT_NETWORK_COVERAGE`；全部中英文 PNG/PDF/JSON、600 dpi 和源文件 SHA256。

- [ ] **Step 2: 运行并确认现有图件不满足契约**

Run: `python -m pytest tests/unit/test_method_overview_visualization.py tests/unit/test_causal_visualization.py tests/visual/test_figure_contracts.py -q`

Expected: new semantic assertions fail.

- [ ] **Step 3: 重绘图 1**

连续构图仅保留：真实 MRI 与冻结 U 形适配示例、八个真实节点状态、同一像素状态轨迹、转移张量/事件类型/持续形成深度三个输出。移除底部折线、柱状、空间热图和任何 CAM/F/G 元素。

- [ ] **Step 4: 重绘图 2**

散点按种子使用不同点形；加入 `<=2 pp` 灰带和固定统计标注；森林图三项共享百分点坐标并标注 `3.80x`；轨迹只叠加青色持续纠正、橙色破坏、紫色错误重编码及白色真值轮廓，不再铺满肿瘤类别色。

- [ ] **Step 5: 重绘图 3 与 V7 补充图**

图 3 使用同一裁剪显示 Clean、Corrupt、Target Restore、Matched Control；恢复新增像素用高饱和青色，持续保留用低饱和青色，丢失用橙色；剂量曲线使用不同点形/线型并直接标注；森林图显示有效患者数。V7 仅保留效应与覆盖两矩阵，灰色为不可评估。

- [ ] **Step 6: 生成中英文正式图并视觉检查**

Run: `python scripts/make_pptt_method_figure.py --workspace-root . --language en --dpi 600`

Run: `python scripts/make_pptt_method_figure.py --workspace-root . --language zh --dpi 600`

Run: `python scripts/make_network_pixel_fate_figure.py --workspace-root . --language en --asset-root "$env:PPTT_ASSET_ROOT/processed_2d" --dpi 600`

Run: `python scripts/make_network_pixel_fate_figure.py --workspace-root . --language zh --asset-root "$env:PPTT_ASSET_ROOT/processed_2d" --dpi 600`

Run: `python scripts/make_v6_causal_figure.py --workspace-root . --language en --dpi 600`

Run: `python scripts/make_v6_causal_figure.py --workspace-root . --language zh --dpi 600`

Run: `python scripts/make_v7_coverage_figure.py --workspace-root . --language en --dpi 600`

Run: `python scripts/make_v7_coverage_figure.py --workspace-root . --language zh --dpi 600`

人工检查：无文字遮挡、同一视野、颜色与图例一致、十秒内分别读出“追踪同像素”“终点近而过程远”“过程候选预测干预”。

- [ ] **Step 7: 运行图件测试并提交**

Run: `python -m pytest tests/unit/test_method_overview_visualization.py tests/unit/test_flagship_visualization.py tests/unit/test_causal_visualization.py tests/visual/test_figure_contracts.py -q`

```bash
git add src/pptt/visualization scripts/make_pptt_method_figure.py scripts/make_network_pixel_fate_figure.py scripts/make_v6_causal_figure.py scripts/make_v7_coverage_figure.py tests/unit/test_method_overview_visualization.py tests/unit/test_causal_visualization.py tests/visual/test_figure_contracts.py
git commit -m "feat: rebuild focused PPTT paper figures from formal data"
```

### Task 7: 增加服务器并行调度和动态 ETA

**Files:**
- Create: `scripts/monitor_v8_progress.py`
- Create: `scripts/server/run_v8_transunet_pipeline.sh`
- Create: `tests/unit/test_v8_progress_monitor.py`

- [ ] **Step 1: 写失败测试，覆盖未开始、运行、失败、无候选和完成状态**

```python
def test_progress_summary_reports_phase_counts_and_dynamic_eta(tmp_path): ...
def test_progress_summary_surfaces_failed_seed_without_restarting(tmp_path): ...
```

- [ ] **Step 2: 实现只读监控器**

输出：当前阶段、候选状态、三个种子各 `completed/250`、总进度、最近患者、失败、GPU/磁盘由外部命令补充的位置、近期吞吐和动态 ETA。`--watch --interval 30` 持续刷新；监控器不得启动、停止或修改实验。

- [ ] **Step 3: 实现服务器流水线**

流水线依次执行：三个验证集轨迹 worker 并行 → 候选统计 → 协议锁 → 若授权则三个测试干预 worker 并行 → 汇总。每个 worker 独立日志和输出目录，失败立即使流水线非零退出；无候选为预注册终止而非执行失败。脚本只读取 `PPTT_ASSET_ROOT` 和工作目录参数，不含连接信息。

- [ ] **Step 4: 运行测试与 Shell 静态检查**

Run: `python -m pytest tests/unit/test_v8_progress_monitor.py -q`

Run: `bash -n scripts/server/run_v8_transunet_pipeline.sh`

- [ ] **Step 5: 提交**

```bash
git add scripts/monitor_v8_progress.py scripts/server/run_v8_transunet_pipeline.sh tests/unit/test_v8_progress_monitor.py
git commit -m "feat: orchestrate and monitor V8 replication"
```

### Task 8: 全量本地审查、服务器正式执行与结论交付

**Files:**
- Modify: `docs/reproducibility_cn.md`
- Modify: `docs/final_experiment_audit_cn.md`
- Create: `docs/v8_transunet_mechanism_replication_results_cn.md`
- Modify: `manifests/result_inventory.json`

- [ ] **Step 1: 运行本地全量测试、格式/导入和隐私检查**

Run: `python -m pytest -q`

Run: `python -m compileall -q src scripts tests`

Run: `git diff --check`

Expected: all tests pass; compile and diff checks exit 0.

- [ ] **Step 2: 推送代码并在服务器核验同一提交**

服务器只拉取已提交代码；正式锁文件记录 `git rev-parse HEAD`。不得上传 `.codex_transfer`、密钥或本地临时文件。

- [ ] **Step 3: 执行验证集阶段并审计候选**

Run on server: `PPTT_ASSET_ROOT=/root/autodl-tmp/A_scheme_workspace/brats2023_data bash scripts/server/run_v8_transunet_pipeline.sh --workspace /root/autodl-tmp/A_scheme_workspace/pptt_process_xai_workspace`

Console monitor: `python scripts/monitor_v8_progress.py --workspace-root . --watch --interval 30`

验证：每种子 125 名验证患者；候选 JSON 最多一条；协议锁时间早于任何测试患者输出。若无候选，记录边界并停止。

- [ ] **Step 4: 审计并行测试与正式统计**

验证：授权时每种子 250 名有序测试患者，无重复、无缺失、无非有限统计；必要性、恢复性、特异性、剂量和算子审计逐项出状态。不得因结果失败更换候选或门槛。

- [ ] **Step 5: 写结果与研究者指引**

结果文档只依据正式 JSON/Parquet，区分：过程诊断创新、干预忠实性结果、跨架构复现状态、失败边界。研究者指引必须给出可执行诊断字段和可检验设计假设，不把 TransUNet 单一候选写成通用结构规律。

- [ ] **Step 6: 更新可复现命令、结果清单和最终审计**

记录所有命令、提交哈希、配置哈希、资产清单、正式/调试边界和图源哈希；V7 继续保持 `INSUFFICIENT_NETWORK_COVERAGE`。

- [ ] **Step 7: 最终对抗式审查并提交**

检查问题：是否仍是过程性方法；是否只用终点 Dice 证明价值；观察器是否被误写为真实机制；是否泄漏测试集；是否有结果导向调参；是否把单路径复现扩写为全网络或跨架构同机制。任一答案不合格则不进入论文主张。

```bash
git add docs/reproducibility_cn.md docs/final_experiment_audit_cn.md docs/v8_transunet_mechanism_replication_results_cn.md manifests/result_inventory.json
git commit -m "docs: audit formal TransUNet mechanism replication"
```

## 完成门槛

- 三张主图和一张 V7 补充图通过数据、语义和视觉契约。
- TransUNet 只可能产生一个验证集候选或明确无候选状态。
- 测试干预只能由不可变协议锁授权，并严格复用患者、路径、算子、剂量和统计门槛。
- 结论只在至少两个种子通过必要性、恢复性、特异性、剂量和算子审计后升级为架构特异的干预忠实性复现。
- 不增加第二数据集、三维网络、CAM/SHAP、注意力头筛选、任意通道搜索或 V7 补救实验。
