# 当前脚本入口：现象原因解释

本目录是 5.12 组会后新的唯一实验脚本入口。后续脚本必须服务于：

**解释 U-Net 有跳接/无跳接曲线中“后期加速”和“后期减速”现象产生的原因。**

当前禁止把新脚本写成继续分析 `skip` 作用、结构依赖排序或模型改进。

## 当前应实现的脚本顺序

1. `run_phase0_lock_stage1_acceleration.sh`：锁定 Stage 1 曲线，导出 `task_dice_gt`、语义速度和语义加速度。
2. `run_phase0_3_phenomenon_cause_gradcam.sh`：当前已实现入口，导出 baseline/no-skip 的 `down1/down2/down3/down4/up1/up2/up3/up4` 分割 CAM、raw feature response、相关性、GT 对齐、熵、速度和加速度。
3. `run_stage1_case_readout_alignment.sh`：导出病例级 Stage 1 readout，并计算同一病例相邻节点上的 `task_dice_gt` 增量与 CAM/raw feature response 增量相关性。
4. `run_stage1_common_case_sets_and_claim_matrix.sh`：构建 `smoke_n16/dev_n64/formal_n128/formal_n512` 统一病例集，并初始化 claim-level evidence matrix。
5. `run_stage1_feature_response_robustness.sh`：E5 raw feature response 聚合鲁棒性实验，比较 L2、mean_abs、max_abs、top-k-channel 和 positive-only。
6. `run_stage1_cam_sensitivity.sh`：E6 CAM target/variant 敏感性实验，检查 `pred_fg/gt_fg` 与 Grad-CAM/HiResCAM-style 是否改变主趋势。
7. `run_stage1_response_guided_perturbation.sh`：E4 response-guided faithfulness perturbation，比较 top-k response、random same-area 和 low-response 扰动。
8. `run_stage1_class_region_response_analysis.sh`：E9 三类区域响应落点分析，默认覆盖 `down1-down4-up1-up4` 全部 Stage 1 节点，量化 CAM/原始特征响应在背景、水肿、坏死/非增强核心、增强肿瘤中的响应质量、top10 富集和与遮挡下降的关系。
9. `run_stage1_statistical_validation.sh`：E10 统计严谨性补充，对 n512 主线证据进行配对检验、FDR 校正和 bootstrap 置信区间，避免只用均值差解释现象。
10. `run_stage1_up_source_decomposition.sh`：E11 上采样节点来源分解，拆解有跳接 `up1-up4` 内部的主路径、跳接输入、拼接输入和卷积输出响应，定位后期恢复主要发生在哪个结构环节。
11. `run_stage1_structural_cause_validation.sh`：E12 结构原因补充，围绕 `up1-up4` 做跳接分支级扰动、张量空间梯度/边界证据、融合卷积前后变化，解释有跳接 U-Net 后期语义恢复主要来自哪个结构环节。
12. `run_stage1_up4_skip_counterfactual_mediation.sh`：E13 核心机制补充，以 E12 定位出的 `up4` 跳接分支为靶点，做病灶/边界区域反事实替换、同面积背景/随机对照和中介式路径分析，检验空间对齐的病灶/边界 skip 信息是否通过融合后任务一致响应影响输出。
13. `run_stage1_up4_skip_counterfactual_visualization.sh`：E13 论文可视化入口，生成反事实替换区域、替换前后 `up4` skip 特征响应、融合响应和最终概率图的放大可视化，并导出病例级数字读数。
14. 后续若拆分脚本，Phase 1 必须覆盖全网络节点；只分析 `up1-up4` 时必须标注为 decoder 子实验。
15. 后续 Phase 2/3 指标必须同时解释早期差距、中部低谷、后期加速/减速和最终交叉。

## 新实验套件输出约定

2026-05-17 后新增实验默认写入：

```text
stage1_explanation_suite_20260517/results/
```

每个新增脚本必须支持 smoke 模式或小样本快速检查，并输出 `run_config.json` 和 `run_manifest.csv`。

## 归档说明

5.12 组会前的旧结构干预脚本已经移入归档目录。归档内容只用于追溯历史，不作为当前实验入口；新脚本只能写在本目录下。
