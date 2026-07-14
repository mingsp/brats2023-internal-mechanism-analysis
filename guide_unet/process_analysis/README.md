# 网络内部推理过程可解释分析代码

本目录包含可解释分析方法核心与当前 U-Net 验证实验。代码围绕同一条内部节点路径组织语义状态、空间响应、组织区域、输出功能和结构来源，避免把 CAM、遮挡和结构干预写成彼此独立的实验列表。

## 方法核心

`internal_inference_framework.py` 提供 F/G 的规范实现：

1. `NodeState` 接收已计算的节点语义、响应、区域和功能指标；
2. `transition_function` 对相邻节点构造带名称的局部转移结果；
3. `global_integration_function` 按网络计算顺序组成全局矩阵和有效性掩码；
4. 缺失或不适用指标保留为无效项，不补零，不参与对应转移；
5. G 不拟合 Dice、不学习组合权重，也不把不同量纲指标压缩为单一分数。

## 验证代码对应关系

- `export_stage1_case_readout_alignment.py`：节点任务语义状态与病例级转移增量；
- `analyze_phenomenon_cause_gradcam.py`：全节点 CAM、原始特征响应及空间对齐；
- `stage1_class_region_response_analysis.py`：类别区域分配和高响应区域富集；
- `stage1_response_guided_perturbation.py`：高响应区域的输出功能检验；
- `stage1_structural_cause_validation.py`：分支置零、空间错位和融合前后分析；
- `stage1_up_source_decomposition.py`：解码融合节点的输入来源分解；
- `stage1_up4_skip_counterfactual_mediation.py`：病灶区域反事实替换；
- `stage1_statistical_validation.py`：患者层级统计与聚类重采样。

## 当前验证边界

当前实验对象为 BraTS2023 二维 U-Net 与无跳接 U-Net。该设置用于验证方法能否从节点曲线定位关键转移，再连接空间落点、输出功能和结构来源；它不构成跨架构或跨数据集的普适性证明。

## 代码约束

- 真实 CAM、扰动和反事实结果必须由对应实验脚本计算，不使用代理热图替代；
- 统计对象、有效样本数和缺失处理必须随结果输出；
- 结构干预结论限定在当前模型实例和实验设置；
- 论文与 Word/LaTeX 生成脚本不属于本目录的公开源代码。
