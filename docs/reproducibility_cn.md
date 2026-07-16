# PPTT 复现说明

## 1. 固定范围

本仓库实现像素预测状态转移追踪（Pixel Prediction Transition Tracing, PPTT）。核心输出包括节点预测状态、相邻节点转移场、类别转移张量、性能变化精确重构、持续状态流、最终决定深度、首次正确深度和错误起源深度。CAM、LayerCAM 与相邻硬掩膜仅用于 V2 独立比较，不参与 PPTT 状态定义。

服务器工作区为 `/root/autodl-tmp/A_scheme_workspace/pptt_process_xai_workspace`，资产根目录为 `/root/autodl-tmp/A_scheme_workspace/brats2023_data`。每次正式运行的代码提交、配置哈希和环境信息由结果清单登记，不在本文档中固定为易失效的单一提交号。

## 2. 已锁定环境

审计环境为 Python 3.10.8、PyTorch 2.1.2+cu118、CUDA runtime 11.8、cuDNN 8700，GPU 为 NVIDIA GeForce RTX 4090（24 GB）。完整依赖快照写入 `manifests/environment_lock.txt`，项目依赖版本写入 `pyproject.toml`。

~~~bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -v
~~~

测试必须通过全部核心用例。两个 Windows junction 检查在 Linux 上跳过属于平台条件，不是核心功能 xfail。

## 3. 数据与权重

`PPTT_ASSET_ROOT` 指向资产根目录。`manifests/assets.json` 登记 train、val、test 的二维图像/标签对数量以及完整 SHA-256 清单；`manifests/model_matrix.yaml` 固定 U-Net、有/无跳接 U-Net 和二维 TransUNet 的模型种子、配置及权重位置。结果清单只登记大权重的路径、大小和 SHA-256，不把大文件提交到 Git。

三种模型的 seed 42、123 和 3407 共 9 个正式分割权重均已生成并完成训练摘要及 SHA-256 核验。对应的 72 个节点观察器均已训练完成；正式 V1 状态仍须由下述冻结观察器参数随机化主控制和最终 V1 汇总共同判定。

## 4. 短流程重放

~~~bash
.venv/bin/python scripts/replay_short_flows.py
~~~

该命令将现有 V0、V2 目录安全移出，在原结果目录不存在的条件下调用固定配置重新生成结果，再同时比较文件 SHA-256 和 JSON 数值内容。审计记录写入 `manifests/short_replay_audit.json`。本次重放中 V0 与 V2 均达到逐字节一致。

## 5. 正式长流程

分割训练矩阵的锁定命令为：

~~~bash
.venv/bin/python scripts/run_training_matrix.py \
  --jobs unet_baseline:123 unet_baseline:3407 \
         unet_noskip:123 unet_noskip:3407 \
         transunet_r50_vit_b16:42 transunet_r50_vit_b16:123 \
         transunet_r50_vit_b16:3407 \
  --resume \
  --asset-root /root/autodl-tmp/A_scheme_workspace/brats2023_data \
  --log-dir logs/segmentation_training
~~~

该矩阵包含 7 个新增训练作业；另两个 seed 42 U-Net 权重由锁定资产提供。日志写入 `logs/segmentation_training/`，`--resume` 从各作业的 `latest_training.pt` 恢复。已完成的 9 个权重及训练摘要均应在进入观察器阶段前再次核验。

正式观察器与后续实验按以下顺序运行：

~~~bash
.venv/bin/python scripts/train_observers.py --matrix manifests/model_matrix.yaml --config configs/observers/default.yaml --asset-root "$PPTT_ASSET_ROOT" --output-root results/v1_observers --resume --device cuda
.venv/bin/python scripts/run_v1_observer.py --matrix manifests/model_matrix.yaml --config configs/observers/default.yaml --asset-root "$PPTT_ASSET_ROOT" --output-root results/v1_observers --device cuda
.venv/bin/python scripts/run_parameter_randomization.py --matrix manifests/model_matrix.yaml --config configs/observers/default.yaml --asset-root "$PPTT_ASSET_ROOT" --output-root results/v1_observers --control-mode frozen_observer --resume --device cuda
.venv/bin/python scripts/run_v1_observer.py --matrix manifests/model_matrix.yaml --config configs/observers/default.yaml --asset-root "$PPTT_ASSET_ROOT" --output-root results/v1_observers --device cuda
.venv/bin/python scripts/run_v3_unet_pair.py --config configs/experiments/v3_unet_pair.yaml --asset-root "$PPTT_ASSET_ROOT" --resume --device cuda
.venv/bin/python scripts/run_v4_intervention.py --config configs/experiments/v4_up4_intervention.yaml --trigger results/v3_unet_pair/v4_trigger.json --asset-root "$PPTT_ASSET_ROOT" --resume --device cuda
.venv/bin/python scripts/run_v5_transunet.py --config configs/experiments/v5_transunet.yaml --asset-root "$PPTT_ASSET_ROOT" --resume --device cuda
~~~

第一次 V1 汇总允许在冻结观察器控制尚未生成时返回 `PENDING_RANDOMIZATION`，用于审查其余观察器控制。参数随机化主控制只在预注册的 seed 42 上运行，固定原观察器并仅随机化所检验的网络模块；其数学定义、原重训诊断结果和输出契约见 `docs/parameter_randomization_protocol_amendment_cn.md`。旧的随机化后重训观察器结果保留为诊断，不参与 V1 门控。只有第二次 V1 汇总的正式矩阵状态为 `PASS` 时才能运行 V3；V3 同时逐作业读取 `v1_status.json` 并硬性要求 `PASS`，状态缺失、失败或待定的作业均不执行轨迹分析。

V3 在内存中使用完整观察器 margin 计算预注册可靠性掩膜，但患者轨迹只持久化离散节点状态、可靠性掩膜、真值、最终模型状态和切片标识。margin 浮点图不参与 V3 任一统计量，且可由锁定权重、观察器和阈值重新计算；不持久化该冗余张量不会改变状态转移、指标重构或患者级统计。该策略由 `trace_storage` 配置和作业清单共同锁定。

V4 仅在 V3 预注册触发条件成立时运行，未触发时只生成 `NOT_TRIGGERED.json`。V5 只检验接口迁移和过程输出完整性，不要求复制 U-Net 曲线。

## 6. 图件与结果清单

~~~bash
.venv/bin/python scripts/make_paper_figures.py --results-root results --language en --dpi 400
.venv/bin/python scripts/make_paper_figures.py --results-root results --language zh --dpi 400
PPTT_ASSET_ROOT=/root/autodl-tmp/A_scheme_workspace/brats2023_data \
  .venv/bin/python scripts/build_result_inventory.py
PPTT_ASSET_ROOT=/root/autodl-tmp/A_scheme_workspace/brats2023_data \
  .venv/bin/python scripts/build_result_inventory.py --verify
~~~

`manifests/result_inventory.json` 记录代码 commit、环境、运行命令、配置哈希、数据清单、模型权重、输出表行数、图件大小和所有结果文件 SHA-256。含 `smoke`、`debug`、`overfit`、`preflight`、`resume_check` 或 `dry_run` 的产物自动标为非正式结果。

## 7. 结果同步

~~~bash
rsync -av --info=progress2 \
  --exclude='.git/' --exclude='.venv/' --exclude='__pycache__/' \
  -e "ssh -p 25410" \
  root@connect.westc.seetacloud.com:/root/autodl-tmp/A_scheme_workspace/pptt_process_xai_workspace/ \
  /e/work/master2026/A_scheme_workspace/pptt_process_xai_workspace/
~~~

同步后应再次运行结果清单校验。密码只在交互式 SSH 提示中输入，不写入命令文件、配置、日志或 Git。
