# PPTT 复现说明

## 1. 固定范围

本仓库实现像素预测状态转移追踪（Pixel Prediction Transition Tracing, PPTT）。核心输出包括节点预测状态、相邻节点转移场、类别转移张量、性能变化精确重构、持续状态流、最终决定深度、首次正确深度和错误起源深度。CAM、LayerCAM 与相邻硬掩膜仅用于 V2 独立比较，不参与 PPTT 状态定义。

当前代码快照为 `71fcef1e872f2deb205c2a24281131326b1fc50a`。服务器工作区为 `/root/autodl-tmp/A_scheme_workspace/pptt_process_xai_workspace`，资产根目录为 `/root/autodl-tmp/A_scheme_workspace/brats2023_data`。

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

当前可用的正式分割权重为 U-Net seed 42 和 no-skip U-Net seed 42。U-Net seed 123/3407、no-skip U-Net seed 123/3407、TransUNet seed 42/123/3407 尚未生成，因此正式多种子分析不得提前执行或下结论。

## 4. 短流程重放

~~~bash
.venv/bin/python scripts/replay_short_flows.py
~~~

该命令将现有 V0、V2 目录安全移出，在原结果目录不存在的条件下调用固定配置重新生成结果，再同时比较文件 SHA-256 和 JSON 数值内容。审计记录写入 `manifests/short_replay_audit.json`。本次重放中 V0 与 V2 均达到逐字节一致。

## 5. 正式长流程

以下训练矩阵超过两小时，本轮没有擅自启动：

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

该矩阵包含 7 个 100 epoch 作业，顺序运行预计为数十 GPU 小时，实际耗时受早停和数据吞吐影响。TransUNet batch size 8 的服务器实测峰值显存为 2.83 GiB，所有作业仍受 20 GiB 预注册上限约束。日志写入 `logs/segmentation_training/`；`--resume` 从各作业的 `latest_training.pt` 恢复。

权重齐备后依次运行：

~~~bash
.venv/bin/python scripts/train_observers.py --matrix manifests/model_matrix.yaml --config configs/observers/default.yaml --asset-root "$PPTT_ASSET_ROOT" --output-root results/v1_observers --resume --device cuda
.venv/bin/python scripts/run_v1_observer.py --matrix manifests/model_matrix.yaml --config configs/observers/default.yaml --asset-root "$PPTT_ASSET_ROOT" --output-root results/v1_observers --device cuda
.venv/bin/python scripts/run_parameter_randomization.py --matrix manifests/model_matrix.yaml --config configs/observers/default.yaml --asset-root "$PPTT_ASSET_ROOT" --output-root results/v1_observers --resume --device cuda
.venv/bin/python scripts/run_v3_unet_pair.py --config configs/experiments/v3_unet_pair.yaml --asset-root "$PPTT_ASSET_ROOT" --resume --device cuda
.venv/bin/python scripts/run_v4_intervention.py --config configs/experiments/v4_up4_intervention.yaml --trigger results/v3_unet_pair/v4_trigger.json --asset-root "$PPTT_ASSET_ROOT" --resume --device cuda
.venv/bin/python scripts/run_v5_transunet.py --config configs/experiments/v5_transunet.yaml --asset-root "$PPTT_ASSET_ROOT" --resume --device cuda
~~~

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
