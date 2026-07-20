# PPTT 复现说明

## 1. 固定范围

本仓库实现像素预测状态转移追踪（Pixel Prediction Transition Tracing, PPTT）。核心输出包括节点预测状态、相邻节点转移场、类别转移张量、性能变化精确重构、持续状态流、最终决定深度、首次正确深度和错误起源深度。候选过程经独立数据划分选择后，可通过像素转移因果追踪检验指定结构变量的必要性、恢复性、区域特异性和剂量关系；全网络扫描进一步在 7 个过程区间与 7 个恢复节点上审计干预响应和覆盖。CAM、LayerCAM 与相邻硬掩膜仅用于 V2 独立比较，不参与 PPTT 状态定义。

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
.venv/bin/python scripts/run_process_effect_gate.py --workspace-root . --config configs/experiments/process_effect_gate.yaml
.venv/bin/python scripts/run_v6_pixel_transition_causal.py --workspace-root . --config configs/experiments/v6_pixel_transition_causal_tracing.yaml --protocol-lock results/v6_pixel_transition_causal/v6_protocol_lock_v2.json --asset-root "$PPTT_ASSET_ROOT" --model-seed 42 --resume --device cuda
.venv/bin/python scripts/run_v6_pixel_transition_causal.py --workspace-root . --config configs/experiments/v6_pixel_transition_causal_tracing.yaml --protocol-lock results/v6_pixel_transition_causal/v6_protocol_lock_v2.json --asset-root "$PPTT_ASSET_ROOT" --model-seed 123 --resume --device cuda
.venv/bin/python scripts/run_v6_pixel_transition_causal.py --workspace-root . --config configs/experiments/v6_pixel_transition_causal_tracing.yaml --protocol-lock results/v6_pixel_transition_causal/v6_protocol_lock_v2.json --asset-root "$PPTT_ASSET_ROOT" --model-seed 3407 --resume --device cuda
.venv/bin/python scripts/summarize_v6_causal.py --workspace-root . --config configs/experiments/v6_pixel_transition_causal_tracing.yaml
.venv/bin/python scripts/export_v6_causal_case.py --workspace-root . --config configs/experiments/v6_pixel_transition_causal_tracing.yaml --protocol-lock results/v6_pixel_transition_causal/v6_protocol_lock_v2.json --asset-root "$PPTT_ASSET_ROOT" --device cuda
~~~

第一次 V1 汇总允许在冻结观察器控制尚未生成时返回 `PENDING_RANDOMIZATION`，用于审查其余观察器控制。参数随机化主控制只在预注册的 seed 42 上运行，固定原观察器并仅随机化所检验的网络模块；其数学定义、原重训诊断结果和输出契约见 `docs/parameter_randomization_protocol_amendment_cn.md`。旧的随机化后重训观察器结果保留为诊断，不参与 V1 门控。只有第二次 V1 汇总的正式矩阵状态为 `PASS` 时才能运行 V3；V3 同时逐作业读取 `v1_status.json` 并硬性要求 `PASS`，状态缺失、失败或待定的作业均不执行轨迹分析。

V3 在内存中使用完整观察器 margin 计算预注册可靠性掩膜，但患者轨迹只持久化离散节点状态、可靠性掩膜、真值、最终模型状态和切片标识。margin 浮点图不参与 V3 任一统计量，且可由锁定权重、观察器和阈值重新计算；不持久化该冗余张量不会改变状态转移、指标重构或患者级统计。该策略由 `trace_storage` 配置和作业清单共同锁定。

V4 仅在 V3 对固定候选 `up3->up4` 的预注册触发条件成立时运行，未触发时只生成 `NOT_TRIGGERED.json`。正式结果中 V4 为 `NOT_TRIGGERED`，不得更换候选转移或降低阈值。V5 只检验接口迁移和过程输出完整性，不要求复制 U-Net 曲线。

V6 是与 V4 分开的候选特异性检验。全局双指标门控仍记录为失败；V6 只针对验证集上跨三个种子通过大效应门控的 `up1->up2` 持续净恢复，检验 `up2` 接收的 `down2` 空间对齐特征。候选、模型种子、测试患者、空间干预、恢复剂量、匹配规则、统计阈值和多重检验均在首次干预前写入 `v6_protocol_lock_v2.json`。该锁仅在首次干预前修正观察器种子身份，并保留被替代锁的路径、哈希和修订原因。正式重放必须直接使用该锁；不得由干预结果重选路径、阈值或病例。

V7 在首次 test 干预前锁定完整的 7×7 过程—恢复节点矩阵。先生成协议锁，再并行运行六个模型—种子作业，最后统一汇总：

~~~bash
.venv/bin/python scripts/lock_v7_network_alignment_protocol.py \
  --workspace-root . \
  --config configs/experiments/v7_network_process_intervention_alignment.yaml \
  --asset-root "$PPTT_ASSET_ROOT"

for model in unet_baseline unet_noskip; do
  for seed in 42 123 3407; do
    .venv/bin/python scripts/run_v7_network_alignment.py \
      --workspace-root . \
      --config configs/experiments/v7_network_process_intervention_alignment.yaml \
      --protocol-lock results/v7_network_process_intervention_alignment/v7_protocol_lock.json \
      --asset-root "$PPTT_ASSET_ROOT" \
      --model "$model" --model-seed "$seed" \
      --execution-mode formal --resume --device cuda &
  done
done
wait

.venv/bin/python scripts/summarize_v7_network_alignment.py \
  --workspace-root . \
  --config configs/experiments/v7_network_process_intervention_alignment.yaml \
  --protocol-lock results/v7_network_process_intervention_alignment/v7_protocol_lock.json
~~~

正式结果必须为六作业各 250 名患者、每名患者 49 个单元。当前门控结果为 `INSUFFICIENT_NETWORK_COVERAGE`；该状态是正式结果，不得降低覆盖阈值或更换过程群。首次汇总曾因 Pandas 尝试序列化 DataFrame 属性而在写表阶段停止；修复只让 Parquet 存储副本清空 `attrs`，不改变数值列。锁定提交、原错误、修订范围和正式结果哈希见 `v7_postprocessing_amendment.json`。

## 6. TransUNet 过程候选与顺序确认

V8 使用验证集过程轨迹在四条预声明 TransUNet 解码输入路径中最多选择一条候选，并在首次测试干预前生成内容寻址协议锁。三个模型种子的验证和正式干预均由服务器脚本并行执行：

~~~bash
cd /root/autodl-tmp/A_scheme_workspace/pptt_process_xai_workspace
PPTT_ASSET_ROOT=/root/autodl-tmp/A_scheme_workspace/brats2023_data \
PYTHON_BIN="$PWD/.venv/bin/python" \
bash scripts/server/run_v8_transunet_pipeline.sh --workspace "$PWD"
~~~

V8 锁定 `bottleneck_to_up1`。必要性、恢复性和剂量关系通过，但区域特异性因匹配位置不足而不可评估。该结果保持在 `results/v8_transunet_mechanism/`，不得修改后重跑为正结论。

V9 在 V8 不变的前提下把验证集匹配可行性加入候选准入，并只对 V8 三个种子均未执行任何干预的共同患者运行：

~~~bash
PPTT_WORKSPACE_ROOT="$PWD" \
PPTT_ASSET_ROOT=/root/autodl-tmp/A_scheme_workspace/brats2023_data \
PPTT_PYTHON_BIN="$PWD/.venv/bin/python" \
bash scripts/server/run_v9_transunet_specificity.sh
~~~

V9 锁定 `skip_down3_to_up1`，但 26 人队列的可评估覆盖不足。V10 保持同一路径、算子和门槛，对剩余 224 名尚无该路径干预结局的测试患者进行最后一次顺序确认：

~~~bash
PPTT_WORKSPACE_ROOT="$PWD" \
PPTT_ASSET_ROOT=/root/autodl-tmp/A_scheme_workspace/brats2023_data \
PPTT_PYTHON_BIN="$PWD/.venv/bin/python" \
bash scripts/server/run_v10_transunet_path_confirmation.sh
~~~

V10 三个种子均通过必要性，恢复性未达到预注册 `d >= 0.8`，区域特异性置信区间均跨 0。最终科学状态为 `TRANSUNET_PROCESS_CANDIDATE_NOT_CAUSALLY_CONFIRMED`。V8-V10 统一按三轮 `alpha=0.05/3` 解释；不得在当前测试集运行第四轮、改选路径或降低门槛。完整数值和独立重构审计见 `docs/v8_transunet_mechanism_replication_results_cn.md`。

上述脚本包含“锁或正式目录已存在即拒绝执行”的污染保护，只适用于从未生成这些正式结果的干净复现实例。已有正式目录不得删除后原位重跑。

## 7. 图件与结果清单

~~~bash
.venv/bin/python scripts/make_paper_figures.py --results-root results --language en --dpi 400
.venv/bin/python scripts/make_paper_figures.py --results-root results --language zh --dpi 400
.venv/bin/python scripts/make_pptt_method_figure.py --workspace-root . --language en --dpi 600
.venv/bin/python scripts/make_pptt_method_figure.py --workspace-root . --language zh --dpi 600
.venv/bin/python scripts/make_network_pixel_fate_figure.py --workspace-root . --language en --dpi 600
.venv/bin/python scripts/make_network_pixel_fate_figure.py --workspace-root . --language zh --dpi 600
.venv/bin/python scripts/make_network_alignment_figure.py --workspace-root . --language en --dpi 600
.venv/bin/python scripts/make_network_alignment_figure.py --workspace-root . --language zh --dpi 600
.venv/bin/python scripts/make_v6_causal_figure.py --workspace-root . --language en --dpi 600
.venv/bin/python scripts/make_v6_causal_figure.py --workspace-root . --language zh --dpi 600
PPTT_ASSET_ROOT=/root/autodl-tmp/A_scheme_workspace/brats2023_data \
  .venv/bin/python scripts/build_result_inventory.py
PPTT_ASSET_ROOT=/root/autodl-tmp/A_scheme_workspace/brats2023_data \
  .venv/bin/python scripts/build_result_inventory.py --verify
~~~

主文图固定为三组：图 1 定义 PPTT 的计算流程，图 2 展示相近终点下的全路径过程差异，图 3 展示完整 7×7 过程—恢复节点矩阵、反事实病例和宏微效应。V6 候选路径图与 `make_paper_figures.py` 生成的其余图件作为补充材料候选。图 3 的灰色单元表示不可评估，不得改绘为零。

`manifests/result_inventory.json` 记录代码 commit、环境、运行命令、配置哈希、数据清单、模型权重、输出表行数、图件大小和所有结果文件 SHA-256。服务器目录不含 Git 元数据时，必须从协议锁注入 `--code-commit` 和 `--code-dirty false`；不得伪造提交。含 `smoke`、`debug`、`overfit`、`preflight`、`resume_check` 或 `dry_run` 的产物自动标为非正式结果。V10 收口后的服务器只读临时清单包含 8,620 个产物并通过内容寻址复核；按照当前同步约束，该临时清单尚未拉取或覆盖本地仓库中的较早快照。

## 8. 结果同步

~~~bash
rsync -av --info=progress2 \
  --exclude='.git/' --exclude='.venv/' --exclude='__pycache__/' \
  -e "ssh -p 25410" \
  root@connect.westc.seetacloud.com:/root/autodl-tmp/A_scheme_workspace/pptt_process_xai_workspace/ \
  /e/work/master2026/A_scheme_workspace/pptt_process_xai_workspace/
~~~

同步后应再次运行结果清单校验。密码只在交互式 SSH 提示中输入，不写入命令文件、配置、日志或 Git。

## 9. V11 跨架构过程因果抽象

V11 只检验一个方法主张：架构特异的内部张量能否通过同一五状态像素决策语言、闭式最小范数状态交换和低容量转移过程，预测 U-Net 与 TransUNet 在全部八个节点干预后的完整下游反事实轨迹。no-skip U-Net 仅作为全路径结构敏感性对照；CAM、逐跳接屏蔽和局部路径排名不属于 V11 方法。

正式 test 干预前按以下顺序生成验证集模型、校准和不可变计划：

~~~bash
export PPTT_ASSET_ROOT=/root/autodl-tmp/A_scheme_workspace/brats2023_data

.venv/bin/python scripts/lock_v11_causal_abstraction_protocol.py \
  --workspace-root . --phase fit-processes \
  --asset-root "$PPTT_ASSET_ROOT"

for model in unet_baseline unet_noskip transunet_r50_vit_b16; do
  for seed in 42 123 3407; do
    .venv/bin/python scripts/prepare_v11_causal_abstraction.py \
      --workspace-root . --phase validation-job \
      --asset-root "$PPTT_ASSET_ROOT" \
      --model "$model" --model-seed "$seed" --device cuda
  done
done

.venv/bin/python scripts/prepare_v11_causal_abstraction.py \
  --workspace-root . --phase aggregate-validation

for model in unet_baseline unet_noskip transunet_r50_vit_b16; do
  for seed in 42 123 3407; do
    .venv/bin/python scripts/prepare_v11_causal_abstraction.py \
      --workspace-root . --phase formal-job \
      --asset-root "$PPTT_ASSET_ROOT" \
      --model "$model" --model-seed "$seed" --device cuda
  done
done

.venv/bin/python scripts/lock_v11_causal_abstraction_protocol.py \
  --workspace-root . --phase lock \
  --asset-root "$PPTT_ASSET_ROOT"
~~~

协议锁固定全部 9 个权重、216 个观察器文件、验证/测试患者顺序、9 份患者干预计划、三个高层过程、校准结果、阈值和源代码哈希。正式汇总额外拒绝缺失剂量、缺失下游深度、task/null 像素集合不一致、选择性可靠性损失、smoke 混入和非有限数值。

锁定后由显存预算调度器运行正式矩阵。调度器先在验证患者上分别测量 U-Net 与 TransUNet 的峰值显存，画像不保存科学输出；随后仅在估计峰值总和低于 21.5 GiB 时并行独立作业，已完整完成的作业不会重复启动。

~~~bash
PPTT_WORKSPACE_ROOT="$PWD" \
PPTT_ASSET_ROOT="$PPTT_ASSET_ROOT" \
PPTT_PYTHON_BIN="$PWD/.venv/bin/python" \
bash scripts/server/run_v11_causal_abstraction.sh
~~~

只读实时监控命令为：

~~~bash
.venv/bin/python scripts/monitor_v11_causal_abstraction.py \
  --workspace-root . --watch-seconds 10
~~~

监控器只读取原子患者产物、结构化日志、PID、GPU 和磁盘状态，不启动、终止或修改作业。只有 `results/v11_causal_abstraction/v11_status.json` 能授权正式结论；结构对照不分离时，即使共享全网络抽象通过，也必须明确标记为不支持结构诊断敏感性。结果图和论文结论均在该状态生成后另行制作。
