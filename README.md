# BraTS2023 Internal Mechanism Analysis Code

This repository contains source code for a Stage 1 feature-map semantic dynamics analysis of 2D U-Net brain tumor segmentation models on BraTS2023-style data.

The code release includes:

- baseline and no-skip U-Net model definitions;
- BraTS preprocessing, dataset loading, training, and evaluation code;
- Stage 1 node readout, CAM, raw feature response, perturbation, statistical validation, structural-source, and counterfactual analysis scripts;
- shell entry points for reproducing the main analysis pipeline.

The repository intentionally excludes datasets, checkpoints, generated results, generated figures, paper drafts, Word/LaTeX build scripts, and local workspace archives.

## Layout

```text
guide_unet/
  Architecture/          U-Net model definitions
  datasets/              BraTS slice dataset loader
  preprocess/            BraTS preprocessing utilities
  train/                 training entry point
  eval/                  evaluation entry point
  internal_mechanism/    internal mechanism analysis utilities
  process_analysis/      Stage 1 explanation and validation scripts

scripts/current_phenomenon_cause/
  run_*.sh               reproducible shell entry points for Stage 1 analyses
```

## Environment

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version from the official PyTorch instructions if GPU execution is required.

## Data And Checkpoints

The code expects preprocessed data and model checkpoints to be supplied locally. The run scripts use environment variables so paths do not need to be edited in source files:

```bash
export DATA_DIR=/path/to/data
export BASELINE_CKPT=/path/to/baseline_seed42_best_val_loss.pth
export NOSKIP_CKPT=/path/to/noskip_unet_seed42_best_val_loss.pth
export OUTPUT_ROOT=/path/to/output
```

Then run a script such as:

```bash
bash scripts/current_phenomenon_cause/run_stage1_statistical_validation.sh
```

## Scope

This is a source-code release for the Stage 1 phenomenon explanation pipeline. It is not a packaged Python library and does not include private data, trained weights, or generated manuscript artifacts.
