# Internal Inference Explainability for Medical Image Segmentation

This repository contains the source code for a process-oriented explainability method that analyzes how task-related information changes along internal feature-tensor paths of trained medical image segmentation networks. The current validation case uses 2D U-Net and no-skip U-Net models on BraTS2023-style data.

The code release includes:

- baseline and no-skip U-Net model definitions;
- BraTS preprocessing, dataset loading, training, and evaluation code;
- a reusable F/G assembly module for local transition analysis and global path integration;
- node readout, CAM, raw feature response, region allocation, perturbation, statistical validation, structural-source, and counterfactual analysis scripts;
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
  process_analysis/      method core plus explanation and validation scripts

scripts/current_phenomenon_cause/
  run_*.sh               reproducible shell entry points for Stage 1 analyses

tests/
  test_internal_inference_framework.py
```

## Method Components

The method represents an ordered internal path as feature-tensor nodes and analyzes every adjacent transition with named metric blocks:

```text
internal nodes -> comparable node states -> F transition records
               -> ordered transition matrix + validity mask -> G path output
```

`guide_unet/process_analysis/internal_inference_framework.py` implements the architecture-independent F/G assembly. It accepts measurements produced by the experiment scripts, keeps semantic, response, region, functional, and structural metrics separate, and preserves missing or inapplicable evidence with an explicit validity mask. G does not learn weights or collapse heterogeneous measurements into a new performance score.

The current U-Net validation maps to the method as follows:

- node semantic state: `export_stage1_case_readout_alignment.py`;
- CAM and raw feature response: `analyze_phenomenon_cause_gradcam.py`;
- class-region allocation and enrichment: `stage1_class_region_response_analysis.py`;
- output-function perturbation: `stage1_response_guided_perturbation.py`;
- structural source and fusion analysis: `stage1_structural_cause_validation.py` and `stage1_up_source_decomposition.py`;
- lesion-region counterfactual analysis: `stage1_up4_skip_counterfactual_mediation.py`;
- patient-level statistical validation: `stage1_statistical_validation.py`.

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

This is a research source release for the explainability method and its current Stage 1 validation pipeline. It is not a packaged clinical product and does not include private data, trained weights, generated results, or manuscript artifacts.
