from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import yaml

from pptt.data.brats2d import BraTS2DDataset
from pptt.models.adapters import build_adapter
from pptt.models.transunet import load_pretrained_npz
from pptt.training.engine import seed_everything
from pptt.training.losses import segmentation_loss


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return payload


def _core_model(adapter: torch.nn.Module) -> torch.nn.Module:
    core = getattr(adapter, "model", None)
    if not isinstance(core, torch.nn.Module):
        raise TypeError("Model adapter does not expose its underlying model")
    return core


def _asset_root(data_config: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        root = override
    else:
        asset = data_config["asset_root"]
        root = Path(
            os.environ.get(
                str(asset["environment_variable"]),
                str(asset["default"]),
            )
        )
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure complete training-step memory by physical batch size."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/models/transunet_r50_vit_b16.yaml"),
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/brats2023_2d.yaml"),
    )
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128],
    )
    parser.add_argument("--memory-limit-gib", type=float)
    parser.add_argument(
        "--output",
        type=Path,
    )
    args = parser.parse_args()

    model_config = _load_yaml(args.config)
    data_config = _load_yaml(args.data_config)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The formal memory profile requires a CUDA device")
    asset_root = _asset_root(data_config, args.asset_root)
    candidates = tuple(sorted(set(int(value) for value in args.batch_sizes)))
    if any(value <= 0 for value in candidates):
        raise ValueError("Candidates must be positive integers")
    train_split = data_config["splits"]["train"]
    dataset = BraTS2DDataset(
        asset_root / str(train_split["image_dir"]),
        asset_root / str(train_split["mask_dir"]),
    )
    maximum_batch = max(candidates)
    samples = [dataset[index] for index in range(maximum_batch)]
    images = torch.stack([sample["image"] for sample in samples])
    labels = torch.stack([sample["label"] for sample in samples])
    configured_limit = model_config["training"].get("memory_limit_gib", 20.0)
    memory_limit_gib = float(
        args.memory_limit_gib if args.memory_limit_gib is not None else configured_limit
    )
    if not 0 < memory_limit_gib < torch.cuda.get_device_properties(device).total_memory / 2**30:
        raise ValueError("memory-limit-gib must be positive and below total CUDA memory")
    limit_bytes = int(memory_limit_gib * 2**30)
    rows: list[dict[str, Any]] = []

    for batch_size in candidates:
        seed_everything(
            args.seed,
            deterministic=bool(model_config["training"]["deterministic"]),
        )
        adapter = build_adapter(
            str(model_config["name"]),
            n_channels=int(model_config["n_channels"]),
            num_classes=int(model_config["num_classes"]),
            bilinear=bool(model_config.get("bilinear", False)),
            img_size=int(model_config.get("img_size", 160)),
        )
        initialization = model_config["initialization"]
        mode = str(initialization["mode"])
        initialization_coverage = None
        if mode == "official_npz":
            source = Path(str(initialization["pretrained_npz"]))
            source = source if source.is_absolute() else asset_root / source
            load_report = load_pretrained_npz(_core_model(adapter), source)
            initialization_coverage = load_report.parameter_coverage
        elif mode != "scratch":
            raise ValueError(f"Unsupported initialization mode: {mode}")
        adapter.to(device).train()
        optimizer = torch.optim.AdamW(
            adapter.parameters(),
            lr=float(model_config["training"]["learning_rate"]),
            weight_decay=float(model_config["training"]["weight_decay"]),
        )
        batch_images = images[:batch_size].to(device)
        batch_labels = labels[:batch_size].to(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        start = perf_counter()
        passed = False
        error: str | None = None
        try:
            optimizer.zero_grad(set_to_none=True)
            logits = adapter(batch_images)
            loss = segmentation_loss(
                logits,
                batch_labels,
                cross_entropy_weight=float(
                    model_config["training"]["cross_entropy_weight"]
                ),
                dice_weight=float(model_config["training"]["dice_weight"]),
            )
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)
            passed = bool(torch.isfinite(loss).item())
            loss_value = float(loss.detach().item())
        except torch.cuda.OutOfMemoryError as exc:
            loss_value = None
            error = f"{type(exc).__name__}: {exc}"
            torch.cuda.empty_cache()
        elapsed = perf_counter() - start
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        peak_cuda = max(peak_allocated, peak_reserved)
        rows.append(
            {
                "batch_size": batch_size,
                "effective_batch_size": batch_size,
                "accumulation_steps": 1,
                "passed": passed and peak_cuda <= limit_bytes,
                "step_completed": passed,
                "peak_cuda_memory_bytes": peak_cuda,
                "peak_cuda_memory_gib": peak_cuda / 2**30,
                "peak_cuda_allocated_bytes": peak_allocated,
                "peak_cuda_allocated_gib": peak_allocated / 2**30,
                "peak_cuda_reserved_bytes": peak_reserved,
                "peak_cuda_reserved_gib": peak_reserved / 2**30,
                "memory_limit_bytes": limit_bytes,
                "elapsed_seconds": elapsed,
                "loss": loss_value,
                "error": error,
                "initialization_coverage": initialization_coverage,
            }
        )
        del optimizer, adapter, batch_images, batch_labels
        gc.collect()
        torch.cuda.empty_cache()

    passing = [row["batch_size"] for row in rows if row["passed"]]
    selected = max(passing) if passing else None
    payload = {
        "status": "PASS" if selected is not None else "FAIL",
        "model": str(model_config["name"]),
        "seed": args.seed,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "configured_effective_batch_size": int(
            model_config["training"]["effective_batch_size"]
        ),
        "memory_limit_gib": memory_limit_gib,
        "selected_batch_size": selected,
        "selected_effective_batch_size": selected,
        "selected_accumulation_steps": 1 if selected is not None else None,
        "profiles": rows,
    }
    output = args.output or Path(
        f"results/training_resource/{model_config['name']}_batch_profile.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
