from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import yaml

from pptt.data.brats2d import BraTS2DDataset
from pptt.models.adapters import build_adapter
from pptt.models.transunet import load_pretrained_npz
from pptt.training.engine import (
    TrainingProgress,
    evaluate,
    load_training_checkpoint,
    save_training_checkpoint,
    seed_everything,
    train_one_epoch,
)


MODEL_CONFIGS = {
    "unet_baseline": Path("configs/models/unet_baseline.yaml"),
    "unet_noskip": Path("configs/models/unet_noskip.yaml"),
    "transunet_r50_vit_b16": Path("configs/models/transunet_r50_vit_b16.yaml"),
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _configuration_digest(*payloads: dict[str, Any]) -> str:
    canonical = json.dumps(
        payloads,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


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
        raise FileNotFoundError(f"Asset root does not exist: {root}")
    return root.resolve()


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _epoch_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    epoch: int,
    num_workers: int,
    device: torch.device,
    drop_last: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(
        (int(seed) * 1_000_003 + int(epoch) * 97_409) % (2**63 - 1)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=drop_last,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=False,
    )


def _core_model(adapter: torch.nn.Module) -> torch.nn.Module:
    core = getattr(adapter, "model", None)
    if not isinstance(core, torch.nn.Module):
        raise TypeError("Model adapter does not expose its underlying model")
    return core


def _save_raw_state(path: Path, model: torch.nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(model.state_dict(), temporary)
    os.replace(temporary, path)


def _build_model(
    model_config: dict[str, Any],
    *,
    seed: int,
    asset_root: Path,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    seed_everything(seed, deterministic=bool(model_config["training"]["deterministic"]))
    name = str(model_config["name"])
    adapter = build_adapter(
        name,
        n_channels=int(model_config["n_channels"]),
        num_classes=int(model_config["num_classes"]),
        bilinear=bool(model_config.get("bilinear", False)),
        img_size=int(model_config.get("img_size", 160)),
    )
    initialization = model_config["initialization"]
    mode = str(initialization["mode"])
    report: dict[str, Any] = {"mode": mode}
    if mode == "official_npz":
        source = Path(str(initialization["pretrained_npz"]))
        source = source if source.is_absolute() else asset_root / source
        report = load_pretrained_npz(_core_model(adapter), source).to_dict()
        report["mode"] = mode
    elif mode != "scratch":
        raise ValueError(f"Unsupported initialization mode: {mode}")
    return adapter, report


def _validate_training_config(training: dict[str, Any], *, full_run: bool) -> None:
    batch_size = int(training["batch_size"])
    effective = int(training["effective_batch_size"])
    if batch_size <= 0 or effective <= 0 or effective % batch_size != 0:
        raise ValueError("effective_batch_size must be a positive multiple of batch_size")
    if full_run and str(training["batch_size_status"]) != "locked":
        raise ValueError("Full training requires a memory-profiled locked batch size")
    if int(training["epochs"]) <= 0 or int(training["patience"]) <= 0:
        raise ValueError("epochs and patience must be positive")
    if (
        float(training["learning_rate"]) <= 0
        or float(training["weight_decay"]) < 0
        or float(training["eta_min"]) < 0
    ):
        raise ValueError("Optimizer and scheduler parameters are invalid")
    if abs(
        float(training["cross_entropy_weight"])
        + float(training["dice_weight"])
        - 1.0
    ) > 1e-12:
        raise ValueError("Loss weights must sum to one")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train one locked segmentation model job.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/brats2023_2d.yaml"),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overfit-batches", type=int)
    parser.add_argument("--overfit-epochs", type=int, default=8)
    parser.add_argument("--max-epochs", type=int)
    args = parser.parse_args()

    model_config = _load_yaml(args.config)
    data_config = _load_yaml(args.data_config)
    full_run = args.overfit_batches is None
    training = model_config["training"]
    _validate_training_config(training, full_run=full_run)
    if args.overfit_batches is not None and args.overfit_batches <= 0:
        raise ValueError("overfit-batches must be positive")
    asset_root = _asset_root(data_config, args.asset_root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    model_name = str(model_config["name"])
    default_root = (
        Path("results/segmentation_overfit")
        if args.overfit_batches is not None
        else Path("results/segmentation_training")
    )
    output = (
        args.output
        if args.output is not None
        else default_root / model_name / f"seed_{args.seed}"
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    latest_path = output / "latest_training.pt"
    digest = _configuration_digest(model_config, data_config)

    train_split = data_config["splits"]["train"]
    val_split = data_config["splits"]["val"]
    train_dataset: Dataset = BraTS2DDataset(
        asset_root / str(train_split["image_dir"]),
        asset_root / str(train_split["mask_dir"]),
    )
    val_dataset: Dataset = BraTS2DDataset(
        asset_root / str(val_split["image_dir"]),
        asset_root / str(val_split["mask_dir"]),
    )
    batch_size = int(training["batch_size"])
    if args.overfit_batches is not None:
        sample_count = min(len(train_dataset), args.overfit_batches * batch_size)
        indices = list(range(sample_count))
        train_dataset = Subset(train_dataset, indices)
        val_dataset = Subset(train_dataset, list(range(len(train_dataset))))

    model, initialization_report = _build_model(
        model_config,
        seed=args.seed,
        asset_root=asset_root,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    configured_epochs = int(training["epochs"])
    epochs = (
        int(args.overfit_epochs)
        if args.overfit_batches is not None
        else configured_epochs
    )
    if args.max_epochs is not None:
        epochs = min(epochs, int(args.max_epochs))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=configured_epochs,
        eta_min=float(training["eta_min"]),
    )
    amp = bool(training["amp"])
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    progress = TrainingProgress()
    history: list[dict[str, Any]] = []
    if args.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {latest_path}")
        progress, metadata = load_training_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        if metadata.get("configuration_sha256") != digest:
            raise ValueError("Resume checkpoint configuration does not match this run")
        if metadata.get("model_name") != model_name or metadata.get("seed") != args.seed:
            raise ValueError("Resume checkpoint belongs to a different model job")
        history = list(metadata.get("history", []))
    elif latest_path.exists():
        raise FileExistsError(
            f"Output already contains a checkpoint; use --resume or a new output: {output}"
        )

    accumulation_steps = int(training["effective_batch_size"]) // batch_size
    num_workers = 0 if args.overfit_batches is not None else int(training["num_workers"])
    start_epoch = progress.completed_epoch + 1
    for epoch in range(start_epoch, epochs):
        train_loader = _epoch_loader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            seed=args.seed,
            epoch=epoch,
            num_workers=num_workers,
            device=device,
            drop_last=True,
        )
        val_loader = _epoch_loader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            seed=args.seed,
            epoch=0,
            num_workers=num_workers,
            device=device,
            drop_last=False,
        )
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device=device,
            accumulation_steps=accumulation_steps,
            amp=amp,
            max_batches=args.overfit_batches,
            cross_entropy_weight=float(training["cross_entropy_weight"]),
            dice_weight=float(training["dice_weight"]),
        )
        val_result = evaluate(
            model,
            val_loader,
            device=device,
            num_classes=int(model_config["num_classes"]),
            amp=amp,
            max_batches=args.overfit_batches,
            cross_entropy_weight=float(training["cross_entropy_weight"]),
            dice_weight=float(training["dice_weight"]),
        )
        scheduler.step()
        improved = val_result.loss < progress.best_val_loss
        progress = TrainingProgress(
            completed_epoch=epoch,
            global_step=progress.global_step + train_result.optimizer_steps,
            best_val_loss=(
                val_result.loss if improved else progress.best_val_loss
            ),
            epochs_without_improvement=(
                0 if improved else progress.epochs_without_improvement + 1
            ),
        )
        epoch_record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": train_result.to_dict(),
            "validation": val_result.to_dict(),
            "improved": improved,
        }
        history.append(epoch_record)
        metadata = {
            "configuration_sha256": digest,
            "model_name": model_name,
            "seed": args.seed,
            "history": history,
            "initialization": initialization_report,
            "overfit_batches": args.overfit_batches,
        }
        save_training_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            progress=progress,
            metadata=metadata,
        )
        if improved:
            save_training_checkpoint(
                output / "best_training.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                progress=progress,
                metadata=metadata,
            )
            _save_raw_state(output / "best_model.pth", _core_model(model))
        _write_json(output / "history.json", {"epochs": history})
        print(json.dumps(epoch_record, ensure_ascii=False, sort_keys=True), flush=True)
        if full_run and progress.epochs_without_improvement >= int(training["patience"]):
            break

    best_model_path = output / "best_model.pth"
    if not best_model_path.is_file():
        raise RuntimeError("Training did not produce a best model")
    _core_model(model).load_state_dict(
        torch.load(best_model_path, map_location=device),
        strict=True,
    )
    audit_loader = _epoch_loader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=args.seed,
        epoch=0,
        num_workers=num_workers,
        device=device,
        drop_last=False,
    )
    audit = evaluate(
        model,
        audit_loader,
        device=device,
        num_classes=int(model_config["num_classes"]),
        amp=amp,
        max_batches=args.overfit_batches,
        cross_entropy_weight=float(training["cross_entropy_weight"]),
        dice_weight=float(training["dice_weight"]),
    )
    audit_payload = audit.to_dict()
    predicted = np.asarray(audit_payload["predicted_pixels_by_class"])
    quality_passed = bool(
        np.all(predicted > 0) and float(audit_payload["mean_tumor_dice"]) >= 0.50
    )
    finite_losses = all(
        np.isfinite(record["train"]["loss"])
        and np.isfinite(record["validation"]["loss"])
        for record in history
    )
    overfit_loss_decreased = bool(
        len(history) >= 2
        and history[-1]["train"]["loss"] < history[0]["train"]["loss"]
    )
    summary = {
        "status": (
            "PASS"
            if finite_losses and (overfit_loss_decreased if not full_run else quality_passed)
            else "FAIL"
        ),
        "model_name": model_name,
        "seed": args.seed,
        "configuration_sha256": digest,
        "progress": {
            "completed_epoch": progress.completed_epoch,
            "global_step": progress.global_step,
            "best_val_loss": progress.best_val_loss,
            "epochs_without_improvement": progress.epochs_without_improvement,
        },
        "initialization": initialization_report,
        "batch_size": batch_size,
        "effective_batch_size": int(training["effective_batch_size"]),
        "accumulation_steps": accumulation_steps,
        "finite_losses": finite_losses,
        "overfit_mode": not full_run,
        "overfit_loss_decreased": overfit_loss_decreased,
        "quality_gate_applicable": full_run,
        "quality_gate_passed": quality_passed if full_run else None,
        "validation_audit": audit_payload,
        "output": str(output),
    }
    _write_json(output / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
