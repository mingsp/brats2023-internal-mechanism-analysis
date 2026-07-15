from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch
from torch import nn

from pptt.training.losses import segmentation_loss_parts
from pptt.transitions.metrics import metrics_from_confusion


@dataclass(frozen=True)
class TrainingProgress:
    completed_epoch: int = -1
    global_step: int = 0
    best_val_loss: float = float("inf")
    epochs_without_improvement: int = 0


@dataclass(frozen=True)
class EpochResult:
    loss: float
    cross_entropy: float
    dice_loss: float
    sample_count: int
    batch_count: int
    optimizer_steps: int
    confusion: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.confusion is not None:
            payload["confusion"] = self.confusion.tolist()
            metrics = metrics_from_confusion(self.confusion)
            payload["dice_by_class"] = metrics.dice.tolist()
            payload["mean_tumor_dice"] = float(metrics.dice[1:].mean())
            payload["predicted_pixels_by_class"] = self.confusion.sum(axis=0).tolist()
        return payload


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic))


def _autocast(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=True,
    )


def train_one_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    *,
    device: torch.device,
    accumulation_steps: int = 1,
    amp: bool = False,
    max_batches: int | None = None,
    cross_entropy_weight: float = 0.2,
    dice_weight: float = 0.8,
) -> EpochResult:
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    available_batches = len(loader)
    batch_limit = available_batches if max_batches is None else min(
        int(max_batches),
        available_batches,
    )
    if batch_limit <= 0:
        raise ValueError("The training loader must provide at least one batch")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = np.zeros(3, dtype=np.float64)
    sample_count = 0
    optimizer_steps = 0

    for batch_index, batch in enumerate(loader):
        if batch_index >= batch_limit:
            break
        image = batch["image"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        group_start = (batch_index // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, batch_limit - group_start)
        with _autocast(device, amp):
            logits = model(image)
            parts = segmentation_loss_parts(
                logits,
                target,
                cross_entropy_weight=cross_entropy_weight,
                dice_weight=dice_weight,
            )
            scaled_loss = parts.total / group_size
        scaler.scale(scaled_loss).backward()
        current_samples = int(image.shape[0])
        totals += np.asarray(
            [
                float(parts.total.detach().item()),
                float(parts.cross_entropy.detach().item()),
                float(parts.dice.detach().item()),
            ]
        ) * current_samples
        sample_count += current_samples
        end_of_group = (
            (batch_index - group_start + 1) == group_size
            or batch_index + 1 == batch_limit
        )
        if end_of_group:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
    return EpochResult(
        loss=float(totals[0] / sample_count),
        cross_entropy=float(totals[1] / sample_count),
        dice_loss=float(totals[2] / sample_count),
        sample_count=sample_count,
        batch_count=batch_limit,
        optimizer_steps=optimizer_steps,
    )


def evaluate(
    model: nn.Module,
    loader: Any,
    *,
    device: torch.device,
    num_classes: int,
    amp: bool = False,
    max_batches: int | None = None,
    cross_entropy_weight: float = 0.2,
    dice_weight: float = 0.8,
) -> EpochResult:
    available_batches = len(loader)
    batch_limit = available_batches if max_batches is None else min(
        int(max_batches),
        available_batches,
    )
    if batch_limit <= 0:
        raise ValueError("The evaluation loader must provide at least one batch")
    model.eval()
    totals = np.zeros(3, dtype=np.float64)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    sample_count = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= batch_limit:
                break
            image = batch["image"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            with _autocast(device, amp):
                logits = model(image)
                parts = segmentation_loss_parts(
                    logits,
                    target,
                    cross_entropy_weight=cross_entropy_weight,
                    dice_weight=dice_weight,
                )
            prediction = logits.argmax(dim=1)
            flat_index = (
                target.reshape(-1).to(torch.int64) * num_classes
                + prediction.reshape(-1).to(torch.int64)
            )
            counts = torch.bincount(flat_index, minlength=num_classes**2)
            confusion += counts.reshape(num_classes, num_classes).cpu().numpy()
            current_samples = int(image.shape[0])
            totals += np.asarray(
                [
                    float(parts.total.item()),
                    float(parts.cross_entropy.item()),
                    float(parts.dice.item()),
                ]
            ) * current_samples
            sample_count += current_samples
    return EpochResult(
        loss=float(totals[0] / sample_count),
        cross_entropy=float(totals[1] / sample_count),
        dice_loss=float(totals[2] / sample_count),
        sample_count=sample_count,
        batch_count=batch_limit,
        optimizer_steps=0,
        confusion=confusion,
    )


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    progress: TrainingProgress,
    metadata: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "progress": asdict(progress),
        "rng_state": _rng_state(),
        "metadata": {} if metadata is None else metadata,
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> tuple[TrainingProgress, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("Unsupported or invalid training checkpoint")
    required = {
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "progress",
        "rng_state",
        "metadata",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Training checkpoint is missing fields: {missing}")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        _restore_rng_state(payload["rng_state"])
    return TrainingProgress(**payload["progress"]), dict(payload["metadata"])
