from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], byteorder="big", signed=False)


def select_evenly_spaced_patients(
    patient_ids: Iterable[str],
    *,
    count: int,
    seed: int,
) -> tuple[str, ...]:
    patients = sorted(set(patient_ids))
    if count <= 0 or len(patients) < count:
        raise ValueError(
            f"count must be positive and no greater than {len(patients)}, got {count}"
        )
    step = len(patients) / count
    offset = np.random.default_rng(seed).uniform(0.0, step)
    positions = np.floor(offset + np.arange(count) * step).astype(np.int64)
    if np.unique(positions).size != count or positions[-1] >= len(patients):
        raise AssertionError("Evenly spaced patient selection produced invalid positions")
    return tuple(patients[int(position)] for position in positions)


def sample_pixel_indices(
    probabilities: torch.Tensor,
    *,
    max_per_class: int,
    seed: int,
) -> torch.Tensor:
    if probabilities.ndim != 3 or probabilities.shape[0] < 2:
        raise ValueError("probabilities must be a CHW tensor")
    if max_per_class <= 0:
        raise ValueError("max_per_class must be positive")
    labels = probabilities.argmax(dim=0).reshape(-1).cpu().numpy()
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for class_index in range(probabilities.shape[0]):
        candidates = np.flatnonzero(labels == class_index)
        if candidates.size:
            selected.append(
                rng.choice(
                    candidates,
                    size=min(max_per_class, candidates.size),
                    replace=False,
                )
            )
    if not selected:
        raise ValueError("No pixels were available for sampling")
    return torch.from_numpy(np.concatenate(selected).astype(np.int64))


def sample_feature_vectors(
    feature: torch.Tensor,
    flat_indices: torch.Tensor,
    *,
    output_size: tuple[int, int],
) -> torch.Tensor:
    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError(f"feature must have shape 1xCxHxW, got {tuple(feature.shape)}")
    if flat_indices.ndim != 1 or flat_indices.dtype != torch.int64:
        raise ValueError("flat_indices must be a one-dimensional int64 tensor")
    output_height, output_width = (int(value) for value in output_size)
    if output_height <= 0 or output_width <= 0:
        raise ValueError("output_size must be positive")
    indices = flat_indices.to(device=feature.device)
    if torch.any(indices < 0) or torch.any(indices >= output_height * output_width):
        raise ValueError("flat_indices contain an out-of-range output pixel")
    y = torch.div(indices, output_width, rounding_mode="floor")
    x = indices % output_width
    grid_x = 2.0 * (x.to(feature.dtype) + 0.5) / output_width - 1.0
    grid_y = 2.0 * (y.to(feature.dtype) + 0.5) / output_height - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        feature,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled[0, :, :, 0].transpose(0, 1).contiguous()


@dataclass(frozen=True)
class BootstrapDifference:
    mean_difference: float
    ci_low: float
    ci_high: float
    iterations: int


def paired_bootstrap_difference(
    real_values: np.ndarray,
    control_values: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> BootstrapDifference:
    real = np.asarray(real_values, dtype=np.float64)
    control = np.asarray(control_values, dtype=np.float64)
    if real.ndim != 1 or control.shape != real.shape or real.size < 2:
        raise ValueError("paired bootstrap requires equal one-dimensional arrays of size >= 2")
    if not np.isfinite(real).all() or not np.isfinite(control).all():
        raise ValueError("paired bootstrap values must be finite")
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    differences = control - real
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, differences.size, size=(iterations, differences.size))
    means = differences[samples].mean(axis=1)
    ci_low, ci_high = np.quantile(means, [0.025, 0.975])
    return BootstrapDifference(
        mean_difference=float(differences.mean()),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        iterations=int(iterations),
    )


@dataclass(frozen=True)
class PatientRows:
    patient_id: str
    features: dict[str, np.ndarray]
    target_probabilities: np.ndarray
    spatial_target_probabilities: np.ndarray
    truth: np.ndarray


@dataclass(frozen=True)
class BalancedRows:
    patient_ids: tuple[str, ...]
    features: dict[str, np.ndarray]
    target_probabilities: np.ndarray
    spatial_target_probabilities: np.ndarray
    truth: np.ndarray
    patient_indices: np.ndarray
    rows_per_patient: int


def save_patient_rows(rows: PatientRows, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    node_names = tuple(rows.features)
    if not node_names:
        raise ValueError("At least one node feature is required")
    payload: dict[str, np.ndarray] = {
        "patient_id": np.asarray(rows.patient_id),
        "node_names": np.asarray(node_names),
        "target_probabilities": np.asarray(
            rows.target_probabilities,
            dtype=np.float32,
        ),
        "spatial_target_probabilities": np.asarray(
            rows.spatial_target_probabilities,
            dtype=np.float32,
        ),
        "truth": np.asarray(rows.truth, dtype=np.int64),
    }
    payload.update(
        {
            f"feature_{index}": np.asarray(rows.features[node], dtype=np.float16)
            for index, node in enumerate(node_names)
        }
    )
    np.savez_compressed(destination, **payload)


def load_patient_rows(path: str | Path) -> PatientRows:
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        patient_id = str(archive["patient_id"].item())
        node_names = tuple(str(value) for value in archive["node_names"].tolist())
        features = {
            node: archive[f"feature_{index}"].astype(np.float16, copy=False)
            for index, node in enumerate(node_names)
        }
        return PatientRows(
            patient_id=patient_id,
            features=features,
            target_probabilities=archive["target_probabilities"].astype(
                np.float32,
                copy=False,
            ),
            spatial_target_probabilities=archive[
                "spatial_target_probabilities"
            ].astype(np.float32, copy=False),
            truth=archive["truth"].astype(np.int64, copy=False),
        )


def balance_patient_rows(
    rows: list[PatientRows],
    *,
    seed: int,
) -> BalancedRows:
    if len(rows) < 2:
        raise ValueError("At least two patients are required for balancing")
    ordered = sorted(rows, key=lambda item: item.patient_id)
    patient_ids = tuple(item.patient_id for item in ordered)
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("Patient rows must have unique patient identifiers")
    node_names = tuple(ordered[0].features)
    if not node_names:
        raise ValueError("At least one node feature is required")
    row_counts = [item.target_probabilities.shape[0] for item in ordered]
    rows_per_patient = min(row_counts)
    if rows_per_patient <= 0:
        raise ValueError("Every patient must provide at least one sampled pixel")
    selections: list[np.ndarray] = []
    for item in ordered:
        count = item.target_probabilities.shape[0]
        if (
            item.spatial_target_probabilities.shape != item.target_probabilities.shape
            or item.truth.shape != (count,)
            or tuple(item.features) != node_names
            or any(feature.shape[0] != count for feature in item.features.values())
        ):
            raise ValueError(f"Misaligned cached rows for patient {item.patient_id}")
        rng = np.random.default_rng(stable_seed(seed, item.patient_id))
        selections.append(rng.permutation(count)[:rows_per_patient])
    features = {
        node: np.concatenate(
            [item.features[node][selection] for item, selection in zip(ordered, selections, strict=True)]
        )
        for node in node_names
    }
    targets = np.concatenate(
        [item.target_probabilities[selection] for item, selection in zip(ordered, selections, strict=True)]
    )
    spatial_targets = np.concatenate(
        [
            item.spatial_target_probabilities[selection]
            for item, selection in zip(ordered, selections, strict=True)
        ]
    )
    truth = np.concatenate(
        [item.truth[selection] for item, selection in zip(ordered, selections, strict=True)]
    )
    patient_indices = np.repeat(
        np.arange(len(ordered), dtype=np.int32),
        rows_per_patient,
    )
    return BalancedRows(
        patient_ids=patient_ids,
        features=features,
        target_probabilities=targets,
        spatial_target_probabilities=spatial_targets,
        truth=truth,
        patient_indices=patient_indices,
        rows_per_patient=rows_per_patient,
    )
