from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as torch_functional



@dataclass(frozen=True)
class FeatureMatches:
    target_indices: np.ndarray
    control_indices: np.ndarray
    requested_target_count: int

    @property
    def matched_count(self) -> int:
        return int(self.target_indices.size)


def match_feature_controls(
    *,
    target_mask: np.ndarray,
    control_mask: np.ndarray,
    truth_class: np.ndarray,
    boundary_distance: np.ndarray,
    activation_norm: np.ndarray,
    boundary_stratum: np.ndarray | None = None,
    activation_stratum: np.ndarray | None = None,
) -> FeatureMatches:
    arrays = [
        np.asarray(value)
        for value in (
            target_mask,
            control_mask,
            truth_class,
            boundary_distance,
            activation_norm,
        )
    ]
    if len({array.shape for array in arrays}) != 1 or arrays[0].ndim != 2:
        raise ValueError("feature matching inputs must be aligned two-dimensional arrays")
    target, control, classes, boundary, norm = arrays
    if target.dtype != np.bool_ or control.dtype != np.bool_:
        raise ValueError("target_mask and control_mask must be boolean")
    if np.any(target & control):
        raise ValueError("target and control masks must be disjoint")
    if not np.isfinite(boundary).all() or not np.isfinite(norm).all():
        raise ValueError("matching values must be finite")
    if (boundary_stratum is None) != (activation_stratum is None):
        raise ValueError("both registered matching strata must be provided together")
    boundary_groups = None
    activation_groups = None
    if boundary_stratum is not None and activation_stratum is not None:
        boundary_groups = np.asarray(boundary_stratum)
        activation_groups = np.asarray(activation_stratum)
        if (
            boundary_groups.shape != target.shape
            or activation_groups.shape != target.shape
            or not np.issubdtype(boundary_groups.dtype, np.integer)
            or not np.issubdtype(activation_groups.dtype, np.integer)
        ):
            raise ValueError("registered matching strata must be aligned integers")

    target_indices = np.flatnonzero(target.reshape(-1))
    available = set(int(value) for value in np.flatnonzero(control.reshape(-1)))
    flat_classes = classes.reshape(-1)
    flat_boundary = boundary.astype(np.float64, copy=False).reshape(-1)
    flat_norm = norm.astype(np.float64, copy=False).reshape(-1)
    flat_boundary_groups = (
        None if boundary_groups is None else boundary_groups.reshape(-1)
    )
    flat_activation_groups = (
        None if activation_groups is None else activation_groups.reshape(-1)
    )
    control_indices = np.asarray(sorted(available), dtype=np.int64)
    scales = np.asarray(
        [
            np.std(flat_boundary[control_indices]) if control_indices.size else 0.0,
            np.std(flat_norm[control_indices]) if control_indices.size else 0.0,
        ]
    )
    scales[scales <= 0] = 1.0

    def candidates(index: int) -> list[int]:
        return sorted(
            value
            for value in available
            if int(flat_classes[value]) == int(flat_classes[index])
            and (
                flat_boundary_groups is None
                or int(flat_boundary_groups[value])
                == int(flat_boundary_groups[index])
            )
            and (
                flat_activation_groups is None
                or int(flat_activation_groups[value])
                == int(flat_activation_groups[index])
            )
        )

    ordered_targets = sorted(
        (int(value) for value in target_indices),
        key=lambda index: (len(candidates(index)), index),
    )
    matched_targets = []
    matched_controls = []
    for target_index in ordered_targets:
        current = candidates(target_index)
        if not current:
            continue
        target_values = np.asarray(
            [flat_boundary[target_index], flat_norm[target_index]]
        )
        candidate_values = np.column_stack(
            [flat_boundary[current], flat_norm[current]]
        )
        distances = np.square((candidate_values - target_values) / scales).sum(axis=1)
        chosen_position = min(
            range(len(current)),
            key=lambda position: (float(distances[position]), current[position]),
        )
        chosen = int(current[chosen_position])
        matched_targets.append(target_index)
        matched_controls.append(chosen)
        available.remove(chosen)
    order = np.argsort(matched_targets)
    return FeatureMatches(
        target_indices=np.asarray(matched_targets, dtype=np.int64)[order],
        control_indices=np.asarray(matched_controls, dtype=np.int64)[order],
        requested_target_count=int(target_indices.size),
    )


def persistent_correction_mask(
    states: np.ndarray,
    truth: np.ndarray,
    *,
    transition_index: int,
    reliable: np.ndarray,
    truth_classes: Sequence[int],
) -> np.ndarray:
    state_path = np.asarray(states)
    labels = np.asarray(truth)
    reliability = np.asarray(reliable)
    if state_path.ndim < 2 or state_path.shape[1:] != labels.shape:
        raise ValueError("states must have shape K followed by the truth shape")
    if reliability.shape != (state_path.shape[0] - 1, *labels.shape):
        raise ValueError("reliable must align with every adjacent transition")
    if reliability.dtype != np.bool_:
        raise ValueError("reliable must be boolean")
    if not 0 <= transition_index < state_path.shape[0] - 1:
        raise ValueError("transition_index is outside the state path")
    classes = tuple(int(value) for value in truth_classes)
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("truth_classes must be a nonempty unique sequence")
    suffix_correct = np.all(
        state_path[transition_index + 1 :] == labels[np.newaxis, ...],
        axis=0,
    )
    return (
        np.isin(labels, classes)
        & reliability[transition_index]
        & (state_path[transition_index] != labels)
        & (state_path[transition_index + 1] == labels)
        & suffix_correct
    )


def select_target_slice(target_mask: np.ndarray) -> int:
    mask = np.asarray(target_mask)
    if mask.ndim != 3 or mask.dtype != np.bool_:
        raise ValueError("target_mask must be boolean with shape SxHxW")
    counts = mask.sum(axis=(1, 2))
    if counts.max(initial=0) == 0:
        raise ValueError("target_mask contains no target pixels")
    return int(np.argmax(counts))


def adaptive_feature_mask(
    mask: np.ndarray,
    *,
    output_shape: tuple[int, int],
) -> np.ndarray:
    selected = np.asarray(mask)
    if selected.ndim != 2 or selected.dtype != np.bool_:
        raise ValueError("mask must be a two-dimensional boolean array")
    if len(output_shape) != 2 or any(int(value) <= 0 for value in output_shape):
        raise ValueError("output_shape must contain two positive dimensions")
    tensor = torch.from_numpy(selected.astype(np.float32))[None, None]
    pooled = torch_functional.adaptive_max_pool2d(tensor, output_shape)
    return pooled[0, 0].numpy() > 0


def adaptive_feature_mean(
    values: np.ndarray,
    *,
    output_shape: tuple[int, int],
) -> np.ndarray:
    selected = np.asarray(values)
    if selected.ndim != 2 or not np.issubdtype(selected.dtype, np.number):
        raise ValueError("values must be a two-dimensional numeric array")
    if not np.isfinite(selected).all():
        raise ValueError("values must be finite")
    if len(output_shape) != 2 or any(int(value) <= 0 for value in output_shape):
        raise ValueError("output_shape must contain two positive dimensions")
    tensor = torch.from_numpy(selected.astype(np.float32, copy=False))[None, None]
    pooled = torch_functional.adaptive_avg_pool2d(tensor, output_shape)
    return pooled[0, 0].numpy()


def dominant_feature_truth_class(
    truth: np.ndarray,
    *,
    output_shape: tuple[int, int],
    truth_classes: Sequence[int],
) -> np.ndarray:
    labels = np.asarray(truth)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("truth must be a two-dimensional integer array")
    classes = tuple(int(value) for value in truth_classes)
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("truth_classes must be a nonempty unique sequence")
    occupancy = torch.cat(
        [
            torch_functional.adaptive_avg_pool2d(
                torch.from_numpy((labels == class_index).astype(np.float32))[None, None],
                output_shape,
            )
            for class_index in classes
        ],
        dim=1,
    )[0]
    dominant = occupancy.argmax(dim=0).numpy()
    has_tumor = occupancy.max(dim=0).values.numpy() > 0
    result = np.zeros(output_shape, dtype=np.uint8)
    class_values = np.asarray(classes, dtype=np.uint8)
    result[has_tumor] = class_values[dominant[has_tumor]]
    return result


def registered_feature_strata(
    boundary_distance: np.ndarray,
    activation_norm: np.ndarray,
    *,
    eligible_mask: np.ndarray,
    boundary_bin_edges: Sequence[float],
    activation_norm_quantile_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boundary = np.asarray(boundary_distance, dtype=np.float64)
    norm = np.asarray(activation_norm, dtype=np.float64)
    eligible = np.asarray(eligible_mask)
    if (
        boundary.shape != norm.shape
        or boundary.shape != eligible.shape
        or boundary.ndim != 2
        or eligible.dtype != np.bool_
    ):
        raise ValueError("registered strata inputs must be aligned two-dimensional arrays")
    if not np.isfinite(boundary).all() or not np.isfinite(norm).all():
        raise ValueError("registered strata values must be finite")
    edges = np.asarray(tuple(float(value) for value in boundary_bin_edges))
    if edges.ndim != 1 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("boundary_bin_edges must be finite and strictly increasing")
    bin_count = int(activation_norm_quantile_bins)
    if bin_count < 1:
        raise ValueError("activation_norm_quantile_bins must be positive")
    eligible_values = norm[eligible]
    if eligible_values.size == 0:
        raise ValueError("eligible_mask must contain at least one feature position")
    quantiles = np.linspace(0.0, 1.0, bin_count + 1)[1:-1]
    norm_edges = (
        np.unique(np.quantile(eligible_values, quantiles))
        if quantiles.size
        else np.empty(0, dtype=np.float64)
    )
    return (
        np.digitize(boundary, edges, right=False).astype(np.int16),
        np.digitize(norm, norm_edges, right=False).astype(np.int16),
        norm_edges.astype(np.float64, copy=False),
    )


def stable_correct_control_mask(
    states: np.ndarray,
    truth: np.ndarray,
    *,
    transition_index: int,
    reliable: np.ndarray,
    truth_classes: Sequence[int],
) -> np.ndarray:
    state_path = np.asarray(states)
    labels = np.asarray(truth)
    reliability = np.asarray(reliable)
    if state_path.ndim < 2 or state_path.shape[1:] != labels.shape:
        raise ValueError("states must have shape K followed by the truth shape")
    if reliability.shape != (state_path.shape[0] - 1, *labels.shape):
        raise ValueError("reliable must align with every adjacent transition")
    if reliability.dtype != np.bool_:
        raise ValueError("reliable must be boolean")
    if not 0 <= transition_index < state_path.shape[0] - 1:
        raise ValueError("transition_index is outside the state path")
    classes = tuple(int(value) for value in truth_classes)
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("truth_classes must be a nonempty unique sequence")
    stable = np.all(
        state_path[transition_index:] == labels[np.newaxis, ...],
        axis=0,
    )
    return np.isin(labels, classes) & reliability[transition_index] & stable


def spatial_corrupt_restore(
    activation: torch.Tensor,
    *,
    restore_mask: torch.Tensor,
    shift_yx: tuple[int, int],
    alpha: float,
) -> torch.Tensor:
    if activation.ndim != 4:
        raise ValueError("activation must have shape BxCxHxW")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must lie inside [0, 1]")
    mask = restore_mask.to(device=activation.device, dtype=torch.bool)
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None]
    if mask.ndim != 4 or mask.shape[-2:] != activation.shape[-2:]:
        raise ValueError("restore_mask must align with activation spatial dimensions")
    if mask.shape[0] not in (1, activation.shape[0]):
        raise ValueError("restore_mask batch dimension does not broadcast")
    shift_y, shift_x = (int(value) for value in shift_yx)
    corrupted = torch.roll(
        activation,
        shifts=(shift_y, shift_x),
        dims=(-2, -1),
    )
    restored = corrupted + float(alpha) * (activation - corrupted)
    return torch.where(mask, restored, corrupted)


def persistent_retention(
    states: np.ndarray,
    truth: np.ndarray,
    target_mask: np.ndarray,
    *,
    transition_index: int,
) -> float:
    state_path = np.asarray(states)
    labels = np.asarray(truth)
    target = np.asarray(target_mask)
    if state_path.ndim < 2 or state_path.shape[1:] != labels.shape:
        raise ValueError("states must align with truth")
    if target.shape != labels.shape or target.dtype != np.bool_:
        raise ValueError("target_mask must be boolean and align with truth")
    if not 0 <= transition_index < state_path.shape[0] - 1:
        raise ValueError("transition_index is outside the state path")
    if not target.any():
        return float("nan")
    persistent = np.all(
        state_path[transition_index + 1 :] == labels[np.newaxis, ...],
        axis=0,
    )
    return float(persistent[target].mean())


def proportion_mediated(
    *,
    clean: float,
    corrupt: float,
    restored: float,
) -> float:
    values = np.asarray([clean, corrupt, restored], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("retention values must be finite")
    total_effect = float(clean - corrupt)
    if total_effect <= 0:
        return float("nan")
    return float((restored - corrupt) / total_effect)


__all__ = [
    "FeatureMatches",
    "adaptive_feature_mask",
    "adaptive_feature_mean",
    "dominant_feature_truth_class",
    "match_feature_controls",
    "persistent_correction_mask",
    "persistent_retention",
    "proportion_mediated",
    "registered_feature_strata",
    "select_target_slice",
    "spatial_corrupt_restore",
    "stable_correct_control_mask",
]
