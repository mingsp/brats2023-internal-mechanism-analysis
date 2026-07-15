from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


def _fixed_derangement(values: np.ndarray, seed: int) -> np.ndarray:
    if values.size < 2:
        raise ValueError("Patient permutation requires at least two patients")
    rng = np.random.default_rng(seed)
    for _ in range(256):
        candidate = rng.permutation(values)
        if np.all(candidate != values):
            return candidate
    return np.roll(values, 1)


def patient_permuted_targets(
    target_probabilities: npt.ArrayLike,
    patient_indices: npt.ArrayLike,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[int, int]]:
    targets = np.asarray(target_probabilities)
    patients = np.asarray(patient_indices)
    if targets.ndim < 2 or patients.shape != (targets.shape[0],):
        raise ValueError("targets and patient_indices must share their first dimension")
    if not np.issubdtype(patients.dtype, np.integer):
        raise ValueError("patient_indices must have integer dtype")
    unique = np.unique(patients).astype(np.int64)
    patient_counts = np.asarray(
        [np.count_nonzero(patients == patient) for patient in unique],
        dtype=np.int64,
    )
    if np.unique(patient_counts).size != 1:
        raise ValueError(
            "Patient permutation requires an equal number of cached rows per patient"
        )
    permuted = _fixed_derangement(unique, seed)
    mapping = {
        int(source): int(destination)
        for source, destination in zip(unique, permuted, strict=True)
    }
    result = np.empty_like(targets)
    for source, destination in mapping.items():
        source_rows = np.flatnonzero(patients == source)
        destination_rows = np.flatnonzero(patients == destination)
        result[source_rows] = targets[destination_rows]
    return result, mapping


def spatially_shifted_targets(
    target_probabilities: npt.ArrayLike,
    *,
    shift_y: int,
    shift_x: int,
) -> np.ndarray:
    targets = np.asarray(target_probabilities)
    if targets.ndim < 3:
        raise ValueError("Spatial targets must have at least three dimensions")
    if abs(int(shift_y)) < 40 or abs(int(shift_x)) < 40:
        raise ValueError("Both spatial shifts must have magnitude of at least 40 pixels")
    return np.roll(targets, shift=(int(shift_y), int(shift_x)), axis=(-2, -1))


def _transition_agreement_and_margin(
    probability_runs: npt.ArrayLike,
    *,
    class_axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probability_runs)
    if probabilities.ndim < 4:
        raise ValueError("Expected probabilities with axes R,K,C and one or more pixel axes")
    axis = class_axis if class_axis >= 0 else probabilities.ndim + class_axis
    if axis < 0 or axis >= probabilities.ndim:
        raise ValueError(f"Invalid class_axis {class_axis}")
    values = np.moveaxis(probabilities, axis, 2)
    if values.shape[0] < 2 or values.shape[1] < 2 or values.shape[2] < 2:
        raise ValueError("At least two restarts, stages, and classes are required")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.allclose(values.sum(axis=2), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("probabilities must sum to one along the class axis")
    states = values.argmax(axis=2)
    ordered = np.sort(values, axis=2)
    margins = ordered[:, :, -1] - ordered[:, :, -2]
    agreement = np.all(states[:, :-1] == states[:1, :-1], axis=0) & np.all(
        states[:, 1:] == states[:1, 1:],
        axis=0,
    )
    minimum_margin = np.minimum(margins[:, :-1], margins[:, 1:]).min(axis=0)
    return agreement, minimum_margin


def reliable_transition_mask(
    probability_runs: npt.ArrayLike,
    *,
    threshold: float,
    class_axis: int = 2,
) -> np.ndarray:
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    agreement, minimum_margin = _transition_agreement_and_margin(
        probability_runs,
        class_axis=class_axis,
    )
    return agreement & (minimum_margin >= float(threshold))


@dataclass(frozen=True)
class ReliabilitySelection:
    threshold: float
    consistency: float
    retention: float


@dataclass(frozen=True)
class ClassReliability:
    retention: np.ndarray
    available: np.ndarray


def class_reliability(
    reliable_mask: npt.ArrayLike,
    truth: npt.ArrayLike,
    *,
    num_classes: int,
    minimum_retention: float = 0.50,
) -> ClassReliability:
    reliable = np.asarray(reliable_mask)
    labels = np.asarray(truth)
    if reliable.ndim < 2 or reliable.shape[1:] != labels.shape:
        raise ValueError(
            "reliable_mask must have transition axis followed by the truth shape"
        )
    if reliable.dtype != np.bool_ or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("reliable_mask must be boolean and truth must be integer")
    if num_classes < 2 or np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("truth labels must be inside the declared class range")
    if not 0 <= minimum_retention <= 1:
        raise ValueError("minimum_retention must be inside [0, 1]")
    retention = np.full((reliable.shape[0], num_classes), np.nan, dtype=np.float64)
    for class_index in range(num_classes):
        class_pixels = labels == class_index
        if class_pixels.any():
            retention[:, class_index] = reliable[:, class_pixels].mean(axis=1)
    available = np.isfinite(retention) & (retention >= minimum_retention)
    return ClassReliability(retention=retention, available=available)


def select_reliability_threshold(
    probability_runs: npt.ArrayLike,
    *,
    grid: tuple[float, ...],
    minimum_consistency: float = 0.90,
    class_axis: int = 2,
) -> ReliabilitySelection:
    if not 0 < minimum_consistency <= 1:
        raise ValueError("minimum_consistency must be inside (0, 1]")
    thresholds = tuple(sorted(set(float(value) for value in grid)))
    if not thresholds or thresholds[0] < 0 or not np.isfinite(thresholds).all():
        raise ValueError("grid must contain finite nonnegative thresholds")
    agreement, minimum_margin = _transition_agreement_and_margin(
        probability_runs,
        class_axis=class_axis,
    )
    for threshold in thresholds:
        eligible = minimum_margin >= threshold
        retained = int(eligible.sum())
        if retained == 0:
            continue
        consistency = float(agreement[eligible].mean())
        if consistency >= minimum_consistency:
            return ReliabilitySelection(
                threshold=threshold,
                consistency=consistency,
                retention=float(eligible.mean()),
            )
    raise ValueError("No reliability threshold satisfies the consistency requirement")
