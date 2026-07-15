from __future__ import annotations

import numpy as np


def selective_path_effect(
    *,
    target_native: float,
    target_intervened: float,
    control_native: float,
    control_intervened: float,
) -> float:
    values = np.asarray(
        [target_native, target_intervened, control_native, control_intervened],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("Error rates must be finite")
    return float(
        (target_intervened - target_native)
        - (control_intervened - control_native)
    )


def masked_error_counts(
    prediction: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> tuple[int, int]:
    predicted = np.asarray(prediction)
    labels = np.asarray(truth)
    selected = np.asarray(mask)
    if predicted.shape != labels.shape or selected.shape != labels.shape:
        raise ValueError("prediction, truth, and mask must have the same shape")
    if selected.dtype != np.bool_:
        raise ValueError("mask must be boolean")
    return int(np.count_nonzero((predicted != labels) & selected)), int(selected.sum())


def dose_direction_consistent(
    alphas: np.ndarray,
    effects: np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> bool:
    dose = np.asarray(alphas, dtype=np.float64)
    values = np.asarray(effects, dtype=np.float64)
    if dose.ndim != 1 or values.shape != dose.shape or dose.size < 2:
        raise ValueError("dose and effect must be equal vectors with at least two values")
    order = np.argsort(dose)
    differences = np.diff(values[order])
    endpoint = values[order][-1] - values[order][0]
    if endpoint >= 0:
        return bool(np.all(differences >= -tolerance))
    return bool(np.all(differences <= tolerance))
