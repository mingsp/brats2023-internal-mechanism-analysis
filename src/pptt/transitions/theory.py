from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _positive_integer(value: int, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def endpoint_hidden_dimension(
    num_classes: int,
    *,
    truth_class_count: int | None = None,
) -> int:
    """Return the interior transport-polytope dimension across truth classes."""
    classes = _positive_integer(num_classes, name="num_classes", minimum=2)
    groups = (
        classes
        if truth_class_count is None
        else _positive_integer(
            truth_class_count,
            name="truth_class_count",
            minimum=1,
        )
    )
    return int(groups * (classes - 1) ** 2)


def additive_event_from_tensor(
    tensor: npt.ArrayLike,
    weights: npt.ArrayLike,
) -> float:
    """Evaluate a truth-conditioned additive adjacent-event functional."""
    counts = np.asarray(tensor)
    coefficients = np.asarray(weights, dtype=np.float64)
    if counts.ndim != 3 or len(set(counts.shape)) != 1:
        raise ValueError("tensor must be a cubic CxCxC array")
    if coefficients.shape != counts.shape:
        raise ValueError("tensor and weights must be aligned CxCxC arrays")
    if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise ValueError("tensor must contain nonnegative integer counts")
    if not np.isfinite(coefficients).all():
        raise ValueError("weights must be finite")
    return float(np.sum(counts.astype(np.float64, copy=False) * coefficients))


def binary_three_node_counterexample() -> tuple[np.ndarray, np.ndarray]:
    """Return pairwise-equivalent paths with different persistent corrections."""
    return (
        np.asarray(
            [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]],
            dtype=np.uint8,
        ),
        np.asarray(
            [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]],
            dtype=np.uint8,
        ),
    )


def adjacent_pair_counts(
    trajectories: npt.ArrayLike,
    *,
    num_classes: int,
) -> tuple[np.ndarray, ...]:
    """Count each adjacent state pair while discarding trajectory identity."""
    classes = _positive_integer(num_classes, name="num_classes", minimum=2)
    paths = np.asarray(trajectories)
    if paths.ndim != 2 or paths.shape[1] < 2:
        raise ValueError("trajectories must be a two-dimensional array with two nodes")
    if not np.issubdtype(paths.dtype, np.integer):
        raise ValueError("trajectory labels must be integers")
    if np.any(paths < 0) or np.any(paths >= classes):
        raise ValueError(f"trajectory labels must lie inside [0, {classes})")
    labels = paths.astype(np.int64, copy=False)
    return tuple(
        np.bincount(
            labels[:, index] * classes + labels[:, index + 1],
            minlength=classes * classes,
        )
        .reshape(classes, classes)
        .astype(np.int64, copy=False)
        for index in range(labels.shape[1] - 1)
    )


def persistent_correction_count(
    trajectories: npt.ArrayLike,
    *,
    truth_class: int,
) -> int:
    """Count paths that become correct after node one and remain correct."""
    if isinstance(truth_class, bool) or not isinstance(
        truth_class,
        (int, np.integer),
    ):
        raise ValueError("truth_class must be an integer")
    paths = np.asarray(trajectories)
    if paths.ndim != 2 or paths.shape[1] < 2:
        raise ValueError("trajectories must be a two-dimensional array with two nodes")
    if not np.issubdtype(paths.dtype, np.integer):
        raise ValueError("trajectory labels must be integers")
    truth = int(truth_class)
    return int(
        np.sum(
            (paths[:, 0] != truth)
            & np.all(paths[:, 1:] == truth, axis=1)
        )
    )


__all__ = [
    "additive_event_from_tensor",
    "adjacent_pair_counts",
    "binary_three_node_counterexample",
    "endpoint_hidden_dimension",
    "persistent_correction_count",
]
