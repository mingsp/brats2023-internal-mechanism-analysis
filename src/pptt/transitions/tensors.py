from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


IntArray = npt.NDArray[np.integer]


def _validate_num_classes(num_classes: int) -> int:
    if isinstance(num_classes, bool) or not isinstance(num_classes, (int, np.integer)):
        raise ValueError(f"num_classes must be an integer, got {num_classes!r}")
    value = int(num_classes)
    if value < 2:
        raise ValueError(f"num_classes must be at least 2, got {value}")
    return value


def validate_label_arrays(
    *arrays: npt.ArrayLike,
    num_classes: int,
) -> tuple[np.ndarray, ...]:
    if not arrays:
        raise ValueError("At least one label array is required")
    c = _validate_num_classes(num_classes)
    converted = tuple(np.asarray(array) for array in arrays)
    expected_shape = converted[0].shape
    if converted[0].size == 0:
        raise ValueError("Label arrays must not be empty")
    for array in converted:
        if array.shape != expected_shape:
            raise ValueError(
                "All label arrays must have identical shape; "
                f"expected {expected_shape}, got {array.shape}"
            )
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"Label arrays must have integer dtype, got {array.dtype}")
        if np.any(array < 0) or np.any(array >= c):
            raise ValueError(f"Label values must be inside [0, {c}); found outside value")
    return tuple(array.astype(np.int64, copy=False) for array in converted)


def _validate_mask(mask: npt.ArrayLike | None, shape: tuple[int, ...]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    selected = np.asarray(mask)
    if selected.shape != shape:
        raise ValueError(
            f"mask must have the same shape as labels; expected {shape}, got {selected.shape}"
        )
    if selected.dtype != np.bool_:
        raise ValueError(f"mask must have boolean dtype, got {selected.dtype}")
    return selected


def build_transition_matrix(
    z0: npt.ArrayLike,
    z1: npt.ArrayLike,
    num_classes: int,
    *,
    mask: npt.ArrayLike | None = None,
) -> np.ndarray:
    c = _validate_num_classes(num_classes)
    state0, state1 = validate_label_arrays(z0, z1, num_classes=c)
    selected = _validate_mask(mask, state0.shape).reshape(-1)
    flat_index = state0.reshape(-1)[selected] * c + state1.reshape(-1)[selected]
    return np.bincount(flat_index, minlength=c * c).reshape(c, c).astype(
        np.int64,
        copy=False,
    )


def build_transition_tensor(
    y: npt.ArrayLike,
    z0: npt.ArrayLike,
    z1: npt.ArrayLike,
    num_classes: int,
    *,
    mask: npt.ArrayLike | None = None,
) -> np.ndarray:
    c = _validate_num_classes(num_classes)
    truth, state0, state1 = validate_label_arrays(y, z0, z1, num_classes=c)
    selected = _validate_mask(mask, truth.shape).reshape(-1)
    flat_index = (
        (truth.reshape(-1)[selected] * c + state0.reshape(-1)[selected]) * c
        + state1.reshape(-1)[selected]
    )
    return np.bincount(flat_index, minlength=c**3).reshape(c, c, c).astype(
        np.int64,
        copy=False,
    )


def _validate_transition_tensor(tensor: npt.ArrayLike) -> np.ndarray:
    counts = np.asarray(tensor)
    if counts.ndim != 3 or len(set(counts.shape)) != 1:
        raise ValueError(f"Expected a CxCxC transition tensor, got {counts.shape}")
    if counts.shape[0] < 2:
        raise ValueError("Transition tensor must contain at least two classes")
    if not np.issubdtype(counts.dtype, np.integer):
        raise ValueError(f"Transition tensor must have integer dtype, got {counts.dtype}")
    if np.any(counts < 0):
        raise ValueError("Transition tensor counts must be nonnegative")
    return counts.astype(np.int64, copy=False)


def confusions_from_transition(
    tensor: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    counts = _validate_transition_tensor(tensor)
    return counts.sum(axis=2), counts.sum(axis=1)


@dataclass(frozen=True)
class StateFlow:
    inflow: np.ndarray
    outflow: np.ndarray
    occupancy_delta: np.ndarray


def state_flows_from_transition(tensor: npt.ArrayLike) -> StateFlow:
    counts = _validate_transition_tensor(tensor)
    before, after = confusions_from_transition(counts)
    stable = np.diagonal(counts, axis1=1, axis2=2)
    return StateFlow(
        inflow=after - stable,
        outflow=before - stable,
        occupancy_delta=after - before,
    )
