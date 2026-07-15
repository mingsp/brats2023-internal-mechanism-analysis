from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


def _validate_states(states: npt.ArrayLike) -> np.ndarray:
    values = np.asarray(states)
    if values.ndim < 2 or values.shape[0] < 2:
        raise ValueError(
            "states must contain at least two stages followed by one or more pixel dimensions"
        )
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"states must have integer dtype, got {values.dtype}")
    return values.astype(np.int64, copy=False)


def _suffix_stability_depth(matches: np.ndarray) -> np.ndarray:
    suffix = np.logical_and.accumulate(matches[::-1], axis=0)[::-1]
    return np.argmax(suffix, axis=0).astype(np.int64) + 1


def final_state_depth(states: npt.ArrayLike) -> np.ndarray:
    values = _validate_states(states)
    final = values[-1]
    return _suffix_stability_depth(values == final[np.newaxis, ...])


def _validate_truth(states: np.ndarray, truth: npt.ArrayLike) -> np.ndarray:
    labels = np.asarray(truth)
    if labels.shape != states.shape[1:]:
        raise ValueError(
            "truth must have the same spatial shape as one state; "
            f"expected {states.shape[1:]}, got {labels.shape}"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"truth must have integer dtype, got {labels.dtype}")
    return labels.astype(np.int64, copy=False)


def correct_state_depth(states: npt.ArrayLike, truth: npt.ArrayLike) -> np.ndarray:
    values = _validate_states(states)
    labels = _validate_truth(values, truth)
    candidate = _suffix_stability_depth(values == labels[np.newaxis, ...])
    return np.where(values[-1] == labels, candidate, -1)


def terminal_error_origin_depth(
    states: npt.ArrayLike,
    truth: npt.ArrayLike,
) -> np.ndarray:
    values = _validate_states(states)
    labels = _validate_truth(values, truth)
    depth = final_state_depth(values)
    return np.where(values[-1] != labels, depth, -1)


@dataclass(frozen=True)
class LineageDepths:
    final_state: np.ndarray
    correct_state: np.ndarray
    terminal_error_origin: np.ndarray


def lineage_depths(states: npt.ArrayLike, truth: npt.ArrayLike) -> LineageDepths:
    values = _validate_states(states)
    labels = _validate_truth(values, truth)
    final_depth = final_state_depth(values)
    return LineageDepths(
        final_state=final_depth,
        correct_state=correct_state_depth(values, labels),
        terminal_error_origin=np.where(values[-1] != labels, final_depth, -1),
    )
