from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from pptt.transitions.tensors import validate_label_arrays


class TransitionEvent(IntEnum):
    STABLE_CORRECT = 0
    CORRECTION = 1
    DESTRUCTION = 2
    STABLE_ERROR = 3
    WRONG_REENCODING = 4
    UNCERTAIN = 5


def states_and_margins(
    probabilities: npt.ArrayLike,
    *,
    class_axis: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(probabilities)
    if values.ndim < 2:
        raise ValueError(f"probabilities must have at least two dimensions, got {values.shape}")
    axis = int(class_axis)
    if axis < 0:
        axis += values.ndim
    if axis < 0 or axis >= values.ndim:
        raise ValueError(f"class_axis {class_axis} is invalid for shape {values.shape}")
    if values.shape[axis] < 2:
        raise ValueError("probabilities must contain at least two classes")
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f"probabilities must have floating dtype, got {values.dtype}")
    if not np.isfinite(values).all():
        raise ValueError("probabilities must be finite")
    if np.any(values < 0):
        raise ValueError("probabilities must be nonnegative")
    probability_sum = values.sum(axis=axis)
    if not np.allclose(probability_sum, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("probabilities must sum to one along class_axis")

    ordered = np.sort(values, axis=axis)
    top = np.take(ordered, -1, axis=axis)
    second = np.take(ordered, -2, axis=axis)
    states = np.argmax(values, axis=axis).astype(np.int64, copy=False)
    return states, top - second


def joint_reliability_mask(
    margin0: npt.ArrayLike,
    margin1: npt.ArrayLike,
    threshold: float,
) -> np.ndarray:
    first = np.asarray(margin0)
    second = np.asarray(margin1)
    if first.shape != second.shape:
        raise ValueError(
            f"margin arrays must have identical shape, got {first.shape} and {second.shape}"
        )
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(f"threshold must be finite and nonnegative, got {threshold}")
    return np.minimum(first, second) >= float(threshold)


def classify_transition_events(
    y: npt.ArrayLike,
    z0: npt.ArrayLike,
    z1: npt.ArrayLike,
    *,
    num_classes: int,
    reliable: npt.ArrayLike | None = None,
) -> np.ndarray:
    truth, state0, state1 = validate_label_arrays(
        y,
        z0,
        z1,
        num_classes=num_classes,
    )
    events = np.full(truth.shape, TransitionEvent.STABLE_ERROR, dtype=np.uint8)
    events[(state0 == truth) & (state1 == truth)] = TransitionEvent.STABLE_CORRECT
    events[(state0 != truth) & (state1 == truth)] = TransitionEvent.CORRECTION
    events[(state0 == truth) & (state1 != truth)] = TransitionEvent.DESTRUCTION
    events[(state0 != truth) & (state1 != truth) & (state0 != state1)] = (
        TransitionEvent.WRONG_REENCODING
    )
    if reliable is not None:
        mask = np.asarray(reliable)
        if mask.shape != truth.shape or mask.dtype != np.bool_:
            raise ValueError("reliable must be a boolean array with the label shape")
        events[~mask] = TransitionEvent.UNCERTAIN
    return events


@dataclass(frozen=True)
class PixelTransitionField:
    state_from: np.ndarray
    state_to: np.ndarray
    reliable: np.ndarray
    truth: np.ndarray | None = None

    @property
    def changed(self) -> np.ndarray:
        return self.state_from != self.state_to
