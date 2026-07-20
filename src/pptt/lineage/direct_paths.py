from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class DirectPathCohorts:
    correction: np.ndarray
    damage: np.ndarray

    def __post_init__(self) -> None:
        if (
            self.correction.shape != self.damage.shape
            or self.correction.dtype != np.bool_
            or self.damage.dtype != np.bool_
        ):
            raise ValueError("direct path cohorts must be aligned boolean arrays")
        if np.any(self.correction & self.damage):
            raise ValueError("correction and damage cohorts must be disjoint")


def _validated_truth_classes(
    truth_classes: Sequence[int],
) -> tuple[int, ...]:
    raw = tuple(truth_classes)
    if not raw or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in raw
    ):
        raise ValueError("truth_classes must be a nonempty integer sequence")
    classes = tuple(int(value) for value in raw)
    if len(classes) != len(set(classes)):
        raise ValueError("truth_classes must contain unique values")
    return classes


def _validated_inputs(
    states: npt.ArrayLike,
    truth: npt.ArrayLike,
    reliable: npt.ArrayLike,
    final_model_state: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state_path = np.asarray(states)
    labels = np.asarray(truth)
    reliability = np.asarray(reliable)
    final_state = np.asarray(final_model_state)
    if state_path.ndim < 2 or state_path.shape[0] < 2:
        raise ValueError("states must contain at least two ordered nodes")
    if not np.issubdtype(state_path.dtype, np.integer):
        raise ValueError("states must have integer dtype")
    if labels.shape != state_path.shape[1:] or not np.issubdtype(
        labels.dtype,
        np.integer,
    ):
        raise ValueError("truth must be an integer array matching one node state")
    if final_state.shape != labels.shape or not np.issubdtype(
        final_state.dtype,
        np.integer,
    ):
        raise ValueError("final_model_state must be an integer array matching truth")
    expected_reliability = (state_path.shape[0] - 1, *labels.shape)
    if reliability.shape != expected_reliability or reliability.dtype != np.bool_:
        raise ValueError(
            f"reliable must be boolean with shape {expected_reliability}"
        )
    return (
        state_path.astype(np.int64, copy=False),
        labels.astype(np.int64, copy=False),
        reliability,
        final_state.astype(np.int64, copy=False),
    )


def output_anchored_direct_path_cohorts(
    states: npt.ArrayLike,
    truth: npt.ArrayLike,
    reliable: npt.ArrayLike,
    final_model_state: npt.ArrayLike,
    *,
    source_index: int,
    receiver_index: int,
    truth_classes: Sequence[int],
) -> DirectPathCohorts:
    """Return persistent correction and damage between two ordered nodes."""
    state_path, labels, reliability, final_state = _validated_inputs(
        states,
        truth,
        reliable,
        final_model_state,
    )
    source = int(source_index)
    receiver = int(receiver_index)
    if not 0 <= source < receiver < state_path.shape[0]:
        raise ValueError(
            "node indices must satisfy 0 <= source_index < receiver_index < K"
        )
    classes = _validated_truth_classes(truth_classes)
    selected_tissue = np.isin(labels, classes)
    suffix_reliable = np.all(reliability[source:], axis=0)
    suffix_correct = np.all(
        state_path[receiver:] == labels[np.newaxis, ...],
        axis=0,
    )
    suffix_wrong = np.all(
        state_path[receiver:] != labels[np.newaxis, ...],
        axis=0,
    )
    correction = (
        selected_tissue
        & suffix_reliable
        & (state_path[source] != labels)
        & suffix_correct
        & (final_state == labels)
    )
    damage = (
        selected_tissue
        & suffix_reliable
        & (state_path[source] == labels)
        & suffix_wrong
        & (final_state != labels)
    )
    return DirectPathCohorts(
        correction=correction.astype(bool, copy=False),
        damage=damage.astype(bool, copy=False),
    )


def class_balanced_direct_net_recovery(
    cohorts: DirectPathCohorts,
    truth: npt.ArrayLike,
    *,
    truth_classes: Sequence[int],
) -> float:
    labels = np.asarray(truth)
    if labels.shape != cohorts.correction.shape or not np.issubdtype(
        labels.dtype,
        np.integer,
    ):
        raise ValueError("truth must be an integer array matching the cohorts")
    classes = _validated_truth_classes(truth_classes)
    rates = []
    for truth_class in classes:
        selected = labels == truth_class
        denominator = int(selected.sum())
        if denominator == 0:
            continue
        correction_count = int(np.count_nonzero(cohorts.correction & selected))
        damage_count = int(np.count_nonzero(cohorts.damage & selected))
        rates.append((correction_count - damage_count) / denominator)
    if not rates:
        raise ValueError("none of the requested truth classes is present")
    return float(np.mean(rates))


__all__ = [
    "DirectPathCohorts",
    "class_balanced_direct_net_recovery",
    "output_anchored_direct_path_cohorts",
]
