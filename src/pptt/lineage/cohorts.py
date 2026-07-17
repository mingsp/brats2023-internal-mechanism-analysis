from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ProcessSliceSelection:
    slice_index: int
    union_pixel_count: int
    transition_pixel_counts: tuple[int, ...]


def _validated_cohort_inputs(
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
        raise ValueError("states must contain at least two nodes followed by pixels")
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
    expected_reliable = (state_path.shape[0] - 1, *labels.shape)
    if reliability.shape != expected_reliable or reliability.dtype != np.bool_:
        raise ValueError(f"reliable must be boolean with shape {expected_reliable}")
    return (
        state_path.astype(np.int64, copy=False),
        labels.astype(np.int64, copy=False),
        reliability,
        final_state.astype(np.int64, copy=False),
    )


def _validated_truth_classes(truth_classes: Sequence[int]) -> tuple[int, ...]:
    raw = tuple(truth_classes)
    if not raw or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in raw
    ):
        raise ValueError("truth_classes must be a nonempty unique integer sequence")
    classes = tuple(int(value) for value in raw)
    if len(classes) != len(set(classes)):
        raise ValueError("truth_classes must be a nonempty unique integer sequence")
    return classes


def output_anchored_persistent_correction_cohorts(
    states: npt.ArrayLike,
    truth: npt.ArrayLike,
    reliable: npt.ArrayLike,
    final_model_state: npt.ArrayLike,
    *,
    truth_classes: Sequence[int],
) -> np.ndarray:
    """Locate reliable wrong-to-correct transitions retained through model output."""
    state_path, labels, reliability, final_state = _validated_cohort_inputs(
        states,
        truth,
        reliable,
        final_model_state,
    )
    classes = _validated_truth_classes(truth_classes)
    target_tissue = np.isin(labels, classes)
    native_output_correct = final_state == labels
    reliable_suffix = np.logical_and.accumulate(reliability[::-1], axis=0)[::-1]
    cohorts = np.zeros_like(reliability, dtype=bool)
    for transition_index in range(state_path.shape[0] - 1):
        correct_suffix = np.all(
            state_path[transition_index + 1 :] == labels[np.newaxis, ...],
            axis=0,
        )
        cohorts[transition_index] = (
            target_tissue
            & native_output_correct
            & reliable_suffix[transition_index]
            & (state_path[transition_index] != labels)
            & (state_path[transition_index + 1] == labels)
            & correct_suffix
        )
    if np.any(cohorts.sum(axis=0) > 1):
        raise RuntimeError("persistent correction cohorts must be pairwise disjoint")
    return cohorts


def output_anchored_stable_correct_cohort(
    states: npt.ArrayLike,
    truth: npt.ArrayLike,
    reliable: npt.ArrayLike,
    final_model_state: npt.ArrayLike,
    *,
    transition_index: int,
    truth_classes: Sequence[int],
    exclude: npt.ArrayLike | None = None,
) -> np.ndarray:
    """Return output-correct pixels that remain correct from one source node onward."""
    state_path, labels, reliability, final_state = _validated_cohort_inputs(
        states,
        truth,
        reliable,
        final_model_state,
    )
    if not 0 <= int(transition_index) < state_path.shape[0] - 1:
        raise ValueError("transition_index is outside the state path")
    classes = _validated_truth_classes(truth_classes)
    index = int(transition_index)
    stable = np.all(
        state_path[index:] == labels[np.newaxis, ...],
        axis=0,
    )
    suffix_reliable = np.all(reliability[index:], axis=0)
    result = (
        np.isin(labels, classes)
        & (final_state == labels)
        & suffix_reliable
        & stable
    )
    if exclude is not None:
        excluded = np.asarray(exclude)
        if excluded.shape != labels.shape or excluded.dtype != np.bool_:
            raise ValueError("exclude must be a boolean array matching truth")
        result &= ~excluded
    return result


def union_persistent_correction_cohorts(cohorts: npt.ArrayLike) -> np.ndarray:
    selected = np.asarray(cohorts)
    if selected.ndim < 2 or selected.dtype != np.bool_:
        raise ValueError("cohorts must be a boolean transition-by-pixel array")
    if np.any(selected.sum(axis=0) > 1):
        raise ValueError("persistent correction cohorts must be pairwise disjoint")
    return selected.any(axis=0)


def select_process_slice(cohorts: npt.ArrayLike) -> ProcessSliceSelection:
    selected = np.asarray(cohorts)
    if selected.ndim != 4 or selected.dtype != np.bool_:
        raise ValueError("cohorts must be boolean with shape transitions x slices x H x W")
    if selected.shape[0] < 1 or selected.shape[1] < 1:
        raise ValueError("cohorts must contain transitions and slices")
    union = union_persistent_correction_cohorts(selected)
    union_counts = union.sum(axis=(1, 2), dtype=np.int64)
    slice_index = int(np.argmax(union_counts))
    transition_counts = selected[:, slice_index].reshape(selected.shape[0], -1).sum(
        axis=1,
        dtype=np.int64,
    )
    return ProcessSliceSelection(
        slice_index=slice_index,
        union_pixel_count=int(union_counts[slice_index]),
        transition_pixel_counts=tuple(int(value) for value in transition_counts),
    )


__all__ = [
    "ProcessSliceSelection",
    "output_anchored_persistent_correction_cohorts",
    "output_anchored_stable_correct_cohort",
    "select_process_slice",
    "union_persistent_correction_cohorts",
]
