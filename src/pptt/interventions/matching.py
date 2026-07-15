from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PixelMatches:
    target_indices: np.ndarray
    control_indices: np.ndarray
    requested_target_count: int
    unmatched_target_count: int

    @property
    def matched_count(self) -> int:
        return int(self.target_indices.size)

    @property
    def matched_fraction(self) -> float:
        if self.requested_target_count == 0:
            return 0.0
        return self.matched_count / self.requested_target_count


def quantile_bins(
    values: np.ndarray,
    *,
    reference_mask: np.ndarray,
    bin_count: int = 4,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    selected = np.asarray(reference_mask)
    if array.shape != selected.shape or selected.dtype != np.bool_:
        raise ValueError("values and reference_mask must have the same shape")
    if bin_count < 2 or not selected.any():
        raise ValueError("quantile binning requires data and at least two bins")
    quantiles = np.linspace(0.0, 1.0, bin_count + 1)[1:-1]
    edges = np.unique(np.quantile(array[selected], quantiles))
    return np.digitize(array, edges, right=False).astype(np.int16)


def match_pixels_without_replacement(
    *,
    truth_class: np.ndarray,
    boundary_bin: np.ndarray,
    confidence_bin: np.ndarray,
    area_bin: np.ndarray,
    norm_bin: np.ndarray,
    target_mask: np.ndarray,
    control_mask: np.ndarray,
    boundary_value: np.ndarray | None = None,
    confidence_value: np.ndarray | None = None,
    area_value: np.ndarray | None = None,
    norm_value: np.ndarray | None = None,
    confidence_caliper: int = 1,
    norm_caliper: int = 1,
) -> PixelMatches:
    arrays = [
        np.asarray(value).reshape(-1)
        for value in (
            truth_class,
            boundary_bin,
            confidence_bin,
            area_bin,
            norm_bin,
            target_mask,
            control_mask,
        )
    ]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("All matching arrays must share one pixel axis")
    truth, boundary, confidence, area, norm, target, control = arrays
    if target.dtype != np.bool_ or control.dtype != np.bool_:
        raise ValueError("target_mask and control_mask must be boolean")
    if np.any(target & control):
        raise ValueError("target and control masks must be disjoint")
    if confidence_caliper < 0 or norm_caliper < 0:
        raise ValueError("calipers must be nonnegative")
    target_indices = np.flatnonzero(target)
    control_indices = np.flatnonzero(control)
    continuous = []
    for supplied, fallback in (
        (boundary_value, boundary),
        (confidence_value, confidence),
        (area_value, area),
        (norm_value, norm),
    ):
        values = np.asarray(fallback if supplied is None else supplied).reshape(-1)
        if values.shape != truth.shape or not np.isfinite(values).all():
            raise ValueError("continuous matching values must be finite and aligned")
        continuous.append(values.astype(np.float64, copy=False))
    boundary_continuous, confidence_continuous, area_continuous, norm_continuous = continuous
    scales = np.asarray(
        [
            np.std(values[control]) if control.any() else 0.0
            for values in continuous
        ],
        dtype=np.float64,
    )
    scales[scales <= 0] = 1.0
    buckets: dict[tuple[int, int, int, int, int], set[int]] = defaultdict(set)
    for index in control_indices:
        key = (
            int(truth[index]),
            int(boundary[index]),
            int(area[index]),
            int(confidence[index]),
            int(norm[index]),
        )
        buckets[key].add(int(index))

    def candidate_indices(index: int) -> list[int]:
        candidates: list[int] = []
        for confidence_delta in range(-confidence_caliper, confidence_caliper + 1):
            for norm_delta in range(-norm_caliper, norm_caliper + 1):
                key = (
                    int(truth[index]),
                    int(boundary[index]),
                    int(area[index]),
                    int(confidence[index] + confidence_delta),
                    int(norm[index] + norm_delta),
                )
                if buckets[key]:
                    candidates.extend(buckets[key])
        return candidates

    ordered_targets = sorted(
        (int(index) for index in target_indices),
        key=lambda index: (len(candidate_indices(index)), index),
    )
    matched_targets = []
    matched_controls = []
    for target_index in ordered_targets:
        candidates = candidate_indices(target_index)
        if not candidates:
            continue
        target_values = np.asarray(
            [
                boundary_continuous[target_index],
                confidence_continuous[target_index],
                area_continuous[target_index],
                norm_continuous[target_index],
            ]
        )
        candidate_values = np.column_stack(
            [
                values[candidates]
                for values in (
                    boundary_continuous,
                    confidence_continuous,
                    area_continuous,
                    norm_continuous,
                )
            ]
        )
        distances = np.square((candidate_values - target_values) / scales).sum(axis=1)
        selected_position = min(
            range(len(candidates)),
            key=lambda position: (distances[position], candidates[position]),
        )
        control_index = int(candidates[selected_position])
        key = (
            int(truth[control_index]),
            int(boundary[control_index]),
            int(area[control_index]),
            int(confidence[control_index]),
            int(norm[control_index]),
        )
        matched_targets.append(target_index)
        matched_controls.append(control_index)
        buckets[key].remove(control_index)
    order = np.argsort(matched_targets)
    target_result = np.asarray(matched_targets, dtype=np.int64)[order]
    control_result = np.asarray(matched_controls, dtype=np.int64)[order]
    return PixelMatches(
        target_indices=target_result,
        control_indices=control_result,
        requested_target_count=int(target_indices.size),
        unmatched_target_count=int(target_indices.size - target_result.size),
    )
