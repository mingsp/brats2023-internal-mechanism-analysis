from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapDifference:
    estimate: float
    ci_low: float
    ci_high: float
    n_pairs: int


@dataclass(frozen=True)
class CaseSelection:
    representative_patient_id: str
    boundary_patient_id: str
    representative_distance: float
    lesion_area_low: float
    lesion_area_high: float
    eligible_patient_count: int


def paired_bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> BootstrapDifference:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.ndim != 1 or second.shape != first.shape or first.size < 2:
        raise ValueError("paired bootstrap requires equal vectors with at least two pairs")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("paired bootstrap values must be finite")
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")
    differences = first - second
    rng = np.random.default_rng(seed)
    positions = rng.integers(
        0,
        differences.size,
        size=(n_resamples, differences.size),
    )
    bootstrap = differences[positions].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
    return BootstrapDifference(
        estimate=float(differences.mean()),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_pairs=int(differences.size),
    )


def select_representative_case(
    frame: pd.DataFrame,
    *,
    vector_columns: tuple[str, ...],
    area_quantiles: tuple[float, float] = (0.25, 0.75),
) -> CaseSelection:
    required = {
        "patient_id",
        "lesion_area",
        "reliable_fraction",
        "class_1_present",
        "class_2_present",
        "class_3_present",
        *vector_columns,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"case selection is missing columns: {sorted(missing)}")
    if frame.patient_id.duplicated().any() or len(frame) < 2:
        raise ValueError("case selection requires unique rows for at least two patients")
    low_quantile, high_quantile = area_quantiles
    if not 0 <= low_quantile < high_quantile <= 1:
        raise ValueError("area_quantiles must be ordered inside [0, 1]")
    numeric = frame[["lesion_area", "reliable_fraction", *vector_columns]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(numeric).all():
        raise ValueError("case selection values must be finite")
    area_low, area_high = np.quantile(
        frame.lesion_area.to_numpy(dtype=np.float64),
        [low_quantile, high_quantile],
    )
    all_classes = frame[
        ["class_1_present", "class_2_present", "class_3_present"]
    ].astype(bool).all(axis=1)
    eligible = frame[
        all_classes
        & frame.lesion_area.between(area_low, area_high, inclusive="both")
    ].copy()
    if eligible.empty:
        raise ValueError("no patient satisfies the preregistered representative-case rule")
    cohort_vectors = frame[list(vector_columns)].to_numpy(dtype=np.float64)
    median = np.median(cohort_vectors, axis=0)
    q25, q75 = np.quantile(cohort_vectors, [0.25, 0.75], axis=0)
    scale = q75 - q25
    fallback = cohort_vectors.std(axis=0, ddof=0)
    scale = np.where(scale > 0, scale, fallback)
    scale = np.where(scale > 0, scale, 1.0)
    eligible_vectors = eligible[list(vector_columns)].to_numpy(dtype=np.float64)
    distances = np.linalg.norm((eligible_vectors - median) / scale, axis=1)
    eligible["_distance"] = distances
    representative = eligible.sort_values(
        ["_distance", "patient_id"],
        kind="mergesort",
    ).iloc[0]
    boundary = frame.sort_values(
        ["reliable_fraction", "patient_id"],
        kind="mergesort",
    ).iloc[0]
    return CaseSelection(
        representative_patient_id=str(representative.patient_id),
        boundary_patient_id=str(boundary.patient_id),
        representative_distance=float(representative._distance),
        lesion_area_low=float(area_low),
        lesion_area_high=float(area_high),
        eligible_patient_count=int(len(eligible)),
    )
