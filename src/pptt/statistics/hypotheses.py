from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import wilcoxon


@dataclass(frozen=True)
class PairedPatientStatistics:
    patient_count: int
    mean_difference: float
    median_difference: float
    ci_low: float
    ci_high: float
    wilcoxon_p: float
    paired_cohens_d: float
    rank_biserial: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def paired_patient_statistics(
    left: np.ndarray,
    right: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> PairedPatientStatistics:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.ndim != 1 or second.shape != first.shape or first.size < 2:
        raise ValueError("paired statistics require equal vectors with at least two patients")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("paired values must be finite")
    if iterations < 100:
        raise ValueError("bootstrap iterations must be at least 100")
    differences = first - second
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(iterations, differences.size))
    bootstrap_means = differences[indices].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        wilcoxon_p = 1.0
        rank_biserial = 0.0
    else:
        wilcoxon_p = float(
            wilcoxon(
                differences,
                zero_method="pratt",
                alternative="two-sided",
                method="auto",
            ).pvalue
        )
        rank_biserial = float(
            (np.count_nonzero(nonzero > 0) - np.count_nonzero(nonzero < 0))
            / nonzero.size
        )
    standard_deviation = float(np.std(differences, ddof=1))
    paired_cohens_d = (
        float(differences.mean() / standard_deviation)
        if standard_deviation > 0
        else (0.0 if differences.mean() == 0 else float("nan"))
    )
    return PairedPatientStatistics(
        patient_count=int(first.size),
        mean_difference=float(differences.mean()),
        median_difference=float(np.median(differences)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        wilcoxon_p=wilcoxon_p,
        paired_cohens_d=paired_cohens_d,
        rank_biserial=rank_biserial,
    )


def adjacent_differences(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] < 2:
        raise ValueError("values must contain at least two ordered stages")
    return np.diff(array, axis=-1)


def adjacent_accelerations(values: np.ndarray) -> np.ndarray:
    differences = adjacent_differences(values)
    if differences.shape[-1] < 2:
        raise ValueError("values must contain at least three ordered stages")
    return np.diff(differences, axis=-1)
