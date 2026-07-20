from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from pptt.statistics.hypotheses import paired_patient_statistics
from pptt.statistics.process_effects import holm_adjust


_PATIENT_COLUMNS = {
    "path_id",
    "topology_order",
    "model_seed",
    "patient_id",
    "net_recovery",
    "selected_slice_target_pixel_count",
    "eligible",
}


def _unique_ints(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must be a nonempty unique sequence")
    return result


def summarize_validation_candidates(
    patient_rows: pd.DataFrame,
    *,
    required_seeds: Sequence[int],
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = _PATIENT_COLUMNS - set(patient_rows.columns)
    if missing:
        raise ValueError(f"candidate patient table lacks columns: {sorted(missing)}")
    seeds = _unique_ints(required_seeds, name="required_seeds")
    frame = patient_rows.copy()
    if frame.empty:
        raise ValueError("candidate patient table must be nonempty")
    frame["model_seed"] = pd.to_numeric(frame["model_seed"], errors="raise").astype(int)
    frame["topology_order"] = pd.to_numeric(
        frame["topology_order"], errors="raise"
    ).astype(int)
    frame["net_recovery"] = pd.to_numeric(
        frame["net_recovery"], errors="raise"
    ).astype(float)
    frame["selected_slice_target_pixel_count"] = pd.to_numeric(
        frame["selected_slice_target_pixel_count"], errors="raise"
    ).astype(int)
    if not np.isfinite(frame["net_recovery"].to_numpy()).all():
        raise ValueError("net_recovery must be finite")
    if (frame["selected_slice_target_pixel_count"] < 0).any():
        raise ValueError("target pixel counts must be nonnegative")
    if set(frame["model_seed"].unique()) != set(seeds):
        raise ValueError("candidate table does not contain exactly the required seeds")
    duplicate = frame.duplicated(["path_id", "model_seed", "patient_id"])
    if duplicate.any():
        raise ValueError("candidate table contains duplicate path-seed-patient rows")
    topology_counts = frame.groupby("path_id")["topology_order"].nunique()
    if (topology_counts != 1).any():
        raise ValueError("one path_id maps to multiple topology orders")
    patient_sets: dict[int, set[str]] = {}
    for seed in seeds:
        groups = frame[frame["model_seed"] == seed].groupby("path_id")
        sets = [set(group["patient_id"].astype(str)) for _, group in groups]
        if not sets or any(current != sets[0] for current in sets[1:]):
            raise ValueError(f"candidate paths do not share one patient set for seed {seed}")
        patient_sets[seed] = sets[0]

    statistics_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    ordered_groups = frame.groupby(
        ["topology_order", "path_id", "model_seed"],
        sort=True,
    )
    for (topology_order, path_id, model_seed), group in ordered_groups:
        if int(model_seed) not in seeds:
            continue
        ordered = group.sort_values("patient_id", kind="mergesort")
        values = ordered["net_recovery"].to_numpy(dtype=np.float64)
        result = paired_patient_statistics(
            values,
            np.zeros_like(values),
            iterations=int(bootstrap_iterations),
            seed=(
                int(bootstrap_seed)
                + int(topology_order) * 100_003
                + int(model_seed)
            ),
        ).to_dict()
        statistics_rows.append(
            {
                "path_id": str(path_id),
                "topology_order": int(topology_order),
                "model_seed": int(model_seed),
                **result,
            }
        )
        eligible = ordered["eligible"].astype(bool)
        coverage_rows.append(
            {
                "path_id": str(path_id),
                "topology_order": int(topology_order),
                "model_seed": int(model_seed),
                "patient_count": int(len(ordered)),
                "eligible_patient_count": int(eligible.sum()),
                "aggregate_target_pixel_count": int(
                    ordered.loc[
                        eligible,
                        "selected_slice_target_pixel_count",
                    ].sum()
                ),
            }
        )
    statistics = pd.DataFrame(statistics_rows).sort_values(
        ["topology_order", "model_seed"],
        kind="mergesort",
    )
    expected_rows = frame["path_id"].nunique() * len(seeds)
    if len(statistics) != expected_rows:
        raise ValueError("candidate statistics matrix is incomplete")
    statistics["holm_p"] = holm_adjust(
        statistics["wilcoxon_p"].to_numpy(dtype=np.float64)
    )
    coverage = pd.DataFrame(coverage_rows).sort_values(
        ["topology_order", "model_seed"],
        kind="mergesort",
    )
    return statistics.reset_index(drop=True), coverage.reset_index(drop=True)


def _records(values: Iterable[Mapping[str, Any]] | pd.DataFrame) -> list[dict[str, Any]]:
    if isinstance(values, pd.DataFrame):
        return values.to_dict("records")
    return [dict(value) for value in values]


def select_registered_candidate(
    statistics_rows: Iterable[Mapping[str, Any]] | pd.DataFrame,
    coverage_rows: Iterable[Mapping[str, Any]] | pd.DataFrame,
    *,
    required_seeds: Sequence[int],
    minimum_supported_seeds: int,
    minimum_patients_per_seed: int,
    minimum_target_pixels_per_seed: int,
    alpha: float,
    minimum_standardized_effect: float,
) -> dict[str, Any]:
    seeds = _unique_ints(required_seeds, name="required_seeds")
    if not 1 <= int(minimum_supported_seeds) <= len(seeds):
        raise ValueError("minimum_supported_seeds is outside the seed count")
    if int(minimum_patients_per_seed) < 1 or int(minimum_target_pixels_per_seed) < 1:
        raise ValueError("coverage thresholds must be positive")
    if not 0 < float(alpha) < 1 or float(minimum_standardized_effect) < 0:
        raise ValueError("statistical thresholds are invalid")
    statistics = _records(statistics_rows)
    coverage = _records(coverage_rows)
    if not statistics or not coverage:
        raise ValueError("candidate statistics and coverage must be nonempty")
    path_ids = sorted({str(row["path_id"]) for row in statistics})
    evaluated = []
    for path_id in path_ids:
        selected_statistics = [
            row for row in statistics if str(row["path_id"]) == path_id
        ]
        selected_coverage = [
            row for row in coverage if str(row["path_id"]) == path_id
        ]
        stats_by_seed = {int(row["model_seed"]): row for row in selected_statistics}
        coverage_by_seed = {int(row["model_seed"]): row for row in selected_coverage}
        if (
            set(stats_by_seed) != set(seeds)
            or set(coverage_by_seed) != set(seeds)
            or len(selected_statistics) != len(seeds)
            or len(selected_coverage) != len(seeds)
        ):
            raise ValueError(f"candidate seed matrix is incomplete for {path_id}")
        topology_orders = {
            int(row["topology_order"])
            for row in (*selected_statistics, *selected_coverage)
        }
        if len(topology_orders) != 1:
            raise ValueError(f"candidate topology is inconsistent for {path_id}")
        seed_results = []
        standardized_effects = []
        all_directions_positive = True
        for seed in seeds:
            row = stats_by_seed[seed]
            current_coverage = coverage_by_seed[seed]
            mean = float(row["mean_difference"])
            ci_low = float(row["ci_low"])
            holm_p = float(row["holm_p"])
            effect = float(row["paired_cohens_d"])
            finite = bool(np.isfinite([mean, ci_low, holm_p, effect]).all())
            coverage_pass = bool(
                int(current_coverage["eligible_patient_count"])
                >= int(minimum_patients_per_seed)
                and int(current_coverage["aggregate_target_pixel_count"])
                >= int(minimum_target_pixels_per_seed)
            )
            statistical_pass = bool(
                finite
                and mean > 0
                and ci_low > 0
                and holm_p < float(alpha)
                and effect >= float(minimum_standardized_effect)
            )
            supported = coverage_pass and statistical_pass
            all_directions_positive &= finite and mean > 0
            standardized_effects.append(effect)
            seed_results.append(
                {
                    "model_seed": seed,
                    "mean_difference": mean,
                    "ci_low": ci_low,
                    "holm_p": holm_p,
                    "paired_cohens_d": effect,
                    "eligible_patient_count": int(
                        current_coverage["eligible_patient_count"]
                    ),
                    "aggregate_target_pixel_count": int(
                        current_coverage["aggregate_target_pixel_count"]
                    ),
                    "coverage_pass": coverage_pass,
                    "statistical_pass": statistical_pass,
                    "supported": supported,
                }
            )
        supported_count = sum(bool(row["supported"]) for row in seed_results)
        finite_effects = np.asarray(standardized_effects, dtype=np.float64)
        candidate = {
            "path_id": path_id,
            "topology_order": topology_orders.pop(),
            "all_seed_directions_positive": bool(all_directions_positive),
            "supported_seed_count": int(supported_count),
            "minimum_seed_effect": (
                float(finite_effects.min())
                if np.isfinite(finite_effects).all()
                else None
            ),
            "mean_seed_effect": (
                float(finite_effects.mean())
                if np.isfinite(finite_effects).all()
                else None
            ),
            "seed_results": seed_results,
        }
        candidate["passed"] = bool(
            candidate["all_seed_directions_positive"]
            and supported_count >= int(minimum_supported_seeds)
            and candidate["minimum_seed_effect"] is not None
        )
        evaluated.append(candidate)
    passing = [row for row in evaluated if row["passed"]]
    passing.sort(
        key=lambda row: (
            -float(row["minimum_seed_effect"]),
            -float(row["mean_seed_effect"]),
            int(row["topology_order"]),
            str(row["path_id"]),
        )
    )
    thresholds = {
        "required_model_seeds": list(seeds),
        "minimum_supported_seeds": int(minimum_supported_seeds),
        "minimum_patients_per_seed": int(minimum_patients_per_seed),
        "minimum_target_pixels_per_seed": int(minimum_target_pixels_per_seed),
        "alpha": float(alpha),
        "minimum_standardized_effect": float(minimum_standardized_effect),
        "multiplicity": "holm_across_all_path_seed_tests",
    }
    if not passing:
        return {
            "status": "NO_REGISTERED_TRANSUNET_CANDIDATE",
            "test_intervention_authorized": False,
            "candidate": None,
            "evaluated_candidates": evaluated,
            "thresholds": thresholds,
            "selection_rule": (
                "positive_all_seeds_then_supported_seed_gate_then_"
                "maximum_minimum_effect_then_mean_effect_then_topology"
            ),
        }
    return {
        "status": "REGISTERED_TRANSUNET_CANDIDATE",
        "test_intervention_authorized": True,
        "candidate": passing[0],
        "evaluated_candidates": evaluated,
        "thresholds": thresholds,
        "selection_rule": (
            "positive_all_seeds_then_supported_seed_gate_then_"
            "maximum_minimum_effect_then_mean_effect_then_topology"
        ),
    }


__all__ = [
    "select_registered_candidate",
    "summarize_validation_candidates",
]
