from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from pptt.causal_abstraction.kernels import TransitionProcess
from pptt.causal_abstraction.states import RelationshipState


STATE_DISTRIBUTION_NAMES = tuple(state.name for state in RelationshipState)
_CONDITION_COLUMNS = (
    "architecture",
    "model_seed",
    "process_source",
    "intervention_node",
    "source_state",
    "downstream_node",
    "downstream_depth",
    "truth_class",
    "condition",
    "dose",
)
_CORE_FIDELITY_COLUMNS = (
    "patient_id",
    "intervention_node",
    "source_state",
    "downstream_node",
    "truth_class",
)


@dataclass(frozen=True)
class PathFidelitySummary:
    patient_rows: pd.DataFrame
    condition_rows: pd.DataFrame
    macro_tv: float
    worst_tv: float
    worst_ci_high: float
    worst_condition: dict[str, Any]
    patient_count: int


@dataclass(frozen=True)
class CrossArchitectureTransfer:
    process_source: str
    network_target: str
    evaluated_rows: pd.DataFrame
    summary: PathFidelitySummary


@dataclass(frozen=True)
class InterventionSpecificitySummary:
    patient_rows: pd.DataFrame
    task_gain: float
    null_gain: float
    delta_gain: float
    delta_ci_low: float
    delta_ci_high: float
    null_clean_tv: float
    patient_count: int


@dataclass(frozen=True)
class DoseDirectionSummary:
    patient_rows: pd.DataFrame
    condition_rows: pd.DataFrame
    aligned_fraction: float
    reverse_condition_count: int


@dataclass(frozen=True)
class StructuralProcessContrast:
    patient_rows: pd.DataFrame
    node_rows: pd.DataFrame
    between: float
    within: float
    delta_distance: float
    ci_low: float
    ci_high: float
    patient_count: int


@dataclass(frozen=True)
class PairedFidelityDifference:
    patient_rows: pd.DataFrame
    estimate: float
    ci_low: float
    ci_high: float
    patient_count: int


def _distribution_columns(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}_{name}" for name in STATE_DISTRIBUTION_NAMES)


def _validate_probability_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size != len(STATE_DISTRIBUTION_NAMES):
        raise ValueError(f"{name} must contain one probability per relationship state")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    if not np.isclose(array.sum(), 1.0, rtol=1.0e-8, atol=1.0e-10):
        raise ValueError(f"{name} must sum to one")
    return array


def _validate_distribution_frame(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    columns = _distribution_columns(prefix)
    missing = tuple(column for column in columns if column not in frame)
    if missing:
        raise ValueError(f"missing {prefix} distribution columns: {missing}")
    values = frame.loc[:, columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"{prefix} distributions must be finite and nonnegative")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1.0e-7, atol=1.0e-9):
        raise ValueError(f"{prefix} distributions must sum to one")
    return values


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    first = _validate_probability_vector(left, name="left")
    second = _validate_probability_vector(right, name="right")
    return float(0.5 * np.abs(first - second).sum())


def counterfactual_gain(
    clean: np.ndarray,
    edited: np.ndarray,
    target: np.ndarray,
) -> float:
    return total_variation(clean, target) - total_variation(edited, target)


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be a nonempty finite vector")
    if int(iterations) < 100:
        raise ValueError("bootstrap_iterations must be at least 100")
    if array.size == 1:
        return float(array[0]), float(array[0])
    generator = np.random.default_rng(int(seed))
    positions = generator.integers(
        0,
        array.size,
        size=(int(iterations), array.size),
    )
    means = array[positions].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _present_condition_columns(
    frame: pd.DataFrame,
    *,
    include_dose: bool = True,
) -> list[str]:
    return [
        column
        for column in _CONDITION_COLUMNS
        if column in frame and (include_dose or column != "dose")
    ]


def summarize_path_fidelity(
    patient_rows: pd.DataFrame,
    *,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 20260720,
) -> PathFidelitySummary:
    """Quantify patient-equal full-path intervention alignment by TV error."""

    frame = patient_rows.copy()
    missing = tuple(column for column in _CORE_FIDELITY_COLUMNS if column not in frame)
    if missing:
        raise ValueError(f"fidelity rows are missing columns: {missing}")
    if frame.empty or frame["patient_id"].isna().any():
        raise ValueError("fidelity rows must be nonempty and patient identified")
    network_columns = _distribution_columns("network")
    high_level_columns = _distribution_columns("high_level")
    _validate_distribution_frame(frame, "network")
    _validate_distribution_frame(frame, "high_level")
    condition_columns = _present_condition_columns(frame)
    patient_group = ["patient_id", *condition_columns]
    aggregated = (
        frame.groupby(patient_group, sort=True, dropna=False)[
            [*network_columns, *high_level_columns]
        ]
        .mean()
        .reset_index()
    )
    network = aggregated.loc[:, network_columns].to_numpy(dtype=np.float64)
    high_level = aggregated.loc[:, high_level_columns].to_numpy(dtype=np.float64)
    aggregated["tv_error"] = 0.5 * np.abs(network - high_level).sum(axis=1)

    condition_output: list[dict[str, Any]] = []
    for group_index, (identity, group) in enumerate(
        aggregated.groupby(condition_columns, sort=True, dropna=False)
    ):
        identity_values = identity if isinstance(identity, tuple) else (identity,)
        errors = group["tv_error"].to_numpy(dtype=np.float64)
        ci_low, ci_high = _bootstrap_mean_interval(
            errors,
            iterations=int(bootstrap_iterations),
            seed=int(bootstrap_seed) + group_index,
        )
        condition_output.append(
            {
                **dict(zip(condition_columns, identity_values, strict=True)),
                "patient_count": int(group["patient_id"].nunique()),
                "mean_tv": float(errors.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    conditions = pd.DataFrame(condition_output)
    if conditions.empty:
        raise ValueError("no fidelity conditions remained after patient aggregation")
    worst_index = int(conditions["mean_tv"].to_numpy().argmax())
    worst = conditions.iloc[worst_index]
    identity_columns = [
        column
        for column in condition_columns
        if column in conditions.columns
    ]
    return PathFidelitySummary(
        patient_rows=aggregated,
        condition_rows=conditions,
        macro_tv=float(conditions["mean_tv"].mean()),
        worst_tv=float(worst["mean_tv"]),
        worst_ci_high=float(worst["ci_high"]),
        worst_condition={column: worst[column] for column in identity_columns},
        patient_count=int(aggregated["patient_id"].nunique()),
    )


def _rollout_distribution(
    process: TransitionProcess,
    *,
    intervention_node: str,
    source_state: int,
    downstream_node: str,
) -> tuple[np.ndarray, int]:
    if intervention_node not in process.node_names:
        raise ValueError(f"unknown intervention node: {intervention_node}")
    start = process.node_names.index(intervention_node)
    if downstream_node == intervention_node:
        distribution = np.zeros(process.state_count, dtype=np.float64)
        distribution[int(source_state)] = 1.0
        return distribution, 0
    if downstream_node == "Y":
        step = len(process.node_names) - start
    elif downstream_node in process.node_names:
        target = process.node_names.index(downstream_node)
        if target <= start:
            raise ValueError("downstream node must follow the intervention node")
        step = target - start
    else:
        raise ValueError(f"unknown downstream node: {downstream_node}")
    rollout = process.rollout(
        intervention_node=start,
        source_state=int(source_state),
    )
    return rollout[step - 1], step


def cross_architecture_transfer(
    process: TransitionProcess,
    intervention_rows: pd.DataFrame,
    *,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 20260720,
) -> CrossArchitectureTransfer:
    """Evaluate one source process on another network's intervention traces."""

    frame = intervention_rows.copy()
    if "architecture" not in frame or frame.empty:
        raise ValueError("cross transfer requires a nonempty architecture column")
    architectures = tuple(sorted(frame["architecture"].astype(str).unique()))
    if len(architectures) != 1:
        raise ValueError("cross transfer rows must contain exactly one target network")
    _validate_distribution_frame(frame, "network")
    predictions: list[np.ndarray] = []
    depths: list[int] = []
    for row in frame.itertuples(index=False):
        predicted, depth = _rollout_distribution(
            process,
            intervention_node=str(row.intervention_node),
            source_state=int(row.source_state),
            downstream_node=str(row.downstream_node),
        )
        predictions.append(predicted)
        depths.append(depth)
    prediction_array = np.stack(predictions)
    for index, column in enumerate(_distribution_columns("high_level")):
        frame[column] = prediction_array[:, index]
    if "downstream_depth" in frame:
        registered = pd.to_numeric(frame["downstream_depth"], errors="raise").to_numpy(
            dtype=np.int64
        )
        if not np.array_equal(registered, np.asarray(depths, dtype=np.int64)):
            raise ValueError("registered downstream_depth disagrees with process topology")
    else:
        frame["downstream_depth"] = depths
    frame["process_source"] = process.source
    summary = summarize_path_fidelity(
        frame,
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=int(bootstrap_seed),
    )
    return CrossArchitectureTransfer(
        process_source=process.source,
        network_target=architectures[0],
        evaluated_rows=frame,
        summary=summary,
    )


def summarize_intervention_specificity(
    rows: pd.DataFrame,
    *,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 20260720,
) -> InterventionSpecificitySummary:
    """Compare task edits with equal-norm observer-nullspace controls."""

    frame = rows.copy()
    if frame.empty or "patient_id" not in frame or frame["patient_id"].isna().any():
        raise ValueError("specificity rows must be nonempty and patient identified")
    distributions = {
        prefix: _validate_distribution_frame(frame, prefix)
        for prefix in ("clean", "task", "null", "target")
    }
    clean_target = 0.5 * np.abs(
        distributions["clean"] - distributions["target"]
    ).sum(axis=1)
    task_target = 0.5 * np.abs(
        distributions["task"] - distributions["target"]
    ).sum(axis=1)
    null_target = 0.5 * np.abs(
        distributions["null"] - distributions["target"]
    ).sum(axis=1)
    frame["task_gain"] = clean_target - task_target
    frame["null_gain"] = clean_target - null_target
    frame["delta_gain"] = frame["task_gain"] - frame["null_gain"]
    frame["null_clean_tv"] = 0.5 * np.abs(
        distributions["null"] - distributions["clean"]
    ).sum(axis=1)
    patient = (
        frame.groupby("patient_id", sort=True)[
            ["task_gain", "null_gain", "delta_gain", "null_clean_tv"]
        ]
        .mean()
        .reset_index()
    )
    low, high = _bootstrap_mean_interval(
        patient["delta_gain"].to_numpy(dtype=np.float64),
        iterations=int(bootstrap_iterations),
        seed=int(bootstrap_seed),
    )
    return InterventionSpecificitySummary(
        patient_rows=patient,
        task_gain=float(patient["task_gain"].mean()),
        null_gain=float(patient["null_gain"].mean()),
        delta_gain=float(patient["delta_gain"].mean()),
        delta_ci_low=low,
        delta_ci_high=high,
        null_clean_tv=float(patient["null_clean_tv"].mean()),
        patient_count=int(patient["patient_id"].nunique()),
    )


def summarize_dose_direction(
    rows: pd.DataFrame,
    *,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 20260720,
) -> DoseDirectionSummary:
    """Estimate patient-level TV slopes over the locked intervention doses."""

    frame = rows.copy()
    required = {"patient_id", "dose", *_CORE_FIDELITY_COLUMNS[1:]}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"dose rows are missing columns: {sorted(missing)}")
    network = _validate_distribution_frame(frame, "network")
    high_level = _validate_distribution_frame(frame, "high_level")
    frame["tv_error"] = 0.5 * np.abs(network - high_level).sum(axis=1)
    doses = pd.to_numeric(frame["dose"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(doses).all() or np.any((doses < 0) | (doses > 1)):
        raise ValueError("doses must be finite values inside [0, 1]")
    frame["dose"] = doses
    condition_columns = _present_condition_columns(frame, include_dose=False)
    patient_slopes: list[dict[str, Any]] = []
    for identity, group in frame.groupby(
        ["patient_id", *condition_columns],
        sort=True,
        dropna=False,
    ):
        identity_values = identity if isinstance(identity, tuple) else (identity,)
        dose_values = group["dose"].to_numpy(dtype=np.float64)
        errors = group["tv_error"].to_numpy(dtype=np.float64)
        unique_doses = np.unique(dose_values)
        if unique_doses.size < 3 or unique_doses.size != dose_values.size:
            raise ValueError("each patient-condition needs at least three unique doses")
        slope = float(np.polyfit(dose_values, errors, deg=1)[0])
        patient_slopes.append(
            {
                **dict(
                    zip(
                        ["patient_id", *condition_columns],
                        identity_values,
                        strict=True,
                    )
                ),
                "tv_slope": slope,
            }
        )
    patient = pd.DataFrame(patient_slopes)
    condition_output: list[dict[str, Any]] = []
    for group_index, (identity, group) in enumerate(
        patient.groupby(condition_columns, sort=True, dropna=False)
    ):
        identity_values = identity if isinstance(identity, tuple) else (identity,)
        slopes = group["tv_slope"].to_numpy(dtype=np.float64)
        low, high = _bootstrap_mean_interval(
            slopes,
            iterations=int(bootstrap_iterations),
            seed=int(bootstrap_seed) + group_index,
        )
        mean = float(slopes.mean())
        condition_output.append(
            {
                **dict(zip(condition_columns, identity_values, strict=True)),
                "patient_count": int(group["patient_id"].nunique()),
                "mean_slope": mean,
                "ci_low": low,
                "ci_high": high,
                "aligned": mean < 0.0,
                "significant_reverse": low > 0.0,
            }
        )
    conditions = pd.DataFrame(condition_output)
    if conditions.empty:
        raise ValueError("no dose conditions remained after patient aggregation")
    return DoseDirectionSummary(
        patient_rows=patient,
        condition_rows=conditions,
        aligned_fraction=float(conditions["aligned"].mean()),
        reverse_condition_count=int(conditions["significant_reverse"].sum()),
    )


def _patient_process_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"patient_id", "model_seed", "node", "source_state"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"process rows are missing columns: {sorted(missing)}")
    _validate_distribution_frame(frame, "process")
    columns = _distribution_columns("process")
    return (
        frame.groupby(
            ["patient_id", "model_seed", "node", "source_state"],
            sort=True,
            dropna=False,
        )[list(columns)]
        .mean()
        .reset_index()
    )


def _pairwise_seed_distance(frame: pd.DataFrame) -> pd.DataFrame:
    probability_columns = _distribution_columns("process")
    rows: list[dict[str, Any]] = []
    for identity, group in frame.groupby(
        ["patient_id", "node", "source_state"],
        sort=True,
        dropna=False,
    ):
        patient_id, node, source_state = identity
        seed_rows = {
            int(row.model_seed): np.asarray(
                [getattr(row, column) for column in probability_columns],
                dtype=np.float64,
            )
            for row in group.itertuples(index=False)
        }
        if len(seed_rows) < 2:
            raise ValueError("within-architecture distance requires multiple seeds")
        distances = [
            total_variation(seed_rows[left], seed_rows[right])
            for left, right in combinations(sorted(seed_rows), 2)
        ]
        rows.append(
            {
                "patient_id": str(patient_id),
                "node": str(node),
                "source_state": int(source_state),
                "within_distance": float(np.mean(distances)),
            }
        )
    return pd.DataFrame(rows)


def structural_process_contrast(
    baseline_rows: pd.DataFrame,
    noskip_rows: pd.DataFrame,
    seed_null_rows: pd.DataFrame,
    *,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 20260720,
) -> StructuralProcessContrast:
    """Compare baseline/no-skip process distance with baseline seed variability."""

    baseline = _patient_process_rows(baseline_rows)
    noskip = _patient_process_rows(noskip_rows)
    seed_null = _patient_process_rows(seed_null_rows)
    probability_columns = _distribution_columns("process")
    keys = ["patient_id", "node", "source_state"]
    baseline_mean = baseline.groupby(keys, sort=True)[list(probability_columns)].mean()
    noskip_mean = noskip.groupby(keys, sort=True)[list(probability_columns)].mean()
    merged = baseline_mean.join(
        noskip_mean,
        how="inner",
        lsuffix="_baseline",
        rsuffix="_noskip",
    ).reset_index()
    if merged.empty:
        raise ValueError("baseline and no-skip rows share no patient conditions")
    left = merged.loc[
        :, [f"{column}_baseline" for column in probability_columns]
    ].to_numpy(dtype=np.float64)
    right = merged.loc[
        :, [f"{column}_noskip" for column in probability_columns]
    ].to_numpy(dtype=np.float64)
    merged["between_distance"] = 0.5 * np.abs(left - right).sum(axis=1)
    within = _pairwise_seed_distance(seed_null)
    comparison = merged.merge(within, on=keys, how="inner", validate="one_to_one")
    if comparison.empty:
        raise ValueError("structural and seed-null rows share no patient conditions")
    comparison["delta_distance"] = (
        comparison["between_distance"] - comparison["within_distance"]
    )
    patient = (
        comparison.groupby("patient_id", sort=True)[
            ["between_distance", "within_distance", "delta_distance"]
        ]
        .mean()
        .reset_index()
    )
    low, high = _bootstrap_mean_interval(
        patient["delta_distance"].to_numpy(dtype=np.float64),
        iterations=int(bootstrap_iterations),
        seed=int(bootstrap_seed),
    )
    node_rows = (
        comparison.groupby("node", sort=True)[
            ["between_distance", "within_distance", "delta_distance"]
        ]
        .mean()
        .reset_index()
    )
    return StructuralProcessContrast(
        patient_rows=patient,
        node_rows=node_rows,
        between=float(patient["between_distance"].mean()),
        within=float(patient["within_distance"].mean()),
        delta_distance=float(patient["delta_distance"].mean()),
        ci_low=low,
        ci_high=high,
        patient_count=int(patient["patient_id"].nunique()),
    )


def paired_fidelity_difference(
    shared: PathFidelitySummary,
    architecture_specific: PathFidelitySummary,
    *,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 20260720,
) -> PairedFidelityDifference:
    """Return shared-minus-specific TV using identical patient conditions."""

    excluded = {"tv_error", "process_source"}
    keys = [
        column
        for column in shared.patient_rows.columns
        if column in architecture_specific.patient_rows.columns
        and column not in excluded
        and not column.startswith("network_")
        and not column.startswith("high_level_")
    ]
    if "patient_id" not in keys:
        raise ValueError("paired fidelity comparison requires patient identity")
    left = shared.patient_rows.loc[:, [*keys, "tv_error"]].rename(
        columns={"tv_error": "shared_tv"}
    )
    right = architecture_specific.patient_rows.loc[:, [*keys, "tv_error"]].rename(
        columns={"tv_error": "specific_tv"}
    )
    paired = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if paired.empty:
        raise ValueError("shared and architecture-specific summaries do not align")
    paired["tv_increment"] = paired["shared_tv"] - paired["specific_tv"]
    patient = (
        paired.groupby("patient_id", sort=True)["tv_increment"]
        .mean()
        .reset_index()
    )
    low, high = _bootstrap_mean_interval(
        patient["tv_increment"].to_numpy(dtype=np.float64),
        iterations=int(bootstrap_iterations),
        seed=int(bootstrap_seed),
    )
    return PairedFidelityDifference(
        patient_rows=patient,
        estimate=float(patient["tv_increment"].mean()),
        ci_low=low,
        ci_high=high,
        patient_count=int(patient["patient_id"].nunique()),
    )


__all__ = [
    "CrossArchitectureTransfer",
    "DoseDirectionSummary",
    "InterventionSpecificitySummary",
    "PairedFidelityDifference",
    "PathFidelitySummary",
    "STATE_DISTRIBUTION_NAMES",
    "StructuralProcessContrast",
    "counterfactual_gain",
    "cross_architecture_transfer",
    "paired_fidelity_difference",
    "structural_process_contrast",
    "summarize_dose_direction",
    "summarize_intervention_specificity",
    "summarize_path_fidelity",
    "total_variation",
]
