from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from pptt.statistics.hypotheses import paired_patient_statistics
from pptt.statistics.process_effects import holm_adjust


CELL_GROUP_COLUMNS = [
    "model",
    "model_seed",
    "transition_index",
    "transition",
    "restore_node",
    "is_receiving_node_cell",
]


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all() or array.size == 0:
        raise ValueError("bootstrap values must be a nonempty finite vector")
    if iterations < 100:
        raise ValueError("bootstrap iterations must be at least 100")
    if array.size == 1:
        return float(array[0]), float(array[0])
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, array.size, size=(iterations, array.size))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_alignment_cells(
    rows: pd.DataFrame,
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    minimum_patients: int = 30,
    minimum_aggregate_pixels: int = 2048,
) -> pd.DataFrame:
    """Return patient-bootstrap cell estimates without selecting matrix cells."""
    missing = set(CELL_GROUP_COLUMNS + [
        "patient_id",
        "status",
        "specific_effect",
        "target_pixel_count",
    ]) - set(rows.columns)
    if missing:
        raise ValueError(f"alignment cell rows lack columns: {sorted(missing)}")
    summaries: list[dict[str, Any]] = []
    grouped = rows.groupby(CELL_GROUP_COLUMNS, sort=True, dropna=False)
    for group_index, (identity, group) in enumerate(grouped):
        values = pd.to_numeric(group["specific_effect"], errors="coerce")
        valid = (group["status"] == "EVALUABLE") & np.isfinite(values)
        selected = group.loc[valid].copy()
        selected_values = values.loc[valid].to_numpy(dtype=np.float64)
        patient_count = int(selected["patient_id"].nunique())
        aggregate_pixels = int(
            pd.to_numeric(selected["target_pixel_count"], errors="raise").sum()
        )
        if selected_values.size:
            ci_low, ci_high = _bootstrap_mean_interval(
                selected_values,
                iterations=bootstrap_iterations,
                seed=int(bootstrap_seed) + group_index,
            )
            mean = float(selected_values.mean())
            median = float(np.median(selected_values))
        else:
            ci_low = ci_high = mean = median = float("nan")
        summary = dict(zip(CELL_GROUP_COLUMNS, identity))
        summary.update(
            {
                "specific_patient_count": patient_count,
                "aggregate_target_pixel_count": aggregate_pixels,
                "mean_specific_effect": mean,
                "median_specific_effect": median,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "target_only_patient_count": int(
                    group.loc[
                        group["status"] == "TARGET_ONLY_CONTROL_UNAVAILABLE",
                        "patient_id",
                    ].nunique()
                ),
                "evaluable": bool(
                    patient_count >= int(minimum_patients)
                    and aggregate_pixels >= int(minimum_aggregate_pixels)
                ),
            }
        )
        summaries.append(summary)
    return pd.DataFrame(summaries)


def compute_patient_alignment_values(
    rows: pd.DataFrame,
    *,
    receiving_node_by_transition: Mapping[str, str],
) -> pd.DataFrame:
    required = {
        "model",
        "model_seed",
        "patient_id",
        "transition",
        "restore_node",
        "status",
        "specific_effect",
        "target_pixel_count",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"alignment rows lack columns: {sorted(missing)}")
    frame = rows.copy()
    frame["registered_receiving_node"] = frame["transition"].map(
        dict(receiving_node_by_transition)
    )
    frame = frame[
        frame["registered_receiving_node"].notna()
        & (frame["restore_node"] == frame["registered_receiving_node"])
    ].copy()
    frame["specific_effect"] = pd.to_numeric(
        frame["specific_effect"], errors="coerce"
    )
    frame["target_pixel_count"] = pd.to_numeric(
        frame["target_pixel_count"], errors="raise"
    )
    frame = frame[
        (frame["status"] == "EVALUABLE")
        & np.isfinite(frame["specific_effect"])
        & (frame["target_pixel_count"] > 0)
    ]
    duplicate_columns = ["model", "model_seed", "patient_id", "transition"]
    if frame.duplicated(duplicate_columns).any():
        raise ValueError("receiving-node rows contain duplicate patient transitions")
    values: list[dict[str, Any]] = []
    for identity, group in frame.groupby(
        ["model", "model_seed", "patient_id"], sort=True
    ):
        effects = group["specific_effect"].to_numpy(dtype=np.float64)
        weights = group["target_pixel_count"].to_numpy(dtype=np.float64)
        model, seed, patient_id = identity
        common = {
            "model": model,
            "model_seed": int(seed),
            "patient_id": patient_id,
            "evaluable_transition_count": int(group["transition"].nunique()),
        }
        values.append(
            {
                **common,
                "endpoint": "macro",
                "value": float(effects.mean()),
                "weight": float(len(effects)),
            }
        )
        values.append(
            {
                **common,
                "endpoint": "micro",
                "value": float(np.average(effects, weights=weights)),
                "weight": float(weights.sum()),
            }
        )
    return pd.DataFrame(values)


def compute_global_alignment_statistics(
    rows: pd.DataFrame,
    *,
    receiving_node_by_transition: Mapping[str, str],
    bootstrap_iterations: int = 10000,
    bootstrap_seed: int = 20260717,
) -> pd.DataFrame:
    """Return macro and micro receiving-node statistics by model and seed."""
    patient_values = compute_patient_alignment_values(
        rows,
        receiving_node_by_transition=receiving_node_by_transition,
    )
    summaries: list[dict[str, Any]] = []
    if patient_values.empty:
        return pd.DataFrame()
    for group_index, (identity, group) in enumerate(
        patient_values.groupby(["model", "model_seed", "endpoint"], sort=True)
    ):
        values = group["value"].to_numpy(dtype=np.float64)
        if values.size >= 2:
            estimate = paired_patient_statistics(
                values,
                np.zeros_like(values),
                iterations=bootstrap_iterations,
                seed=int(bootstrap_seed) + group_index,
            ).to_dict()
        else:
            estimate = {
                "patient_count": 1,
                "mean_difference": float(values[0]),
                "median_difference": float(values[0]),
                "ci_low": float(values[0]),
                "ci_high": float(values[0]),
                "wilcoxon_p": 1.0,
                "paired_cohens_d": float("nan"),
                "rank_biserial": float(np.sign(values[0])),
            }
        model, seed, endpoint = identity
        summaries.append(
            {
                "model": model,
                "model_seed": int(seed),
                "endpoint": endpoint,
                **estimate,
                "total_weight": float(group["weight"].sum()),
                "median_evaluable_transition_count": float(
                    group["evaluable_transition_count"].median()
                ),
            }
        )
    result = pd.DataFrame(summaries)
    result["holm_p"] = holm_adjust(result["wilcoxon_p"].to_numpy(dtype=np.float64))
    result.attrs["patient_values"] = patient_values
    return result


def evaluate_network_alignment_gate(
    global_rows: pd.DataFrame,
    cell_rows: pd.DataFrame,
    audit: Mapping[str, object],
    *,
    minimum_supported_seeds: int,
    minimum_evaluable_transitions: int,
    minimum_positive_receiving_transitions: int,
    alpha: float,
) -> dict[str, object]:
    """Return one registered status and the complete gate audit."""
    required_audits = (
        "mathematical_properties_pass",
        "protocol_identity_pass",
        "formal_job_matrix_pass",
        "operator_audit_pass",
    )
    audit_pass = all(audit.get(name) is True for name in required_audits)
    models = sorted(
        set(global_rows.get("model", pd.Series(dtype=object)).dropna().tolist())
        | set(cell_rows.get("model", pd.Series(dtype=object)).dropna().tolist())
    )
    coverage: dict[str, Any] = {}
    model_alignment: dict[str, Any] = {}
    for model in models:
        selected_cells = cell_rows[
            (cell_rows["model"] == model)
            & cell_rows["is_receiving_node_cell"].astype(bool)
            & cell_rows["evaluable"].astype(bool)
        ]
        seed_transition_counts = (
            selected_cells.groupby("model_seed")["transition_index"].nunique().to_dict()
        )
        coverage_seeds = sorted(
            int(seed)
            for seed, count in seed_transition_counts.items()
            if int(count) >= int(minimum_evaluable_transitions)
        )
        coverage[model] = {
            "seed_transition_counts": {
                str(int(seed)): int(count)
                for seed, count in seed_transition_counts.items()
            },
            "coverage_supported_seeds": coverage_seeds,
            "coverage_pass": len(coverage_seeds) >= int(minimum_supported_seeds),
        }

        endpoint_support: dict[str, list[int]] = {}
        model_global = global_rows[global_rows["model"] == model]
        for endpoint in ("micro", "macro"):
            endpoint_rows = model_global[model_global["endpoint"] == endpoint]
            endpoint_support[endpoint] = sorted(
                int(row.model_seed)
                for row in endpoint_rows.itertuples()
                if float(row.mean_difference) > 0
                and float(row.ci_low) > 0
                and float(row.holm_p) < float(alpha)
            )
        positive_counts = (
            selected_cells.assign(
                positive=pd.to_numeric(
                    selected_cells["mean_specific_effect"], errors="coerce"
                )
                > 0
            )
            .groupby("model_seed")["positive"]
            .sum()
            .to_dict()
        )
        positive_seeds = sorted(
            int(seed)
            for seed, count in positive_counts.items()
            if int(count) >= int(minimum_positive_receiving_transitions)
        )
        endpoints_pass = all(
            len(endpoint_support[endpoint]) >= int(minimum_supported_seeds)
            for endpoint in ("micro", "macro")
        )
        positive_pass = len(positive_seeds) >= int(minimum_supported_seeds)
        model_alignment[model] = {
            "endpoint_supported_seeds": endpoint_support,
            "positive_receiving_transition_counts": {
                str(int(seed)): int(count) for seed, count in positive_counts.items()
            },
            "positive_supported_seeds": positive_seeds,
            "alignment_pass": bool(endpoints_pass and positive_pass),
        }

    coverage_pass = bool(models) and all(
        detail["coverage_pass"] for detail in coverage.values()
    )
    passed_models = sorted(
        model for model, detail in model_alignment.items() if detail["alignment_pass"]
    )
    if not audit_pass:
        status = "OBSERVATIONAL_PROCESS_ONLY"
    elif not coverage_pass:
        status = "INSUFFICIENT_NETWORK_COVERAGE"
    elif len(passed_models) == len(models) and len(models) >= 2:
        status = "PASS_NETWORK_WIDE"
    elif len(passed_models) == 1:
        status = "PARTIAL_NETWORK_ALIGNMENT"
    else:
        status = "OBSERVATIONAL_PROCESS_ONLY"
    return {
        "status": status,
        "network_wide_claim_authorized": status == "PASS_NETWORK_WIDE",
        "audit_pass": audit_pass,
        "audit": dict(audit),
        "coverage": coverage,
        "model_alignment": model_alignment,
        "passed_models": passed_models,
    }


__all__ = [
    "compute_global_alignment_statistics",
    "compute_patient_alignment_values",
    "evaluate_network_alignment_gate",
    "summarize_alignment_cells",
]
