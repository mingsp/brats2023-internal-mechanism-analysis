from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pptt.experiments.pilot import stable_seed
from pptt.statistics.causal_effects import evaluate_causal_conclusion_gate
from pptt.statistics.hypotheses import paired_patient_statistics
from pptt.statistics.process_effects import holm_adjust
from pptt.visualization.causal import bootstrap_mean_interval


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must contain a mapping: {path}")
    return payload


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            _json_ready(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _resolve_under(workspace: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def _collect(
    root: Path,
    *,
    model: str,
    seeds: tuple[int, ...],
    expected_patient_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligibility_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    for seed in seeds:
        job_root = root / model / f"seed_{seed}"
        status_path = job_root / "job_status.json"
        if not status_path.is_file():
            raise FileNotFoundError(status_path)
        job_status = json.loads(status_path.read_text(encoding="utf-8"))
        if (
            job_status.get("status") != "COMPLETE"
            or int(job_status.get("patient_count", -1)) != expected_patient_count
        ):
            raise ValueError(f"V6 job is incomplete: {status_path}")
        patient_files = sorted((job_root / "patient_results").glob("*.json"))
        if len(patient_files) != expected_patient_count:
            raise ValueError(f"unexpected patient JSON count for seed {seed}")
        patient_ids = [path.stem for path in patient_files]
        if len(patient_ids) != len(set(patient_ids)):
            raise ValueError(f"duplicate patient JSON for seed {seed}")
        for path in patient_files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            patient_id = str(payload["patient_id"])
            status = str(payload["status"])
            eligibility_rows.append(
                {
                    "model_seed": seed,
                    "patient_id": patient_id,
                    "status": status,
                    "selected_slice_id": payload.get("selected_slice_id"),
                    "target_pixel_count": payload.get("target_pixel_count"),
                    "target_feature_count": payload.get("target_feature_count"),
                    "matched_feature_count": payload.get("matched_feature_count"),
                    "specificity_evaluable": bool(
                        payload.get("specificity_evaluable", False)
                    ),
                }
            )
            if status != "PASS":
                continue
            effects = payload["effects"]
            effect_rows.append(
                {
                    "model_seed": seed,
                    "patient_id": patient_id,
                    "selected_slice_id": payload["selected_slice_id"],
                    "target_pixel_count": int(payload["target_pixel_count"]),
                    "target_feature_count": int(payload["target_feature_count"]),
                    "matched_feature_count": int(payload["matched_feature_count"]),
                    "specificity_evaluable": bool(
                        payload["specificity_evaluable"]
                    ),
                    "total_effect": float(effects["total_effect"]),
                    "target_indirect_effect": float(
                        effects["target_indirect_effect"]
                    ),
                    "matched_target_indirect_effect": (
                        np.nan
                        if effects["matched_target_indirect_effect"] is None
                        else float(effects["matched_target_indirect_effect"])
                    ),
                    "control_indirect_effect": (
                        np.nan
                        if effects["control_indirect_effect"] is None
                        else float(effects["control_indirect_effect"])
                    ),
                    "target_minus_control": (
                        np.nan
                        if effects["target_minus_control"] is None
                        else float(effects["target_minus_control"])
                    ),
                    "target_proportion_mediated": (
                        np.nan
                        if effects["target_proportion_mediated"] is None
                        else float(effects["target_proportion_mediated"])
                    ),
                    "target_activation_error": float(
                        payload["audit"][
                            "target_alpha_1_activation_max_abs_error"
                        ]
                    ),
                    "baseline_hook_count_after": int(
                        payload["audit"]["baseline_hook_count_after"]
                    ),
                    "clean_anchor_mismatch_count": int(
                        sum(payload["audit"]["clean_anchor"].values())
                    ),
                }
            )
            for condition in payload["conditions"]:
                condition_rows.append(
                    {
                        "model_seed": seed,
                        "patient_id": patient_id,
                        **condition,
                    }
                )
            negative_rows.append(
                {
                    "model_seed": seed,
                    "patient_id": patient_id,
                    **payload["negative_control"],
                }
            )
    return (
        pd.DataFrame(eligibility_rows),
        pd.DataFrame(effect_rows),
        pd.DataFrame(condition_rows),
        pd.DataFrame(negative_rows),
    )


def _primary_statistics(
    effects: pd.DataFrame,
    *,
    seeds: tuple[int, ...],
    iterations: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    endpoint_columns = {
        "necessity": "total_effect",
        "restoration": "target_indirect_effect",
        "specificity": "target_minus_control",
    }
    rows = []
    for endpoint, column in endpoint_columns.items():
        for model_seed in seeds:
            values = effects.loc[effects.model_seed == model_seed, column].dropna()
            if values.size < 2:
                continue
            result = paired_patient_statistics(
                values.to_numpy(dtype=np.float64),
                np.zeros(values.size, dtype=np.float64),
                iterations=iterations,
                seed=stable_seed(
                    bootstrap_seed,
                    "v6",
                    endpoint,
                    model_seed,
                ),
            )
            rows.append(
                {
                    "endpoint": endpoint,
                    "model_seed": model_seed,
                    **result.to_dict(),
                }
            )
    statistics = pd.DataFrame(rows)
    if statistics.shape[0] != len(endpoint_columns) * len(seeds):
        raise ValueError("V6 primary family lacks a registered seed-endpoint test")
    statistics["holm_p"] = holm_adjust(
        statistics.wilcoxon_p.to_numpy(dtype=np.float64)
    )
    return statistics


def _dose_summary(
    conditions: pd.DataFrame,
    *,
    seeds: tuple[int, ...],
    alphas: tuple[float, ...],
    iterations: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    names = ["corrupt", *[f"restore_target_{alpha:.2f}" for alpha in alphas]]
    doses = [0.0, *alphas]
    rows = []
    for model_seed in seeds:
        means = []
        for condition, alpha in zip(names, doses, strict=True):
            selected = conditions[
                (conditions.model_seed == model_seed)
                & (conditions.condition == condition)
            ].persistent_retention.astype(float)
            if selected.empty:
                raise ValueError(f"missing V6 dose condition {condition}/seed_{model_seed}")
            values = selected.to_numpy(dtype=np.float64)
            ci_low, ci_high = bootstrap_mean_interval(
                values,
                iterations=iterations,
                seed=stable_seed(
                    bootstrap_seed,
                    "v6_dose",
                    model_seed,
                    condition,
                ),
            )
            means.append(float(selected.mean()))
            rows.append(
                {
                    "model_seed": model_seed,
                    "condition": condition,
                    "alpha": alpha,
                    "patient_count": int(selected.size),
                    "mean_persistent_retention": float(selected.mean()),
                    "median_persistent_retention": float(selected.median()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "monotonic": None,
                }
            )
        monotonic = bool(np.all(np.diff(means) >= -1.0e-12))
        for row in rows[-len(names) :]:
            row["monotonic"] = monotonic
    return pd.DataFrame(rows)


def _representative_case(
    effects: pd.DataFrame,
    conditions: pd.DataFrame,
) -> dict[str, Any]:
    dose_names = [
        "corrupt",
        "restore_target_0.25",
        "restore_target_0.50",
        "restore_target_0.75",
        "restore_target_1.00",
    ]
    dose_pivot = conditions[conditions.condition.isin(dose_names)].pivot(
        index=["model_seed", "patient_id"],
        columns="condition",
        values="persistent_retention",
    )
    dose_pivot["patient_dose_monotonic"] = np.all(
        np.diff(dose_pivot[dose_names].to_numpy(dtype=np.float64), axis=1)
        >= -1.0e-12,
        axis=1,
    )
    candidates = effects.merge(
        dose_pivot[["patient_dose_monotonic"]].reset_index(),
        on=["model_seed", "patient_id"],
        validate="one_to_one",
    )
    candidates = candidates[
        candidates.specificity_evaluable
        & candidates.patient_dose_monotonic
        & (candidates.total_effect > 0)
        & (candidates.target_indirect_effect > 0)
        & (candidates.target_minus_control > 0)
    ].copy()
    if candidates.empty:
        raise ValueError("no V6 case satisfies the transparent visualization rule")
    metrics = ["total_effect", "target_indirect_effect", "target_minus_control"]
    for metric in metrics:
        scale = float(candidates[metric].std(ddof=0))
        if scale <= 0:
            scale = 1.0
        candidates[f"distance_{metric}"] = (
            (candidates[metric] - candidates[metric].median()) / scale
        ).abs()
    candidates["median_profile_distance"] = candidates[
        [f"distance_{metric}" for metric in metrics]
    ].sum(axis=1)
    selected = candidates.sort_values(
        ["median_profile_distance", "model_seed", "patient_id"],
        kind="mergesort",
    ).iloc[0]
    return {
        "selection_rule": (
            "Among evaluable patients with positive necessity, restoration, "
            "specificity, and monotonic patient-level dose response, select the "
            "case nearest the multivariate median effect profile."
        ),
        "model_seed": int(selected.model_seed),
        "patient_id": str(selected.patient_id),
        "selected_slice_id": str(selected.selected_slice_id),
        "total_effect": float(selected.total_effect),
        "target_indirect_effect": float(selected.target_indirect_effect),
        "target_minus_control": float(selected.target_minus_control),
        "target_pixel_count": int(selected.target_pixel_count),
        "median_profile_distance": float(selected.median_profile_distance),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize formal V6 causal tracing.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/v6_pixel_transition_causal_tracing.yaml"),
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    config_path = _resolve_under(workspace, args.config)
    config = _load_yaml(config_path)
    output_root = (
        _resolve_under(workspace, args.output_root)
        if args.output_root is not None
        else (workspace / str(config["output_root"])).resolve()
    )
    seeds = tuple(int(value) for value in config["model_seeds"])
    model = str(config["models"]["causal"])
    eligibility, effects, conditions, negative = _collect(
        output_root,
        model=model,
        seeds=seeds,
        expected_patient_count=250,
    )
    statistics_config = config["statistics"]
    statistics = _primary_statistics(
        effects,
        seeds=seeds,
        iterations=int(statistics_config["bootstrap_iterations"]),
        bootstrap_seed=int(statistics_config["bootstrap_seed"]),
    )
    alphas = tuple(float(value) for value in config["intervention"]["restoration_alphas"])
    dose = _dose_summary(
        conditions,
        seeds=seeds,
        alphas=alphas,
        iterations=int(statistics_config["bootstrap_iterations"]),
        bootstrap_seed=int(statistics_config["bootstrap_seed"]),
    )
    tolerance = float(config["intervention"]["restore_logit_tolerance"])
    negative_pass = bool(
        not negative.empty
        and (negative.max_abs_logit_error.astype(float) <= tolerance).all()
        and (negative.state_mismatch_count.astype(int) == 0).all()
        and (negative.reliability_mismatch_count.astype(int) == 0).all()
        and (negative.final_state_mismatch_count.astype(int) == 0).all()
        and (negative.hook_count_after.astype(int) == 0).all()
    )
    restoration_pass = bool(
        not effects.empty
        and (effects.target_activation_error <= tolerance).all()
        and (effects.baseline_hook_count_after == 0).all()
        and (effects.clean_anchor_mismatch_count == 0).all()
    )
    dose_rows = [
        {
            "model_seed": int(seed),
            "monotonic": bool(
                dose.loc[dose.model_seed == seed, "monotonic"].iloc[0]
            ),
        }
        for seed in seeds
    ]
    conclusion = evaluate_causal_conclusion_gate(
        statistics.to_dict("records"),
        dose_rows=dose_rows,
        negative_control_pass=negative_pass,
        restoration_audit_pass=restoration_pass,
        minimum_supported_seeds=int(
            statistics_config["minimum_supported_model_seeds"]
        ),
        alpha=float(statistics_config["alpha"]),
        minimum_standardized_effect=float(
            statistics_config["minimum_standardized_effect"]
        ),
    )
    representative = _representative_case(effects, conditions)

    outputs = {
        "eligibility": "causal_eligibility_audit.parquet",
        "effects": "causal_patient_effects.parquet",
        "conditions": "causal_condition_retention.parquet",
        "negative": "causal_negative_control_audit.parquet",
        "statistics": "causal_patient_statistics.parquet",
        "dose": "causal_dose_summary.parquet",
    }
    for frame, filename in (
        (eligibility, outputs["eligibility"]),
        (effects, outputs["effects"]),
        (conditions, outputs["conditions"]),
        (negative, outputs["negative"]),
        (statistics, outputs["statistics"]),
        (dose, outputs["dose"]),
    ):
        frame.to_parquet(output_root / filename, index=False)
    _write_json(output_root / "causal_conclusion_gate.json", conclusion)
    _write_json(output_root / "causal_representative_case.json", representative)
    status = {
        "status": "PASS",
        "formal_patient_files": int(eligibility.shape[0]),
        "evaluable_patient_rows": int(effects.shape[0]),
        "specificity_evaluable_rows": int(effects.specificity_evaluable.sum()),
        "status_counts": {
            f"seed_{seed}": eligibility.loc[
                eligibility.model_seed == seed, "status"
            ].value_counts().sort_index().to_dict()
            for seed in seeds
        },
        "negative_control_pass": negative_pass,
        "restoration_audit_pass": restoration_pass,
        "causal_conclusion_passed": bool(conclusion["passed"]),
        "outputs": outputs,
    }
    _write_json(output_root / "causal_status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    print(
        statistics[
            [
                "endpoint",
                "model_seed",
                "patient_count",
                "mean_difference",
                "ci_low",
                "ci_high",
                "holm_p",
                "paired_cohens_d",
                "rank_biserial",
            ]
        ].to_string(index=False)
    )
    print(
        dose[
            [
                "model_seed",
                "alpha",
                "patient_count",
                "mean_persistent_retention",
                "monotonic",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
