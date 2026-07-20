from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pptt.experiments.pilot import stable_seed
from pptt.statistics.causal_replication import evaluate_replication_gate
from pptt.statistics.hypotheses import paired_patient_statistics
from pptt.statistics.process_effects import holm_adjust
try:
    from scripts.lock_v8_transunet_protocol import (
        sha256_file,
        source_tree_sha256,
    )
except ModuleNotFoundError:
    from lock_v8_transunet_protocol import (
        sha256_file,
        source_tree_sha256,
    )


ENDPOINTS = ("necessity", "restoration", "specificity")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must contain a mapping: {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must contain an object: {path}")
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


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            _json_ready(dict(payload)),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_under(workspace: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def _verified_condition(
    condition: Mapping[str, Any],
    *,
    target_pixel_count: int,
    target_flat_indices: Sequence[int],
) -> dict[str, Any]:
    count = int(condition["primary_target_count"])
    correct = int(condition["primary_target_correct_count"])
    retention = float(condition["primary_target_retention"])
    if count != int(target_pixel_count) or not 0 <= correct <= count:
        raise ValueError("condition target counts do not reconstruct the fixed cohort")
    if not np.isclose(retention, correct / count, atol=1.0e-12, rtol=0.0):
        raise ValueError("condition retention does not reconstruct from pixel counts")
    indices = tuple(int(value) for value in condition["correct_target_flat_indices"])
    if len(indices) != correct or len(indices) != len(set(indices)):
        raise ValueError("condition target-index record is inconsistent")
    if not set(indices).issubset(set(int(value) for value in target_flat_indices)):
        raise ValueError("condition correct pixels fall outside the fixed target cohort")
    return {
        "condition": str(condition["condition"]),
        "region": str(condition["region"]),
        "alpha": condition.get("alpha"),
        "primary_target_retention": retention,
        "primary_target_correct_count": correct,
        "primary_target_count": count,
        "observer_persistent_retention": float(
            condition["observer_persistent_retention"]
        ),
    }


def _collect_formal_results(
    root: Path,
    *,
    lock: Mapping[str, Any],
    model: str,
    seeds: Sequence[int],
    expected_patient_count: int,
    protocol_lock_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligibility_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    expected_patients = tuple(str(value) for value in lock["test_patient_ids"])
    if len(expected_patients) != int(expected_patient_count):
        raise ValueError("protocol lock has the wrong test patient count")
    candidate_path_id = str(lock["candidate"]["path_id"])
    for model_seed in (int(value) for value in seeds):
        job_root = root / model / f"seed_{model_seed}"
        status_path = job_root / "job_status.json"
        if not status_path.is_file():
            raise FileNotFoundError(status_path)
        job_status = _load_json(status_path)
        operator_rows.append(
            {
                "model_seed": model_seed,
                "status": str(job_status.get("status")),
                "passed": bool(job_status.get("operator_audit_pass", False)),
                "audited_patient_count": int(
                    job_status.get("audited_patient_count", 0)
                ),
                "graph_identity_exact": bool(
                    job_status.get("graph_identity_audit", {}).get(
                        "exact_equal",
                        False,
                    )
                ),
                "graph_identity_max_abs_error": float(
                    job_status.get("graph_identity_audit", {}).get(
                        "max_abs_error",
                        np.nan,
                    )
                ),
                "patient_operator_audit_pass": bool(
                    job_status.get("patient_operator_audit_pass", False)
                ),
            }
        )
        if job_status.get("status") != "COMPLETE":
            raise ValueError(
                f"V8 seed {model_seed} did not pass its formal audit: {status_path}"
            )
        if int(job_status.get("patient_count", -1)) != expected_patient_count:
            raise ValueError(f"V8 seed {model_seed} has the wrong patient count")
        if job_status.get("protocol_lock_sha256") != protocol_lock_sha256:
            raise ValueError(f"V8 seed {model_seed} used a different protocol lock")
        if job_status.get("path", {}).get("path_id") != candidate_path_id:
            raise ValueError(f"V8 seed {model_seed} used a different path")
        patient_paths = sorted((job_root / "patient_results").glob("*.json"))
        observed_patients = tuple(path.stem for path in patient_paths)
        if observed_patients != expected_patients:
            raise ValueError(f"V8 seed {model_seed} patient registry is incomplete")
        for patient_path in patient_paths:
            payload = _load_json(patient_path)
            patient_id = str(payload["patient_id"])
            if patient_id != patient_path.stem:
                raise ValueError(f"patient id differs from filename: {patient_path}")
            identity = payload.get("run_identity", {})
            if identity.get("protocol_lock_sha256") != protocol_lock_sha256:
                raise ValueError(f"patient {patient_id} used a different lock")
            if identity.get("path", {}).get("path_id") != candidate_path_id:
                raise ValueError(f"patient {patient_id} used a different path")
            status = str(payload["status"])
            eligibility_rows.append(
                {
                    "model_seed": model_seed,
                    "patient_id": patient_id,
                    "status": status,
                    "selected_slice_id": payload.get("selected_slice_id"),
                    "target_pixel_count": int(payload.get("target_pixel_count", 0)),
                    "target_feature_count": payload.get("target_feature_count"),
                    "control_feature_pool_count": payload.get(
                        "control_feature_pool_count"
                    ),
                    "matched_feature_count": payload.get("matched_feature_count"),
                    "specificity_evaluable": bool(
                        payload.get("specificity_evaluable", False)
                    ),
                }
            )
            if status != "PASS":
                continue
            target_pixel_count = int(payload["target_pixel_count"])
            target_indices = tuple(
                int(value) for value in payload["target_pixel_flat_indices"]
            )
            output_shape = tuple(int(value) for value in payload["output_shape"])
            if (
                len(target_indices) != target_pixel_count
                or len(target_indices) != len(set(target_indices))
                or len(output_shape) != 2
                or any(value <= 0 for value in output_shape)
                or any(
                    value < 0 or value >= int(np.prod(output_shape))
                    for value in target_indices
                )
            ):
                raise ValueError(f"patient {patient_id} target record is inconsistent")
            verified_conditions = [
                _verified_condition(
                    condition,
                    target_pixel_count=target_pixel_count,
                    target_flat_indices=target_indices,
                )
                for condition in payload["conditions"]
            ]
            by_name = {
                str(condition["condition"]): condition
                for condition in verified_conditions
            }
            required_names = {
                "clean",
                "corrupt",
                "restore_target_0.25",
                "restore_target_0.50",
                "restore_target_0.75",
                "restore_target_1.00",
            }
            if not required_names.issubset(by_name):
                raise ValueError(f"patient {patient_id} lacks a registered condition")
            clean = float(by_name["clean"]["primary_target_retention"])
            corrupt = float(by_name["corrupt"]["primary_target_retention"])
            restored = float(
                by_name["restore_target_1.00"]["primary_target_retention"]
            )
            specificity = payload["effects"].get("specificity")
            if not np.isclose(
                float(payload["effects"]["necessity"]),
                clean - corrupt,
                atol=1.0e-12,
                rtol=0.0,
            ):
                raise ValueError(f"patient {patient_id} necessity is not reconstructible")
            if not np.isclose(
                float(payload["effects"]["restoration"]),
                restored - corrupt,
                atol=1.0e-12,
                rtol=0.0,
            ):
                raise ValueError(f"patient {patient_id} restoration is not reconstructible")
            if payload["specificity_evaluable"]:
                target_matched = float(
                    by_name["restore_target_matched_1.00"][
                        "primary_target_retention"
                    ]
                )
                control_matched = float(
                    by_name["restore_control_matched_1.00"][
                        "primary_target_retention"
                    ]
                )
                if specificity is None or not np.isclose(
                    float(specificity),
                    target_matched - control_matched,
                    atol=1.0e-12,
                    rtol=0.0,
                ):
                    raise ValueError(
                        f"patient {patient_id} specificity is not reconstructible"
                    )
            audit = payload["audit"]
            effect_rows.append(
                {
                    "model_seed": model_seed,
                    "patient_id": patient_id,
                    "selected_slice_id": str(payload["selected_slice_id"]),
                    "target_pixel_count": target_pixel_count,
                    "target_feature_count": int(payload["target_feature_count"]),
                    "matched_feature_count": int(payload["matched_feature_count"]),
                    "specificity_evaluable": bool(
                        payload["specificity_evaluable"]
                    ),
                    "necessity": float(payload["effects"]["necessity"]),
                    "restoration": float(payload["effects"]["restoration"]),
                    "specificity": (
                        np.nan if specificity is None else float(specificity)
                    ),
                    "proportion_mediated": (
                        np.nan
                        if payload["effects"]["proportion_mediated"] is None
                        else float(payload["effects"]["proportion_mediated"])
                    ),
                    "full_restore_max_abs_error": float(
                        audit["full_restore_max_abs_error"]
                    ),
                    "target_alpha_1_max_abs_error": float(
                        audit["target_alpha_1_max_abs_error"]
                    ),
                    "spatial_shift_channel_multiset_max_abs_error": float(
                        audit["spatial_shift_channel_multiset_max_abs_error"]
                    ),
                    "receiver_hook_count_delta": int(
                        audit["receiver_hook_count_delta"]
                    ),
                    "clean_anchor_mismatch_count": int(
                        sum(audit["clean_anchor"].values())
                    ),
                }
            )
            condition_rows.extend(
                {
                    "model_seed": model_seed,
                    "patient_id": patient_id,
                    "selected_slice_id": str(payload["selected_slice_id"]),
                    **condition,
                }
                for condition in verified_conditions
            )
    eligibility_columns = (
        "model_seed",
        "patient_id",
        "status",
        "selected_slice_id",
        "target_pixel_count",
        "target_feature_count",
        "control_feature_pool_count",
        "matched_feature_count",
        "specificity_evaluable",
    )
    effect_columns = (
        "model_seed",
        "patient_id",
        "selected_slice_id",
        "target_pixel_count",
        "target_feature_count",
        "matched_feature_count",
        "specificity_evaluable",
        "necessity",
        "restoration",
        "specificity",
        "proportion_mediated",
        "full_restore_max_abs_error",
        "target_alpha_1_max_abs_error",
        "spatial_shift_channel_multiset_max_abs_error",
        "receiver_hook_count_delta",
        "clean_anchor_mismatch_count",
    )
    condition_columns = (
        "model_seed",
        "patient_id",
        "selected_slice_id",
        "condition",
        "region",
        "alpha",
        "primary_target_retention",
        "primary_target_correct_count",
        "primary_target_count",
        "observer_persistent_retention",
    )
    operator_columns = (
        "model_seed",
        "status",
        "passed",
        "audited_patient_count",
        "graph_identity_exact",
        "graph_identity_max_abs_error",
        "patient_operator_audit_pass",
    )
    return (
        pd.DataFrame(eligibility_rows, columns=eligibility_columns),
        pd.DataFrame(effect_rows, columns=effect_columns),
        pd.DataFrame(condition_rows, columns=condition_columns),
        pd.DataFrame(operator_rows, columns=operator_columns),
    )


def _primary_statistics(
    effects: pd.DataFrame,
    *,
    seeds: Sequence[int],
    iterations: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for endpoint in ENDPOINTS:
        for model_seed in (int(value) for value in seeds):
            values = effects.loc[
                effects["model_seed"] == model_seed,
                endpoint,
            ].dropna().to_numpy(dtype=np.float64)
            if values.size >= 2:
                result = paired_patient_statistics(
                    values,
                    np.zeros_like(values),
                    iterations=int(iterations),
                    seed=stable_seed(
                        int(bootstrap_seed),
                        "v8_transunet",
                        endpoint,
                        model_seed,
                    ),
                ).to_dict()
            else:
                result = {
                    "patient_count": int(values.size),
                    "mean_difference": np.nan,
                    "median_difference": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "wilcoxon_p": 1.0,
                    "paired_cohens_d": np.nan,
                    "rank_biserial": np.nan,
                }
            rows.append(
                {
                    "endpoint": endpoint,
                    "model_seed": model_seed,
                    **result,
                }
            )
    statistics = pd.DataFrame(rows)
    if len(statistics) != len(ENDPOINTS) * len(tuple(seeds)):
        raise ValueError("V8 primary endpoint family is incomplete")
    statistics["holm_p"] = holm_adjust(
        statistics["wilcoxon_p"].to_numpy(dtype=np.float64)
    )
    return statistics


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    selected = np.asarray(values, dtype=np.float64)
    if selected.ndim != 1 or selected.size < 2 or not np.isfinite(selected).all():
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, selected.size, size=(int(iterations), selected.size))
    means = selected[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _dose_summary(
    conditions: pd.DataFrame,
    *,
    seeds: Sequence[int],
    alphas: Sequence[float],
    iterations: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    ordered_alphas = (0.0, *(float(value) for value in alphas))
    names = (
        "corrupt",
        *(f"restore_target_{value:.2f}" for value in ordered_alphas[1:]),
    )
    summary_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for model_seed in (int(value) for value in seeds):
        selected = conditions[
            (conditions["model_seed"] == model_seed)
            & conditions["condition"].isin(names)
        ]
        pivot = selected.pivot(
            index="patient_id",
            columns="condition",
            values="primary_target_retention",
        )
        complete = pivot.dropna(subset=list(names))
        means: list[float] = []
        for condition_name, alpha in zip(names, ordered_alphas, strict=True):
            values = complete[condition_name].to_numpy(dtype=np.float64)
            low, high = _bootstrap_mean_interval(
                values,
                iterations=int(iterations),
                seed=stable_seed(
                    int(bootstrap_seed),
                    "v8_transunet_dose",
                    model_seed,
                    condition_name,
                ),
            )
            mean = float(values.mean()) if values.size else np.nan
            means.append(mean)
            summary_rows.append(
                {
                    "model_seed": model_seed,
                    "condition": condition_name,
                    "alpha": alpha,
                    "patient_count": int(values.size),
                    "mean_primary_target_retention": mean,
                    "median_primary_target_retention": (
                        float(np.median(values)) if values.size else np.nan
                    ),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
        patient_curves = complete.loc[:, list(names)].to_numpy(dtype=np.float64)
        patient_monotonic = (
            np.all(np.diff(patient_curves, axis=1) >= -1.0e-12, axis=1)
            if patient_curves.size
            else np.empty(0, dtype=bool)
        )
        aggregate_monotonic = bool(
            np.isfinite(means).all()
            and np.all(np.diff(np.asarray(means)) >= -1.0e-12)
        )
        gate_rows.append(
            {
                "model_seed": model_seed,
                "patient_count": int(len(complete)),
                "monotonic": aggregate_monotonic,
                "patient_monotonic_count": int(patient_monotonic.sum()),
                "patient_monotonic_fraction": (
                    float(patient_monotonic.mean())
                    if patient_monotonic.size
                    else np.nan
                ),
            }
        )
    gate_by_seed = {int(row["model_seed"]): row for row in gate_rows}
    for row in summary_rows:
        gate = gate_by_seed[int(row["model_seed"])]
        row["aggregate_curve_monotonic"] = bool(gate["monotonic"])
        row["patient_monotonic_fraction"] = gate["patient_monotonic_fraction"]
    return pd.DataFrame(summary_rows), gate_rows


def _terminal_no_candidate(
    output_root: Path,
    *,
    lock: Mapping[str, Any],
    protocol_lock_sha256: str,
) -> None:
    status = {
        "execution_status": "NOT_RUN_BY_PREREGISTERED_RULE",
        "scientific_status": "NO_REGISTERED_TRANSUNET_CANDIDATE",
        "test_intervention_authorized": False,
        "protocol_lock_sha256": protocol_lock_sha256,
        "candidate_registration": lock.get("candidate_registration"),
        "claim_boundary": (
            "No TransUNet test intervention was performed because validation "
            "data did not yield a preregistered candidate."
        ),
    }
    _write_json_atomic(output_root / "v8_status.json", status)
    _write_json_atomic(output_root / "process_diagnostic_report.json", status)


def _diagnostic_report(
    *,
    lock: Mapping[str, Any],
    eligibility: pd.DataFrame,
    statistics: pd.DataFrame,
    dose_gate_rows: Sequence[Mapping[str, Any]],
    conclusion: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = lock["candidate"]
    endpoint_summary = []
    for row in statistics.to_dict("records"):
        endpoint_summary.append(
            {
                "endpoint": row["endpoint"],
                "model_seed": int(row["model_seed"]),
                "patient_count": int(row["patient_count"]),
                "mean_effect": row["mean_difference"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "holm_p": row["holm_p"],
                "paired_cohens_d": row["paired_cohens_d"],
            }
        )
    return {
        "method": "Pixel Prediction Transition Tracing",
        "diagnostic_object": (
            "Output-anchored persistent pixel-state transitions across ordered "
            "internal checkpoints."
        ),
        "registered_candidate": {
            "path_id": candidate["path_id"],
            "source_node": candidate["path"]["source_node"],
            "receiver_node": candidate["path"]["receiver_node"],
            "source_module_path": candidate["path"]["source_module_path"],
            "receiver_module_path": candidate["path"]["receiver_module_path"],
            "selection_split": "val",
            "confirmation_split": "test",
        },
        "eligible_patient_counts": {
            f"seed_{seed}": int(
                (
                    (eligibility["model_seed"] == seed)
                    & (eligibility["status"] == "PASS")
                ).sum()
            )
            for seed in sorted(eligibility["model_seed"].unique())
        },
        "endpoint_summary": endpoint_summary,
        "dose_summary": [dict(row) for row in dose_gate_rows],
        "conclusion": dict(conclusion),
        "research_use": [
            (
                "Locate where output-persistent corrections are formed or lost "
                "instead of ranking models only by terminal Dice."
            ),
            (
                "Translate a stable process interval into one preregistered "
                "structure-variable intervention for independent confirmation."
            ),
            (
                "Use necessity, restoration, regional specificity and dose "
                "response to decide whether a process observation supports a "
                "bounded mechanism hypothesis."
            ),
            (
                "Retain no-candidate, insufficient-coverage and failed-"
                "intervention outcomes as explicit diagnostic boundaries."
            ),
        ],
        "claim_boundary": (
            "A passing result supports intervention fidelity for the locked "
            "TransUNet path, target cohort and operator; it does not identify a "
            "unique causal mechanism shared by all segmentation networks."
        ),
        "cam_is_part_of_method": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize the locked TransUNet V8 mechanism replication."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/v8_transunet_mechanism_replication.yaml"
        ),
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--protocol-lock", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    config_path = _resolve_under(workspace, args.config)
    config = _load_yaml(config_path)
    lock_path = (
        _resolve_under(workspace, args.protocol_lock)
        if args.protocol_lock is not None
        else (workspace / str(config["protocol_lock"])).resolve()
    )
    lock = _load_json(lock_path)
    if lock.get("configuration_sha256") != sha256_file(config_path):
        raise ValueError("active V8 configuration differs from the protocol lock")
    if lock.get("source_identity", {}).get("source_tree_sha256") != source_tree_sha256(
        workspace
    ):
        raise ValueError("active V8 source tree differs from the protocol lock")
    output_root = (
        _resolve_under(workspace, args.output_root)
        if args.output_root is not None
        else (workspace / str(config["formal_output_root"])).resolve()
    )
    configured_output = (workspace / str(config["formal_output_root"])).resolve()
    if output_root != configured_output:
        raise ValueError("formal V8 summary root must match the locked configuration")
    protocol_lock_sha256 = sha256_file(lock_path)
    if lock.get("status") == "NO_REGISTERED_TRANSUNET_CANDIDATE":
        _terminal_no_candidate(
            output_root,
            lock=lock,
            protocol_lock_sha256=protocol_lock_sha256,
        )
        print("NO_REGISTERED_TRANSUNET_CANDIDATE", flush=True)
        return 0
    if (
        lock.get("status") != "LOCKED_BEFORE_FIRST_TEST_INTERVENTION"
        or lock.get("test_intervention_authorized") is not True
    ):
        raise ValueError("V8 protocol lock does not authorize formal summarization")

    seeds = tuple(int(value) for value in config["model_seeds"])
    expected = int(config["formal"]["expected_patient_count_per_seed"])
    eligibility, effects, conditions, operators = _collect_formal_results(
        output_root,
        lock=lock,
        model=str(config["model"]),
        seeds=seeds,
        expected_patient_count=expected,
        protocol_lock_sha256=protocol_lock_sha256,
    )
    statistics_config = config["statistics"]
    statistics = _primary_statistics(
        effects,
        seeds=seeds,
        iterations=int(statistics_config["bootstrap_iterations"]),
        bootstrap_seed=int(statistics_config["bootstrap_seed"]),
    )
    dose, dose_gate_rows = _dose_summary(
        conditions,
        seeds=seeds,
        alphas=tuple(float(value) for value in config["intervention"]["restoration_alphas"]),
        iterations=int(statistics_config["bootstrap_iterations"]),
        bootstrap_seed=int(statistics_config["bootstrap_seed"]),
    )
    conclusion = evaluate_replication_gate(
        statistics.to_dict("records"),
        dose_rows=dose_gate_rows,
        operator_rows=operators.to_dict("records"),
        minimum_supported_seeds=int(statistics_config["minimum_supported_seeds"]),
        minimum_patients_per_seed=int(
            config["formal"][
                "minimum_evaluable_patients_per_endpoint_per_seed"
            ]
        ),
        alpha=float(statistics_config["alpha"]),
        minimum_standardized_effect=float(
            statistics_config["minimum_standardized_effect"]
        ),
    )
    report = _diagnostic_report(
        lock=lock,
        eligibility=eligibility,
        statistics=statistics,
        dose_gate_rows=dose_gate_rows,
        conclusion=conclusion,
    )
    outputs = {
        "eligibility": "causal_eligibility_audit.parquet",
        "effects": "causal_patient_effects.parquet",
        "conditions": "causal_condition_retention.parquet",
        "statistics": "causal_patient_statistics.parquet",
        "dose": "causal_dose_summary.parquet",
        "operator": "causal_operator_audit.parquet",
        "conclusion": "causal_conclusion_gate.json",
        "diagnostic_report": "process_diagnostic_report.json",
    }
    for frame, filename in (
        (eligibility, outputs["eligibility"]),
        (effects, outputs["effects"]),
        (conditions, outputs["conditions"]),
        (statistics, outputs["statistics"]),
        (dose, outputs["dose"]),
        (operators, outputs["operator"]),
    ):
        _write_parquet_atomic(output_root / filename, frame)
    _write_json_atomic(output_root / outputs["conclusion"], conclusion)
    _write_json_atomic(output_root / outputs["diagnostic_report"], report)
    status = {
        "execution_status": "PASS",
        "scientific_status": conclusion["status"],
        "formal_patient_files": int(len(eligibility)),
        "expected_formal_patient_files": int(expected * len(seeds)),
        "evaluable_patient_rows": int(len(effects)),
        "specificity_evaluable_rows": int(effects["specificity_evaluable"].sum()),
        "status_counts": {
            f"seed_{seed}": eligibility.loc[
                eligibility["model_seed"] == seed,
                "status",
            ].value_counts().sort_index().to_dict()
            for seed in seeds
        },
        "operator_audit_pass": bool(operators["passed"].all()),
        "causal_conclusion_passed": bool(conclusion["passed"]),
        "protocol_lock_sha256": protocol_lock_sha256,
        "outputs": outputs,
    }
    _write_json_atomic(output_root / "v8_status.json", status)
    print(json.dumps(_json_ready(status), ensure_ascii=False, indent=2), flush=True)
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
            ]
        ].to_string(index=False),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
