from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from pptt.experiments.model_matrix import load_model_matrix
from pptt.experiments.pilot import paired_bootstrap_difference, stable_seed
from pptt.observers.controls import (
    class_reliability,
    select_reliability_threshold,
    transition_reliability_statistics,
)
from pptt.observers.evaluation import predict_observer_probabilities
from pptt.observers.full_cache import load_indexed_patient_cache
from pptt.observers.linear import LinearObserver
from pptt.observers.objective import mean_js_divergence
from pptt.observers.randomization import parameter_randomization_admission


CONTROL_NAMES = ("real", "patient_permutation", "spatial_shift")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _job_selector(value: str) -> str:
    model, separator, seed = value.rpartition(":")
    if not separator:
        raise ValueError(f"Invalid job selector {value!r}")
    return f"{model}/seed_{int(seed)}"


def _load_observers(
    output: Path,
    *,
    nodes: tuple[str, ...],
    controls: tuple[str, ...],
    seeds: tuple[int, ...],
    in_channels: dict[str, int],
    num_classes: int,
    device: torch.device,
) -> dict[str, dict[str, dict[int, LinearObserver]]]:
    observers: dict[str, dict[str, dict[int, LinearObserver]]] = {
        node: {control: {} for control in controls} for node in nodes
    }
    for node in nodes:
        for control in controls:
            for seed in seeds:
                state_path = output / "observers" / node / control / f"seed_{seed}.pt"
                if not state_path.is_file():
                    raise FileNotFoundError(state_path)
                observer = LinearObserver(in_channels[node], num_classes)
                observer.load_state_dict(
                    torch.load(state_path, map_location="cpu"),
                    strict=True,
                )
                observers[node][control][seed] = observer.to(device).eval()
    return observers


def _scope_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
) -> list[dict[str, Any]]:
    target_state = target.argmax(axis=1)
    prediction_state = prediction.argmax(axis=1)
    scopes: list[tuple[str, np.ndarray]] = [
        ("overall", np.ones(target.shape[0], dtype=bool)),
        ("foreground", target_state != 0),
    ]
    scopes.extend(
        (f"class_{class_index}", target_state == class_index)
        for class_index in range(target.shape[1])
    )
    rows: list[dict[str, Any]] = []
    for scope, selected in scopes:
        if selected.any():
            rows.append(
                {
                    "scope": scope,
                    "available": True,
                    "row_count": int(selected.sum()),
                    "mean_js_divergence": mean_js_divergence(
                        target[selected],
                        prediction[selected],
                    ),
                    "hard_agreement": float(
                        np.mean(target_state[selected] == prediction_state[selected])
                    ),
                }
            )
        else:
            rows.append(
                {
                    "scope": scope,
                    "available": False,
                    "row_count": 0,
                    "mean_js_divergence": np.nan,
                    "hard_agreement": np.nan,
                }
            )
    return rows


def _path_runs(
    real_probabilities: dict[str, dict[int, list[np.ndarray]]],
    final_targets: list[np.ndarray],
    *,
    nodes: tuple[str, ...],
    seeds: tuple[int, ...],
) -> np.ndarray:
    final = np.concatenate(final_targets).T
    runs = []
    for seed in seeds:
        stages = [np.concatenate(real_probabilities[node][seed]).T for node in nodes]
        stages.append(final)
        runs.append(np.stack(stages))
    return np.stack(runs)


def _minimum_restart_margin(predictions: list[np.ndarray]) -> np.ndarray:
    values = np.stack(predictions)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[2] < 2:
        raise ValueError("Expected at least two observer runs and two classes")
    ordered = np.sort(values, axis=2)
    margins = ordered[:, :, -1] - ordered[:, :, -2]
    return margins.min(axis=0)


def _evaluate_job(
    *,
    job_id: str,
    model: str,
    model_seed: int,
    model_config: dict[str, Any],
    observer_config: dict[str, Any],
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    nodes = tuple(str(node) for node in model_config["checkpoint_nodes"])
    seeds = tuple(int(seed) for seed in observer_config["observer_seeds"])
    hyperparameters = json.loads(
        (output / "selected_hyperparameters.json").read_text(encoding="utf-8")
    )
    first_test_path = sorted((output / "cache" / "test" / "patients").glob("*.npz"))[0]
    first_cache = load_indexed_patient_cache(first_test_path)
    in_channels = {node: int(first_cache.features[node].shape[1]) for node in nodes}
    observers = _load_observers(
        output,
        nodes=nodes,
        controls=CONTROL_NAMES,
        seeds=seeds,
        in_channels=in_channels,
        num_classes=int(model_config["num_classes"]),
        device=device,
    )
    quality_rows: list[dict[str, Any]] = []
    restart_rows: list[dict[str, Any]] = []
    splits = ("train", "val", "test")
    ensemble_overall: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {
        split: {node: {control: [] for control in CONTROL_NAMES} for node in nodes}
        for split in splits
    }
    real_probabilities: dict[str, dict[str, dict[int, list[np.ndarray]]]] = {
        split: {node: {seed: [] for seed in seeds} for node in nodes}
        for split in splits
    }
    final_targets: dict[str, list[np.ndarray]] = {split: [] for split in splits}
    truth_values: dict[str, list[np.ndarray]] = {split: [] for split in splits}

    for split in splits:
        patient_paths = sorted((output / "cache" / split / "patients").glob("*.npz"))
        if not patient_paths:
            raise FileNotFoundError(f"No cached patients for {job_id}/{split}")
        for path in patient_paths:
            cache = load_indexed_patient_cache(path)
            final_targets[split].append(cache.target_probabilities)
            truth_values[split].append(cache.truth)
            patient_real_states: list[np.ndarray] = []
            for node in nodes:
                temperature = float(hyperparameters[node]["temperature"])
                node_real_states: list[np.ndarray] = []
                evaluated_controls = ("real",) if split == "train" else CONTROL_NAMES
                for control in evaluated_controls:
                    seed_predictions: list[np.ndarray] = []
                    for seed in seeds:
                        prediction = predict_observer_probabilities(
                            observers[node][control][seed],
                            cache.features[node],
                            device=device,
                        )
                        seed_predictions.append(prediction)
                        if control == "real":
                            real_probabilities[split][node][seed].append(prediction)
                        for scope in _scope_metrics(
                            cache.target_probabilities,
                            prediction,
                        ):
                            quality_rows.append(
                                {
                                    "model": model,
                                    "model_seed": model_seed,
                                    "split": split,
                                    "patient_id": cache.patient_id,
                                    "node": node,
                                    "control": control,
                                    "observer_seed": seed,
                                    "temperature": temperature,
                                    "l2": float(hyperparameters[node]["l2"]),
                                    **scope,
                                }
                            )
                    ensemble = np.mean(np.stack(seed_predictions), axis=0)
                    ensemble_metrics = _scope_metrics(
                        cache.target_probabilities,
                        ensemble,
                    )
                    for scope in ensemble_metrics:
                        quality_rows.append(
                            {
                                "model": model,
                                "model_seed": model_seed,
                                "split": split,
                                "patient_id": cache.patient_id,
                                "node": node,
                                "control": control,
                                "observer_seed": -1,
                                "temperature": temperature,
                                "l2": float(hyperparameters[node]["l2"]),
                                **scope,
                            }
                        )
                    overall = next(row for row in ensemble_metrics if row["scope"] == "overall")
                    ensemble_overall[split][node][control].append(
                        {"patient_id": cache.patient_id, **overall}
                    )
                    if control == "real":
                        states = np.stack(
                            [prediction.argmax(axis=1) for prediction in seed_predictions]
                        )
                        node_real_states.append(states)
                        restart_rows.append(
                            {
                                "model": model,
                                "model_seed": model_seed,
                                "split": split,
                                "patient_id": cache.patient_id,
                                "kind": "node",
                                "name": node,
                                "agreement": float(np.all(states == states[:1], axis=0).mean()),
                                "row_count": cache.row_count,
                            }
                        )
                patient_real_states.append(node_real_states[0])
            patient_real_states.append(
                np.repeat(
                    cache.target_probabilities.argmax(axis=1)[None],
                    len(seeds),
                    axis=0,
                )
            )
            stage_names = (*nodes, "final")
            for transition_index in range(len(stage_names) - 1):
                left = patient_real_states[transition_index]
                right = patient_real_states[transition_index + 1]
                agreement = np.all(left == left[:1], axis=0) & np.all(
                    right == right[:1],
                    axis=0,
                )
                restart_rows.append(
                    {
                        "model": model,
                        "model_seed": model_seed,
                        "split": split,
                        "patient_id": cache.patient_id,
                        "kind": "transition",
                        "name": (
                            f"{stage_names[transition_index]}"
                            f"->{stage_names[transition_index + 1]}"
                        ),
                        "agreement": float(agreement.mean()),
                        "row_count": cache.row_count,
                    }
                )

    statistics_config = observer_config.get("statistics", {})
    bootstrap_iterations = int(statistics_config.get("bootstrap_iterations", 10000))
    bootstrap_seed = int(statistics_config.get("bootstrap_seed", 20260715))
    selectivity_rows: list[dict[str, Any]] = []
    node_control_pass: dict[str, bool] = {}
    for node in nodes:
        real = np.asarray(
            [
                row["mean_js_divergence"]
                for row in ensemble_overall["test"][node]["real"]
            ]
        )
        node_passes = []
        for control in ("patient_permutation", "spatial_shift"):
            control_values = np.asarray(
                [
                    row["mean_js_divergence"]
                    for row in ensemble_overall["test"][node][control]
                ]
            )
            estimate = paired_bootstrap_difference(
                real,
                control_values,
                iterations=bootstrap_iterations,
                seed=stable_seed(bootstrap_seed, job_id, node, control),
            )
            passed = bool(estimate.mean_difference > 0 and estimate.ci_low > 0)
            node_passes.append(passed)
            for patient_index, patient_id in enumerate(
                row["patient_id"]
                for row in ensemble_overall["test"][node]["real"]
            ):
                selectivity_rows.append(
                    {
                        "model": model,
                        "model_seed": model_seed,
                        "node": node,
                        "control": control,
                        "patient_id": patient_id,
                        "real_jsd": real[patient_index],
                        "control_jsd": control_values[patient_index],
                        "control_minus_real_jsd": (
                            control_values[patient_index] - real[patient_index]
                        ),
                        "mean_difference": estimate.mean_difference,
                        "ci_low": estimate.ci_low,
                        "ci_high": estimate.ci_high,
                        "passed": passed,
                    }
                )
        node_control_pass[node] = all(node_passes)

    generalization_gap: dict[str, dict[str, float | int]] = {}
    for node in nodes:
        split_means = {
            split: float(
                np.mean(
                    [
                        row["mean_js_divergence"]
                        for row in ensemble_overall[split][node]["real"]
                    ]
                )
            )
            for split in splits
        }
        generalization_gap[node] = {
            "train_patient_count": len(ensemble_overall["train"][node]["real"]),
            "val_patient_count": len(ensemble_overall["val"][node]["real"]),
            "test_patient_count": len(ensemble_overall["test"][node]["real"]),
            "train_mean_jsd": split_means["train"],
            "val_mean_jsd": split_means["val"],
            "test_mean_jsd": split_means["test"],
            "val_minus_train_jsd": split_means["val"] - split_means["train"],
            "test_minus_train_jsd": split_means["test"] - split_means["train"],
        }

    val_runs = _path_runs(
        real_probabilities["val"],
        final_targets["val"],
        nodes=nodes,
        seeds=seeds,
    )
    reliability = observer_config["reliability"]
    threshold_grid = tuple(float(value) for value in reliability["threshold_grid"])
    threshold_selection_error: str | None = None
    try:
        threshold_selection = select_reliability_threshold(
            val_runs,
            grid=threshold_grid,
            minimum_consistency=float(reliability["minimum_transition_consistency"]),
            class_axis=2,
        )
        evaluation_threshold = threshold_selection.threshold
    except ValueError as error:
        threshold_selection = None
        threshold_selection_error = str(error)
        evaluation_threshold = max(threshold_grid)
    test_runs = _path_runs(
        real_probabilities["test"],
        final_targets["test"],
        nodes=nodes,
        seeds=seeds,
    )
    val_statistics = transition_reliability_statistics(
        val_runs,
        threshold=evaluation_threshold,
        class_axis=2,
    )
    test_statistics = transition_reliability_statistics(
        test_runs,
        threshold=evaluation_threshold,
        class_axis=2,
    )
    test_truth = np.concatenate(truth_values["test"])
    class_status = class_reliability(
        test_statistics.reliable_mask,
        test_truth,
        num_classes=int(model_config["num_classes"]),
        minimum_retention=float(reliability["minimum_class_retention"]),
    )
    present_tumor_classes = [
        class_index
        for class_index in range(1, int(model_config["num_classes"]))
        if np.any(test_truth == class_index)
    ]
    stage_names = (*nodes, "final")
    transition_names = [
        f"{left}->{right}"
        for left, right in zip(stage_names[:-1], stage_names[1:], strict=True)
    ]
    failed: dict[str, list[str]] = {node: [] for node in nodes}
    if threshold_selection is None:
        for node in nodes:
            failed[node].append(
                "no validation reliability threshold satisfied the consistency gate"
            )
    for node in nodes:
        if not node_control_pass[node]:
            failed[node].append("real observer did not beat both controls on test patients")
    minimum_consistency = float(reliability["minimum_transition_consistency"])
    for index, transition in enumerate(transition_names):
        reasons = []
        if (
            not np.isfinite(test_statistics.consistency[index])
            or test_statistics.consistency[index] < minimum_consistency
        ):
            reasons.append("restart transition consistency below threshold")
        unavailable_classes = [
            class_index
            for class_index in present_tumor_classes
            if not class_status.available[index, class_index]
        ]
        if unavailable_classes:
            reasons.append(f"tumor-class retention failed for {unavailable_classes}")
        if reasons:
            left, right = transition.split("->")
            if left in failed:
                failed[left].extend(reasons)
            if right in failed:
                failed[right].extend(reasons)
    frozen_report_path = output / "parameter_randomization_frozen_status.json"
    retrained_report_path = output / "parameter_randomization_status.json"
    frozen_report = (
        json.loads(frozen_report_path.read_text(encoding="utf-8"))
        if frozen_report_path.is_file()
        else None
    )
    retrained_report = (
        json.loads(retrained_report_path.read_text(encoding="utf-8"))
        if retrained_report_path.is_file()
        else None
    )
    registered_model_seeds = tuple(
        int(value)
        for value in observer_config["controls"].get(
            "parameter_randomization_model_seeds",
            (42,),
        )
    )
    randomization_admission = parameter_randomization_admission(
        model_seed=model_seed,
        registered_model_seeds=registered_model_seeds,
        frozen_report=frozen_report,
        retrained_report=retrained_report,
    )
    for node, reasons in randomization_admission.failed_nodes.items():
        if node not in failed:
            raise ValueError(f"Randomization report contains unknown node: {node}")
        failed[node].extend(reasons)
    parameter_randomization_status = randomization_admission.primary_status
    failed = {node: sorted(set(reasons)) for node, reasons in failed.items() if reasons}

    for split in splits:
        patient_ids = [
            row["patient_id"] for row in ensemble_overall[split][nodes[0]]["real"]
        ]
        for node in nodes:
            for patient_index, patient_id in enumerate(patient_ids):
                predictions = [
                    real_probabilities[split][node][seed][patient_index]
                    for seed in seeds
                ]
                minimum_margin = _minimum_restart_margin(predictions)
                restart_rows.append(
                    {
                        "model": model,
                        "model_seed": model_seed,
                        "split": split,
                        "patient_id": patient_id,
                        "kind": "node_confidence",
                        "name": node,
                        "agreement": np.nan,
                        "row_count": int(minimum_margin.size),
                        "minimum_margin_mean": float(minimum_margin.mean()),
                        "low_confidence_fraction": float(
                            np.mean(minimum_margin < evaluation_threshold)
                        ),
                        "threshold": evaluation_threshold,
                        "threshold_is_formally_selected": threshold_selection is not None,
                    }
                )

    quality_frame = pd.DataFrame(quality_rows)
    selectivity_frame = pd.DataFrame(selectivity_rows)
    restart_frame = pd.DataFrame(restart_rows)
    quality_frame.to_parquet(output / "observer_quality_by_node.parquet", index=False)
    selectivity_frame.to_parquet(
        output / "observer_control_selectivity.parquet",
        index=False,
    )
    restart_frame.to_parquet(
        output / "observer_restart_agreement.parquet",
        index=False,
    )
    reliability_payload = {
        "selected_on": "val patients",
        "applied_to": "test patients without retuning",
        "selection_status": "PASS" if threshold_selection is not None else "FAIL",
        "selection_error": threshold_selection_error,
        "threshold": (
            threshold_selection.threshold if threshold_selection is not None else None
        ),
        "diagnostic_threshold": evaluation_threshold,
        "selection_consistency": (
            threshold_selection.consistency if threshold_selection is not None else None
        ),
        "selection_retention": (
            threshold_selection.retention if threshold_selection is not None else None
        ),
        "transition_names": transition_names,
        "val_consistency": val_statistics.consistency,
        "val_margin_retention": val_statistics.margin_retention,
        "val_reliable_retention": val_statistics.reliable_retention,
        "test_consistency": test_statistics.consistency,
        "test_margin_retention": test_statistics.margin_retention,
        "test_reliable_retention": test_statistics.reliable_retention,
        "test_class_retention": class_status.retention,
        "test_class_available": class_status.available,
        "present_tumor_classes": present_tumor_classes,
        "generalization_gap_by_node": generalization_gap,
    }
    _write_json(output / "reliability_thresholds.json", reliability_payload)
    _write_json(output / "failed_nodes.json", {"failed_nodes": failed})

    convergence = pd.read_parquet(output / "convergence_audit.parquet")
    convergence_payload = {
        "fit_count": len(convergence),
        "all_finite": bool(
            np.isfinite(
                convergence[
                    [
                        "initial_loss",
                        "final_loss",
                        "final_gradient_norm",
                        "relative_loss_change",
                    ]
                ].to_numpy(dtype=np.float64)
            ).all()
        ),
        "maximum_final_gradient_norm": float(convergence.final_gradient_norm.max()),
        "minimum_relative_loss_change": float(convergence.relative_loss_change.min()),
        "rows": convergence.to_dict(orient="records"),
    }
    _write_json(output / "convergence_audit.json", convergence_payload)
    if (
        failed
        or not convergence_payload["all_finite"]
        or parameter_randomization_status == "FAIL"
    ):
        status = "FAIL"
    elif parameter_randomization_status in {"PASS", "NOT_APPLICABLE"}:
        status = "PASS"
    else:
        status = "PENDING_RANDOMIZATION"
    result = {
        "job": job_id,
        "status": status,
        "failed_nodes": failed,
        "quality_rows": len(quality_rows),
        "selectivity_rows": len(selectivity_rows),
        "restart_rows": len(restart_rows),
        "reliability": reliability_payload,
        "parameter_randomization_status": parameter_randomization_status,
        "parameter_randomization_control_mode": "frozen_observer",
        "parameter_randomization_registered_model_seeds": registered_model_seeds,
        "retrained_randomization_diagnostic_status": (
            randomization_admission.retrained_diagnostic_status
        ),
    }
    _write_json(output / "v1_status.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate controlled observers and fixed reliability gates."
    )
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/observers/default.yaml"),
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results/v1_observers"))
    parser.add_argument("--jobs", nargs="+")
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    matrix = load_model_matrix(
        args.matrix,
        workspace_root=args.workspace_root.resolve(),
        asset_root_override=args.asset_root,
    )
    selected = (
        {_job_selector(value) for value in args.jobs}
        if args.jobs is not None
        else {job.job_id for job in matrix.jobs}
    )
    observer_config = _load_yaml(args.config)
    device = torch.device(args.device)
    results = []
    for job in matrix.jobs:
        if job.job_id not in selected:
            continue
        output = args.output_root.resolve() / job.model / f"seed_{job.seed}"
        if not (output / "training_status.json").is_file():
            results.append({"job": job.job_id, "status": "MISSING_OBSERVERS"})
            continue
        results.append(
            _evaluate_job(
                job_id=job.job_id,
                model=job.model,
                model_seed=job.seed,
                model_config=_load_yaml(job.model_config),
                observer_config=observer_config,
                output=output,
                device=device,
            )
        )
    complete = len(results) == len(selected) and all(row["status"] == "PASS" for row in results)
    summary = {"status": "PASS" if complete else "INCOMPLETE", "jobs": results}
    _write_json(args.output_root.resolve() / "v1_matrix_status.json", summary)
    print(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
