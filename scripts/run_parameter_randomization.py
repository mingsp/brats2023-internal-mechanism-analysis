from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from pptt.data.brats2d import SliceRecord, discover_slice_records
from pptt.data.patient_splits import patient_id_from_slice
from pptt.experiments.model_matrix import ModelMatrixJob, load_model_matrix
from pptt.experiments.pilot import paired_bootstrap_difference, stable_seed
from pptt.models.adapters import build_adapter
from pptt.observers.cache import build_observer_cache
from pptt.observers.evaluation import (
    patient_probability_metrics,
    predict_observer_probabilities,
)
from pptt.observers.full_cache import (
    load_balanced_node_cache,
    load_indexed_patient_cache,
    save_indexed_patient_cache,
)
from pptt.observers.linear import LinearObserver
from pptt.observers.objective import class_balanced_pixel_weights
from pptt.observers.randomization import (
    extract_randomized_node_features,
    randomize_checkpoint_module,
)
from pptt.observers.trainer import ObserverTrainingConfig, train_observer


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


def _records_by_patient(
    asset_root: Path,
    split_config: dict[str, Any],
) -> dict[str, dict[str, SliceRecord]]:
    grouped: dict[str, dict[str, SliceRecord]] = defaultdict(dict)
    records = discover_slice_records(
        asset_root / str(split_config["image_dir"]),
        asset_root / str(split_config["mask_dir"]),
    )
    for record in records:
        patient = patient_id_from_slice(record.slice_id)
        grouped[patient][record.slice_id] = record
    if len(grouped) != int(split_config["patient_count"]):
        raise ValueError("Randomization source data patient count does not match config")
    return dict(grouped)


def _training_config(
    observer_config: dict[str, Any],
    selected: dict[str, float],
) -> ObserverTrainingConfig:
    base = observer_config["training"]
    return ObserverTrainingConfig(
        l2=float(selected["l2"]),
        temperature=float(selected["temperature"]),
        max_iter=int(base["max_iter"]),
        tolerance_grad=float(base["tolerance_grad"]),
        tolerance_change=float(base["tolerance_change"]),
        history_size=int(base["history_size"]),
    )


def _build_randomized_caches(
    job: ModelMatrixJob,
    *,
    node: str,
    model_config: dict[str, Any],
    split_records: dict[str, dict[str, dict[str, SliceRecord]]],
    job_output: Path,
    observer_config: dict[str, Any],
    device: torch.device,
    resume: bool,
) -> tuple[dict[str, list[Path]], dict[str, Any]]:
    adapter = build_adapter(
        job.model,
        n_channels=int(model_config["n_channels"]),
        num_classes=int(model_config["num_classes"]),
        bilinear=bool(model_config.get("bilinear", False)),
        img_size=int(model_config.get("img_size", 160)),
    )
    checkpoint_digest = adapter.load_checkpoint(job.checkpoint, map_location="cpu")
    randomization_seed = stable_seed(
        int(observer_config["controls"]["parameter_randomization_seed"]),
        job.job_id,
        node,
    )
    before_hash, after_hash = randomize_checkpoint_module(
        adapter,
        node,
        seed=randomization_seed,
    )
    adapter.to(device).eval()
    metadata_path = (
        job_output / "parameter_randomization" / node / "randomization_metadata.json"
    )
    existing_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if resume and metadata_path.is_file()
        else None
    )
    expected_identity = {
        "model": job.model,
        "model_seed": job.seed,
        "node": node,
        "checkpoint_sha256": checkpoint_digest,
        "randomization_seed": randomization_seed,
        "module_state_sha256_before": before_hash,
        "module_state_sha256_after": after_hash,
    }
    if existing_metadata is not None:
        mismatches = {
            key: (existing_metadata.get(key), value)
            for key, value in expected_identity.items()
            if existing_metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Randomization cache identity changed for {job.job_id}/{node}: "
                f"{mismatches}"
            )
    randomized_paths: dict[str, list[Path]] = {}
    extraction_seconds = 0.0
    for split in ("train", "val", "test"):
        source_paths = sorted((job_output / "cache" / split / "patients").glob("*.npz"))
        if not source_paths:
            raise FileNotFoundError(f"Original observer cache is absent for {job.job_id}/{split}")
        destinations: list[Path] = []
        for source_path in source_paths:
            source = load_indexed_patient_cache(source_path)
            destination = (
                job_output
                / "parameter_randomization"
                / node
                / "cache"
                / split
                / "patients"
                / source_path.name
            )
            if resume and destination.is_file():
                if existing_metadata is None:
                    raise ValueError(
                        f"Randomization cache lacks identity metadata: {destination}"
                    )
                destinations.append(destination)
                continue
            start = perf_counter()
            features = extract_randomized_node_features(
                adapter,
                source,
                split_records[split][source.patient_id],
                node=node,
                device=device,
            )
            extraction_seconds += perf_counter() - start
            randomized = replace(source, features={node: features})
            save_indexed_patient_cache(randomized, destination)
            destinations.append(destination)
        randomized_paths[split] = destinations
    adapter.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metadata = {
        "model": job.model,
        "model_seed": job.seed,
        "node": node,
        "checkpoint": job.checkpoint,
        "checkpoint_sha256": checkpoint_digest,
        "scope": "only the declared checkpoint module; all other model state is retained",
        "randomization_seed": randomization_seed,
        "module_state_sha256_before": before_hash,
        "module_state_sha256_after": after_hash,
        "target": "original frozen model final probabilities at identical cached pixels",
        "extraction_seconds": extraction_seconds,
    }
    _write_json(metadata_path, metadata)
    return randomized_paths, metadata


def _fit_and_evaluate_node(
    job: ModelMatrixJob,
    *,
    node: str,
    randomized_paths: dict[str, list[Path]],
    selected: dict[str, float],
    randomized_module_hash: str,
    observer_config: dict[str, Any],
    original_quality: pd.DataFrame,
    job_output: Path,
    device: torch.device,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    capacity = int(observer_config["cache"]["max_rows_per_split_node"])
    sampling_seed = int(observer_config["cache"]["sampling_seed"])
    train_rows = load_balanced_node_cache(
        randomized_paths["train"],
        node=node,
        capacity=capacity,
        seed=sampling_seed,
    )
    weights = class_balanced_pixel_weights(train_rows.target_probabilities)
    cache = build_observer_cache(
        train_rows.features,
        train_rows.target_probabilities,
        patient_indices=train_rows.patient_indices,
        pixel_weights=weights,
    )
    config = _training_config(observer_config, selected)
    observers = []
    fit_reports = []
    for seed in (int(value) for value in observer_config["observer_seeds"]):
        state_path = (
            job_output
            / "parameter_randomization"
            / node
            / "observers"
            / f"seed_{seed}.pt"
        )
        report_path = state_path.with_suffix(".json")
        if resume and state_path.is_file() and report_path.is_file():
            observer = LinearObserver(
                train_rows.features.shape[1],
                train_rows.target_probabilities.shape[1],
            )
            observer.load_state_dict(torch.load(state_path, map_location="cpu"), strict=True)
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
            expected_report_identity = {
                "temperature": config.temperature,
                "l2": config.l2,
                "randomized_module_sha256": randomized_module_hash,
            }
            if any(
                report_payload.get(key) != value
                for key, value in expected_report_identity.items()
            ):
                raise ValueError(f"Randomized observer identity changed: {report_path}")
        else:
            observer, report = train_observer(
                cache,
                config=config,
                seed=seed,
                device=device,
            )
            state_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {name: value.cpu() for name, value in observer.state_dict().items()},
                state_path,
            )
            report_payload = {
                "model": job.model,
                "model_seed": job.seed,
                "node": node,
                "observer_seed": seed,
                "temperature": config.temperature,
                "l2": config.l2,
                "randomized_module_sha256": randomized_module_hash,
                **report.to_dict(),
            }
            _write_json(report_path, report_payload)
        fit_reports.append(report_payload)
        observers.append(observer.to(device).eval())

    randomized_by_patient: dict[str, float] = {}
    patient_rows: list[dict[str, Any]] = []
    for path in randomized_paths["test"]:
        patient = load_indexed_patient_cache(path)
        predictions = [
            predict_observer_probabilities(
                observer,
                patient.features[node],
                temperature=config.temperature,
                device=device,
            )
            for observer in observers
        ]
        ensemble = np.mean(np.stack(predictions), axis=0)
        metric = patient_probability_metrics(
            patient.target_probabilities,
            ensemble,
            np.zeros(patient.row_count, dtype=np.int32),
        )[0]
        randomized_by_patient[patient.patient_id] = float(metric["mean_js_divergence"])

    original = original_quality[
        (original_quality.node == node)
        & (original_quality.split == "test")
        & (original_quality.control == "real")
        & (original_quality.observer_seed == -1)
        & (original_quality.scope == "overall")
    ].set_index("patient_id")
    patient_ids = sorted(randomized_by_patient)
    if set(patient_ids) != set(original.index.tolist()):
        raise ValueError(f"Original and randomized test patients differ at {node}")
    original_values = np.asarray(
        [float(original.loc[patient_id, "mean_js_divergence"]) for patient_id in patient_ids]
    )
    randomized_values = np.asarray(
        [randomized_by_patient[patient_id] for patient_id in patient_ids]
    )
    statistics = observer_config.get("statistics", {})
    estimate = paired_bootstrap_difference(
        original_values,
        randomized_values,
        iterations=int(statistics.get("bootstrap_iterations", 10000)),
        seed=stable_seed(
            int(statistics.get("bootstrap_seed", 20260715)),
            job.job_id,
            node,
            "parameter_randomization",
        ),
    )
    finite_fits = all(
        np.isfinite(
            [
                report["initial_loss"],
                report["final_loss"],
                report["final_gradient_norm"],
                report["relative_loss_change"],
            ]
        ).all()
        for report in fit_reports
    )
    passed = bool(estimate.mean_difference > 0 and estimate.ci_low > 0 and finite_fits)
    for index, patient_id in enumerate(patient_ids):
        patient_rows.append(
            {
                "model": job.model,
                "model_seed": job.seed,
                "node": node,
                "patient_id": patient_id,
                "original_jsd": original_values[index],
                "randomized_jsd": randomized_values[index],
                "randomized_minus_original_jsd": randomized_values[index]
                - original_values[index],
                "mean_difference": estimate.mean_difference,
                "ci_low": estimate.ci_low,
                "ci_high": estimate.ci_high,
                "passed": passed,
            }
        )
    for observer in observers:
        observer.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return patient_rows, {
        "node": node,
        "status": "PASS" if passed else "FAIL",
        "patient_count": len(patient_ids),
        "mean_randomized_minus_original_jsd": estimate.mean_difference,
        "ci_low": estimate.ci_low,
        "ci_high": estimate.ci_high,
        "all_fits_finite": finite_fits,
        "fit_reports": fit_reports,
    }


def _run_job(
    job: ModelMatrixJob,
    *,
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    observer_config: dict[str, Any],
    asset_root: Path,
    output_root: Path,
    device: torch.device,
    resume: bool,
) -> dict[str, Any]:
    job_output = output_root / job.model / f"seed_{job.seed}"
    quality_path = job_output / "observer_quality_by_node.parquet"
    if not quality_path.is_file():
        raise FileNotFoundError(
            f"Run controlled observer evaluation before randomization: {quality_path}"
        )
    nodes = tuple(str(value) for value in model_config["checkpoint_nodes"])
    selected_nodes = (nodes[0], nodes[len(nodes) // 2 - 1], nodes[-1])
    if len(set(selected_nodes)) != 3:
        raise ValueError("Parameter randomization requires three distinct checkpoints")
    selected_hyperparameters = json.loads(
        (job_output / "selected_hyperparameters.json").read_text(encoding="utf-8")
    )
    split_records = {
        split: _records_by_patient(asset_root, data_config["splits"][split])
        for split in ("train", "val", "test")
    }
    original_quality = pd.read_parquet(quality_path)
    patient_rows: list[dict[str, Any]] = []
    node_reports: list[dict[str, Any]] = []
    for node in selected_nodes:
        randomized_paths, metadata = _build_randomized_caches(
            job,
            node=node,
            model_config=model_config,
            split_records=split_records,
            job_output=job_output,
            observer_config=observer_config,
            device=device,
            resume=resume,
        )
        rows, report = _fit_and_evaluate_node(
            job,
            node=node,
            randomized_paths=randomized_paths,
            selected=selected_hyperparameters[node],
            randomized_module_hash=metadata["module_state_sha256_after"],
            observer_config=observer_config,
            original_quality=original_quality,
            job_output=job_output,
            device=device,
            resume=resume,
        )
        patient_rows.extend(rows)
        node_reports.append({**metadata, **report})
    destination = job_output / "parameter_randomization_sensitivity.parquet"
    pd.DataFrame(patient_rows).to_parquet(destination, index=False)
    status = "PASS" if all(row["status"] == "PASS" for row in node_reports) else "FAIL"
    result = {
        "job": job.job_id,
        "status": status,
        "selected_nodes": selected_nodes,
        "nodes": node_reports,
        "output": destination,
    }
    _write_json(job_output / "parameter_randomization_status.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run checkpoint-local parameter-randomization construct checks."
    )
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/brats2023_2d.yaml"),
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results/v1_observers"))
    parser.add_argument("--jobs", nargs="+")
    parser.add_argument("--resume", action="store_true")
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
        else {job.job_id for job in matrix.jobs if job.seed == 42}
    )
    jobs = [job for job in matrix.jobs if job.job_id in selected]
    unknown = selected - {job.job_id for job in matrix.jobs}
    if unknown:
        raise ValueError(f"Unknown model matrix jobs: {sorted(unknown)}")
    if any(job.seed != 42 for job in jobs):
        raise ValueError("Parameter randomization is pre-registered for model seed 42 only")
    observer_config = _load_yaml(args.config)
    data_config = _load_yaml(args.data_config)
    device = torch.device(args.device)
    results = []
    for job in jobs:
        if not job.checkpoint.is_file():
            results.append({"job": job.job_id, "status": "MISSING_CHECKPOINT"})
            continue
        results.append(
            _run_job(
                job,
                model_config=_load_yaml(job.model_config),
                data_config=data_config,
                observer_config=observer_config,
                asset_root=matrix.asset_root,
                output_root=args.output_root.resolve(),
                device=device,
                resume=args.resume,
            )
        )
    complete = len(results) == len(selected) and all(
        row["status"] == "PASS" for row in results
    )
    summary = {"status": "PASS" if complete else "INCOMPLETE", "jobs": results}
    _write_json(
        args.output_root.resolve() / "parameter_randomization_matrix_status.json",
        summary,
    )
    print(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
