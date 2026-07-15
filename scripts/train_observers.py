from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from pptt.data.brats2d import discover_slice_records
from pptt.data.patient_splits import patient_id_from_slice
from pptt.experiments.model_matrix import ModelMatrixJob, load_model_matrix
from pptt.hooks.checkpoints import checkpoint_sha256
from pptt.models.adapters import build_adapter
from pptt.observers.cache import build_observer_cache
from pptt.observers.controls import patient_permuted_targets
from pptt.observers.evaluation import (
    patient_probability_metrics,
    predict_observer_probabilities,
)
from pptt.observers.full_cache import (
    extract_indexed_patient_cache,
    load_balanced_node_cache,
    save_indexed_patient_cache,
    write_split_cache_index,
)
from pptt.observers.objective import class_balanced_pixel_weights
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


def _convergence_table_row(payload: dict[str, Any]) -> dict[str, Any]:
    row = dict(payload)
    mapping = row.pop("patient_mapping", None)
    row["patient_mapping_json"] = (
        json.dumps(mapping, sort_keys=True) if mapping is not None else None
    )
    return row


def _group_records(records: list[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        grouped[patient_id_from_slice(record.slice_id)].append(record)
    return {
        patient: sorted(items, key=lambda item: item.slice_id)
        for patient, items in grouped.items()
    }


def _job_selector(value: str) -> str:
    model, separator, seed = value.rpartition(":")
    if not separator:
        raise ValueError(f"Invalid job selector {value!r}")
    return f"{model}/seed_{int(seed)}"


def _build_caches(
    job: ModelMatrixJob,
    *,
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    observer_config: dict[str, Any],
    asset_root: Path,
    output: Path,
    device: torch.device,
    resume: bool,
    max_patients_per_split: int | None,
) -> tuple[str, dict[str, list[Path]]]:
    adapter = build_adapter(
        job.model,
        n_channels=int(model_config["n_channels"]),
        num_classes=int(model_config["num_classes"]),
        bilinear=bool(model_config.get("bilinear", False)),
        img_size=int(model_config.get("img_size", 160)),
    )
    checkpoint_digest = adapter.load_checkpoint(job.checkpoint, map_location="cpu")
    if checkpoint_digest != checkpoint_sha256(job.checkpoint):
        raise RuntimeError("Checkpoint hash changed during cache initialization")
    adapter.to(device).eval()
    nodes = tuple(str(node) for node in model_config["checkpoint_nodes"])
    cache_config = observer_config["cache"]
    control_config = observer_config["controls"]
    cache_paths: dict[str, list[Path]] = {}
    profile_rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        split_config = data_config["splits"][split]
        grouped = _group_records(
            discover_slice_records(
                asset_root / str(split_config["image_dir"]),
                asset_root / str(split_config["mask_dir"]),
            )
        )
        if len(grouped) != int(split_config["patient_count"]):
            raise ValueError(f"Unexpected patient count for {split}")
        if max_patients_per_split is not None:
            grouped = {
                patient: grouped[patient]
                for patient in sorted(grouped)[:max_patients_per_split]
            }
        patient_directory = output / "cache" / split / "patients"
        patient_directory.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for patient_id in sorted(grouped):
            destination = patient_directory / f"{patient_id}.npz"
            if resume and destination.is_file():
                paths.append(destination)
                continue
            cache, profile = extract_indexed_patient_cache(
                adapter,
                grouped[patient_id],
                patient_id=patient_id,
                nodes=nodes,
                num_classes=int(model_config["num_classes"]),
                max_lesion_slices=int(
                    cache_config["max_lesion_slices_per_patient"]
                ),
                max_nonlesion_slices=int(
                    cache_config["max_nonlesion_slices_per_patient"]
                ),
                max_pixels_per_class=int(
                    cache_config["max_pixels_per_predicted_class_per_slice"]
                ),
                spatial_shift_y=int(control_config["spatial_shift_y"]),
                spatial_shift_x=int(control_config["spatial_shift_x"]),
                seed=int(cache_config["sampling_seed"]),
                device=device,
            )
            save_indexed_patient_cache(cache, destination)
            profile_rows.append({"split": split, **profile, "bytes": destination.stat().st_size})
            paths.append(destination)
            print(
                f"cache {job.job_id}/{split}/{patient_id}: rows={cache.row_count}",
                flush=True,
            )
        cache_paths[split] = paths
        index_path = output / "cache" / split / "cache_index.parquet"
        if not (resume and index_path.is_file()):
            index_rows = write_split_cache_index(
                paths,
                output=index_path,
                split=split,
                model=job.model,
                model_seed=job.seed,
                checkpoint_sha256=checkpoint_digest,
            )
        else:
            index_rows = int(pd.read_parquet(index_path, columns=["node"]).shape[0])
        _write_json(
            output / "cache" / split / "metadata.json",
            {
                "model": job.model,
                "model_seed": job.seed,
                "checkpoint": job.checkpoint,
                "checkpoint_sha256": checkpoint_digest,
                "nodes": nodes,
                "patient_count": len(paths),
                "index_rows": index_rows,
                "sampling_seed": int(cache_config["sampling_seed"]),
                "smoke_max_patients_per_split": max_patients_per_split,
            },
        )
    _write_json(output / "cache" / "extraction_profiles.json", {"rows": profile_rows})
    adapter.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return checkpoint_digest, cache_paths


def _selection_config(
    base: dict[str, Any],
    *,
    temperature: float,
    l2: float,
) -> ObserverTrainingConfig:
    return ObserverTrainingConfig(
        l2=l2,
        temperature=temperature,
        max_iter=int(base["max_iter"]),
        tolerance_grad=float(base["tolerance_grad"]),
        tolerance_change=float(base["tolerance_change"]),
        history_size=int(base["history_size"]),
    )


def _fit_job_observers(
    job: ModelMatrixJob,
    *,
    model_config: dict[str, Any],
    observer_config: dict[str, Any],
    cache_paths: dict[str, list[Path]],
    output: Path,
    device: torch.device,
    resume: bool,
) -> dict[str, Any]:
    nodes = tuple(str(node) for node in model_config["checkpoint_nodes"])
    capacity = int(observer_config["cache"]["max_rows_per_split_node"])
    selection = observer_config["selection"]
    selection_seed = int(observer_config["selection_seed"])
    observer_seeds = tuple(int(seed) for seed in observer_config["observer_seeds"])
    batch_size = 65536
    selection_rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []
    selected_parameters: dict[str, dict[str, float]] = {}

    for node in nodes:
        train_rows = load_balanced_node_cache(
            cache_paths["train"],
            node=node,
            capacity=capacity,
            seed=int(observer_config["controls"]["patient_permutation_seed"]),
        )
        val_rows = load_balanced_node_cache(
            cache_paths["val"],
            node=node,
            capacity=capacity,
            seed=int(observer_config["controls"]["patient_permutation_seed"]),
        )
        weights = class_balanced_pixel_weights(train_rows.target_probabilities)
        real_cache = build_observer_cache(
            train_rows.features,
            train_rows.target_probabilities,
            patient_indices=train_rows.patient_indices,
            pixel_weights=weights,
        )
        node_candidates: list[dict[str, Any]] = []
        for temperature in (float(value) for value in selection["temperatures"]):
            for l2 in (float(value) for value in selection["l2_values"]):
                candidate_config = _selection_config(
                    observer_config["training"],
                    temperature=temperature,
                    l2=l2,
                )
                observer, report = train_observer(
                    real_cache,
                    config=candidate_config,
                    seed=selection_seed,
                    device=device,
                )
                prediction = predict_observer_probabilities(
                    observer,
                    val_rows.features,
                    temperature=temperature,
                    device=device,
                    batch_size=batch_size,
                )
                patient_metrics = patient_probability_metrics(
                    val_rows.target_probabilities,
                    prediction,
                    val_rows.patient_indices,
                )
                mean_patient_jsd = float(
                    np.mean([row["mean_js_divergence"] for row in patient_metrics])
                )
                candidate = {
                    "model": job.model,
                    "model_seed": job.seed,
                    "node": node,
                    "temperature": temperature,
                    "l2": l2,
                    "mean_patient_jsd": mean_patient_jsd,
                    "mean_patient_hard_agreement": float(
                        np.mean([row["hard_agreement"] for row in patient_metrics])
                    ),
                    **report.to_dict(),
                }
                node_candidates.append(candidate)
                selection_rows.append(candidate)
                observer.to("cpu")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        best_jsd = min(row["mean_patient_jsd"] for row in node_candidates)
        eligible = [
            row
            for row in node_candidates
            if row["mean_patient_jsd"]
            <= best_jsd + float(selection["jsd_tie_tolerance"])
        ]
        chosen = sorted(
            eligible,
            key=lambda row: (-row["l2"], abs(row["temperature"] - 1.0), row["temperature"]),
        )[0]
        selected_parameters[node] = {
            "temperature": float(chosen["temperature"]),
            "l2": float(chosen["l2"]),
            "validation_mean_patient_jsd": float(chosen["mean_patient_jsd"]),
        }

        patient_targets, patient_mapping = patient_permuted_targets(
            train_rows.target_probabilities,
            train_rows.patient_indices,
            seed=int(observer_config["controls"]["patient_permutation_seed"]),
        )
        controls = {
            "real": train_rows.target_probabilities,
            "patient_permutation": patient_targets,
            "spatial_shift": train_rows.spatial_target_probabilities,
        }
        final_config = _selection_config(
            observer_config["training"],
            temperature=chosen["temperature"],
            l2=chosen["l2"],
        )
        for control, targets in controls.items():
            control_weights = class_balanced_pixel_weights(targets)
            cache = build_observer_cache(
                train_rows.features,
                targets,
                patient_indices=train_rows.patient_indices,
                pixel_weights=control_weights,
            )
            for seed in observer_seeds:
                state_path = output / "observers" / node / control / f"seed_{seed}.pt"
                report_path = state_path.with_suffix(".json")
                if resume and state_path.is_file() and report_path.is_file():
                    convergence_rows.append(
                        _convergence_table_row(
                            json.loads(report_path.read_text(encoding="utf-8"))
                        )
                    )
                    continue
                start = perf_counter()
                observer, report = train_observer(
                    cache,
                    config=final_config,
                    seed=seed,
                    device=device,
                )
                elapsed = perf_counter() - start
                state_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {key: value.cpu() for key, value in observer.state_dict().items()},
                    state_path,
                )
                report_payload = {
                    "model": job.model,
                    "model_seed": job.seed,
                    "node": node,
                    "control": control,
                    "temperature": chosen["temperature"],
                    "l2": chosen["l2"],
                    "fit_seconds": elapsed,
                    "patient_mapping": (
                        patient_mapping if control == "patient_permutation" else None
                    ),
                    **report.to_dict(),
                }
                _write_json(report_path, report_payload)
                convergence_rows.append(_convergence_table_row(report_payload))
                observer.to("cpu")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(
                    f"observer {job.job_id}/{node}/{control}/seed{seed}: "
                    f"loss={report.final_loss:.6g}",
                    flush=True,
                )
        del train_rows, val_rows

    pd.DataFrame(selection_rows).to_parquet(
        output / "hyperparameter_selection.parquet",
        index=False,
    )
    pd.DataFrame(convergence_rows).to_parquet(
        output / "convergence_audit.parquet",
        index=False,
    )
    _write_json(output / "selected_hyperparameters.json", selected_parameters)
    finite = all(
        np.isfinite(
            [
                row["initial_loss"],
                row["final_loss"],
                row["final_gradient_norm"],
                row["relative_loss_change"],
            ]
        ).all()
        for row in convergence_rows
    )
    return {
        "status": "PASS" if finite else "FAIL",
        "selected_hyperparameters": selected_parameters,
        "observer_count": len(convergence_rows),
        "finite_optimization": finite,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build fixed caches and train controlled observers for a model matrix."
    )
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/brats2023_2d.yaml"),
    )
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("results/v1_observers"))
    parser.add_argument("--jobs", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-patients-per-split", type=int)
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    matrix = load_model_matrix(
        args.matrix,
        workspace_root=workspace,
        asset_root_override=args.asset_root,
    )
    observer_config = _load_yaml(args.config)
    data_config = _load_yaml(args.data_config)
    selected = (
        {_job_selector(value) for value in args.jobs}
        if args.jobs is not None
        else {job.job_id for job in matrix.jobs}
    )
    unknown = selected - {job.job_id for job in matrix.jobs}
    if unknown:
        raise ValueError(f"Unknown model matrix jobs: {sorted(unknown)}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.max_patients_per_split is not None and args.max_patients_per_split < 2:
        raise ValueError("Smoke mode requires at least two patients per split")
    statuses: list[dict[str, Any]] = []
    for job in matrix.jobs:
        if job.job_id not in selected:
            continue
        if not job.checkpoint.is_file():
            statuses.append(
                {
                    "job": job.job_id,
                    "status": "MISSING_CHECKPOINT",
                    "checkpoint": job.checkpoint,
                }
            )
            continue
        model_config = _load_yaml(job.model_config)
        output = args.output_root.resolve() / job.model / f"seed_{job.seed}"
        output.mkdir(parents=True, exist_ok=True)
        checkpoint_digest, cache_paths = _build_caches(
            job,
            model_config=model_config,
            data_config=data_config,
            observer_config=observer_config,
            asset_root=matrix.asset_root,
            output=output,
            device=device,
            resume=args.resume,
            max_patients_per_split=args.max_patients_per_split,
        )
        observer_status = _fit_job_observers(
            job,
            model_config=model_config,
            observer_config=observer_config,
            cache_paths=cache_paths,
            output=output,
            device=device,
            resume=args.resume,
        )
        status = {
            "job": job.job_id,
            "checkpoint": job.checkpoint,
            "checkpoint_sha256": checkpoint_digest,
            **observer_status,
        }
        _write_json(output / "training_status.json", status)
        statuses.append(status)
    complete = all(row["status"] == "PASS" for row in statuses) and len(statuses) == len(selected)
    summary = {"status": "PASS" if complete else "INCOMPLETE", "jobs": statuses}
    _write_json(args.output_root.resolve() / "matrix_training_status.json", summary)
    print(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
