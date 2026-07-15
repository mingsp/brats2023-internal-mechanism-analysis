from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from io import BytesIO
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from pptt.data.brats2d import discover_slice_records, ensure_chw, normalize_label
from pptt.data.patient_splits import patient_id_from_slice
from pptt.experiments.pilot import (
    BalancedRows,
    PatientRows,
    balance_patient_rows,
    load_patient_rows,
    paired_bootstrap_difference,
    sample_feature_vectors,
    sample_pixel_indices,
    save_patient_rows,
    select_evenly_spaced_patients,
    stable_seed,
)
from pptt.models.adapters import build_adapter
from pptt.models.transunet import load_pretrained_npz
from pptt.observers.cache import (
    build_observer_cache,
    select_patient_slice_indices,
)
from pptt.observers.controls import (
    class_reliability,
    patient_permuted_targets,
    spatially_shifted_targets,
    transition_reliability_statistics,
)
from pptt.observers.linear import LinearObserver
from pptt.observers.objective import (
    class_balanced_pixel_weights,
    mean_js_divergence,
)
from pptt.observers.trainer import ObserverTrainingConfig, train_observer
from pptt.transitions.metrics import confusion_from_labels, metrics_from_confusion
from pptt.transitions.tensors import (
    build_transition_tensor_views,
    confusions_from_transition,
)


CONTROL_NAMES = ("real", "patient_permutation", "spatial_shift")


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


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("The experiment configuration must contain a mapping")
    return payload


def _asset_root(config: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        root = override
    else:
        data = config["data"]
        root = Path(
            os.environ.get(
                str(data["asset_root_environment"]),
                str(data["asset_root_default"]),
            )
        )
    if not root.is_dir():
        raise FileNotFoundError(f"Asset root does not exist: {root}")
    return root.resolve()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _group_records(records: list[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        grouped[patient_id_from_slice(record.slice_id)].append(record)
    return {
        patient: sorted(items, key=lambda item: item.slice_id)
        for patient, items in grouped.items()
    }


def _freeze_pilot_manifest(
    manifest_path: Path,
    patient_ids: tuple[str, ...],
) -> None:
    expected = "".join(f"{patient}\n" for patient in patient_ids)
    if manifest_path.exists():
        current = manifest_path.read_text(encoding="utf-8")
        if current != expected:
            raise ValueError(
                f"Frozen pilot manifest differs from deterministic selection: {manifest_path}"
            )
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(expected, encoding="utf-8")


def _split_patients(
    selected: tuple[str, ...],
    *,
    calibration_count: int,
    selection_count: int,
    evaluation_count: int,
    seed: int,
) -> dict[str, tuple[str, ...]]:
    if calibration_count + selection_count + evaluation_count != len(selected):
        raise ValueError("Pilot role counts must sum to the selected patient count")
    rng = np.random.default_rng(stable_seed(seed, "pilot-role-split"))
    shuffled = np.asarray(selected)[rng.permutation(len(selected))]
    calibration_end = calibration_count
    selection_end = calibration_end + selection_count
    return {
        "calibration": tuple(sorted(shuffled[:calibration_end].tolist())),
        "threshold_selection": tuple(
            sorted(shuffled[calibration_end:selection_end].tolist())
        ),
        "evaluation": tuple(sorted(shuffled[selection_end:].tolist())),
    }


def _choose_patient_slices(
    records: list[Any],
    *,
    max_lesion: int,
    max_nonlesion: int,
) -> list[Any]:
    lesion_areas = np.asarray(
        [
            np.count_nonzero(
                normalize_label(np.load(record.mask_path, allow_pickle=False))
            )
            for record in records
        ],
        dtype=np.int64,
    )
    selected = select_patient_slice_indices(
        lesion_areas,
        max_lesion_slices=max_lesion,
        max_nonlesion_slices=max_nonlesion,
    )
    return [records[int(index)] for index in selected]


def _extract_patient(
    adapter: torch.nn.Module,
    records: list[Any],
    *,
    patient_id: str,
    nodes: tuple[str, ...],
    num_classes: int,
    max_lesion_slices: int,
    max_nonlesion_slices: int,
    max_pixels_per_class: int,
    spatial_shift_y: int,
    spatial_shift_x: int,
    seed: int,
    device: torch.device,
) -> tuple[PatientRows, dict[str, Any]]:
    chosen = _choose_patient_slices(
        records,
        max_lesion=max_lesion_slices,
        max_nonlesion=max_nonlesion_slices,
    )
    feature_parts: dict[str, list[np.ndarray]] = {node: [] for node in nodes}
    target_parts: list[np.ndarray] = []
    spatial_parts: list[np.ndarray] = []
    truth_parts: list[np.ndarray] = []
    slice_profiles: list[dict[str, Any]] = []
    activation_shapes: dict[str, list[int]] = {}

    for record in chosen:
        image = torch.from_numpy(
            ensure_chw(np.load(record.image_path, allow_pickle=False))
        )[None].to(device)
        truth = normalize_label(np.load(record.mask_path, allow_pickle=False))
        _synchronize(device)
        start = perf_counter()
        with torch.inference_mode():
            trace = adapter.trace(image)
            probabilities = torch.softmax(trace.logits, dim=1)
        _synchronize(device)
        elapsed = perf_counter() - start
        if probabilities.shape != (1, num_classes, 160, 160):
            raise ValueError(
                f"Unexpected final probability shape: {tuple(probabilities.shape)}"
            )
        for name, activation in trace.activations.items():
            activation_shapes.setdefault(name, list(activation.shape[1:]))

        flat_indices = sample_pixel_indices(
            probabilities[0],
            max_per_class=max_pixels_per_class,
            seed=stable_seed(seed, patient_id, record.slice_id, "pixels"),
        )
        flat_numpy = flat_indices.cpu().numpy()
        probability_numpy = probabilities[0].detach().cpu().numpy()
        spatial_numpy = spatially_shifted_targets(
            probability_numpy,
            shift_y=spatial_shift_y,
            shift_x=spatial_shift_x,
        )
        target_parts.append(
            probability_numpy.transpose(1, 2, 0).reshape(-1, num_classes)[flat_numpy]
        )
        spatial_parts.append(
            spatial_numpy.transpose(1, 2, 0).reshape(-1, num_classes)[flat_numpy]
        )
        truth_parts.append(truth.reshape(-1)[flat_numpy])
        for node in nodes:
            if node not in trace.activations:
                raise KeyError(f"Trace does not contain requested node {node!r}")
            sampled = sample_feature_vectors(
                trace.activations[node],
                flat_indices,
                output_size=(160, 160),
            )
            feature_parts[node].append(
                sampled.detach().cpu().numpy().astype(np.float16, copy=False)
            )
        slice_profiles.append(
            {
                "slice_id": record.slice_id,
                "lesion_pixels": int(np.count_nonzero(truth)),
                "sampled_pixels": int(flat_indices.numel()),
                "inference_seconds": float(elapsed),
            }
        )
        del trace, probabilities, image

    if not target_parts:
        raise ValueError(f"No slices selected for patient {patient_id}")
    rows = PatientRows(
        patient_id=patient_id,
        features={
            node: np.concatenate(parts, axis=0).astype(np.float16, copy=False)
            for node, parts in feature_parts.items()
        },
        target_probabilities=np.concatenate(target_parts, axis=0).astype(
            np.float32,
            copy=False,
        ),
        spatial_target_probabilities=np.concatenate(spatial_parts, axis=0).astype(
            np.float32,
            copy=False,
        ),
        truth=np.concatenate(truth_parts).astype(np.int64, copy=False),
    )
    cache_bytes = sum(value.nbytes for value in rows.features.values())
    cache_bytes += rows.target_probabilities.nbytes
    cache_bytes += rows.spatial_target_probabilities.nbytes + rows.truth.nbytes
    return rows, {
        "patient_id": patient_id,
        "selected_slice_count": len(chosen),
        "sampled_pixel_count": int(rows.truth.size),
        "inference_seconds": float(
            sum(item["inference_seconds"] for item in slice_profiles)
        ),
        "cache_bytes_uncompressed": int(cache_bytes),
        "feature_cache_bytes_by_node": {
            node: int(value.nbytes) for node, value in rows.features.items()
        },
        "activation_shapes": activation_shapes,
        "slices": slice_profiles,
    }


def _balanced_for_ids(
    rows: dict[str, PatientRows],
    patient_ids: tuple[str, ...],
    *,
    seed: int,
) -> BalancedRows:
    return balance_patient_rows([rows[patient] for patient in patient_ids], seed=seed)


def _predict_probabilities(
    observer: LinearObserver,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    observer = observer.to(device).eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, features.shape[0], batch_size):
            batch = torch.from_numpy(
                features[start : start + batch_size].astype(np.float32, copy=False)
            ).to(device)
            logits = observer(batch[:, :, None, None], output_size=(1, 1))[:, :, 0, 0]
            outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def _patient_metrics(
    balanced: BalancedRows,
    predictions: np.ndarray,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, patient_id in enumerate(balanced.patient_ids):
        selected = balanced.patient_indices == index
        target = balanced.target_probabilities[selected]
        prediction = predictions[selected]
        output.append(
            {
                "patient_id": patient_id,
                "mean_js_divergence": mean_js_divergence(target, prediction),
                "hard_agreement": float(
                    np.mean(target.argmax(axis=1) == prediction.argmax(axis=1))
                ),
                "sampled_pixels": int(selected.sum()),
            }
        )
    return output


def _path_probability_runs(
    predictions: dict[str, dict[str, dict[int, np.ndarray]]],
    balanced: BalancedRows,
    *,
    nodes: tuple[str, ...],
    seeds: tuple[int, ...],
) -> np.ndarray:
    runs = []
    final = balanced.target_probabilities.T
    for seed in seeds:
        stages = [predictions[node]["real"][seed].T for node in nodes]
        stages.append(final)
        runs.append(np.stack(stages, axis=0))
    return np.stack(runs, axis=0)


def _choose_path_threshold(
    probability_runs: np.ndarray,
    *,
    grid: tuple[float, ...],
    minimum_consistency: float,
) -> tuple[float, dict[str, Any], bool]:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for threshold in sorted(set(grid)):
        statistics = transition_reliability_statistics(
            probability_runs,
            threshold=threshold,
            class_axis=2,
        )
        payload = {
            "threshold": float(threshold),
            "consistency": statistics.consistency,
            "margin_retention": statistics.margin_retention,
            "reliable_retention": statistics.reliable_retention,
        }
        candidates.append((float(threshold), payload))
        if (
            np.isfinite(statistics.consistency).all()
            and np.all(statistics.consistency >= minimum_consistency)
        ):
            return float(threshold), payload, True
    valid = [
        candidate
        for candidate in candidates
        if np.isfinite(np.asarray(candidate[1]["consistency"])).all()
    ]
    if not valid:
        raise ValueError("No reliability threshold retained any transition pixels")
    fallback = max(
        valid,
        key=lambda item: (
            float(np.min(item[1]["consistency"])),
            float(np.mean(item[1]["reliable_retention"])),
        ),
    )
    return fallback[0], fallback[1], False


def _metric_reconstruction(
    truth: np.ndarray,
    states: list[np.ndarray],
    reliable_mask: np.ndarray,
    *,
    num_classes: int,
    transition_names: tuple[str, ...],
    output: Path,
) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    archive: dict[str, np.ndarray] = {}
    maximum_error = 0.0
    for index, name in enumerate(transition_names):
        views = build_transition_tensor_views(
            truth,
            states[index],
            states[index + 1],
            reliable_mask[index],
            num_classes,
        )
        archive[f"full_{index}"] = views.full
        archive[f"reliable_{index}"] = views.reliable
        for view_name, tensor, mask in (
            ("full", views.full, None),
            ("reliable", views.reliable, reliable_mask[index]),
        ):
            tensor_before, tensor_after = confusions_from_transition(tensor)
            direct_before = confusion_from_labels(
                truth,
                states[index],
                num_classes,
                mask=mask,
            )
            direct_after = confusion_from_labels(
                truth,
                states[index + 1],
                num_classes,
                mask=mask,
            )
            count_error = max(
                float(np.max(np.abs(tensor_before - direct_before))),
                float(np.max(np.abs(tensor_after - direct_after))),
            )
            before_metrics = metrics_from_confusion(tensor_before)
            after_metrics = metrics_from_confusion(tensor_after)
            direct_before_metrics = metrics_from_confusion(direct_before)
            direct_after_metrics = metrics_from_confusion(direct_after)
            metric_error = max(
                float(
                    np.max(
                        np.abs(
                            getattr(before_metrics, metric)
                            - getattr(direct_before_metrics, metric)
                        )
                    )
                )
                for metric in before_metrics.as_dict()
            )
            metric_error = max(
                metric_error,
                max(
                    float(
                        np.max(
                            np.abs(
                                getattr(after_metrics, metric)
                                - getattr(direct_after_metrics, metric)
                            )
                        )
                    )
                    for metric in after_metrics.as_dict()
                ),
            )
            maximum_error = max(maximum_error, count_error, metric_error)
            rows.append(
                {
                    "transition": name,
                    "view": view_name,
                    "pixel_count": int(tensor.sum()),
                    "count_reconstruction_error": count_error,
                    "metric_reconstruction_error": metric_error,
                    "dice_before": before_metrics.dice.tolist(),
                    "dice_after": after_metrics.dice.tolist(),
                    "dice_delta": (
                        after_metrics.dice - before_metrics.dice
                    ).tolist(),
                }
            )
    np.savez_compressed(output / "transition_tensors.npz", **archive)
    return rows, maximum_error


def _measure_patient_transition_archive_bytes(
    truth: np.ndarray,
    states: list[np.ndarray],
    reliable_mask: np.ndarray,
    patient_indices: np.ndarray,
    *,
    num_classes: int,
) -> list[int]:
    measured: list[int] = []
    for patient_index in np.unique(patient_indices):
        selected = patient_indices == patient_index
        arrays: dict[str, np.ndarray] = {}
        for transition_index in range(len(states) - 1):
            views = build_transition_tensor_views(
                truth[selected],
                states[transition_index][selected],
                states[transition_index + 1][selected],
                reliable_mask[transition_index, selected],
                num_classes,
            )
            arrays[f"full_{transition_index}"] = views.full
            arrays[f"reliable_{transition_index}"] = views.reliable
        stream = BytesIO()
        np.savez_compressed(stream, **arrays)
        measured.append(stream.tell())
    return measured


def _profile_secondary_model(
    config: dict[str, Any],
    *,
    asset_root: Path,
    records: list[Any],
    device: torch.device,
) -> dict[str, Any] | None:
    secondary = config["resource_extrapolation"]["secondary_model"]
    if not bool(secondary.get("enabled", False)):
        return None
    adapter = build_adapter(
        str(secondary["name"]),
        n_channels=int(config["model"]["n_channels"]),
        num_classes=int(config["model"]["num_classes"]),
        img_size=160,
    )
    report = load_pretrained_npz(
        adapter.model,
        asset_root / str(secondary["pretrained_npz"]),
    )
    adapter.to(device).eval()
    warmup_record = records[0]
    warmup_image = torch.from_numpy(
        ensure_chw(np.load(warmup_record.image_path, allow_pickle=False))
    )[None].to(device)
    with torch.inference_mode():
        adapter.trace(warmup_image)
    _synchronize(device)
    del warmup_image
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    elapsed: list[float] = []
    shapes: dict[str, list[int]] = {}
    for record in records[: int(secondary["sample_slices"])]:
        image = torch.from_numpy(
            ensure_chw(np.load(record.image_path, allow_pickle=False))
        )[None].to(device)
        _synchronize(device)
        start = perf_counter()
        with torch.inference_mode():
            trace = adapter.trace(image)
        _synchronize(device)
        elapsed.append(perf_counter() - start)
        shapes = {name: list(value.shape[1:]) for name, value in trace.activations.items()}
        del trace, image
    peak = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    adapter.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "name": str(secondary["name"]),
        "pretrained_load": report.to_dict(),
        "sample_slices": len(elapsed),
        "seconds_per_slice": elapsed,
        "mean_seconds_per_slice": float(np.mean(elapsed)),
        "peak_cuda_memory_bytes": peak,
        "activation_shapes": shapes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the locked 16-patient observer and transition gates."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/v1_observer.yaml"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/v1_pilot"))
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--nodes", nargs="+")
    parser.add_argument("--patients", type=Path)
    parser.add_argument("--observer-seeds", nargs="+", type=int)
    args = parser.parse_args()

    config = _load_config(args.config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    asset_root = _asset_root(config, args.asset_root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    data_config = config["data"]
    records = discover_slice_records(
        asset_root / str(data_config["image_dir"]),
        asset_root / str(data_config["mask_dir"]),
    )
    grouped = _group_records(records)
    if len(grouped) != int(data_config["expected_patient_count"]):
        raise ValueError(
            f"Expected {data_config['expected_patient_count']} test patients, "
            f"found {len(grouped)}"
        )

    pilot_config = config["pilot"]
    selected_patients = select_evenly_spaced_patients(
        grouped,
        count=int(pilot_config["patient_count"]),
        seed=int(config["seed"]),
    )
    manifest_path = args.patients or Path(str(pilot_config["manifest"]))
    _freeze_pilot_manifest(manifest_path, selected_patients)
    roles = _split_patients(
        selected_patients,
        calibration_count=int(pilot_config["calibration_patients"]),
        selection_count=int(pilot_config["threshold_selection_patients"]),
        evaluation_count=int(pilot_config["evaluation_patients"]),
        seed=int(config["seed"]),
    )
    _write_json(
        output / "pilot_split.json",
        {
            "selection_method": "seeded evenly spaced selection over sorted test patients",
            "seed": int(config["seed"]),
            "selected_patients": selected_patients,
            "roles": roles,
        },
    )

    model_config = config["model"]
    model_name = args.model or str(model_config["name"])
    model_seed = (
        int(args.model_seed) if args.model_seed is not None else int(model_config["seed"])
    )
    nodes = tuple(args.nodes or (str(node) for node in model_config["nodes"]))
    adapter = build_adapter(
        model_name,
        n_channels=int(model_config["n_channels"]),
        num_classes=int(model_config["num_classes"]),
        bilinear=bool(model_config["bilinear"]),
    )
    checkpoint_path = args.checkpoint or Path(str(model_config["checkpoint"]))
    checkpoint_sha256 = adapter.load_checkpoint(
        checkpoint_path if checkpoint_path.is_absolute() else asset_root / checkpoint_path,
        map_location="cpu",
    )
    adapter.to(device).eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    cache_directory = output / "patient_cache"
    patient_rows: dict[str, PatientRows] = {}
    extraction_profiles: list[dict[str, Any]] = []
    for patient_id in selected_patients:
        cache_path = cache_directory / f"{patient_id}.npz"
        if args.reuse_cache and cache_path.is_file():
            rows = load_patient_rows(cache_path)
            if tuple(rows.features) != nodes or rows.patient_id != patient_id:
                raise ValueError(f"Incompatible cached patient rows: {cache_path}")
            profile_path = cache_path.with_suffix(".profile.json")
            if not profile_path.is_file():
                raise FileNotFoundError(
                    f"Cache reuse requires its measured profile: {profile_path}"
                )
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            profile["feature_cache_bytes_by_node"] = {
                node: int(value.nbytes) for node, value in rows.features.items()
            }
            profile["cache_bytes_uncompressed"] = int(
                sum(value.nbytes for value in rows.features.values())
                + rows.target_probabilities.nbytes
                + rows.spatial_target_probabilities.nbytes
                + rows.truth.nbytes
            )
            profile["sampled_pixel_count"] = int(rows.truth.size)
            profile["cache_file_bytes"] = cache_path.stat().st_size
            profile["reused_cache"] = True
            _write_json(profile_path, profile)
        else:
            rows, profile = _extract_patient(
                adapter,
                grouped[patient_id],
                patient_id=patient_id,
                nodes=nodes,
                num_classes=int(model_config["num_classes"]),
                max_lesion_slices=int(pilot_config["max_lesion_slices_per_patient"]),
                max_nonlesion_slices=int(
                    pilot_config["max_nonlesion_slices_per_patient"]
                ),
                max_pixels_per_class=int(
                    pilot_config["max_pixels_per_predicted_class_per_slice"]
                ),
                spatial_shift_y=int(config["controls"]["spatial_shift_y"]),
                spatial_shift_x=int(config["controls"]["spatial_shift_x"]),
                seed=int(config["seed"]),
                device=device,
            )
            save_patient_rows(rows, cache_path)
            profile["cache_file_bytes"] = cache_path.stat().st_size
            profile["reused_cache"] = False
            _write_json(cache_path.with_suffix(".profile.json"), profile)
        patient_rows[patient_id] = rows
        extraction_profiles.append(profile)
        print(
            f"cached {patient_id}: rows={rows.truth.size}, "
            f"file={cache_path.stat().st_size / 2**20:.2f} MiB",
            flush=True,
        )

    baseline_peak = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    adapter.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    balanced = {
        role: _balanced_for_ids(
            patient_rows,
            patient_ids,
            seed=stable_seed(config["seed"], role),
        )
        for role, patient_ids in roles.items()
    }
    calibration = balanced["calibration"]
    patient_targets, patient_mapping = patient_permuted_targets(
        calibration.target_probabilities,
        calibration.patient_indices,
        seed=int(config["controls"]["patient_permutation_seed"]),
    )
    training_targets = {
        "real": calibration.target_probabilities,
        "patient_permutation": patient_targets,
        "spatial_shift": calibration.spatial_target_probabilities,
    }
    real_weights = class_balanced_pixel_weights(calibration.target_probabilities)
    observer_config = config["observer"]
    training_config = ObserverTrainingConfig(
        l2=float(observer_config["l2"]),
        temperature=float(observer_config["temperature"]),
        max_iter=int(observer_config["max_iter"]),
        tolerance_grad=float(observer_config["tolerance_grad"]),
        tolerance_change=float(observer_config["tolerance_change"]),
        history_size=int(observer_config["history_size"]),
    )
    observer_seeds = tuple(
        args.observer_seeds
        if args.observer_seeds is not None
        else (int(seed) for seed in observer_config["seeds"])
    )
    prediction_batch_size = int(observer_config["prediction_batch_size"])
    predictions: dict[
        str, dict[str, dict[str, dict[int, np.ndarray]]]
    ] = {
        role: {
            node: {control: {} for control in CONTROL_NAMES} for node in nodes
        }
        for role in ("threshold_selection", "evaluation")
    }
    table_rows: list[dict[str, Any]] = []
    fit_seconds: list[float] = []
    state_directory = output / "observer_states"
    state_directory.mkdir(parents=True, exist_ok=True)

    for node in nodes:
        for control in CONTROL_NAMES:
            cache = build_observer_cache(
                calibration.features[node],
                training_targets[control],
                patient_indices=calibration.patient_indices,
                pixel_weights=real_weights,
            )
            for observer_seed in observer_seeds:
                _synchronize(device)
                start = perf_counter()
                observer, report = train_observer(
                    cache,
                    config=training_config,
                    seed=observer_seed,
                    device=device,
                )
                _synchronize(device)
                elapsed = perf_counter() - start
                fit_seconds.append(elapsed)
                state_path = (
                    state_directory / f"{node}_{control}_seed{observer_seed}.pt"
                )
                torch.save(
                    {key: value.detach().cpu() for key, value in observer.state_dict().items()},
                    state_path,
                )
                table_rows.append(
                    {
                        "row_type": "optimization",
                        "node": node,
                        "control": control,
                        "seed": observer_seed,
                        "split": "calibration",
                        "patient_id": None,
                        "fit_seconds": elapsed,
                        **report.to_dict(),
                    }
                )
                for role in ("threshold_selection", "evaluation"):
                    role_rows = balanced[role]
                    probability = _predict_probabilities(
                        observer,
                        role_rows.features[node],
                        device=device,
                        batch_size=prediction_batch_size,
                    )
                    predictions[role][node][control][observer_seed] = probability
                    for patient_metric in _patient_metrics(role_rows, probability):
                        table_rows.append(
                            {
                                "row_type": "patient_evaluation",
                                "node": node,
                                "control": control,
                                "seed": observer_seed,
                                "split": role,
                                "fit_seconds": elapsed,
                                **patient_metric,
                            }
                        )
                observer.to("cpu")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(
                    f"fit {node}/{control}/seed{observer_seed}: "
                    f"loss={report.final_loss:.6g}, time={elapsed:.2f}s",
                    flush=True,
                )

    ensemble_metrics: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {
        role: {node: {} for node in nodes}
        for role in ("threshold_selection", "evaluation")
    }
    for role in ensemble_metrics:
        for node in nodes:
            for control in CONTROL_NAMES:
                ensemble = np.mean(
                    np.stack(
                        [
                            predictions[role][node][control][seed]
                            for seed in observer_seeds
                        ]
                    ),
                    axis=0,
                )
                metrics = _patient_metrics(balanced[role], ensemble)
                ensemble_metrics[role][node][control] = metrics
                for patient_metric in metrics:
                    table_rows.append(
                        {
                            "row_type": "patient_evaluation",
                            "node": node,
                            "control": control,
                            "seed": -1,
                            "split": role,
                            "fit_seconds": None,
                            **patient_metric,
                        }
                    )

    control_gates: dict[str, Any] = {}
    bootstrap_config = config["statistics"]
    for node in nodes:
        real = np.asarray(
            [
                item["mean_js_divergence"]
                for item in ensemble_metrics["evaluation"][node]["real"]
            ]
        )
        node_gate: dict[str, Any] = {}
        for control in ("patient_permutation", "spatial_shift"):
            control_values = np.asarray(
                [
                    item["mean_js_divergence"]
                    for item in ensemble_metrics["evaluation"][node][control]
                ]
            )
            estimate = paired_bootstrap_difference(
                real,
                control_values,
                iterations=int(bootstrap_config["bootstrap_iterations"]),
                seed=stable_seed(
                    bootstrap_config["bootstrap_seed"], node, control
                ),
            )
            node_gate[control] = {
                **asdict(estimate),
                "passed": bool(estimate.mean_difference > 0 and estimate.ci_low > 0),
                "real_patient_jsd": real,
                "control_patient_jsd": control_values,
            }
        node_gate["passed"] = all(
            node_gate[control]["passed"]
            for control in ("patient_permutation", "spatial_shift")
        )
        control_gates[node] = node_gate

    selection_runs = _path_probability_runs(
        predictions["threshold_selection"],
        balanced["threshold_selection"],
        nodes=nodes,
        seeds=observer_seeds,
    )
    reliability_config = config["reliability"]
    selected_threshold, selection_reliability, selection_gate = _choose_path_threshold(
        selection_runs,
        grid=tuple(float(value) for value in reliability_config["threshold_grid"]),
        minimum_consistency=float(
            reliability_config["minimum_transition_consistency"]
        ),
    )
    evaluation_runs = _path_probability_runs(
        predictions["evaluation"],
        balanced["evaluation"],
        nodes=nodes,
        seeds=observer_seeds,
    )
    evaluation_reliability = transition_reliability_statistics(
        evaluation_runs,
        threshold=selected_threshold,
        class_axis=2,
    )
    truth = balanced["evaluation"].truth
    class_gate = class_reliability(
        evaluation_reliability.reliable_mask,
        truth,
        num_classes=int(model_config["num_classes"]),
        minimum_retention=float(reliability_config["minimum_class_retention"]),
    )
    present_tumor_classes = [
        class_index
        for class_index in range(1, int(model_config["num_classes"]))
        if np.any(truth == class_index)
    ]
    class_retention_passed = bool(
        present_tumor_classes
        and np.all(class_gate.available[:, present_tumor_classes])
    )
    evaluation_consistency_passed = bool(
        np.isfinite(evaluation_reliability.consistency).all()
        and np.all(
            evaluation_reliability.consistency
            >= float(reliability_config["minimum_transition_consistency"])
        )
    )

    canonical_probabilities = [
        np.mean(
            np.stack(
                [predictions["evaluation"][node]["real"][seed] for seed in observer_seeds]
            ),
            axis=0,
        )
        for node in nodes
    ]
    canonical_probabilities.append(balanced["evaluation"].target_probabilities)
    canonical_states = [probability.argmax(axis=1) for probability in canonical_probabilities]
    transition_names = tuple(
        f"{left}->{right}"
        for left, right in zip((*nodes, "final")[:-1], (*nodes, "final")[1:])
    )
    transition_rows, maximum_metric_error = _metric_reconstruction(
        truth,
        canonical_states,
        evaluation_reliability.reliable_mask,
        num_classes=int(model_config["num_classes"]),
        transition_names=transition_names,
        output=output,
    )
    patient_transition_archive_bytes = _measure_patient_transition_archive_bytes(
        truth,
        canonical_states,
        evaluation_reliability.reliable_mask,
        balanced["evaluation"].patient_indices,
        num_classes=int(model_config["num_classes"]),
    )
    for item in transition_rows:
        table_rows.append(
            {
                "row_type": "transition_summary",
                "node": item["transition"],
                "control": item["view"],
                "seed": -1,
                "split": "evaluation",
                "patient_id": None,
                **item,
            }
        )

    table = pd.DataFrame(table_rows)
    table.to_parquet(output / "observer_node_controls.parquet", index=False)

    optimization_rows = [row for row in table_rows if row["row_type"] == "optimization"]
    optimization_values = np.asarray(
        [
            [
                row["initial_loss"],
                row["final_loss"],
                row["final_data_loss"],
                row["final_regularization"],
                row["final_gradient_norm"],
                row["relative_loss_change"],
            ]
            for row in optimization_rows
        ],
        dtype=np.float64,
    )
    optimization_passed = bool(
        np.isfinite(optimization_values).all()
        and all(row["final_loss"] <= row["initial_loss"] for row in optimization_rows)
    )
    metric_passed = bool(
        maximum_metric_error
        < float(bootstrap_config["maximum_metric_reconstruction_error"])
    )
    all_controls_passed = all(control_gates[node]["passed"] for node in nodes)
    reliability_passed = bool(
        selection_gate
        and evaluation_consistency_passed
        and class_retention_passed
    )
    gate_report = {
        "experiment": str(config["experiment"]),
        "status": "PASS"
        if all_controls_passed
        and reliability_passed
        and optimization_passed
        and metric_passed
        else "FAIL",
        "model": {
            "name": model_name,
            "seed": model_seed,
            "checkpoint_sha256": checkpoint_sha256,
            "nodes": nodes,
        },
        "control_separation": {
            "passed": all_controls_passed,
            "definition": "control minus real JSD; positive paired bootstrap CI",
            "nodes": control_gates,
        },
        "reliability": {
            "passed": reliability_passed,
            "selected_threshold": selected_threshold,
            "selection_gate_passed": selection_gate,
            "selection": selection_reliability,
            "evaluation_consistency_passed": evaluation_consistency_passed,
            "evaluation_consistency": evaluation_reliability.consistency,
            "evaluation_margin_retention": evaluation_reliability.margin_retention,
            "evaluation_reliable_retention": evaluation_reliability.reliable_retention,
            "present_tumor_classes": present_tumor_classes,
            "class_retention": class_gate.retention,
            "class_available": class_gate.available,
            "minimum_consistency": float(
                reliability_config["minimum_transition_consistency"]
            ),
            "minimum_class_retention": float(
                reliability_config["minimum_class_retention"]
            ),
        },
        "optimization": {
            "passed": optimization_passed,
            "fit_count": len(optimization_rows),
            "maximum_final_gradient_norm": float(
                max(row["final_gradient_norm"] for row in optimization_rows)
            ),
            "minimum_relative_loss_change": float(
                min(row["relative_loss_change"] for row in optimization_rows)
            ),
        },
        "metric_reconstruction": {
            "passed": metric_passed,
            "maximum_error": maximum_metric_error,
            "required_maximum": float(
                bootstrap_config["maximum_metric_reconstruction_error"]
            ),
            "transitions": transition_rows,
        },
        "patient_permutation_mapping": patient_mapping,
    }
    _write_json(output / "gate_report.json", gate_report)

    secondary_profile = _profile_secondary_model(
        config,
        asset_root=asset_root,
        records=grouped[selected_patients[0]],
        device=device,
    )
    measured_patient_seconds = [
        float(profile["inference_seconds"])
        for profile in extraction_profiles
        if "inference_seconds" in profile
    ]
    measured_cache_bytes = [
        int(profile["cache_file_bytes"]) for profile in extraction_profiles
    ]
    extrapolation = config["resource_extrapolation"]
    patient_scale = int(data_config["expected_patient_count"]) / len(selected_patients)
    shape_profiles = [
        profile["activation_shapes"]
        for profile in extraction_profiles
        if profile.get("activation_shapes")
    ]
    if not shape_profiles:
        raise ValueError("Resource extrapolation requires measured activation shapes")
    activation_shapes = shape_profiles[0]
    selected_channels = sum(int(activation_shapes[node][0]) for node in nodes)
    all_channels = sum(int(shape[0]) for shape in activation_shapes.values())
    channel_scale = all_channels / selected_channels
    full_model_seed_count = int(extrapolation["full_model_seed_count"])
    unet_model_seed_count = int(extrapolation["unet_model_seed_count"])
    transunet_model_seed_count = int(extrapolation["transunet_model_seed_count"])
    if unet_model_seed_count + transunet_model_seed_count != full_model_seed_count:
        raise ValueError("Resource model seed counts are inconsistent")
    mean_patient_inference = (
        float(np.mean(measured_patient_seconds)) if measured_patient_seconds else None
    )
    median_fit = float(np.median(fit_seconds))
    mean_selected_slices = float(
        np.mean(
            [
                int(profile["selected_slice_count"])
                for profile in extraction_profiles
                if "selected_slice_count" in profile
            ]
        )
    )
    transunet_patient_seconds = (
        float(secondary_profile["mean_seconds_per_slice"]) * mean_selected_slices
        if secondary_profile is not None
        else None
    )
    full_inference = None
    if mean_patient_inference is not None:
        per_patient_models = mean_patient_inference * full_model_seed_count
        if transunet_patient_seconds is not None:
            per_patient_models = (
                mean_patient_inference * unet_model_seed_count
                + transunet_patient_seconds * transunet_model_seed_count
            )
        full_inference = per_patient_models * int(data_config["expected_patient_count"])
    full_fit_count = (
        full_model_seed_count
        * int(extrapolation["full_node_count"])
        * int(extrapolation["control_count"])
        * int(extrapolation["observer_seed_count"])
    )
    measured_feature_bytes_by_node = {
        node: [
            int(profile["feature_cache_bytes_by_node"][node])
            for profile in extraction_profiles
        ]
        for node in nodes
    }
    mean_sampled_rows = float(
        np.mean([int(profile["sampled_pixel_count"]) for profile in extraction_profiles])
    )
    baseline_channels = {
        node: int(shape[0]) for node, shape in activation_shapes.items()
    }
    transunet_shapes = (
        secondary_profile["activation_shapes"] if secondary_profile is not None else {}
    )
    transunet_channels = {
        node: int(shape[0]) for node, shape in transunet_shapes.items()
    }
    rows_in_full_test = mean_sampled_rows * int(data_config["expected_patient_count"])
    baseline_feature_bytes = rows_in_full_test * sum(baseline_channels.values()) * 2
    transunet_feature_bytes = (
        rows_in_full_test * sum(transunet_channels.values()) * 2
        if transunet_channels
        else baseline_feature_bytes
    )
    target_bytes_per_row = int(model_config["num_classes"]) * 4 * 2 + 8
    target_bytes = rows_in_full_test * target_bytes_per_row * full_model_seed_count
    v1_uncompressed_cache = (
        baseline_feature_bytes * unet_model_seed_count
        + transunet_feature_bytes * transunet_model_seed_count
        + target_bytes
    )
    measured_uncompressed = sum(
        int(profile["cache_bytes_uncompressed"]) for profile in extraction_profiles
    )
    measured_compressed = sum(measured_cache_bytes)
    compression_ratio = measured_compressed / measured_uncompressed
    transition_scale = (
        int(extrapolation["full_transition_count"]) / len(transition_names)
    )
    mean_patient_transition_bytes = float(np.mean(patient_transition_archive_bytes))
    v3_disk_bytes = (
        mean_patient_transition_bytes
        * transition_scale
        * int(data_config["expected_patient_count"])
        * unet_model_seed_count
    )
    v5_disk_bytes = (
        mean_patient_transition_bytes
        * transition_scale
        * int(data_config["expected_patient_count"])
        * transunet_model_seed_count
    )
    resource_profile = {
        "measured": {
            "baseline_patient_profiles": extraction_profiles,
            "baseline_peak_cuda_memory_bytes": baseline_peak,
            "observer_fit_seconds": fit_seconds,
            "observer_fit_seconds_median": median_fit,
            "observer_fit_seconds_mean": float(np.mean(fit_seconds)),
            "patient_cache_file_bytes": measured_cache_bytes,
            "patient_cache_uncompressed_bytes": [
                int(profile["cache_bytes_uncompressed"])
                for profile in extraction_profiles
            ],
            "feature_cache_bytes_by_node": measured_feature_bytes_by_node,
            "transition_archive_bytes_per_evaluation_patient": (
                patient_transition_archive_bytes
            ),
            "secondary_model": secondary_profile,
        },
        "extrapolation_assumptions": {
            **extrapolation,
            "patient_scale": patient_scale,
            "selected_node_channels": selected_channels,
            "all_node_channels": all_channels,
            "cache_channel_scale": channel_scale,
            "full_model_seed_count": full_model_seed_count,
            "unet_model_seed_count": unet_model_seed_count,
            "transunet_model_seed_count": transunet_model_seed_count,
            "mean_selected_slices_per_patient": mean_selected_slices,
            "mean_sampled_rows_per_patient": mean_sampled_rows,
            "measured_cache_compression_ratio": compression_ratio,
            "estimated_transunet_seconds_per_patient": transunet_patient_seconds,
            "rule": "linear extrapolation from measured pilot inference, cache, and fit costs",
        },
        "estimated": {
            "v1_full_inference_seconds": full_inference,
            "v1_full_observer_fit_count": full_fit_count,
            "v1_full_observer_fit_seconds": median_fit * full_fit_count,
            "v1_full_total_seconds": (
                full_inference + median_fit * full_fit_count
                if full_inference is not None
                else None
            ),
            "v1_full_cache_uncompressed_bytes": int(v1_uncompressed_cache),
            "v1_full_cache_bytes": int(v1_uncompressed_cache * compression_ratio),
            "v1_full_feature_cache_bytes_by_architecture": {
                "unet_single_seed": int(baseline_feature_bytes),
                "transunet_single_seed": int(transunet_feature_bytes),
            },
            "v1_full_feature_cache_bytes_by_unet_node_single_seed": {
                node: int(rows_in_full_test * channels * 2)
                for node, channels in baseline_channels.items()
            },
            "v1_full_feature_cache_bytes_by_transunet_node_single_seed": {
                node: int(rows_in_full_test * channels * 2)
                for node, channels in transunet_channels.items()
            },
            "v3_transition_disk_bytes": int(v3_disk_bytes),
            "v5_transition_disk_bytes": int(v5_disk_bytes),
            "v1_v3_v5_total_disk_bytes": int(
                v1_uncompressed_cache * compression_ratio + v3_disk_bytes + v5_disk_bytes
            ),
            "v3_forward_seconds": (
                full_inference * int(extrapolation["v3_forward_variants_per_model"])
                if full_inference is not None
                else None
            ),
            "v5_forward_seconds": (
                full_inference * int(extrapolation["v5_forward_variants_per_model"])
                if full_inference is not None
                else None
            ),
        },
    }
    _write_json(output / "resource_profile.json", resource_profile)
    print(json.dumps(_json_ready(gate_report), ensure_ascii=False, indent=2))
    return 0 if gate_report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
