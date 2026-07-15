from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import ndimage
import torch
import yaml

from pptt.data.brats2d import (
    SliceRecord,
    discover_slice_records,
    ensure_chw,
    normalize_label,
)
from pptt.data.patient_splits import patient_id_from_slice
from pptt.experiments.model_matrix import ModelMatrixJob, load_model_matrix
from pptt.experiments.pilot import stable_seed
from pptt.interventions.activation_swap import (
    capture_module_output,
    replace_activation_vectors,
    spatially_shift_activation,
    transform_module_input,
)
from pptt.interventions.effects import (
    dose_direction_consistent,
    masked_error_counts,
    selective_path_effect,
)
from pptt.interventions.matching import (
    match_pixels_without_replacement,
    quantile_bins,
)
from pptt.interventions.ood import (
    OODReference,
    fit_ood_reference,
    load_ood_reference,
    local_context_descriptor,
    save_ood_reference,
    score_ood,
)
from pptt.io.artifacts import CaseTrace, load_case_trace
from pptt.models.adapters import build_adapter
from pptt.statistics.hypotheses import paired_patient_statistics


@dataclass(frozen=True)
class NativeSlice:
    record: SliceRecord
    image: torch.Tensor
    truth: np.ndarray
    native_logits: np.ndarray
    native_prediction: np.ndarray
    boundary_distance: np.ndarray


@dataclass(frozen=True)
class PatientCandidates:
    slices: tuple[NativeSlice, ...]
    vectors: np.ndarray
    ood_vectors: np.ndarray
    shifted_ood_vectors: np.ndarray
    slice_indices: np.ndarray
    spatial_indices: np.ndarray
    truth_class: np.ndarray
    boundary_distance: np.ndarray
    confidence: np.ndarray
    component_area: np.ndarray
    activation_norm: np.ndarray
    is_target: np.ndarray
    is_control: np.ndarray
    anchor_mismatch_count: int


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


def _group_records(records: list[SliceRecord]) -> dict[str, list[SliceRecord]]:
    grouped: dict[str, list[SliceRecord]] = defaultdict(list)
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


def _component_area_map(truth: np.ndarray, classes: tuple[int, ...]) -> np.ndarray:
    result = np.zeros_like(truth, dtype=np.float64)
    structure = np.ones((3, 3), dtype=np.uint8)
    for class_index in classes:
        components, count = ndimage.label(truth == class_index, structure=structure)
        if count == 0:
            continue
        sizes = np.bincount(components.reshape(-1))
        result[components > 0] = sizes[components[components > 0]]
    return result


def _boundary_distance(truth: np.ndarray, classes: tuple[int, ...]) -> np.ndarray:
    foreground = np.isin(truth, classes)
    return ndimage.distance_transform_edt(foreground).astype(np.float32)


def _standardized_mean_difference(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.size == 0 or second.size == 0:
        return float("nan")
    pooled_variance = 0.5 * (first.var(ddof=0) + second.var(ddof=0))
    if pooled_variance <= 0:
        return 0.0 if first.mean() == second.mean() else float("inf")
    return float((first.mean() - second.mean()) / np.sqrt(pooled_variance))


def _capture_native(
    adapter: torch.nn.Module,
    image: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        logits, activation = capture_module_output(
            adapter,
            adapter.model.inc,
            image.to(device),
        )
    return (
        logits[0].detach().cpu().numpy().astype(np.float32, copy=False),
        activation[0].detach().cpu().numpy().astype(np.float32, copy=False),
    )


def _patient_candidates(
    adapter: torch.nn.Module,
    trace: CaseTrace,
    records: list[SliceRecord],
    *,
    nodes: tuple[str, ...],
    truth_classes: tuple[int, ...],
    shift_y: int,
    shift_x: int,
    maximum_controls_per_slice: int,
    patient_seed: int,
    device: torch.device,
) -> PatientCandidates:
    if trace.truth is None or trace.margins is None:
        raise ValueError("V4 requires V3 traces with truth and canonical margins")
    if trace.final_model_state is None:
        raise ValueError("V4 requires the original final model state anchor")
    if trace.slice_ids != tuple(record.slice_id for record in records):
        raise ValueError("V3 trace and source slices do not align")
    up3_index = nodes.index("up3")
    up4_index = nodes.index("up4")
    if up4_index != up3_index + 1:
        raise ValueError("V4 requires adjacent up3 and up4 checkpoints")

    slices: list[NativeSlice] = []
    vector_rows: list[np.ndarray] = []
    ood_rows: list[np.ndarray] = []
    shifted_ood_rows: list[np.ndarray] = []
    slice_rows: list[np.ndarray] = []
    spatial_rows: list[np.ndarray] = []
    truth_rows: list[np.ndarray] = []
    boundary_rows: list[np.ndarray] = []
    confidence_rows: list[np.ndarray] = []
    area_rows: list[np.ndarray] = []
    norm_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    control_rows: list[np.ndarray] = []
    anchor_mismatch_count = 0

    for slice_index, record in enumerate(records):
        image = torch.from_numpy(
            ensure_chw(np.load(record.image_path, allow_pickle=False))
        )[None]
        truth = normalize_label(np.load(record.mask_path, allow_pickle=False))
        native_logits, activation = _capture_native(adapter, image, device=device)
        native_prediction = native_logits.argmax(axis=0).astype(np.uint8)
        anchor_mismatch_count += int(
            np.count_nonzero(native_prediction != trace.final_model_state[slice_index])
        )
        boundary = _boundary_distance(truth, truth_classes)
        area = _component_area_map(truth, truth_classes)
        reliable = trace.reliable[up3_index, slice_index]
        up3_state = trace.states[up3_index, slice_index]
        up4_state = trace.states[up4_index, slice_index]
        tumor = np.isin(truth, truth_classes)
        target = tumor & reliable & (up3_state != truth) & (up4_state == truth)
        control = tumor & reliable & (up3_state == truth) & (up4_state == truth)

        control_indices = np.flatnonzero(control.reshape(-1))
        if control_indices.size > maximum_controls_per_slice:
            rng = np.random.default_rng(
                stable_seed(patient_seed, record.slice_id, "control_subsample")
            )
            kept = rng.choice(
                control_indices,
                size=maximum_controls_per_slice,
                replace=False,
            )
            control = np.zeros_like(control)
            control.reshape(-1)[kept] = True

        target_indices = np.flatnonzero(target.reshape(-1))
        control_indices = np.flatnonzero(control.reshape(-1))
        spatial_indices = np.concatenate([target_indices, control_indices])
        roles_target = np.concatenate(
            [
                np.ones(target_indices.size, dtype=bool),
                np.zeros(control_indices.size, dtype=bool),
            ]
        )
        roles_control = ~roles_target

        if spatial_indices.size:
            activation_flat = activation.reshape(activation.shape[0], -1)
            shifted_activation = np.roll(
                activation,
                shift=(shift_y, shift_x),
                axis=(-2, -1),
            )
            ood_flat = local_context_descriptor(activation).reshape(
                activation.shape[0] * 3,
                -1,
            )
            shifted_ood_flat = local_context_descriptor(
                shifted_activation
            ).reshape(activation.shape[0] * 3, -1)
            vectors = activation_flat[:, spatial_indices].T
            vector_rows.append(vectors)
            ood_rows.append(ood_flat[:, spatial_indices].T)
            shifted_ood_rows.append(shifted_ood_flat[:, spatial_indices].T)
            slice_rows.append(
                np.full(spatial_indices.size, slice_index, dtype=np.int32)
            )
            spatial_rows.append(spatial_indices.astype(np.int32, copy=False))
            truth_rows.append(truth.reshape(-1)[spatial_indices])
            boundary_rows.append(boundary.reshape(-1)[spatial_indices])
            confidence_rows.append(
                trace.margins[up3_index, slice_index].reshape(-1)[spatial_indices]
            )
            area_rows.append(area.reshape(-1)[spatial_indices])
            norm_rows.append(np.linalg.norm(vectors, axis=1))
            target_rows.append(roles_target)
            control_rows.append(roles_control)

        slices.append(
            NativeSlice(
                record=record,
                image=image,
                truth=truth,
                native_logits=native_logits,
                native_prediction=native_prediction,
                boundary_distance=boundary,
            )
        )

    channel_count = 64
    if vector_rows:
        channel_count = int(vector_rows[0].shape[1])

    def concatenate(rows: list[np.ndarray], *, dtype: np.dtype) -> np.ndarray:
        if rows:
            return np.concatenate(rows).astype(dtype, copy=False)
        return np.empty(0, dtype=dtype)

    return PatientCandidates(
        slices=tuple(slices),
        vectors=(
            np.concatenate(vector_rows).astype(np.float32, copy=False)
            if vector_rows
            else np.empty((0, channel_count), dtype=np.float32)
        ),
        ood_vectors=(
            np.concatenate(ood_rows).astype(np.float32, copy=False)
            if ood_rows
            else np.empty((0, channel_count * 3), dtype=np.float32)
        ),
        shifted_ood_vectors=(
            np.concatenate(shifted_ood_rows).astype(np.float32, copy=False)
            if shifted_ood_rows
            else np.empty((0, channel_count * 3), dtype=np.float32)
        ),
        slice_indices=concatenate(slice_rows, dtype=np.int32),
        spatial_indices=concatenate(spatial_rows, dtype=np.int32),
        truth_class=concatenate(truth_rows, dtype=np.uint8),
        boundary_distance=concatenate(boundary_rows, dtype=np.float32),
        confidence=concatenate(confidence_rows, dtype=np.float32),
        component_area=concatenate(area_rows, dtype=np.float64),
        activation_norm=concatenate(norm_rows, dtype=np.float32),
        is_target=concatenate(target_rows, dtype=bool),
        is_control=concatenate(control_rows, dtype=bool),
        anchor_mismatch_count=anchor_mismatch_count,
    )


def _fit_native_ood_references(
    adapter: torch.nn.Module,
    grouped_records: dict[str, list[SliceRecord]],
    *,
    config: dict[str, Any],
    truth_classes: tuple[int, ...],
    model_seed: int,
    device: torch.device,
    max_patients: int | None,
) -> tuple[dict[int, OODReference], list[str], dict[int, int]]:
    if not bool(config.get("class_conditional", False)):
        raise ValueError("V4 OOD auditing must be conditioned on the matched truth class")
    if str(config.get("descriptor")) != "center_mean_std_3x3":
        raise ValueError("V4 OOD auditing requires the registered local descriptor")
    patient_ids = np.asarray(sorted(grouped_records))
    configured_count = int(config["reference_patient_count"])
    if max_patients is not None:
        configured_count = min(configured_count, max_patients)
    selected_count = min(configured_count, patient_ids.size)
    rng = np.random.default_rng(stable_seed(int(config["seed"]), model_seed, "ood"))
    selected = sorted(
        str(value)
        for value in rng.choice(patient_ids, size=selected_count, replace=False)
    )
    samples: dict[int, list[np.ndarray]] = {value: [] for value in truth_classes}
    vectors_per_slice = int(config["vectors_per_slice"])
    for patient_id in selected:
        for record in grouped_records[patient_id]:
            image = torch.from_numpy(
                ensure_chw(np.load(record.image_path, allow_pickle=False))
            )[None]
            _, activation = _capture_native(adapter, image, device=device)
            flat = local_context_descriptor(activation).reshape(
                activation.shape[0] * 3,
                -1,
            ).T
            truth = normalize_label(np.load(record.mask_path, allow_pickle=False)).reshape(-1)
            for class_index in truth_classes:
                positions = np.flatnonzero(truth == class_index)
                if positions.size == 0:
                    continue
                sample_count = min(vectors_per_slice, positions.size)
                selected_positions = rng.choice(
                    positions,
                    size=sample_count,
                    replace=False,
                )
                samples[class_index].append(flat[selected_positions])
    references = {}
    counts = {}
    for class_index in truth_classes:
        if not samples[class_index]:
            raise ValueError(
                f"No native class-{class_index} activations were available for OOD fitting"
            )
        values = np.concatenate(samples[class_index]).astype(np.float32, copy=False)
        references[class_index] = fit_ood_reference(
            values,
            components=int(config["pca_components"]),
            neighbors=int(config["neighbors"]),
            max_samples=int(config["max_samples"]),
            seed=stable_seed(
                int(config["seed"]),
                model_seed,
                "ood_fit",
                class_index,
            ),
            threshold_quantile=float(config["threshold_quantile"]),
        )
        counts[class_index] = int(values.shape[0])
    return references, selected, counts


def _run_transformed_forward(
    adapter: torch.nn.Module,
    image: torch.Tensor,
    *,
    device: torch.device,
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> np.ndarray:
    with torch.inference_mode(), transform_module_input(
        adapter.model.up4,
        argument_index=1,
        transform=transform,
    ):
        logits = adapter(image.to(device))
    return logits[0].detach().cpu().numpy().astype(np.float32, copy=False)


def _condition_effect(
    native_predictions: tuple[np.ndarray, ...],
    intervened_predictions: tuple[np.ndarray, ...],
    slices: tuple[NativeSlice, ...],
    target_masks: tuple[np.ndarray, ...],
    control_masks: tuple[np.ndarray, ...],
) -> dict[str, Any]:
    counts = {
        "target_native_error_count": 0,
        "target_intervened_error_count": 0,
        "target_pixel_count": 0,
        "control_native_error_count": 0,
        "control_intervened_error_count": 0,
        "control_pixel_count": 0,
        "global_native_error_count": 0,
        "global_intervened_error_count": 0,
        "global_pixel_count": 0,
        "target_rollback_count": 0,
        "target_native_correct_count": 0,
    }
    for native, changed, context, target, control in zip(
        native_predictions,
        intervened_predictions,
        slices,
        target_masks,
        control_masks,
        strict=True,
    ):
        native_target_error, target_count = masked_error_counts(
            native,
            context.truth,
            target,
        )
        changed_target_error, _ = masked_error_counts(changed, context.truth, target)
        native_control_error, control_count = masked_error_counts(
            native,
            context.truth,
            control,
        )
        changed_control_error, _ = masked_error_counts(changed, context.truth, control)
        counts["target_native_error_count"] += native_target_error
        counts["target_intervened_error_count"] += changed_target_error
        counts["target_pixel_count"] += target_count
        counts["control_native_error_count"] += native_control_error
        counts["control_intervened_error_count"] += changed_control_error
        counts["control_pixel_count"] += control_count
        counts["global_native_error_count"] += int(np.count_nonzero(native != context.truth))
        counts["global_intervened_error_count"] += int(
            np.count_nonzero(changed != context.truth)
        )
        counts["global_pixel_count"] += int(context.truth.size)
        native_correct = target & (native == context.truth)
        counts["target_native_correct_count"] += int(native_correct.sum())
        counts["target_rollback_count"] += int(
            np.count_nonzero(native_correct & (changed != context.truth))
        )
    if counts["target_pixel_count"] == 0 or counts["control_pixel_count"] == 0:
        raise ValueError("Matched target and control masks must be nonempty")
    target_native = counts["target_native_error_count"] / counts["target_pixel_count"]
    target_changed = (
        counts["target_intervened_error_count"] / counts["target_pixel_count"]
    )
    control_native = counts["control_native_error_count"] / counts["control_pixel_count"]
    control_changed = (
        counts["control_intervened_error_count"] / counts["control_pixel_count"]
    )
    return {
        **counts,
        "target_native_error_rate": target_native,
        "target_intervened_error_rate": target_changed,
        "control_native_error_rate": control_native,
        "control_intervened_error_rate": control_changed,
        "target_error_rate_change": target_changed - target_native,
        "control_error_rate_change": control_changed - control_native,
        "selective_path_effect": selective_path_effect(
            target_native=target_native,
            target_intervened=target_changed,
            control_native=control_native,
            control_intervened=control_changed,
        ),
        "global_native_error_rate": (
            counts["global_native_error_count"] / counts["global_pixel_count"]
        ),
        "global_intervened_error_rate": (
            counts["global_intervened_error_count"] / counts["global_pixel_count"]
        ),
        "target_rollback_rate": (
            counts["target_rollback_count"] / counts["target_native_correct_count"]
            if counts["target_native_correct_count"]
            else float("nan")
        ),
    }


def _ood_rows(
    references: dict[int, OODReference],
    *,
    condition: str,
    target_vectors: np.ndarray,
    control_vectors: np.ndarray,
    target_classes: np.ndarray,
    control_classes: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for role, vectors, classes in (
        ("target", target_vectors, target_classes),
        ("control", control_vectors, control_classes),
        (
            "all",
            np.concatenate([target_vectors, control_vectors]),
            np.concatenate([target_classes, control_classes]),
        ),
    ):
        class_groups = [-1, *sorted(references)]
        for class_index in class_groups:
            selected = np.ones(classes.shape, dtype=bool) if class_index == -1 else classes == class_index
            if not selected.any():
                continue
            mahalanobis_values = []
            mahalanobis_ratios = []
            knn_values = []
            knn_ratios = []
            for current_class in sorted(references):
                class_mask = selected & (classes == current_class)
                if not class_mask.any():
                    continue
                reference = references[current_class]
                scores = score_ood(reference, vectors[class_mask])
                mahalanobis_values.append(scores.mahalanobis)
                mahalanobis_ratios.append(
                    scores.mahalanobis / reference.mahalanobis_threshold
                )
                knn_values.append(scores.knn)
                knn_ratios.append(scores.knn / reference.knn_threshold)
            mahalanobis = np.concatenate(mahalanobis_values)
            mahalanobis_ratio = np.concatenate(mahalanobis_ratios)
            knn = np.concatenate(knn_values)
            knn_ratio = np.concatenate(knn_ratios)
            class_reference = references.get(class_index)
            rows.append(
                {
                    "condition": condition,
                    "role": role,
                    "class_index": class_index,
                    "vector_count": int(mahalanobis.size),
                    "mahalanobis_p50": float(np.quantile(mahalanobis, 0.50)),
                    "mahalanobis_p95": float(np.quantile(mahalanobis, 0.95)),
                    "mahalanobis_p99": float(np.quantile(mahalanobis, 0.99)),
                    "mahalanobis_max": float(mahalanobis.max()),
                    "mahalanobis_threshold": (
                        class_reference.mahalanobis_threshold
                        if class_reference is not None
                        else float("nan")
                    ),
                    "mahalanobis_ratio_p99": float(
                        np.quantile(mahalanobis_ratio, 0.99)
                    ),
                    "mahalanobis_exceedance_fraction": float(
                        np.mean(mahalanobis_ratio > 1.0)
                    ),
                    "knn_p50": float(np.quantile(knn, 0.50)),
                    "knn_p95": float(np.quantile(knn, 0.95)),
                    "knn_p99": float(np.quantile(knn, 0.99)),
                    "knn_max": float(knn.max()),
                    "knn_threshold": (
                        class_reference.knn_threshold
                        if class_reference is not None
                        else float("nan")
                    ),
                    "knn_ratio_p99": float(np.quantile(knn_ratio, 0.99)),
                    "knn_exceedance_fraction": float(np.mean(knn_ratio > 1.0)),
                }
            )
    return rows


def _subgroup_rows(
    *,
    condition: str,
    native_predictions: tuple[np.ndarray, ...],
    changed_predictions: tuple[np.ndarray, ...],
    slices: tuple[NativeSlice, ...],
    target_masks: tuple[np.ndarray, ...],
    truth_classes: tuple[int, ...],
    boundary_band_pixels: float,
) -> list[dict[str, Any]]:
    groups: list[tuple[str, str, int | None]] = [
        ("boundary", "boundary", None),
        ("boundary", "interior", None),
    ]
    groups.extend(("truth_class", f"class_{value}", value) for value in truth_classes)
    rows = []
    for group_type, group_name, class_index in groups:
        pixel_count = 0
        native_correct_count = 0
        rollback_count = 0
        for native, changed, context, target in zip(
            native_predictions,
            changed_predictions,
            slices,
            target_masks,
            strict=True,
        ):
            selected = target.copy()
            if group_type == "boundary":
                if group_name == "boundary":
                    selected &= context.boundary_distance <= boundary_band_pixels
                else:
                    selected &= context.boundary_distance > boundary_band_pixels
            else:
                selected &= context.truth == class_index
            native_correct = selected & (native == context.truth)
            pixel_count += int(selected.sum())
            native_correct_count += int(native_correct.sum())
            rollback_count += int(
                np.count_nonzero(native_correct & (changed != context.truth))
            )
        rows.append(
            {
                "condition": condition,
                "group_type": group_type,
                "group": group_name,
                "pixel_count": pixel_count,
                "native_correct_count": native_correct_count,
                "rollback_count": rollback_count,
                "rollback_rate": (
                    rollback_count / native_correct_count
                    if native_correct_count
                    else float("nan")
                ),
            }
        )
    return rows


def _run_patient(
    adapter: torch.nn.Module,
    references: dict[int, OODReference],
    trace: CaseTrace,
    records: list[SliceRecord],
    *,
    experiment_config: dict[str, Any],
    model_seed: int,
    patient_id: str,
    minimum_pairs: int,
    device: torch.device,
) -> dict[str, Any]:
    nodes = tuple(str(value) for value in experiment_config["nodes"])
    truth_classes = tuple(int(value) for value in experiment_config["target"]["truth_classes"])
    matching_config = experiment_config["matching"]
    intervention_config = experiment_config["intervention"]
    shift_y, shift_x = (
        int(value) for value in intervention_config["spatial_shift_yx"]
    )
    candidates = _patient_candidates(
        adapter,
        trace,
        records,
        nodes=nodes,
        truth_classes=truth_classes,
        shift_y=shift_y,
        shift_x=shift_x,
        maximum_controls_per_slice=int(
            matching_config["maximum_controls_per_slice"]
        ),
        patient_seed=stable_seed(model_seed, patient_id),
        device=device,
    )
    target_count = int(candidates.is_target.sum())
    control_count = int(candidates.is_control.sum())
    matching_row: dict[str, Any] = {
        "target_candidate_count": target_count,
        "control_candidate_count": control_count,
        "anchor_mismatch_count": candidates.anchor_mismatch_count,
        "minimum_required_pairs": minimum_pairs,
    }
    if target_count == 0 or control_count == 0:
        matching_row.update(
            {
                "status": "FAILED",
                "failure_reason": "no_target_candidates" if target_count == 0 else "no_control_candidates",
                "matched_pair_count": 0,
                "unmatched_target_count": target_count,
                "matched_fraction": 0.0,
            }
        )
        return {
            "matching": [matching_row],
            "effects": [],
            "subgroups": [],
            "ood": [],
            "restore": [],
        }

    boundary_bin = np.digitize(
        candidates.boundary_distance,
        np.asarray(matching_config["boundary_bin_edges"], dtype=np.float64),
    )
    reference_mask = candidates.is_target | candidates.is_control
    bin_count = int(matching_config["quantile_bin_count"])
    confidence_bin = quantile_bins(
        candidates.confidence,
        reference_mask=reference_mask,
        bin_count=bin_count,
    )
    area_bin = quantile_bins(
        candidates.component_area,
        reference_mask=reference_mask,
        bin_count=bin_count,
    )
    norm_bin = quantile_bins(
        candidates.activation_norm,
        reference_mask=reference_mask,
        bin_count=bin_count,
    )
    matches = match_pixels_without_replacement(
        truth_class=candidates.truth_class,
        boundary_bin=boundary_bin,
        confidence_bin=confidence_bin,
        area_bin=area_bin,
        norm_bin=norm_bin,
        target_mask=candidates.is_target,
        control_mask=candidates.is_control,
        boundary_value=candidates.boundary_distance,
        confidence_value=candidates.confidence,
        area_value=candidates.component_area,
        norm_value=candidates.activation_norm,
        confidence_caliper=int(matching_config["confidence_caliper"]),
        norm_caliper=int(matching_config["activation_norm_caliper"]),
    )
    matching_row.update(
        {
            "matched_pair_count": matches.matched_count,
            "unmatched_target_count": matches.unmatched_target_count,
            "matched_fraction": matches.matched_fraction,
        }
    )
    if matches.matched_count < minimum_pairs:
        matching_row.update(
            {
                "status": "FAILED",
                "failure_reason": "insufficient_calipered_matches",
            }
        )
        return {
            "matching": [matching_row],
            "effects": [],
            "subgroups": [],
            "ood": [],
            "restore": [],
        }

    target_rows = matches.target_indices
    control_rows = matches.control_indices
    matching_row.update(
        {
            "status": "PASS",
            "failure_reason": "",
            "smd_boundary_distance": _standardized_mean_difference(
                candidates.boundary_distance[target_rows],
                candidates.boundary_distance[control_rows],
            ),
            "smd_up3_confidence": _standardized_mean_difference(
                candidates.confidence[target_rows],
                candidates.confidence[control_rows],
            ),
            "smd_component_area": _standardized_mean_difference(
                candidates.component_area[target_rows],
                candidates.component_area[control_rows],
            ),
            "smd_activation_norm": _standardized_mean_difference(
                candidates.activation_norm[target_rows],
                candidates.activation_norm[control_rows],
            ),
        }
    )

    destination_rows = np.concatenate([target_rows, control_rows])
    source_rows = np.concatenate([control_rows, target_rows])
    target_masks: list[np.ndarray] = []
    control_masks: list[np.ndarray] = []
    replacement_plans: list[tuple[torch.Tensor, torch.Tensor]] = []
    for slice_index, context in enumerate(candidates.slices):
        target_mask = np.zeros_like(context.truth, dtype=bool)
        control_mask = np.zeros_like(context.truth, dtype=bool)
        selected_targets = target_rows[candidates.slice_indices[target_rows] == slice_index]
        selected_controls = control_rows[candidates.slice_indices[control_rows] == slice_index]
        target_mask.reshape(-1)[candidates.spatial_indices[selected_targets]] = True
        control_mask.reshape(-1)[candidates.spatial_indices[selected_controls]] = True
        target_masks.append(target_mask)
        control_masks.append(control_mask)
        selected_destinations = destination_rows[
            candidates.slice_indices[destination_rows] == slice_index
        ]
        corresponding_sources = source_rows[
            candidates.slice_indices[destination_rows] == slice_index
        ]
        replacement_plans.append(
            (
                torch.from_numpy(
                    candidates.spatial_indices[selected_destinations].astype(np.int64)
                ),
                torch.from_numpy(candidates.vectors[corresponding_sources]),
            )
        )

    native_predictions = tuple(context.native_prediction for context in candidates.slices)
    condition_logits: dict[str, tuple[np.ndarray, ...]] = {}
    alphas = tuple(float(value) for value in intervention_config["replacement_alphas"])
    for alpha in alphas:
        name = f"replacement_alpha_{alpha:.2f}"
        outputs = []
        for context, (destinations, sources) in zip(
            candidates.slices,
            replacement_plans,
            strict=True,
        ):
            outputs.append(
                _run_transformed_forward(
                    adapter,
                    context.image,
                    device=device,
                    transform=lambda activation, d=destinations, s=sources, a=alpha: replace_activation_vectors(
                        activation,
                        d,
                        s,
                        alpha=a,
                    ),
                )
            )
        condition_logits[name] = tuple(outputs)

    shifted_outputs = []
    zero_outputs = []
    for context in candidates.slices:
        shifted_outputs.append(
            _run_transformed_forward(
                adapter,
                context.image,
                device=device,
                transform=lambda activation: spatially_shift_activation(
                    activation,
                    shift_y=shift_y,
                    shift_x=shift_x,
                ),
            )
        )
        zero_outputs.append(
            _run_transformed_forward(
                adapter,
                context.image,
                device=device,
                transform=torch.zeros_like,
            )
        )
    condition_logits["spatial_shift"] = tuple(shifted_outputs)
    condition_logits["zero"] = tuple(zero_outputs)

    restored_logits = []
    max_restore_error = 0.0
    restore_state_mismatch_count = 0
    for context in candidates.slices:
        with torch.inference_mode():
            output = adapter(context.image.to(device))[0].detach().cpu().numpy()
        restored_logits.append(output.astype(np.float32, copy=False))
        max_restore_error = max(
            max_restore_error,
            float(np.max(np.abs(output - context.native_logits))),
        )
        restore_state_mismatch_count += int(
            np.count_nonzero(output.argmax(axis=0) != context.native_prediction)
        )
    condition_logits["restore_native"] = tuple(restored_logits)

    effects = []
    subgroups = []
    for condition, logits in condition_logits.items():
        changed_predictions = tuple(
            value.argmax(axis=0).astype(np.uint8) for value in logits
        )
        alpha = (
            float(condition.rsplit("_", 1)[-1])
            if condition.startswith("replacement_alpha_")
            else None
        )
        effects.append(
            {
                "condition": condition,
                "alpha": alpha,
                **_condition_effect(
                    native_predictions,
                    changed_predictions,
                    candidates.slices,
                    tuple(target_masks),
                    tuple(control_masks),
                ),
            }
        )
        subgroups.extend(
            _subgroup_rows(
                condition=condition,
                native_predictions=native_predictions,
                changed_predictions=changed_predictions,
                slices=candidates.slices,
                target_masks=tuple(target_masks),
                truth_classes=truth_classes,
                boundary_band_pixels=float(
                    intervention_config["boundary_band_pixels"]
                ),
            )
        )

    target_native_vectors = candidates.vectors[target_rows]
    control_native_vectors = candidates.vectors[control_rows]
    target_native_ood = candidates.ood_vectors[target_rows]
    control_native_ood = candidates.ood_vectors[control_rows]
    channel_count = candidates.vectors.shape[1]
    target_truth_classes = candidates.truth_class[target_rows]
    control_truth_classes = candidates.truth_class[control_rows]
    ood = _ood_rows(
        references,
        condition="native",
        target_vectors=target_native_ood,
        control_vectors=control_native_ood,
        target_classes=target_truth_classes,
        control_classes=control_truth_classes,
    )
    for alpha in alphas:
        target_changed_center = (
            (1.0 - alpha) * target_native_vectors
            + alpha * control_native_vectors
        )
        control_changed_center = (
            (1.0 - alpha) * control_native_vectors
            + alpha * target_native_vectors
        )
        ood.extend(
            _ood_rows(
                references,
                condition=f"replacement_alpha_{alpha:.2f}",
                target_vectors=np.concatenate(
                    [
                        target_changed_center,
                        target_native_ood[:, channel_count:],
                    ],
                    axis=1,
                ),
                control_vectors=np.concatenate(
                    [
                        control_changed_center,
                        control_native_ood[:, channel_count:],
                    ],
                    axis=1,
                ),
                target_classes=target_truth_classes,
                control_classes=control_truth_classes,
            )
        )
    ood.extend(
        _ood_rows(
            references,
            condition="spatial_shift",
            target_vectors=candidates.shifted_ood_vectors[target_rows],
            control_vectors=candidates.shifted_ood_vectors[control_rows],
            target_classes=target_truth_classes,
            control_classes=control_truth_classes,
        )
    )
    ood.extend(
        _ood_rows(
            references,
            condition="zero",
            target_vectors=np.concatenate(
                [
                    np.zeros_like(target_native_vectors),
                    target_native_ood[:, channel_count:],
                ],
                axis=1,
            ),
            control_vectors=np.concatenate(
                [
                    np.zeros_like(control_native_vectors),
                    control_native_ood[:, channel_count:],
                ],
                axis=1,
            ),
            target_classes=target_truth_classes,
            control_classes=control_truth_classes,
        )
    )
    ood.extend(
        _ood_rows(
            references,
            condition="restore_native",
            target_vectors=target_native_ood,
            control_vectors=control_native_ood,
            target_classes=target_truth_classes,
            control_classes=control_truth_classes,
        )
    )
    return {
        "matching": [matching_row],
        "effects": effects,
        "subgroups": subgroups,
        "ood": ood,
        "restore": [
            {
                "max_abs_logit_error": max_restore_error,
                "state_mismatch_count": restore_state_mismatch_count,
                "forward_pre_hook_count": len(adapter.model.up4._forward_pre_hooks),
            }
        ],
    }


def _extend_with_metadata(
    destination: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    destination.extend({**metadata, **row} for row in rows)


def _summarize_conclusion(
    effects: pd.DataFrame,
    ood: pd.DataFrame,
    restore: pd.DataFrame,
    *,
    trigger: dict[str, Any],
    experiment_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    statistics_config = experiment_config["statistics"]
    primary_alpha = float(experiment_config["intervention"]["primary_alpha"])
    primary_condition = f"replacement_alpha_{primary_alpha:.2f}"
    statistic_rows = []
    dose_rows = []
    for model_seed in sorted(effects.model_seed.unique()) if not effects.empty else []:
        selected = effects[
            (effects.model_seed == model_seed)
            & (effects.condition == primary_condition)
        ]
        values = selected.selective_path_effect.to_numpy(dtype=np.float64)
        if values.size >= 2:
            result = paired_patient_statistics(
                values,
                np.zeros_like(values),
                iterations=int(statistics_config["bootstrap_iterations"]),
                seed=stable_seed(
                    int(statistics_config["bootstrap_seed"]),
                    "v4_spe",
                    int(model_seed),
                ),
            )
            statistic_rows.append(
                {
                    "model_seed": int(model_seed),
                    "endpoint": "selective_path_effect",
                    "condition": primary_condition,
                    **result.to_dict(),
                }
            )
        seed_dose = effects[
            (effects.model_seed == model_seed)
            & effects.condition.str.startswith("replacement_alpha_")
        ]
        means = seed_dose.groupby("alpha", as_index=False).selective_path_effect.mean()
        alpha_values = np.concatenate([[0.0], means.alpha.to_numpy(dtype=np.float64)])
        effect_values = np.concatenate(
            [[0.0], means.selective_path_effect.to_numpy(dtype=np.float64)]
        )
        dose_rows.append(
            {
                "model_seed": int(model_seed),
                "alphas": alpha_values,
                "mean_selective_path_effects": effect_values,
                "direction_consistent": dose_direction_consistent(
                    alpha_values,
                    effect_values,
                ),
            }
        )
    statistics = pd.DataFrame(statistic_rows)
    required_seed_count = int(trigger["minimum_consistent_model_seeds"])
    spe_supported = (
        int((statistics.ci_low > 0).sum()) if not statistics.empty else 0
    )
    dose_supported = sum(bool(row["direction_consistent"]) for row in dose_rows)
    restore_tolerance = float(
        experiment_config["intervention"]["restore_logit_tolerance"]
    )
    restoration_pass = bool(
        not restore.empty
        and (restore.max_abs_logit_error <= restore_tolerance).all()
        and (restore.state_mismatch_count == 0).all()
        and (restore.forward_pre_hook_count == 0).all()
    )
    primary_ood = ood[
        (ood.condition == primary_condition)
        & (ood.role == "all")
        & (ood.class_index == -1)
    ]
    native_ood = ood[
        (ood.condition == "native")
        & (ood.role == "all")
        & (ood.class_index == -1)
    ]
    ood_config = experiment_config["ood"]
    p99_tolerance = float(ood_config["relative_p99_tolerance"])
    exceedance_tolerance = float(ood_config["exceedance_fraction_tolerance"])
    if primary_ood.empty or native_ood.empty:
        ood_pass = False
        ood_comparison_rows: list[dict[str, Any]] = []
    else:
        comparison = primary_ood.merge(
            native_ood,
            on=["model", "model_seed", "split", "patient_id", "role", "class_index"],
            suffixes=("_primary", "_native"),
            validate="one_to_one",
        )
        comparison["mahalanobis_within_limit"] = (
            comparison.mahalanobis_ratio_p99_primary
            <= np.maximum(
                1.0,
                comparison.mahalanobis_ratio_p99_native + p99_tolerance,
            )
        )
        comparison["knn_within_limit"] = (
            comparison.knn_ratio_p99_primary
            <= np.maximum(
                1.0,
                comparison.knn_ratio_p99_native + p99_tolerance,
            )
        )
        comparison["mahalanobis_exceedance_within_limit"] = (
            comparison.mahalanobis_exceedance_fraction_primary
            <= comparison.mahalanobis_exceedance_fraction_native
            + exceedance_tolerance
        )
        comparison["knn_exceedance_within_limit"] = (
            comparison.knn_exceedance_fraction_primary
            <= comparison.knn_exceedance_fraction_native
            + exceedance_tolerance
        )
        ood_columns = [
            "model_seed",
            "patient_id",
            "mahalanobis_ratio_p99_primary",
            "mahalanobis_ratio_p99_native",
            "knn_ratio_p99_primary",
            "knn_ratio_p99_native",
            "mahalanobis_exceedance_fraction_primary",
            "mahalanobis_exceedance_fraction_native",
            "knn_exceedance_fraction_primary",
            "knn_exceedance_fraction_native",
            "mahalanobis_within_limit",
            "knn_within_limit",
            "mahalanobis_exceedance_within_limit",
            "knn_exceedance_within_limit",
        ]
        ood_comparison_rows = comparison[ood_columns].to_dict("records")
        ood_pass = bool(
            comparison[
                [
                    "mahalanobis_within_limit",
                    "knn_within_limit",
                    "mahalanobis_exceedance_within_limit",
                    "knn_exceedance_within_limit",
                ]
            ].to_numpy(dtype=bool).all()
        )
    gates = {
        "spe_patient_ci_above_zero_in_required_seeds": spe_supported
        >= required_seed_count,
        "dose_direction_consistent_in_required_seeds": dose_supported
        >= required_seed_count,
        "native_restoration_within_tolerance": restoration_pass,
        "primary_replacement_not_more_ood_than_registered_native_limit": ood_pass,
    }
    conclusion = {
        "candidate_transition": trigger["candidate_transition"],
        "primary_condition": primary_condition,
        "minimum_required_model_seeds": required_seed_count,
        "spe_supported_model_seed_count": spe_supported,
        "dose_supported_model_seed_count": dose_supported,
        "gates": gates,
        "selective_functional_contribution_supported": all(gates.values()),
        "allowed_claim_if_supported": (
            "The up4 skip path has a selective functional contribution to the "
            "prelocalized persistent corrections under the registered model, "
            "data, path, and natural-activation replacement operator."
        ),
        "otherwise_report": "path sensitivity without selective source attribution",
        "dose_audit": dose_rows,
        "ood_comparison_audit": ood_comparison_rows,
    }
    return statistics, conclusion


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the discovery-locked up4 path intervention for PPTT V4."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trigger", type=Path, required=True)
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/brats2023_2d.yaml"),
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--v3-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--jobs", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-patients-per-split", type=int)
    parser.add_argument("--minimum-pairs-per-patient", type=int)
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    experiment_config = _load_yaml(args.config)
    data_config = _load_yaml(args.data_config)
    trigger = json.loads(args.trigger.read_text(encoding="utf-8"))
    output_root = (
        args.output_root
        if args.output_root is not None
        else workspace / str(experiment_config["output_root"])
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if str(trigger.get("candidate_transition")) != str(
        experiment_config["target"]["transition"]
    ):
        raise ValueError("V4 trigger and configured transition do not match")
    if not bool(trigger.get("triggered", False)):
        _write_json(
            output_root / "NOT_TRIGGERED.json",
            {
                "status": "NOT_TRIGGERED",
                "trigger": trigger,
                "no_fallback_transition": True,
            },
        )
        print(f"V4 not triggered; wrote {output_root / 'NOT_TRIGGERED.json'}")
        return 0

    matrix = load_model_matrix(
        workspace / str(experiment_config["model_matrix"]),
        workspace_root=workspace,
        asset_root_override=args.asset_root,
    )
    asset_root = matrix.asset_root
    configured_model = str(experiment_config["model"])
    eligible_seeds = {
        int(row["model_seed"])
        for row in trigger.get("seed_results", [])
        if bool(row.get("all_conditions_met", False))
    }
    jobs = [
        job
        for job in matrix.jobs
        if job.model == configured_model and job.seed in eligible_seeds
    ]
    if args.jobs:
        requested = {_job_selector(value) for value in args.jobs}
        jobs = [job for job in jobs if job.job_id in requested]
        missing = requested - {job.job_id for job in jobs}
        if missing:
            raise ValueError(f"Requested V4 jobs are not trigger-eligible: {sorted(missing)}")
    if not jobs:
        raise ValueError("Triggered V4 has no eligible baseline model jobs")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    v3_root = (
        args.v3_root
        if args.v3_root is not None
        else workspace / str(experiment_config["v3_root"])
    ).resolve()
    discovery_split = str(experiment_config["discovery_split"])
    primary_split = str(experiment_config["primary_split"])
    minimum_pairs = (
        int(args.minimum_pairs_per_patient)
        if args.minimum_pairs_per_patient is not None
        else int(experiment_config["matching"]["minimum_pairs_per_patient"])
    )
    all_rows: dict[str, list[dict[str, Any]]] = {
        "matching": [],
        "effects": [],
        "subgroups": [],
        "ood": [],
        "restore": [],
    }
    run_jobs = []

    for job in jobs:
        if not job.checkpoint.is_file():
            raise FileNotFoundError(f"Missing V4 checkpoint: {job.checkpoint}")
        model_config = _load_yaml(job.model_config)
        adapter = build_adapter(
            job.model,
            n_channels=int(model_config["n_channels"]),
            num_classes=int(model_config["num_classes"]),
            bilinear=bool(model_config.get("bilinear", False)),
            img_size=int(model_config.get("img_size", 160)),
        )
        checkpoint_digest = adapter.load_checkpoint(job.checkpoint, map_location="cpu")
        adapter.to(device).eval()
        job_output = output_root / job.model / f"seed_{job.seed}"
        manifest = {
            "schema_version": 1,
            "experiment": experiment_config,
            "trigger": trigger,
            "job": job.job_id,
            "checkpoint": job.checkpoint,
            "checkpoint_sha256": checkpoint_digest,
            "v3_root": v3_root,
            "minimum_pairs_per_patient": minimum_pairs,
            "max_patients_per_split": args.max_patients_per_split,
        }
        manifest_path = job_output / "job_manifest.json"
        if args.resume and manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing != _json_ready(manifest):
                raise ValueError(f"V4 resume identity changed: {manifest_path}")
        else:
            _write_json(manifest_path, manifest)

        discovery_config = data_config["splits"][discovery_split]
        discovery_grouped = _group_records(
            discover_slice_records(
                asset_root / str(discovery_config["image_dir"]),
                asset_root / str(discovery_config["mask_dir"]),
            )
        )
        truth_classes = tuple(
            int(value) for value in experiment_config["target"]["truth_classes"]
        )
        reference_paths = {
            class_index: job_output
            / f"native_ood_reference_class_{class_index}.npz"
            for class_index in truth_classes
        }
        reference_meta_path = job_output / "native_ood_reference.json"
        if (
            args.resume
            and all(path.is_file() for path in reference_paths.values())
            and reference_meta_path.is_file()
        ):
            references = {
                class_index: load_ood_reference(path)
                for class_index, path in reference_paths.items()
            }
            reference_metadata = json.loads(
                reference_meta_path.read_text(encoding="utf-8")
            )
        else:
            references, reference_patients, sampled_counts = _fit_native_ood_references(
                adapter,
                discovery_grouped,
                config=experiment_config["ood"],
                truth_classes=truth_classes,
                model_seed=job.seed,
                device=device,
                max_patients=args.max_patients_per_split,
            )
            job_output.mkdir(parents=True, exist_ok=True)
            for class_index, reference in references.items():
                save_ood_reference(reference_paths[class_index], reference)
            reference_metadata = {
                "patient_ids": reference_patients,
                "class_conditional": True,
                "classes": {
                    str(class_index): {
                        "sampled_vector_count_before_fit_cap": sampled_counts[
                            class_index
                        ],
                        "mahalanobis_threshold": reference.mahalanobis_threshold,
                        "knn_threshold": reference.knn_threshold,
                    }
                    for class_index, reference in references.items()
                },
            }
            _write_json(reference_meta_path, reference_metadata)

        primary_config = data_config["splits"][primary_split]
        primary_grouped = _group_records(
            discover_slice_records(
                asset_root / str(primary_config["image_dir"]),
                asset_root / str(primary_config["mask_dir"]),
            )
        )
        patient_ids = sorted(primary_grouped)
        if args.max_patients_per_split is not None:
            patient_ids = patient_ids[: args.max_patients_per_split]
        passed = 0
        for patient_id in patient_ids:
            patient_path = job_output / "patient_results" / f"{patient_id}.json"
            if args.resume and patient_path.is_file():
                payload = json.loads(patient_path.read_text(encoding="utf-8"))
            else:
                trace_path = (
                    v3_root
                    / job.model
                    / f"seed_{job.seed}"
                    / primary_split
                    / "case_traces"
                    / f"{patient_id}.npz"
                )
                if not trace_path.is_file():
                    raise FileNotFoundError(f"Missing V3 trace for V4: {trace_path}")
                payload = _run_patient(
                    adapter,
                    references,
                    load_case_trace(trace_path),
                    primary_grouped[patient_id],
                    experiment_config=experiment_config,
                    model_seed=job.seed,
                    patient_id=patient_id,
                    minimum_pairs=minimum_pairs,
                    device=device,
                )
                _write_json(patient_path, payload)
            metadata = {
                "model": job.model,
                "model_seed": job.seed,
                "split": primary_split,
                "patient_id": patient_id,
            }
            for name in all_rows:
                _extend_with_metadata(all_rows[name], payload[name], metadata)
            if payload["matching"][0]["status"] == "PASS":
                passed += 1
            print(f"intervention {job.job_id}/{patient_id}", flush=True)
        run_jobs.append(
            {
                "job": job.job_id,
                "patient_count": len(patient_ids),
                "matched_patient_count": passed,
                "checkpoint_sha256": checkpoint_digest,
                "ood_reference": reference_metadata,
            }
        )
        adapter.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    frames = {name: pd.DataFrame(rows) for name, rows in all_rows.items()}
    output_files = {
        "matching": "intervention_matching_audit.parquet",
        "effects": "intervention_effects_by_patient.parquet",
        "subgroups": "intervention_subgroup_rollback.parquet",
        "ood": "intervention_ood_audit.parquet",
        "restore": "intervention_restore_audit.parquet",
    }
    for name, filename in output_files.items():
        frames[name].to_parquet(output_root / filename, index=False)
    statistics, conclusion = _summarize_conclusion(
        frames["effects"],
        frames["ood"],
        frames["restore"],
        trigger=trigger,
        experiment_config=experiment_config,
    )
    statistics.to_parquet(
        output_root / "intervention_patient_statistics.parquet",
        index=False,
    )
    _write_json(output_root / "conclusion_gate.json", conclusion)
    _write_json(
        output_root / "run_manifest.json",
        {
            "status": "PASS",
            "trigger": trigger,
            "jobs": run_jobs,
            "output_files": output_files,
            "conclusion_gate": conclusion,
        },
    )
    print(f"V4 complete: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
