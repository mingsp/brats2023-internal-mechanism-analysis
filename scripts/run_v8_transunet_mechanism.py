from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping

import numpy as np
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
from pptt.interventions.activation_swap import capture_module_output
from pptt.interventions.causal_runtime import trace_spatial_intervention
from pptt.interventions.causal_tracing import (
    adaptive_feature_mask,
    adaptive_feature_mean,
    dominant_feature_truth_class,
    match_feature_controls,
    persistent_retention,
    proportion_mediated,
    registered_feature_strata,
    spatial_corrupt_restore,
)
from pptt.interventions.transunet_paths import (
    InterventionPath,
    audit_path_activation_identity,
    resolve_module,
)
from pptt.io.artifacts import CaseTrace, load_case_trace
from pptt.lineage.cohorts import output_anchored_stable_correct_cohort
from pptt.lineage.direct_paths import output_anchored_direct_path_cohorts
from pptt.models.adapters import build_adapter
from pptt.models.protocol import ModelAdapter
from pptt.pipeline.trace_model import (
    ObserverPathTracer,
    SlicePathTrace,
    load_formal_reliability_threshold,
    load_real_observer_path,
)
try:
    from scripts.lock_v8_transunet_protocol import (
        load_and_validate_protocol_lock,
        sha256_file,
        sha256_named_values,
        source_tree_sha256,
    )
except ModuleNotFoundError:
    from lock_v8_transunet_protocol import (
        load_and_validate_protocol_lock,
        sha256_file,
        sha256_named_values,
        source_tree_sha256,
    )


@dataclass(frozen=True)
class ModelRuntime:
    job: ModelMatrixJob
    adapter: ModelAdapter
    tracer: ObserverPathTracer
    checkpoint_sha256: str
    reliability_threshold: float


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


def _resolve_under(workspace: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def _group_records(records: list[SliceRecord]) -> dict[str, list[SliceRecord]]:
    grouped: dict[str, list[SliceRecord]] = defaultdict(list)
    for record in records:
        grouped[patient_id_from_slice(record.slice_id)].append(record)
    return {
        patient: sorted(items, key=lambda item: item.slice_id)
        for patient, items in grouped.items()
    }


def resolve_registered_patient_ids(
    all_patient_ids: tuple[str, ...] | list[str],
    protocol_lock: Mapping[str, Any],
    *,
    registry_mode: str,
    expected_split_count: int,
    expected_formal_count: int,
) -> list[str]:
    available = sorted(str(value) for value in all_patient_ids)
    if len(available) != int(expected_split_count) or len(set(available)) != len(
        available
    ):
        raise ValueError("test-patient inventory differs from the data configuration")
    registered = [str(value) for value in protocol_lock.get("test_patient_ids", ())]
    if registry_mode == "full_split":
        if int(expected_formal_count) != int(expected_split_count):
            raise ValueError("full-split formal count differs from the split count")
        if registered != available:
            raise ValueError("full-split protocol registry differs from test data")
        return available
    if registry_mode not in {"protocol_lock_subset", "common_v8_unintervened"}:
        raise ValueError(f"unknown formal patient registry mode: {registry_mode}")
    if (
        len(registered) != int(expected_formal_count)
        or len(set(registered)) != len(registered)
        or registered != sorted(registered)
        or not set(registered).issubset(available)
    ):
        raise ValueError("locked formal subset is invalid")
    return registered


def _load_slice(record: SliceRecord) -> tuple[torch.Tensor, np.ndarray]:
    image = torch.from_numpy(
        ensure_chw(np.load(record.image_path, allow_pickle=False))
    )[None]
    truth = normalize_label(np.load(record.mask_path, allow_pickle=False))
    return image, truth


def _job_for(
    jobs: tuple[ModelMatrixJob, ...],
    *,
    model: str,
    seed: int,
) -> ModelMatrixJob:
    matches = [job for job in jobs if job.model == model and job.seed == seed]
    if len(matches) != 1:
        raise ValueError(f"expected one model-matrix job for {model}/seed_{seed}")
    return matches[0]


def _load_runtime(
    job: ModelMatrixJob,
    *,
    config: Mapping[str, Any],
    observer_root: Path,
    device: torch.device,
) -> ModelRuntime:
    if not job.checkpoint.is_file():
        raise FileNotFoundError(job.checkpoint)
    model_config = _load_yaml(job.model_config)
    nodes = tuple(str(value) for value in config["nodes"])
    observer_seeds = tuple(int(value) for value in config["observer_seeds"])
    observer_directory = observer_root / job.model / f"seed_{job.seed}"
    threshold = load_formal_reliability_threshold(observer_directory)
    observers = load_real_observer_path(
        observer_directory,
        nodes=nodes,
        seeds=observer_seeds,
    )
    adapter = build_adapter(
        job.model,
        n_channels=int(model_config["n_channels"]),
        num_classes=int(model_config["num_classes"]),
        bilinear=bool(model_config.get("bilinear", False)),
        img_size=int(model_config.get("img_size", 160)),
    )
    checkpoint_digest = adapter.load_checkpoint(job.checkpoint, map_location="cpu")
    tracer = ObserverPathTracer(
        adapter,
        observers,
        nodes=nodes,
        seeds=observer_seeds,
        reliability_threshold=threshold,
        device=device,
    )
    return ModelRuntime(
        job=job,
        adapter=adapter,
        tracer=tracer,
        checkpoint_sha256=checkpoint_digest,
        reliability_threshold=threshold,
    )


def _active_observer_bundle_sha256(
    observer_directory: Path,
    *,
    nodes: tuple[str, ...],
    observer_seeds: tuple[int, ...],
) -> str:
    required = [
        observer_directory / name
        for name in (
            "observer_control_selectivity.parquet",
            "observer_quality_by_node.parquet",
            "observer_restart_agreement.parquet",
        )
    ]
    required.extend(
        observer_directory / "observers" / node / "real" / f"seed_{seed}.pt"
        for node in nodes
        for seed in observer_seeds
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return sha256_named_values(
        {
            path.relative_to(observer_directory).as_posix(): sha256_file(path)
            for path in sorted(required)
        }
    )


def _anchor_audit(
    clean: SlicePathTrace,
    formal: CaseTrace,
    *,
    slice_index: int,
) -> dict[str, int]:
    if formal.final_model_state is None:
        raise ValueError("formal TransUNet trace lacks final model state")
    return {
        "state_mismatch_count": int(
            np.count_nonzero(clean.states != formal.states[:, slice_index])
        ),
        "reliability_mismatch_count": int(
            np.count_nonzero(clean.reliable != formal.reliable[:, slice_index])
        ),
        "final_state_mismatch_count": int(
            np.count_nonzero(
                clean.final_model_state != formal.final_model_state[slice_index]
            )
        ),
    }


def _condition_payload(
    name: str,
    traced: SlicePathTrace,
    *,
    truth: np.ndarray,
    target_pixels: np.ndarray,
    receiver_index: int,
    alpha: float | None,
    region: str,
) -> dict[str, Any]:
    final_correct = traced.final_model_state[target_pixels] == truth[target_pixels]
    final_retention = float(final_correct.mean())
    target_flat_indices = np.flatnonzero(target_pixels.reshape(-1))
    return {
        "condition": name,
        "region": region,
        "alpha": alpha,
        "primary_target_retention": final_retention,
        "primary_target_correct_count": int(final_correct.sum()),
        "primary_target_count": int(final_correct.size),
        "correct_target_flat_indices": target_flat_indices[final_correct].tolist(),
        "observer_persistent_retention": persistent_retention(
            traced.states,
            truth,
            target_pixels,
            transition_index=int(receiver_index) - 1,
        ),
    }


def _ineligible(
    *,
    patient_id: str,
    status: str,
    target_pixel_count: int,
    selected_slice_id: str | None = None,
    target_feature_count: int | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "patient_id": patient_id,
        "target_pixel_count": int(target_pixel_count),
        "selected_slice_id": selected_slice_id,
        "target_feature_count": target_feature_count,
        "specificity_evaluable": False,
        "conditions": [],
        "effects": {},
        "audit": {},
    }


def _trace_intervention(
    runtime: ModelRuntime,
    image: torch.Tensor,
    truth: np.ndarray,
    *,
    path: InterventionPath,
    restore_mask: torch.Tensor,
    shift_yx: tuple[int, int],
    alpha: float,
) -> SlicePathTrace:
    receiver = resolve_module(runtime.adapter.model, path.receiver_module_path)
    selector = (
        {"argument_name": path.argument_name}
        if path.argument_name is not None
        else {"argument_index": path.argument_index}
    )
    return trace_spatial_intervention(
        runtime.tracer,
        image,
        truth,
        module=receiver,
        restore_mask=restore_mask,
        shift_yx=shift_yx,
        alpha=alpha,
        **selector,
    )


def _run_patient(
    runtime: ModelRuntime,
    formal_trace: CaseTrace,
    records: list[SliceRecord],
    *,
    config: Mapping[str, Any],
    path: InterventionPath,
    patient_id: str,
    device: torch.device,
) -> dict[str, Any]:
    if formal_trace.truth is None or formal_trace.final_model_state is None:
        raise ValueError("formal TransUNet trace lacks truth or native output")
    expected_slices = tuple(record.slice_id for record in records)
    if formal_trace.slice_ids != expected_slices:
        raise ValueError(f"formal trace and source slices differ for {patient_id}")
    truth_classes = tuple(int(value) for value in config["truth_classes"])
    eligibility = config["eligibility"]
    matching = config["matching"]
    intervention = config["intervention"]
    cohorts = output_anchored_direct_path_cohorts(
        formal_trace.states,
        formal_trace.truth,
        formal_trace.reliable,
        formal_trace.final_model_state,
        source_index=path.source_index,
        receiver_index=path.receiver_index,
        truth_classes=truth_classes,
    )
    counts = cohorts.correction.sum(axis=(1, 2), dtype=np.int64)
    maximum_count = int(counts.max(initial=0))
    if maximum_count < int(eligibility["minimum_target_pixels_on_selected_slice"]):
        return _ineligible(
            patient_id=patient_id,
            status="INELIGIBLE_TARGET_PIXELS",
            target_pixel_count=maximum_count,
        )
    slice_index = int(np.argmax(counts))
    target_pixels = cohorts.correction[slice_index]
    record = records[slice_index]
    image, truth = _load_slice(record)
    if not np.array_equal(truth, formal_trace.truth[slice_index]):
        raise ValueError(f"formal truth and source mask differ for {record.slice_id}")
    image_device = image.to(device)
    source_module = resolve_module(runtime.adapter.model, path.source_module_path)
    with torch.inference_mode():
        _, clean_activation = capture_module_output(
            runtime.adapter,
            source_module,
            image_device,
        )
    feature_shape = tuple(int(value) for value in clean_activation.shape[-2:])
    target_feature = adaptive_feature_mask(target_pixels, output_shape=feature_shape)
    target_feature_count = int(target_feature.sum())
    if target_feature_count < int(eligibility["minimum_target_feature_positions"]):
        return _ineligible(
            patient_id=patient_id,
            status="INELIGIBLE_TARGET_FEATURE_POSITIONS",
            target_pixel_count=maximum_count,
            selected_slice_id=record.slice_id,
            target_feature_count=target_feature_count,
        )
    stable_control = output_anchored_stable_correct_cohort(
        formal_trace.states[:, slice_index],
        truth,
        formal_trace.reliable[:, slice_index],
        formal_trace.final_model_state[slice_index],
        transition_index=path.source_index,
        truth_classes=truth_classes,
        exclude=target_pixels,
    )
    control_feature = adaptive_feature_mask(
        stable_control,
        output_shape=feature_shape,
    ) & ~target_feature
    feature_truth_class = dominant_feature_truth_class(
        truth,
        output_shape=feature_shape,
        truth_classes=truth_classes,
    )
    boundary = ndimage.distance_transform_edt(
        np.isin(truth, truth_classes)
    ).astype(np.float32)
    feature_boundary = adaptive_feature_mean(boundary, output_shape=feature_shape)
    activation_norm = (
        clean_activation[0]
        .detach()
        .float()
        .square()
        .sum(dim=0)
        .sqrt()
        .cpu()
        .numpy()
    )
    boundary_stratum, activation_stratum, activation_edges = (
        registered_feature_strata(
            feature_boundary,
            activation_norm,
            eligible_mask=target_feature | control_feature,
            boundary_bin_edges=tuple(matching["boundary_bin_edges"]),
            activation_norm_quantile_bins=int(
                matching["activation_norm_quantile_bins"]
            ),
        )
    )
    matches = match_feature_controls(
        target_mask=target_feature,
        control_mask=control_feature,
        truth_class=feature_truth_class,
        boundary_distance=feature_boundary,
        activation_norm=activation_norm,
        boundary_stratum=boundary_stratum,
        activation_stratum=activation_stratum,
    )
    minimum_matches = max(
        int(eligibility["minimum_target_feature_positions"]),
        int(eligibility["minimum_control_feature_positions"]),
    )
    specificity_evaluable = matches.matched_count >= minimum_matches
    matched_target = np.zeros(feature_shape, dtype=bool)
    matched_control = np.zeros(feature_shape, dtype=bool)
    if specificity_evaluable:
        matched_target.reshape(-1)[matches.target_indices] = True
        matched_control.reshape(-1)[matches.control_indices] = True
    target_tensor = torch.from_numpy(target_feature)
    matched_target_tensor = torch.from_numpy(matched_target)
    matched_control_tensor = torch.from_numpy(matched_control)
    shift_yx = tuple(int(value) for value in intervention["feature_shift_yx"])
    receiver_module = resolve_module(
        runtime.adapter.model,
        path.receiver_module_path,
    )
    receiver_hook_count_before = len(receiver_module._forward_pre_hooks)

    clean = runtime.tracer.trace_slice(image, truth)
    anchor = _anchor_audit(clean, formal_trace, slice_index=slice_index)
    if any(anchor.values()):
        raise RuntimeError(f"clean rerun does not reproduce formal trace for {patient_id}")
    conditions = [
        _condition_payload(
            "clean",
            clean,
            truth=truth,
            target_pixels=target_pixels,
            receiver_index=path.receiver_index,
            alpha=None,
            region="none",
        )
    ]
    corrupt = _trace_intervention(
        runtime,
        image,
        truth,
        path=path,
        restore_mask=target_tensor,
        shift_yx=shift_yx,
        alpha=0.0,
    )
    conditions.append(
        _condition_payload(
            "corrupt",
            corrupt,
            truth=truth,
            target_pixels=target_pixels,
            receiver_index=path.receiver_index,
            alpha=0.0,
            region="full_path_shift",
        )
    )
    target_traces: dict[float, SlicePathTrace] = {}
    for alpha_value in intervention["restoration_alphas"]:
        alpha = float(alpha_value)
        traced = _trace_intervention(
            runtime,
            image,
            truth,
            path=path,
            restore_mask=target_tensor,
            shift_yx=shift_yx,
            alpha=alpha,
        )
        target_traces[alpha] = traced
        conditions.append(
            _condition_payload(
                f"restore_target_{alpha:.2f}",
                traced,
                truth=truth,
                target_pixels=target_pixels,
                receiver_index=path.receiver_index,
                alpha=alpha,
                region="target",
            )
        )
    matched_target_trace: SlicePathTrace | None = None
    control_trace: SlicePathTrace | None = None
    if specificity_evaluable:
        matched_target_trace = _trace_intervention(
            runtime,
            image,
            truth,
            path=path,
            restore_mask=matched_target_tensor,
            shift_yx=shift_yx,
            alpha=1.0,
        )
        conditions.append(
            _condition_payload(
                "restore_target_matched_1.00",
                matched_target_trace,
                truth=truth,
                target_pixels=target_pixels,
                receiver_index=path.receiver_index,
                alpha=1.0,
                region="matched_target",
            )
        )
        control_trace = _trace_intervention(
            runtime,
            image,
            truth,
            path=path,
            restore_mask=matched_control_tensor,
            shift_yx=shift_yx,
            alpha=1.0,
        )
        conditions.append(
            _condition_payload(
                "restore_control_matched_1.00",
                control_trace,
                truth=truth,
                target_pixels=target_pixels,
                receiver_index=path.receiver_index,
                alpha=1.0,
                region="matched_control",
            )
        )
    q_by_condition = {
        str(row["condition"]): float(row["primary_target_retention"])
        for row in conditions
    }
    primary_alpha = float(intervention["primary_alpha"])
    q_clean = q_by_condition["clean"]
    q_corrupt = q_by_condition["corrupt"]
    q_target = q_by_condition[f"restore_target_{primary_alpha:.2f}"]
    q_matched_target = q_by_condition.get("restore_target_matched_1.00", np.nan)
    q_control = q_by_condition.get("restore_control_matched_1.00", np.nan)

    full_restore = spatial_corrupt_restore(
        clean_activation,
        restore_mask=torch.ones(feature_shape, dtype=torch.bool),
        shift_yx=shift_yx,
        alpha=1.0,
    )
    target_restore = spatial_corrupt_restore(
        clean_activation,
        restore_mask=target_tensor,
        shift_yx=shift_yx,
        alpha=1.0,
    )
    rolled = torch.roll(clean_activation, shifts=shift_yx, dims=(-2, -1))
    sorted_clean = torch.sort(clean_activation.flatten(2), dim=2).values
    sorted_rolled = torch.sort(rolled.flatten(2), dim=2).values
    target_index = target_tensor.to(
        device=clean_activation.device,
        dtype=torch.bool,
    )
    target_activation_error = float(
        (target_restore - clean_activation).abs()[0, :, target_index].max().item()
    )
    receiver_hook_count_after = len(receiver_module._forward_pre_hooks)
    return {
        "status": "PASS",
        "patient_id": patient_id,
        "selected_slice_id": record.slice_id,
        "selected_slice_index": slice_index,
        "path_id": path.path_id,
        "target_pixel_count": int(target_pixels.sum()),
        "output_shape": list(target_pixels.shape),
        "target_pixel_flat_indices": np.flatnonzero(
            target_pixels.reshape(-1)
        ).tolist(),
        "target_feature_count": target_feature_count,
        "control_feature_pool_count": int(control_feature.sum()),
        "matched_feature_count": matches.matched_count,
        "specificity_evaluable": specificity_evaluable,
        "feature_shape": list(feature_shape),
        "activation_norm_quantile_edges": activation_edges,
        "conditions": conditions,
        "effects": {
            "necessity": q_clean - q_corrupt,
            "restoration": q_target - q_corrupt,
            "specificity": (
                float(q_matched_target - q_control)
                if specificity_evaluable
                else None
            ),
            "proportion_mediated": proportion_mediated(
                clean=q_clean,
                corrupt=q_corrupt,
                restored=q_target,
            ),
        },
        "audit": {
            "clean_anchor": anchor,
            "clean_primary_retention": q_clean,
            "full_restore_max_abs_error": float(
                (full_restore - clean_activation).abs().max().item()
            ),
            "target_alpha_1_max_abs_error": target_activation_error,
            "spatial_shift_channel_multiset_max_abs_error": float(
                (sorted_clean - sorted_rolled).abs().max().item()
            ),
            "receiver_hook_count_before": receiver_hook_count_before,
            "receiver_hook_count_after": receiver_hook_count_after,
            "receiver_hook_count_delta": (
                receiver_hook_count_after - receiver_hook_count_before
            ),
        },
    }


def _progress_payload(
    *,
    seed: int,
    started_at: str,
    started_monotonic: float,
    completed: int,
    total: int,
    last_patient_id: str | None,
    status_counts: Mapping[str, int],
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    elapsed = max(0.0, time.monotonic() - started_monotonic)
    throughput = completed / elapsed if elapsed > 0 and completed > 0 else 0.0
    eta = (total - completed) / throughput if throughput > 0 else None
    return {
        "phase": "v8_transunet_formal_test_intervention",
        "status": status,
        "model_seed": int(seed),
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed_patients": int(completed),
        "total_patients": int(total),
        "last_patient_id": last_patient_id,
        "status_counts": dict(sorted(status_counts.items())),
        "elapsed_seconds": elapsed,
        "patients_per_second": throughput,
        "eta_seconds": eta,
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the locked TransUNet V8 mechanism replication."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/v8_transunet_mechanism_replication.yaml"
        ),
    )
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path("results/v8_transunet_mechanism/v8_protocol_lock.json"),
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/brats2023_2d.yaml"),
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--observer-root", type=Path)
    parser.add_argument("--test-trace-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--execution-mode",
        choices=("formal", "smoke"),
        default="formal",
    )
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--smoke-patient-id")
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    config_path = _resolve_under(workspace, args.config)
    lock_path = _resolve_under(workspace, args.protocol_lock)
    data_config_path = _resolve_under(workspace, args.data_config)
    config = _load_yaml(config_path)
    data_config = _load_yaml(data_config_path)
    if args.execution_mode == "formal" and (
        args.max_patients is not None or args.smoke_patient_id is not None
    ):
        raise ValueError("formal V8 cannot truncate or select the patient registry")
    if args.execution_mode == "smoke" and args.max_patients is None:
        raise ValueError("smoke mode requires --max-patients")
    matrix = load_model_matrix(
        workspace / str(config["model_matrix"]),
        workspace_root=workspace,
        asset_root_override=args.asset_root,
    )
    split = str(config["intervention_split"])
    split_config = data_config["splits"][split]
    grouped = _group_records(
        discover_slice_records(
            matrix.asset_root / str(split_config["image_dir"]),
            matrix.asset_root / str(split_config["mask_dir"]),
        )
    )
    preliminary_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(preliminary_lock, dict):
        raise ValueError("protocol lock must contain a JSON object")
    formal_config = config.get("formal", {})
    patient_ids = resolve_registered_patient_ids(
        sorted(grouped),
        preliminary_lock,
        registry_mode=str(formal_config.get("patient_registry_mode", "full_split")),
        expected_split_count=int(split_config["patient_count"]),
        expected_formal_count=int(
            formal_config.get(
                "expected_patient_count_per_seed",
                split_config["patient_count"],
            )
        ),
    )
    lock = load_and_validate_protocol_lock(
        lock_path,
        active_configuration_sha256=sha256_file(config_path),
        active_source_tree_sha256=source_tree_sha256(workspace),
        patient_ids=patient_ids,
        model_seed=args.model_seed,
    )
    path = InterventionPath(**lock["candidate"]["path"])
    if args.execution_mode == "smoke":
        if args.smoke_patient_id is not None:
            if args.smoke_patient_id not in grouped:
                raise ValueError("smoke patient is absent from the test split")
            patient_ids = [str(args.smoke_patient_id)]
        else:
            patient_ids = patient_ids[: int(args.max_patients)]
    configured_output = workspace / str(config["formal_output_root"])
    output_root = (
        _resolve_under(workspace, args.output_root)
        if args.output_root is not None
        else configured_output.resolve()
    )
    if args.execution_mode == "formal" and output_root != configured_output.resolve():
        raise ValueError("formal V8 output root must match the locked configuration")
    if args.execution_mode == "smoke" and output_root == configured_output.resolve():
        raise ValueError("smoke output cannot contaminate the formal V8 directory")
    model = str(config["model"])
    job = _job_for(matrix.jobs, model=model, seed=args.model_seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    observer_root = (
        _resolve_under(workspace, args.observer_root)
        if args.observer_root is not None
        else (workspace / str(config["observer_root"])).resolve()
    )
    runtime = _load_runtime(
        job,
        config=config,
        observer_root=observer_root,
        device=device,
    )
    locked_model = next(
        row
        for row in lock["registered_model_jobs"]
        if int(row["model_seed"]) == args.model_seed
    )
    if runtime.checkpoint_sha256 != locked_model["checkpoint_sha256"]:
        raise ValueError("active TransUNet checkpoint differs from the V8 lock")
    locked_observer = next(
        row
        for row in lock["registered_observer_jobs"]
        if int(row["model_seed"]) == args.model_seed
    )
    observer_directory = observer_root / model / f"seed_{args.model_seed}"
    observer_hash = _active_observer_bundle_sha256(
        observer_directory,
        nodes=tuple(str(value) for value in config["nodes"]),
        observer_seeds=tuple(int(value) for value in config["observer_seeds"]),
    )
    if observer_hash != locked_observer["observer_bundle_sha256"]:
        raise ValueError("active observer bundle differs from the V8 lock")
    trace_root = (
        _resolve_under(workspace, args.test_trace_root)
        if args.test_trace_root is not None
        else (workspace / str(config["test_trace_root"])).resolve()
    )
    first_image, _ = _load_slice(grouped[patient_ids[0]][0])
    graph_identity = audit_path_activation_identity(
        runtime.adapter,
        path,
        first_image.to(device),
    )
    if not graph_identity["exact_equal"] or graph_identity["max_abs_error"] != 0.0:
        raise RuntimeError("registered graph path failed the runtime identity audit")
    job_output = output_root / model / f"seed_{args.model_seed}"
    run_identity = {
        "execution_mode": args.execution_mode,
        "model_seed": args.model_seed,
        "path": path.to_dict(),
        "protocol_lock_sha256": sha256_file(lock_path),
        "configuration_sha256": sha256_file(config_path),
        "source_tree_sha256": source_tree_sha256(workspace),
        "checkpoint_sha256": runtime.checkpoint_sha256,
        "observer_bundle_sha256": observer_hash,
    }
    manifest = {
        "schema_version": 1,
        **run_identity,
        "patient_count": len(patient_ids),
        "reliability_threshold": runtime.reliability_threshold,
        "graph_identity_audit": graph_identity,
    }
    manifest_path = job_output / "job_manifest.json"
    if args.resume and manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != _json_ready(manifest):
            raise ValueError("V8 resume identity differs from the locked job")
    else:
        _write_json_atomic(manifest_path, manifest)

    started_at = datetime.now(timezone.utc).isoformat()
    started_monotonic = time.monotonic()
    counts: dict[str, int] = defaultdict(int)
    progress_path = job_output / "progress.json"
    try:
        for index, patient_id in enumerate(patient_ids, start=1):
            patient_path = job_output / "patient_results" / f"{patient_id}.json"
            if args.resume and patient_path.is_file():
                payload = json.loads(patient_path.read_text(encoding="utf-8"))
                if payload.get("run_identity") != _json_ready(run_identity):
                    raise ValueError(f"V8 resume identity differs for {patient_id}")
            else:
                trace_path = (
                    trace_root
                    / model
                    / f"seed_{args.model_seed}"
                    / split
                    / "case_traces"
                    / f"{patient_id}.npz"
                )
                if not trace_path.is_file():
                    raise FileNotFoundError(trace_path)
                payload = _run_patient(
                    runtime,
                    load_case_trace(trace_path),
                    grouped[patient_id],
                    config=config,
                    path=path,
                    patient_id=patient_id,
                    device=device,
                )
                payload["run_identity"] = run_identity
                _write_json_atomic(patient_path, payload)
            counts[str(payload["status"])] += 1
            progress = _progress_payload(
                seed=args.model_seed,
                started_at=started_at,
                started_monotonic=started_monotonic,
                completed=index,
                total=len(patient_ids),
                last_patient_id=patient_id,
                status_counts=counts,
                status="RUNNING" if index < len(patient_ids) else "COMPLETE",
            )
            _write_json_atomic(progress_path, progress)
            print(
                f"v8 seed={args.model_seed} {index}/{len(patient_ids)} "
                f"patient={patient_id} status={payload['status']} "
                f"eta_s={progress['eta_seconds']}",
                flush=True,
            )
    except Exception as error:
        _write_json_atomic(
            progress_path,
            _progress_payload(
                seed=args.model_seed,
                started_at=started_at,
                started_monotonic=started_monotonic,
                completed=sum(counts.values()),
                total=len(patient_ids),
                last_patient_id=None,
                status_counts=counts,
                status="FAILED",
                error=f"{type(error).__name__}: {error}",
            ),
        )
        raise
    tolerance = float(config["statistics"]["activation_restore_tolerance"])
    completed_payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((job_output / "patient_results").glob("*.json"))
    ]
    pass_payloads = [
        payload for payload in completed_payloads if payload.get("status") == "PASS"
    ]
    patient_audits_pass = bool(
        all(
            sum(payload["audit"]["clean_anchor"].values()) == 0
            and float(payload["audit"]["clean_primary_retention"]) == 1.0
            and float(payload["audit"]["full_restore_max_abs_error"]) <= tolerance
            and float(payload["audit"]["target_alpha_1_max_abs_error"]) <= tolerance
            and float(
                payload["audit"]["spatial_shift_channel_multiset_max_abs_error"]
            )
            <= tolerance
            and int(payload["audit"]["receiver_hook_count_delta"]) == 0
            for payload in pass_payloads
        )
    )
    operator_audit_pass = bool(
        graph_identity["exact_equal"]
        and graph_identity["max_abs_error"] == 0.0
        and graph_identity["source_hook_count_after"] == 0
        and graph_identity["receiver_hook_count_after"] == 0
        and patient_audits_pass
    )
    status = {
        "status": (
            (
                "COMPLETE" if operator_audit_pass else "FAILED_AUDIT"
            )
            if args.execution_mode == "formal"
            else (
                "SMOKE_COMPLETE_NOT_FOR_CONCLUSION"
                if operator_audit_pass
                else "SMOKE_FAILED_AUDIT_NOT_FOR_CONCLUSION"
            )
        ),
        "execution_mode": args.execution_mode,
        "model_seed": args.model_seed,
        "patient_count": len(patient_ids),
        "status_counts": dict(sorted(counts.items())),
        "path": path.to_dict(),
        "protocol_lock_sha256": sha256_file(lock_path),
        "graph_identity_audit": graph_identity,
        "audited_patient_count": len(pass_payloads),
        "patient_operator_audit_pass": patient_audits_pass,
        "operator_audit_pass": operator_audit_pass,
    }
    _write_json_atomic(job_output / "job_status.json", status)
    runtime.adapter.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(json.dumps(status, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
