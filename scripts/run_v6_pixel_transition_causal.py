from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

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
    persistent_correction_mask,
    persistent_retention,
    proportion_mediated,
    registered_feature_strata,
    select_target_slice,
    spatial_corrupt_restore,
    stable_correct_control_mask,
)
from pptt.io.artifacts import CaseTrace, load_case_trace
from pptt.models.adapters import build_adapter
from pptt.models.protocol import ModelAdapter
from pptt.pipeline.trace_model import (
    ObserverPathTracer,
    SlicePathTrace,
    load_formal_reliability_threshold,
    load_real_observer_path,
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            _json_ready(payload),
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


def _group_records(records: list[SliceRecord]) -> dict[str, list[SliceRecord]]:
    grouped: dict[str, list[SliceRecord]] = defaultdict(list)
    for record in records:
        grouped[patient_id_from_slice(record.slice_id)].append(record)
    return {
        patient: sorted(items, key=lambda item: item.slice_id)
        for patient, items in grouped.items()
    }


def _load_slice(record: SliceRecord) -> tuple[torch.Tensor, np.ndarray]:
    image = torch.from_numpy(
        ensure_chw(np.load(record.image_path, allow_pickle=False))
    )[None]
    truth = normalize_label(np.load(record.mask_path, allow_pickle=False))
    return image, truth


def _resolve_under(workspace: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def _validate_protocol_lock(
    *,
    lock_path: Path,
    config_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_BEFORE_FIRST_INTERVENTION":
        raise ValueError("V6 protocol is not locked before intervention")
    if not bool(lock.get("intervention_authorized", False)):
        raise ValueError("V6 protocol lock does not authorize intervention")
    if not bool(lock.get("candidate_specific_validation_gate", {}).get("passed")):
        raise ValueError("candidate-specific validation gate is not PASS")
    if lock.get("configuration_sha256") != _sha256(config_path):
        raise ValueError("V6 configuration changed after protocol lock")
    if lock.get("configuration") != _json_ready(config):
        raise ValueError("V6 lock payload and active configuration differ")
    target = config["target"]
    expected_identity = {
        "transition": str(target["transition"]),
        "transition_index": int(target["transition_index"]),
        "module": str(target["module"]),
        "argument_index": int(target["argument_index"]),
        "source": str(target["source"]),
    }
    if lock.get("candidate_identity") != expected_identity:
        raise ValueError("V6 candidate identity changed after validation selection")
    amendment = lock.get("supersedes", {})
    if (
        amendment.get("scope") != "observer_seed_identity_only"
        or not bool(amendment.get("before_first_intervention", False))
    ):
        raise ValueError("V6 v2 observer identity amendment is not auditable")
    if "global_two_metric_gate_passed" not in lock:
        raise ValueError("global two-metric gate must remain reported")
    return lock


def _load_runtime(
    job: ModelMatrixJob,
    *,
    config: dict[str, Any],
    observer_root: Path,
    device: torch.device,
) -> ModelRuntime:
    if not job.checkpoint.is_file():
        raise FileNotFoundError(job.checkpoint)
    model_config = _load_yaml(job.model_config)
    observer_directory = observer_root / job.model / f"seed_{job.seed}"
    nodes = tuple(str(value) for value in config["nodes"])
    observer_seeds = tuple(int(value) for value in config["observer_seeds"])
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


def _condition_payload(
    name: str,
    traced: SlicePathTrace,
    *,
    truth: np.ndarray,
    target_pixels: np.ndarray,
    transition_index: int,
    alpha: float | None,
    region: str,
) -> dict[str, Any]:
    return {
        "condition": name,
        "region": region,
        "alpha": alpha,
        "persistent_retention": persistent_retention(
            traced.states,
            truth,
            target_pixels,
            transition_index=transition_index,
        ),
        "final_model_target_retention": float(
            (traced.final_model_state[target_pixels] == truth[target_pixels]).mean()
        ),
    }


def _anchor_matches_formal_trace(
    clean: SlicePathTrace,
    formal: CaseTrace,
    *,
    slice_index: int,
) -> dict[str, int]:
    if formal.final_model_state is None:
        raise ValueError("formal V3 trace lacks final model state")
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


def _ineligible_payload(
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
        "target_pixel_count": target_pixel_count,
        "selected_slice_id": selected_slice_id,
        "target_feature_count": target_feature_count,
        "conditions": [],
        "effects": {},
        "negative_control": {},
    }


def _run_patient(
    baseline: ModelRuntime,
    no_skip: ModelRuntime,
    formal_trace: CaseTrace,
    records: list[SliceRecord],
    *,
    config: dict[str, Any],
    patient_id: str,
    device: torch.device,
) -> dict[str, Any]:
    if formal_trace.truth is None:
        raise ValueError("formal V3 trace lacks truth")
    expected_slices = tuple(record.slice_id for record in records)
    if formal_trace.slice_ids != expected_slices:
        raise ValueError(f"formal trace and source slices differ for {patient_id}")
    target_config = config["target"]
    eligibility = config["eligibility"]
    intervention = config["intervention"]
    matching = config["matching"]
    transition_index = int(target_config["transition_index"])
    truth_classes = tuple(int(value) for value in target_config["truth_classes"])
    target_path = persistent_correction_mask(
        formal_trace.states,
        formal_trace.truth,
        transition_index=transition_index,
        reliable=formal_trace.reliable,
        truth_classes=truth_classes,
    )
    counts = target_path.sum(axis=(1, 2))
    maximum_count = int(counts.max(initial=0))
    if maximum_count < int(eligibility["minimum_target_pixels_on_selected_slice"]):
        return _ineligible_payload(
            patient_id=patient_id,
            status="INELIGIBLE_TARGET_PIXELS",
            target_pixel_count=maximum_count,
        )
    slice_index = select_target_slice(target_path)
    record = records[slice_index]
    target_pixels = target_path[slice_index]
    image, truth = _load_slice(record)
    if not np.array_equal(truth, formal_trace.truth[slice_index]):
        raise ValueError(f"formal truth and source mask differ for {record.slice_id}")

    with torch.inference_mode():
        _, clean_activation = capture_module_output(
            baseline.adapter,
            getattr(baseline.adapter.model, str(target_config["source"])),
            image.to(device),
        )
    feature_shape = tuple(int(value) for value in clean_activation.shape[-2:])
    target_feature = adaptive_feature_mask(
        target_pixels,
        output_shape=feature_shape,
    )
    target_feature_count = int(target_feature.sum())
    if target_feature_count < int(eligibility["minimum_target_feature_positions"]):
        return _ineligible_payload(
            patient_id=patient_id,
            status="INELIGIBLE_TARGET_FEATURE_POSITIONS",
            target_pixel_count=maximum_count,
            selected_slice_id=record.slice_id,
            target_feature_count=target_feature_count,
        )

    stable_control = stable_correct_control_mask(
        formal_trace.states[:, slice_index],
        truth,
        transition_index=transition_index,
        reliable=formal_trace.reliable[:, slice_index],
        truth_classes=truth_classes,
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
    specificity_evaluable = bool(
        matches.matched_count
        >= max(
            int(eligibility["minimum_target_feature_positions"]),
            int(eligibility["minimum_control_feature_positions"]),
        )
    )
    matched_target_restore = np.zeros(feature_shape, dtype=bool)
    control_restore = np.zeros(feature_shape, dtype=bool)
    if specificity_evaluable:
        matched_target_restore.reshape(-1)[matches.target_indices] = True
        control_restore.reshape(-1)[matches.control_indices] = True
    target_restore_tensor = torch.from_numpy(target_feature)
    matched_target_restore_tensor = torch.from_numpy(matched_target_restore)
    control_restore_tensor = torch.from_numpy(control_restore)
    shift_yx = tuple(int(value) for value in intervention["feature_shift_yx"])

    clean = baseline.tracer.trace_slice(image, truth)
    anchor_audit = _anchor_matches_formal_trace(
        clean,
        formal_trace,
        slice_index=slice_index,
    )
    if any(anchor_audit.values()):
        raise RuntimeError(
            f"clean rerun does not reproduce formal V3 trace for {patient_id}"
        )
    conditions = [
        _condition_payload(
            "clean",
            clean,
            truth=truth,
            target_pixels=target_pixels,
            transition_index=transition_index,
            alpha=None,
            region="none",
        )
    ]
    corrupt = trace_spatial_intervention(
        baseline.tracer,
        image,
        truth,
        module=getattr(baseline.adapter.model, str(target_config["module"])),
        argument_index=int(target_config["argument_index"]),
        restore_mask=target_restore_tensor,
        shift_yx=shift_yx,
        alpha=0.0,
    )
    conditions.append(
        _condition_payload(
            "corrupt",
            corrupt,
            truth=truth,
            target_pixels=target_pixels,
            transition_index=transition_index,
            alpha=0.0,
            region="full_path_shift",
        )
    )
    target_traces: dict[float, SlicePathTrace] = {}
    for alpha_value in intervention["restoration_alphas"]:
        alpha = float(alpha_value)
        traced = trace_spatial_intervention(
            baseline.tracer,
            image,
            truth,
            module=getattr(baseline.adapter.model, str(target_config["module"])),
            argument_index=int(target_config["argument_index"]),
            restore_mask=target_restore_tensor,
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
                transition_index=transition_index,
                alpha=alpha,
                region="target",
            )
        )

    matched_target_trace: SlicePathTrace | None = None
    control_trace: SlicePathTrace | None = None
    if specificity_evaluable:
        matched_target_trace = trace_spatial_intervention(
            baseline.tracer,
            image,
            truth,
            module=getattr(baseline.adapter.model, str(target_config["module"])),
            argument_index=int(target_config["argument_index"]),
            restore_mask=matched_target_restore_tensor,
            shift_yx=shift_yx,
            alpha=1.0,
        )
        conditions.append(
            _condition_payload(
                "restore_target_matched_1.00",
                matched_target_trace,
                truth=truth,
                target_pixels=target_pixels,
                transition_index=transition_index,
                alpha=1.0,
                region="matched_target",
            )
        )
        control_trace = trace_spatial_intervention(
            baseline.tracer,
            image,
            truth,
            module=getattr(baseline.adapter.model, str(target_config["module"])),
            argument_index=int(target_config["argument_index"]),
            restore_mask=control_restore_tensor,
            shift_yx=shift_yx,
            alpha=1.0,
        )
        conditions.append(
            _condition_payload(
                "restore_control_matched_1.00",
                control_trace,
                truth=truth,
                target_pixels=target_pixels,
                transition_index=transition_index,
                alpha=1.0,
                region="matched_control",
            )
        )

    q_clean = float(conditions[0]["persistent_retention"])
    q_corrupt = float(conditions[1]["persistent_retention"])
    primary_alpha = float(intervention["primary_alpha"])
    q_target = persistent_retention(
        target_traces[primary_alpha].states,
        truth,
        target_pixels,
        transition_index=transition_index,
    )
    q_control = (
        persistent_retention(
            control_trace.states,
            truth,
            target_pixels,
            transition_index=transition_index,
        )
        if control_trace is not None
        else float("nan")
    )
    q_matched_target = (
        persistent_retention(
            matched_target_trace.states,
            truth,
            target_pixels,
            transition_index=transition_index,
        )
        if matched_target_trace is not None
        else float("nan")
    )
    total_effect = q_clean - q_corrupt
    indirect_target = q_target - q_corrupt
    indirect_matched_target = q_matched_target - q_corrupt
    indirect_control = q_control - q_corrupt

    with torch.inference_mode():
        restored_activation = spatial_corrupt_restore(
            clean_activation,
            restore_mask=target_restore_tensor,
            shift_yx=shift_yx,
            alpha=1.0,
        )
    activation_difference = (restored_activation - clean_activation).abs()[
        0, :, target_feature
    ]
    target_activation_error = float(activation_difference.max().item())

    no_skip_clean = no_skip.tracer.trace_slice(image, truth)
    no_skip_changed = trace_spatial_intervention(
        no_skip.tracer,
        image,
        truth,
        module=getattr(no_skip.adapter.model, str(target_config["module"])),
        argument_index=int(target_config["argument_index"]),
        restore_mask=target_restore_tensor,
        shift_yx=shift_yx,
        alpha=0.0,
    )
    if no_skip_clean.final_logits is None or no_skip_changed.final_logits is None:
        raise RuntimeError("no-skip trace lacks final logits")
    negative_control = {
        "max_abs_logit_error": float(
            np.max(np.abs(no_skip_clean.final_logits - no_skip_changed.final_logits))
        ),
        "state_mismatch_count": int(
            np.count_nonzero(no_skip_clean.states != no_skip_changed.states)
        ),
        "reliability_mismatch_count": int(
            np.count_nonzero(no_skip_clean.reliable != no_skip_changed.reliable)
        ),
        "final_state_mismatch_count": int(
            np.count_nonzero(
                no_skip_clean.final_model_state
                != no_skip_changed.final_model_state
            )
        ),
        "hook_count_after": len(
            getattr(no_skip.adapter.model, str(target_config["module"]))._forward_pre_hooks
        ),
    }

    return {
        "status": "PASS",
        "patient_id": patient_id,
        "selected_slice_id": record.slice_id,
        "selected_slice_index": slice_index,
        "target_pixel_count": int(target_pixels.sum()),
        "target_feature_count": target_feature_count,
        "control_feature_pool_count": int(control_feature.sum()),
        "matched_feature_count": matches.matched_count,
        "specificity_evaluable": specificity_evaluable,
        "feature_shape": feature_shape,
        "activation_norm_quantile_edges": activation_edges,
        "conditions": conditions,
        "effects": {
            "total_effect": total_effect,
            "target_indirect_effect": indirect_target,
            "matched_target_indirect_effect": indirect_matched_target,
            "control_indirect_effect": indirect_control,
            "target_minus_control": indirect_matched_target - indirect_control,
            "target_proportion_mediated": proportion_mediated(
                clean=q_clean,
                corrupt=q_corrupt,
                restored=q_target,
            ),
        },
        "negative_control": negative_control,
        "audit": {
            "clean_anchor": anchor_audit,
            "target_alpha_1_activation_max_abs_error": target_activation_error,
            "baseline_hook_count_after": len(
                getattr(
                    baseline.adapter.model,
                    str(target_config["module"]),
                )._forward_pre_hooks
            ),
        },
    }


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the locked V6 pixel-transition causal tracing protocol."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/v6_pixel_transition_causal_tracing.yaml"),
    )
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path("results/v6_pixel_transition_causal/v6_protocol_lock_v2.json"),
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/brats2023_2d.yaml"),
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--v3-root", type=Path)
    parser.add_argument("--observer-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execution-mode", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--smoke-patient-id")
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    config_path = _resolve_under(workspace, args.config)
    data_config_path = _resolve_under(workspace, args.data_config)
    lock_path = _resolve_under(workspace, args.protocol_lock)
    config = _load_yaml(config_path)
    data_config = _load_yaml(data_config_path)
    lock = _validate_protocol_lock(
        lock_path=lock_path,
        config_path=config_path,
        config=config,
    )
    registered_seeds = tuple(int(value) for value in config["model_seeds"])
    if args.model_seed not in registered_seeds:
        raise ValueError("model seed is outside the locked V6 protocol")
    if args.execution_mode == "formal" and args.max_patients is not None:
        raise ValueError("formal V6 cannot truncate the registered patient set")
    if args.execution_mode == "formal" and args.smoke_patient_id is not None:
        raise ValueError("formal V6 cannot select a patient")
    if args.execution_mode == "smoke" and args.max_patients is None:
        raise ValueError("smoke mode requires --max-patients")

    configured_output = workspace / str(config["output_root"])
    output_root = (
        _resolve_under(workspace, args.output_root)
        if args.output_root is not None
        else configured_output.resolve()
    )
    if args.execution_mode == "formal" and output_root != configured_output.resolve():
        raise ValueError("formal V6 output root must match the locked configuration")
    if args.execution_mode == "smoke" and output_root == configured_output.resolve():
        raise ValueError("smoke output cannot contaminate the formal V6 directory")

    matrix = load_model_matrix(
        workspace / str(config["model_matrix"]),
        workspace_root=workspace,
        asset_root_override=args.asset_root,
    )
    causal_model = str(config["models"]["causal"])
    control_model = str(config["models"]["negative_control"])
    baseline_job = _job_for(
        matrix.jobs,
        model=causal_model,
        seed=args.model_seed,
    )
    no_skip_job = _job_for(
        matrix.jobs,
        model=control_model,
        seed=args.model_seed,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    observer_root = (
        _resolve_under(workspace, args.observer_root)
        if args.observer_root is not None
        else (workspace / str(config["observer_root"])).resolve()
    )
    v3_root = (
        _resolve_under(workspace, args.v3_root)
        if args.v3_root is not None
        else (workspace / str(config["v3_root"])).resolve()
    )
    baseline = _load_runtime(
        baseline_job,
        config=config,
        observer_root=observer_root,
        device=device,
    )
    no_skip = _load_runtime(
        no_skip_job,
        config=config,
        observer_root=observer_root,
        device=device,
    )

    split = str(config["primary_split"])
    split_config = data_config["splits"][split]
    grouped = _group_records(
        discover_slice_records(
            matrix.asset_root / str(split_config["image_dir"]),
            matrix.asset_root / str(split_config["mask_dir"]),
        )
    )
    if len(grouped) != int(split_config["patient_count"]):
        raise ValueError("V6 test-patient inventory differs from the locked data config")
    patient_ids = sorted(grouped)
    if args.smoke_patient_id is not None:
        if args.smoke_patient_id not in grouped:
            raise ValueError("smoke patient is absent from the registered test split")
        patient_ids = [str(args.smoke_patient_id)]
    elif args.max_patients is not None:
        patient_ids = patient_ids[: int(args.max_patients)]

    job_output = output_root / causal_model / f"seed_{args.model_seed}"
    manifest = {
        "schema_version": 1,
        "execution_mode": args.execution_mode,
        "model_seed": args.model_seed,
        "patient_count": len(patient_ids),
        "protocol_lock": lock_path,
        "protocol_lock_sha256": _sha256(lock_path),
        "global_two_metric_gate_passed": bool(
            lock["global_two_metric_gate_passed"]
        ),
        "baseline_checkpoint": baseline_job.checkpoint,
        "baseline_checkpoint_sha256": baseline.checkpoint_sha256,
        "no_skip_checkpoint": no_skip_job.checkpoint,
        "no_skip_checkpoint_sha256": no_skip.checkpoint_sha256,
        "observer_seeds": config["observer_seeds"],
        "baseline_reliability_threshold": baseline.reliability_threshold,
        "no_skip_reliability_threshold": no_skip.reliability_threshold,
    }
    manifest_path = job_output / "job_manifest.json"
    if args.resume and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != _json_ready(manifest):
            raise ValueError("V6 resume identity differs from the locked job")
    else:
        _write_json_atomic(manifest_path, manifest)

    counts: dict[str, int] = defaultdict(int)
    for patient_index, patient_id in enumerate(patient_ids, start=1):
        patient_path = job_output / "patient_results" / f"{patient_id}.json"
        if args.resume and patient_path.is_file():
            payload = json.loads(patient_path.read_text(encoding="utf-8"))
        else:
            trace_path = (
                v3_root
                / causal_model
                / f"seed_{args.model_seed}"
                / split
                / "case_traces"
                / f"{patient_id}.npz"
            )
            if not trace_path.is_file():
                raise FileNotFoundError(trace_path)
            payload = _run_patient(
                baseline,
                no_skip,
                load_case_trace(trace_path),
                grouped[patient_id],
                config=config,
                patient_id=patient_id,
                device=device,
            )
            _write_json_atomic(patient_path, payload)
        counts[str(payload["status"])] += 1
        print(
            f"v6 seed={args.model_seed} {patient_index}/{len(patient_ids)} "
            f"patient={patient_id} status={payload['status']}",
            flush=True,
        )

    status = {
        "status": "COMPLETE" if args.execution_mode == "formal" else "SMOKE_COMPLETE",
        "execution_mode": args.execution_mode,
        "model_seed": args.model_seed,
        "patient_count": len(patient_ids),
        "status_counts": dict(sorted(counts.items())),
        "protocol_lock_sha256": _sha256(lock_path),
    }
    _write_json_atomic(job_output / "job_status.json", status)
    print(json.dumps(status, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
