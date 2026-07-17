from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
import torch

from pptt.data.brats2d import discover_slice_records
from pptt.experiments.model_matrix import load_model_matrix
from pptt.interventions.activation_swap import capture_module_output
from pptt.interventions.causal_runtime import trace_spatial_intervention
from pptt.interventions.causal_tracing import (
    adaptive_feature_mask,
    adaptive_feature_mean,
    dominant_feature_truth_class,
    match_feature_controls,
    persistent_correction_mask,
    persistent_retention,
    registered_feature_strata,
    select_target_slice,
    stable_correct_control_mask,
)
from pptt.io.artifacts import load_case_trace
from pptt.visualization.causal import display_channel

from run_v6_pixel_transition_causal import (
    _group_records,
    _job_for,
    _load_runtime,
    _load_slice,
    _load_yaml,
    _resolve_under,
    _validate_protocol_lock,
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _condition_by_name(payload: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in payload["conditions"] if item["condition"] == name]
    if len(matches) != 1:
        raise ValueError(f"formal patient result lacks unique condition: {name}")
    return matches[0]


def _assert_close(actual: float, expected: float, *, label: str) -> float:
    error = abs(float(actual) - float(expected))
    if error > 1.0e-10:
        raise RuntimeError(f"representative rerun differs for {label}: {error}")
    return error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the locked representative V6 case for the causal figure."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
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
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    config_path = _resolve_under(workspace, args.config)
    lock_path = _resolve_under(workspace, args.protocol_lock)
    data_config_path = _resolve_under(workspace, args.data_config)
    config = _load_yaml(config_path)
    data_config = _load_yaml(data_config_path)
    _validate_protocol_lock(
        lock_path=lock_path,
        config_path=config_path,
        config=config,
    )
    output_root = (workspace / str(config["output_root"])).resolve()
    representative_path = output_root / "causal_representative_case.json"
    representative = json.loads(representative_path.read_text(encoding="utf-8"))
    model_seed = int(representative["model_seed"])
    patient_id = str(representative["patient_id"])
    selected_slice_id = str(representative["selected_slice_id"])
    stored_path = (
        output_root
        / str(config["models"]["causal"])
        / f"seed_{model_seed}"
        / "patient_results"
        / f"{patient_id}.json"
    )
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    if stored.get("status") != "PASS":
        raise ValueError("representative formal patient result is not PASS")

    matrix = load_model_matrix(
        workspace / str(config["model_matrix"]),
        workspace_root=workspace,
        asset_root_override=args.asset_root,
    )
    causal_model = str(config["models"]["causal"])
    job = _job_for(matrix.jobs, model=causal_model, seed=model_seed)
    device = torch.device(args.device)
    runtime = _load_runtime(
        job,
        config=config,
        observer_root=(workspace / str(config["observer_root"])).resolve(),
        device=device,
    )
    split = str(config["primary_split"])
    split_config = data_config["splits"][split]
    records_by_patient = _group_records(
        discover_slice_records(
            matrix.asset_root / str(split_config["image_dir"]),
            matrix.asset_root / str(split_config["mask_dir"]),
        )
    )
    records = records_by_patient[patient_id]
    formal_trace_path = (
        workspace
        / str(config["v3_root"])
        / causal_model
        / f"seed_{model_seed}"
        / split
        / "case_traces"
        / f"{patient_id}.npz"
    )
    formal = load_case_trace(formal_trace_path)
    if formal.truth is None:
        raise ValueError("representative V3 trace lacks truth")
    if formal.slice_ids != tuple(record.slice_id for record in records):
        raise ValueError("representative source inventory differs from V3 trace")

    target_config = config["target"]
    transition_index = int(target_config["transition_index"])
    truth_classes = tuple(int(value) for value in target_config["truth_classes"])
    target_path = persistent_correction_mask(
        formal.states,
        formal.truth,
        transition_index=transition_index,
        reliable=formal.reliable,
        truth_classes=truth_classes,
    )
    selected_index = select_target_slice(target_path)
    if formal.slice_ids[selected_index] != selected_slice_id:
        raise RuntimeError("representative slice is not reproduced by the locked rule")
    target_pixels = target_path[selected_index]
    image, truth = _load_slice(records[selected_index])

    source_module = getattr(runtime.adapter.model, str(target_config["source"]))
    with torch.inference_mode():
        _, clean_activation = capture_module_output(
            runtime.adapter,
            source_module,
            image.to(device),
        )
    feature_shape = tuple(int(value) for value in clean_activation.shape[-2:])
    target_feature = adaptive_feature_mask(target_pixels, output_shape=feature_shape)
    stable_control = stable_correct_control_mask(
        formal.states[:, selected_index],
        truth,
        transition_index=transition_index,
        reliable=formal.reliable[:, selected_index],
        truth_classes=truth_classes,
    )
    control_pool = (
        adaptive_feature_mask(stable_control, output_shape=feature_shape)
        & ~target_feature
    )
    feature_truth = dominant_feature_truth_class(
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
    matching = config["matching"]
    boundary_stratum, activation_stratum, _ = registered_feature_strata(
        feature_boundary,
        activation_norm,
        eligible_mask=target_feature | control_pool,
        boundary_bin_edges=tuple(matching["boundary_bin_edges"]),
        activation_norm_quantile_bins=int(matching["activation_norm_quantile_bins"]),
    )
    matches = match_feature_controls(
        target_mask=target_feature,
        control_mask=control_pool,
        truth_class=feature_truth,
        boundary_distance=feature_boundary,
        activation_norm=activation_norm,
        boundary_stratum=boundary_stratum,
        activation_stratum=activation_stratum,
    )
    if matches.matched_count != int(stored["matched_feature_count"]):
        raise RuntimeError("representative feature matching is not reproducible")
    matched_target = np.zeros(feature_shape, dtype=bool)
    matched_control = np.zeros(feature_shape, dtype=bool)
    matched_target.reshape(-1)[matches.target_indices] = True
    matched_control.reshape(-1)[matches.control_indices] = True

    module = getattr(runtime.adapter.model, str(target_config["module"]))
    argument_index = int(target_config["argument_index"])
    shift_yx = tuple(int(value) for value in config["intervention"]["feature_shift_yx"])
    clean = runtime.tracer.trace_slice(image, truth)
    corrupt = trace_spatial_intervention(
        runtime.tracer,
        image,
        truth,
        module=module,
        argument_index=argument_index,
        restore_mask=torch.from_numpy(target_feature),
        shift_yx=shift_yx,
        alpha=0.0,
    )
    restore_full_target = trace_spatial_intervention(
        runtime.tracer,
        image,
        truth,
        module=module,
        argument_index=argument_index,
        restore_mask=torch.from_numpy(target_feature),
        shift_yx=shift_yx,
        alpha=1.0,
    )
    restore_matched_target = trace_spatial_intervention(
        runtime.tracer,
        image,
        truth,
        module=module,
        argument_index=argument_index,
        restore_mask=torch.from_numpy(matched_target),
        shift_yx=shift_yx,
        alpha=1.0,
    )
    restore_control = trace_spatial_intervention(
        runtime.tracer,
        image,
        truth,
        module=module,
        argument_index=argument_index,
        restore_mask=torch.from_numpy(matched_control),
        shift_yx=shift_yx,
        alpha=1.0,
    )

    traces = {
        "clean": clean,
        "corrupt": corrupt,
        "restore_target_full": restore_full_target,
        "restore_target": restore_matched_target,
        "restore_control": restore_control,
    }
    q_values = {
        name: persistent_retention(
            trace.states,
            truth,
            target_pixels,
            transition_index=transition_index,
        )
        for name, trace in traces.items()
    }
    stored_q = {
        "clean": float(_condition_by_name(stored, "clean")["persistent_retention"]),
        "corrupt": float(_condition_by_name(stored, "corrupt")["persistent_retention"]),
        "restore_target_full": float(
            _condition_by_name(stored, "restore_target_1.00")["persistent_retention"]
        ),
        "restore_target": float(
            _condition_by_name(stored, "restore_target_matched_1.00")[
                "persistent_retention"
            ]
        ),
        "restore_control": float(
            _condition_by_name(stored, "restore_control_matched_1.00")[
                "persistent_retention"
            ]
        ),
    }
    rerun_errors = {
        name: _assert_close(q_values[name], stored_q[name], label=name)
        for name in q_values
    }
    rerun_effects = {
        "total_effect": q_values["clean"] - q_values["corrupt"],
        "target_indirect_effect": (
            q_values["restore_target_full"] - q_values["corrupt"]
        ),
        "target_minus_control": (
            q_values["restore_target"] - q_values["restore_control"]
        ),
    }
    for name, value in rerun_effects.items():
        rerun_errors[f"effect_{name}"] = _assert_close(
            value,
            float(stored["effects"][name]),
            label=name,
        )

    artifact_path = output_root / "causal_representative_case_artifacts.npz"
    np.savez_compressed(
        artifact_path,
        image=display_channel(
            image.detach().cpu().numpy(),
            channel_index=0,
        ).astype(np.float32, copy=False),
        truth=truth.astype(np.uint8, copy=False),
        target_pixels=target_pixels.astype(bool, copy=False),
        target_feature=target_feature.astype(bool, copy=False),
        matched_target_feature=matched_target.astype(bool, copy=False),
        matched_control_feature=matched_control.astype(bool, copy=False),
        clean_states=clean.states,
        clean_reliable=clean.reliable,
        corrupt_states=corrupt.states,
        restore_target_states=restore_matched_target.states,
        restore_control_states=restore_control.states,
        clean_final_state=clean.final_model_state,
        corrupt_final_state=corrupt.final_model_state,
        restore_target_final_state=restore_matched_target.final_model_state,
        restore_control_final_state=restore_control.final_model_state,
    )
    manifest = {
        "status": "PASS",
        "selection": representative,
        "transition_index": transition_index,
        "transition": target_config["transition"],
        "path_variable": {
            "module": target_config["module"],
            "argument_index": argument_index,
            "source": target_config["source"],
        },
        "q_values": q_values,
        "rerun_effects": rerun_effects,
        "maximum_scalar_reproduction_error": max(rerun_errors.values()),
        "target_pixel_count": int(target_pixels.sum()),
        "target_feature_count": int(target_feature.sum()),
        "matched_feature_count": int(matches.matched_count),
        "source_patient_result": stored_path,
        "source_v3_trace": formal_trace_path,
        "artifact": artifact_path,
    }
    manifest_path = output_root / "causal_representative_case_artifacts.json"
    manifest_path.write_text(
        json.dumps(_json_ready(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_ready(manifest), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
