from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage
import torch
import yaml

from pptt.data.brats2d import discover_slice_records
from pptt.experiments.model_matrix import load_model_matrix
from pptt.interventions.activation_swap import capture_module_output
from pptt.interventions.causal_tracing import (
    adaptive_feature_mask,
    adaptive_feature_mean,
    dominant_feature_truth_class,
    match_feature_controls,
    registered_feature_strata,
)
from pptt.interventions.transunet_paths import (
    declared_transunet_decoder_paths,
    resolve_module,
)
from pptt.io.artifacts import load_case_trace
from pptt.lineage.cohorts import output_anchored_stable_correct_cohort
from pptt.lineage.direct_paths import output_anchored_direct_path_cohorts
from scripts.run_v8_transunet_mechanism import (
    _group_records,
    _job_for,
    _load_runtime,
    _load_slice,
)


_SUMMARY_COLUMNS = {
    "path_id",
    "topology_order",
    "model_seed",
    "patient_id",
    "status",
    "matched_feature_count",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must contain a mapping: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
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
    temporary.replace(path)


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def summarize_feasibility_rows(rows: pd.DataFrame) -> pd.DataFrame:
    missing = _SUMMARY_COLUMNS - set(rows.columns)
    if missing:
        raise ValueError(f"feasibility rows lack columns: {sorted(missing)}")
    if rows.empty:
        raise ValueError("feasibility rows must be nonempty")
    frame = rows.copy()
    frame["matched_feature_count"] = pd.to_numeric(
        frame["matched_feature_count"], errors="raise"
    ).astype(int)
    if (frame["matched_feature_count"] < 0).any():
        raise ValueError("matched feature counts must be nonnegative")
    if frame.duplicated(["path_id", "model_seed", "patient_id"]).any():
        raise ValueError("feasibility rows contain duplicate path-seed-patient rows")
    records = []
    for (order, path_id, seed), group in frame.groupby(
        ["topology_order", "path_id", "model_seed"], sort=True
    ):
        passed = group["status"].astype(str) == "PASS"
        records.append(
            {
                "path_id": str(path_id),
                "topology_order": int(order),
                "model_seed": int(seed),
                "patient_count": int(len(group)),
                "specificity_evaluable_patient_count": int(passed.sum()),
                "aggregate_matched_feature_count": int(
                    group.loc[passed, "matched_feature_count"].sum()
                ),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["topology_order", "model_seed"], kind="mergesort"
    ).reset_index(drop=True)


def _trace_directory(
    workspace: Path,
    config: dict[str, Any],
    *,
    model_seed: int,
) -> Path:
    return (
        workspace
        / str(config["validation"]["worker_root"])
        / f"seed_{model_seed}"
        / str(config["model"])
        / f"seed_{model_seed}"
        / str(config["discovery_split"])
        / "case_traces"
    )


def _evaluate_patient_path(
    *,
    runtime: Any,
    trace: Any,
    records: list[Any],
    path: Any,
    patient_id: str,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    truth_classes = tuple(int(value) for value in config["truth_classes"])
    eligibility = config["eligibility"]
    matching = config["matching"]
    if trace.truth is None or trace.final_model_state is None:
        raise ValueError(f"validation trace lacks truth or final state: {patient_id}")
    if trace.slice_ids != tuple(record.slice_id for record in records):
        raise ValueError(f"validation trace and source slices differ: {patient_id}")
    cohorts = output_anchored_direct_path_cohorts(
        trace.states,
        trace.truth,
        trace.reliable,
        trace.final_model_state,
        source_index=path.source_index,
        receiver_index=path.receiver_index,
        truth_classes=truth_classes,
    )
    counts = cohorts.correction.sum(axis=(1, 2), dtype=np.int64)
    maximum_count = int(counts.max(initial=0))
    base = {
        "path_id": path.path_id,
        "topology_order": path.topology_order,
        "source_node": path.source_node,
        "receiver_node": path.receiver_node,
        "model_seed": int(runtime.job.seed),
        "patient_id": patient_id,
        "target_pixel_count": maximum_count,
        "target_feature_count": 0,
        "control_feature_pool_count": 0,
        "matched_feature_count": 0,
        "feature_height": 0,
        "feature_width": 0,
    }
    if maximum_count < int(eligibility["minimum_target_pixels_on_selected_slice"]):
        return {**base, "status": "INELIGIBLE_TARGET_PIXELS"}
    slice_index = int(np.argmax(counts))
    target_pixels = cohorts.correction[slice_index]
    image, truth = _load_slice(records[slice_index])
    if not np.array_equal(truth, trace.truth[slice_index]):
        raise ValueError(f"validation truth differs: {patient_id}")
    source_module = resolve_module(runtime.adapter.model, path.source_module_path)
    with torch.inference_mode():
        _, activation = capture_module_output(
            runtime.adapter,
            source_module,
            image.to(device),
        )
    feature_shape = tuple(int(value) for value in activation.shape[-2:])
    target_feature = adaptive_feature_mask(target_pixels, output_shape=feature_shape)
    target_count = int(target_feature.sum())
    base.update(
        {
            "selected_slice_id": records[slice_index].slice_id,
            "target_feature_count": target_count,
            "feature_height": feature_shape[0],
            "feature_width": feature_shape[1],
        }
    )
    if target_count < int(eligibility["minimum_target_feature_positions"]):
        return {**base, "status": "INELIGIBLE_TARGET_FEATURE_POSITIONS"}
    stable_control = output_anchored_stable_correct_cohort(
        trace.states[:, slice_index],
        truth,
        trace.reliable[:, slice_index],
        trace.final_model_state[slice_index],
        transition_index=path.source_index,
        truth_classes=truth_classes,
        exclude=target_pixels,
    )
    control_feature = adaptive_feature_mask(
        stable_control, output_shape=feature_shape
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
        activation[0]
        .detach()
        .float()
        .square()
        .sum(dim=0)
        .sqrt()
        .cpu()
        .numpy()
    )
    boundary_stratum, activation_stratum, _ = registered_feature_strata(
        feature_boundary,
        activation_norm,
        eligible_mask=target_feature | control_feature,
        boundary_bin_edges=tuple(matching["boundary_bin_edges"]),
        activation_norm_quantile_bins=int(
            matching["activation_norm_quantile_bins"]
        ),
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
    return {
        **base,
        "status": (
            "PASS"
            if matches.matched_count >= minimum_matches
            else "INELIGIBLE_MATCHED_FEATURE_POSITIONS"
        ),
        "control_feature_pool_count": int(control_feature.sum()),
        "matched_feature_count": int(matches.matched_count),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit TransUNet candidate matching feasibility on validation data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/v9_transunet_specificity.yaml"),
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    config_path = args.config if args.config.is_absolute() else workspace / args.config
    config = _load_yaml(config_path.resolve())
    seeds = tuple(int(value) for value in config["model_seeds"])
    if args.model_seed not in seeds:
        raise ValueError("model seed is outside the registered V9 matrix")
    matrix = load_model_matrix(
        workspace / str(config["model_matrix"]),
        workspace_root=workspace,
        asset_root_override=args.asset_root.resolve(),
    )
    data_config = _load_yaml(workspace / "configs/data/brats2023_2d.yaml")
    split = str(config["discovery_split"])
    split_config = data_config["splits"][split]
    grouped = _group_records(
        discover_slice_records(
            matrix.asset_root / str(split_config["image_dir"]),
            matrix.asset_root / str(split_config["mask_dir"]),
        )
    )
    patient_ids = sorted(grouped)
    expected = int(config["validation"]["expected_patient_count_per_seed"])
    if len(patient_ids) != expected:
        raise ValueError("validation patient inventory differs from V9 configuration")
    if args.max_patients is not None:
        if int(args.max_patients) < 1:
            raise ValueError("max-patients must be positive")
        patient_ids = patient_ids[: int(args.max_patients)]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    job = _job_for(
        matrix.jobs,
        model=str(config["model"]),
        seed=args.model_seed,
    )
    runtime = _load_runtime(
        job,
        config=config,
        observer_root=(workspace / str(config["observer_root"])).resolve(),
        device=device,
    )
    trace_directory = _trace_directory(
        workspace, config, model_seed=args.model_seed
    )
    paths = declared_transunet_decoder_paths()
    if tuple(path.path_id for path in paths) != tuple(config["candidate_paths"]):
        raise ValueError("V9 candidate paths differ from the declared graph")
    rows = []
    for index, patient_id in enumerate(patient_ids, start=1):
        trace = load_case_trace(trace_directory / f"{patient_id}.npz")
        for path in paths:
            rows.append(
                _evaluate_patient_path(
                    runtime=runtime,
                    trace=trace,
                    records=grouped[patient_id],
                    path=path,
                    patient_id=patient_id,
                    config=config,
                    device=device,
                )
            )
        print(
            f"v9 feasibility seed={args.model_seed} {index}/{len(patient_ids)} "
            f"patient={patient_id}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    summary = summarize_feasibility_rows(frame)
    configured_root = (
        workspace / str(config["validation"]["feasibility_root"])
    ).resolve()
    selected_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else configured_root
    )
    if args.max_patients is None and selected_root != configured_root:
        raise ValueError("formal feasibility output must match the configuration")
    if args.max_patients is not None and selected_root == configured_root:
        raise ValueError("smoke feasibility cannot write into the formal directory")
    output = selected_root / f"seed_{args.model_seed}"
    _write_parquet_atomic(output / "patient_feasibility.parquet", frame)
    _write_parquet_atomic(output / "feasibility_summary.parquet", summary)
    status = {
        "status": (
            "SMOKE_COMPLETE_NOT_FOR_SELECTION"
            if args.max_patients is not None
            else "COMPLETE_VALIDATION_FEASIBILITY"
        ),
        "model_seed": args.model_seed,
        "patient_count": len(patient_ids),
        "candidate_path_count": len(paths),
        "row_count": len(frame),
        "test_data_read": False,
    }
    _write_json_atomic(output / "feasibility_status.json", status)
    runtime.adapter.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(json.dumps(status, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
