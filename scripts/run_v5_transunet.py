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

from pptt.data.brats2d import (
    SliceRecord,
    discover_slice_records,
    ensure_chw,
    normalize_label,
)
from pptt.data.patient_splits import patient_id_from_slice
from pptt.experiments.model_matrix import ModelMatrixJob, load_model_matrix
from pptt.io.artifacts import CaseTrace, load_case_trace, save_case_trace
from pptt.models.adapters import build_adapter
from pptt.pipeline.summarize_trace import summarize_case_trace
from pptt.pipeline.trace_model import ObserverPathTracer, load_real_observer_path
from pptt.pipeline.transfer import (
    contiguous_node_segments,
    mask_discontinuous_reliability,
)


CONTROL_NAMES = ("patient_permutation", "spatial_shift")


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


def _observer_transfer_rows(
    directory: Path,
    *,
    job: ModelMatrixJob,
    nodes: tuple[str, ...],
    observer_seeds: tuple[int, ...],
) -> tuple[list[dict[str, Any]], tuple[str, ...], str | None]:
    if not directory.is_dir():
        return (
            [
                {
                    "model": job.model,
                    "model_seed": job.seed,
                    "node": node,
                    "observer_artifacts_available": False,
                    "real_better_than_patient_permutation": False,
                    "real_better_than_spatial_shift": False,
                    "node_admitted": False,
                    "failure_reason": "missing_observer_directory",
                }
                for node in nodes
            ],
            (),
            "missing_observer_directory",
        )
    required = (
        directory / "failed_nodes.json",
        directory / "observer_control_selectivity.parquet",
        directory / "reliability_thresholds.json",
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        reason = "missing_observer_audit_files:" + ",".join(missing)
        return (
            [
                {
                    "model": job.model,
                    "model_seed": job.seed,
                    "node": node,
                    "observer_artifacts_available": False,
                    "real_better_than_patient_permutation": False,
                    "real_better_than_spatial_shift": False,
                    "node_admitted": False,
                    "failure_reason": reason,
                }
                for node in nodes
            ],
            (),
            reason,
        )
    failed_payload = json.loads(
        (directory / "failed_nodes.json").read_text(encoding="utf-8")
    )
    failed_nodes = failed_payload.get("failed_nodes", {})
    selectivity = pd.read_parquet(directory / "observer_control_selectivity.parquet")
    rows = []
    admitted = []
    for node in nodes:
        node_frame = selectivity[selectivity.node == node]
        control_status = {}
        control_ci_low = {}
        for control in CONTROL_NAMES:
            control_frame = node_frame[node_frame.control == control]
            control_status[control] = bool(
                not control_frame.empty and control_frame.passed.astype(bool).all()
            )
            control_ci_low[control] = (
                float(control_frame.ci_low.min())
                if not control_frame.empty
                else float("nan")
            )
        state_files_available = all(
            (
                directory
                / "observers"
                / node
                / "real"
                / f"seed_{observer_seed}.pt"
            ).is_file()
            for observer_seed in observer_seeds
        )
        node_admitted = bool(
            node not in failed_nodes
            and state_files_available
            and all(control_status.values())
        )
        failure_reasons = []
        if node in failed_nodes:
            failure_reasons.append(f"v1_failed:{failed_nodes[node]}")
        if not state_files_available:
            failure_reasons.append("missing_real_observer_restart")
        for control, passed in control_status.items():
            if not passed:
                failure_reasons.append(f"control_not_passed:{control}")
        if node_admitted:
            admitted.append(node)
        rows.append(
            {
                "model": job.model,
                "model_seed": job.seed,
                "node": node,
                "observer_artifacts_available": state_files_available,
                "real_better_than_patient_permutation": control_status[
                    "patient_permutation"
                ],
                "patient_permutation_ci_low": control_ci_low[
                    "patient_permutation"
                ],
                "real_better_than_spatial_shift": control_status["spatial_shift"],
                "spatial_shift_ci_low": control_ci_low["spatial_shift"],
                "node_admitted": node_admitted,
                "failure_reason": ";".join(failure_reasons),
            }
        )
    return rows, tuple(admitted), None


def _load_reliability_threshold(directory: Path) -> float:
    payload = json.loads(
        (directory / "reliability_thresholds.json").read_text(encoding="utf-8")
    )
    threshold = payload.get("threshold")
    if threshold is None or not np.isfinite(float(threshold)) or float(threshold) < 0:
        raise ValueError(f"Invalid V1 reliability threshold: {directory}")
    return float(threshold)


def _trace_patient(
    tracer: ObserverPathTracer,
    records: list[SliceRecord],
    *,
    declared_nodes: tuple[str, ...],
    traced_nodes: tuple[str, ...],
) -> CaseTrace:
    states = []
    reliable = []
    margins = []
    truth = []
    final_model_state = []
    for record in records:
        image = torch.from_numpy(
            ensure_chw(np.load(record.image_path, allow_pickle=False))
        )[None]
        label = normalize_label(np.load(record.mask_path, allow_pickle=False))
        traced = tracer.trace_slice(image, label)
        states.append(traced.states)
        reliable.append(
            mask_discontinuous_reliability(
                traced.reliable,
                declared_nodes=declared_nodes,
                traced_nodes=traced_nodes,
            )
        )
        margins.append(traced.margins)
        truth.append(traced.truth)
        final_model_state.append(traced.final_model_state)
    return CaseTrace(
        states=np.stack(states, axis=1),
        reliable=np.stack(reliable, axis=1),
        margins=np.stack(margins, axis=1),
        truth=np.stack(truth),
        final_model_state=np.stack(final_model_state),
        slice_ids=tuple(record.slice_id for record in records),
    )


def _segment_trace(
    trace: CaseTrace,
    *,
    traced_nodes: tuple[str, ...],
    segment: tuple[str, ...],
) -> CaseTrace:
    start = traced_nodes.index(segment[0])
    stop = start + len(segment)
    if traced_nodes[start:stop] != segment:
        raise ValueError("segment must be contiguous inside traced_nodes")
    return CaseTrace(
        states=trace.states[start:stop],
        reliable=trace.reliable[start : max(stop - 1, start)],
        margins=None if trace.margins is None else trace.margins[start:stop],
        truth=trace.truth,
        final_model_state=trace.final_model_state,
        slice_ids=trace.slice_ids,
    )


def _extend(
    destination: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    destination.extend({**metadata, **row} for row in rows)


def _prepare_manifest(path: Path, payload: dict[str, Any], *, resume: bool) -> None:
    if resume and path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != _json_ready(payload):
            raise ValueError(f"V5 resume identity changed: {path}")
    else:
        _write_json(path, payload)


def _run_job(
    job: ModelMatrixJob,
    *,
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    experiment_config: dict[str, Any],
    asset_root: Path,
    observer_root: Path,
    output_root: Path,
    device: torch.device,
    resume: bool,
    max_patients: int | None,
    tables: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    nodes = tuple(str(value) for value in experiment_config["nodes"])
    observer_seeds = tuple(
        int(value) for value in experiment_config["observer_seeds"]
    )
    observer_directory = observer_root / job.model / f"seed_{job.seed}"
    transfer_rows, admitted_nodes, observer_blocker = _observer_transfer_rows(
        observer_directory,
        job=job,
        nodes=nodes,
        observer_seeds=observer_seeds,
    )
    checkpoint_available = job.checkpoint.is_file()
    for row in transfer_rows:
        row["checkpoint_available"] = checkpoint_available
        tables["observer_transfer"].append(row)
    if not checkpoint_available or observer_blocker is not None:
        blockers = []
        if not checkpoint_available:
            blockers.append("missing_segmentation_checkpoint")
        if observer_blocker is not None:
            blockers.append(observer_blocker)
        return {
            "job": job.job_id,
            "status": "PENDING_ASSETS",
            "blockers": blockers,
            "admitted_nodes": list(admitted_nodes),
            "failed_nodes": [node for node in nodes if node not in admitted_nodes],
            "segments": [],
        }

    adapter = build_adapter(
        job.model,
        n_channels=int(model_config["n_channels"]),
        num_classes=int(model_config["num_classes"]),
        bilinear=bool(model_config.get("bilinear", False)),
        img_size=int(model_config.get("img_size", 160)),
    )
    if tuple(adapter.checkpoint_names) != nodes:
        raise ValueError("TransUNet checkpoint adapter does not match V5 declaration")
    checkpoint_digest = adapter.load_checkpoint(job.checkpoint, map_location="cpu")
    if not admitted_nodes:
        return {
            "job": job.job_id,
            "status": "FAILED_TRANSFER",
            "blockers": ["no_observer_node_passed_v1"],
            "checkpoint_sha256": checkpoint_digest,
            "admitted_nodes": [],
            "failed_nodes": list(nodes),
            "segments": [],
        }
    observers = load_real_observer_path(
        observer_directory,
        nodes=admitted_nodes,
        seeds=observer_seeds,
    )
    threshold = _load_reliability_threshold(observer_directory)
    tracer = ObserverPathTracer(
        adapter,
        observers,
        nodes=admitted_nodes,
        seeds=observer_seeds,
        reliability_threshold=threshold,
        device=device,
    )
    segments = contiguous_node_segments(nodes, admitted_nodes)
    job_output = output_root / job.model / f"seed_{job.seed}"
    _prepare_manifest(
        job_output / "job_manifest.json",
        {
            "schema_version": 1,
            "job": job.job_id,
            "checkpoint": job.checkpoint,
            "checkpoint_sha256": checkpoint_digest,
            "observer_directory": observer_directory,
            "observer_seeds": observer_seeds,
            "declared_nodes": nodes,
            "admitted_nodes": admitted_nodes,
            "segments": segments,
            "reliability_threshold": threshold,
            "max_patients": max_patients,
        },
        resume=resume,
    )
    split = str(experiment_config["primary_split"])
    split_config = data_config["splits"][split]
    grouped = _group_records(
        discover_slice_records(
            asset_root / str(split_config["image_dir"]),
            asset_root / str(split_config["mask_dir"]),
        )
    )
    if len(grouped) != int(split_config["patient_count"]):
        raise ValueError(f"Unexpected patient count for {split}")
    patient_ids = sorted(grouped)
    if max_patients is not None:
        patient_ids = patient_ids[:max_patients]
    for patient_id in patient_ids:
        artifact_path = job_output / split / "case_traces" / f"{patient_id}.npz"
        expected_slices = tuple(record.slice_id for record in grouped[patient_id])
        if resume and artifact_path.is_file():
            trace = load_case_trace(artifact_path)
            if trace.slice_ids != expected_slices:
                raise ValueError(f"V5 case trace slices changed: {artifact_path}")
        else:
            trace = _trace_patient(
                tracer,
                grouped[patient_id],
                declared_nodes=nodes,
                traced_nodes=admitted_nodes,
            )
            save_case_trace(
                artifact_path,
                states=trace.states,
                reliable=trace.reliable,
                margins=trace.margins,
                truth=trace.truth,
                final_model_state=trace.final_model_state,
                slice_ids=trace.slice_ids,
            )
        for segment_index, segment in enumerate(segments):
            summary = summarize_case_trace(
                _segment_trace(
                    trace,
                    traced_nodes=admitted_nodes,
                    segment=segment,
                ),
                nodes=segment,
                num_classes=int(model_config["num_classes"]),
            )
            metadata = {
                "model": job.model,
                "model_seed": job.seed,
                "split": split,
                "patient_id": patient_id,
                "segment_index": segment_index,
                "segment_start": segment[0],
                "segment_end": segment[-1],
            }
            _extend(tables["nodes"], summary.node_rows, metadata)
            _extend(tables["transitions"], summary.transition_rows, metadata)
            _extend(tables["tensors"], summary.tensor_rows, metadata)
            _extend(tables["flows"], summary.flow_rows, metadata)
            _extend(tables["reconstruction"], summary.reconstruction_rows, metadata)
            _extend(tables["lineage"], summary.lineage_rows, metadata)
        print(f"transfer {job.job_id}/{patient_id}", flush=True)
    tracer.adapter.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    tolerance = float(experiment_config["statistics"]["reconstruction_tolerance"])
    job_reconstruction = [
        row for row in tables["reconstruction"] if row["model_seed"] == job.seed
    ]
    reconstruction_pass = bool(
        job_reconstruction
        and all(
            row["left_confusion_max_abs_error"] == 0
            and row["right_confusion_max_abs_error"] == 0
            and row["metric_max_abs_error"] <= tolerance
            and row["telescoping_macro_dice_error"] <= tolerance
            for row in job_reconstruction
        )
    )
    lineage_kinds = {
        row["kind"] for row in tables["lineage"] if row["model_seed"] == job.seed
    }
    required_lineage = set(experiment_config["required_lineage_outputs"])
    signature_outputs = {
        "node_trajectory": any(
            row["model_seed"] == job.seed for row in tables["nodes"]
        ),
        "transition_field_and_tensor": any(
            row["model_seed"] == job.seed for row in tables["tensors"]
        ),
        "lineage_dF_dY_o": required_lineage.issubset(lineage_kinds),
    }
    complete_nodes = len(admitted_nodes) == len(nodes)
    transfer_complete = complete_nodes and reconstruction_pass and all(
        signature_outputs.values()
    )
    return {
        "job": job.job_id,
        "status": "COMPLETE_TRANSFER" if transfer_complete else "PARTIAL_TRANSFER",
        "checkpoint_sha256": checkpoint_digest,
        "patient_count": len(patient_ids),
        "admitted_nodes": list(admitted_nodes),
        "failed_nodes": [node for node in nodes if node not in admitted_nodes],
        "segments": [list(segment) for segment in segments],
        "metric_reconstruction_pass": reconstruction_pass,
        "signature_outputs": signature_outputs,
    }


def _frame(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> pd.DataFrame:
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=list(columns))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate PPTT transfer to the U-shaped 2D TransUNet."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/brats2023_2d.yaml"),
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--observer-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--jobs", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-patients", type=int)
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    experiment_config = _load_yaml(args.config)
    data_config = _load_yaml(args.data_config)
    matrix = load_model_matrix(
        workspace / str(experiment_config["model_matrix"]),
        workspace_root=workspace,
        asset_root_override=args.asset_root,
    )
    configured_model = str(experiment_config["model"])
    configured_seeds = {int(value) for value in experiment_config["model_seeds"]}
    jobs = [
        job
        for job in matrix.jobs
        if job.model == configured_model and job.seed in configured_seeds
    ]
    if args.jobs:
        requested = {_job_selector(value) for value in args.jobs}
        jobs = [job for job in jobs if job.job_id in requested]
        missing = requested - {job.job_id for job in jobs}
        if missing:
            raise ValueError(f"Requested V5 jobs are not configured: {sorted(missing)}")
    if not jobs:
        raise ValueError("V5 has no configured model jobs")
    output_root = (
        args.output_root
        if args.output_root is not None
        else workspace / str(experiment_config["output_root"])
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    observer_root = (
        args.observer_root
        if args.observer_root is not None
        else workspace / str(experiment_config["observer_root"])
    ).resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    tables: dict[str, list[dict[str, Any]]] = {
        "observer_transfer": [],
        "nodes": [],
        "transitions": [],
        "tensors": [],
        "flows": [],
        "reconstruction": [],
        "lineage": [],
    }
    statuses = []
    for job in jobs:
        statuses.append(
            _run_job(
                job,
                model_config=_load_yaml(job.model_config),
                data_config=data_config,
                experiment_config=experiment_config,
                asset_root=matrix.asset_root,
                observer_root=observer_root,
                output_root=output_root,
                device=device,
                resume=args.resume,
                max_patients=args.max_patients,
                tables=tables,
            )
        )

    outputs = {
        "observer_transfer": (
            "observer_transfer_table.parquet",
            ("model", "model_seed", "node", "node_admitted"),
        ),
        "reconstruction": (
            "metric_reconstruction_error.parquet",
            ("model", "model_seed", "patient_id", "transition"),
        ),
        "flows": (
            "state_flow_by_node.parquet",
            ("model", "model_seed", "patient_id", "transition"),
        ),
        "lineage": (
            "lineage_depth_summary.parquet",
            ("model", "model_seed", "patient_id", "kind"),
        ),
        "nodes": (
            "signature_node_trajectory.parquet",
            ("model", "model_seed", "patient_id", "node"),
        ),
        "transitions": (
            "signature_transition_field.parquet",
            ("model", "model_seed", "patient_id", "transition"),
        ),
        "tensors": (
            "transition_tensor_by_patient.parquet",
            ("model", "model_seed", "patient_id", "transition"),
        ),
    }
    for table, (filename, columns) in outputs.items():
        _frame(tables[table], columns).to_parquet(
            output_root / filename,
            index=False,
        )
    complete_count = sum(status["status"] == "COMPLETE_TRANSFER" for status in statuses)
    pending_count = sum(status["status"] == "PENDING_ASSETS" for status in statuses)
    overall = {
        "status": (
            "COMPLETE_TRANSFER"
            if complete_count == len(statuses)
            else ("PENDING_ASSETS" if pending_count else "PARTIAL_OR_FAILED_TRANSFER")
        ),
        "method_scope": "interface transfer only; no U-Net curve replication claim",
        "jobs": statuses,
        "configured_job_count": len(statuses),
        "complete_transfer_job_count": complete_count,
        "pending_asset_job_count": pending_count,
        "output_files": {table: value[0] for table, value in outputs.items()},
    }
    _write_json(output_root / "transfer_status.json", overall)
    print(json.dumps(_json_ready(overall), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
