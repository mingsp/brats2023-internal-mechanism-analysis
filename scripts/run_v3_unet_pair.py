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
from pptt.experiments.pilot import stable_seed
from pptt.hooks.checkpoints import checkpoint_sha256
from pptt.io.artifacts import CaseTrace, load_case_trace, save_case_trace
from pptt.models.adapters import build_adapter
from pptt.pipeline.summarize_trace import summarize_case_trace
from pptt.pipeline.trace_model import (
    ObserverPathTracer,
    load_formal_reliability_threshold,
    load_real_observer_path,
)
from pptt.statistics.hypotheses import paired_patient_statistics


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


def _group_records(records: list[SliceRecord]) -> dict[str, list[SliceRecord]]:
    grouped: dict[str, list[SliceRecord]] = defaultdict(list)
    for record in records:
        grouped[patient_id_from_slice(record.slice_id)].append(record)
    return {
        patient: sorted(items, key=lambda item: item.slice_id)
        for patient, items in grouped.items()
    }


def _observer_hashes(
    observer_directory: Path,
    *,
    nodes: tuple[str, ...],
    seeds: tuple[int, ...],
) -> dict[str, str]:
    return {
        f"{node}/seed_{seed}": checkpoint_sha256(
            observer_directory / "observers" / node / "real" / f"seed_{seed}.pt"
        )
        for node in nodes
        for seed in seeds
    }


def _prepare_job_manifest(
    path: Path,
    *,
    payload: dict[str, Any],
    resume: bool,
) -> None:
    if resume and path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != _json_ready(payload):
            raise ValueError(f"V3 resume identity changed: {path}")
    else:
        case_files = list(path.parent.glob("*/case_traces/*.npz"))
        if resume and case_files:
            raise ValueError(f"V3 case traces lack a matching job manifest: {path}")
        _write_json(path, payload)


def _trace_patient(
    tracer: ObserverPathTracer,
    records: list[SliceRecord],
) -> CaseTrace:
    states = []
    reliable = []
    truth = []
    final_model_state = []
    for record in records:
        image = torch.from_numpy(
            ensure_chw(np.load(record.image_path, allow_pickle=False))
        )[None]
        label = normalize_label(np.load(record.mask_path, allow_pickle=False))
        traced = tracer.trace_slice(image, label)
        states.append(traced.states)
        reliable.append(traced.reliable)
        truth.append(traced.truth)
        final_model_state.append(traced.final_model_state)
    return CaseTrace(
        states=np.stack(states, axis=1),
        reliable=np.stack(reliable, axis=1),
        truth=np.stack(truth),
        final_model_state=np.stack(final_model_state),
        slice_ids=tuple(record.slice_id for record in records),
    )


def _extend(
    target: list[dict[str, Any]],
    rows: list[dict],
    metadata: dict[str, Any],
) -> None:
    target.extend({**metadata, **row} for row in rows)


def _paired_vectors(
    frame: pd.DataFrame,
    *,
    seed: int,
    split: str,
    metric: str,
    filters: dict[str, Any],
) -> tuple[list[str], np.ndarray, np.ndarray] | None:
    selected = frame[(frame.model_seed == seed) & (frame.split == split)]
    for key, value in filters.items():
        selected = selected[selected[key] == value]
    baseline = selected[selected.model == "unet_baseline"].set_index("patient_id")
    noskip = selected[selected.model == "unet_noskip"].set_index("patient_id")
    patients = sorted(set(baseline.index) & set(noskip.index))
    if len(patients) < 2:
        return None
    return (
        patients,
        baseline.loc[patients, metric].to_numpy(dtype=np.float64),
        noskip.loc[patients, metric].to_numpy(dtype=np.float64),
    )


def _add_statistics(
    rows: list[dict[str, Any]],
    *,
    question: str,
    contrast: str,
    seed: int,
    split: str,
    location: str,
    metric: str,
    left: np.ndarray,
    right: np.ndarray,
    iterations: int,
    bootstrap_seed: int,
) -> None:
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 2:
        return
    result = paired_patient_statistics(
        left[finite],
        right[finite],
        iterations=iterations,
        seed=stable_seed(
            bootstrap_seed,
            question,
            contrast,
            seed,
            split,
            location,
            metric,
        ),
    )
    rows.append(
        {
            "question": question,
            "contrast": contrast,
            "model_seed": seed,
            "split": split,
            "location": location,
            "metric": metric,
            **result.to_dict(),
        }
    )


def _hypothesis_statistics(
    node_frame: pd.DataFrame,
    transition_frame: pd.DataFrame,
    *,
    split: str,
    seeds: tuple[int, ...],
    iterations: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for node in ("down1", "down2"):
            for metric in ("dice", "correct_occupancy_rate", "persistent_correct_rate"):
                paired = _paired_vectors(
                    node_frame,
                    seed=seed,
                    split=split,
                    metric=metric,
                    filters={"node": node, "class_index": -1},
                )
                if paired is not None:
                    patients, baseline, noskip = paired
                    _add_statistics(
                        rows,
                        question="early_gap",
                        contrast="baseline_minus_noskip",
                        seed=seed,
                        split=split,
                        location=node,
                        metric=metric,
                        left=baseline,
                        right=noskip,
                        iterations=iterations,
                        bootstrap_seed=bootstrap_seed,
                    )
        for transition in ("down3->down4", "down4->up1"):
            for metric in (
                "destruction_rate",
                "wrong_reencoding_rate",
                "persistent_destruction_rate",
                "macro_tumor_delta_dice",
            ):
                paired = _paired_vectors(
                    transition_frame,
                    seed=seed,
                    split=split,
                    metric=metric,
                    filters={"transition": transition},
                )
                if paired is not None:
                    patients, baseline, noskip = paired
                    _add_statistics(
                        rows,
                        question="middle_trough",
                        contrast="baseline_minus_noskip",
                        seed=seed,
                        split=split,
                        location=transition,
                        metric=metric,
                        left=baseline,
                        right=noskip,
                        iterations=iterations,
                        bootstrap_seed=bootstrap_seed,
                    )
        for transition in ("up1->up2", "up2->up3", "up3->up4"):
            for metric in (
                "persistent_net_rate",
                "macro_tumor_delta_dice",
                "macro_tumor_dice_acceleration",
            ):
                paired = _paired_vectors(
                    transition_frame,
                    seed=seed,
                    split=split,
                    metric=metric,
                    filters={"transition": transition},
                )
                if paired is not None:
                    patients, baseline, noskip = paired
                    _add_statistics(
                        rows,
                        question="late_acceleration_deceleration",
                        contrast="baseline_minus_noskip",
                        seed=seed,
                        split=split,
                        location=transition,
                        metric=metric,
                        left=baseline,
                        right=noskip,
                        iterations=iterations,
                        bootstrap_seed=bootstrap_seed,
                    )
        paired = _paired_vectors(
            node_frame,
            seed=seed,
            split=split,
            metric="dice",
            filters={"node": "up4", "class_index": -1},
        )
        if paired is not None:
            patients, baseline, noskip = paired
            _add_statistics(
                rows,
                question="terminal_crossover",
                contrast="baseline_minus_noskip",
                seed=seed,
                split=split,
                location="up4",
                metric="dice",
                left=baseline,
                right=noskip,
                iterations=iterations,
                bootstrap_seed=bootstrap_seed,
            )
    return rows


def _v4_trigger(
    transition_frame: pd.DataFrame,
    *,
    split: str,
    seeds: tuple[int, ...],
    trigger_config: dict[str, Any],
    iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    candidate = str(trigger_config["candidate_transition"])
    decoder_transitions = tuple(str(value) for value in trigger_config["decoder_transitions"])
    seed_rows = []
    for seed in seeds:
        selected = transition_frame[
            (transition_frame.split == split)
            & (transition_frame.model_seed == seed)
        ]
        baseline = selected[selected.model == "unet_baseline"]
        noskip = selected[selected.model == "unet_noskip"]
        if baseline.empty or noskip.empty:
            continue
        means = {
            transition: float(
                baseline[baseline.transition == transition]["persistent_net_rate"].mean()
            )
            for transition in decoder_transitions
        }
        candidate_baseline = baseline[baseline.transition == candidate].set_index(
            "patient_id"
        )
        candidate_noskip = noskip[noskip.transition == candidate].set_index("patient_id")
        patients = sorted(set(candidate_baseline.index) & set(candidate_noskip.index))
        if len(patients) < 2:
            continue
        baseline_net = candidate_baseline.loc[
            patients,
            "persistent_net_rate",
        ].to_numpy(dtype=np.float64)
        noskip_net = candidate_noskip.loc[
            patients,
            "persistent_net_rate",
        ].to_numpy(dtype=np.float64)
        baseline_positive = paired_patient_statistics(
            baseline_net,
            np.zeros_like(baseline_net),
            iterations=iterations,
            seed=stable_seed(bootstrap_seed, "v4", seed, "baseline_positive"),
        )
        architecture_difference = paired_patient_statistics(
            baseline_net,
            noskip_net,
            iterations=iterations,
            seed=stable_seed(bootstrap_seed, "v4", seed, "architecture_difference"),
        )
        delta_dice = candidate_baseline.loc[
            patients,
            "macro_tumor_delta_dice",
        ].to_numpy(dtype=np.float64)
        conditions = {
            "candidate_is_largest_decoder_net_flow": bool(
                np.isfinite(means[candidate])
                and means[candidate]
                > max(
                    value
                    for transition, value in means.items()
                    if transition != candidate
                )
            ),
            "baseline_candidate_ci_above_zero": bool(baseline_positive.ci_low > 0),
            "baseline_minus_noskip_ci_above_zero": bool(
                architecture_difference.ci_low > 0
            ),
            "baseline_candidate_delta_dice_positive": bool(np.nanmean(delta_dice) > 0),
        }
        seed_rows.append(
            {
                "model_seed": seed,
                "patient_count": len(patients),
                "decoder_mean_persistent_net_rate": means,
                "baseline_candidate": baseline_positive.to_dict(),
                "architecture_difference": architecture_difference.to_dict(),
                "baseline_candidate_mean_delta_dice": float(np.nanmean(delta_dice)),
                "conditions": conditions,
                "all_conditions_met": all(conditions.values()),
            }
        )
    consistent_count = sum(row["all_conditions_met"] for row in seed_rows)
    required = int(trigger_config["minimum_consistent_model_seeds"])
    triggered = consistent_count >= required
    reasons = []
    if len(seed_rows) < required:
        reasons.append("fewer than the pre-registered number of paired model seeds")
    if consistent_count < required:
        reasons.append("candidate conditions were not met in enough model seeds")
    return {
        "triggered": triggered,
        "candidate_transition": candidate,
        "discovery_split": split,
        "minimum_consistent_model_seeds": required,
        "consistent_model_seed_count": consistent_count,
        "seed_results": seed_rows,
        "reasons": reasons,
        "fallback_transition_allowed": False,
    }


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
    max_patients_per_split: int | None,
    table_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    nodes = tuple(str(value) for value in experiment_config["nodes"])
    observer_seeds = tuple(int(value) for value in experiment_config["observer_seeds"])
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
    job_output = output_root / job.model / f"seed_{job.seed}"
    manifest = {
        "model": job.model,
        "model_seed": job.seed,
        "checkpoint": job.checkpoint,
        "checkpoint_sha256": checkpoint_digest,
        "observer_directory": observer_directory,
        "observer_state_sha256": _observer_hashes(
            observer_directory,
            nodes=nodes,
            seeds=observer_seeds,
        ),
        "nodes": nodes,
        "observer_seeds": observer_seeds,
        "reliability_threshold": threshold,
    }
    _prepare_job_manifest(
        job_output / "job_manifest.json",
        payload=manifest,
        resume=resume,
    )
    patient_count = 0
    for split in (
        str(experiment_config["discovery_split"]),
        str(experiment_config["primary_split"]),
    ):
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
        if max_patients_per_split is not None:
            patient_ids = patient_ids[:max_patients_per_split]
        for patient_id in patient_ids:
            artifact_path = job_output / split / "case_traces" / f"{patient_id}.npz"
            expected_slices = tuple(record.slice_id for record in grouped[patient_id])
            if resume and artifact_path.is_file():
                trace = load_case_trace(artifact_path)
                if trace.slice_ids != expected_slices:
                    raise ValueError(f"Case trace slices changed: {artifact_path}")
            else:
                trace = _trace_patient(tracer, grouped[patient_id])
                save_case_trace(
                    artifact_path,
                    states=trace.states,
                    reliable=trace.reliable,
                    truth=trace.truth,
                    final_model_state=trace.final_model_state,
                    slice_ids=trace.slice_ids,
                )
            summary = summarize_case_trace(
                trace,
                nodes=nodes,
                num_classes=int(model_config["num_classes"]),
            )
            metadata = {
                "model": job.model,
                "model_seed": job.seed,
                "split": split,
                "patient_id": patient_id,
                "slice_count": len(grouped[patient_id]),
            }
            _extend(table_rows["nodes"], summary.node_rows, metadata)
            _extend(table_rows["confusions"], summary.confusion_rows, metadata)
            _extend(table_rows["transitions"], summary.transition_rows, metadata)
            _extend(table_rows["flows"], summary.flow_rows, metadata)
            _extend(
                table_rows["metric_deltas"],
                summary.metric_delta_rows,
                metadata,
            )
            _extend(table_rows["tensors"], summary.tensor_rows, metadata)
            _extend(table_rows["lineage"], summary.lineage_rows, metadata)
            _extend(
                table_rows["reconstruction"],
                summary.reconstruction_rows,
                metadata,
            )
            patient_count += 1
            print(f"trace {job.job_id}/{split}/{patient_id}", flush=True)
    tracer.adapter.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "job": job.job_id,
        "status": "PASS",
        "patient_split_count": patient_count,
        "checkpoint_sha256": checkpoint_digest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run patient-level PPTT validation for baseline and no-skip U-Nets."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/brats2023_2d.yaml"))
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--observer-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--jobs", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-patients-per-split", type=int)
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    experiment_config = _load_yaml(args.config)
    data_config = _load_yaml(args.data_config)
    matrix = load_model_matrix(
        workspace / str(experiment_config["model_matrix"]),
        workspace_root=workspace,
        asset_root_override=args.asset_root,
    )
    configured = {
        f"{model}/seed_{seed}"
        for model in experiment_config["models"]
        for seed in experiment_config["model_seeds"]
    }
    selected = (
        {_job_selector(value) for value in args.jobs}
        if args.jobs is not None
        else configured
    )
    if not selected <= configured:
        raise ValueError(f"Jobs are outside the V3 configuration: {sorted(selected - configured)}")
    observer_root = (
        args.observer_root
        if args.observer_root is not None
        else workspace / str(experiment_config["observer_root"])
    ).resolve()
    output_root = (
        args.output_root
        if args.output_root is not None
        else workspace / str(experiment_config["output_root"])
    ).resolve()
    device = torch.device(args.device)
    table_rows: dict[str, list[dict[str, Any]]] = {
        name: []
        for name in (
            "nodes",
            "confusions",
            "transitions",
            "flows",
            "metric_deltas",
            "tensors",
            "lineage",
            "reconstruction",
        )
    }
    statuses = []
    matrix_jobs = {job.job_id: job for job in matrix.jobs}
    for job_id in sorted(selected):
        job = matrix_jobs[job_id]
        observer_directory = observer_root / job.model / f"seed_{job.seed}"
        if not job.checkpoint.is_file():
            statuses.append({"job": job_id, "status": "MISSING_CHECKPOINT"})
            continue
        if not (observer_directory / "v1_status.json").is_file():
            statuses.append({"job": job_id, "status": "MISSING_V1"})
            continue
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
                max_patients_per_split=args.max_patients_per_split,
                table_rows=table_rows,
            )
        )

    node_frame = pd.DataFrame(table_rows["nodes"])
    confusion_frame = pd.DataFrame(table_rows["confusions"])
    transition_frame = pd.DataFrame(table_rows["transitions"])
    flow_frame = pd.DataFrame(table_rows["flows"])
    metric_delta_frame = pd.DataFrame(table_rows["metric_deltas"])
    tensor_frame = pd.DataFrame(table_rows["tensors"])
    lineage_frame = pd.DataFrame(table_rows["lineage"])
    reconstruction_frame = pd.DataFrame(table_rows["reconstruction"])
    tolerance = float(experiment_config["statistics"]["reconstruction_tolerance"])
    if not reconstruction_frame.empty:
        failed_reconstruction = reconstruction_frame[
            (reconstruction_frame.left_confusion_max_abs_error != 0)
            | (reconstruction_frame.right_confusion_max_abs_error != 0)
            | (reconstruction_frame.metric_max_abs_error > tolerance)
            | (reconstruction_frame.telescoping_macro_dice_error > tolerance)
        ]
        if not failed_reconstruction.empty:
            raise RuntimeError("Metric reconstruction gate failed")
    output_root.mkdir(parents=True, exist_ok=True)
    node_frame.to_parquet(output_root / "node_state_by_patient.parquet", index=False)
    confusion_frame.to_parquet(
        output_root / "node_confusion_by_patient.parquet",
        index=False,
    )
    transition_frame.to_parquet(
        output_root / "transition_event_by_patient.parquet",
        index=False,
    )
    flow_frame.to_parquet(output_root / "state_flow_by_node.parquet", index=False)
    metric_delta_frame.to_parquet(
        output_root / "metric_delta_by_patient.parquet",
        index=False,
    )
    tensor_frame.to_parquet(output_root / "transition_tensor_by_patient.parquet", index=False)
    lineage_frame.to_parquet(output_root / "lineage_depth_summary.parquet", index=False)
    reconstruction_frame.to_parquet(
        output_root / "metric_reconstruction_error.parquet",
        index=False,
    )
    statistics_config = experiment_config["statistics"]
    hypothesis_rows = (
        _hypothesis_statistics(
            node_frame,
            transition_frame,
            split=str(experiment_config["primary_split"]),
            seeds=tuple(int(value) for value in experiment_config["model_seeds"]),
            iterations=int(statistics_config["bootstrap_iterations"]),
            bootstrap_seed=int(statistics_config["bootstrap_seed"]),
        )
        if not node_frame.empty and not transition_frame.empty
        else []
    )
    hypothesis_columns = [
        "question",
        "contrast",
        "model_seed",
        "split",
        "location",
        "metric",
        "patient_count",
        "mean_difference",
        "median_difference",
        "ci_low",
        "ci_high",
        "wilcoxon_p",
        "paired_cohens_d",
        "rank_biserial",
    ]
    hypothesis_frame = pd.DataFrame(hypothesis_rows, columns=hypothesis_columns)
    hypothesis_frame.to_parquet(
        output_root / "hypothesis_statistics.parquet",
        index=False,
    )
    seed_summary_columns = [
        "question",
        "contrast",
        "split",
        "location",
        "metric",
        "model_seed_count",
        "model_seeds_json",
        "mean_of_seed_effects",
        "positive_seed_count",
        "negative_seed_count",
        "zero_seed_count",
        "direction_consistent",
        "statistical_unit",
    ]
    seed_summary_rows = []
    if not hypothesis_frame.empty:
        grouping = ["question", "contrast", "split", "location", "metric"]
        for keys, group in hypothesis_frame.groupby(grouping, sort=True):
            effects = group.mean_difference.to_numpy(dtype=np.float64)
            seed_summary_rows.append(
                {
                    **dict(zip(grouping, keys, strict=True)),
                    "model_seed_count": len(group),
                    "model_seeds_json": json.dumps(
                        sorted(group.model_seed.astype(int).tolist())
                    ),
                    "mean_of_seed_effects": float(np.mean(effects)),
                    "positive_seed_count": int(np.count_nonzero(effects > 0)),
                    "negative_seed_count": int(np.count_nonzero(effects < 0)),
                    "zero_seed_count": int(np.count_nonzero(effects == 0)),
                    "direction_consistent": bool(
                        np.all(effects > 0) or np.all(effects < 0) or np.all(effects == 0)
                    ),
                    "statistical_unit": "descriptive model-seed summary only",
                }
            )
    pd.DataFrame(seed_summary_rows, columns=seed_summary_columns).to_parquet(
        output_root / "architecture_seed_summary.parquet",
        index=False,
    )
    trigger = (
        _v4_trigger(
            transition_frame,
            split=str(experiment_config["discovery_split"]),
            seeds=tuple(int(value) for value in experiment_config["model_seeds"]),
            trigger_config=experiment_config["v4_trigger"],
            iterations=int(statistics_config["bootstrap_iterations"]),
            bootstrap_seed=int(statistics_config["bootstrap_seed"]),
        )
        if not transition_frame.empty
        else {
            "triggered": False,
            "reasons": ["no paired V3 transition results"],
            "fallback_transition_allowed": False,
        }
    )
    _write_json(output_root / "v4_trigger.json", trigger)
    selected_complete = len(statuses) == len(selected) and all(
        row["status"] == "PASS" for row in statuses
    )
    full_matrix_complete = selected == configured and selected_complete
    status = {
        "execution_status": "PASS" if selected_complete else "INCOMPLETE",
        "full_matrix_status": "PASS" if full_matrix_complete else "INCOMPLETE",
        "selected_jobs": sorted(selected),
        "configured_jobs": sorted(configured),
        "jobs": statuses,
        "v4_triggered": trigger["triggered"],
    }
    _write_json(output_root / "v3_status.json", status)
    print(json.dumps(_json_ready(status), ensure_ascii=False, indent=2))
    return 0 if selected_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
