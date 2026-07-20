from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage
import torch
from torch.nn import functional as F
import yaml

from pptt.causal_abstraction.interventions import (
    bilinear_resize_rows,
    minimum_norm_state_exchange,
    project_feature_edit,
    stack_observer_weights,
    stack_restart_logit_deltas,
)
from pptt.causal_abstraction.matching import MatchingRules, match_natural_sources
from pptt.causal_abstraction.planning import (
    build_balanced_base_requests,
    prune_matches_to_independent_resize_rows,
    select_node_slices,
)
from pptt.causal_abstraction.states import relationship_states
from pptt.causal_abstraction.trace_process import node_reliability
from pptt.data.brats2d import (
    SliceRecord,
    discover_slice_records,
    ensure_chw,
    normalize_label,
)
from pptt.data.patient_splits import patient_id_from_slice
from pptt.experiments.model_matrix import ModelMatrixJob, load_model_matrix
from pptt.io.artifacts import CaseTrace, load_case_trace
from pptt.models.adapters import build_adapter
from pptt.models.protocol import ModelAdapter
from pptt.observers.linear import LinearObserver
from pptt.pipeline.trace_model import (
    load_formal_reliability_threshold,
    load_real_observer_path,
)
from scripts.lock_v11_causal_abstraction_protocol import (
    EXPECTED_CONTROL_MODELS,
    EXPECTED_MAIN_MODELS,
    EXPECTED_MODEL_SEEDS,
    EXPECTED_NODES,
    EXPECTED_OBSERVER_SEEDS,
    sha256_file,
    sha256_named_values,
    validate_v11_configuration,
)


@dataclass(frozen=True)
class PlanningRuntime:
    job: ModelMatrixJob
    adapter: ModelAdapter
    observers: Mapping[str, Mapping[int, LinearObserver]]
    reliability_threshold: float
    checkpoint_sha256: str


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must contain a mapping: {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return payload


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
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
    encoded = json.dumps(
        _json_ready(payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.parquet")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _stable_order(seed: int, *values: Any) -> int:
    encoded = "|".join((str(int(seed)), *(str(value) for value in values))).encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _group_records(records: Sequence[SliceRecord]) -> dict[str, list[SliceRecord]]:
    grouped: dict[str, list[SliceRecord]] = defaultdict(list)
    for record in records:
        grouped[patient_id_from_slice(record.slice_id)].append(record)
    return {
        patient_id: sorted(items, key=lambda item: item.slice_id)
        for patient_id, items in grouped.items()
    }


def _load_slice(record: SliceRecord) -> tuple[torch.Tensor, np.ndarray]:
    image = torch.from_numpy(
        ensure_chw(np.load(record.image_path, allow_pickle=False))
    )[None]
    truth = normalize_label(np.load(record.mask_path, allow_pickle=False))
    return image, truth


def _job_for(
    jobs: Sequence[ModelMatrixJob],
    *,
    model: str,
    model_seed: int,
) -> ModelMatrixJob:
    matches = [
        job for job in jobs if job.model == str(model) and job.seed == int(model_seed)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one model job for {model}/seed_{model_seed}")
    return matches[0]


def _load_runtime(
    job: ModelMatrixJob,
    *,
    configuration: Mapping[str, Any],
    workspace: Path,
    device: torch.device,
) -> PlanningRuntime:
    model_config = _load_yaml(job.model_config)
    adapter = build_adapter(
        job.model,
        n_channels=int(model_config["n_channels"]),
        num_classes=int(model_config["num_classes"]),
        bilinear=bool(model_config.get("bilinear", False)),
        img_size=int(model_config.get("img_size", 160)),
    )
    if tuple(adapter.checkpoint_names) != EXPECTED_NODES:
        raise ValueError(f"adapter node order differs for {job.job_id}")
    checkpoint_sha = adapter.load_checkpoint(job.checkpoint, map_location="cpu")
    observer_directory = (
        workspace
        / str(configuration["observer_root"])
        / job.model
        / f"seed_{job.seed}"
    )
    threshold = load_formal_reliability_threshold(observer_directory)
    observers = load_real_observer_path(
        observer_directory,
        nodes=EXPECTED_NODES,
        seeds=EXPECTED_OBSERVER_SEEDS,
    )
    adapter = adapter.to(device).eval()
    observer_device = {
        node: {
            seed: observers[node][seed].to(device).eval()
            for seed in EXPECTED_OBSERVER_SEEDS
        }
        for node in EXPECTED_NODES
    }
    return PlanningRuntime(
        job=job,
        adapter=adapter,
        observers=observer_device,
        reliability_threshold=threshold,
        checkpoint_sha256=checkpoint_sha,
    )


def _trace_directory(
    workspace: Path,
    configuration: Mapping[str, Any],
    *,
    split_role: str,
    model: str,
    model_seed: int,
) -> Path:
    template = str(configuration["clean_trace_templates"][split_role][model])
    return workspace / template.format(model=model, seed=int(model_seed))


def _restart_logits(
    activation: torch.Tensor,
    observers: Mapping[int, LinearObserver],
    *,
    output_shape: tuple[int, int],
) -> torch.Tensor:
    return torch.stack(
        tuple(
            observers[seed](activation, output_size=output_shape)[0]
            for seed in EXPECTED_OBSERVER_SEEDS
        ),
        dim=0,
    )


def _observer_weight_stack(
    observers: Mapping[int, LinearObserver],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    weights = tuple(
        observers[seed].projection.weight[:, :, 0, 0].detach().to(
            device=device,
            dtype=dtype,
        )
        for seed in EXPECTED_OBSERVER_SEEDS
    )
    return stack_observer_weights(weights)


def _boundary_distance(truth: np.ndarray, truth_class: int) -> np.ndarray:
    selected = np.asarray(truth) == int(truth_class)
    return ndimage.distance_transform_edt(selected).astype(np.float32, copy=False)


def _candidate_rows_for_job(
    runtime: PlanningRuntime,
    *,
    grouped_records: Mapping[str, Sequence[SliceRecord]],
    patient_ids: Sequence[str],
    trace_directory: Path,
    configuration: Mapping[str, Any],
    split: str,
    device: torch.device,
    retain_logits: bool,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    candidate_limit = int(
        configuration["matching"][
            "source_candidates_per_patient_node_state_class"
        ]
    )
    rows: list[dict[str, Any]] = []
    logit_bank: dict[str, np.ndarray] = {}
    selection_registry: dict[str, Any] = {}
    mismatch_count = 0
    for patient_index, patient_id in enumerate(patient_ids, start=1):
        trace_path = trace_directory / f"{patient_id}.npz"
        if not trace_path.is_file():
            raise FileNotFoundError(trace_path)
        trace = load_case_trace(trace_path)
        records = list(grouped_records[patient_id])
        expected_slices = tuple(record.slice_id for record in records)
        if trace.slice_ids != expected_slices:
            raise ValueError(f"clean trace slice registry differs for {patient_id}")
        selections = select_node_slices(trace, nodes=EXPECTED_NODES, num_classes=4)
        selection_registry[patient_id] = selections
        record_by_id = {record.slice_id: record for record in records}
        nodes_by_slice: dict[str, list[str]] = defaultdict(list)
        for node, selection in selections.items():
            nodes_by_slice[str(selection["slice_id"])].append(node)
        for slice_id in sorted(nodes_by_slice):
            record = record_by_id[slice_id]
            slice_index = expected_slices.index(slice_id)
            image, truth = _load_slice(record)
            if trace.truth is None or not np.array_equal(truth, trace.truth[slice_index]):
                raise ValueError(f"trace truth differs from source mask: {slice_id}")
            with torch.inference_mode():
                clean = runtime.adapter.trace(image.to(device))
                for node in sorted(
                    nodes_by_slice[slice_id],
                    key=EXPECTED_NODES.index,
                ):
                    node_index = EXPECTED_NODES.index(node)
                    activation = clean.activations[node]
                    restart_logits = _restart_logits(
                        activation,
                        runtime.observers[node],
                        output_shape=truth.shape,
                    )
                    canonical_prediction = (
                        torch.softmax(restart_logits, dim=1)
                        .mean(dim=0)
                        .argmax(dim=0)
                        .cpu()
                        .numpy()
                        .astype(np.uint8, copy=False)
                    )
                    mismatch_count += int(
                        np.count_nonzero(
                            canonical_prediction
                            != trace.states[node_index, slice_index]
                        )
                    )
                    relation = relationship_states(
                        truth,
                        canonical_prediction,
                        num_classes=4,
                    )
                    reliable = node_reliability(trace, node_index)[slice_index]
                    feature_norm_native = activation[0].float().square().sum(dim=0).sqrt()
                    feature_norm = (
                        F.interpolate(
                            feature_norm_native[None, None],
                            size=truth.shape,
                            mode="bilinear",
                            align_corners=False,
                        )[0, 0]
                        .cpu()
                        .numpy()
                    )
                    native_h, native_w = (
                        int(activation.shape[-2]),
                        int(activation.shape[-1]),
                    )
                    logits_flat = restart_logits.permute(0, 2, 3, 1).reshape(
                        len(EXPECTED_OBSERVER_SEEDS),
                        -1,
                        int(restart_logits.shape[1]),
                    )
                    for truth_class in range(4):
                        distances = _boundary_distance(truth, truth_class).reshape(-1)
                        for state in range(5):
                            eligible = np.flatnonzero(
                                reliable.reshape(-1)
                                & (truth.reshape(-1) == truth_class)
                                & (relation.reshape(-1) == state)
                            )
                            if eligible.size == 0:
                                continue
                            ordered = sorted(
                                eligible.astype(int).tolist(),
                                key=lambda index: (
                                    _stable_order(
                                        int(configuration["high_level_process"]["fold_seed"]),
                                        runtime.job.model,
                                        runtime.job.seed,
                                        patient_id,
                                        node,
                                        truth_class,
                                        state,
                                        index,
                                    ),
                                    index,
                                ),
                            )[:candidate_limit]
                            for output_index in ordered:
                                row_id = f"{patient_id}|{slice_id}|{node}|{output_index}"
                                rows.append(
                                    {
                                        "row_id": row_id,
                                        "patient_id": patient_id,
                                        "node": node,
                                        "truth_class": truth_class,
                                        "state": state,
                                        "boundary_distance": float(distances[output_index]),
                                        "spatial_scale": f"{native_h}x{native_w}",
                                        "feature_norm": float(
                                            feature_norm.reshape(-1)[output_index]
                                        ),
                                        "split": split,
                                        "architecture": runtime.job.model,
                                        "model_seed": runtime.job.seed,
                                        "slice_id": slice_id,
                                        "output_index": output_index,
                                        "native_h": native_h,
                                        "native_w": native_w,
                                    }
                                )
                                if retain_logits:
                                    logit_bank[row_id] = (
                                        logits_flat[:, output_index]
                                        .detach()
                                        .cpu()
                                        .numpy()
                                        .astype(np.float64, copy=False)
                                    )
        print(
            f"v11-plan {runtime.job.job_id} {split} "
            f"{patient_index}/{len(patient_ids)} patient={patient_id}",
            flush=True,
        )
    if mismatch_count:
        raise ValueError(
            f"clean observer predictions differ from frozen traces: {mismatch_count} pixels"
        )
    candidates = pd.DataFrame(rows)
    if candidates.empty or not candidates["row_id"].is_unique:
        raise ValueError("candidate extraction produced an empty or duplicate registry")
    return candidates, logit_bank, selection_registry


def _feature_norm_edges(candidates: pd.DataFrame) -> dict[str, list[float]]:
    output = {}
    for node, group in candidates.groupby("node", sort=True):
        edges = np.unique(
            np.quantile(group["feature_norm"].to_numpy(dtype=np.float64), (0.25, 0.5, 0.75))
        )
        if len(edges) != 3 or not np.isfinite(edges).all():
            raise ValueError(f"feature-norm calibration is degenerate for {node}")
        output[str(node)] = edges.astype(float).tolist()
    if set(output) != set(EXPECTED_NODES):
        raise ValueError("feature-norm calibration lacks a registered node")
    return output


def _matched_pairs(
    candidates: pd.DataFrame,
    *,
    feature_edges: Mapping[str, Sequence[float]],
    configuration: Mapping[str, Any],
    model_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    requests = build_balanced_base_requests(
        candidates,
        maximum_targets=int(
            configuration["matching"]["maximum_target_pixels_per_node_patient"]
        ),
        seed=int(configuration["high_level_process"]["fold_seed"]) + int(model_seed),
    )
    matches = []
    for node_index, node in enumerate(EXPECTED_NODES):
        node_candidates = candidates[candidates["node"] == node]
        node_requests = requests[requests["node"] == node]
        if node_requests.empty:
            continue
        rules = MatchingRules(
            same_patient_first=bool(configuration["matching"]["same_patient_first"]),
            max_pixels_per_patient_node_state=int(
                configuration["matching"]["max_pixels_per_patient_node_state"]
            ),
            boundary_edges=tuple(
                float(value) for value in configuration["matching"]["boundary_edges"]
            ),
            feature_norm_quantile_bins=int(
                configuration["matching"]["feature_norm_quantile_bins"]
            ),
            seed=int(configuration["high_level_process"]["fold_seed"])
            + int(model_seed)
            + node_index,
            feature_norm_edges=tuple(float(value) for value in feature_edges[node]),
        )
        matches.append(match_natural_sources(node_requests, node_candidates, rules))
    if not matches:
        raise ValueError("natural-source matching produced no node results")
    all_matches = pd.concat(matches, ignore_index=True)
    retained, rank_audit = prune_matches_to_independent_resize_rows(
        all_matches,
        candidates,
        output_shape=(160, 160),
        rcond=float(configuration["intervention"]["pseudoinverse_rcond"]),
    )
    return retained, rank_audit


def _patient_plan_payloads(
    *,
    model: str,
    model_seed: int,
    split: str,
    patient_ids: Sequence[str],
    candidates: pd.DataFrame,
    matches: pd.DataFrame,
    selection_registry: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    candidate_index = candidates.set_index("row_id", verify_integrity=True)
    payloads: dict[str, dict[str, Any]] = {}
    for patient_id in patient_ids:
        node_plans = []
        for node in EXPECTED_NODES:
            selected = matches[
                (matches["patient_id"] == patient_id) & (matches["node"] == node)
            ].sort_values("base_row_id", kind="mergesort")
            pairs = []
            for _, match in selected.iterrows():
                base = candidate_index.loc[str(match["base_row_id"])]
                source = candidate_index.loc[str(match["source_row_id"])]
                if int(source["state"]) != int(match["requested_source_state"]):
                    raise ValueError("matched source state differs from the request")
                pairs.append(
                    {
                        "base_output_index": int(base["output_index"]),
                        "base_state": int(base["state"]),
                        "target_state": int(source["state"]),
                        "truth_class": int(base["truth_class"]),
                        "source_patient_id": str(source["patient_id"]),
                        "source_slice_id": str(source["slice_id"]),
                        "source_output_index": int(source["output_index"]),
                        "same_patient": bool(match["same_patient"]),
                        "boundary_stratum": int(match["base_boundary_stratum"]),
                        "feature_norm_stratum": int(match["base_norm_stratum"]),
                    }
                )
            selection = selection_registry[patient_id][node]
            native_shape = None
            if pairs:
                base = candidate_index.loc[
                    f"{patient_id}|{selection['slice_id']}|{node}|{pairs[0]['base_output_index']}"
                ]
                native_shape = [int(base["native_h"]), int(base["native_w"])]
            node_plans.append(
                {
                    "node": node,
                    "status": "MATCHED" if pairs else "NO_MATCH",
                    "base_slice_id": str(selection["slice_id"]),
                    "native_shape": native_shape,
                    "pairs": pairs,
                }
            )
        payloads[patient_id] = {
            "schema_version": 1,
            "status": "REGISTERED_PATIENT_PLAN",
            "model": model,
            "model_seed": int(model_seed),
            "split": split,
            "patient_id": patient_id,
            "nodes": node_plans,
        }
    return payloads


def _operator_audits(
    runtime: PlanningRuntime,
    *,
    payloads: Mapping[str, Mapping[str, Any]],
    logit_bank: Mapping[str, np.ndarray],
    configuration: Mapping[str, Any],
) -> pd.DataFrame:
    rows = []
    for patient_id, payload in payloads.items():
        for node_plan in payload["nodes"]:
            pairs = node_plan["pairs"]
            if not pairs:
                continue
            node = str(node_plan["node"])
            base_ids = [
                f"{patient_id}|{node_plan['base_slice_id']}|{node}|{pair['base_output_index']}"
                for pair in pairs
            ]
            source_ids = [
                f"{pair['source_patient_id']}|{pair['source_slice_id']}|{node}|{pair['source_output_index']}"
                for pair in pairs
            ]
            base_logits = torch.from_numpy(np.stack([logit_bank[key] for key in base_ids]))
            source_logits = torch.from_numpy(
                np.stack([logit_bank[key] for key in source_ids])
            )
            base_logits = base_logits.permute(1, 0, 2).to(torch.float64)
            source_logits = source_logits.permute(1, 0, 2).to(torch.float64)
            stacked_delta = stack_restart_logit_deltas(source_logits, base_logits)
            stacked_weight = _observer_weight_stack(
                runtime.observers[node],
                dtype=torch.float64,
                device=torch.device("cpu"),
            )
            output_indices = torch.tensor(
                [pair["base_output_index"] for pair in pairs],
                dtype=torch.int64,
            )
            native_shape = tuple(int(value) for value in node_plan["native_shape"])
            resize_rows = bilinear_resize_rows(
                native_shape,
                (160, 160),
                output_indices,
                dtype=torch.float64,
            )
            exchange = minimum_norm_state_exchange(
                resize_rows,
                stacked_weight,
                stacked_delta,
                rcond=float(configuration["intervention"]["pseudoinverse_rcond"]),
            )
            projected = project_feature_edit(
                exchange.delta_h,
                native_shape=native_shape,
                observer_weight=stacked_weight,
                output_shape=(160, 160),
            )
            magnitude = torch.linalg.vector_norm(projected, dim=0).numpy()
            target_mask = np.zeros((160, 160), dtype=bool)
            target_mask.reshape(-1)[output_indices.numpy()] = True
            target_mean = float(magnitude[target_mask].mean())
            outside_q99 = float(np.quantile(magnitude[~target_mask], 0.99))
            leakage_ratio = outside_q99 / max(target_mean, np.finfo(float).eps)
            reconstructed = (
                base_logits
                + exchange.reconstructed_delta_logits.reshape(
                    len(pairs),
                    len(EXPECTED_OBSERVER_SEEDS),
                    -1,
                ).permute(1, 0, 2)
            )
            predictions = (
                torch.softmax(reconstructed, dim=2)
                .mean(dim=0)
                .argmax(dim=1)
                .numpy()
            )
            truth_classes = np.asarray(
                [pair["truth_class"] for pair in pairs],
                dtype=np.uint8,
            )
            realized_states = relationship_states(
                truth_classes,
                predictions.astype(np.uint8),
                num_classes=4,
            )
            target_states = np.asarray(
                [pair["target_state"] for pair in pairs],
                dtype=np.uint8,
            )
            rows.append(
                {
                    "model": runtime.job.model,
                    "model_seed": runtime.job.seed,
                    "patient_id": patient_id,
                    "node": node,
                    "pair_count": len(pairs),
                    "target_max_abs_error": exchange.target_max_abs_error,
                    "state_realization_rate": float(
                        np.mean(realized_states == target_states)
                    ),
                    "edit_norm_per_sqrt_pixel": exchange.frobenius_norm
                    / np.sqrt(len(pairs)),
                    "leakage_ratio_q99_to_target_mean": leakage_ratio,
                    "joint_observer_rank": exchange.effective_rank_channel,
                    "joint_observer_nullity": int(
                        stacked_weight.shape[1] - exchange.effective_rank_channel
                    ),
                    "spatial_rank": exchange.effective_rank_spatial,
                }
            )
    return pd.DataFrame(rows)


def _coverage_summary(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output = {}
    for node in EXPECTED_NODES:
        state_patients: dict[int, set[str]] = defaultdict(set)
        state_pixels: dict[int, int] = defaultdict(int)
        patients = set()
        for patient_id, payload in payloads.items():
            node_plan = next(value for value in payload["nodes"] if value["node"] == node)
            if node_plan["pairs"]:
                patients.add(patient_id)
            for pair in node_plan["pairs"]:
                state = int(pair["target_state"])
                state_patients[state].add(patient_id)
                state_pixels[state] += 1
        output[node] = {
            "patient_count": len(patients),
            "target_states": {
                str(state): {
                    "patient_count": len(state_patients[state]),
                    "pixel_count": int(state_pixels[state]),
                }
                for state in sorted(state_patients)
            },
        }
    return output


def _prepare_job(
    *,
    workspace: Path,
    asset_root: Path,
    configuration: Mapping[str, Any],
    model: str,
    model_seed: int,
    split_role: str,
    device: torch.device,
) -> dict[str, Any]:
    validated = validate_v11_configuration(configuration)
    matrix = load_model_matrix(
        workspace / str(configuration["model_matrix"]),
        workspace_root=workspace,
        asset_root_override=asset_root,
    )
    job = _job_for(matrix.jobs, model=model, model_seed=model_seed)
    runtime = _load_runtime(
        job,
        configuration=configuration,
        workspace=workspace,
        device=device,
    )
    data_config = _load_yaml(workspace / "configs/data/brats2023_2d.yaml")
    split = str(configuration["fit_split"] if split_role == "validation" else configuration["formal_split"])
    split_config = data_config["splits"][split]
    grouped = _group_records(
        discover_slice_records(
            asset_root / str(split_config["image_dir"]),
            asset_root / str(split_config["mask_dir"]),
        )
    )
    expected_count = (
        validated.validation_patient_count
        if split_role == "validation"
        else validated.formal_patient_count
    )
    patient_ids = sorted(grouped)
    if len(patient_ids) != expected_count:
        raise ValueError(f"{split} patient count differs from the protocol")
    trace_directory = _trace_directory(
        workspace,
        configuration,
        split_role=split_role,
        model=model,
        model_seed=model_seed,
    )
    candidates, logit_bank, selection_registry = _candidate_rows_for_job(
        runtime,
        grouped_records=grouped,
        patient_ids=patient_ids,
        trace_directory=trace_directory,
        configuration=configuration,
        split=split,
        device=device,
        retain_logits=split_role == "validation",
    )
    identity = f"{model}/seed_{model_seed}"
    fit_root = workspace / str(configuration["validation_fit_root"])
    if split_role == "validation":
        feature_edges = _feature_norm_edges(candidates)
    else:
        matching_calibration = _load_json(fit_root / "matching_calibration.json")
        if matching_calibration.get("status") != "PASS":
            raise ValueError("matching calibration is not PASS")
        feature_edges = matching_calibration["jobs"][identity]["feature_norm_edges"]
    matches, rank_audit = _matched_pairs(
        candidates,
        feature_edges=feature_edges,
        configuration=configuration,
        model_seed=model_seed,
    )
    payloads = _patient_plan_payloads(
        model=model,
        model_seed=model_seed,
        split=split,
        patient_ids=patient_ids,
        candidates=candidates,
        matches=matches,
        selection_registry=selection_registry,
    )
    coverage = _coverage_summary(payloads)
    if split_role == "validation":
        job_root = fit_root / "planning_jobs" / model / f"seed_{model_seed}"
    else:
        job_root = (
            workspace
            / str(configuration["intervention_plan_root"])
            / model
            / f"seed_{model_seed}"
        )
    plan_root = job_root / "patient_plans"
    plan_hashes = {}
    for patient_id in patient_ids:
        path = plan_root / f"{patient_id}.json"
        _write_json_atomic(path, payloads[patient_id])
        plan_hashes[patient_id] = sha256_file(path)
    _write_parquet_atomic(job_root / "rank_audit.parquet", rank_audit)
    common = {
        "schema_version": 1,
        "model": model,
        "model_seed": model_seed,
        "split": split,
        "patient_ids": patient_ids,
        "patient_count": len(patient_ids),
        "patient_plan_hashes": plan_hashes,
        "patient_plan_registry_sha256": sha256_named_values(plan_hashes),
        "checkpoint_sha256": runtime.checkpoint_sha256,
        "feature_norm_edges": feature_edges,
        "coverage": coverage,
        "candidate_count": int(len(candidates)),
        "matched_pair_count": int(sum(len(value["pairs"]) for payload in payloads.values() for value in payload["nodes"])),
        "full_activations_persisted": False,
    }
    if split_role == "validation":
        audits = _operator_audits(
            runtime,
            payloads=payloads,
            logit_bank=logit_bank,
            configuration=configuration,
        )
        if audits.empty:
            raise ValueError("validation operator audit is empty")
        audit_path = job_root / "operator_audits.parquet"
        _write_parquet_atomic(audit_path, audits)
        status = "PASS"
        if (
            audits["target_max_abs_error"].max()
            > float(configuration["intervention"]["maximum_reconstruction_error"])
            or audits["state_realization_rate"].min()
            < float(configuration["intervention"]["minimum_state_realization"])
            or audits["joint_observer_nullity"].min() <= 0
        ):
            status = "FAILED_OPERATOR_CALIBRATION"
        result = {
            **common,
            "status": status,
            "operator_audit_sha256": sha256_file(audit_path),
        }
        _write_json_atomic(job_root / "job_calibration.json", result)
    else:
        gates = configuration["coverage"]
        failed_nodes = []
        for node, summary in coverage.items():
            evaluable_states = [
                state
                for state, values in summary["target_states"].items()
                if int(values["patient_count"]) >= int(gates["minimum_state_patients"])
                and int(values["pixel_count"]) >= int(gates["minimum_state_pixels"])
            ]
            if (
                int(summary["patient_count"]) < int(gates["minimum_node_patients"])
                or len(evaluable_states) < int(gates["minimum_source_states"])
            ):
                failed_nodes.append(node)
            summary["evaluable_target_states"] = evaluable_states
        result = {
            **common,
            "status": (
                "COMPLETE_LOCKABLE_PLAN"
                if not failed_nodes
                else "INSUFFICIENT_INTERVENTION_SUPPORT"
            ),
            "failed_nodes": failed_nodes,
        }
        _write_json_atomic(job_root / "plan_manifest.json", result)
    runtime.adapter.to("cpu")
    for group in runtime.observers.values():
        for observer in group.values():
            observer.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def aggregate_validation(
    workspace: Path,
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    fit_root = workspace / str(configuration["validation_fit_root"])
    jobs = {}
    audit_frames = []
    for model in EXPECTED_MAIN_MODELS + EXPECTED_CONTROL_MODELS:
        for model_seed in EXPECTED_MODEL_SEEDS:
            identity = f"{model}/seed_{model_seed}"
            root = fit_root / "planning_jobs" / model / f"seed_{model_seed}"
            calibration_path = root / "job_calibration.json"
            audit_path = root / "operator_audits.parquet"
            for path in (calibration_path, audit_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            calibration = _load_json(calibration_path)
            if calibration.get("status") != "PASS":
                raise ValueError(f"validation planning did not pass: {identity}")
            jobs[identity] = {
                "feature_norm_edges": calibration["feature_norm_edges"],
                "candidate_count": calibration["candidate_count"],
                "matched_pair_count": calibration["matched_pair_count"],
                "coverage": calibration["coverage"],
                "job_calibration_sha256": sha256_file(calibration_path),
            }
            audit_frames.append(pd.read_parquet(audit_path))
    audits = pd.concat(audit_frames, ignore_index=True)
    matching_payload = {
        "status": "PASS",
        "jobs": jobs,
        "boundary_edges": configuration["matching"]["boundary_edges"],
        "feature_norm_quantile_bins": configuration["matching"][
            "feature_norm_quantile_bins"
        ],
    }
    matching_path = fit_root / "matching_calibration.json"
    _write_json_atomic(matching_path, matching_payload)
    thresholds = {}
    for (model, node), group in audits.groupby(["model", "node"], sort=True):
        seed_quantiles = []
        for _, seed_group in group.groupby("model_seed", sort=True):
            seed_quantiles.append(
                {
                    "edit_norm": float(
                        np.quantile(seed_group["edit_norm_per_sqrt_pixel"], 0.99)
                    ),
                    "leakage": float(
                        np.quantile(
                            seed_group["leakage_ratio_q99_to_target_mean"],
                            0.99,
                        )
                    ),
                }
            )
        thresholds[f"{model}/{node}"] = {
            "maximum_edit_norm_per_sqrt_pixel": max(
                value["edit_norm"] for value in seed_quantiles
            ),
            "maximum_leakage_ratio_q99_to_target_mean": max(
                value["leakage"] for value in seed_quantiles
            ),
            "validation_patient_count": int(group["patient_id"].nunique()),
            "minimum_joint_observer_nullity": int(
                group["joint_observer_nullity"].min()
            ),
        }
    norm_payload = {
        "status": "PASS",
        "quantile": 0.99,
        "seed_aggregation": "maximum_seed_specific_quantile",
        "thresholds": thresholds,
        "maximum_reconstruction_error": configuration["intervention"][
            "maximum_reconstruction_error"
        ],
        "minimum_state_realization": configuration["intervention"][
            "minimum_state_realization"
        ],
    }
    norm_path = fit_root / "norm_and_leakage_calibration.json"
    _write_json_atomic(norm_path, norm_payload)
    history_path = fit_root / "history_admission.json"
    history = _load_json(history_path)
    status = "PASS" if history.get("status") == "PASS" else "HIGH_LEVEL_MODEL_MISSPECIFIED"
    manifest = {
        "status": status,
        "matching_calibration_sha256": sha256_file(matching_path),
        "norm_and_leakage_calibration_sha256": sha256_file(norm_path),
        "history_admission_sha256": sha256_file(history_path),
        "job_count": len(jobs),
        "full_activations_persisted": False,
    }
    _write_json_atomic(fit_root / "calibration_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare validation calibration or formal plans for V11."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/v11_causal_abstraction.yaml"),
    )
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument(
        "--phase",
        choices=("validation-job", "aggregate-validation", "formal-job"),
        required=True,
    )
    parser.add_argument("--model")
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    workspace = args.workspace_root.resolve()
    config_path = args.config if args.config.is_absolute() else workspace / args.config
    configuration = _load_yaml(config_path.resolve())
    validated = validate_v11_configuration(configuration)
    if args.phase == "aggregate-validation":
        result = aggregate_validation(workspace, configuration)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    if args.asset_root is None or args.model is None or args.model_seed is None:
        raise ValueError("job phases require --asset-root, --model, and --model-seed")
    allowed_models = validated.main_models + validated.control_models
    if args.model not in allowed_models or int(args.model_seed) not in validated.model_seeds:
        raise ValueError("requested model/seed is outside the V11 matrix")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    result = _prepare_job(
        workspace=workspace,
        asset_root=args.asset_root.resolve(),
        configuration=configuration,
        model=args.model,
        model_seed=int(args.model_seed),
        split_role=("validation" if args.phase == "validation-job" else "formal"),
        device=device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] in {"PASS", "COMPLETE_LOCKABLE_PLAN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
