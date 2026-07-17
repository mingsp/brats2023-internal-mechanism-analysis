from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
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
from pptt.interventions.causal_tracing import (
    adaptive_feature_mean,
    dominant_feature_truth_class,
)
from pptt.interventions.network_alignment import (
    NodeRestoreMasks,
    map_transition_mask,
    match_node_restore_masks,
)
from pptt.interventions.network_runtime import (
    NetworkAlignmentForwardSet,
    run_network_alignment_forwards,
)
from pptt.io.artifacts import CaseTrace, load_case_trace
from pptt.lineage.cohorts import (
    output_anchored_persistent_correction_cohorts,
    output_anchored_stable_correct_cohort,
    select_process_slice,
    union_persistent_correction_cohorts,
)
from pptt.models.adapters import build_adapter
from pptt.models.protocol import ModelAdapter
try:
    from scripts.lock_v7_network_alignment_protocol import (
        load_and_validate_formal_lock,
        sha256_file,
        source_tree_sha256,
    )
except ModuleNotFoundError:
    from lock_v7_network_alignment_protocol import (
        load_and_validate_formal_lock,
        sha256_file,
        source_tree_sha256,
    )


@dataclass(frozen=True)
class ModelRuntime:
    job: ModelMatrixJob
    adapter: ModelAdapter
    checkpoint_sha256: str


class JobAlreadyRunningError(RuntimeError):
    pass


class JobRunLock:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._stream: Any = None

    def __enter__(self) -> "JobRunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a+", encoding="utf-8")
        try:
            import fcntl

            fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._stream.close()
            self._stream = None
            raise JobAlreadyRunningError(
                f"JOB_ALREADY_RUNNING: {self.path}"
            ) from error
        self._stream.seek(0)
        self._stream.truncate()
        self._stream.write(
            json.dumps({"pid": os.getpid()}, ensure_ascii=False, sort_keys=True) + "\n"
        )
        self._stream.flush()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._stream is None:
            return
        try:
            import fcntl

            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        finally:
            self._stream.close()
            self._stream = None


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


def _validate_patient_payload(
    payload: Mapping[str, Any],
    *,
    patient_id: str,
    job_identity: Mapping[str, Any],
) -> None:
    if payload.get("patient_id") != patient_id:
        raise ValueError(f"patient output identity differs for {patient_id}")
    if payload.get("job_identity") != dict(job_identity):
        raise ValueError(f"patient output job identity differs for {patient_id}")
    cells = payload.get("matrix_cells")
    if not isinstance(cells, list) or len(cells) != 49:
        raise ValueError(f"patient output must contain 49 matrix cells: {patient_id}")


def run_patient_registry(
    job_root: str | Path,
    *,
    patient_ids: Sequence[str],
    job_identity: Mapping[str, Any],
    process_patient: Callable[[str], dict[str, Any]],
    resume: bool,
    execution_mode: str,
) -> dict[str, Any]:
    root = Path(job_root)
    patients = list(patient_ids)
    if len(patients) != len(set(patients)) or not patients:
        raise ValueError("job patient registry must be nonempty and unique")
    manifest = {
        "schema_version": 1,
        "execution_mode": str(execution_mode),
        "job_identity": dict(job_identity),
        "patient_ids": patients,
        "patient_count": len(patients),
    }
    manifest_path = root / "job_manifest.json"
    status_path = root / "job_status.json"
    with JobRunLock(root / ".run.lock"):
        if manifest_path.is_file():
            if _load_json(manifest_path) != _json_ready(manifest):
                raise ValueError("resume job identity differs from the registered manifest")
        else:
            _write_json_atomic(manifest_path, manifest)

        if resume and status_path.is_file():
            existing_status = _load_json(status_path)
            expected_complete = (
                "COMPLETE" if execution_mode == "formal" else "SMOKE_COMPLETE"
            )
            if existing_status.get("status") == expected_complete:
                for patient_id in patients:
                    patient_path = root / "patient_results" / f"{patient_id}.json"
                    if not patient_path.is_file():
                        raise ValueError("complete job is missing a registered patient output")
                    _validate_patient_payload(
                        _load_json(patient_path),
                        patient_id=patient_id,
                        job_identity=job_identity,
                    )
                return existing_status

        counts: dict[str, int] = defaultdict(int)
        for patient_index, patient_id in enumerate(patients, start=1):
            patient_path = root / "patient_results" / f"{patient_id}.json"
            if patient_path.is_file():
                if not resume:
                    raise ValueError(f"patient output already exists without --resume: {patient_id}")
                payload = _load_json(patient_path)
                _validate_patient_payload(
                    payload,
                    patient_id=patient_id,
                    job_identity=job_identity,
                )
            else:
                payload = process_patient(patient_id)
                payload["patient_id"] = patient_id
                payload["job_identity"] = dict(job_identity)
                _validate_patient_payload(
                    payload,
                    patient_id=patient_id,
                    job_identity=job_identity,
                )
                _write_json_atomic(patient_path, payload)
            counts[str(payload.get("status", "UNKNOWN"))] += 1
            print(
                f"v7 {job_identity.get('model')} seed={job_identity.get('model_seed')} "
                f"{patient_index}/{len(patients)} patient={patient_id} "
                f"status={payload.get('status')}",
                flush=True,
            )

        status = {
            "status": "COMPLETE" if execution_mode == "formal" else "SMOKE_COMPLETE",
            "execution_mode": execution_mode,
            "patient_count": len(patients),
            "status_counts": dict(sorted(counts.items())),
            "job_identity": dict(job_identity),
        }
        _write_json_atomic(status_path, status)
        return status


def _state_from_logits(logits: np.ndarray) -> np.ndarray:
    array = np.asarray(logits)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError("logits must have shape CxHxW or 1xCxHxW")
    if not np.isfinite(array).all():
        raise ValueError("logits must be finite")
    return array.argmax(axis=0).astype(np.uint8, copy=False)


def _retention(
    logits: np.ndarray,
    truth: np.ndarray,
    target_mask: np.ndarray,
) -> float:
    target = np.asarray(target_mask)
    labels = np.asarray(truth)
    if target.shape != labels.shape or target.dtype != np.bool_:
        raise ValueError("target mask must be boolean and align with truth")
    if not target.any():
        return float("nan")
    state = _state_from_logits(logits)
    if state.shape != labels.shape:
        raise ValueError("final model state does not align with truth")
    return float((state[target] == labels[target]).mean())


def build_matrix_cells(
    forward_set: NetworkAlignmentForwardSet,
    *,
    truth: np.ndarray,
    transition_masks: Mapping[int, np.ndarray],
    nodes: Sequence[str],
    cell_metadata: Mapping[tuple[int, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ordered_nodes = tuple(str(value) for value in nodes)
    if len(ordered_nodes) != 8:
        raise ValueError("V7 matrix requires exactly eight registered nodes")
    cells: list[dict[str, Any]] = []
    for transition_index in range(7):
        transition = f"{ordered_nodes[transition_index]}->{ordered_nodes[transition_index + 1]}"
        target_pixels = np.asarray(transition_masks[transition_index])
        if target_pixels.dtype != np.bool_:
            raise ValueError("transition masks must be boolean")
        clean_retention = (
            _retention(forward_set.clean_logits, truth, target_pixels)
            if target_pixels.any()
            else None
        )
        corrupt_retention = (
            _retention(forward_set.corrupt_logits, truth, target_pixels)
            if target_pixels.any()
            else None
        )
        total_effect = (
            float(clean_retention - corrupt_retention)
            if clean_retention is not None and corrupt_retention is not None
            else None
        )
        receiving_node = ordered_nodes[transition_index + 1]
        for restore_node in ordered_nodes[1:]:
            metadata = dict(cell_metadata[(transition_index, restore_node)])
            status = str(metadata["status"])
            target_retention: float | None = None
            target_effect: float | None = None
            control_retention: float | None = None
            control_effect: float | None = None
            specific_effect: float | None = None
            key = f"{restore_node}/transition_{transition_index}"
            target_allowed = status in {
                "EVALUABLE",
                "TARGET_ONLY_CONTROL_UNAVAILABLE",
            }
            if target_allowed:
                if key not in forward_set.target_restored_logits:
                    raise RuntimeError(f"missing target restore output for {key}")
                target_retention = _retention(
                    forward_set.target_restored_logits[key],
                    truth,
                    target_pixels,
                )
                target_effect = float(target_retention - corrupt_retention)
                control_logits = forward_set.control_restored_logits.get(key)
                if status == "EVALUABLE":
                    if control_logits is None:
                        raise RuntimeError(f"missing control restore output for {key}")
                    control_retention = _retention(
                        control_logits,
                        truth,
                        target_pixels,
                    )
                    control_effect = float(control_retention - corrupt_retention)
                    specific_effect = float(target_retention - control_retention)
            cells.append(
                {
                    "transition_index": transition_index,
                    "transition": transition,
                    "restore_node": restore_node,
                    "receiving_node": receiving_node,
                    "is_receiving_node_cell": restore_node == receiving_node,
                    "status": status,
                    "target_pixel_count": int(metadata.get("target_pixel_count", 0)),
                    "target_feature_count": int(metadata.get("target_feature_count", 0)),
                    "control_feature_count": int(metadata.get("control_feature_count", 0)),
                    "clean_retention": clean_retention,
                    "corrupt_retention": corrupt_retention,
                    "total_effect": total_effect,
                    "target_retention": target_retention,
                    "target_effect": target_effect,
                    "control_retention": control_retention,
                    "control_effect": control_effect,
                    "specific_effect": specific_effect,
                }
            )
    return cells


def _null_matrix_cells(
    *,
    nodes: Sequence[str],
    status: str,
    transition_pixel_counts: Sequence[int],
) -> list[dict[str, Any]]:
    cells = []
    for transition_index in range(7):
        transition = f"{nodes[transition_index]}->{nodes[transition_index + 1]}"
        for restore_node in nodes[1:]:
            cells.append(
                {
                    "transition_index": transition_index,
                    "transition": transition,
                    "restore_node": restore_node,
                    "receiving_node": nodes[transition_index + 1],
                    "is_receiving_node_cell": restore_node == nodes[transition_index + 1],
                    "status": status,
                    "target_pixel_count": int(transition_pixel_counts[transition_index]),
                    "target_feature_count": 0,
                    "control_feature_count": 0,
                    "clean_retention": None,
                    "corrupt_retention": None,
                    "total_effect": None,
                    "target_retention": None,
                    "target_effect": None,
                    "control_retention": None,
                    "control_effect": None,
                    "specific_effect": None,
                }
            )
    return cells


def _group_records(records: list[SliceRecord]) -> dict[str, list[SliceRecord]]:
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


def _load_runtime(job: ModelMatrixJob, *, device: torch.device) -> ModelRuntime:
    if not job.checkpoint.is_file():
        raise FileNotFoundError(job.checkpoint)
    model_config = _load_yaml(job.model_config)
    adapter = build_adapter(
        job.model,
        n_channels=int(model_config["n_channels"]),
        num_classes=int(model_config["num_classes"]),
        bilinear=bool(model_config.get("bilinear", False)),
        img_size=int(model_config.get("img_size", 160)),
    )
    checkpoint_sha = adapter.load_checkpoint(job.checkpoint, map_location="cpu")
    adapter = adapter.to(device).eval()
    return ModelRuntime(job=job, adapter=adapter, checkpoint_sha256=checkpoint_sha)


def _job_for(
    jobs: Sequence[ModelMatrixJob], *, model: str, model_seed: int
) -> ModelMatrixJob:
    matches = [job for job in jobs if job.model == model and job.seed == model_seed]
    if len(matches) != 1:
        raise ValueError(f"expected one model job for {model}/seed_{model_seed}")
    return matches[0]


def _registered_checkpoint(lock: Mapping[str, Any], model: str, seed: int) -> str:
    matches = [
        row
        for row in lock.get("registered_model_jobs", [])
        if row.get("model") == model and int(row.get("model_seed", -1)) == seed
    ]
    if len(matches) != 1:
        raise ValueError(f"protocol lock lacks model job {model}/seed_{seed}")
    return str(matches[0]["checkpoint_sha256"])


def _cell_masks_and_metadata(
    *,
    clean_activations: Mapping[str, torch.Tensor],
    cohorts: np.ndarray,
    states: np.ndarray,
    truth: np.ndarray,
    reliable: np.ndarray,
    final_model_state: np.ndarray,
    nodes: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[
    dict[str, dict[int, NodeRestoreMasks]],
    dict[tuple[int, str], dict[str, Any]],
]:
    eligibility = config["eligibility"]
    matching = config["matching"]
    truth_classes = tuple(int(value) for value in config["truth_classes"])
    union = union_persistent_correction_cohorts(cohorts)
    boundary = ndimage.distance_transform_edt(
        np.isin(truth, truth_classes)
    ).astype(np.float32)
    masks: dict[str, dict[int, NodeRestoreMasks]] = {node: {} for node in nodes[1:]}
    metadata: dict[tuple[int, str], dict[str, Any]] = {}
    stable_controls = {
        transition_index: output_anchored_stable_correct_cohort(
            states,
            truth,
            reliable,
            final_model_state,
            transition_index=transition_index,
            truth_classes=truth_classes,
            exclude=union,
        )
        for transition_index in range(7)
    }
    for node in nodes[1:]:
        clean_activation = clean_activations[node]
        if clean_activation.ndim != 4 or clean_activation.shape[0] != 1:
            raise ValueError(f"clean activation for {node} must have shape 1xCxHxW")
        feature_shape = tuple(int(value) for value in clean_activation.shape[-2:])
        feature_truth = dominant_feature_truth_class(
            truth,
            output_shape=feature_shape,
            truth_classes=truth_classes,
        )
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
        for transition_index in range(7):
            target_pixels = cohorts[transition_index]
            target_pixel_count = int(target_pixels.sum())
            base = {
                "target_pixel_count": target_pixel_count,
                "target_feature_count": 0,
                "control_feature_count": 0,
            }
            if target_pixel_count < int(eligibility["minimum_transition_pixels"]):
                metadata[(transition_index, node)] = {
                    **base,
                    "status": "INELIGIBLE_TRANSITION_PIXELS",
                }
                continue
            target_feature = map_transition_mask(
                target_pixels,
                output_shape=feature_shape,
            )
            target_feature_count = int(target_feature.sum())
            base["target_feature_count"] = target_feature_count
            if target_feature_count < int(
                eligibility["minimum_target_feature_positions"]
            ):
                metadata[(transition_index, node)] = {
                    **base,
                    "status": "INELIGIBLE_TARGET_FEATURE_POSITIONS",
                }
                continue
            control_feature = map_transition_mask(
                stable_controls[transition_index],
                output_shape=feature_shape,
            ) & ~target_feature
            registered = match_node_restore_masks(
                target_mask=target_feature,
                control_pool=control_feature,
                truth_class=feature_truth,
                boundary_distance=feature_boundary,
                activation_norm=activation_norm,
                boundary_bin_edges=tuple(matching["boundary_bin_edges"]),
                activation_norm_quantile_bins=int(
                    matching["activation_norm_quantile_bins"]
                ),
            )
            control_count = int(registered.control_count)
            base["control_feature_count"] = control_count
            status = "EVALUABLE"
            if (
                registered.control is None
                or control_count
                < int(eligibility["minimum_control_feature_positions"])
            ):
                registered = NodeRestoreMasks(
                    target=registered.target,
                    control=None,
                    target_count=registered.target_count,
                    control_count=0,
                )
                base["control_feature_count"] = 0
                status = "TARGET_ONLY_CONTROL_UNAVAILABLE"
            masks[node][transition_index] = registered
            metadata[(transition_index, node)] = {**base, "status": status}
    return masks, metadata


def _run_patient(
    runtime: ModelRuntime,
    formal_trace: CaseTrace,
    records: Sequence[SliceRecord],
    *,
    config: Mapping[str, Any],
    patient_id: str,
    device: torch.device,
) -> dict[str, Any]:
    if formal_trace.truth is None or formal_trace.final_model_state is None:
        raise ValueError("formal V3 trace must retain truth and native final model state")
    nodes = tuple(str(value) for value in config["nodes"])
    expected_slices = tuple(record.slice_id for record in records)
    if formal_trace.slice_ids != expected_slices:
        raise ValueError(f"formal trace and source slices differ for {patient_id}")
    cohorts_all = output_anchored_persistent_correction_cohorts(
        formal_trace.states,
        formal_trace.truth,
        formal_trace.reliable,
        formal_trace.final_model_state,
        truth_classes=tuple(int(value) for value in config["truth_classes"]),
    )
    selection = select_process_slice(cohorts_all)
    if selection.union_pixel_count < int(
        config["eligibility"]["minimum_union_pixels_on_selected_slice"]
    ):
        return {
            "status": "INELIGIBLE_UNION_PIXELS",
            "selected_slice_id": expected_slices[selection.slice_index],
            "selected_slice_index": selection.slice_index,
            "union_pixel_count": selection.union_pixel_count,
            "transition_pixel_counts": list(selection.transition_pixel_counts),
            "matrix_cells": _null_matrix_cells(
                nodes=nodes,
                status="INELIGIBLE_UNION_PIXELS",
                transition_pixel_counts=selection.transition_pixel_counts,
            ),
            "audit": {},
        }

    slice_index = selection.slice_index
    record = records[slice_index]
    image, truth = _load_slice(record)
    if not np.array_equal(truth, formal_trace.truth[slice_index]):
        raise ValueError(f"formal truth and source mask differ for {record.slice_id}")
    image = image.to(device)
    with torch.inference_mode():
        clean_trace = runtime.adapter.trace(image)
    clean_final = clean_trace.logits.argmax(dim=1)[0].detach().cpu().numpy()
    final_mismatch = int(
        np.count_nonzero(clean_final != formal_trace.final_model_state[slice_index])
    )
    if final_mismatch:
        raise RuntimeError(
            f"clean rerun does not reproduce the formal V3 final state for {patient_id}"
        )

    cohorts = cohorts_all[:, slice_index]
    masks, metadata = _cell_masks_and_metadata(
        clean_activations=clean_trace.activations,
        cohorts=cohorts,
        states=formal_trace.states[:, slice_index],
        truth=truth,
        reliable=formal_trace.reliable[:, slice_index],
        final_model_state=formal_trace.final_model_state[slice_index],
        nodes=nodes,
        config=config,
    )
    forward_set = run_network_alignment_forwards(
        runtime.adapter,
        image,
        root_node=str(config["intervention"]["root_node"]),
        restore_nodes=tuple(str(value) for value in config["intervention"]["restore_nodes"]),
        masks=masks,
        input_equivalent_shift_yx=tuple(
            int(value) for value in config["intervention"]["input_equivalent_shift_yx"]
        ),
        logit_tolerance=float(config["intervention"]["logit_tolerance"]),
        clean_trace=clean_trace,
    )
    cells = build_matrix_cells(
        forward_set,
        truth=truth,
        transition_masks={index: cohorts[index] for index in range(7)},
        nodes=nodes,
        cell_metadata=metadata,
    )
    evaluable_count = sum(cell["status"] == "EVALUABLE" for cell in cells)
    return {
        "status": "PASS" if evaluable_count else "NO_EVALUABLE_SPECIFICITY",
        "selected_slice_id": record.slice_id,
        "selected_slice_index": slice_index,
        "union_pixel_count": selection.union_pixel_count,
        "transition_pixel_counts": list(selection.transition_pixel_counts),
        "evaluable_matrix_cell_count": evaluable_count,
        "matrix_cells": cells,
        "audit": {
            "clean_final_state_mismatch_count": final_mismatch,
            **forward_set.audits,
        },
    }


def _validate_historical_state(
    workspace: Path,
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> None:
    v4_path = workspace / str(config["v3_root"]) / "v4_trigger.json"
    v6_path = (
        workspace
        / "results"
        / "v6_pixel_transition_causal"
        / "v6_protocol_lock_v2.json"
    )
    if sha256_file(v4_path) != lock["v4_state"]["sha256"]:
        raise ValueError("V4 state changed after the V7 protocol lock")
    if sha256_file(v6_path) != lock["v6_scope"]["sha256"]:
        raise ValueError("V6 lock changed after the V7 protocol lock")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the locked network-wide PPTT process-intervention alignment scan."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/v7_network_process_intervention_alignment.yaml"
        ),
    )
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path(
            "results/v7_network_process_intervention_alignment/v7_protocol_lock.json"
        ),
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/brats2023_2d.yaml"),
    )
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execution-mode", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--max-patients", type=int)
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    config_path = args.config if args.config.is_absolute() else workspace / args.config
    lock_path = (
        args.protocol_lock
        if args.protocol_lock.is_absolute()
        else workspace / args.protocol_lock
    )
    data_config_path = (
        args.data_config
        if args.data_config.is_absolute()
        else workspace / args.data_config
    )
    config = _load_yaml(config_path.resolve())
    data_config = _load_yaml(data_config_path.resolve())
    if args.model not in config["models"] or args.model_seed not in config["model_seeds"]:
        raise ValueError("model identity is outside the registered V7 matrix")
    if args.execution_mode == "formal" and args.max_patients is not None:
        raise ValueError("formal V7 cannot truncate the registered patient set")
    if args.execution_mode == "smoke" and args.max_patients is None:
        raise ValueError("smoke V7 requires --max-patients")

    matrix = load_model_matrix(
        workspace / str(config["model_matrix"]),
        workspace_root=workspace,
        asset_root_override=args.asset_root,
    )
    split = str(config["primary_split"])
    split_config = data_config["splits"][split]
    grouped = _group_records(
        discover_slice_records(
            matrix.asset_root / str(split_config["image_dir"]),
            matrix.asset_root / str(split_config["mask_dir"]),
        )
    )
    split_payload = _load_json(matrix.asset_root / str(data_config["split_file"]))
    full_patient_ids = [str(value) for value in split_payload[split]]
    if len(full_patient_ids) != 250 or set(full_patient_ids) != set(grouped):
        raise ValueError("active data patient registry differs from the locked test split")
    lock = load_and_validate_formal_lock(
        lock_path.resolve(),
        active_configuration_sha256=sha256_file(config_path.resolve()),
        active_source_tree_sha256=source_tree_sha256(workspace),
        patient_ids=full_patient_ids,
    )
    _validate_historical_state(workspace, config, lock)

    job = _job_for(matrix.jobs, model=args.model, model_seed=args.model_seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    runtime = _load_runtime(job, device=device)
    registered_checkpoint = _registered_checkpoint(
        lock, args.model, args.model_seed
    )
    if runtime.checkpoint_sha256 != registered_checkpoint:
        raise ValueError("active model checkpoint differs from the V7 protocol lock")

    patient_ids = full_patient_ids
    output_root = workspace / str(config["output_root"])
    if args.execution_mode == "smoke":
        patient_ids = full_patient_ids[: int(args.max_patients)]
        output_root = workspace / "results" / "v7_smoke"
    job_root = output_root / args.model / f"seed_{args.model_seed}"
    protocol_sha = sha256_file(lock_path.resolve())
    job_identity = {
        "model": args.model,
        "model_seed": args.model_seed,
        "checkpoint_sha256": runtime.checkpoint_sha256,
        "protocol_lock_sha256": protocol_sha,
        "execution_mode": args.execution_mode,
    }
    trace_root = workspace / str(config["v3_root"])

    def process_patient(patient_id: str) -> dict[str, Any]:
        trace_path = (
            trace_root
            / args.model
            / f"seed_{args.model_seed}"
            / split
            / "case_traces"
            / f"{patient_id}.npz"
        )
        if not trace_path.is_file():
            raise FileNotFoundError(trace_path)
        return _run_patient(
            runtime,
            load_case_trace(trace_path),
            grouped[patient_id],
            config=config,
            patient_id=patient_id,
            device=device,
        )

    status = run_patient_registry(
        job_root,
        patient_ids=patient_ids,
        job_identity=job_identity,
        process_patient=process_patient,
        resume=args.resume,
        execution_mode=args.execution_mode,
    )
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
