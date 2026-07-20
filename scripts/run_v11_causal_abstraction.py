from __future__ import annotations

import argparse
from collections import OrderedDict
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import torch
from torch import nn
import yaml

from pptt.causal_abstraction.interventions import (
    bilinear_resize_rows,
    equal_norm_nullspace_control,
    minimum_norm_state_exchange,
    project_feature_edit,
    stack_observer_contrast_weights,
    stack_restart_logit_contrasts,
)
from pptt.causal_abstraction.runtime import CounterfactualTrace, run_state_exchange
from pptt.causal_abstraction.states import relationship_states
from pptt.data.brats2d import (
    SliceRecord,
    discover_slice_records,
    ensure_chw,
    normalize_label,
)
from pptt.data.patient_splits import patient_id_from_slice
from pptt.experiments.model_matrix import ModelMatrixJob, load_model_matrix
from pptt.models.adapters import HookedModelAdapter, build_adapter
from pptt.models.protocol import ModelAdapter
from pptt.observers.linear import LinearObserver
from pptt.pipeline.trace_model import (
    load_formal_reliability_threshold,
    load_real_observer_path,
)
from pptt.types import ForwardTrace
from scripts.lock_v11_causal_abstraction_protocol import (
    EXPECTED_CONTROL_MODELS,
    EXPECTED_MAIN_MODELS,
    EXPECTED_MODEL_SEEDS,
    EXPECTED_OBSERVER_SEEDS,
    load_and_validate_v11_protocol_lock,
    sha256_file,
    sha256_named_values,
    source_tree_sha256,
    validate_v11_configuration,
)


NODE_NAMES = (
    "down1",
    "down2",
    "down3",
    "down4",
    "up1",
    "up2",
    "up3",
    "up4",
)
FORMAL_DOSES = (0.0, 0.25, 0.5, 0.75, 1.0)
CONDITION_NAMES = ("task", "null")


@dataclass(frozen=True)
class FormalRuntime:
    job: ModelMatrixJob
    adapter: ModelAdapter
    observers: Mapping[str, Mapping[int, LinearObserver]]
    reliability_threshold: float
    checkpoint_sha256: str
    observer_hashes: Mapping[str, str]


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


def _stable_seed(*values: Any) -> int:
    encoded = "|".join(str(value) for value in values).encode("utf-8")
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
    selected = [
        job
        for job in jobs
        if job.model == str(model) and job.seed == int(model_seed)
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one model job for {model}/seed_{model_seed}")
    return selected[0]


def _observer_registry_hashes(
    workspace: Path,
    configuration: Mapping[str, Any],
    *,
    model: str,
    model_seed: int,
) -> dict[str, str]:
    root = (
        workspace
        / str(configuration["observer_root"])
        / model
        / f"seed_{model_seed}"
        / "observers"
    )
    hashes = {}
    for node in NODE_NAMES:
        for observer_seed in EXPECTED_OBSERVER_SEEDS:
            path = root / node / "real" / f"seed_{observer_seed}.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            hashes[f"{node}/seed_{observer_seed}"] = sha256_file(path)
    return hashes


def _load_formal_runtime(
    job: ModelMatrixJob,
    *,
    workspace: Path,
    configuration: Mapping[str, Any],
    device: torch.device,
) -> FormalRuntime:
    model_config = _load_yaml(job.model_config)
    adapter = build_adapter(
        job.model,
        n_channels=int(model_config["n_channels"]),
        num_classes=int(model_config["num_classes"]),
        bilinear=bool(model_config.get("bilinear", False)),
        img_size=int(model_config.get("img_size", 160)),
    )
    if tuple(adapter.checkpoint_names) != NODE_NAMES:
        raise ValueError(f"adapter node order differs for {job.job_id}")
    checkpoint_hash = adapter.load_checkpoint(job.checkpoint, map_location="cpu")
    observer_directory = (
        workspace
        / str(configuration["observer_root"])
        / job.model
        / f"seed_{job.seed}"
    )
    threshold = load_formal_reliability_threshold(observer_directory)
    observers = load_real_observer_path(
        observer_directory,
        nodes=NODE_NAMES,
        seeds=EXPECTED_OBSERVER_SEEDS,
    )
    return FormalRuntime(
        job=job,
        adapter=adapter.to(device).eval(),
        observers={
            node: {
                seed: observers[node][seed].to(device).eval()
                for seed in EXPECTED_OBSERVER_SEEDS
            }
            for node in NODE_NAMES
        },
        reliability_threshold=threshold,
        checkpoint_sha256=checkpoint_hash,
        observer_hashes=_observer_registry_hashes(
            workspace,
            configuration,
            model=job.model,
            model_seed=job.seed,
        ),
    )


class JobAlreadyRunningError(RuntimeError):
    pass


class JobRunLock:
    """A cross-platform, fail-closed single-writer job lock."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._owned = False

    def __enter__(self) -> "JobRunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            descriptor = os.open(self.path, flags)
        except FileExistsError as error:
            raise JobAlreadyRunningError(
                f"JOB_ALREADY_RUNNING_OR_STALE_LOCK: {self.path}"
            ) from error
        try:
            payload = json.dumps(
                {"pid": os.getpid()},
                ensure_ascii=False,
                sort_keys=True,
            ) + "\n"
            os.write(descriptor, payload.encode("utf-8"))
            os.fsync(descriptor)
            self._owned = True
        finally:
            os.close(descriptor)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._owned:
            self.path.unlink(missing_ok=True)
            self._owned = False


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


def validate_execution_request(
    *,
    execution_mode: str,
    patient_limit: int | None,
    protocol_lock: Path,
) -> None:
    mode = str(execution_mode)
    if mode not in {"smoke", "formal"}:
        raise ValueError("execution mode must be smoke or formal")
    if mode == "smoke":
        if patient_limit is None or int(patient_limit) <= 0:
            raise ValueError("smoke mode requires a positive patient limit")
        return
    if patient_limit is not None:
        raise ValueError("formal mode rejects every patient limit")
    if not Path(protocol_lock).is_file():
        raise FileNotFoundError(f"formal protocol lock is missing: {protocol_lock}")


def validate_patient_plan(
    payload: Mapping[str, Any],
    *,
    model: str,
    model_seed: int,
    patient_id: str,
    split: str = "test",
    output_size: int = 160 * 160,
    maximum_pairs_per_node: int = 32,
) -> None:
    """Validate one immutable patient plan before any model forward pass."""

    expected_identity = {
        "model": str(model),
        "model_seed": int(model_seed),
        "patient_id": str(patient_id),
        "split": str(split),
    }
    for field, expected in expected_identity.items():
        if payload.get(field) != expected:
            raise ValueError(f"patient plan {field} differs for {patient_id}")
    if payload.get("schema_version") != 1 or payload.get("status") != "REGISTERED_PATIENT_PLAN":
        raise ValueError(f"patient plan schema or status differs for {patient_id}")
    node_plans = payload.get("nodes")
    if not isinstance(node_plans, list) or [row.get("node") for row in node_plans] != list(NODE_NAMES):
        raise ValueError(f"patient plan lacks the complete ordered node path: {patient_id}")
    for node_plan in node_plans:
        node = str(node_plan["node"])
        base_slice_id = node_plan.get("base_slice_id")
        pairs = node_plan.get("pairs")
        if not isinstance(base_slice_id, str) or not base_slice_id:
            raise ValueError(f"patient plan has no base slice for {patient_id}/{node}")
        if not isinstance(pairs, list):
            raise ValueError(f"patient plan pairs must be a list for {patient_id}/{node}")
        if len(pairs) > int(maximum_pairs_per_node):
            raise ValueError(f"patient plan exceeds the locked pair cap for {patient_id}/{node}")
        expected_status = "MATCHED" if pairs else "NO_MATCH"
        if node_plan.get("status") != expected_status:
            raise ValueError(f"patient plan status differs for {patient_id}/{node}")
        native_shape = node_plan.get("native_shape")
        if pairs:
            if (
                not isinstance(native_shape, list)
                or len(native_shape) != 2
                or any(int(value) <= 0 for value in native_shape)
            ):
                raise ValueError(f"patient plan native shape is invalid for {patient_id}/{node}")
        elif native_shape is not None:
            raise ValueError(f"unmatched patient plan must not register a native shape: {patient_id}/{node}")
        base_indices: set[int] = set()
        source_keys: set[tuple[str, str, int]] = set()
        for pair in pairs:
            if not isinstance(pair, dict):
                raise ValueError(f"patient plan pair is not a mapping for {patient_id}/{node}")
            base_index = int(pair.get("base_output_index", -1))
            source_index = int(pair.get("source_output_index", -1))
            base_state = int(pair.get("base_state", -1))
            target_state = int(pair.get("target_state", -1))
            truth_class = int(pair.get("truth_class", -1))
            source_patient = str(pair.get("source_patient_id", ""))
            source_slice = str(pair.get("source_slice_id", ""))
            if not 0 <= base_index < int(output_size) or not 0 <= source_index < int(output_size):
                raise ValueError(f"patient plan contains an out-of-range pixel for {patient_id}/{node}")
            if base_index in base_indices:
                raise ValueError(f"patient plan repeats a base pixel for {patient_id}/{node}")
            base_indices.add(base_index)
            source_key = (source_patient, source_slice, source_index)
            if not source_patient or not source_slice or source_key in source_keys:
                raise ValueError(f"patient plan repeats or omits a source pixel for {patient_id}/{node}")
            source_keys.add(source_key)
            if not 0 <= base_state < 5 or not 0 <= target_state < 5 or base_state == target_state:
                raise ValueError(f"patient plan has an invalid state exchange for {patient_id}/{node}")
            if not 0 <= truth_class < 4:
                raise ValueError(f"patient plan has an invalid truth class for {patient_id}/{node}")


def validate_locked_plan_manifest(
    manifest_path: Path,
    *,
    lock: Mapping[str, Any],
    model: str,
    model_seed: int,
) -> tuple[dict[str, Any], Path]:
    """Verify the job manifest and every patient-plan hash against the lock."""

    identity = f"{model}/seed_{int(model_seed)}"
    registered_hash = lock.get("intervention_plan_hashes", {}).get(identity)
    if registered_hash is None or sha256_file(manifest_path) != registered_hash:
        raise ValueError(f"active intervention-plan manifest differs from the V11 lock: {identity}")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("status") != "COMPLETE_LOCKABLE_PLAN"
        or manifest.get("model") != model
        or int(manifest.get("model_seed", -1)) != int(model_seed)
        or manifest.get("split") != "test"
        or manifest.get("failed_nodes") != []
    ):
        raise ValueError(f"intervention-plan manifest is not lockable: {identity}")
    patients = list(lock.get("test_patient_ids", []))
    if manifest.get("patient_ids") != patients or manifest.get("patient_count") != len(patients):
        raise ValueError(f"intervention-plan patient registry differs from the lock: {identity}")
    plan_hashes = manifest.get("patient_plan_hashes")
    if not isinstance(plan_hashes, dict) or list(plan_hashes) != patients:
        raise ValueError(f"intervention-plan hashes are incomplete or unordered: {identity}")
    if manifest.get("patient_plan_registry_sha256") != sha256_named_values(plan_hashes):
        raise ValueError(f"intervention-plan registry hash differs: {identity}")
    coverage = manifest.get("coverage")
    coverage_limits = lock.get("configuration", {}).get("coverage", {})
    if not isinstance(coverage, dict) or set(coverage) != set(NODE_NAMES):
        raise ValueError(f"intervention-plan coverage is incomplete: {identity}")
    for node in NODE_NAMES:
        summary = coverage[node]
        target_states = summary.get("target_states", {})
        evaluable = [
            state
            for state, values in target_states.items()
            if int(values.get("patient_count", -1))
            >= int(coverage_limits.get("minimum_state_patients", -1))
            and int(values.get("pixel_count", -1))
            >= int(coverage_limits.get("minimum_state_pixels", -1))
        ]
        if (
            int(summary.get("patient_count", -1))
            < int(coverage_limits.get("minimum_node_patients", -1))
            or len(evaluable)
            < int(coverage_limits.get("minimum_source_states", -1))
            or sorted(str(value) for value in evaluable)
            != sorted(str(value) for value in summary.get("evaluable_target_states", []))
        ):
            raise ValueError(f"intervention-plan coverage gate failed: {identity}/{node}")
    if manifest.get("full_activations_persisted") is not False:
        raise ValueError(f"intervention plan may not persist full activations: {identity}")
    plan_root = manifest_path.parent / "patient_plans"
    for patient_id in patients:
        path = plan_root / f"{patient_id}.json"
        if not path.is_file() or sha256_file(path) != str(plan_hashes[patient_id]):
            raise ValueError(f"patient intervention plan differs from the manifest: {patient_id}")
    return manifest, plan_root


def validate_runtime_against_lock(
    runtime: FormalRuntime,
    lock: Mapping[str, Any],
) -> None:
    model_row = [
        row
        for row in lock.get("registered_model_jobs", [])
        if row.get("model") == runtime.job.model
        and int(row.get("model_seed", -1)) == runtime.job.seed
    ]
    if len(model_row) != 1 or model_row[0].get("checkpoint_sha256") != runtime.checkpoint_sha256:
        raise ValueError(f"active checkpoint differs from the V11 lock: {runtime.job.job_id}")
    expected_observers = {
        f"{row['node']}/seed_{int(row['observer_seed'])}": str(row["sha256"])
        for row in lock.get("registered_observer_jobs", [])
        if row.get("model") == runtime.job.model
        and int(row.get("model_seed", -1)) == runtime.job.seed
    }
    if expected_observers != dict(runtime.observer_hashes):
        raise ValueError(f"active observers differ from the V11 lock: {runtime.job.job_id}")
    thresholds = {
        float(row["reliability_threshold"])
        for row in lock.get("registered_observer_jobs", [])
        if row.get("model") == runtime.job.model
        and int(row.get("model_seed", -1)) == runtime.job.seed
    }
    if thresholds != {float(runtime.reliability_threshold)}:
        raise ValueError(f"active reliability threshold differs from the V11 lock: {runtime.job.job_id}")


def _active_validation_hashes(
    workspace: Path,
    configuration: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    fit_root = workspace / str(configuration["validation_fit_root"])
    processes = {
        name: sha256_file(fit_root / f"{name}.json")
        for name in ("H_U", "H_T", "H_shared")
    }
    calibrations = {
        "matching": sha256_file(fit_root / "matching_calibration.json"),
        "norm_and_leakage": sha256_file(
            fit_root / "norm_and_leakage_calibration.json"
        ),
        "history_admission": sha256_file(fit_root / "history_admission.json"),
    }
    return processes, calibrations


def _validate_data_inventory(
    workspace: Path,
    asset_root: Path,
    lock: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    data_config_path = workspace / "configs" / "data" / "brats2023_2d.yaml"
    data_config = _load_yaml(data_config_path)
    split_path = asset_root / str(data_config["split_file"])
    split_payload = _load_json(split_path)
    inventory = {
        "data_config_sha256": sha256_file(data_config_path),
        "split_sha256": sha256_file(split_path),
    }
    if inventory != lock.get("data_inventory"):
        raise ValueError("active BraTS data inventory differs from the V11 lock")
    patients = sorted(str(value) for value in split_payload.get("test", []))
    if patients != list(lock.get("test_patient_ids", [])):
        raise ValueError("active BraTS test registry differs from the V11 lock")
    return data_config, patients


def _validate_patient_payload(
    payload: Mapping[str, Any],
    *,
    patient_id: str,
    job_identity: Mapping[str, Any],
) -> None:
    if payload.get("patient_id") != patient_id:
        raise ValueError(f"patient output identity differs for {patient_id}")
    if payload.get("job_identity") != dict(job_identity):
        raise ValueError(f"patient job identity differs for {patient_id}")
    if payload.get("nodes") != list(NODE_NAMES):
        raise ValueError(f"patient output lacks the complete node path: {patient_id}")
    node_results = payload.get("node_results")
    if not isinstance(node_results, list) or len(node_results) != len(NODE_NAMES):
        raise ValueError(f"patient output must contain eight node results: {patient_id}")
    for node_index, node_result in enumerate(node_results):
        if not isinstance(node_result, dict) or node_result.get("node") != NODE_NAMES[node_index]:
            raise ValueError(f"patient node order differs for {patient_id}")
        conditions = node_result.get("conditions")
        if not isinstance(conditions, dict) or set(conditions) != set(CONDITION_NAMES):
            raise ValueError(f"patient conditions are incomplete for {patient_id}")
        expected_path = [*NODE_NAMES[node_index:], "Y"]
        for condition in conditions.values():
            if condition.get("doses") != list(FORMAL_DOSES):
                raise ValueError(f"patient doses differ for {patient_id}")
            if condition.get("downstream_path") != expected_path:
                raise ValueError(f"patient downstream path differs for {patient_id}")


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
    patients = [str(value) for value in patient_ids]
    if not patients or len(patients) != len(set(patients)):
        raise ValueError("job patient registry must be nonempty and unique")
    if patients != sorted(patients):
        raise ValueError("job patient registry must be sorted")
    manifest = {
        "schema_version": 1,
        "execution_mode": str(execution_mode),
        "job_identity": dict(job_identity),
        "patient_ids": patients,
        "patient_count": len(patients),
        "full_activations_persisted": False,
    }
    manifest_path = root / "job_manifest.json"
    status_path = root / "job_status.json"
    completed = 0
    skipped = 0
    with JobRunLock(root / ".run.lock"):
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing != _json_ready(manifest):
                raise ValueError("resume identity differs from the registered job manifest")
        else:
            _write_json_atomic(manifest_path, manifest)

        result_root = root / "patient_results"
        result_root.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(
            status_path,
            {
                "status": "RUNNING",
                "execution_mode": execution_mode,
                "formal_claim_eligible": False,
                "formal_summary_eligible": False,
                "patient_count": len(patients),
                "completed_patients": 0,
                "skipped_completed_patients": 0,
                "job_identity": dict(job_identity),
            },
        )
        active_patient: str | None = None
        try:
            for patient_index, patient_id in enumerate(patients, start=1):
                active_patient = patient_id
                destination = result_root / patient_id
                result_path = destination / "patient_result.json"
                if destination.exists():
                    if not resume:
                        raise ValueError(
                            f"patient output already exists without resume: {patient_id}"
                        )
                    if not result_path.is_file():
                        raise ValueError(f"partial patient directory exists: {destination}")
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                    _validate_patient_payload(
                        payload,
                        patient_id=patient_id,
                        job_identity=job_identity,
                    )
                    skipped += 1
                else:
                    temporary: Path | None = Path(
                        tempfile.mkdtemp(
                            dir=result_root,
                            prefix=f".{patient_id}.",
                        )
                    )
                    try:
                        payload = process_patient(patient_id)
                        payload["patient_id"] = patient_id
                        payload["job_identity"] = dict(job_identity)
                        _validate_patient_payload(
                            payload,
                            patient_id=patient_id,
                            job_identity=job_identity,
                        )
                        _write_json_atomic(temporary / "patient_result.json", payload)
                        os.replace(temporary, destination)
                        temporary = None
                    finally:
                        if temporary is not None and temporary.exists():
                            shutil.rmtree(temporary)
                completed += 1
                print(
                    f"v11 {job_identity.get('model')} seed={job_identity.get('model_seed')} "
                    f"{patient_index}/{len(patients)} patient={patient_id}",
                    flush=True,
                )
        except Exception as error:
            _write_json_atomic(
                status_path,
                {
                    "status": "FAILED",
                    "execution_mode": execution_mode,
                    "formal_claim_eligible": False,
                    "formal_summary_eligible": False,
                    "patient_count": len(patients),
                    "completed_patients": completed,
                    "skipped_completed_patients": skipped,
                    "failed_patient_id": active_patient,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "job_identity": dict(job_identity),
                },
            )
            raise

        status = {
            "status": "COMPLETE" if execution_mode == "formal" else "SMOKE_COMPLETE",
            "execution_mode": execution_mode,
            "formal_claim_eligible": False,
            "formal_summary_eligible": execution_mode == "formal",
            "patient_count": len(patients),
            "completed_patients": completed,
            "skipped_completed_patients": skipped,
            "node_count": len(NODE_NAMES),
            "condition_names": list(CONDITION_NAMES),
            "dose_count": len(FORMAL_DOSES),
            "job_identity": dict(job_identity),
        }
        _write_json_atomic(status_path, status)
        return status


class _SyntheticProcessNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for name in NODE_NAMES:
            layer = nn.Conv2d(3, 3, kernel_size=1, bias=False)
            with torch.no_grad():
                layer.weight.copy_(torch.eye(3).reshape(3, 3, 1, 1))
            setattr(self, name, layer)
        self.head = nn.Conv2d(3, 2, kernel_size=1, bias=False)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.weight[0, 0, 0, 0] = 1.0
            self.head.weight[1, 1, 0, 0] = 1.0

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        state = image
        for name in NODE_NAMES:
            state = getattr(self, name)(state)
        return self.head(state)


def _synthetic_adapter() -> HookedModelAdapter:
    model = _SyntheticProcessNet()
    return HookedModelAdapter(
        model,
        OrderedDict((name, getattr(model, name)) for name in NODE_NAMES),
    ).eval()


def _synthetic_observers() -> dict[str, dict[int, LinearObserver]]:
    output: dict[str, dict[int, LinearObserver]] = {}
    for node in NODE_NAMES:
        output[node] = {}
        for seed in (17, 29, 43):
            observer = LinearObserver(3, 2)
            with torch.no_grad():
                observer.projection.weight.zero_()
                observer.projection.weight[0, 0, 0, 0] = 1.0
                observer.projection.weight[1, 1, 0, 0] = 1.0
                observer.projection.bias.zero_()
            output[node][seed] = observer.eval()
    return output


def _state_count_rows(
    trace: CounterfactualTrace,
    *,
    output_indices: np.ndarray,
    source_states: np.ndarray | None = None,
    truth_classes: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    indices = np.asarray(output_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0 or len(np.unique(indices)) != indices.size:
        raise ValueError("output_indices must be a nonempty unique vector")
    if source_states is None and truth_classes is None:
        groups = ((-1, -1, np.arange(indices.size, dtype=np.int64)),)
    else:
        states = np.asarray(source_states)
        classes = np.asarray(truth_classes)
        if (
            states.shape != indices.shape
            or classes.shape != indices.shape
            or not np.issubdtype(states.dtype, np.integer)
            or not np.issubdtype(classes.dtype, np.integer)
            or np.any((states < 0) | (states >= 5))
            or np.any((classes < 0) | (classes >= 4))
        ):
            raise ValueError("source_states and truth_classes must align with output_indices")
        groups = tuple(
            (
                int(source_state),
                int(truth_class),
                np.flatnonzero((states == source_state) & (classes == truth_class)),
            )
            for source_state, truth_class in sorted(
                set(zip(states.astype(int), classes.astype(int), strict=True))
            )
        )
    for dose in trace.doses:
        for downstream_depth, (depth, state_map) in enumerate(
            trace.downstream_states[dose].items()
        ):
            flattened = np.asarray(state_map).reshape(-1)
            reliable = np.asarray(trace.downstream_reliable[dose][depth]).reshape(-1)
            for source_state, truth_class, positions in groups:
                group_indices = indices[positions]
                selected = group_indices[reliable[group_indices]]
                counts = (
                    np.bincount(flattened[selected], minlength=5)
                    if selected.size
                    else np.zeros(5, dtype=np.int64)
                )
                rows.append(
                    {
                        "dose": float(dose),
                        "depth": depth,
                        "downstream_depth": downstream_depth,
                        "source_state": source_state,
                        "truth_class": truth_class,
                        "selected_pixel_count": int(group_indices.size),
                        "reliable_pixel_count": int(selected.size),
                        "state_counts": counts.astype(int).tolist(),
                    }
                )
    return rows


def _condition_payload(
    trace: CounterfactualTrace,
    *,
    output_indices: np.ndarray,
    condition: str,
    source_states: np.ndarray | None = None,
    truth_classes: np.ndarray | None = None,
) -> dict[str, Any]:
    return {
        "condition": condition,
        "doses": list(trace.doses),
        "downstream_path": list(trace.audits["downstream_path"]),
        "state_counts": _state_count_rows(
            trace,
            output_indices=output_indices,
            source_states=source_states,
            truth_classes=truth_classes,
        ),
        "audits": dict(trace.audits),
    }


def _restart_logits_at_indices(
    activation: torch.Tensor,
    observers: Mapping[int, LinearObserver],
    *,
    output_shape: tuple[int, int],
    output_indices: Sequence[int],
) -> torch.Tensor:
    indices = torch.as_tensor(
        tuple(int(value) for value in output_indices),
        dtype=torch.int64,
        device=activation.device,
    )
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("observer target indices must be nonempty")
    runs = torch.stack(
        tuple(
            observers[seed](activation, output_size=output_shape)[0]
            for seed in EXPECTED_OBSERVER_SEEDS
        ),
        dim=0,
    )
    return runs.permute(0, 2, 3, 1).reshape(
        len(EXPECTED_OBSERVER_SEEDS),
        -1,
        int(runs.shape[1]),
    )[:, indices]


def _stacked_observer_weight(
    observers: Mapping[int, LinearObserver],
) -> torch.Tensor:
    return stack_observer_contrast_weights(
        tuple(
            observers[seed]
            .projection.weight[:, :, 0, 0]
            .detach()
            .cpu()
            .to(torch.float64)
            for seed in EXPECTED_OBSERVER_SEEDS
        )
    )


def _canonical_states_at_targets(
    restart_logits: torch.Tensor,
    truth_classes: np.ndarray,
) -> np.ndarray:
    logits = restart_logits.detach().cpu().to(torch.float64)
    predictions = torch.softmax(logits, dim=2).mean(dim=0).argmax(dim=1).numpy()
    return relationship_states(
        np.asarray(truth_classes, dtype=np.uint8),
        predictions.astype(np.uint8, copy=False),
        num_classes=int(logits.shape[2]),
    )


def _empty_condition_payload(node_index: int, condition: str) -> dict[str, Any]:
    return {
        "condition": condition,
        "doses": list(FORMAL_DOSES),
        "downstream_path": [*NODE_NAMES[node_index:], "Y"],
        "state_counts": [],
        "audits": {"status": "NO_MATCH"},
    }


class FormalPatientProcessor:
    """Execute one locked patient plan without persisting feature tensors."""

    def __init__(
        self,
        *,
        runtime: FormalRuntime,
        patient_plan_root: Path,
        patient_plan_hashes: Mapping[str, str],
        records_by_slice: Mapping[str, SliceRecord],
        configuration: Mapping[str, Any],
        norm_calibration: Mapping[str, Any],
        device: torch.device,
        dose_batch_size: int,
        split: str = "test",
    ) -> None:
        self.runtime = runtime
        self.patient_plan_root = Path(patient_plan_root)
        self.patient_plan_hashes = dict(patient_plan_hashes)
        self.records_by_slice = dict(records_by_slice)
        self.configuration = configuration
        self.norm_calibration = norm_calibration
        self.device = device
        self.dose_batch_size = int(dose_batch_size)
        self.split = str(split)
        if self.dose_batch_size <= 0:
            raise ValueError("dose_batch_size must be positive")
        if not self.split:
            raise ValueError("split must be non-empty")

    def _patient_plan(self, patient_id: str) -> dict[str, Any]:
        path = self.patient_plan_root / f"{patient_id}.json"
        if not path.is_file() or sha256_file(path) != self.patient_plan_hashes.get(patient_id):
            raise ValueError(f"active patient plan differs from the locked manifest: {patient_id}")
        payload = _load_json(path)
        validate_patient_plan(
            payload,
            model=self.runtime.job.model,
            model_seed=self.runtime.job.seed,
            patient_id=patient_id,
            split=self.split,
        )
        return payload

    def _record(self, slice_id: str) -> SliceRecord:
        try:
            return self.records_by_slice[str(slice_id)]
        except KeyError as error:
            raise ValueError(f"locked plan references an unknown test slice: {slice_id}") from error

    def _source_logit_bank(
        self,
        patient_plan: Mapping[str, Any],
    ) -> dict[tuple[str, str, str, int], torch.Tensor]:
        requests: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for node_plan in patient_plan["nodes"]:
            node = str(node_plan["node"])
            for pair in node_plan["pairs"]:
                requests[str(pair["source_slice_id"])][node].append(pair)
        bank: dict[tuple[str, str, str, int], torch.Tensor] = {}
        with torch.inference_mode():
            for source_slice_id in sorted(requests):
                record = self._record(source_slice_id)
                source_patient = patient_id_from_slice(source_slice_id)
                image, truth = _load_slice(record)
                trace = self.runtime.adapter.trace(image.to(self.device))
                for node in sorted(requests[source_slice_id], key=NODE_NAMES.index):
                    pairs = requests[source_slice_id][node]
                    indices = [int(pair["source_output_index"]) for pair in pairs]
                    logits = _restart_logits_at_indices(
                        trace.activations[node],
                        self.runtime.observers[node],
                        output_shape=tuple(int(value) for value in truth.shape),
                        output_indices=indices,
                    ).detach().cpu().to(torch.float64)
                    truth_classes = truth.reshape(-1)[np.asarray(indices, dtype=np.int64)]
                    expected_classes = np.asarray(
                        [pair["truth_class"] for pair in pairs],
                        dtype=np.uint8,
                    )
                    if not np.array_equal(truth_classes, expected_classes):
                        raise ValueError(f"source truth class differs from the plan: {source_slice_id}/{node}")
                    states = _canonical_states_at_targets(logits, truth_classes)
                    expected_states = np.asarray(
                        [pair["target_state"] for pair in pairs],
                        dtype=np.uint8,
                    )
                    if not np.array_equal(states, expected_states):
                        raise ValueError(f"source state differs from the plan: {source_slice_id}/{node}")
                    for position, pair in enumerate(pairs):
                        registered_patient = str(pair["source_patient_id"])
                        if registered_patient != source_patient:
                            raise ValueError(f"source patient differs from its slice identifier: {source_slice_id}")
                        key = (
                            node,
                            registered_patient,
                            source_slice_id,
                            int(pair["source_output_index"]),
                        )
                        if key in bank:
                            raise ValueError(f"source pixel is duplicated inside a patient plan: {key}")
                        bank[key] = logits[:, position]
                del trace
        return bank

    @staticmethod
    def _empty_node_result(node: str) -> dict[str, Any]:
        node_index = NODE_NAMES.index(node)
        return {
            "node": node,
            "status": "NO_MATCH",
            "conditions": {
                condition: _empty_condition_payload(node_index, condition)
                for condition in CONDITION_NAMES
            },
            "effect_summary": {
                "pair_count": 0,
                "evaluable": False,
                "failure_reasons": ["NO_MATCH"],
            },
        }

    def _node_result(
        self,
        node_plan: Mapping[str, Any],
        *,
        patient_id: str,
        image: torch.Tensor,
        truth: np.ndarray,
        clean_trace: ForwardTrace,
        source_bank: Mapping[tuple[str, str, str, int], torch.Tensor],
    ) -> dict[str, Any]:
        node = str(node_plan["node"])
        node_index = NODE_NAMES.index(node)
        pairs = list(node_plan["pairs"])
        if not pairs:
            return self._empty_node_result(node)

        output_indices = np.asarray(
            [pair["base_output_index"] for pair in pairs],
            dtype=np.int64,
        )
        source_states = np.asarray(
            [pair["target_state"] for pair in pairs],
            dtype=np.uint8,
        )
        truth_classes = np.asarray(
            [pair["truth_class"] for pair in pairs],
            dtype=np.uint8,
        )
        actual_truth = truth.reshape(-1)[output_indices]
        if not np.array_equal(actual_truth, truth_classes):
            raise ValueError(f"base truth class differs from the plan: {node}")
        activation = clean_trace.activations[node]
        native_shape = tuple(int(value) for value in activation.shape[-2:])
        if list(native_shape) != list(node_plan["native_shape"]):
            raise ValueError(f"base activation geometry differs from the plan: {node}")
        base_logits = _restart_logits_at_indices(
            activation,
            self.runtime.observers[node],
            output_shape=tuple(int(value) for value in truth.shape),
            output_indices=output_indices,
        ).detach().cpu().to(torch.float64)
        base_states = _canonical_states_at_targets(base_logits, truth_classes)
        expected_base_states = np.asarray(
            [pair["base_state"] for pair in pairs],
            dtype=np.uint8,
        )
        if not np.array_equal(base_states, expected_base_states):
            raise ValueError(f"base state differs from the plan: {node}")

        source_logits = torch.stack(
            tuple(
                source_bank[
                    (
                        node,
                        str(pair["source_patient_id"]),
                        str(pair["source_slice_id"]),
                        int(pair["source_output_index"]),
                    )
                ]
                for pair in pairs
            ),
            dim=1,
        )
        stacked_weight = _stacked_observer_weight(self.runtime.observers[node])
        stacked_delta = stack_restart_logit_contrasts(source_logits, base_logits)
        resize_rows = bilinear_resize_rows(
            native_shape,
            tuple(int(value) for value in truth.shape),
            torch.from_numpy(output_indices),
            dtype=torch.float64,
        )
        rcond = float(self.configuration["intervention"]["pseudoinverse_rcond"])
        exchange = minimum_norm_state_exchange(
            resize_rows,
            stacked_weight,
            stacked_delta,
            rcond=rcond,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            _stable_seed(
                self.configuration["high_level_process"]["fold_seed"],
                self.runtime.job.model,
                self.runtime.job.seed,
                patient_id_from_slice(str(node_plan["base_slice_id"])),
                node,
                "nullspace",
            )
        )
        null_edit = equal_norm_nullspace_control(
            stacked_weight,
            exchange.delta_h,
            generator=generator,
            rcond=rcond,
        )

        projected = project_feature_edit(
            exchange.delta_h,
            native_shape=native_shape,
            observer_weight=stacked_weight,
            output_shape=tuple(int(value) for value in truth.shape),
        )
        projected_magnitude = torch.linalg.vector_norm(projected, dim=0).numpy()
        target_mask = np.zeros(truth.shape, dtype=bool)
        target_mask.reshape(-1)[output_indices] = True
        target_mean = float(projected_magnitude[target_mask].mean())
        outside_q99 = float(np.quantile(projected_magnitude[~target_mask], 0.99))
        leakage_ratio = outside_q99 / max(target_mean, np.finfo(float).eps)
        edit_norm = exchange.frobenius_norm / np.sqrt(len(pairs))
        limit_key = f"{self.runtime.job.model}/{node}"
        limits = self.norm_calibration.get("thresholds", {}).get(limit_key)
        if not isinstance(limits, dict):
            raise ValueError(f"validation norm/leakage limit is missing: {limit_key}")
        norm_pass = edit_norm <= float(limits["maximum_edit_norm_per_sqrt_pixel"])
        leakage_pass = leakage_ratio <= float(
            limits["maximum_leakage_ratio_q99_to_target_mean"]
        )
        null_target_max_abs_logit_change = float(
            torch.max(torch.abs(resize_rows @ null_edit @ stacked_weight.T)).item()
        )

        print(
            f"v11-progress model={self.runtime.job.model} "
            f"seed={self.runtime.job.seed} patient={patient_id} "
            f"node={node} condition=task status=RUNNING",
            flush=True,
        )
        task = run_state_exchange(
            self.runtime.adapter,
            image,
            node=node,
            delta_h=exchange.delta_h,
            doses=FORMAL_DOSES,
            observers=self.runtime.observers,
            truth=truth,
            reliability_threshold=self.runtime.reliability_threshold,
            dose_batch_size=self.dose_batch_size,
            clean_trace=clean_trace,
        )
        print(
            f"v11-progress model={self.runtime.job.model} "
            f"seed={self.runtime.job.seed} patient={patient_id} "
            f"node={node} condition=task status=COMPLETE",
            flush=True,
        )
        print(
            f"v11-progress model={self.runtime.job.model} "
            f"seed={self.runtime.job.seed} patient={patient_id} "
            f"node={node} condition=null status=RUNNING",
            flush=True,
        )
        null = run_state_exchange(
            self.runtime.adapter,
            image,
            node=node,
            delta_h=null_edit,
            doses=FORMAL_DOSES,
            observers=self.runtime.observers,
            truth=truth,
            reliability_threshold=self.runtime.reliability_threshold,
            dose_batch_size=self.dose_batch_size,
            clean_trace=clean_trace,
        )
        print(
            f"v11-progress model={self.runtime.job.model} "
            f"seed={self.runtime.job.seed} patient={patient_id} "
            f"node={node} condition=null status=COMPLETE",
            flush=True,
        )
        realized = np.asarray(task.downstream_states[1.0][node]).reshape(-1)[
            output_indices
        ]
        realization_rate = float(np.mean(realized == source_states))
        restore_error = max(
            float(task.audits["dose_zero_max_abs_logit_error"]),
            float(null.audits["dose_zero_max_abs_logit_error"]),
        )
        intervention = self.configuration["intervention"]
        failures = []
        if exchange.target_max_abs_error > float(intervention["maximum_reconstruction_error"]):
            failures.append("RECONSTRUCTION_ERROR")
        if realization_rate < float(intervention["minimum_state_realization"]):
            failures.append("STATE_REALIZATION")
        if restore_error > float(intervention["maximum_restore_error"]):
            failures.append("RESTORE_ERROR")
        if not norm_pass or not leakage_pass:
            failures.append("INTERVENTION_OOD")
        if null_target_max_abs_logit_change > float(intervention["maximum_reconstruction_error"]):
            failures.append("NULLSPACE_READOUT_CHANGED")

        return {
            "node": node,
            "status": "PASS" if not failures else "FAILED_OPERATOR_OR_OOD_AUDIT",
            "conditions": {
                "task": _condition_payload(
                    task,
                    output_indices=output_indices,
                    source_states=source_states,
                    truth_classes=truth_classes,
                    condition="task",
                ),
                "null": _condition_payload(
                    null,
                    output_indices=output_indices,
                    source_states=source_states,
                    truth_classes=truth_classes,
                    condition="null",
                ),
            },
            "effect_summary": {
                "pair_count": len(pairs),
                "evaluable": not failures,
                "failure_reasons": failures,
                "operator_reconstruction_max_abs_error": exchange.target_max_abs_error,
                "target_state_realization_rate": realization_rate,
                "restore_max_abs_logit_error": restore_error,
                "task_edit_frobenius_norm": exchange.frobenius_norm,
                "task_edit_norm_per_sqrt_pixel": edit_norm,
                "null_edit_frobenius_norm": float(
                    torch.linalg.vector_norm(null_edit).item()
                ),
                "null_target_max_abs_logit_change": null_target_max_abs_logit_change,
                "leakage_ratio_q99_to_target_mean": leakage_ratio,
                "validation_maximum_edit_norm_per_sqrt_pixel": float(
                    limits["maximum_edit_norm_per_sqrt_pixel"]
                ),
                "validation_maximum_leakage_ratio_q99_to_target_mean": float(
                    limits["maximum_leakage_ratio_q99_to_target_mean"]
                ),
                "norm_pass": norm_pass,
                "leakage_pass": leakage_pass,
                "joint_observer_rank": exchange.effective_rank_channel,
                "joint_observer_nullity": int(
                    stacked_weight.shape[1] - exchange.effective_rank_channel
                ),
                "spatial_rank": exchange.effective_rank_spatial,
                "base_state_verification": True,
                "source_state_verification": True,
            },
        }

    def __call__(self, patient_id: str) -> dict[str, Any]:
        print(
            f"v11-progress model={self.runtime.job.model} "
            f"seed={self.runtime.job.seed} patient={patient_id} "
            "node=- condition=- status=STARTED",
            flush=True,
        )
        patient_plan = self._patient_plan(patient_id)
        source_bank = self._source_logit_bank(patient_plan)
        matched_by_slice: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        results_by_node: dict[str, dict[str, Any]] = {}
        for node_plan in patient_plan["nodes"]:
            node = str(node_plan["node"])
            if node_plan["pairs"]:
                matched_by_slice[str(node_plan["base_slice_id"])].append(node_plan)
            else:
                results_by_node[node] = self._empty_node_result(node)
        with torch.inference_mode():
            for base_slice_id in sorted(matched_by_slice):
                if patient_id_from_slice(base_slice_id) != patient_id:
                    raise ValueError(f"base slice does not belong to its patient: {base_slice_id}")
                image, truth = _load_slice(self._record(base_slice_id))
                image_device = image.to(self.device)
                clean_trace = self.runtime.adapter.trace(image_device)
                for node_plan in sorted(
                    matched_by_slice[base_slice_id],
                    key=lambda value: NODE_NAMES.index(str(value["node"])),
                ):
                    node = str(node_plan["node"])
                    results_by_node[node] = self._node_result(
                        node_plan,
                        patient_id=patient_id,
                        image=image_device,
                        truth=truth,
                        clean_trace=clean_trace,
                        source_bank=source_bank,
                    )
                del clean_trace
        return {
            "schema_version": 1,
            "execution_mode": "formal",
            "formal_claim_eligible": False,
            "formal_summary_eligible": True,
            "nodes": list(NODE_NAMES),
            "node_results": [results_by_node[node] for node in NODE_NAMES],
            "patient_plan_sha256": self.patient_plan_hashes[patient_id],
            "storage_audit": {
                "full_feature_tensors_persisted": False,
                "full_output_logits_persisted": False,
                "only_state_counts_and_audits_persisted": True,
            },
        }


def _synthetic_patient_payload(patient_id: str) -> dict[str, Any]:
    if patient_id != "synthetic-patient-000":
        raise ValueError(f"unknown synthetic patient: {patient_id}")
    adapter = _synthetic_adapter()
    observers = _synthetic_observers()
    image = torch.tensor(
        [
            [
                [[2.0, -1.0], [3.0, -2.0]],
                [[-2.0, 1.0], [-3.0, 2.0]],
                [[0.5, -0.5], [0.25, -0.25]],
            ]
        ],
        dtype=torch.float32,
    )
    truth = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    target_indices = torch.tensor([0], dtype=torch.int64)
    observer_weight = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    resize_rows = bilinear_resize_rows(
        (2, 2),
        (2, 2),
        target_indices,
    )
    exchange = minimum_norm_state_exchange(
        resize_rows,
        observer_weight,
        torch.tensor([[-4.0, 4.0]], dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(20260720)
    null_edit = equal_norm_nullspace_control(
        observer_weight,
        exchange.delta_h,
        generator=generator,
    )
    node_results: list[dict[str, Any]] = []
    for node in NODE_NAMES:
        task = run_state_exchange(
            adapter,
            image,
            node=node,
            delta_h=exchange.delta_h,
            doses=FORMAL_DOSES,
            observers=observers,
            truth=truth,
            reliability_threshold=0.0,
        )
        null = run_state_exchange(
            adapter,
            image,
            node=node,
            delta_h=null_edit,
            doses=FORMAL_DOSES,
            observers=observers,
            truth=truth,
            reliability_threshold=0.0,
        )
        selected = target_indices.numpy()
        node_results.append(
            {
                "node": node,
                "conditions": {
                    "task": _condition_payload(
                        task,
                        output_indices=selected,
                        condition="task",
                    ),
                    "null": _condition_payload(
                        null,
                        output_indices=selected,
                        condition="null",
                    ),
                },
                "effect_summary": {
                    "target_state": 2,
                    "operator_reconstruction_max_abs_error": exchange.target_max_abs_error,
                    "task_edit_frobenius_norm": exchange.frobenius_norm,
                    "null_edit_frobenius_norm": float(torch.linalg.vector_norm(null_edit).item()),
                },
            }
        )
    return {
        "schema_version": 1,
        "execution_mode": "smoke",
        "formal_claim_eligible": False,
        "nodes": list(NODE_NAMES),
        "node_results": node_results,
        "storage_audit": {
            "full_feature_tensors_persisted": False,
            "full_output_logits_persisted": False,
        },
    }


def run_v11_smoke(
    output_root: str | Path,
    *,
    patient_limit: int = 1,
) -> dict[str, Any]:
    validate_execution_request(
        execution_mode="smoke",
        patient_limit=patient_limit,
        protocol_lock=Path("unused-in-smoke"),
    )
    if int(patient_limit) != 1:
        raise ValueError("the deterministic V11 smoke fixture contains exactly one patient")
    identity = {
        "model": "synthetic",
        "model_seed": 0,
        "protocol": "v11-smoke-not-formal",
        "nodes": list(NODE_NAMES),
        "doses": list(FORMAL_DOSES),
        "conditions": list(CONDITION_NAMES),
    }
    return run_patient_registry(
        Path(output_root) / "synthetic" / "seed_0",
        patient_ids=("synthetic-patient-000",),
        job_identity=identity,
        process_patient=_synthetic_patient_payload,
        resume=True,
        execution_mode="smoke",
    )


def run_v11_formal(
    *,
    workspace: Path,
    config_path: Path,
    protocol_lock_path: Path,
    asset_root: Path,
    model: str,
    model_seed: int,
    device: torch.device,
    dose_batch_size: int,
    resume: bool,
    output_root: Path | None = None,
) -> dict[str, Any]:
    configuration = _load_yaml(config_path)
    validated = validate_v11_configuration(configuration)
    if model not in validated.main_models + validated.control_models:
        raise ValueError(f"model is outside the V11 matrix: {model}")
    if int(model_seed) not in validated.model_seeds:
        raise ValueError(f"model seed is outside the V11 matrix: {model_seed}")
    raw_lock = _load_json(protocol_lock_path)
    data_config, patient_ids = _validate_data_inventory(
        workspace,
        asset_root,
        raw_lock,
    )
    process_hashes, calibration_hashes = _active_validation_hashes(
        workspace,
        configuration,
    )
    lock = load_and_validate_v11_protocol_lock(
        protocol_lock_path,
        active_configuration_sha256=sha256_file(config_path),
        active_source_tree_sha256=source_tree_sha256(workspace),
        test_patient_ids=patient_ids,
        process_hashes=process_hashes,
        model=model,
        model_seed=int(model_seed),
    )
    if lock.get("calibration_hashes") != calibration_hashes:
        raise ValueError("active validation calibrations differ from the V11 lock")
    history = _load_json(
        workspace
        / str(configuration["validation_fit_root"])
        / "history_admission.json"
    )
    if history.get("status") != "PASS":
        raise ValueError("the locked first-order high-level process admission is not PASS")

    plan_manifest_path = (
        workspace
        / str(configuration["intervention_plan_root"])
        / model
        / f"seed_{int(model_seed)}"
        / "plan_manifest.json"
    )
    plan_manifest, plan_root = validate_locked_plan_manifest(
        plan_manifest_path,
        lock=lock,
        model=model,
        model_seed=int(model_seed),
    )
    matrix = load_model_matrix(
        workspace / str(configuration["model_matrix"]),
        workspace_root=workspace,
        asset_root_override=asset_root,
    )
    job = _job_for(matrix.jobs, model=model, model_seed=int(model_seed))
    runtime = _load_formal_runtime(
        job,
        workspace=workspace,
        configuration=configuration,
        device=device,
    )
    validate_runtime_against_lock(runtime, lock)
    if plan_manifest.get("checkpoint_sha256") != runtime.checkpoint_sha256:
        raise ValueError("intervention plan was prepared with a different checkpoint")

    split_config = data_config["splits"][str(configuration["formal_split"])]
    records = discover_slice_records(
        asset_root / str(split_config["image_dir"]),
        asset_root / str(split_config["mask_dir"]),
    )
    grouped = _group_records(records)
    if sorted(grouped) != patient_ids:
        raise ValueError("discovered test slices differ from the locked patient registry")
    records_by_slice = {record.slice_id: record for record in records}
    if len(records_by_slice) != len(records):
        raise ValueError("test slice identifiers are not unique")
    norm_calibration = _load_json(
        workspace
        / str(configuration["validation_fit_root"])
        / "norm_and_leakage_calibration.json"
    )
    if norm_calibration.get("status") != "PASS":
        raise ValueError("norm and leakage calibration is not PASS")
    processor = FormalPatientProcessor(
        runtime=runtime,
        patient_plan_root=plan_root,
        patient_plan_hashes=plan_manifest["patient_plan_hashes"],
        records_by_slice=records_by_slice,
        configuration=configuration,
        norm_calibration=norm_calibration,
        device=device,
        dose_batch_size=int(dose_batch_size),
    )
    registered_output_root = workspace / str(configuration["formal_job_root"])
    if output_root is not None and output_root.resolve() != registered_output_root.resolve():
        raise ValueError("formal output_root may not differ from the locked configuration")
    job_root = registered_output_root / model / f"seed_{int(model_seed)}"
    identity = {
        "model": model,
        "model_seed": int(model_seed),
        "protocol_lock_sha256": sha256_file(protocol_lock_path),
        "configuration_sha256": sha256_file(config_path),
        "source_tree_sha256": source_tree_sha256(workspace),
        "checkpoint_sha256": runtime.checkpoint_sha256,
        "observer_registry_sha256": sha256_named_values(runtime.observer_hashes),
        "plan_manifest_sha256": sha256_file(plan_manifest_path),
        "process_hashes": process_hashes,
        "calibration_hashes": calibration_hashes,
        "nodes": list(NODE_NAMES),
        "doses": list(FORMAL_DOSES),
        "conditions": list(CONDITION_NAMES),
    }
    try:
        return run_patient_registry(
            job_root,
            patient_ids=patient_ids,
            job_identity=identity,
            process_patient=processor,
            resume=bool(resume),
            execution_mode="formal",
        )
    finally:
        runtime.adapter.to("cpu")
        for observer_group in runtime.observers.values():
            for observer in observer_group.values():
                observer.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run resumable V11 full-path causal-abstraction interventions."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/v11_causal_abstraction.yaml"),
    )
    parser.add_argument("--protocol-lock", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--model", default="synthetic")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument(
        "--execution-mode",
        choices=("smoke", "formal"),
        required=True,
    )
    parser.add_argument("--patient-limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dose-batch-size", type=int, default=5)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    workspace = args.workspace_root.resolve()
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (workspace / args.config).resolve()
    )
    lock_path = (
        args.protocol_lock.resolve()
        if args.protocol_lock is not None and args.protocol_lock.is_absolute()
        else (
            (workspace / args.protocol_lock).resolve()
            if args.protocol_lock is not None
            else workspace / "missing-v11-protocol-lock.json"
        )
    )
    validate_execution_request(
        execution_mode=args.execution_mode,
        patient_limit=args.patient_limit,
        protocol_lock=lock_path,
    )
    if args.execution_mode == "smoke":
        output = args.output_root or Path("results/v11_causal_abstraction_smoke")
        status = run_v11_smoke(output, patient_limit=int(args.patient_limit))
        print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.asset_root is None:
        raise ValueError("formal mode requires --asset-root")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    formal_output = None
    if args.output_root is not None:
        formal_output = (
            args.output_root.resolve()
            if args.output_root.is_absolute()
            else (workspace / args.output_root).resolve()
        )
    status = run_v11_formal(
        workspace=workspace,
        config_path=config_path,
        protocol_lock_path=lock_path,
        asset_root=args.asset_root.resolve(),
        model=str(args.model),
        model_seed=int(args.model_seed),
        device=device,
        dose_batch_size=int(args.dose_batch_size),
        resume=bool(args.resume),
        output_root=formal_output,
    )
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
