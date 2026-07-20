from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import yaml

from pptt.causal_abstraction.kernels import (
    cross_validated_history_comparison,
    estimate_patient_equal_process,
    evaluate_history_dependence,
    pool_architecture_processes,
)
from pptt.causal_abstraction.trace_process import collect_case_trace_rows


EXPECTED_NAME = "v11_cross_architecture_causal_abstraction"
EXPECTED_DATASET = "brats2023_2d"
EXPECTED_MAIN_MODELS = ("unet_baseline", "transunet_r50_vit_b16")
EXPECTED_CONTROL_MODELS = ("unet_noskip",)
EXPECTED_MODEL_SEEDS = (42, 123, 3407)
EXPECTED_OBSERVER_SEEDS = (17, 29, 43)
EXPECTED_NODES = (
    "down1",
    "down2",
    "down3",
    "down4",
    "up1",
    "up2",
    "up3",
    "up4",
)
EXPECTED_RELATIONSHIP_STATES = ("BC", "FC", "FP", "FN", "FW")
EXPECTED_TRUTH_CLASSES = (0, 1, 2, 3)
EXPECTED_DOSES = (0.0, 0.25, 0.5, 0.75, 1.0)
EXPECTED_TRANSFER_DIRECTIONS = (
    "unet_baseline->transunet_r50_vit_b16",
    "transunet_r50_vit_b16->unet_baseline",
)
EXPECTED_PROCESS_NAMES = ("H_U", "H_T", "H_shared")
EXPECTED_CALIBRATION_NAMES = (
    "matching",
    "norm_and_leakage",
    "history_admission",
)
EXPECTED_MODEL_COUNT = 9
EXPECTED_OBSERVER_COUNT = 216


@dataclass(frozen=True)
class V11Configuration:
    dataset: str
    main_models: tuple[str, ...]
    control_models: tuple[str, ...]
    model_seeds: tuple[int, ...]
    observer_seeds: tuple[int, ...]
    nodes: tuple[str, ...]
    relationship_states: tuple[str, ...]
    truth_classes: tuple[int, ...]
    doses: tuple[float, ...]
    transfer_directions: tuple[str, ...]
    validation_patient_count: int
    formal_patient_count: int


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
    if hasattr(value, "item"):
        try:
            return _json_ready(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, Path):
        return str(value)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_named_values(values: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        encoded_name = str(name).encode("utf-8")
        encoded_value = str(value).encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(encoded_value).to_bytes(8, "big"))
        digest.update(encoded_value)
    return digest.hexdigest()


def _require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise ValueError(f"registered {field} changed: {actual!r} != {expected!r}")


def _require_sha256(value: Any, field: str) -> str:
    encoded = str(value)
    if len(encoded) != 64 or any(character not in "0123456789abcdef" for character in encoded.lower()):
        raise ValueError(f"{field} must be a 64-character SHA-256 digest")
    return encoded.lower()


def validate_v11_configuration(config: Mapping[str, Any]) -> V11Configuration:
    _require_equal(config.get("name"), EXPECTED_NAME, "name")
    _require_equal(config.get("dataset"), EXPECTED_DATASET, "dataset")
    _require_equal(config.get("fit_split"), "val", "fit_split")
    _require_equal(config.get("formal_split"), "test", "formal_split")
    _require_equal(tuple(config.get("main_models", ())), EXPECTED_MAIN_MODELS, "main_models")
    _require_equal(tuple(config.get("control_models", ())), EXPECTED_CONTROL_MODELS, "control_models")
    _require_equal(tuple(config.get("model_seeds", ())), EXPECTED_MODEL_SEEDS, "model_seeds")
    _require_equal(tuple(config.get("observer_seeds", ())), EXPECTED_OBSERVER_SEEDS, "observer_seeds")
    _require_equal(tuple(config.get("nodes", ())), EXPECTED_NODES, "nodes")
    _require_equal(
        tuple(config.get("relationship_states", ())),
        EXPECTED_RELATIONSHIP_STATES,
        "relationship_states",
    )
    _require_equal(tuple(config.get("truth_classes", ())), EXPECTED_TRUTH_CLASSES, "truth_classes")
    _require_equal(tuple(float(value) for value in config.get("doses", ())), EXPECTED_DOSES, "doses")
    _require_equal(
        tuple(config.get("transfer_directions", ())),
        EXPECTED_TRANSFER_DIRECTIONS,
        "transfer_directions",
    )
    _require_equal(config.get("model_matrix"), "manifests/model_matrix.yaml", "model_matrix")
    _require_equal(config.get("observer_root"), "results/v1_observers", "observer_root")
    _require_equal(
        config.get("validation_fit_root"),
        "results/v11_causal_abstraction/validation_fit",
        "validation_fit_root",
    )
    _require_equal(
        config.get("formal_job_root"),
        "results/v11_causal_abstraction/formal_jobs",
        "formal_job_root",
    )
    _require_equal(config.get("output_root"), "results/v11_causal_abstraction", "output_root")
    _require_equal(
        config.get("protocol_lock"),
        "results/v11_causal_abstraction/v11_protocol_lock.json",
        "protocol_lock",
    )
    _require_equal(
        config.get("intervention_plan_root"),
        "results/v11_causal_abstraction/intervention_plans",
        "intervention_plan_root",
    )
    _require_equal(
        config.get("calibration_manifest"),
        "results/v11_causal_abstraction/validation_fit/calibration_manifest.json",
        "calibration_manifest",
    )
    _require_equal(
        config.get("clean_trace_templates"),
        {
            "validation": {
                "unet_baseline": "results/v3_unet_pair/{model}/seed_{seed}/val/case_traces",
                "unet_noskip": "results/v3_unet_pair/{model}/seed_{seed}/val/case_traces",
                "transunet_r50_vit_b16": "results/v8_transunet_mechanism/validation_workers/seed_{seed}/{model}/seed_{seed}/val/case_traces",
            },
            "formal": {
                "unet_baseline": "results/v3_unet_pair/{model}/seed_{seed}/test/case_traces",
                "unet_noskip": "results/v3_unet_pair/{model}/seed_{seed}/test/case_traces",
                "transunet_r50_vit_b16": "results/v5_transunet/{model}/seed_{seed}/test/case_traces",
            },
        },
        "clean_trace_templates",
    )
    _require_equal(
        config.get("high_level_process"),
        {
            "state_count": 5,
            "dirichlet_alpha": 0.5,
            "patient_folds": 5,
            "fold_seed": 20260720,
            "history_tv_tolerance": 0.03,
        },
        "high_level_process",
    )
    _require_equal(
        config.get("matching"),
        {
            "same_patient_first": True,
            "same_truth_class": True,
            "without_replacement": True,
            "max_pixels_per_patient_node_state": 128,
            "boundary_edges": [1.5, 3.5, 7.5],
            "feature_norm_quantile_bins": 4,
            "validation_locked_feature_norm_edges": True,
            "selected_slice_rule": "maximum_matched_state_coverage_then_stable_id",
            "source_candidates_per_patient_node_state_class": 8,
            "maximum_target_pixels_per_node_patient": 32,
        },
        "matching",
    )
    _require_equal(
        config.get("intervention"),
        {
            "operator": "restart_ensemble_output_constrained_minimum_norm_state_exchange",
            "observer_constraint": "all_registered_restarts",
            "pseudoinverse_rcond": 1.0e-7,
            "null_control": "equal_norm_observer_nullspace",
            "maximum_reconstruction_error": 1.0e-5,
            "minimum_state_realization": 0.95,
            "maximum_restore_error": 1.0e-6,
            "validation_locked_norm_and_leakage_limits": True,
        },
        "intervention",
    )
    _require_equal(
        config.get("controls"),
        {
            "depth_permutation": [7, 6, 5, 4, 3, 2, 1, 0],
            "state_permutation": [1, 0, 3, 4, 2],
            "minimum_state_permutation_advantage": 0.10,
            "minimum_state_permutation_ci_low": 0.05,
            "paired_test": "one_sided_wilcoxon_signed_rank",
            "holm_family": "depth_order_all_main_model_seed_node_cells",
            "observer_randomization_source": "v1_frozen_observer_admission",
        },
        "controls",
    )
    expected_coverage = {
        "expected_validation_patients": 125,
        "expected_formal_patients": 250,
        "minimum_node_patients": 100,
        "minimum_source_states": 2,
        "minimum_state_patients": 50,
        "minimum_state_pixels": 128,
        "minimum_downstream_reliable_fraction": 0.90,
    }
    _require_equal(config.get("coverage"), expected_coverage, "coverage")
    _require_equal(
        config.get("statistics"),
        {
            "bootstrap_iterations": 10000,
            "bootstrap_seed": 20260720,
            "holm_alpha": 0.05,
            "patient_equal": True,
            "seed_equal": True,
            "architecture_equal_for_shared_process": True,
        },
        "statistics",
    )
    _require_equal(
        config.get("gates"),
        {
            "maximum_macro_tv": 0.15,
            "maximum_worst_tv": 0.20,
            "maximum_worst_tv_ci_high": 0.25,
            "maximum_shared_increment": 0.03,
            "maximum_shared_increment_ci_high": 0.05,
            "minimum_specificity_delta": 0.10,
            "minimum_specificity_ci_low": 0.05,
            "maximum_null_clean_tv": 0.05,
            "minimum_order_advantage": 0.10,
            "minimum_dose_aligned_fraction": 0.90,
            "required_randomized_controls": ["state_permutation"],
            "structure_delta_ci_low_must_exceed_zero": True,
        },
        "gates",
    )
    _require_equal(
        config.get("protocol"),
        {
            "lock_before_formal_intervention": True,
            "thresholds_may_not_change_after_lock": True,
            "patients_may_not_change_after_lock": True,
            "validation_processes_may_not_change_after_lock": True,
            "formal_mode_rejects_patient_limit": True,
            "smoke_results_are_never_formal": True,
        },
        "protocol",
    )
    return V11Configuration(
        dataset=EXPECTED_DATASET,
        main_models=EXPECTED_MAIN_MODELS,
        control_models=EXPECTED_CONTROL_MODELS,
        model_seeds=EXPECTED_MODEL_SEEDS,
        observer_seeds=EXPECTED_OBSERVER_SEEDS,
        nodes=EXPECTED_NODES,
        relationship_states=EXPECTED_RELATIONSHIP_STATES,
        truth_classes=EXPECTED_TRUTH_CLASSES,
        doses=EXPECTED_DOSES,
        transfer_directions=EXPECTED_TRANSFER_DIRECTIONS,
        validation_patient_count=int(expected_coverage["expected_validation_patients"]),
        formal_patient_count=int(expected_coverage["expected_formal_patients"]),
    )


def _validated_patients(
    patient_ids: Sequence[str],
    *,
    expected_count: int,
    name: str,
) -> list[str]:
    patients = [str(value) for value in patient_ids]
    if len(patients) != expected_count or len(set(patients)) != expected_count:
        raise ValueError(
            f"{name} patient registry must contain {expected_count} unique patients"
        )
    if patients != sorted(patients) or any(not value for value in patients):
        raise ValueError(f"{name} patient registry must be sorted and nonempty")
    return patients


def _validated_model_jobs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = [dict(row) for row in rows]
    expected = {
        (model, seed)
        for model in EXPECTED_MAIN_MODELS + EXPECTED_CONTROL_MODELS
        for seed in EXPECTED_MODEL_SEEDS
    }
    identities = {
        (str(row.get("model")), int(row.get("model_seed", -1)))
        for row in records
    }
    if len(records) != EXPECTED_MODEL_COUNT or identities != expected:
        raise ValueError("registered model job matrix is incomplete or contains duplicates")
    for row in records:
        row["model"] = str(row["model"])
        row["model_seed"] = int(row["model_seed"])
        row["checkpoint_sha256"] = _require_sha256(
            row.get("checkpoint_sha256"),
            "checkpoint_sha256",
        )
    return sorted(
        records,
        key=lambda row: (
            (EXPECTED_MAIN_MODELS + EXPECTED_CONTROL_MODELS).index(row["model"]),
            row["model_seed"],
        ),
    )


def _validated_observer_jobs(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records = [dict(row) for row in rows]
    expected = {
        (model, model_seed, node, observer_seed)
        for model in EXPECTED_MAIN_MODELS + EXPECTED_CONTROL_MODELS
        for model_seed in EXPECTED_MODEL_SEEDS
        for node in EXPECTED_NODES
        for observer_seed in EXPECTED_OBSERVER_SEEDS
    }
    identities = {
        (
            str(row.get("model")),
            int(row.get("model_seed", -1)),
            str(row.get("node")),
            int(row.get("observer_seed", -1)),
        )
        for row in records
    }
    if len(records) != EXPECTED_OBSERVER_COUNT or identities != expected:
        raise ValueError("registered observer matrix is incomplete or contains duplicates")
    for row in records:
        if row.get("status") != "PASS":
            raise ValueError("registered observer did not pass admission")
        row["model"] = str(row["model"])
        row["model_seed"] = int(row["model_seed"])
        row["node"] = str(row["node"])
        row["observer_seed"] = int(row["observer_seed"])
        row["sha256"] = _require_sha256(row.get("sha256"), "observer sha256")
    model_order = EXPECTED_MAIN_MODELS + EXPECTED_CONTROL_MODELS
    return sorted(
        records,
        key=lambda row: (
            model_order.index(row["model"]),
            row["model_seed"],
            EXPECTED_NODES.index(row["node"]),
            row["observer_seed"],
        ),
    )


def _validated_hash_mapping(
    values: Mapping[str, Any],
    *,
    field: str,
    expected_names: Sequence[str] | None = None,
) -> dict[str, str]:
    output = {str(name): _require_sha256(value, f"{field}.{name}") for name, value in values.items()}
    if not output:
        raise ValueError(f"{field} must be nonempty")
    if expected_names is not None and set(output) != set(expected_names):
        raise ValueError(f"{field} must contain exactly {tuple(expected_names)!r}")
    return {name: output[name] for name in sorted(output)}


def _validated_plan_hashes(values: Mapping[str, Any]) -> dict[str, str]:
    expected = {
        f"{model}/seed_{seed}"
        for model in EXPECTED_MAIN_MODELS + EXPECTED_CONTROL_MODELS
        for seed in EXPECTED_MODEL_SEEDS
    }
    if set(values) != expected:
        raise ValueError("intervention plan matrix is incomplete")
    return {
        name: _require_sha256(values[name], f"intervention plan {name}")
        for name in sorted(values)
    }


def build_v11_protocol_lock_payload(
    *,
    configuration: Mapping[str, Any],
    configuration_sha256: str,
    source_identity: Mapping[str, Any],
    model_jobs: Sequence[Mapping[str, Any]],
    observer_jobs: Sequence[Mapping[str, Any]],
    validation_patient_ids: Sequence[str],
    test_patient_ids: Sequence[str],
    process_hashes: Mapping[str, str],
    calibration_hashes: Mapping[str, str],
    intervention_plan_hashes: Mapping[str, str],
    data_inventory: Mapping[str, str],
    environment_identity: Mapping[str, str],
) -> dict[str, Any]:
    validated = validate_v11_configuration(configuration)
    configuration_hash = _require_sha256(configuration_sha256, "configuration_sha256")
    source = dict(source_identity)
    if source.get("code_dirty") is not False:
        raise ValueError("formal V11 protocol requires a clean source tree")
    if not source.get("commit") or not source.get("source_tree_sha256"):
        raise ValueError("formal V11 protocol requires commit and source tree hashes")
    source["source_tree_sha256"] = _require_sha256(
        source["source_tree_sha256"],
        "source_tree_sha256",
    )
    validation_patients = _validated_patients(
        validation_patient_ids,
        expected_count=validated.validation_patient_count,
        name="validation",
    )
    test_patients = _validated_patients(
        test_patient_ids,
        expected_count=validated.formal_patient_count,
        name="test",
    )
    processes = _validated_hash_mapping(
        process_hashes,
        field="process hashes",
        expected_names=EXPECTED_PROCESS_NAMES,
    )
    calibrations = _validated_hash_mapping(
        calibration_hashes,
        field="calibration hashes",
        expected_names=EXPECTED_CALIBRATION_NAMES,
    )
    inventory = _validated_hash_mapping(data_inventory, field="data inventory")
    environment = {str(name): str(value) for name, value in environment_identity.items()}
    if not environment or any(not name or not value for name, value in environment.items()):
        raise ValueError("environment identity must be nonempty")
    return {
        "status": "LOCKED_BEFORE_FORMAL_INTERVENTION",
        "formal_intervention_authorized": True,
        "method_scope": {
            "dataset": validated.dataset,
            "main_architectures": list(validated.main_models),
            "structural_control": list(validated.control_models),
            "high_level_mechanism": "pixel_decision_state_transition_process",
            "intervention_scope": "all_registered_nodes_and_all_downstream_states",
        },
        "configuration_sha256": configuration_hash,
        "source_identity": source,
        "data_inventory": inventory,
        "environment_identity": dict(sorted(environment.items())),
        "registered_model_jobs": _validated_model_jobs(model_jobs),
        "registered_observer_jobs": _validated_observer_jobs(observer_jobs),
        "validation_patient_ids": validation_patients,
        "validation_patient_registry_sha256": sha256_named_values(
            {str(index): value for index, value in enumerate(validation_patients)}
        ),
        "test_patient_ids": test_patients,
        "test_patient_registry_sha256": sha256_named_values(
            {str(index): value for index, value in enumerate(test_patients)}
        ),
        "process_hashes": processes,
        "calibration_hashes": calibrations,
        "intervention_plan_hashes": _validated_plan_hashes(
            intervention_plan_hashes
        ),
        "gate_table": dict(configuration["gates"]),
        "thresholds_may_not_change_after_lock": True,
        "patients_may_not_change_after_lock": True,
        "validation_processes_may_not_change_after_lock": True,
        "smoke_results_are_never_formal": True,
        "configuration": dict(configuration),
    }


def _formal_outputs_exist(root: Path) -> bool:
    if not root.exists():
        return False
    return any(path.is_file() for path in root.rglob("*"))


def write_protocol_lock(
    path: Path,
    payload: Mapping[str, Any],
    *,
    formal_output_root: Path,
) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"existing protocol lock differs: {path}")
        return
    if _formal_outputs_exist(formal_output_root):
        raise ValueError("formal intervention outputs already exist before protocol lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_and_validate_v11_protocol_lock(
    path: Path,
    *,
    active_configuration_sha256: str,
    active_source_tree_sha256: str,
    test_patient_ids: Sequence[str],
    process_hashes: Mapping[str, str],
    model: str,
    model_seed: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"V11 protocol lock is missing: {path}")
    lock = _load_json(path)
    if lock.get("status") != "LOCKED_BEFORE_FORMAL_INTERVENTION" or lock.get(
        "formal_intervention_authorized"
    ) is not True:
        raise ValueError("V11 protocol does not authorize formal intervention")
    if lock.get("configuration_sha256") != _require_sha256(
        active_configuration_sha256,
        "active_configuration_sha256",
    ):
        raise ValueError("active configuration SHA-256 differs from the V11 lock")
    if lock.get("source_identity", {}).get("source_tree_sha256") != _require_sha256(
        active_source_tree_sha256,
        "active_source_tree_sha256",
    ):
        raise ValueError("active source tree SHA-256 differs from the V11 lock")
    registered_patients = lock.get("test_patient_ids", [])
    if list(test_patient_ids) != registered_patients:
        raise ValueError("formal test patient registry differs from the V11 lock")
    active_processes = _validated_hash_mapping(
        process_hashes,
        field="process hashes",
        expected_names=EXPECTED_PROCESS_NAMES,
    )
    if lock.get("process_hashes") != active_processes:
        raise ValueError("active process hashes differ from the V11 lock")
    registered_jobs = {
        (str(row.get("model")), int(row.get("model_seed", -1)))
        for row in lock.get("registered_model_jobs", [])
    }
    if (str(model), int(model_seed)) not in registered_jobs:
        raise ValueError(f"model/seed {model}/{model_seed} is absent from the V11 lock")
    validate_v11_configuration(lock.get("configuration", {}))
    return lock


def source_tree_sha256(workspace: Path) -> str:
    files: list[Path] = []
    for root_name in ("src", "scripts", "configs", "tests"):
        root = workspace / root_name
        if root.is_dir():
            files.extend(path for path in root.rglob("*") if path.is_file())
    for relative in ("pyproject.toml", "manifests/model_matrix.yaml"):
        path = workspace / relative
        if path.is_file():
            files.append(path)
    selected = sorted(
        {
            path.resolve()
            for path in files
            if "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}
        },
        key=lambda path: path.relative_to(workspace).as_posix(),
    )
    digest = hashlib.sha256()
    for path in selected:
        relative = path.relative_to(workspace).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def resolve_source_identity(workspace: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(workspace), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(workspace), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "commit": commit,
        "source_tree_sha256": source_tree_sha256(workspace),
        "code_dirty": bool(status.strip()),
        "identity_source": "git",
    }


def _resolve_checkpoint(
    workspace: Path,
    asset_root: Path,
    checkpoint: Mapping[str, Any],
) -> Path:
    root = str(checkpoint["root"])
    if root == "workspace":
        return workspace / str(checkpoint["path"])
    if root == "asset":
        return asset_root / str(checkpoint["path"])
    raise ValueError(f"unsupported checkpoint root: {root}")


def _collect_model_jobs(
    workspace: Path,
    asset_root: Path,
    configuration: Mapping[str, Any],
) -> list[dict[str, Any]]:
    matrix = _load_yaml(workspace / str(configuration["model_matrix"]))
    expected_models = set(EXPECTED_MAIN_MODELS + EXPECTED_CONTROL_MODELS)
    output: list[dict[str, Any]] = []
    for job in matrix.get("jobs", []):
        model = str(job.get("model"))
        seed = int(job.get("seed", -1))
        if model not in expected_models or seed not in EXPECTED_MODEL_SEEDS:
            continue
        checkpoint = _resolve_checkpoint(workspace, asset_root, job["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        output.append(
            {
                "model": model,
                "model_seed": seed,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
    return _validated_model_jobs(output)


def _collect_observer_jobs(
    workspace: Path,
    configuration: Mapping[str, Any],
) -> list[dict[str, Any]]:
    observer_root = workspace / str(configuration["observer_root"])
    output: list[dict[str, Any]] = []
    for model in EXPECTED_MAIN_MODELS + EXPECTED_CONTROL_MODELS:
        for model_seed in EXPECTED_MODEL_SEEDS:
            directory = observer_root / model / f"seed_{model_seed}"
            status_path = directory / "v1_status.json"
            failed_path = directory / "failed_nodes.json"
            reliability_path = directory / "reliability_thresholds.json"
            for path in (status_path, failed_path, reliability_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            if _load_json(status_path).get("status") != "PASS":
                raise ValueError(f"observer admission is not PASS: {directory}")
            if _load_json(failed_path).get("failed_nodes"):
                raise ValueError(f"observer admission contains failed nodes: {directory}")
            reliability = _load_json(reliability_path).get("threshold")
            if reliability is None:
                raise ValueError(f"observer reliability threshold is missing: {directory}")
            for node in EXPECTED_NODES:
                for observer_seed in EXPECTED_OBSERVER_SEEDS:
                    path = (
                        directory
                        / "observers"
                        / node
                        / "real"
                        / f"seed_{observer_seed}.pt"
                    )
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    output.append(
                        {
                            "model": model,
                            "model_seed": model_seed,
                            "node": node,
                            "observer_seed": observer_seed,
                            "path": str(path.resolve()),
                            "sha256": sha256_file(path),
                            "status": "PASS",
                            "reliability_threshold": float(reliability),
                        }
                    )
    return _validated_observer_jobs(output)


def _registered_patients(
    workspace: Path,
    asset_root: Path,
) -> tuple[list[str], list[str], dict[str, str]]:
    data_config_path = workspace / "configs" / "data" / "brats2023_2d.yaml"
    data_config = _load_yaml(data_config_path)
    split_path = asset_root / str(data_config["split_file"])
    split_payload = _load_json(split_path)
    validation = _validated_patients(
        sorted(split_payload.get("val", [])),
        expected_count=125,
        name="validation",
    )
    test = _validated_patients(
        sorted(split_payload.get("test", [])),
        expected_count=250,
        name="test",
    )
    return validation, test, {
        "data_config_sha256": sha256_file(data_config_path),
        "split_sha256": sha256_file(split_path),
    }


def _validation_asset_hashes(
    workspace: Path,
    configuration: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    fit_root = workspace / str(configuration["validation_fit_root"])
    process_paths = {
        name: fit_root / f"{name}.json" for name in EXPECTED_PROCESS_NAMES
    }
    calibration_paths = {
        "matching": fit_root / "matching_calibration.json",
        "norm_and_leakage": fit_root / "norm_and_leakage_calibration.json",
        "history_admission": fit_root / "history_admission.json",
    }
    for path in (*process_paths.values(), *calibration_paths.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    history = _load_json(calibration_paths["history_admission"])
    if history.get("status") != "PASS":
        raise ValueError("history admission did not pass")
    return (
        {name: sha256_file(path) for name, path in process_paths.items()},
        {name: sha256_file(path) for name, path in calibration_paths.items()},
    )


def _collect_plan_hashes(
    workspace: Path,
    configuration: Mapping[str, Any],
    *,
    test_patient_ids: Sequence[str],
) -> dict[str, str]:
    root = workspace / str(configuration["intervention_plan_root"])
    patients = _validated_patients(
        test_patient_ids,
        expected_count=250,
        name="test",
    )
    coverage_limits = configuration["coverage"]
    hashes: dict[str, str] = {}
    for model in EXPECTED_MAIN_MODELS + EXPECTED_CONTROL_MODELS:
        for model_seed in EXPECTED_MODEL_SEEDS:
            identity = f"{model}/seed_{model_seed}"
            path = root / model / f"seed_{model_seed}" / "plan_manifest.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            manifest = _load_json(path)
            if (
                manifest.get("status") != "COMPLETE_LOCKABLE_PLAN"
                or manifest.get("model") != model
                or int(manifest.get("model_seed", -1)) != model_seed
                or manifest.get("split") != "test"
                or manifest.get("failed_nodes") != []
                or manifest.get("patient_ids") != patients
                or int(manifest.get("patient_count", -1)) != len(patients)
                or manifest.get("full_activations_persisted") is not False
            ):
                raise ValueError(f"intervention plan is not lockable: {identity}")
            plan_hashes = manifest.get("patient_plan_hashes")
            if not isinstance(plan_hashes, dict) or list(plan_hashes) != patients:
                raise ValueError(f"intervention plan patient hashes differ: {identity}")
            if manifest.get("patient_plan_registry_sha256") != sha256_named_values(
                plan_hashes
            ):
                raise ValueError(f"intervention plan registry hash differs: {identity}")
            patient_root = path.parent / "patient_plans"
            for patient_id in patients:
                patient_path = patient_root / f"{patient_id}.json"
                if (
                    not patient_path.is_file()
                    or sha256_file(patient_path) != str(plan_hashes[patient_id])
                ):
                    raise ValueError(
                        f"intervention patient plan differs: {identity}/{patient_id}"
                    )
            coverage = manifest.get("coverage")
            if not isinstance(coverage, dict) or set(coverage) != set(EXPECTED_NODES):
                raise ValueError(f"intervention plan coverage is incomplete: {identity}")
            for node in EXPECTED_NODES:
                summary = coverage[node]
                evaluable = [
                    state
                    for state, values in summary.get("target_states", {}).items()
                    if int(values.get("patient_count", -1))
                    >= int(coverage_limits["minimum_state_patients"])
                    and int(values.get("pixel_count", -1))
                    >= int(coverage_limits["minimum_state_pixels"])
                ]
                if (
                    int(summary.get("patient_count", -1))
                    < int(coverage_limits["minimum_node_patients"])
                    or len(evaluable)
                    < int(coverage_limits["minimum_source_states"])
                    or sorted(str(value) for value in evaluable)
                    != sorted(
                        str(value)
                        for value in summary.get("evaluable_target_states", [])
                    )
                ):
                    raise ValueError(
                        f"intervention plan coverage gate failed: {identity}/{node}"
                    )
            hashes[identity] = sha256_file(path)
    return _validated_plan_hashes(hashes)


def _environment_identity() -> dict[str, str]:
    import numpy
    import pandas
    import torch

    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": str(numpy.__version__),
        "pandas": str(pandas.__version__),
        "cuda": str(torch.version.cuda),
    }


def _trace_directory(
    workspace: Path,
    config: Mapping[str, Any],
    *,
    split_role: str,
    model: str,
    model_seed: int,
) -> Path:
    templates = config.get("clean_trace_templates", {})
    try:
        template = str(templates[split_role][model])
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"clean trace template is missing for {split_role}/{model}"
        ) from error
    return workspace / template.format(model=model, seed=int(model_seed))


def fit_validation_processes(
    workspace: Path,
    configuration: Mapping[str, Any],
    *,
    validation_patient_ids: Sequence[str],
) -> dict[str, Any]:
    """Fit H_U, H_T, and H_shared from validation-only clean trajectories."""

    validated = validate_v11_configuration(configuration)
    patients = _validated_patients(
        validation_patient_ids,
        expected_count=validated.validation_patient_count,
        name="validation",
    )
    fit_root = workspace / str(configuration["validation_fit_root"])
    fit_root.mkdir(parents=True, exist_ok=True)
    process_by_model = {}
    history_admission: dict[str, Any] = {}
    row_inventory: dict[str, Any] = {}
    process_names = {
        "unet_baseline": "H_U",
        "transunet_r50_vit_b16": "H_T",
    }
    for model in validated.main_models:
        transition_frames = []
        history_frames = []
        trace_hashes: dict[str, str] = {}
        for model_seed in validated.model_seeds:
            directory = _trace_directory(
                workspace,
                configuration,
                split_role="validation",
                model=model,
                model_seed=model_seed,
            )
            paths = [directory / f"{patient_id}.npz" for patient_id in patients]
            missing = [path for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(missing[0])
            transitions, histories = collect_case_trace_rows(
                paths,
                model_seed=model_seed,
                num_classes=4,
            )
            transition_frames.append(transitions)
            history_frames.append(histories)
            trace_hashes[f"seed_{model_seed}_patient_registry"] = sha256_named_values(
                {patient_id: sha256_file(path) for patient_id, path in zip(patients, paths, strict=True)}
            )
        import pandas as pd

        transition_rows = pd.concat(transition_frames, ignore_index=True)
        history_rows = pd.concat(history_frames, ignore_index=True)
        process = estimate_patient_equal_process(
            transition_rows,
            node_names=validated.nodes,
            state_count=len(validated.relationship_states),
            alpha=float(configuration["high_level_process"]["dirichlet_alpha"]),
            source=model,
        )
        comparison = cross_validated_history_comparison(
            history_rows,
            node_names=validated.nodes,
            state_count=len(validated.relationship_states),
            alpha=float(configuration["high_level_process"]["dirichlet_alpha"]),
            fold_count=int(configuration["high_level_process"]["patient_folds"]),
            fold_seed=int(configuration["high_level_process"]["fold_seed"]),
        )
        admission = evaluate_history_dependence(
            comparison,
            tolerance=float(configuration["high_level_process"]["history_tv_tolerance"]),
            bootstrap_iterations=int(configuration["statistics"]["bootstrap_iterations"]),
            bootstrap_seed=int(configuration["statistics"]["bootstrap_seed"]),
        )
        process_by_model[model] = process
        process_name = process_names[model]
        _write_json_atomic(fit_root / f"{process_name}.json", process.to_payload())
        transition_rows.to_parquet(
            fit_root / f"{process_name}_transition_counts.parquet",
            index=False,
        )
        history_rows.to_parquet(
            fit_root / f"{process_name}_history_counts.parquet",
            index=False,
        )
        comparison.to_parquet(
            fit_root / f"{process_name}_history_comparison.parquet",
            index=False,
        )
        history_admission[process_name] = admission
        row_inventory[process_name] = {
            "transition_row_count": int(len(transition_rows)),
            "history_row_count": int(len(history_rows)),
            "history_comparison_row_count": int(len(comparison)),
            "trace_hashes": trace_hashes,
        }
    shared = pool_architecture_processes(
        {
            "unet_baseline": process_by_model["unet_baseline"],
            "transunet_r50_vit_b16": process_by_model[
                "transunet_r50_vit_b16"
            ],
        }
    )
    _write_json_atomic(fit_root / "H_shared.json", shared.to_payload())
    history_status = (
        "PASS"
        if all(value["status"] == "PASS" for value in history_admission.values())
        else "HIGH_LEVEL_MODEL_MISSPECIFIED"
    )
    history_payload = {
        "status": history_status,
        "architectures": history_admission,
        "tolerance": float(configuration["high_level_process"]["history_tv_tolerance"]),
        "patient_folds": int(configuration["high_level_process"]["patient_folds"]),
        "fold_seed": int(configuration["high_level_process"]["fold_seed"]),
    }
    _write_json_atomic(fit_root / "history_admission.json", history_payload)
    process_hashes = {
        name: sha256_file(fit_root / f"{name}.json")
        for name in EXPECTED_PROCESS_NAMES
    }
    status = {
        "status": history_status,
        "process_hashes": process_hashes,
        "history_admission_sha256": sha256_file(
            fit_root / "history_admission.json"
        ),
        "row_inventory": row_inventory,
        "validation_patient_registry_sha256": sha256_named_values(
            {str(index): value for index, value in enumerate(patients)}
        ),
    }
    _write_json_atomic(fit_root / "process_fit_status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and inspect the immutable V11 causal-abstraction configuration."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/v11_causal_abstraction.yaml"),
    )
    parser.add_argument(
        "--phase",
        choices=("inspect", "fit-processes", "lock"),
        default="lock",
    )
    parser.add_argument("--asset-root", type=Path)
    args = parser.parse_args()
    workspace = args.workspace_root.resolve()
    config_path = args.config if args.config.is_absolute() else workspace / args.config
    configuration = _load_yaml(config_path.resolve())
    validated = validate_v11_configuration(configuration)
    if args.phase == "fit-processes":
        if args.asset_root is None:
            raise ValueError("--asset-root is required for validation process fitting")
        validation_patients, _, _ = _registered_patients(
            workspace,
            args.asset_root.resolve(),
        )
        result = fit_validation_processes(
            workspace,
            configuration,
            validation_patient_ids=validation_patients,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    if args.phase == "lock":
        if args.asset_root is None:
            raise ValueError("--asset-root is required for the formal V11 lock")
        asset_root = args.asset_root.resolve()
        validation_patients, test_patients, data_inventory = _registered_patients(
            workspace,
            asset_root,
        )
        process_hashes, calibration_hashes = _validation_asset_hashes(
            workspace,
            configuration,
        )
        payload = build_v11_protocol_lock_payload(
            configuration=configuration,
            configuration_sha256=sha256_file(config_path.resolve()),
            source_identity=resolve_source_identity(workspace),
            model_jobs=_collect_model_jobs(workspace, asset_root, configuration),
            observer_jobs=_collect_observer_jobs(workspace, configuration),
            validation_patient_ids=validation_patients,
            test_patient_ids=test_patients,
            process_hashes=process_hashes,
            calibration_hashes=calibration_hashes,
            intervention_plan_hashes=_collect_plan_hashes(
                workspace,
                configuration,
                test_patient_ids=test_patients,
            ),
            data_inventory=data_inventory,
            environment_identity=_environment_identity(),
        )
        lock_path = workspace / str(configuration["protocol_lock"])
        formal_root = workspace / str(configuration["formal_job_root"])
        write_protocol_lock(
            lock_path,
            payload,
            formal_output_root=formal_root,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    result = {
        "status": "CONFIGURATION_VALID",
        "configuration_sha256": sha256_file(config_path.resolve()),
        "source_identity": resolve_source_identity(workspace),
        "scope": {
            "dataset": validated.dataset,
            "main_models": list(validated.main_models),
            "control_models": list(validated.control_models),
            "node_count": len(validated.nodes),
            "model_job_count": EXPECTED_MODEL_COUNT,
            "observer_job_count": EXPECTED_OBSERVER_COUNT,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
