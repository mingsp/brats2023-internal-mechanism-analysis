from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import yaml


EXPECTED_MODELS = ("unet_baseline", "unet_noskip")
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
EXPECTED_RESTORE_NODES = EXPECTED_NODES[1:]
EXPECTED_V6_IDENTITY = {
    "transition": "up1->up2",
    "transition_index": 4,
    "module": "up2",
    "argument_index": 1,
    "source": "down2",
}
FORBIDDEN_CANDIDATE_KEYS = {
    "candidate",
    "candidate_transition",
    "discovery_split",
    "target",
    "target_transition",
}


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise ValueError(f"registered {field} changed: {actual!r} != {expected!r}")


def validate_registered_configuration(config: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = FORBIDDEN_CANDIDATE_KEYS.intersection(config)
    if forbidden:
        raise ValueError(
            "V7 is network-wide; candidate-specific fields are forbidden: "
            f"{sorted(forbidden)}"
        )

    _require_equal(
        config.get("name"),
        "v7_network_process_intervention_alignment",
        "name",
    )
    _require_equal(tuple(config.get("models", ())), EXPECTED_MODELS, "models")
    _require_equal(
        tuple(config.get("model_seeds", ())), EXPECTED_MODEL_SEEDS, "model_seeds"
    )
    _require_equal(
        tuple(config.get("observer_seeds", ())),
        EXPECTED_OBSERVER_SEEDS,
        "observer_seeds",
    )
    nodes = tuple(config.get("nodes", ()))
    _require_equal(nodes, EXPECTED_NODES, "nodes")
    _require_equal(config.get("primary_split"), "test", "primary_split")
    _require_equal(tuple(config.get("truth_classes", ())), (1, 2, 3), "truth_classes")
    _require_equal(
        config.get("process_event"),
        "output_anchored_persistent_correction",
        "process_event",
    )

    eligibility = config.get("eligibility", {})
    expected_eligibility = {
        "minimum_union_pixels_on_selected_slice": 32,
        "minimum_transition_pixels": 8,
        "minimum_target_feature_positions": 4,
        "minimum_control_feature_positions": 4,
        "minimum_patients_per_transition": 30,
        "minimum_aggregate_pixels_per_transition": 2048,
    }
    _require_equal(eligibility, expected_eligibility, "eligibility")

    intervention = config.get("intervention", {})
    _require_equal(intervention.get("root_node"), "down1", "intervention.root_node")
    _require_equal(
        intervention.get("operator"),
        "spatial_roll_module_output",
        "intervention.operator",
    )
    _require_equal(
        tuple(intervention.get("input_equivalent_shift_yx", ())),
        (8, 8),
        "intervention.input_equivalent_shift_yx",
    )
    _require_equal(
        tuple(intervention.get("restore_nodes", ())),
        EXPECTED_RESTORE_NODES,
        "intervention.restore_nodes",
    )
    _require_equal(
        float(intervention.get("logit_tolerance", float("nan"))),
        1.0e-6,
        "intervention.logit_tolerance",
    )

    matching = config.get("matching", {})
    expected_matching = {
        "same_dominant_truth_class": True,
        "boundary_bin_edges": [1.5, 3.5, 7.5],
        "activation_norm_quantile_bins": 4,
        "without_replacement": True,
    }
    _require_equal(matching, expected_matching, "matching")

    statistics = config.get("statistics", {})
    expected_statistics = {
        "bootstrap_iterations": 10000,
        "bootstrap_seed": 20260717,
        "alpha": 0.05,
        "multiplicity": "holm_global_macro_micro",
        "minimum_supported_seeds": 2,
        "minimum_evaluable_transitions": 5,
        "minimum_positive_receiving_transitions": 4,
    }
    _require_equal(statistics, expected_statistics, "statistics")

    protocol = config.get("protocol", {})
    expected_protocol = {
        "lock_before_test_intervention": True,
        "thresholds_may_not_change_after_lock": True,
        "v4_must_remain_not_triggered": True,
        "v6_must_remain_candidate_specific": True,
    }
    _require_equal(protocol, expected_protocol, "protocol")

    transitions = [f"{left}->{right}" for left, right in zip(nodes[:-1], nodes[1:])]
    return {"transitions": transitions, "matrix_shape": [7, 7]}


def validate_patient_registry(
    patient_ids: Sequence[str], *, expected: Sequence[str] | None = None
) -> list[str]:
    registered = list(patient_ids)
    if len(registered) != 250:
        raise ValueError(f"formal patient registry must contain exactly 250 patients, got {len(registered)}")
    if len(set(registered)) != len(registered):
        raise ValueError("formal patient registry contains duplicate identities")
    if not all(isinstance(patient_id, str) and patient_id for patient_id in registered):
        raise ValueError("formal patient registry contains an invalid identity")
    if expected is not None and registered != list(expected):
        raise ValueError("formal patient ordered registry differs from the registered split")
    return registered


def validate_v4_status(status: Mapping[str, Any]) -> None:
    if status.get("triggered") is not False:
        raise ValueError("V4 must remain NOT_TRIGGERED (triggered=false)")


def validate_v6_scope(lock: Mapping[str, Any]) -> None:
    _require_equal(
        lock.get("status"),
        "LOCKED_BEFORE_FIRST_INTERVENTION",
        "V6 lock status",
    )
    _require_equal(
        lock.get("candidate_identity"), EXPECTED_V6_IDENTITY, "V6 candidate identity"
    )
    if lock.get("global_two_metric_gate_passed") is not False:
        raise ValueError("V6 must not be promoted to global network-wide evidence")
    if lock.get("global_two_metric_gate_must_remain_reported") is not True:
        raise ValueError("V6 global-gate limitation must remain reported")


def _expected_job_identities() -> set[tuple[str, int]]:
    return {
        (model, seed)
        for model in EXPECTED_MODELS
        for seed in EXPECTED_MODEL_SEEDS
    }


def _validate_job_records(
    records: Sequence[Mapping[str, Any]], *, record_name: str
) -> list[dict[str, Any]]:
    normalized = [dict(record) for record in records]
    identities = {
        (str(record["model"]), int(record["model_seed"])) for record in normalized
    }
    _require_equal(identities, _expected_job_identities(), record_name)
    if record_name == "observer_jobs":
        failed = [record for record in normalized if record.get("status") != "PASS"]
        if failed:
            raise ValueError(f"observer admission failed: {failed}")
    if record_name == "model_jobs":
        missing = [record for record in normalized if not record.get("checkpoint_sha256")]
        if missing:
            raise ValueError(f"model checkpoint SHA-256 missing: {missing}")
    return sorted(normalized, key=lambda item: (item["model"], int(item["model_seed"])))


def build_protocol_lock_payload(
    *,
    configuration: Mapping[str, Any],
    configuration_sha256: str,
    source_identity: Mapping[str, Any],
    data_inventory: Mapping[str, Any],
    model_jobs: Sequence[Mapping[str, Any]],
    observer_jobs: Sequence[Mapping[str, Any]],
    patient_ids: Sequence[str],
    v4_status: Mapping[str, Any],
    v4_sha256: str,
    v6_lock: Mapping[str, Any],
    v6_sha256: str,
) -> dict[str, Any]:
    design = validate_registered_configuration(configuration)
    patients = validate_patient_registry(patient_ids)
    validate_v4_status(v4_status)
    validate_v6_scope(v6_lock)
    models = _validate_job_records(model_jobs, record_name="model_jobs")
    observers = _validate_job_records(observer_jobs, record_name="observer_jobs")

    if source_identity.get("code_dirty") is not False:
        raise ValueError("formal protocol requires code_dirty=false")
    if not source_identity.get("commit"):
        raise ValueError("formal protocol requires a source commit identity")
    if not source_identity.get("source_tree_sha256"):
        raise ValueError("formal protocol requires a source tree SHA-256")
    if not data_inventory.get("sha256"):
        raise ValueError("formal protocol requires a data inventory SHA-256")

    return {
        "status": "LOCKED_BEFORE_TEST_INTERVENTION",
        "formal_intervention_authorized": True,
        "configuration_sha256": str(configuration_sha256),
        "source_identity": dict(source_identity),
        "data_inventory": dict(data_inventory),
        "registered_model_jobs": models,
        "registered_observer_jobs": observers,
        "registered_patient_ids": patients,
        "registered_patient_count": len(patients),
        "registered_transitions": design["transitions"],
        "registered_restore_nodes": list(EXPECTED_RESTORE_NODES),
        "registered_matrix_shape": design["matrix_shape"],
        "v4_state": {
            "triggered": False,
            "sha256": str(v4_sha256),
        },
        "v6_scope": {
            "candidate_identity": EXPECTED_V6_IDENTITY,
            "global_two_metric_gate_passed": False,
            "sha256": str(v6_sha256),
        },
        "configuration": dict(configuration),
    }


def write_protocol_lock(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"existing protocol lock differs: {path}")
        return
    if path.parent.exists():
        existing = [item for item in path.parent.rglob("*") if item.is_file()]
        if existing:
            raise ValueError(
                "formal output directory is not empty before protocol lock: "
                + ", ".join(str(item) for item in existing[:5])
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def load_and_validate_formal_lock(
    path: Path,
    *,
    active_configuration_sha256: str,
    active_source_tree_sha256: str,
    patient_ids: Sequence[str],
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"formal protocol lock is missing: {path}")
    lock = _load_json(path)
    if lock.get("status") != "LOCKED_BEFORE_TEST_INTERVENTION":
        raise ValueError("formal protocol lock has an invalid status")
    if lock.get("formal_intervention_authorized") is not True:
        raise ValueError("formal intervention is not authorized")
    if lock.get("configuration_sha256") != active_configuration_sha256:
        raise ValueError("active configuration SHA-256 differs from the formal lock")
    source = lock.get("source_identity", {})
    if source.get("code_dirty") is not False:
        raise ValueError("formal lock records a dirty source tree")
    if source.get("source_tree_sha256") != active_source_tree_sha256:
        raise ValueError("active source tree SHA-256 differs from the formal lock")
    registered = lock.get("registered_patient_ids", [])
    try:
        validate_patient_registry(patient_ids, expected=registered)
    except ValueError as error:
        raise ValueError(f"formal patient registry changed after lock: {error}") from error
    return lock


def source_tree_sha256(workspace: Path) -> str:
    files: list[Path] = []
    for root_name in ("src", "scripts", "configs", "tests"):
        root = workspace / root_name
        if root.is_dir():
            files.extend(path for path in root.rglob("*") if path.is_file())
    for relative in (
        "pyproject.toml",
        "manifests/model_matrix.yaml",
        "manifests/run_commands.yaml",
    ):
        path = workspace / relative
        if path.is_file():
            files.append(path)
    files = sorted(
        {
            path.resolve()
            for path in files
            if "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}
        },
        key=lambda path: path.relative_to(workspace).as_posix(),
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(workspace).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def resolve_source_identity(workspace: Path) -> dict[str, Any]:
    tree_sha = source_tree_sha256(workspace)
    try:
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
            "source_tree_sha256": tree_sha,
            "code_dirty": bool(status.strip()),
            "identity_source": "git",
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        manifest_path = workspace / "manifests" / "v7_source_identity.json"
        if not manifest_path.is_file():
            raise ValueError(
                "workspace is not a Git checkout and manifests/v7_source_identity.json is missing"
            )
        identity = _load_json(manifest_path)
        if identity.get("source_tree_sha256") != tree_sha:
            raise ValueError("server source tree differs from the registered source identity")
        if identity.get("code_dirty") is not False or not identity.get("commit"):
            raise ValueError("registered source identity is not a clean commit")
        identity["identity_source"] = "verified_manifest"
        return identity


def _resolve_checkpoint(
    workspace: Path, asset_root: Path, checkpoint: Mapping[str, Any]
) -> Path:
    root = str(checkpoint["root"])
    base = asset_root if root == "asset" else workspace if root == "workspace" else None
    if base is None:
        raise ValueError(f"unsupported checkpoint root: {root}")
    return base / str(checkpoint["path"])


def _collect_model_jobs(
    workspace: Path,
    asset_root: Path,
    matrix: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected = _expected_job_identities()
    output: list[dict[str, Any]] = []
    for job in matrix.get("jobs", []):
        identity = (str(job["model"]), int(job["seed"]))
        if identity not in expected:
            continue
        checkpoint = _resolve_checkpoint(workspace, asset_root, job["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        output.append(
            {
                "model": identity[0],
                "model_seed": identity[1],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
    return _validate_job_records(output, record_name="model_jobs")


def _job_map(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["job"]): row for row in payload.get("jobs", [])}


def _collect_observer_jobs(workspace: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    observer_root = workspace / str(config["observer_root"])
    training_path = observer_root / "matrix_training_status.json"
    evaluation_path = observer_root / "v1_matrix_status.json"
    for path in (training_path, evaluation_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    training = _load_json(training_path)
    evaluation = _load_json(evaluation_path)
    if training.get("status") != "PASS" or evaluation.get("status") != "PASS":
        raise ValueError("observer matrix admission did not pass")
    training_jobs = _job_map(training)
    evaluation_jobs = _job_map(evaluation)
    output: list[dict[str, Any]] = []
    for model in EXPECTED_MODELS:
        for seed in EXPECTED_MODEL_SEEDS:
            job_id = f"{model}/seed_{seed}"
            train_row = training_jobs.get(job_id)
            eval_row = evaluation_jobs.get(job_id)
            if train_row is None or eval_row is None:
                raise ValueError(f"observer status missing for {job_id}")
            status = "PASS" if train_row.get("status") == eval_row.get("status") == "PASS" else "FAIL"
            output.append(
                {
                    "model": model,
                    "model_seed": seed,
                    "status": status,
                    "trained_observer_count": int(train_row.get("observer_count", 0)),
                    "selection_status": eval_row.get("reliability", {}).get("selection_status"),
                    "failed_nodes": eval_row.get("failed_nodes", {}),
                }
            )
    return _validate_job_records(output, record_name="observer_jobs")


def _load_registered_patients(
    workspace: Path,
    asset_root: Path,
    config: Mapping[str, Any],
) -> tuple[list[str], dict[str, str]]:
    data_config_path = workspace / "configs" / "data" / "brats2023_2d.yaml"
    data_config = _load_yaml(data_config_path)
    split_path = asset_root / str(data_config["split_file"])
    split_payload = _load_json(split_path)
    split = str(config["primary_split"])
    patients = validate_patient_registry(split_payload.get(split, []))

    v3_path = workspace / str(config["v3_root"]) / "transition_event_by_patient.parquet"
    if not v3_path.is_file():
        raise FileNotFoundError(v3_path)
    frame = pd.read_parquet(
        v3_path,
        columns=["model", "model_seed", "split", "patient_id"],
    )
    frame = frame[
        frame["model"].isin(EXPECTED_MODELS)
        & frame["model_seed"].isin(EXPECTED_MODEL_SEEDS)
        & (frame["split"] == split)
    ]
    for (model, seed), group in frame.groupby(["model", "model_seed"], sort=True):
        observed = sorted(group["patient_id"].drop_duplicates().tolist())
        if observed != sorted(patients):
            raise ValueError(f"V3 patient registry differs for {model}/seed_{seed}")
    return patients, {
        "path": str(split_path),
        "sha256": sha256_file(split_path),
        "data_config_sha256": sha256_file(data_config_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lock the network-wide V7 intervention protocol before test intervention."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/v7_network_process_intervention_alignment.yaml"
        ),
    )
    parser.add_argument("--asset-root", type=Path, required=True)
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = workspace / config_path
    config_path = config_path.resolve()
    asset_root = args.asset_root.resolve()
    config = _load_yaml(config_path)
    validate_registered_configuration(config)

    matrix_path = workspace / str(config["model_matrix"])
    matrix = _load_yaml(matrix_path)
    model_jobs = _collect_model_jobs(workspace, asset_root, matrix)
    observer_jobs = _collect_observer_jobs(workspace, config)
    patient_ids, data_inventory = _load_registered_patients(
        workspace, asset_root, config
    )

    v4_path = workspace / str(config["v3_root"]) / "v4_trigger.json"
    v6_path = (
        workspace
        / "results"
        / "v6_pixel_transition_causal"
        / "v6_protocol_lock_v2.json"
    )
    for path in (v4_path, v6_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    v4_status = _load_json(v4_path)
    v6_lock = _load_json(v6_path)
    source_identity = resolve_source_identity(workspace)

    payload = build_protocol_lock_payload(
        configuration=config,
        configuration_sha256=sha256_file(config_path),
        source_identity=source_identity,
        data_inventory=data_inventory,
        model_jobs=model_jobs,
        observer_jobs=observer_jobs,
        patient_ids=patient_ids,
        v4_status=v4_status,
        v4_sha256=sha256_file(v4_path),
        v6_lock=v6_lock,
        v6_sha256=sha256_file(v6_path),
    )
    output_root = workspace / str(config["output_root"])
    lock_path = output_root / "v7_protocol_lock.json"
    write_protocol_lock(lock_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"V7 protocol locked: {lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
