from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from pptt.interventions.transunet_paths import declared_transunet_decoder_paths


EXPECTED_MODEL = "transunet_r50_vit_b16"
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


def validate_registered_configuration(config: Mapping[str, Any]) -> dict[str, Any]:
    _require_equal(
        config.get("name"),
        "v8_transunet_mechanism_replication",
        "name",
    )
    _require_equal(config.get("model"), EXPECTED_MODEL, "model")
    _require_equal(
        tuple(config.get("model_seeds", ())),
        EXPECTED_MODEL_SEEDS,
        "model_seeds",
    )
    _require_equal(
        tuple(config.get("observer_seeds", ())),
        EXPECTED_OBSERVER_SEEDS,
        "observer_seeds",
    )
    _require_equal(tuple(config.get("nodes", ())), EXPECTED_NODES, "nodes")
    _require_equal(config.get("primary_split"), "val", "primary_split")
    _require_equal(config.get("discovery_split"), "val", "discovery_split")
    _require_equal(
        config.get("intervention_split"),
        "test",
        "intervention_split",
    )
    declared_ids = tuple(path.path_id for path in declared_transunet_decoder_paths())
    _require_equal(
        tuple(config.get("candidate_paths", ())),
        declared_ids,
        "candidate_paths",
    )
    _require_equal(tuple(config.get("truth_classes", ())), (1, 2, 3), "truth_classes")
    _require_equal(
        config.get("eligibility"),
        {
            "minimum_target_pixels_on_selected_slice": 32,
            "minimum_target_feature_positions": 8,
            "minimum_control_feature_positions": 8,
            "minimum_candidate_patients_per_seed": 20,
            "minimum_candidate_target_pixels_per_seed": 512,
        },
        "eligibility",
    )
    _require_equal(
        config.get("intervention"),
        {
            "operator": "spatial_cyclic_shift_with_local_clean_restore",
            "feature_shift_yx": [2, 2],
            "restoration_alphas": [0.25, 0.50, 0.75, 1.00],
            "primary_alpha": 1.00,
        },
        "intervention",
    )
    _require_equal(
        config.get("matching"),
        {
            "same_truth_class": True,
            "boundary_bin_edges": [1.5, 3.5, 7.5],
            "activation_norm_quantile_bins": 4,
        },
        "matching",
    )
    statistics = config.get("statistics", {})
    _require_equal(statistics.get("bootstrap_iterations"), 10000, "bootstrap_iterations")
    _require_equal(statistics.get("bootstrap_seed"), 20260720, "bootstrap_seed")
    _require_equal(float(statistics.get("alpha", -1)), 0.05, "alpha")
    _require_equal(
        float(statistics.get("minimum_standardized_effect", -1)),
        0.8,
        "minimum_standardized_effect",
    )
    _require_equal(
        int(statistics.get("minimum_supported_seeds", -1)),
        2,
        "minimum_supported_seeds",
    )
    _require_equal(statistics.get("multiplicity"), "holm_global", "multiplicity")
    _require_equal(
        int(config.get("validation", {}).get("expected_patient_count_per_seed", -1)),
        125,
        "validation patient count",
    )
    _require_equal(
        int(config.get("formal", {}).get("expected_patient_count_per_seed", -1)),
        250,
        "test patient count",
    )
    _require_equal(
        int(
            config.get("formal", {}).get(
                "minimum_evaluable_patients_per_endpoint_per_seed",
                -1,
            )
        ),
        20,
        "minimum evaluable patients per test endpoint and seed",
    )
    return {
        "candidate_path_count": len(declared_ids),
        "validation_patient_count": 125,
        "test_patient_count": 250,
    }


def _validated_patients(
    patient_ids: Sequence[str],
    *,
    expected_count: int,
    name: str,
) -> list[str]:
    patients = list(patient_ids)
    if len(patients) != expected_count or len(set(patients)) != expected_count:
        raise ValueError(
            f"{name} patient registry must contain {expected_count} unique patients"
        )
    if patients != sorted(patients) or any(not value for value in patients):
        raise ValueError(f"{name} patient registry must be sorted and nonempty")
    return patients


def _validated_job_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    observer: bool,
) -> list[dict[str, Any]]:
    records = [dict(row) for row in rows]
    identities = {
        (str(row.get("model")), int(row.get("model_seed", -1)))
        for row in records
    }
    expected = {(EXPECTED_MODEL, seed) for seed in EXPECTED_MODEL_SEEDS}
    if identities != expected or len(records) != len(expected):
        raise ValueError("registered model/observer job matrix is incomplete")
    for row in records:
        required = "observer_bundle_sha256" if observer else "checkpoint_sha256"
        if not row.get(required):
            raise ValueError(f"registered job lacks {required}")
        if observer and row.get("status") != "PASS":
            raise ValueError("observer bundle did not pass admission")
    return sorted(records, key=lambda row: int(row["model_seed"]))


def build_protocol_lock_payload(
    *,
    configuration: Mapping[str, Any],
    configuration_sha256: str,
    source_identity: Mapping[str, Any],
    candidate_registration: Mapping[str, Any],
    candidate_registration_sha256: str,
    validation_asset_hashes: Mapping[str, str],
    model_jobs: Sequence[Mapping[str, Any]],
    observer_jobs: Sequence[Mapping[str, Any]],
    validation_patient_ids: Sequence[str],
    test_patient_ids: Sequence[str],
    data_inventory: Mapping[str, str],
) -> dict[str, Any]:
    validate_registered_configuration(configuration)
    if source_identity.get("code_dirty") is not False:
        raise ValueError("formal V8 protocol requires a clean source tree")
    if not source_identity.get("commit") or not source_identity.get(
        "source_tree_sha256"
    ):
        raise ValueError("formal V8 protocol requires commit and source tree hashes")
    validation_patients = _validated_patients(
        validation_patient_ids,
        expected_count=125,
        name="validation",
    )
    test_patients = _validated_patients(
        test_patient_ids,
        expected_count=250,
        name="test",
    )
    models = _validated_job_records(model_jobs, observer=False)
    observers = _validated_job_records(observer_jobs, observer=True)
    registration = dict(candidate_registration)
    registration_status = registration.get("status")
    if registration_status not in {
        "REGISTERED_TRANSUNET_CANDIDATE",
        "NO_REGISTERED_TRANSUNET_CANDIDATE",
    }:
        raise ValueError("candidate registration has an invalid status")
    registered_candidate = registration.get("candidate")
    authorized = registration_status == "REGISTERED_TRANSUNET_CANDIDATE"
    if authorized:
        if registration.get("test_intervention_authorized") is not True:
            raise ValueError("registered candidate does not authorize test intervention")
        if not isinstance(registered_candidate, dict):
            raise ValueError("registered candidate payload is missing")
        declared = {
            path.path_id: path.to_dict()
            for path in declared_transunet_decoder_paths()
        }
        path_id = str(registered_candidate.get("path_id"))
        if path_id not in declared or registered_candidate.get("path") != declared[path_id]:
            raise ValueError("registered candidate differs from the declared graph path")
        status = "LOCKED_BEFORE_FIRST_TEST_INTERVENTION"
    else:
        if registration.get("test_intervention_authorized") is not False:
            raise ValueError("no-candidate status must not authorize intervention")
        if registered_candidate is not None:
            raise ValueError("no-candidate status must not contain a candidate")
        status = "NO_REGISTERED_TRANSUNET_CANDIDATE"
    if not configuration_sha256 or not candidate_registration_sha256:
        raise ValueError("configuration and candidate hashes are required")
    if not validation_asset_hashes or not all(validation_asset_hashes.values()):
        raise ValueError("validation asset hashes are incomplete")
    if not data_inventory or not all(data_inventory.values()):
        raise ValueError("data inventory hashes are incomplete")
    return {
        "status": status,
        "test_intervention_authorized": authorized,
        "candidate": registered_candidate,
        "candidate_registration": registration,
        "candidate_registration_sha256": str(candidate_registration_sha256),
        "validation_asset_hashes": dict(validation_asset_hashes),
        "configuration_sha256": str(configuration_sha256),
        "source_identity": dict(source_identity),
        "data_inventory": dict(data_inventory),
        "registered_model_jobs": models,
        "registered_observer_jobs": observers,
        "validation_patient_ids": validation_patients,
        "validation_patient_registry_sha256": sha256_named_values(
            {str(index): value for index, value in enumerate(validation_patients)}
        ),
        "test_patient_ids": test_patients,
        "test_patient_registry_sha256": sha256_named_values(
            {str(index): value for index, value in enumerate(test_patients)}
        ),
        "candidate_may_not_change_after_lock": True,
        "thresholds_may_not_change_after_lock": True,
        "test_is_single_confirmation_run": True,
        "configuration": dict(configuration),
    }


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
    existing_outputs = (
        list(formal_output_root.glob("**/patient_results/*.json"))
        if formal_output_root.exists()
        else []
    )
    if existing_outputs:
        raise ValueError("test intervention outputs already exist before protocol lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def load_and_validate_protocol_lock(
    path: Path,
    *,
    active_configuration_sha256: str,
    active_source_tree_sha256: str,
    patient_ids: Sequence[str],
    model_seed: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"V8 protocol lock is missing: {path}")
    lock = _load_json(path)
    if lock.get("status") != "LOCKED_BEFORE_FIRST_TEST_INTERVENTION" or lock.get(
        "test_intervention_authorized"
    ) is not True:
        raise ValueError("V8 protocol does not authorize test intervention")
    if lock.get("configuration_sha256") != str(active_configuration_sha256):
        raise ValueError("active configuration SHA-256 differs from the V8 lock")
    if lock.get("source_identity", {}).get("source_tree_sha256") != str(
        active_source_tree_sha256
    ):
        raise ValueError("active source tree SHA-256 differs from the V8 lock")
    registered_patients = lock.get("test_patient_ids", [])
    if list(patient_ids) != registered_patients:
        raise ValueError("formal test patient registry differs from the V8 lock")
    registered_seeds = {
        int(row["model_seed"]) for row in lock.get("registered_model_jobs", [])
    }
    if int(model_seed) not in registered_seeds:
        raise ValueError(f"model seed {model_seed} is absent from the V8 lock")
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
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    matrix = _load_yaml(workspace / str(config["model_matrix"]))
    output = []
    for job in matrix.get("jobs", []):
        if str(job.get("model")) != EXPECTED_MODEL or int(job.get("seed", -1)) not in EXPECTED_MODEL_SEEDS:
            continue
        checkpoint = _resolve_checkpoint(workspace, asset_root, job["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        output.append(
            {
                "model": EXPECTED_MODEL,
                "model_seed": int(job["seed"]),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
    return _validated_job_records(output, observer=False)


def _bundle_sha256(root: Path, files: Sequence[Path]) -> str:
    hashes = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(files)
    }
    return sha256_named_values(hashes)


def _collect_observer_jobs(
    workspace: Path,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    observer_root = workspace / str(config["observer_root"])
    output = []
    audit_names = (
        "observer_control_selectivity.parquet",
        "observer_quality_by_node.parquet",
        "observer_restart_agreement.parquet",
    )
    for model_seed in EXPECTED_MODEL_SEEDS:
        directory = observer_root / EXPECTED_MODEL / f"seed_{model_seed}"
        required = [directory / name for name in audit_names]
        required.extend(
            directory / "observers" / node / "real" / f"seed_{observer_seed}.pt"
            for node in EXPECTED_NODES
            for observer_seed in EXPECTED_OBSERVER_SEEDS
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
        output.append(
            {
                "model": EXPECTED_MODEL,
                "model_seed": model_seed,
                "status": "PASS",
                "observer_file_count": len(required),
                "observer_bundle_sha256": _bundle_sha256(directory, required),
            }
        )
    return _validated_job_records(output, observer=True)


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


def _audit_trace_registry(
    workspace: Path,
    config: Mapping[str, Any],
    *,
    patient_ids: Sequence[str],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for model_seed in EXPECTED_MODEL_SEEDS:
        worker_root = (
            workspace
            / str(config["validation"]["worker_root"])
            / f"seed_{model_seed}"
        )
        status_path = worker_root / "transfer_status.json"
        manifest_path = (
            worker_root
            / EXPECTED_MODEL
            / f"seed_{model_seed}"
            / "job_manifest.json"
        )
        for path in (status_path, manifest_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        status = _load_json(status_path)
        if status.get("status") != "COMPLETE_TRANSFER":
            raise ValueError(f"validation transfer did not complete for seed {model_seed}")
        traces = (
            worker_root
            / EXPECTED_MODEL
            / f"seed_{model_seed}"
            / "val"
            / "case_traces"
        )
        observed = sorted(path.stem for path in traces.glob("*.npz"))
        if observed != list(patient_ids):
            raise ValueError(f"validation trace registry differs for seed {model_seed}")
        hashes[f"seed_{model_seed}_status"] = sha256_file(status_path)
        hashes[f"seed_{model_seed}_manifest"] = sha256_file(manifest_path)
        hashes[f"seed_{model_seed}_patient_registry"] = sha256_named_values(
            {str(index): value for index, value in enumerate(observed)}
        )
    return hashes


def _audit_test_trace_registry(
    workspace: Path,
    config: Mapping[str, Any],
    *,
    patient_ids: Sequence[str],
) -> dict[str, str]:
    root = workspace / str(config["test_trace_root"])
    hashes: dict[str, str] = {}
    for model_seed in EXPECTED_MODEL_SEEDS:
        job_root = root / EXPECTED_MODEL / f"seed_{model_seed}"
        manifest_path = job_root / "job_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = _load_json(manifest_path)
        if tuple(manifest.get("admitted_nodes", ())) != EXPECTED_NODES:
            raise ValueError(f"test trace lacks the full node path for seed {model_seed}")
        traces = job_root / "test" / "case_traces"
        observed = sorted(path.stem for path in traces.glob("*.npz"))
        if observed != list(patient_ids):
            raise ValueError(f"test trace registry differs for seed {model_seed}")
        hashes[f"seed_{model_seed}_manifest"] = sha256_file(manifest_path)
        hashes[f"seed_{model_seed}_patient_registry"] = sha256_named_values(
            {str(index): value for index, value in enumerate(observed)}
        )
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lock V8 before the first TransUNet test intervention."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/v8_transunet_mechanism_replication.yaml"
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
    candidate_root = workspace / str(config["validation"]["candidate_root"])
    candidate_path = candidate_root / "candidate_registration.json"
    candidate_files = {
        "candidate_registration": candidate_path,
        "candidate_patient_scores": (
            candidate_root / "validation_candidate_patient_scores.parquet"
        ),
        "candidate_statistics": (
            candidate_root / "validation_candidate_statistics.parquet"
        ),
        "candidate_coverage": (
            candidate_root / "validation_candidate_coverage.parquet"
        ),
    }
    for path in candidate_files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    registration = _load_json(candidate_path)
    validation_patients, test_patients, data_inventory = _registered_patients(
        workspace,
        asset_root,
    )
    validation_asset_hashes = {
        name: sha256_file(path) for name, path in candidate_files.items()
    }
    validation_asset_hashes.update(
        _audit_trace_registry(
            workspace,
            config,
            patient_ids=validation_patients,
        )
    )
    validation_asset_hashes.update(
        {
            f"test_trace_{name}": value
            for name, value in _audit_test_trace_registry(
                workspace,
                config,
                patient_ids=test_patients,
            ).items()
        }
    )
    payload = build_protocol_lock_payload(
        configuration=config,
        configuration_sha256=sha256_file(config_path),
        source_identity=resolve_source_identity(workspace),
        candidate_registration=registration,
        candidate_registration_sha256=sha256_file(candidate_path),
        validation_asset_hashes=validation_asset_hashes,
        model_jobs=_collect_model_jobs(workspace, asset_root, config),
        observer_jobs=_collect_observer_jobs(workspace, config),
        validation_patient_ids=validation_patients,
        test_patient_ids=test_patients,
        data_inventory=data_inventory,
    )
    lock_path = workspace / str(config["protocol_lock"])
    formal_root = workspace / str(config["formal_output_root"])
    write_protocol_lock(lock_path, payload, formal_output_root=formal_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
