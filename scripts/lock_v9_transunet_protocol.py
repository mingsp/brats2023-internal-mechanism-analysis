from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from pptt.interventions.transunet_paths import declared_transunet_decoder_paths
from pptt.validation.v9_followup import derive_common_unintervened_cohort
from scripts.lock_v8_transunet_protocol import (
    _collect_model_jobs,
    _collect_observer_jobs,
    _validated_job_records,
    resolve_source_identity,
    sha256_file,
    sha256_named_values,
    write_protocol_lock,
)


EXPECTED_MODEL = "transunet_r50_vit_b16"
EXPECTED_SEEDS = (42, 123, 3407)


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


def _validated_patients(
    patient_ids: Sequence[str], *, expected_count: int
) -> list[str]:
    patients = [str(value) for value in patient_ids]
    if (
        len(patients) != int(expected_count)
        or len(set(patients)) != int(expected_count)
        or patients != sorted(patients)
        or any(not value for value in patients)
    ):
        raise ValueError(
            f"V9 patient registry must contain {expected_count} sorted unique ids"
        )
    return patients


def validate_v9_configuration(config: Mapping[str, Any]) -> None:
    if str(config.get("experiment_id")) != "v9":
        raise ValueError("V9 experiment id differs")
    if str(config.get("model")) != EXPECTED_MODEL:
        raise ValueError("V9 model differs")
    if tuple(int(value) for value in config.get("model_seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("V9 model seeds differ")
    declared = tuple(path.path_id for path in declared_transunet_decoder_paths())
    if tuple(config.get("candidate_paths", ())) != declared:
        raise ValueError("V9 candidate paths differ from the declared graph")
    eligibility = config.get("eligibility", {})
    expected_eligibility = {
        "minimum_target_pixels_on_selected_slice": 32,
        "minimum_target_feature_positions": 8,
        "minimum_control_feature_positions": 8,
        "minimum_candidate_patients_per_seed": 20,
        "minimum_candidate_target_pixels_per_seed": 512,
        "minimum_feasible_patients_per_seed": 20,
    }
    if dict(eligibility) != expected_eligibility:
        raise ValueError("V9 eligibility thresholds differ")
    statistics = config.get("statistics", {})
    if float(statistics.get("alpha", -1)) != 0.025:
        raise ValueError("V9 alpha differs from the registered alpha spending")
    if float(statistics.get("minimum_standardized_effect", -1)) != 0.8:
        raise ValueError("V9 standardized-effect threshold differs")
    if int(statistics.get("minimum_supported_seeds", -1)) != 2:
        raise ValueError("V9 seed support threshold differs")
    if str(statistics.get("multiplicity")) != (
        "holm_global_with_two_round_alpha_spending"
    ):
        raise ValueError("V9 multiplicity rule differs")
    formal = config.get("formal", {})
    if int(formal.get("expected_patient_count_per_seed", -1)) != 26:
        raise ValueError("V9 formal patient count differs")
    if int(formal.get("minimum_evaluable_patients_per_endpoint_per_seed", -1)) != 20:
        raise ValueError("V9 endpoint patient threshold differs")


def build_v9_protocol_lock_payload(
    *,
    configuration: Mapping[str, Any],
    configuration_sha256: str,
    source_identity: Mapping[str, Any],
    candidate_registration: Mapping[str, Any],
    candidate_registration_sha256: str,
    validation_asset_hashes: Mapping[str, str],
    model_jobs: Sequence[Mapping[str, Any]],
    observer_jobs: Sequence[Mapping[str, Any]],
    test_patient_ids: Sequence[str],
    expected_patient_count: int,
    data_inventory: Mapping[str, str],
    v8_audit: Mapping[str, str],
) -> dict[str, Any]:
    if source_identity.get("code_dirty") is not False:
        raise ValueError("formal V9 protocol requires a clean source tree")
    if not source_identity.get("commit") or not source_identity.get(
        "source_tree_sha256"
    ):
        raise ValueError("formal V9 protocol requires source identity hashes")
    registration = dict(candidate_registration)
    if registration.get("status") != "REGISTERED_FEASIBLE_TRANSUNET_CANDIDATE":
        raise ValueError("V9 candidate did not pass validation feasibility")
    if registration.get("test_intervention_authorized") is not True:
        raise ValueError("V9 candidate does not authorize formal intervention")
    candidate = registration.get("candidate")
    if not isinstance(candidate, dict) or not candidate.get("path_id"):
        raise ValueError("V9 candidate payload is missing")
    patients = _validated_patients(
        test_patient_ids, expected_count=int(expected_patient_count)
    )
    if not configuration_sha256 or not candidate_registration_sha256:
        raise ValueError("V9 configuration and candidate hashes are required")
    if not validation_asset_hashes or not all(validation_asset_hashes.values()):
        raise ValueError("V9 validation asset hashes are incomplete")
    if not data_inventory or not all(data_inventory.values()):
        raise ValueError("V9 data inventory hashes are incomplete")
    if not v8_audit or not all(v8_audit.values()):
        raise ValueError("V8 nonintervention audit hashes are incomplete")
    return {
        "status": "LOCKED_BEFORE_FIRST_TEST_INTERVENTION",
        "test_intervention_authorized": True,
        "candidate": candidate,
        "candidate_registration": registration,
        "candidate_registration_sha256": candidate_registration_sha256,
        "validation_asset_hashes": dict(validation_asset_hashes),
        "configuration": dict(configuration),
        "configuration_sha256": configuration_sha256,
        "source_identity": dict(source_identity),
        "registered_model_jobs": _validated_job_records(model_jobs, observer=False),
        "registered_observer_jobs": _validated_job_records(
            observer_jobs, observer=True
        ),
        "test_patient_ids": patients,
        "test_patient_registry_sha256": sha256_named_values(
            {str(index): value for index, value in enumerate(patients)}
        ),
        "data_inventory": dict(data_inventory),
        "v8_nonintervention_audit": dict(v8_audit),
        "candidate_may_not_change_after_lock": True,
        "thresholds_may_not_change_after_lock": True,
        "test_is_single_confirmation_run": True,
        "prior_v8_result_may_not_change": True,
    }


def _v8_patient_rows(
    workspace: Path, config: Mapping[str, Any]
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, str]]:
    root = workspace / str(config["v8_formal_root"]) / EXPECTED_MODEL
    rows_by_seed: dict[int, list[dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    for seed in EXPECTED_SEEDS:
        directory = root / f"seed_{seed}" / "patient_results"
        paths = sorted(directory.glob("*.json"))
        if len(paths) != 250:
            raise ValueError(f"V8 seed {seed} patient registry is incomplete")
        rows_by_seed[seed] = [_load_json(path) for path in paths]
        hashes[f"seed_{seed}_patient_registry"] = sha256_named_values(
            {str(index): path.stem for index, path in enumerate(paths)}
        )
    return rows_by_seed, hashes


def _selected_trace_hash(
    workspace: Path,
    config: Mapping[str, Any],
    patient_ids: Sequence[str],
) -> str:
    root = workspace / str(config["test_trace_root"]) / EXPECTED_MODEL
    values: dict[str, str] = {}
    for seed in EXPECTED_SEEDS:
        for patient_id in patient_ids:
            path = root / f"seed_{seed}" / "test" / "case_traces" / f"{patient_id}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            values[f"seed_{seed}/{patient_id}"] = sha256_file(path)
    return sha256_named_values(values)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lock V9 before the first follow-up test intervention."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/v9_transunet_specificity.yaml"),
    )
    parser.add_argument("--asset-root", type=Path, required=True)
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    config_path = args.config if args.config.is_absolute() else workspace / args.config
    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    validate_v9_configuration(config)
    candidate_root = workspace / str(config["validation"]["candidate_root"])
    candidate_path = candidate_root / "candidate_registration.json"
    candidate_registration = _load_json(candidate_path)
    candidate = candidate_registration.get("candidate")
    declared = {
        path.path_id: path.to_dict() for path in declared_transunet_decoder_paths()
    }
    if not isinstance(candidate, dict) or candidate.get("path") != declared.get(
        str(candidate.get("path_id"))
    ):
        raise ValueError("V9 candidate differs from the declared graph")

    rows_by_seed, v8_hashes = _v8_patient_rows(workspace, config)
    expected_count = int(config["formal"]["expected_patient_count_per_seed"])
    patients = derive_common_unintervened_cohort(
        rows_by_seed,
        required_seeds=EXPECTED_SEEDS,
        minimum_patient_count=expected_count,
    )
    if len(patients) != expected_count:
        raise ValueError("V9 common unintervened cohort differs from the lock")
    selected_v8_hashes = {}
    for seed in EXPECTED_SEEDS:
        rows = {str(row["patient_id"]): row for row in rows_by_seed[seed]}
        selected_v8_hashes[f"seed_{seed}_selected_payloads"] = sha256_named_values(
            {
                patient_id: json.dumps(
                    rows[patient_id], sort_keys=True, separators=(",", ":")
                )
                for patient_id in patients
            }
        )
    v8_status_path = workspace / str(config["v8_status"])
    if not v8_status_path.is_file():
        raise FileNotFoundError(v8_status_path)
    v8_audit = {
        "v8_status_sha256": sha256_file(v8_status_path),
        **v8_hashes,
        **selected_v8_hashes,
    }

    validation_files = [
        candidate_path,
        candidate_root / "validation_feasibility_patient_rows.parquet",
        candidate_root / "validation_feasibility_summary.parquet",
    ]
    for seed in EXPECTED_SEEDS:
        seed_root = (
            workspace
            / str(config["validation"]["feasibility_root"])
            / f"seed_{seed}"
        )
        validation_files.extend(
            [
                seed_root / "patient_feasibility.parquet",
                seed_root / "feasibility_summary.parquet",
                seed_root / "feasibility_status.json",
            ]
        )
    for path in validation_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    validation_hashes = {
        path.relative_to(workspace).as_posix(): sha256_file(path)
        for path in validation_files
    }
    asset_root = args.asset_root.resolve()
    split_path = asset_root / "data" / "patient_splits_seed42.json"
    if not split_path.is_file():
        split_path = asset_root / "patient_splits_seed42.json"
    if not split_path.is_file():
        data_config = _load_yaml(workspace / "configs/data/brats2023_2d.yaml")
        split_path = asset_root / str(data_config["split_file"])
    data_inventory = {
        "data_config_sha256": sha256_file(workspace / "configs/data/brats2023_2d.yaml"),
        "split_sha256": sha256_file(split_path),
        "selected_test_trace_sha256": _selected_trace_hash(
            workspace, config, patients
        ),
    }
    source_identity = resolve_source_identity(workspace)
    payload = build_v9_protocol_lock_payload(
        configuration=config,
        configuration_sha256=sha256_file(config_path),
        source_identity=source_identity,
        candidate_registration=candidate_registration,
        candidate_registration_sha256=sha256_file(candidate_path),
        validation_asset_hashes=validation_hashes,
        model_jobs=_collect_model_jobs(workspace, asset_root, config),
        observer_jobs=_collect_observer_jobs(workspace, config),
        test_patient_ids=patients,
        expected_patient_count=expected_count,
        data_inventory=data_inventory,
        v8_audit=v8_audit,
    )
    lock_path = workspace / str(config["protocol_lock"])
    formal_root = workspace / str(config["formal_output_root"])
    write_protocol_lock(lock_path, payload, formal_output_root=formal_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
