from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from pptt.interventions.transunet_paths import declared_transunet_decoder_paths
from pptt.validation.v9_followup import derive_remaining_path_naive_cohort
from scripts.lock_v8_transunet_protocol import (
    _collect_model_jobs,
    _collect_observer_jobs,
    resolve_source_identity,
    sha256_file,
    sha256_named_values,
    write_protocol_lock,
)
from scripts.lock_v9_transunet_protocol import build_v9_protocol_lock_payload


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


def validate_v10_configuration(config: Mapping[str, Any]) -> None:
    if str(config.get("experiment_id")) != "v10":
        raise ValueError("V10 experiment id differs")
    if str(config.get("model")) != EXPECTED_MODEL:
        raise ValueError("V10 model differs")
    if tuple(int(value) for value in config.get("model_seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("V10 model seeds differ")
    declared = tuple(path.path_id for path in declared_transunet_decoder_paths())
    if tuple(config.get("candidate_paths", ())) != declared:
        raise ValueError("V10 declared paths differ")
    eligibility = config.get("eligibility", {})
    if (
        int(eligibility.get("minimum_target_pixels_on_selected_slice", -1)) != 32
        or int(eligibility.get("minimum_target_feature_positions", -1)) != 8
        or int(eligibility.get("minimum_control_feature_positions", -1)) != 8
    ):
        raise ValueError("V10 spatial eligibility thresholds differ")
    statistics = config.get("statistics", {})
    if abs(float(statistics.get("alpha", -1)) - (0.05 / 3.0)) > 1.0e-15:
        raise ValueError("V10 alpha differs from three-round spending")
    if float(statistics.get("minimum_standardized_effect", -1)) != 0.8:
        raise ValueError("V10 standardized-effect threshold differs")
    if float(statistics.get("minimum_patient_monotonic_fraction", -1)) != 0.8:
        raise ValueError("V10 patient dose threshold differs")
    if str(statistics.get("multiplicity")) != (
        "holm_global_with_three_round_alpha_spending"
    ):
        raise ValueError("V10 multiplicity rule differs")
    formal = config.get("formal", {})
    if int(formal.get("expected_patient_count_per_seed", -1)) != 224:
        raise ValueError("V10 formal patient count differs")
    if int(formal.get("minimum_evaluable_patients_per_endpoint_per_seed", -1)) != 20:
        raise ValueError("V10 endpoint patient threshold differs")


def _split_patients(workspace: Path, asset_root: Path) -> tuple[list[str], Path]:
    data_config = _load_yaml(workspace / "configs/data/brats2023_2d.yaml")
    split_path = asset_root / str(data_config["split_file"])
    split = _load_json(split_path)
    patients = sorted(str(value) for value in split.get("test", ()))
    if len(patients) != 250 or len(set(patients)) != 250:
        raise ValueError("test split does not contain 250 unique patients")
    return patients, split_path


def _selected_trace_hash(
    workspace: Path,
    config: Mapping[str, Any],
    patient_ids: tuple[str, ...],
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
        description="Lock V10 before the first remaining-cohort path intervention."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/v10_transunet_path_confirmation.yaml"),
    )
    parser.add_argument("--asset-root", type=Path, required=True)
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    config_path = args.config if args.config.is_absolute() else workspace / args.config
    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    validate_v10_configuration(config)
    asset_root = args.asset_root.resolve()

    candidate_root = workspace / str(config["validation"]["candidate_root"])
    candidate_path = candidate_root / "candidate_registration.json"
    registration = _load_json(candidate_path)
    candidate = registration.get("candidate")
    if registration.get("status") != "REGISTERED_FEASIBLE_TRANSUNET_CANDIDATE":
        raise ValueError("V10 candidate did not pass V9 validation gates")
    declared = {
        path.path_id: path.to_dict() for path in declared_transunet_decoder_paths()
    }
    if not isinstance(candidate, dict) or candidate.get("path") != declared.get(
        str(candidate.get("path_id"))
    ):
        raise ValueError("V10 candidate differs from the declared graph")

    v9_lock_path = workspace / str(config["v9_protocol_lock"])
    v9_lock = _load_json(v9_lock_path)
    if v9_lock.get("candidate", {}).get("path") != candidate.get("path"):
        raise ValueError("V10 candidate differs from the locked V9 path")
    all_patients, split_path = _split_patients(workspace, asset_root)
    expected = int(config["formal"]["expected_patient_count_per_seed"])
    patients = derive_remaining_path_naive_cohort(
        all_patients,
        tuple(str(value) for value in v9_lock.get("test_patient_ids", ())),
        expected_remaining_count=expected,
    )
    v9_formal = workspace / str(config["v9_formal_root"]) / EXPECTED_MODEL
    for seed in EXPECTED_SEEDS:
        observed = {
            path.stem
            for path in (v9_formal / f"seed_{seed}" / "patient_results").glob("*.json")
        }
        if observed != set(v9_lock["test_patient_ids"]):
            raise ValueError(f"V9 formal patient registry differs for seed {seed}")
        if observed & set(patients):
            raise ValueError("V10 cohort contains a prior same-path intervention")

    v8_status_path = workspace / str(config["v8_status"])
    v9_status_path = workspace / str(config["v9_status"])
    validation_files = [
        candidate_path,
        candidate_root / "validation_feasibility_patient_rows.parquet",
        candidate_root / "validation_feasibility_summary.parquet",
    ]
    for path in (*validation_files, v8_status_path, v9_status_path, v9_lock_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    validation_hashes = {
        path.relative_to(workspace).as_posix(): sha256_file(path)
        for path in validation_files
    }
    prior_audit = {
        "v8_status_sha256": sha256_file(v8_status_path),
        "v9_status_sha256": sha256_file(v9_status_path),
        "v9_protocol_lock_sha256": sha256_file(v9_lock_path),
        "v9_patient_registry_sha256": str(v9_lock["test_patient_registry_sha256"]),
    }
    data_inventory = {
        "data_config_sha256": sha256_file(workspace / "configs/data/brats2023_2d.yaml"),
        "split_sha256": sha256_file(split_path),
        "selected_test_trace_sha256": _selected_trace_hash(
            workspace, config, patients
        ),
    }
    payload = build_v9_protocol_lock_payload(
        configuration=config,
        configuration_sha256=sha256_file(config_path),
        source_identity=resolve_source_identity(workspace),
        candidate_registration=registration,
        candidate_registration_sha256=sha256_file(candidate_path),
        validation_asset_hashes=validation_hashes,
        model_jobs=_collect_model_jobs(workspace, asset_root, config),
        observer_jobs=_collect_observer_jobs(workspace, config),
        test_patient_ids=patients,
        expected_patient_count=expected,
        data_inventory=data_inventory,
        v8_audit=prior_audit,
    )
    payload.update(
        {
            "sequential_confirmation_round": 3,
            "prior_v9_result_may_not_change": True,
            "same_path_intervention_outcomes_previously_read": False,
        }
    )
    lock_path = workspace / str(config["protocol_lock"])
    formal_root = workspace / str(config["formal_output_root"])
    write_protocol_lock(lock_path, payload, formal_output_root=formal_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
