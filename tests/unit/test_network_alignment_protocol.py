from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from scripts.lock_v7_network_alignment_protocol import (
    build_protocol_lock_payload,
    load_and_validate_formal_lock,
    validate_registered_configuration,
    validate_v4_status,
    validate_v6_scope,
    validate_patient_registry,
    write_protocol_lock,
)


def _configuration() -> dict:
    return {
        "name": "v7_network_process_intervention_alignment",
        "model_matrix": "manifests/model_matrix.yaml",
        "observer_root": "results/v1_observers",
        "v3_root": "results/v3_unet_pair",
        "output_root": "results/v7_network_process_intervention_alignment",
        "models": ["unet_baseline", "unet_noskip"],
        "model_seeds": [42, 123, 3407],
        "observer_seeds": [17, 29, 43],
        "nodes": [
            "down1",
            "down2",
            "down3",
            "down4",
            "up1",
            "up2",
            "up3",
            "up4",
        ],
        "primary_split": "test",
        "truth_classes": [1, 2, 3],
        "process_event": "output_anchored_persistent_correction",
        "eligibility": {
            "minimum_union_pixels_on_selected_slice": 32,
            "minimum_transition_pixels": 8,
            "minimum_target_feature_positions": 4,
            "minimum_control_feature_positions": 4,
            "minimum_patients_per_transition": 30,
            "minimum_aggregate_pixels_per_transition": 2048,
        },
        "intervention": {
            "root_node": "down1",
            "operator": "spatial_roll_module_output",
            "input_equivalent_shift_yx": [8, 8],
            "restore_nodes": [
                "down2",
                "down3",
                "down4",
                "up1",
                "up2",
                "up3",
                "up4",
            ],
            "logit_tolerance": 1.0e-6,
        },
        "matching": {
            "same_dominant_truth_class": True,
            "boundary_bin_edges": [1.5, 3.5, 7.5],
            "activation_norm_quantile_bins": 4,
            "without_replacement": True,
        },
        "statistics": {
            "bootstrap_iterations": 10000,
            "bootstrap_seed": 20260717,
            "alpha": 0.05,
            "multiplicity": "holm_global_macro_micro",
            "minimum_supported_seeds": 2,
            "minimum_evaluable_transitions": 5,
            "minimum_positive_receiving_transitions": 4,
        },
        "protocol": {
            "lock_before_test_intervention": True,
            "thresholds_may_not_change_after_lock": True,
            "v4_must_remain_not_triggered": True,
            "v6_must_remain_candidate_specific": True,
        },
    }


def _patients() -> list[str]:
    return [f"BraTS-GLI-{index:05d}-000" for index in range(250)]


def _v6_lock() -> dict:
    return {
        "status": "LOCKED_BEFORE_FIRST_INTERVENTION",
        "global_two_metric_gate_passed": False,
        "global_two_metric_gate_must_remain_reported": True,
        "candidate_identity": {
            "transition": "up1->up2",
            "transition_index": 4,
            "module": "up2",
            "argument_index": 1,
            "source": "down2",
        },
    }


def _lock_payload() -> dict:
    return build_protocol_lock_payload(
        configuration=_configuration(),
        configuration_sha256="config-sha",
        source_identity={
            "commit": "abc123",
            "source_tree_sha256": "source-sha",
            "code_dirty": False,
        },
        data_inventory={
            "path": "processed_2d/splits/brats2023_gli_seed42.json",
            "sha256": "data-sha",
        },
        model_jobs=[
            {
                "model": model,
                "model_seed": seed,
                "checkpoint_sha256": f"{model}-{seed}-sha",
            }
            for model in ("unet_baseline", "unet_noskip")
            for seed in (42, 123, 3407)
        ],
        observer_jobs=[
            {"model": model, "model_seed": seed, "status": "PASS"}
            for model in ("unet_baseline", "unet_noskip")
            for seed in (42, 123, 3407)
        ],
        patient_ids=_patients(),
        v4_status={"triggered": False},
        v4_sha256="v4-sha",
        v6_lock=_v6_lock(),
        v6_sha256="v6-sha",
    )


def test_registered_configuration_accepts_only_locked_network_wide_design():
    result = validate_registered_configuration(_configuration())

    assert result["transitions"] == [
        "down1->down2",
        "down2->down3",
        "down3->down4",
        "down4->up1",
        "up1->up2",
        "up2->up3",
        "up3->up4",
    ]
    assert result["matrix_shape"] == [7, 7]


def test_checked_in_configuration_matches_registered_design():
    config_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "experiments"
        / "v7_network_process_intervention_alignment.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    validate_registered_configuration(config)


def test_protocol_rejects_model_or_seed_matrix_change():
    config = _configuration()
    config["model_seeds"] = [42, 123]

    with pytest.raises(ValueError, match="model_seeds"):
        validate_registered_configuration(config)


def test_protocol_rejects_node_order_change():
    config = _configuration()
    config["nodes"][2], config["nodes"][3] = config["nodes"][3], config["nodes"][2]

    with pytest.raises(ValueError, match="nodes"):
        validate_registered_configuration(config)


def test_protocol_rejects_v6_candidate_path_inserted_into_v7():
    config = _configuration()
    config["target"] = {
        "transition": "up1->up2",
        "module": "up2",
        "source": "down2",
    }

    with pytest.raises(ValueError, match="candidate-specific"):
        validate_registered_configuration(config)


def test_protocol_rejects_truncated_or_reordered_patient_registry():
    patients = _patients()

    with pytest.raises(ValueError, match="exactly 250"):
        validate_patient_registry(patients[:-1], expected=patients)
    with pytest.raises(ValueError, match="ordered registry"):
        validate_patient_registry(list(reversed(patients)), expected=patients)


def test_protocol_rejects_rewritten_v4_status():
    with pytest.raises(ValueError, match="V4 must remain NOT_TRIGGERED"):
        validate_v4_status({"triggered": True})


def test_protocol_keeps_v6_candidate_specific_and_not_global():
    validate_v6_scope(_v6_lock())
    globalized = deepcopy(_v6_lock())
    globalized["global_two_metric_gate_passed"] = True

    with pytest.raises(ValueError, match="global network-wide evidence"):
        validate_v6_scope(globalized)


def test_protocol_lock_rejects_polluted_formal_output_directory(tmp_path: Path):
    output_root = tmp_path / "formal"
    polluted = output_root / "unet_baseline" / "seed_42" / "patient_results"
    polluted.mkdir(parents=True)
    (polluted / "patient.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="formal output directory is not empty"):
        write_protocol_lock(output_root / "v7_protocol_lock.json", _lock_payload())


def test_formal_run_rejects_missing_lock(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="formal protocol lock is missing"):
        load_and_validate_formal_lock(
            tmp_path / "missing.json",
            active_configuration_sha256="config-sha",
            active_source_tree_sha256="source-sha",
            patient_ids=_patients(),
        )


def test_formal_run_rejects_configuration_or_source_hash_change(tmp_path: Path):
    lock_path = tmp_path / "v7_protocol_lock.json"
    lock_path.write_text(json.dumps(_lock_payload()), encoding="utf-8")

    with pytest.raises(ValueError, match="configuration SHA-256"):
        load_and_validate_formal_lock(
            lock_path,
            active_configuration_sha256="changed-config",
            active_source_tree_sha256="source-sha",
            patient_ids=_patients(),
        )
    with pytest.raises(ValueError, match="source tree SHA-256"):
        load_and_validate_formal_lock(
            lock_path,
            active_configuration_sha256="config-sha",
            active_source_tree_sha256="changed-source",
            patient_ids=_patients(),
        )


def test_formal_run_rejects_patient_registry_change_after_lock(tmp_path: Path):
    lock_path = tmp_path / "v7_protocol_lock.json"
    lock_path.write_text(json.dumps(_lock_payload()), encoding="utf-8")

    with pytest.raises(ValueError, match="patient registry"):
        load_and_validate_formal_lock(
            lock_path,
            active_configuration_sha256="config-sha",
            active_source_tree_sha256="source-sha",
            patient_ids=_patients()[:-1],
        )
