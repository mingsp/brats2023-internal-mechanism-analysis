import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from scripts.lock_v8_transunet_protocol import (
    build_protocol_lock_payload,
    load_and_validate_protocol_lock,
    validate_registered_configuration,
    write_protocol_lock,
)


def _configuration() -> dict:
    path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "experiments"
        / "v8_transunet_mechanism_replication.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _patients(count: int, suffix: str) -> list[str]:
    return [f"BraTS-GLI-{index:05d}-{suffix}" for index in range(count)]


def _registration(status: str = "REGISTERED_TRANSUNET_CANDIDATE") -> dict:
    return {
        "status": status,
        "test_intervention_authorized": status
        == "REGISTERED_TRANSUNET_CANDIDATE",
        "candidate": (
            {
                "path_id": "skip_down2_to_up2",
                "minimum_seed_effect": 1.1,
                "path": {
                    "path_id": "skip_down2_to_up2",
                    "source_node": "down2",
                    "receiver_node": "up2",
                    "source_index": 1,
                    "receiver_index": 5,
                    "source_module_path": (
                        "transformer.embeddings.hybrid_model.feature_taps.1"
                    ),
                    "receiver_module_path": "decoder.blocks.1",
                    "argument_index": 1,
                    "argument_name": "skip",
                    "topology_order": 2,
                },
            }
            if status == "REGISTERED_TRANSUNET_CANDIDATE"
            else None
        ),
    }


def _payload(registration: dict | None = None) -> dict:
    return build_protocol_lock_payload(
        configuration=_configuration(),
        configuration_sha256="config-sha",
        source_identity={
            "commit": "abc123",
            "source_tree_sha256": "source-sha",
            "code_dirty": False,
        },
        candidate_registration=registration or _registration(),
        candidate_registration_sha256="candidate-sha",
        validation_asset_hashes={"statistics": "statistics-sha"},
        model_jobs=[
            {
                "model": "transunet_r50_vit_b16",
                "model_seed": seed,
                "checkpoint_sha256": f"checkpoint-{seed}",
            }
            for seed in (42, 123, 3407)
        ],
        observer_jobs=[
            {
                "model": "transunet_r50_vit_b16",
                "model_seed": seed,
                "status": "PASS",
                "observer_bundle_sha256": f"observer-{seed}",
            }
            for seed in (42, 123, 3407)
        ],
        validation_patient_ids=_patients(125, "000"),
        test_patient_ids=_patients(250, "001"),
        data_inventory={"split_sha256": "split-sha"},
    )


def test_checked_in_configuration_matches_registered_design():
    audit = validate_registered_configuration(_configuration())

    assert audit["candidate_path_count"] == 4
    assert audit["validation_patient_count"] == 125
    assert audit["test_patient_count"] == 250


def test_protocol_lock_is_content_addressed_and_precedes_test_outputs(tmp_path):
    lock_path = tmp_path / "v8_protocol_lock.json"
    formal_root = tmp_path / "formal_test"
    payload = _payload()

    write_protocol_lock(lock_path, payload, formal_output_root=formal_root)
    first = lock_path.read_text(encoding="utf-8")
    write_protocol_lock(lock_path, payload, formal_output_root=formal_root)

    assert lock_path.read_text(encoding="utf-8") == first
    assert json.loads(first)["status"] == "LOCKED_BEFORE_FIRST_TEST_INTERVENTION"


def test_protocol_lock_rejects_changed_candidate_or_existing_patient_output(tmp_path):
    lock_path = tmp_path / "v8_protocol_lock.json"
    formal_root = tmp_path / "formal_test"
    payload = _payload()
    write_protocol_lock(lock_path, payload, formal_output_root=formal_root)
    changed = deepcopy(payload)
    changed["candidate"]["path_id"] = "bottleneck_to_up1"

    with pytest.raises(ValueError, match="existing protocol lock differs"):
        write_protocol_lock(lock_path, changed, formal_output_root=formal_root)

    second_lock = tmp_path / "second_lock.json"
    patient_output = formal_root / "seed_42" / "patient_results" / "p.json"
    patient_output.parent.mkdir(parents=True)
    patient_output.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="test intervention outputs already exist"):
        write_protocol_lock(second_lock, payload, formal_output_root=formal_root)


def test_no_candidate_writes_terminal_status_without_authorizing_test(tmp_path):
    payload = _payload(_registration("NO_REGISTERED_TRANSUNET_CANDIDATE"))
    lock_path = tmp_path / "v8_protocol_lock.json"
    write_protocol_lock(lock_path, payload, formal_output_root=tmp_path / "formal")

    assert payload["status"] == "NO_REGISTERED_TRANSUNET_CANDIDATE"
    assert not payload["test_intervention_authorized"]
    with pytest.raises(ValueError, match="does not authorize test intervention"):
        load_and_validate_protocol_lock(
            lock_path,
            active_configuration_sha256="config-sha",
            active_source_tree_sha256="source-sha",
            patient_ids=_patients(250, "001"),
            model_seed=42,
        )


def test_formal_lock_rejects_source_or_patient_change(tmp_path):
    lock_path = tmp_path / "v8_protocol_lock.json"
    payload = _payload()
    write_protocol_lock(lock_path, payload, formal_output_root=tmp_path / "formal")

    with pytest.raises(ValueError, match="source tree SHA-256"):
        load_and_validate_protocol_lock(
            lock_path,
            active_configuration_sha256="config-sha",
            active_source_tree_sha256="changed",
            patient_ids=_patients(250, "001"),
            model_seed=42,
        )
    changed_patients = _patients(249, "001") + ["different"]
    with pytest.raises(ValueError, match="patient registry"):
        load_and_validate_protocol_lock(
            lock_path,
            active_configuration_sha256="config-sha",
            active_source_tree_sha256="source-sha",
            patient_ids=changed_patients,
            model_seed=42,
        )
