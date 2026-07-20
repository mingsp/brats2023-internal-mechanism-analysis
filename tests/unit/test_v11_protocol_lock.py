from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from scripts.lock_v11_causal_abstraction_protocol import (
    build_v11_protocol_lock_payload,
    load_and_validate_v11_protocol_lock,
    validate_v11_configuration,
    write_protocol_lock,
)


def _configuration() -> dict:
    path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "experiments"
        / "v11_causal_abstraction.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _patients(count: int, suffix: str) -> list[str]:
    return [f"BraTS-GLI-{index:05d}-{suffix}" for index in range(count)]


def _model_jobs() -> list[dict[str, object]]:
    return [
        {
            "model": model,
            "model_seed": seed,
            "checkpoint_sha256": f"{model}-{seed}".encode().hex().ljust(64, "0")[:64],
        }
        for model in (
            "unet_baseline",
            "transunet_r50_vit_b16",
            "unet_noskip",
        )
        for seed in (42, 123, 3407)
    ]


def _observer_jobs() -> list[dict[str, object]]:
    rows = []
    for model in (
        "unet_baseline",
        "transunet_r50_vit_b16",
        "unet_noskip",
    ):
        for model_seed in (42, 123, 3407):
            for node in (
                "down1",
                "down2",
                "down3",
                "down4",
                "up1",
                "up2",
                "up3",
                "up4",
            ):
                for observer_seed in (17, 29, 43):
                    identity = f"{model}-{model_seed}-{node}-{observer_seed}"
                    rows.append(
                        {
                            "model": model,
                            "model_seed": model_seed,
                            "node": node,
                            "observer_seed": observer_seed,
                            "sha256": identity.encode().hex().ljust(64, "0")[:64],
                            "status": "PASS",
                        }
                    )
    return rows


def _plan_hashes() -> dict[str, str]:
    return {
        f"{model}/seed_{seed}": f"{model}-{seed}-plan".encode().hex().ljust(64, "0")[:64]
        for model in (
            "unet_baseline",
            "transunet_r50_vit_b16",
            "unet_noskip",
        )
        for seed in (42, 123, 3407)
    }


def _payload() -> dict:
    return build_v11_protocol_lock_payload(
        configuration=_configuration(),
        configuration_sha256="a" * 64,
        source_identity={
            "commit": "abc123",
            "source_tree_sha256": "b" * 64,
            "code_dirty": False,
        },
        model_jobs=_model_jobs(),
        observer_jobs=_observer_jobs(),
        validation_patient_ids=_patients(125, "000"),
        test_patient_ids=_patients(250, "001"),
        process_hashes={
            "H_U": "c" * 64,
            "H_T": "d" * 64,
            "H_shared": "e" * 64,
        },
        calibration_hashes={
            "matching": "f" * 64,
            "norm_and_leakage": "1" * 64,
            "history_admission": "2" * 64,
        },
        intervention_plan_hashes=_plan_hashes(),
        data_inventory={
            "data_config_sha256": "3" * 64,
            "split_sha256": "4" * 64,
        },
        environment_identity={"python": "3.10.8", "torch": "2.1.2+cu118"},
    )


def test_checked_in_scope_is_exactly_two_main_architectures_one_control_and_brats():
    validated = validate_v11_configuration(_configuration())

    assert validated.main_models == (
        "unet_baseline",
        "transunet_r50_vit_b16",
    )
    assert validated.control_models == ("unet_noskip",)
    assert validated.dataset == "brats2023_2d"
    assert validated.nodes == (
        "down1",
        "down2",
        "down3",
        "down4",
        "up1",
        "up2",
        "up3",
        "up4",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset", "acdc"),
        ("main_models", ["unet_baseline", "transunet_r50_vit_b16", "swin_unet"]),
        ("control_models", ["unet_noskip", "attention_unet"]),
        ("nodes", ["down1", "up4"]),
        ("doses", [0.0, 0.5, 1.0]),
    ],
)
def test_configuration_rejects_scope_or_protocol_drift(field, value):
    changed = deepcopy(_configuration())
    changed[field] = value

    with pytest.raises(ValueError, match="changed"):
        validate_v11_configuration(changed)


def test_configuration_rejects_unregistered_matching_candidate_budget():
    changed = deepcopy(_configuration())
    changed["matching"]["source_candidates_per_patient_node_state_class"] = 16

    with pytest.raises(ValueError, match="matching changed"):
        validate_v11_configuration(changed)


def test_lock_requires_clean_source_complete_models_and_all_observers():
    payload = _payload()
    assert len(payload["registered_model_jobs"]) == 9
    assert len(payload["registered_observer_jobs"]) == 216
    assert payload["status"] == "LOCKED_BEFORE_FORMAL_INTERVENTION"

    dirty = deepcopy(payload)
    configuration = dirty.pop("configuration")
    del dirty
    with pytest.raises(ValueError, match="clean source tree"):
        build_v11_protocol_lock_payload(
            configuration=configuration,
            configuration_sha256="a" * 64,
            source_identity={
                "commit": "abc123",
                "source_tree_sha256": "b" * 64,
                "code_dirty": True,
            },
            model_jobs=_model_jobs(),
            observer_jobs=_observer_jobs(),
            validation_patient_ids=_patients(125, "000"),
            test_patient_ids=_patients(250, "001"),
            process_hashes={"H_U": "c" * 64, "H_T": "d" * 64, "H_shared": "e" * 64},
            calibration_hashes={"matching": "f" * 64, "norm_and_leakage": "1" * 64, "history_admission": "2" * 64},
            intervention_plan_hashes=_plan_hashes(),
            data_inventory={"data_config_sha256": "3" * 64, "split_sha256": "4" * 64},
            environment_identity={"python": "3.10.8", "torch": "2.1.2+cu118"},
        )

    missing_observer = _observer_jobs()[:-1]
    with pytest.raises(ValueError, match="observer matrix"):
        build_v11_protocol_lock_payload(
            configuration=_configuration(),
            configuration_sha256="a" * 64,
            source_identity={"commit": "abc123", "source_tree_sha256": "b" * 64, "code_dirty": False},
            model_jobs=_model_jobs(),
            observer_jobs=missing_observer,
            validation_patient_ids=_patients(125, "000"),
            test_patient_ids=_patients(250, "001"),
            process_hashes={"H_U": "c" * 64, "H_T": "d" * 64, "H_shared": "e" * 64},
            calibration_hashes={"matching": "f" * 64, "norm_and_leakage": "1" * 64, "history_admission": "2" * 64},
            intervention_plan_hashes=_plan_hashes(),
            data_inventory={"data_config_sha256": "3" * 64, "split_sha256": "4" * 64},
            environment_identity={"python": "3.10.8", "torch": "2.1.2+cu118"},
        )


def test_protocol_lock_is_atomic_idempotent_and_rejects_changed_content(tmp_path):
    lock_path = tmp_path / "v11_protocol_lock.json"
    formal_root = tmp_path / "formal_jobs"
    payload = _payload()

    write_protocol_lock(lock_path, payload, formal_output_root=formal_root)
    first = lock_path.read_text(encoding="utf-8")
    write_protocol_lock(lock_path, payload, formal_output_root=formal_root)
    assert lock_path.read_text(encoding="utf-8") == first

    changed = deepcopy(payload)
    changed["process_hashes"]["H_shared"] = "9" * 64
    with pytest.raises(ValueError, match="existing protocol lock differs"):
        write_protocol_lock(lock_path, changed, formal_output_root=formal_root)

    second_lock = tmp_path / "second.json"
    patient_result = formal_root / "unet" / "seed_42" / "patient_results" / "p.json"
    patient_result.parent.mkdir(parents=True)
    patient_result.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="formal intervention outputs already exist"):
        write_protocol_lock(second_lock, payload, formal_output_root=formal_root)


def test_active_lock_rejects_changed_source_patients_or_process_hashes(tmp_path):
    lock_path = tmp_path / "v11_protocol_lock.json"
    payload = _payload()
    write_protocol_lock(lock_path, payload, formal_output_root=tmp_path / "formal")

    loaded = load_and_validate_v11_protocol_lock(
        lock_path,
        active_configuration_sha256="a" * 64,
        active_source_tree_sha256="b" * 64,
        test_patient_ids=_patients(250, "001"),
        process_hashes=payload["process_hashes"],
        model="unet_baseline",
        model_seed=42,
    )
    assert loaded["formal_intervention_authorized"] is True

    with pytest.raises(ValueError, match="source tree"):
        load_and_validate_v11_protocol_lock(
            lock_path,
            active_configuration_sha256="a" * 64,
            active_source_tree_sha256="8" * 64,
            test_patient_ids=_patients(250, "001"),
            process_hashes=payload["process_hashes"],
            model="unet_baseline",
            model_seed=42,
        )
    with pytest.raises(ValueError, match="patient registry"):
        load_and_validate_v11_protocol_lock(
            lock_path,
            active_configuration_sha256="a" * 64,
            active_source_tree_sha256="b" * 64,
            test_patient_ids=_patients(249, "001") + ["different"],
            process_hashes=payload["process_hashes"],
            model="unet_baseline",
            model_seed=42,
        )
    with pytest.raises(ValueError, match="process hashes"):
        load_and_validate_v11_protocol_lock(
            lock_path,
            active_configuration_sha256="a" * 64,
            active_source_tree_sha256="b" * 64,
            test_patient_ids=_patients(250, "001"),
            process_hashes={**payload["process_hashes"], "H_shared": "0" * 64},
            model="unet_baseline",
            model_seed=42,
        )


def test_serialized_lock_contains_no_unregistered_dataset_or_architecture():
    payload = _payload()
    encoded = json.dumps(payload, sort_keys=True)

    assert "acdc" not in encoded.lower()
    assert "swin" not in encoded.lower()
    assert "attention_unet" not in encoded.lower()
