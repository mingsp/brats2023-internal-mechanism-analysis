from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pptt.causal_abstraction.runtime import CounterfactualTrace
from scripts.run_v11_causal_abstraction import (
    NODE_NAMES,
    _state_count_rows,
    run_patient_registry,
    run_v11_smoke,
    validate_locked_plan_manifest,
    validate_patient_plan,
    validate_execution_request,
)
from scripts.lock_v11_causal_abstraction_protocol import (
    sha256_file,
    sha256_named_values,
)


def test_v11_smoke_is_complete_resumable_and_nonformal(tmp_path: Path):
    output_root = tmp_path / "v11_smoke"

    first = run_v11_smoke(output_root, patient_limit=1)
    second = run_v11_smoke(output_root, patient_limit=1)

    assert first["status"] == "SMOKE_COMPLETE"
    assert first["execution_mode"] == "smoke"
    assert first["formal_claim_eligible"] is False
    assert first["node_count"] == 8
    assert first["condition_names"] == ["task", "null"]
    assert first["dose_count"] == 5
    assert first["completed_patients"] == 1
    assert first["skipped_completed_patients"] == 0
    assert second["completed_patients"] == 1
    assert second["skipped_completed_patients"] == 1

    patient_path = (
        output_root
        / "synthetic"
        / "seed_0"
        / "patient_results"
        / "synthetic-patient-000"
        / "patient_result.json"
    )
    payload = json.loads(patient_path.read_text(encoding="utf-8"))
    assert payload["patient_id"] == "synthetic-patient-000"
    assert payload["nodes"] == list(NODE_NAMES)
    assert len(payload["node_results"]) == 8
    for node_index, node_result in enumerate(payload["node_results"]):
        assert node_result["node"] == NODE_NAMES[node_index]
        assert set(node_result["conditions"]) == {"task", "null"}
        expected_downstream = [*NODE_NAMES[node_index:], "Y"]
        for condition in node_result["conditions"].values():
            assert condition["doses"] == [0.0, 0.25, 0.5, 0.75, 1.0]
            assert condition["downstream_path"] == expected_downstream
            assert len(condition["state_counts"]) == 5 * len(expected_downstream)

    def keys(value):
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value), set())
        return set()

    persisted_keys = keys(payload)
    assert "activations" not in persisted_keys
    assert "feature_tensor" not in persisted_keys
    assert "final_logits" not in persisted_keys
    assert payload["storage_audit"]["full_feature_tensors_persisted"] is False
    assert payload["storage_audit"]["full_output_logits_persisted"] is False


def test_formal_request_rejects_patient_limit_or_missing_lock(tmp_path: Path):
    with pytest.raises(ValueError, match="patient limit"):
        validate_execution_request(
            execution_mode="formal",
            patient_limit=1,
            protocol_lock=tmp_path / "lock.json",
        )
    with pytest.raises(FileNotFoundError, match="protocol lock"):
        validate_execution_request(
            execution_mode="formal",
            patient_limit=None,
            protocol_lock=tmp_path / "lock.json",
        )


def test_smoke_request_requires_a_positive_patient_limit(tmp_path: Path):
    with pytest.raises(ValueError, match="positive patient limit"):
        validate_execution_request(
            execution_mode="smoke",
            patient_limit=0,
            protocol_lock=tmp_path / "unused.json",
        )


def test_patient_registry_persists_failure_without_partial_patient_output(
    tmp_path: Path,
):
    root = tmp_path / "failed_job"

    def fail(_patient_id: str) -> dict:
        raise RuntimeError("intentional test failure")

    with pytest.raises(RuntimeError, match="intentional test failure"):
        run_patient_registry(
            root,
            patient_ids=["p1"],
            job_identity={"model": "synthetic", "model_seed": 0},
            process_patient=fail,
            resume=False,
            execution_mode="smoke",
        )

    status = json.loads((root / "job_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "FAILED"
    assert status["completed_patients"] == 0
    assert status["failed_patient_id"] == "p1"
    assert not (root / "patient_results/p1").exists()


def _patient_plan(patient_id: str) -> dict:
    return {
        "schema_version": 1,
        "status": "REGISTERED_PATIENT_PLAN",
        "model": "unet_baseline",
        "model_seed": 42,
        "split": "test",
        "patient_id": patient_id,
        "nodes": [
            {
                "node": node,
                "status": "MATCHED",
                "base_slice_id": f"{patient_id}_001",
                "native_shape": [2, 2],
                "pairs": [
                    {
                        "base_output_index": node_index,
                        "base_state": 3,
                        "target_state": 1,
                        "truth_class": 1,
                        "source_patient_id": patient_id,
                        "source_slice_id": f"{patient_id}_001",
                        "source_output_index": node_index + 8,
                        "same_patient": True,
                        "boundary_stratum": 0,
                        "feature_norm_stratum": 0,
                    }
                ],
            }
            for node_index, node in enumerate(NODE_NAMES)
        ],
    }


def test_patient_plan_validation_rejects_duplicate_base_pixels():
    patient_id = "BraTS-GLI-00001-000"
    plan = _patient_plan(patient_id)
    validate_patient_plan(
        plan,
        model="unet_baseline",
        model_seed=42,
        patient_id=patient_id,
    )
    plan["nodes"][0]["pairs"].append(dict(plan["nodes"][0]["pairs"][0]))

    with pytest.raises(ValueError, match="repeats a base pixel"):
        validate_patient_plan(
            plan,
            model="unet_baseline",
            model_seed=42,
            patient_id=patient_id,
        )


def test_locked_plan_manifest_verifies_every_patient_hash(tmp_path: Path):
    patients = ["BraTS-GLI-00001-000", "BraTS-GLI-00002-000"]
    root = tmp_path / "unet_baseline" / "seed_42"
    patient_root = root / "patient_plans"
    patient_root.mkdir(parents=True)
    patient_hashes = {}
    for patient_id in patients:
        path = patient_root / f"{patient_id}.json"
        path.write_text(
            json.dumps(_patient_plan(patient_id), sort_keys=True),
            encoding="utf-8",
        )
        patient_hashes[patient_id] = sha256_file(path)
    coverage = {
        node: {
            "patient_count": 2,
            "target_states": {"1": {"patient_count": 2, "pixel_count": 2}},
            "evaluable_target_states": ["1"],
        }
        for node in NODE_NAMES
    }
    manifest = {
        "status": "COMPLETE_LOCKABLE_PLAN",
        "model": "unet_baseline",
        "model_seed": 42,
        "split": "test",
        "failed_nodes": [],
        "patient_ids": patients,
        "patient_count": 2,
        "patient_plan_hashes": patient_hashes,
        "patient_plan_registry_sha256": sha256_named_values(patient_hashes),
        "coverage": coverage,
        "full_activations_persisted": False,
    }
    manifest_path = root / "plan_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    lock = {
        "test_patient_ids": patients,
        "intervention_plan_hashes": {
            "unet_baseline/seed_42": sha256_file(manifest_path)
        },
        "configuration": {
            "coverage": {
                "minimum_node_patients": 1,
                "minimum_source_states": 1,
                "minimum_state_patients": 1,
                "minimum_state_pixels": 1,
            }
        },
    }

    loaded, loaded_root = validate_locked_plan_manifest(
        manifest_path,
        lock=lock,
        model="unet_baseline",
        model_seed=42,
    )
    assert loaded["patient_count"] == 2
    assert loaded_root == patient_root

    (patient_root / f"{patients[0]}.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="patient intervention plan differs"):
        validate_locked_plan_manifest(
            manifest_path,
            lock=lock,
            model="unet_baseline",
            model_seed=42,
        )


def test_state_counts_preserve_target_state_and_truth_strata():
    states = {
        0.0: {"up4": np.array([[3, 3], [4, 0]], dtype=np.uint8), "Y": np.array([[3, 1], [4, 0]], dtype=np.uint8)},
        1.0: {"up4": np.array([[1, 1], [2, 0]], dtype=np.uint8), "Y": np.array([[1, 1], [2, 0]], dtype=np.uint8)},
    }
    reliable = {
        dose: {depth: np.ones((2, 2), dtype=bool) for depth in values}
        for dose, values in states.items()
    }
    trace = CounterfactualTrace(
        node="up4",
        doses=(0.0, 1.0),
        downstream_states=states,
        downstream_reliable=reliable,
        final_logits={dose: np.zeros((2, 2, 2), dtype=np.float32) for dose in states},
        audits={"downstream_path": ["up4", "Y"]},
    )
    rows = _state_count_rows(
        trace,
        output_indices=np.array([0, 1, 2]),
        source_states=np.array([1, 1, 2], dtype=np.uint8),
        truth_classes=np.array([1, 1, 2], dtype=np.uint8),
    )

    assert len(rows) == 8
    full_up4_state1 = next(
        row
        for row in rows
        if row["dose"] == 1.0
        and row["depth"] == "up4"
        and row["source_state"] == 1
        and row["truth_class"] == 1
    )
    assert full_up4_state1["state_counts"] == [0, 2, 0, 0, 0]
