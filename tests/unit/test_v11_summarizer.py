from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pptt.causal_abstraction.kernels import TransitionProcess
from scripts.run_v11_causal_abstraction import FORMAL_DOSES, NODE_NAMES
from scripts.summarize_v11_causal_abstraction import (
    _coverage_and_operator_rows,
    _holm_adjust,
    _patient_error_advantage,
    _permuted_depth_process,
    collect_formal_job,
)


def _condition(node_index: int, name: str) -> dict:
    path = [*NODE_NAMES[node_index:], "Y"]
    return {
        "condition": name,
        "doses": list(FORMAL_DOSES),
        "downstream_path": path,
        "audits": {"dose_zero_max_abs_logit_error": 0.0},
        "state_counts": [
            {
                "dose": dose,
                "depth": depth,
                "downstream_depth": downstream_depth,
                "source_state": 1,
                "truth_class": 1,
                "selected_pixel_count": 1,
                "reliable_pixel_count": 1,
                "state_counts": [0, 1, 0, 0, 0],
            }
            for dose in FORMAL_DOSES
            for downstream_depth, depth in enumerate(path)
        ],
    }


def _formal_patient(patient_id: str, identity: dict) -> dict:
    return {
        "schema_version": 1,
        "execution_mode": "formal",
        "formal_claim_eligible": False,
        "formal_summary_eligible": True,
        "patient_id": patient_id,
        "job_identity": identity,
        "nodes": list(NODE_NAMES),
        "node_results": [
            {
                "node": node,
                "status": "PASS",
                "conditions": {
                    "task": _condition(node_index, "task"),
                    "null": _condition(node_index, "null"),
                },
                "effect_summary": {
                    "pair_count": 1,
                    "evaluable": True,
                    "operator_reconstruction_max_abs_error": 0.0,
                    "target_state_realization_rate": 1.0,
                    "restore_max_abs_logit_error": 0.0,
                    "norm_pass": True,
                    "leakage_pass": True,
                    "null_target_max_abs_logit_change": 0.0,
                },
            }
            for node_index, node in enumerate(NODE_NAMES)
        ],
        "storage_audit": {
            "full_feature_tensors_persisted": False,
            "full_output_logits_persisted": False,
        },
    }


def test_collect_formal_job_rejects_smoke_and_loads_every_depth(tmp_path: Path):
    patient_id = "BraTS-GLI-00001-000"
    identity = {"model": "unet_baseline", "model_seed": 42}
    root = tmp_path / "job"
    patient_path = root / "patient_results" / patient_id / "patient_result.json"
    patient_path.parent.mkdir(parents=True)
    payload = _formal_patient(patient_id, identity)
    patient_path.write_text(json.dumps(payload), encoding="utf-8")
    (root / "job_manifest.json").write_text(
        json.dumps(
            {
                "execution_mode": "formal",
                "job_identity": identity,
                "patient_ids": [patient_id],
                "patient_count": 1,
                "full_activations_persisted": False,
            }
        ),
        encoding="utf-8",
    )
    (root / "job_status.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "execution_mode": "formal",
                "formal_summary_eligible": True,
                "completed_patients": 1,
                "job_identity": identity,
            }
        ),
        encoding="utf-8",
    )

    states, effects, audit = collect_formal_job(
        root,
        model="unet_baseline",
        model_seed=42,
        patient_ids=[patient_id],
    )
    assert len(states) == 2 * len(FORMAL_DOSES) * sum(
        len(NODE_NAMES) - index + 1 for index in range(len(NODE_NAMES))
    )
    assert len(effects) == len(NODE_NAMES)
    assert audit["patient_count"] == 1

    payload["execution_mode"] = "smoke"
    patient_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not an admissible formal artifact"):
        collect_formal_job(
            root,
            model="unet_baseline",
            model_seed=42,
            patient_ids=[patient_id],
        )


def test_collect_formal_job_rejects_an_incomplete_condition_grid(tmp_path: Path):
    patient_id = "BraTS-GLI-00001-000"
    identity = {"model": "unet_baseline", "model_seed": 42}
    root = tmp_path / "job"
    patient_path = root / "patient_results" / patient_id / "patient_result.json"
    patient_path.parent.mkdir(parents=True)
    payload = _formal_patient(patient_id, identity)
    payload["node_results"][0]["conditions"]["task"]["state_counts"].pop()
    patient_path.write_text(json.dumps(payload), encoding="utf-8")
    (root / "job_manifest.json").write_text(
        json.dumps(
            {
                "execution_mode": "formal",
                "job_identity": identity,
                "patient_ids": [patient_id],
                "patient_count": 1,
                "full_activations_persisted": False,
            }
        ),
        encoding="utf-8",
    )
    (root / "job_status.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "execution_mode": "formal",
                "formal_summary_eligible": True,
                "completed_patients": 1,
                "job_identity": identity,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="condition grid is incomplete"):
        collect_formal_job(
            root,
            model="unet_baseline",
            model_seed=42,
            patient_ids=[patient_id],
        )


def test_reliability_summary_keeps_the_worst_downstream_condition():
    state_rows = pd.DataFrame(
        [
            {
                "architecture": "unet_baseline",
                "model_seed": 42,
                "patient_id": "p1",
                "intervention_node": "down1",
                "source_state": 1,
                "truth_class": 1,
                "condition": "task",
                "dose": 1.0,
                "downstream_node": "down1",
                "selected_pixel_count": 10,
                "reliable_pixel_count": 9,
            },
            {
                "architecture": "unet_baseline",
                "model_seed": 42,
                "patient_id": "p1",
                "intervention_node": "down1",
                "source_state": 1,
                "truth_class": 1,
                "condition": "null",
                "dose": 0.5,
                "downstream_node": "up4",
                "selected_pixel_count": 10,
                "reliable_pixel_count": 5,
            },
        ]
    )
    effects = pd.DataFrame(
        [
            {
                "architecture": "unet_baseline",
                "model_seed": 42,
                "patient_id": "p1",
                "intervention_node": "down1",
                "node_status": "PASS",
                "pair_count": 10,
                "operator_reconstruction_max_abs_error": 0.0,
                "target_state_realization_rate": 1.0,
                "restore_max_abs_logit_error": 0.0,
                "norm_pass": True,
                "leakage_pass": True,
                "null_target_max_abs_logit_change": 0.0,
            }
        ]
    )
    configuration = {
        "coverage": {
            "minimum_state_patients": 1,
            "minimum_state_pixels": 1,
        }
    }

    rows = _coverage_and_operator_rows(state_rows, effects, configuration)
    reliability = [row for row in rows if row["kind"] == "reliability"]

    assert len(reliability) == 1
    assert reliability[0]["minimum_reliable_fraction"] == pytest.approx(0.5)


def test_holm_adjustment_is_monotone_in_sorted_p_values():
    raw = np.array([0.03, 0.001, 0.02])
    adjusted = _holm_adjust(raw)

    assert np.all((0.0 <= adjusted) & (adjusted <= 1.0))
    ordered = adjusted[np.argsort(raw)]
    assert np.all(np.diff(ordered) >= 0)
    np.testing.assert_allclose(adjusted, [0.04, 0.003, 0.04])


def test_paired_advantage_is_patient_equal():
    correct = pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "intervention_node": ["down1", "down1"],
            "source_state": [1, 1],
            "downstream_node": ["down2", "down2"],
            "truth_class": [1, 1],
            "tv_error": [0.1, 0.2],
        }
    )
    control = correct.copy()
    control["tv_error"] += 0.2

    patient, estimate, low, high, _ = _patient_error_advantage(
        correct,
        control,
        bootstrap_iterations=200,
        bootstrap_seed=7,
    )

    assert len(patient) == 2
    assert estimate == pytest.approx(0.2)
    assert low == pytest.approx(0.2)
    assert high == pytest.approx(0.2)


def test_depth_permutation_changes_kernel_order_without_changing_capacity():
    kernels = np.stack(
        [
            np.full((5, 5), 1.0 / 5) + np.eye(5) * index * 0.0
            for index in range(8)
        ]
    )
    kernels[0] = np.eye(5)
    kernels[-1] = np.roll(np.eye(5), 1, axis=1)
    process = TransitionProcess(
        node_names=NODE_NAMES,
        kernels=kernels,
        coverage=np.ones((8, 5), dtype=np.int64),
        alpha=0.5,
        source="original",
    )

    changed = _permuted_depth_process(process, list(reversed(range(8))))

    np.testing.assert_allclose(changed.kernels[0], process.kernels[-1])
    assert changed.kernels.shape == process.kernels.shape
    assert changed.node_names == process.node_names
