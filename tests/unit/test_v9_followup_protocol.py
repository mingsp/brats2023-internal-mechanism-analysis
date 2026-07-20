from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from scripts.audit_v9_transunet_feasibility import summarize_feasibility_rows
from scripts.lock_v9_transunet_protocol import build_v9_protocol_lock_payload
from scripts.run_v8_transunet_mechanism import resolve_registered_patient_ids
from pptt.statistics.mechanism_replication import (
    select_feasible_registered_candidate,
)
from pptt.validation.v9_followup import (
    derive_common_unintervened_cohort,
    derive_remaining_path_naive_cohort,
)


SEEDS = (42, 123, 3407)


def _candidate(path_id: str, order: int, minimum_effect: float) -> dict:
    return {
        "path_id": path_id,
        "topology_order": order,
        "passed": True,
        "all_seed_directions_positive": True,
        "supported_seed_count": 3,
        "minimum_seed_effect": minimum_effect,
        "mean_seed_effect": minimum_effect + 0.1,
        "seed_results": [],
    }


def _feasibility(path_id: str, order: int, count: int) -> list[dict]:
    return [
        {
            "path_id": path_id,
            "topology_order": order,
            "model_seed": seed,
            "specificity_evaluable_patient_count": count,
            "aggregate_matched_feature_count": count * 8,
        }
        for seed in SEEDS
    ]


def test_feasibility_gate_rejects_strongest_unmatchable_path():
    process_registration = {
        "evaluated_candidates": [
            _candidate("coarse", 0, 2.2),
            _candidate("fine", 3, 0.9),
        ]
    }
    feasibility = pd.DataFrame(
        _feasibility("coarse", 0, 0) + _feasibility("fine", 3, 24)
    )

    result = select_feasible_registered_candidate(
        process_registration,
        feasibility,
        required_seeds=SEEDS,
        minimum_feasible_patients_per_seed=20,
    )

    assert result["status"] == "REGISTERED_FEASIBLE_TRANSUNET_CANDIDATE"
    assert result["test_intervention_authorized"] is True
    assert result["candidate"]["path_id"] == "fine"
    coarse = next(
        row for row in result["evaluated_candidates"] if row["path_id"] == "coarse"
    )
    assert coarse["process_passed"] is True
    assert coarse["feasibility_passed"] is False
    assert coarse["passed"] is False


def test_feasibility_gate_stops_when_no_path_is_testable():
    process_registration = {
        "evaluated_candidates": [_candidate("coarse", 0, 2.2)]
    }

    result = select_feasible_registered_candidate(
        process_registration,
        pd.DataFrame(_feasibility("coarse", 0, 19)),
        required_seeds=SEEDS,
        minimum_feasible_patients_per_seed=20,
    )

    assert result["status"] == "NO_FEASIBLE_TRANSUNET_CANDIDATE"
    assert result["test_intervention_authorized"] is False
    assert result["candidate"] is None


def _payload(patient_id: str, status: str) -> dict:
    return {
        "patient_id": patient_id,
        "status": status,
        "conditions": [],
        "effects": {},
        "audit": {},
    }


def test_common_cohort_contains_only_pairs_without_prior_intervention():
    rows_by_seed = {
        42: [_payload("p1", "INELIGIBLE_TARGET_FEATURE_POSITIONS"), _payload("p2", "PASS")],
        123: [_payload("p1", "INELIGIBLE_TARGET_FEATURE_POSITIONS"), _payload("p2", "INELIGIBLE_TARGET_FEATURE_POSITIONS")],
        3407: [_payload("p1", "INELIGIBLE_TARGET_FEATURE_POSITIONS"), _payload("p2", "PASS")],
    }

    cohort = derive_common_unintervened_cohort(
        rows_by_seed,
        required_seeds=SEEDS,
        minimum_patient_count=1,
    )

    assert cohort == ("p1",)


def test_common_cohort_rejects_nonempty_intervention_payload():
    rows_by_seed = {
        seed: [_payload("p1", "INELIGIBLE_TARGET_FEATURE_POSITIONS")]
        for seed in SEEDS
    }
    rows_by_seed = deepcopy(rows_by_seed)
    rows_by_seed[123][0]["conditions"] = [{"condition": "clean"}]

    with pytest.raises(ValueError, match="contains prior intervention data"):
        derive_common_unintervened_cohort(
            rows_by_seed,
            required_seeds=SEEDS,
            minimum_patient_count=1,
        )


def test_feasibility_summary_counts_only_fully_matched_patients():
    rows = pd.DataFrame(
        [
            {
                "path_id": "path",
                "topology_order": 1,
                "model_seed": seed,
                "patient_id": patient,
                "status": status,
                "matched_feature_count": matched,
            }
            for seed in SEEDS
            for patient, status, matched in (
                ("p1", "PASS", 9),
                ("p2", "INELIGIBLE_MATCHED_FEATURE_POSITIONS", 7),
            )
        ]
    )

    summary = summarize_feasibility_rows(rows)

    assert len(summary) == len(SEEDS)
    assert set(summary.specificity_evaluable_patient_count) == {1}
    assert set(summary.aggregate_matched_feature_count) == {9}


def test_v9_lock_records_feasible_candidate_and_unintervened_cohort():
    candidate = _candidate("skip_down3_to_up1", 1, 0.9)
    candidate["path"] = {
        "path_id": "skip_down3_to_up1",
        "source_node": "down3",
        "receiver_node": "up1",
        "source_index": 2,
        "receiver_index": 4,
        "source_module_path": "source",
        "receiver_module_path": "receiver",
        "argument_index": 1,
        "argument_name": "skip",
        "topology_order": 1,
    }
    patients = tuple(f"p{index:02d}" for index in range(26))

    payload = build_v9_protocol_lock_payload(
        configuration={"name": "v9"},
        configuration_sha256="config",
        source_identity={
            "commit": "abc",
            "source_tree_sha256": "tree",
            "code_dirty": False,
        },
        candidate_registration={
            "status": "REGISTERED_FEASIBLE_TRANSUNET_CANDIDATE",
            "test_intervention_authorized": True,
            "candidate": candidate,
        },
        candidate_registration_sha256="candidate",
        validation_asset_hashes={"feasibility": "hash"},
        model_jobs=[
            {
                "model": "transunet_r50_vit_b16",
                "model_seed": seed,
                "checkpoint_sha256": f"m{seed}",
            }
            for seed in SEEDS
        ],
        observer_jobs=[
            {
                "model": "transunet_r50_vit_b16",
                "model_seed": seed,
                "status": "PASS",
                "observer_bundle_sha256": f"o{seed}",
            }
            for seed in SEEDS
        ],
        test_patient_ids=patients,
        expected_patient_count=26,
        data_inventory={"split": "hash"},
        v8_audit={"status_sha256": "v8"},
    )

    assert payload["status"] == "LOCKED_BEFORE_FIRST_TEST_INTERVENTION"
    assert payload["test_patient_ids"] == list(patients)
    assert payload["prior_v8_result_may_not_change"] is True


def test_runner_uses_locked_subset_without_changing_split_inventory():
    all_patients = ("p1", "p2", "p3", "p4")
    selected = resolve_registered_patient_ids(
        all_patients,
        {"test_patient_ids": ["p2", "p4"]},
        registry_mode="protocol_lock_subset",
        expected_split_count=4,
        expected_formal_count=2,
    )

    assert selected == ["p2", "p4"]


def test_remaining_path_naive_cohort_excludes_every_v9_patient():
    remaining = derive_remaining_path_naive_cohort(
        ("p1", "p2", "p3", "p4"),
        ("p2", "p4"),
        expected_remaining_count=2,
    )

    assert remaining == ("p1", "p3")
