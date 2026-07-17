from __future__ import annotations

import numpy as np
import pandas as pd

from pptt.statistics.network_alignment import (
    compute_global_alignment_statistics,
    evaluate_network_alignment_gate,
    summarize_alignment_cells,
)


def _receiving_map(count: int = 7) -> dict[str, str]:
    return {f"t{index}": f"n{index + 1}" for index in range(count)}


def test_macro_and_micro_use_only_receiving_node_cells():
    rows = []
    for patient_id in ("a", "b"):
        rows.extend(
            [
                {
                    "model": "m",
                    "model_seed": 1,
                    "patient_id": patient_id,
                    "transition": "t0",
                    "restore_node": "n1",
                    "status": "EVALUABLE",
                    "specific_effect": 0.2,
                    "target_pixel_count": 1,
                },
                {
                    "model": "m",
                    "model_seed": 1,
                    "patient_id": patient_id,
                    "transition": "t1",
                    "restore_node": "n2",
                    "status": "EVALUABLE",
                    "specific_effect": 0.6,
                    "target_pixel_count": 3,
                },
                {
                    "model": "m",
                    "model_seed": 1,
                    "patient_id": patient_id,
                    "transition": "t0",
                    "restore_node": "off_path",
                    "status": "EVALUABLE",
                    "specific_effect": 100.0,
                    "target_pixel_count": 100,
                },
            ]
        )

    summary = compute_global_alignment_statistics(
        pd.DataFrame(rows),
        receiving_node_by_transition={"t0": "n1", "t1": "n2"},
        bootstrap_iterations=200,
        bootstrap_seed=7,
    )

    macro = summary.loc[summary.endpoint == "macro", "mean_difference"].item()
    micro = summary.loc[summary.endpoint == "micro", "mean_difference"].item()
    assert np.isclose(macro, 0.4)
    assert np.isclose(micro, 0.5)


def test_micro_average_does_not_double_count_disjoint_cohorts():
    rows = pd.DataFrame(
        [
            {
                "model": "m",
                "model_seed": 1,
                "patient_id": patient,
                "transition": transition,
                "restore_node": node,
                "status": "EVALUABLE",
                "specific_effect": effect,
                "target_pixel_count": count,
            }
            for patient in ("a", "b")
            for transition, node, effect, count in (
                ("t0", "n1", 0.25, 2),
                ("t1", "n2", 0.75, 2),
            )
        ]
    )

    summary = compute_global_alignment_statistics(
        rows,
        receiving_node_by_transition={"t0": "n1", "t1": "n2"},
        bootstrap_iterations=200,
        bootstrap_seed=11,
    )

    assert np.isclose(
        summary.loc[summary.endpoint == "micro", "mean_difference"].item(),
        0.5,
    )
    assert summary.loc[summary.endpoint == "micro", "total_weight"].item() == 8


def _global_rows(
    *,
    macro_positive: bool = True,
    micro_positive: bool = True,
) -> pd.DataFrame:
    rows = []
    for model in ("unet_baseline", "unet_noskip"):
        for seed in (42, 123):
            for endpoint, positive in (
                ("macro", macro_positive),
                ("micro", micro_positive),
            ):
                rows.append(
                    {
                        "model": model,
                        "model_seed": seed,
                        "endpoint": endpoint,
                        "mean_difference": 0.1 if positive else -0.01,
                        "ci_low": 0.05 if positive else -0.05,
                        "holm_p": 0.01 if positive else 0.5,
                    }
                )
    return pd.DataFrame(rows)


def _cell_rows(*, transition_count: int, positive_count: int) -> pd.DataFrame:
    rows = []
    for model in ("unet_baseline", "unet_noskip"):
        for seed in (42, 123):
            for transition_index in range(transition_count):
                rows.append(
                    {
                        "model": model,
                        "model_seed": seed,
                        "transition_index": transition_index,
                        "restore_node": f"n{transition_index + 1}",
                        "is_receiving_node_cell": True,
                        "evaluable": True,
                        "mean_specific_effect": (
                            0.1 if transition_index < positive_count else -0.01
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _audit() -> dict[str, bool]:
    return {
        "mathematical_properties_pass": True,
        "protocol_identity_pass": True,
        "formal_job_matrix_pass": True,
        "operator_audit_pass": True,
    }


def test_gate_rejects_only_one_evaluable_transition():
    result = evaluate_network_alignment_gate(
        _global_rows(),
        _cell_rows(transition_count=1, positive_count=1),
        _audit(),
        minimum_supported_seeds=2,
        minimum_evaluable_transitions=5,
        minimum_positive_receiving_transitions=4,
        alpha=0.05,
    )

    assert result["status"] == "INSUFFICIENT_NETWORK_COVERAGE"


def test_gate_rejects_positive_micro_but_nonpositive_macro():
    result = evaluate_network_alignment_gate(
        _global_rows(macro_positive=False, micro_positive=True),
        _cell_rows(transition_count=5, positive_count=5),
        _audit(),
        minimum_supported_seeds=2,
        minimum_evaluable_transitions=5,
        minimum_positive_receiving_transitions=4,
        alpha=0.05,
    )

    assert result["status"] == "OBSERVATIONAL_PROCESS_ONLY"


def test_gate_rejects_effect_concentrated_in_fewer_than_four_transitions():
    result = evaluate_network_alignment_gate(
        _global_rows(),
        _cell_rows(transition_count=5, positive_count=3),
        _audit(),
        minimum_supported_seeds=2,
        minimum_evaluable_transitions=5,
        minimum_positive_receiving_transitions=4,
        alpha=0.05,
    )

    assert result["status"] == "OBSERVATIONAL_PROCESS_ONLY"


def test_gate_passes_two_models_two_seeds_and_five_transitions():
    result = evaluate_network_alignment_gate(
        _global_rows(),
        _cell_rows(transition_count=5, positive_count=4),
        _audit(),
        minimum_supported_seeds=2,
        minimum_evaluable_transitions=5,
        minimum_positive_receiving_transitions=4,
        alpha=0.05,
    )

    assert result["status"] == "PASS_NETWORK_WIDE"
    assert result["network_wide_claim_authorized"] is True


def test_undefined_specificity_remains_null_not_zero():
    rows = pd.DataFrame(
        [
            {
                "model": "m",
                "model_seed": 1,
                "patient_id": "p",
                "transition_index": 0,
                "transition": "t0",
                "restore_node": "n1",
                "is_receiving_node_cell": True,
                "status": "TARGET_ONLY_CONTROL_UNAVAILABLE",
                "specific_effect": np.nan,
                "target_pixel_count": 20,
            }
        ]
    )

    summary = summarize_alignment_cells(
        rows,
        bootstrap_iterations=200,
        bootstrap_seed=3,
    )

    assert summary.loc[0, "specific_patient_count"] == 0
    assert np.isnan(summary.loc[0, "mean_specific_effect"])
