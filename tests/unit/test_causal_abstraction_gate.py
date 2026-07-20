from __future__ import annotations

import pandas as pd

from pptt.causal_abstraction.gates import (
    ConclusionStatus,
    evaluate_causal_abstraction_gate,
)


ARCHITECTURES = ("unet", "transunet")
NODES = ("down1", "up3")
SEEDS = (42,)
DIRECTIONS = ("unet->transunet", "transunet->unet")


def _thresholds() -> dict[str, object]:
    return {
        "main_architectures": ARCHITECTURES,
        "model_seeds": SEEDS,
        "nodes": NODES,
        "transfer_directions": DIRECTIONS,
        "minimum_node_patients": 2,
        "minimum_source_states": 2,
        "minimum_state_patients": 2,
        "minimum_state_pixels": 4,
        "maximum_reconstruction_error": 1.0e-5,
        "minimum_state_realization": 0.95,
        "maximum_restore_error": 1.0e-6,
        "maximum_macro_tv": 0.15,
        "maximum_worst_tv": 0.20,
        "maximum_worst_tv_ci_high": 0.25,
        "maximum_shared_increment": 0.03,
        "maximum_shared_increment_ci_high": 0.05,
        "minimum_specificity_delta": 0.10,
        "minimum_specificity_ci_low": 0.05,
        "maximum_null_clean_tv": 0.05,
        "minimum_order_advantage": 0.10,
        "holm_alpha": 0.05,
        "minimum_dose_aligned_fraction": 0.90,
        "required_randomized_controls": ("random_observer", "state_permutation"),
    }


def _audit() -> dict[str, object]:
    return {
        "asset_identity_pass": True,
        "protocol_lock_pass": True,
        "patient_registry_pass": True,
        "observer_admission_pass": True,
        "source_clean_before_formal_pass": True,
        "history_status": "PASS",
    }


def _rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, object]] = []
    controls: list[dict[str, object]] = []
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            for node in NODES:
                for source_state in (1, 3):
                    metrics.append(
                        {
                            "kind": "coverage",
                            "architecture": architecture,
                            "model_seed": seed,
                            "node": node,
                            "source_state": source_state,
                            "patient_count": 10,
                            "pixel_count": 20,
                            "evaluable": True,
                        }
                    )
                metrics.extend(
                    [
                        {
                            "kind": "operator",
                            "architecture": architecture,
                            "model_seed": seed,
                            "node": node,
                            "reconstruction_error": 1.0e-7,
                            "state_realization": 0.99,
                            "restore_error": 1.0e-8,
                            "ood_pass": True,
                        },
                        {
                            "kind": "fidelity",
                            "evaluation": "within",
                            "architecture": architecture,
                            "model_seed": seed,
                            "node": node,
                            "macro_tv": 0.08,
                            "worst_tv": 0.12,
                            "worst_ci_high": 0.16,
                        },
                        {
                            "kind": "fidelity",
                            "evaluation": "shared",
                            "architecture": architecture,
                            "model_seed": seed,
                            "node": node,
                            "macro_tv": 0.09,
                            "worst_tv": 0.13,
                            "worst_ci_high": 0.17,
                        },
                        {
                            "kind": "shared_noninferiority",
                            "architecture": architecture,
                            "model_seed": seed,
                            "node": node,
                            "tv_increment": 0.01,
                            "ci_high": 0.02,
                        },
                        {
                            "kind": "dose",
                            "architecture": architecture,
                            "model_seed": seed,
                            "node": node,
                            "aligned_fraction": 0.95,
                            "reverse_condition_count": 0,
                        },
                    ]
                )
                controls.extend(
                    [
                        {
                            "kind": "specificity",
                            "architecture": architecture,
                            "model_seed": seed,
                            "node": node,
                            "delta_gain": 0.20,
                            "delta_ci_low": 0.12,
                            "null_clean_tv": 0.02,
                        },
                        {
                            "kind": "order",
                            "architecture": architecture,
                            "model_seed": seed,
                            "node": node,
                            "advantage": 0.15,
                            "holm_p": 0.01,
                        },
                    ]
                )
                for control_name in ("random_observer", "state_permutation"):
                    controls.append(
                        {
                            "kind": "randomized",
                            "control_name": control_name,
                            "architecture": architecture,
                            "model_seed": seed,
                            "node": node,
                            "separation_pass": True,
                        }
                    )
    direction_targets = {
        "unet->transunet": "transunet",
        "transunet->unet": "unet",
    }
    for direction, target in direction_targets.items():
        for seed in SEEDS:
            for node in NODES:
                metrics.append(
                    {
                        "kind": "fidelity",
                        "evaluation": "cross",
                        "direction": direction,
                        "architecture": target,
                        "model_seed": seed,
                        "node": node,
                        "macro_tv": 0.10,
                        "worst_tv": 0.14,
                        "worst_ci_high": 0.18,
                    }
                )
    metrics.append(
        {
            "kind": "structure",
            "delta_distance": 0.20,
            "ci_low": 0.10,
            "reported_node_count": len(NODES),
        }
    )
    return pd.DataFrame(metrics), pd.DataFrame(controls)


def test_complete_matrix_authorizes_shared_full_network_claim():
    metrics, controls = _rows()

    report = evaluate_causal_abstraction_gate(
        metrics,
        controls,
        _audit(),
        _thresholds(),
    )

    assert report.status == ConclusionStatus.PASS_SHARED_FULL_NETWORK
    assert report.full_network_claim_authorized is True
    assert report.cross_architecture_claim_authorized is True
    assert report.structural_sensitivity_supported is True


def test_one_failed_node_blocks_full_network_claim():
    metrics, controls = _rows()
    failed = (
        (metrics.kind == "fidelity")
        & (metrics.evaluation == "within")
        & (metrics.architecture == "unet")
        & (metrics.node == "up3")
    )
    metrics.loc[failed, "worst_tv"] = 0.30

    report = evaluate_causal_abstraction_gate(
        metrics, controls, _audit(), _thresholds()
    )

    assert report.status == ConclusionStatus.PARTIAL_NODE_RANGE_ONLY
    assert report.full_network_claim_authorized is False
    assert any(item["node"] == "up3" for item in report.failed_conditions)


def test_shared_claim_requires_both_cross_transfer_directions():
    metrics, controls = _rows()
    metrics = metrics[
        ~(
            (metrics.kind == "fidelity")
            & (metrics.evaluation == "cross")
            & (metrics.direction == "transunet->unet")
        )
    ]

    report = evaluate_causal_abstraction_gate(
        metrics, controls, _audit(), _thresholds()
    )

    assert report.status == ConclusionStatus.ARCHITECTURE_SPECIFIC_PROCESS_ONLY
    assert report.cross_architecture_claim_authorized is False


def test_shared_noninferiority_failure_preserves_only_bidirectional_transfer():
    metrics, controls = _rows()
    failed = (
        (metrics.kind == "shared_noninferiority")
        & (metrics.architecture == "transunet")
        & (metrics.node == "down1")
    )
    metrics.loc[failed, "tv_increment"] = 0.08

    report = evaluate_causal_abstraction_gate(
        metrics, controls, _audit(), _thresholds()
    )

    assert report.status == ConclusionStatus.BIDIRECTIONAL_TRANSFER_ONLY
    assert report.cross_architecture_claim_authorized is True
    assert report.shared_model_claim_authorized is False


def test_nonseparating_null_or_randomized_control_blocks_causal_claim():
    metrics, controls = _rows()
    controls.loc[controls.kind == "specificity", "delta_gain"] = 0.01

    report = evaluate_causal_abstraction_gate(
        metrics, controls, _audit(), _thresholds()
    )

    assert report.status == ConclusionStatus.OBSERVATIONAL_PROCESS_ONLY
    assert report.causal_claim_authorized is False


def test_failed_asset_audit_has_priority_over_favorable_results():
    metrics, controls = _rows()
    audit = _audit()
    audit["asset_identity_pass"] = False

    report = evaluate_causal_abstraction_gate(
        metrics, controls, audit, _thresholds()
    )

    assert report.status == ConclusionStatus.FAILED_AUDIT
    assert report.allowed_claim == "No formal result is authorized."


def test_history_misspecification_has_explicit_status():
    metrics, controls = _rows()
    audit = _audit()
    audit["history_status"] = "HIGH_LEVEL_MODEL_MISSPECIFIED"

    report = evaluate_causal_abstraction_gate(
        metrics, controls, audit, _thresholds()
    )

    assert report.status == ConclusionStatus.HIGH_LEVEL_MODEL_MISSPECIFIED
    assert report.full_network_claim_authorized is False

