from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class ConclusionStatus(str, Enum):
    PASS_SHARED_FULL_NETWORK = "PASS_SHARED_FULL_NETWORK"
    PASS_SHARED_FULL_NETWORK_NO_STRUCTURAL_SENSITIVITY = (
        "PASS_SHARED_FULL_NETWORK_NO_STRUCTURAL_SENSITIVITY"
    )
    BIDIRECTIONAL_TRANSFER_ONLY = "BIDIRECTIONAL_TRANSFER_ONLY"
    ARCHITECTURE_SPECIFIC_PROCESS_ONLY = "ARCHITECTURE_SPECIFIC_PROCESS_ONLY"
    PARTIAL_NODE_RANGE_ONLY = "PARTIAL_NODE_RANGE_ONLY"
    OBSERVATIONAL_PROCESS_ONLY = "OBSERVATIONAL_PROCESS_ONLY"
    INSUFFICIENT_INTERVENTION_SUPPORT = "INSUFFICIENT_INTERVENTION_SUPPORT"
    HIGH_LEVEL_MODEL_MISSPECIFIED = "HIGH_LEVEL_MODEL_MISSPECIFIED"
    FAILED_AUDIT = "FAILED_AUDIT"


@dataclass(frozen=True)
class GateReport:
    status: ConclusionStatus
    full_network_claim_authorized: bool
    causal_claim_authorized: bool
    cross_architecture_claim_authorized: bool
    shared_model_claim_authorized: bool
    structural_sensitivity_supported: bool
    allowed_claim: str
    forbidden_claims: tuple[str, ...]
    failed_conditions: tuple[dict[str, Any], ...]
    passed_stage_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "full_network_claim_authorized": self.full_network_claim_authorized,
            "causal_claim_authorized": self.causal_claim_authorized,
            "cross_architecture_claim_authorized": (
                self.cross_architecture_claim_authorized
            ),
            "shared_model_claim_authorized": self.shared_model_claim_authorized,
            "structural_sensitivity_supported": (
                self.structural_sensitivity_supported
            ),
            "allowed_claim": self.allowed_claim,
            "forbidden_claims": list(self.forbidden_claims),
            "failed_conditions": list(self.failed_conditions),
            "passed_stage_names": list(self.passed_stage_names),
        }


_REQUIRED_AUDIT_FLAGS = (
    "asset_identity_pass",
    "protocol_lock_pass",
    "patient_registry_pass",
    "observer_admission_pass",
    "observer_randomization_pass",
    "source_clean_before_formal_pass",
)
_REQUIRED_THRESHOLDS = (
    "main_architectures",
    "model_seeds",
    "nodes",
    "transfer_directions",
    "minimum_node_patients",
    "minimum_source_states",
    "minimum_state_patients",
    "minimum_state_pixels",
    "minimum_downstream_reliable_fraction",
    "maximum_reconstruction_error",
    "minimum_state_realization",
    "maximum_restore_error",
    "maximum_macro_tv",
    "maximum_worst_tv",
    "maximum_worst_tv_ci_high",
    "maximum_shared_increment",
    "maximum_shared_increment_ci_high",
    "minimum_specificity_delta",
    "minimum_specificity_ci_low",
    "maximum_null_clean_tv",
    "minimum_order_advantage",
    "holm_alpha",
    "minimum_dose_aligned_fraction",
    "required_randomized_controls",
)


def _tuple_setting(
    thresholds: Mapping[str, object],
    name: str,
) -> tuple[Any, ...]:
    value = thresholds[name]
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"threshold {name} must be a non-string sequence")
    result = tuple(value)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"threshold {name} must be nonempty and unique")
    return result


def _number(
    thresholds: Mapping[str, object],
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = float(thresholds[name])
    if not np.isfinite(value):
        raise ValueError(f"threshold {name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"threshold {name} is below {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"threshold {name} is above {maximum}")
    return value


def _identity(architecture: Any, seed: Any, node: Any) -> dict[str, Any]:
    return {
        "architecture": str(architecture),
        "model_seed": int(seed),
        "node": str(node),
    }


def _rows_for_cell(
    frame: pd.DataFrame,
    *,
    kind: str,
    architecture: str,
    seed: int,
    node: str,
) -> pd.DataFrame:
    required = {"kind", "architecture", "model_seed", "node"}
    if not required.issubset(frame.columns):
        return frame.iloc[0:0]
    return frame[
        (frame["kind"] == kind)
        & (frame["architecture"] == architecture)
        & (frame["model_seed"] == seed)
        & (frame["node"] == node)
    ]


def _finite_row_value(row: pd.Series, name: str) -> float | None:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _fidelity_failures(
    metrics: pd.DataFrame,
    *,
    evaluation: str,
    cells: tuple[tuple[str, int, str], ...],
    thresholds: Mapping[str, object],
    direction: str | None = None,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for architecture, seed, node in cells:
        selected = _rows_for_cell(
            metrics,
            kind="fidelity",
            architecture=architecture,
            seed=seed,
            node=node,
        )
        if "evaluation" in selected:
            selected = selected[selected["evaluation"] == evaluation]
        if direction is not None:
            if "direction" not in selected:
                selected = selected.iloc[0:0]
            else:
                selected = selected[selected["direction"] == direction]
        identity = {
            "stage": f"fidelity:{evaluation}",
            **_identity(architecture, seed, node),
        }
        if direction is not None:
            identity["direction"] = direction
        if len(selected) != 1:
            failures.append(
                {**identity, "reason": "missing_or_duplicate_fidelity_row"}
            )
            continue
        row = selected.iloc[0]
        macro = _finite_row_value(row, "macro_tv")
        worst = _finite_row_value(row, "worst_tv")
        worst_high = _finite_row_value(row, "worst_ci_high")
        if (
            macro is None
            or worst is None
            or worst_high is None
            or macro > float(thresholds["maximum_macro_tv"])
            or worst > float(thresholds["maximum_worst_tv"])
            or worst_high > float(thresholds["maximum_worst_tv_ci_high"])
        ):
            failures.append(
                {
                    **identity,
                    "reason": "absolute_full_path_fidelity_failed",
                    "macro_tv": macro,
                    "worst_tv": worst,
                    "worst_ci_high": worst_high,
                }
            )
    return failures


def _report(
    status: ConclusionStatus,
    *,
    failed: list[dict[str, Any]],
    passed: list[str],
    structural: bool = False,
) -> GateReport:
    settings = {
        ConclusionStatus.PASS_SHARED_FULL_NETWORK: (
            True,
            True,
            True,
            True,
            "Within BraTS2023 and the registered U-Net and TransUNet models, "
            "one shared pixel-decision process is supported as a full-network "
            "approximate causal abstraction.",
        ),
        ConclusionStatus.PASS_SHARED_FULL_NETWORK_NO_STRUCTURAL_SENSITIVITY: (
            True,
            True,
            True,
            True,
            "Within BraTS2023 and the registered U-Net and TransUNet models, "
            "one shared pixel-decision process is supported as a full-network "
            "approximate causal abstraction, but structural diagnostic "
            "sensitivity is not supported.",
        ),
        ConclusionStatus.BIDIRECTIONAL_TRANSFER_ONLY: (
            False,
            True,
            True,
            False,
            "The registered high-level intervention dynamics transfer in both "
            "directions, but one shared parameterization is not supported.",
        ),
        ConclusionStatus.ARCHITECTURE_SPECIFIC_PROCESS_ONLY: (
            False,
            True,
            False,
            False,
            "Architecture-specific full-path process abstractions are supported; "
            "cross-architecture abstraction is not supported.",
        ),
        ConclusionStatus.PARTIAL_NODE_RANGE_ONLY: (
            False,
            False,
            False,
            False,
            "Only the explicitly passing node ranges are supported.",
        ),
        ConclusionStatus.OBSERVATIONAL_PROCESS_ONLY: (
            False,
            False,
            False,
            False,
            "The method supports observational process diagnosis only.",
        ),
        ConclusionStatus.INSUFFICIENT_INTERVENTION_SUPPORT: (
            False,
            False,
            False,
            False,
            "Intervention support is insufficient for a formal process claim.",
        ),
        ConclusionStatus.HIGH_LEVEL_MODEL_MISSPECIFIED: (
            False,
            False,
            False,
            False,
            "The registered first-order high-level process is misspecified.",
        ),
        ConclusionStatus.FAILED_AUDIT: (
            False,
            False,
            False,
            False,
            "No formal result is authorized.",
        ),
    }
    full, causal, cross, shared, allowed = settings[status]
    forbidden = [
        "A universal mechanism for all segmentation networks.",
        "Recovery of the unique true internal reasoning mechanism.",
        "Identical low-level computational paths across architectures.",
        "A claim that skip connections are the unique causal explanation.",
    ]
    if not full:
        forbidden.append("A full-network shared causal abstraction.")
    if not cross:
        forbidden.append("A cross-architecture common process.")
    if not causal:
        forbidden.append("An interventionally supported causal mechanism.")
    if not structural:
        forbidden.append("Reliable diagnosis of structural process change.")
    return GateReport(
        status=status,
        full_network_claim_authorized=full,
        causal_claim_authorized=causal,
        cross_architecture_claim_authorized=cross,
        shared_model_claim_authorized=shared,
        structural_sensitivity_supported=structural,
        allowed_claim=allowed,
        forbidden_claims=tuple(forbidden),
        failed_conditions=tuple(failed),
        passed_stage_names=tuple(passed),
    )


def evaluate_causal_abstraction_gate(
    metrics: pd.DataFrame,
    controls: pd.DataFrame,
    audit: Mapping[str, object],
    thresholds: Mapping[str, object],
) -> GateReport:
    """Apply the locked causal-abstraction decision hierarchy."""

    missing_thresholds = tuple(
        name for name in _REQUIRED_THRESHOLDS if name not in thresholds
    )
    if missing_thresholds:
        raise ValueError(f"gate thresholds are missing: {missing_thresholds}")
    architectures = tuple(str(value) for value in _tuple_setting(
        thresholds, "main_architectures"
    ))
    seeds = tuple(int(value) for value in _tuple_setting(thresholds, "model_seeds"))
    nodes = tuple(str(value) for value in _tuple_setting(thresholds, "nodes"))
    directions = tuple(str(value) for value in _tuple_setting(
        thresholds, "transfer_directions"
    ))
    randomized_controls = tuple(str(value) for value in _tuple_setting(
        thresholds, "required_randomized_controls"
    ))
    for name in (
        "minimum_node_patients",
        "minimum_source_states",
        "minimum_state_patients",
        "minimum_state_pixels",
    ):
        _number(thresholds, name, minimum=1)
    for name in (
        "maximum_reconstruction_error",
        "maximum_restore_error",
        "maximum_macro_tv",
        "maximum_worst_tv",
        "maximum_worst_tv_ci_high",
        "maximum_shared_increment",
        "maximum_shared_increment_ci_high",
        "minimum_specificity_delta",
        "minimum_specificity_ci_low",
        "maximum_null_clean_tv",
        "minimum_order_advantage",
    ):
        _number(thresholds, name, minimum=0.0)
    for name in (
        "minimum_state_realization",
        "minimum_downstream_reliable_fraction",
        "holm_alpha",
        "minimum_dose_aligned_fraction",
    ):
        _number(thresholds, name, minimum=0.0, maximum=1.0)
    if metrics.empty:
        raise ValueError("metrics must be nonempty")

    passed: list[str] = []
    audit_failures = [
        {"stage": "audit", "check": name, "reason": "missing_or_false"}
        for name in _REQUIRED_AUDIT_FLAGS
        if audit.get(name) is not True
    ]
    if audit_failures:
        return _report(
            ConclusionStatus.FAILED_AUDIT,
            failed=audit_failures,
            passed=passed,
        )
    passed.append("audit")
    if audit.get("history_status") == "HIGH_LEVEL_MODEL_MISSPECIFIED":
        return _report(
            ConclusionStatus.HIGH_LEVEL_MODEL_MISSPECIFIED,
            failed=[
                {
                    "stage": "high_level_model",
                    "reason": "first_order_history_dependence_failed",
                }
            ],
            passed=passed,
        )
    if audit.get("history_status") != "PASS":
        return _report(
            ConclusionStatus.FAILED_AUDIT,
            failed=[
                {
                    "stage": "high_level_model",
                    "reason": "missing_history_admission",
                }
            ],
            passed=passed,
        )
    passed.append("first_order_admission")

    cells = tuple(
        (architecture, seed, node)
        for architecture in architectures
        for seed in seeds
        for node in nodes
    )
    support_failures: list[dict[str, Any]] = []
    for architecture, seed, node in cells:
        identity = _identity(architecture, seed, node)
        coverage = _rows_for_cell(
            metrics,
            kind="coverage",
            architecture=architecture,
            seed=seed,
            node=node,
        )
        if "evaluable" in coverage:
            coverage = coverage[coverage["evaluable"].astype(bool)]
        valid_states = []
        for row in coverage.itertuples(index=False):
            patient_count = getattr(row, "patient_count", np.nan)
            pixel_count = getattr(row, "pixel_count", np.nan)
            if (
                np.isfinite(patient_count)
                and np.isfinite(pixel_count)
                and float(patient_count) >= float(thresholds["minimum_state_patients"])
                and float(pixel_count) >= float(thresholds["minimum_state_pixels"])
            ):
                valid_states.append(int(row.source_state))
        node_patient_column = (
            "node_patient_count"
            if "node_patient_count" in coverage
            else "patient_count"
        )
        maximum_patients = (
            float(pd.to_numeric(coverage[node_patient_column], errors="coerce").max())
            if not coverage.empty and node_patient_column in coverage
            else float("nan")
        )
        if (
            len(set(valid_states)) < int(thresholds["minimum_source_states"])
            or not np.isfinite(maximum_patients)
            or maximum_patients < float(thresholds["minimum_node_patients"])
        ):
            support_failures.append(
                {
                    "stage": "coverage",
                    **identity,
                    "reason": "insufficient_state_or_patient_coverage",
                    "valid_source_state_count": len(set(valid_states)),
                    "maximum_patient_count": maximum_patients,
                }
            )
        reliability = _rows_for_cell(
            metrics,
            kind="reliability",
            architecture=architecture,
            seed=seed,
            node=node,
        )
        if len(reliability) != 1:
            support_failures.append(
                {
                    "stage": "reliability",
                    **identity,
                    "reason": "missing_or_duplicate_reliability_row",
                }
            )
        else:
            fraction = _finite_row_value(
                reliability.iloc[0], "minimum_reliable_fraction"
            )
            if (
                fraction is None
                or fraction
                < float(thresholds["minimum_downstream_reliable_fraction"])
            ):
                support_failures.append(
                    {
                        "stage": "reliability",
                        **identity,
                        "reason": "selective_downstream_reliability_failed",
                        "minimum_reliable_fraction": fraction,
                    }
                )
        operator = _rows_for_cell(
            metrics,
            kind="operator",
            architecture=architecture,
            seed=seed,
            node=node,
        )
        if len(operator) != 1:
            support_failures.append(
                {
                    "stage": "operator",
                    **identity,
                    "reason": "missing_or_duplicate_operator_row",
                }
            )
            continue
        row = operator.iloc[0]
        reconstruction = _finite_row_value(row, "reconstruction_error")
        realization = _finite_row_value(row, "state_realization")
        restore = _finite_row_value(row, "restore_error")
        if (
            reconstruction is None
            or realization is None
            or restore is None
            or reconstruction > float(thresholds["maximum_reconstruction_error"])
            or realization < float(thresholds["minimum_state_realization"])
            or restore > float(thresholds["maximum_restore_error"])
            or not bool(row.get("ood_pass", False))
        ):
            support_failures.append(
                {
                    "stage": "operator",
                    **identity,
                    "reason": "operator_or_ood_gate_failed",
                }
            )
    if support_failures:
        return _report(
            ConclusionStatus.INSUFFICIENT_INTERVENTION_SUPPORT,
            failed=support_failures,
            passed=passed,
        )
    passed.append("coverage_and_operator")

    within_failures = _fidelity_failures(
        metrics,
        evaluation="within",
        cells=cells,
        thresholds=thresholds,
    )
    if within_failures:
        return _report(
            ConclusionStatus.PARTIAL_NODE_RANGE_ONLY,
            failed=within_failures,
            passed=passed,
        )
    passed.append("within_architecture_full_path")

    control_failures: list[dict[str, Any]] = []
    for architecture, seed, node in cells:
        identity = _identity(architecture, seed, node)
        specificity = _rows_for_cell(
            controls,
            kind="specificity",
            architecture=architecture,
            seed=seed,
            node=node,
        )
        if len(specificity) != 1:
            control_failures.append(
                {
                    "stage": "specificity",
                    **identity,
                    "reason": "missing_or_duplicate_specificity_row",
                }
            )
        else:
            row = specificity.iloc[0]
            delta = _finite_row_value(row, "delta_gain")
            delta_low = _finite_row_value(row, "delta_ci_low")
            null_tv = _finite_row_value(row, "null_clean_tv")
            if (
                delta is None
                or delta_low is None
                or null_tv is None
                or delta < float(thresholds["minimum_specificity_delta"])
                or delta_low <= float(thresholds["minimum_specificity_ci_low"])
                or null_tv > float(thresholds["maximum_null_clean_tv"])
            ):
                control_failures.append(
                    {
                        "stage": "specificity",
                        **identity,
                        "reason": "task_edit_not_specific",
                    }
                )
        order = _rows_for_cell(
            controls,
            kind="order",
            architecture=architecture,
            seed=seed,
            node=node,
        )
        if len(order) != 1:
            control_failures.append(
                {
                    "stage": "order_control",
                    **identity,
                    "reason": "missing_or_duplicate_order_row",
                }
            )
        else:
            row = order.iloc[0]
            advantage = _finite_row_value(row, "advantage")
            holm_p = _finite_row_value(row, "holm_p")
            if (
                advantage is None
                or holm_p is None
                or advantage < float(thresholds["minimum_order_advantage"])
                or holm_p >= float(thresholds["holm_alpha"])
            ):
                control_failures.append(
                    {
                        "stage": "order_control",
                        **identity,
                        "reason": "correct_depth_order_not_specific",
                    }
                )
        randomized = _rows_for_cell(
            controls,
            kind="randomized",
            architecture=architecture,
            seed=seed,
            node=node,
        )
        for control_name in randomized_controls:
            selected = (
                randomized[randomized["control_name"] == control_name]
                if "control_name" in randomized
                else randomized.iloc[0:0]
            )
            if len(selected) != 1 or not bool(
                selected.iloc[0].get("separation_pass", False)
            ):
                control_failures.append(
                    {
                        "stage": "randomized_control",
                        **identity,
                        "control_name": control_name,
                        "reason": "control_not_separated",
                    }
                )
        dose = _rows_for_cell(
            metrics,
            kind="dose",
            architecture=architecture,
            seed=seed,
            node=node,
        )
        if len(dose) != 1:
            control_failures.append(
                {
                    "stage": "dose",
                    **identity,
                    "reason": "missing_or_duplicate_dose_row",
                }
            )
        else:
            row = dose.iloc[0]
            fraction = _finite_row_value(row, "aligned_fraction")
            reverse = _finite_row_value(row, "reverse_condition_count")
            if (
                fraction is None
                or reverse is None
                or fraction < float(thresholds["minimum_dose_aligned_fraction"])
                or reverse != 0.0
            ):
                control_failures.append(
                    {
                        "stage": "dose",
                        **identity,
                        "reason": "dose_direction_failed",
                    }
                )
    if control_failures:
        return _report(
            ConclusionStatus.OBSERVATIONAL_PROCESS_ONLY,
            failed=control_failures,
            passed=passed,
        )
    passed.append("nonempty_intervention_controls")

    transfer_failures: list[dict[str, Any]] = []
    for direction in directions:
        target = direction.split("->")[-1]
        direction_cells = tuple((target, seed, node) for seed in seeds for node in nodes)
        transfer_failures.extend(
            _fidelity_failures(
                metrics,
                evaluation="cross",
                cells=direction_cells,
                thresholds=thresholds,
                direction=direction,
            )
        )
    if transfer_failures:
        return _report(
            ConclusionStatus.ARCHITECTURE_SPECIFIC_PROCESS_ONLY,
            failed=transfer_failures,
            passed=passed,
        )
    passed.append("bidirectional_cross_architecture_transfer")

    shared_failures = _fidelity_failures(
        metrics,
        evaluation="shared",
        cells=cells,
        thresholds=thresholds,
    )
    for architecture, seed, node in cells:
        selected = _rows_for_cell(
            metrics,
            kind="shared_noninferiority",
            architecture=architecture,
            seed=seed,
            node=node,
        )
        identity = {
            "stage": "shared_noninferiority",
            **_identity(architecture, seed, node),
        }
        if len(selected) != 1:
            shared_failures.append(
                {**identity, "reason": "missing_or_duplicate_shared_row"}
            )
            continue
        row = selected.iloc[0]
        increment = _finite_row_value(row, "tv_increment")
        ci_high = _finite_row_value(row, "ci_high")
        if (
            increment is None
            or ci_high is None
            or increment > float(thresholds["maximum_shared_increment"])
            or ci_high > float(thresholds["maximum_shared_increment_ci_high"])
        ):
            shared_failures.append(
                {
                    **identity,
                    "reason": "shared_process_noninferiority_failed",
                    "tv_increment": increment,
                    "ci_high": ci_high,
                }
            )
    if shared_failures:
        return _report(
            ConclusionStatus.BIDIRECTIONAL_TRANSFER_ONLY,
            failed=shared_failures,
            passed=passed,
        )
    passed.append("shared_process_noninferiority")

    structure = metrics[metrics["kind"] == "structure"]
    structural = False
    if len(structure) == 1:
        row = structure.iloc[0]
        delta = _finite_row_value(row, "delta_distance")
        ci_low = _finite_row_value(row, "ci_low")
        node_count = _finite_row_value(row, "reported_node_count")
        structural = bool(
            delta is not None
            and ci_low is not None
            and node_count is not None
            and delta > 0.0
            and ci_low > 0.0
            and node_count >= len(nodes)
            and _finite_row_value(row, "patient_count") is not None
            and float(row["patient_count"])
            >= float(thresholds["minimum_node_patients"])
        )
    structure_failures: list[dict[str, Any]] = []
    if structural:
        passed.append("structural_diagnostic_sensitivity")
    else:
        structure_failures.append(
            {
                "stage": "structural_control",
                "reason": "baseline_noskip_did_not_exceed_seed_variability",
            }
        )
    return _report(
        (
            ConclusionStatus.PASS_SHARED_FULL_NETWORK
            if structural
            else ConclusionStatus.PASS_SHARED_FULL_NETWORK_NO_STRUCTURAL_SENSITIVITY
        ),
        failed=structure_failures,
        passed=passed,
        structural=structural,
    )


__all__ = [
    "ConclusionStatus",
    "GateReport",
    "evaluate_causal_abstraction_gate",
]
