from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def evaluate_causal_conclusion_gate(
    rows: Iterable[Mapping[str, Any]],
    *,
    dose_rows: Iterable[Mapping[str, Any]],
    negative_control_pass: bool,
    restoration_audit_pass: bool,
    minimum_supported_seeds: int,
    alpha: float,
    minimum_standardized_effect: float,
) -> dict[str, Any]:
    records = [dict(row) for row in rows]
    if minimum_supported_seeds < 1:
        raise ValueError("minimum_supported_seeds must be positive")
    endpoints = ("necessity", "restoration", "specificity")
    endpoint_audit: dict[str, dict[str, Any]] = {}
    for endpoint in endpoints:
        selected = [row for row in records if row.get("endpoint") == endpoint]
        supported = [
            row
            for row in selected
            if float(row["mean_difference"]) > 0
            and float(row["ci_low"]) > 0
            and float(row["holm_p"]) < float(alpha)
            and abs(float(row["paired_cohens_d"]))
            >= float(minimum_standardized_effect)
        ]
        endpoint_audit[endpoint] = {
            "evaluated_seed_count": len(selected),
            "supported_seed_count": len(supported),
            "supported_model_seeds": sorted(
                int(row["model_seed"]) for row in supported
            ),
            "passed": len(supported) >= minimum_supported_seeds,
        }
    dose_records = [dict(row) for row in dose_rows]
    dose_supported = sorted(
        int(row["model_seed"])
        for row in dose_records
        if bool(row.get("monotonic", False))
    )
    gates = {
        "necessity": endpoint_audit["necessity"]["passed"],
        "restoration": endpoint_audit["restoration"]["passed"],
        "specificity": endpoint_audit["specificity"]["passed"],
        "dose_response": len(dose_supported) >= minimum_supported_seeds,
        "no_skip_negative_control": bool(negative_control_pass),
        "activation_and_hook_restoration": bool(restoration_audit_pass),
    }
    return {
        "passed": all(gates.values()),
        "minimum_supported_seeds": int(minimum_supported_seeds),
        "alpha": float(alpha),
        "minimum_standardized_effect": float(minimum_standardized_effect),
        "endpoint_audit": endpoint_audit,
        "dose_supported_model_seeds": dose_supported,
        "gates": gates,
        "allowed_claim_if_passed": (
            "Under the registered high-level pixel states, low-level up2 skip "
            "variable, and spatial corruption/restoration operators, the selected "
            "path has a replicated interventionally faithful contribution to "
            "persistent tumor-pixel correction."
        ),
        "claim_if_failed": (
            "The selected path is process-sensitive under the registered "
            "intervention, without complete causal-faithfulness support."
        ),
    }


__all__ = ["evaluate_causal_conclusion_gate"]
