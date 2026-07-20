from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def _supported_seed(
    row: Mapping[str, Any],
    *,
    alpha: float,
    minimum_standardized_effect: float,
    minimum_patients_per_seed: int,
) -> bool:
    values = np.asarray(
        [
            row.get("mean_difference"),
            row.get("ci_low"),
            row.get("holm_p"),
            row.get("paired_cohens_d"),
        ],
        dtype=np.float64,
    )
    return bool(
        int(row.get("patient_count", 0)) >= int(minimum_patients_per_seed)
        and np.isfinite(values).all()
        and values[0] > 0
        and values[1] > 0
        and values[2] < float(alpha)
        and values[3] >= float(minimum_standardized_effect)
    )


def evaluate_replication_gate(
    rows: Iterable[Mapping[str, Any]],
    *,
    dose_rows: Iterable[Mapping[str, Any]],
    operator_rows: Iterable[Mapping[str, Any]],
    minimum_supported_seeds: int,
    minimum_patients_per_seed: int,
    alpha: float,
    minimum_standardized_effect: float,
    minimum_patient_monotonic_fraction: float = 0.0,
) -> dict[str, Any]:
    if int(minimum_supported_seeds) < 1:
        raise ValueError("minimum_supported_seeds must be positive")
    if int(minimum_patients_per_seed) < 2:
        raise ValueError("minimum_patients_per_seed must be at least two")
    if not 0.0 <= float(minimum_patient_monotonic_fraction) <= 1.0:
        raise ValueError("minimum_patient_monotonic_fraction must be in [0, 1]")
    endpoints = ("necessity", "restoration", "specificity")
    records = [dict(row) for row in rows]
    endpoint_audit: dict[str, dict[str, Any]] = {}
    for endpoint in endpoints:
        selected = [row for row in records if row.get("endpoint") == endpoint]
        if not selected:
            raise ValueError(f"registered endpoint is missing: {endpoint}")
        seeds = [int(row["model_seed"]) for row in selected]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate model seed for endpoint {endpoint}")
        supported = sorted(
            int(row["model_seed"])
            for row in selected
            if _supported_seed(
                row,
                alpha=float(alpha),
                minimum_standardized_effect=float(minimum_standardized_effect),
                minimum_patients_per_seed=int(minimum_patients_per_seed),
            )
        )
        endpoint_audit[endpoint] = {
            "evaluated_seed_count": len(selected),
            "supported_seed_count": len(supported),
            "supported_model_seeds": supported,
            "passed": len(supported) >= int(minimum_supported_seeds),
        }
    dose_records = [dict(row) for row in dose_rows]
    operator_records = [dict(row) for row in operator_rows]
    dose_supported = sorted(
        int(row["model_seed"])
        for row in dose_records
        if bool(row.get("monotonic", False))
        and int(row.get("patient_count", 0)) >= int(minimum_patients_per_seed)
        and float(row.get("patient_monotonic_fraction", 1.0))
        >= float(minimum_patient_monotonic_fraction)
    )
    operator_supported = sorted(
        int(row["model_seed"])
        for row in operator_records
        if bool(row.get("passed", False))
    )
    gates = {
        **{
            endpoint: bool(endpoint_audit[endpoint]["passed"])
            for endpoint in endpoints
        },
        "dose_response": len(set(dose_supported))
        >= int(minimum_supported_seeds),
        "operator_audit": len(set(operator_supported))
        >= int(minimum_supported_seeds),
    }
    passed = all(gates.values())
    return {
        "passed": passed,
        "status": (
            "INTERVENTIONALLY_FAITHFUL_TRANSUNET_REPLICATION"
            if passed
            else "TRANSUNET_PROCESS_CANDIDATE_NOT_CAUSALLY_CONFIRMED"
        ),
        "minimum_supported_seeds": int(minimum_supported_seeds),
        "minimum_patients_per_seed": int(minimum_patients_per_seed),
        "alpha": float(alpha),
        "minimum_standardized_effect": float(minimum_standardized_effect),
        "minimum_patient_monotonic_fraction": float(
            minimum_patient_monotonic_fraction
        ),
        "endpoint_audit": endpoint_audit,
        "dose_supported_model_seeds": sorted(set(dose_supported)),
        "operator_supported_model_seeds": sorted(set(operator_supported)),
        "gates": gates,
        "allowed_claim_if_passed": (
            "PPTT prospectively generated and interventionally confirmed one "
            "architecture-specific process-structure candidate in TransUNet "
            "under the registered variables and operators."
        ),
        "claim_if_failed": (
            "PPTT generated a TransUNet process candidate that was not fully "
            "confirmed by the registered intervention family."
        ),
    }


__all__ = ["evaluate_replication_gate"]
