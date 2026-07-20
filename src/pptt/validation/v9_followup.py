from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_UNINTERVENED_STATUS = "INELIGIBLE_TARGET_FEATURE_POSITIONS"


def derive_common_unintervened_cohort(
    rows_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    required_seeds: Sequence[int],
    minimum_patient_count: int,
) -> tuple[str, ...]:
    """Return patients that exited before every registered V8 intervention."""
    seeds = tuple(int(seed) for seed in required_seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("required_seeds must be a nonempty unique sequence")
    if set(int(seed) for seed in rows_by_seed) != set(seeds):
        raise ValueError("rows_by_seed differs from the required seed registry")
    if int(minimum_patient_count) < 1:
        raise ValueError("minimum_patient_count must be positive")

    unintervened_sets: list[set[str]] = []
    for seed in seeds:
        rows = [dict(row) for row in rows_by_seed[seed]]
        patient_ids = [str(row.get("patient_id", "")) for row in rows]
        if any(not value for value in patient_ids) or len(patient_ids) != len(
            set(patient_ids)
        ):
            raise ValueError(f"seed {seed} has invalid or duplicate patient ids")
        current: set[str] = set()
        for row in rows:
            if str(row.get("status")) != _UNINTERVENED_STATUS:
                continue
            if row.get("conditions") or row.get("effects") or row.get("audit"):
                raise ValueError(
                    f"patient {row['patient_id']} contains prior intervention data"
                )
            current.add(str(row["patient_id"]))
        unintervened_sets.append(current)

    common = tuple(sorted(set.intersection(*unintervened_sets)))
    if len(common) < int(minimum_patient_count):
        raise ValueError(
            "common unintervened cohort is smaller than the registered minimum"
        )
    return common


__all__ = ["derive_common_unintervened_cohort"]
