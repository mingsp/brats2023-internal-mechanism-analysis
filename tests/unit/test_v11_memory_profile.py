from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.profile_v11_causal_abstraction_memory import _select_profile_patient


def _plan(patient_id: str, pair_counts: list[int]) -> dict:
    return {
        "patient_id": patient_id,
        "nodes": [
            {"node": f"n{index}", "pairs": [{} for _ in range(count)]}
            for index, count in enumerate(pair_counts)
        ],
    }


def test_memory_profile_prefers_broad_node_coverage_then_pair_count(tmp_path: Path):
    plans = {
        "p1": _plan("p1", [4, 4, 4, 0]),
        "p2": _plan("p2", [1, 1, 1, 1]),
        "p3": _plan("p3", [2, 2, 2, 2]),
    }
    for patient_id, payload in plans.items():
        (tmp_path / f"{patient_id}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    patient_id, payload = _select_profile_patient(tmp_path, list(reversed(plans)))

    assert patient_id == "p3"
    assert payload == plans["p3"]


def test_memory_profile_rejects_registry_without_matches(tmp_path: Path):
    payload = _plan("p1", [0, 0])
    (tmp_path / "p1.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="no matched intervention pairs"):
        _select_profile_patient(tmp_path, ["p1"])
