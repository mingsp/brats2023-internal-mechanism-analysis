from __future__ import annotations

import numpy as np
import pandas as pd

from pptt.causal_abstraction.planning import (
    build_balanced_base_requests,
    prune_matches_to_independent_resize_rows,
    select_node_slices,
)
from pptt.io.artifacts import CaseTrace


NODES = (
    "down1",
    "down2",
    "down3",
    "down4",
    "up1",
    "up2",
    "up3",
    "up4",
)


def _selection_trace() -> CaseTrace:
    truth = np.zeros((2, 2, 2), dtype=np.uint8)
    truth[1, 0, 0] = 1
    truth[1, 0, 1] = 1
    states = np.zeros((8, 2, 2, 2), dtype=np.uint8)
    states[:, 1, 0, 0] = 1
    states[:, 1, 0, 1] = 0
    states[:, 1, 1, 1] = 1
    reliable = np.ones((7, 2, 2, 2), dtype=bool)
    return CaseTrace(
        states=states,
        reliable=reliable,
        truth=truth,
        final_model_state=states[-1],
        slice_ids=("p_000", "p_001"),
    )


def test_slice_selection_prefers_exchangeable_state_coverage():
    selected = select_node_slices(_selection_trace(), nodes=NODES)

    assert set(selected) == set(NODES)
    assert all(value["slice_id"] == "p_001" for value in selected.values())
    assert all(value["exchangeable_stratum_count"] >= 2 for value in selected.values())


def _candidate_frame() -> pd.DataFrame:
    rows = []
    states = [0, 2, 0, 2, 1, 3, 4, 1, 3, 4]
    truths = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    for index, (truth, state) in enumerate(zip(truths, states, strict=True)):
        rows.append(
            {
                "row_id": f"r{index}",
                "patient_id": "p1",
                "node": "down4",
                "truth_class": truth,
                "state": state,
                "output_index": index,
                "native_h": 2,
                "native_w": 2,
                "boundary_distance": 1.0,
                "spatial_scale": "2x2",
                "feature_norm": float(index + 1),
            }
        )
    return pd.DataFrame(rows)


def test_balanced_requests_are_deterministic_unique_and_nonidentity():
    candidates = _candidate_frame()
    first = build_balanced_base_requests(candidates, maximum_targets=6, seed=7)
    second = build_balanced_base_requests(candidates, maximum_targets=6, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert first["row_id"].is_unique
    assert first["output_index"].is_unique
    assert np.all(first["state"] != first["requested_source_state"])
    assert len(first) == 6


def test_rank_pruning_removes_duplicate_resize_rows():
    candidates = _candidate_frame().iloc[:3].copy()
    candidates.loc[:, "output_index"] = [0, 1, 2]
    matches = pd.DataFrame(
        {
            "status": ["MATCHED"] * 3,
            "patient_id": ["p1"] * 3,
            "node": ["down4"] * 3,
            "base_row_id": candidates["row_id"].tolist(),
        }
    )

    retained, diagnostics = prune_matches_to_independent_resize_rows(
        matches,
        candidates,
        output_shape=(4, 4),
    )

    assert 1 <= len(retained) <= 3
    assert diagnostics.loc[0, "retained_count"] == len(retained)
    assert diagnostics.loc[0, "spatial_rank"] == len(retained)
