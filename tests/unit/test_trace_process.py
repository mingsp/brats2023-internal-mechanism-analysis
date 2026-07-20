from __future__ import annotations

import numpy as np

from pptt.causal_abstraction.states import RelationshipState
from pptt.causal_abstraction.trace_process import (
    case_trace_history_rows,
    case_trace_transition_rows,
    node_reliability,
    relationship_state_path,
)
from pptt.io.artifacts import CaseTrace


def _trace() -> CaseTrace:
    truth = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    stages = np.stack(
        (
            np.array([[0, 0], [2, 1]], dtype=np.uint8),
            np.array([[0, 1], [2, 1]], dtype=np.uint8),
            np.array([[0, 1], [2, 3]], dtype=np.uint8),
            np.array([[0, 1], [2, 3]], dtype=np.uint8),
            np.array([[0, 1], [2, 3]], dtype=np.uint8),
            np.array([[0, 1], [2, 3]], dtype=np.uint8),
            np.array([[0, 1], [2, 3]], dtype=np.uint8),
            np.array([[0, 1], [2, 3]], dtype=np.uint8),
        )
    )
    reliable = np.ones((7, 2, 2), dtype=bool)
    reliable[0, 1, 1] = False
    final = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    return CaseTrace(
        states=stages,
        reliable=reliable,
        truth=truth,
        final_model_state=final,
    )


def test_trace_maps_nodes_and_output_to_relationship_states():
    path = relationship_state_path(_trace(), num_classes=4)

    assert path.shape == (9, 2, 2)
    assert path[0, 0, 1] == RelationshipState.FN
    assert path[0, 1, 1] == RelationshipState.FW
    assert path[2, 1, 1] == RelationshipState.FC
    assert np.count_nonzero(path[-1] == RelationshipState.FC) == 3
    assert path[-1, 0, 0] == RelationshipState.BC


def test_transition_rows_include_all_eight_transitions_and_patient_identity():
    rows = case_trace_transition_rows(
        _trace(),
        patient_id="p1",
        model_seed=42,
    )

    assert set(rows["transition_index"]) == set(range(8))
    assert set(rows["patient_id"]) == {"p1"}
    assert set(rows["model_seed"]) == {42}
    assert rows.groupby("transition_index")["count"].sum().loc[0] == 3
    assert rows.groupby("transition_index")["count"].sum().loc[7] == 4


def test_history_rows_start_at_second_transition_and_use_joint_reliability():
    rows = case_trace_history_rows(
        _trace(),
        patient_id="p1",
        model_seed=123,
    )

    assert set(rows["transition_index"]) == set(range(1, 8))
    assert "previous_state" in rows
    assert rows.groupby("transition_index")["count"].sum().loc[1] == 3
    assert rows.groupby("transition_index")["count"].sum().loc[7] == 4


def test_internal_node_reliability_requires_both_adjacent_pairs():
    trace = _trace()

    assert node_reliability(trace, 0).sum() == 3
    assert node_reliability(trace, 1).sum() == 3
    assert node_reliability(trace, 7).sum() == 4
