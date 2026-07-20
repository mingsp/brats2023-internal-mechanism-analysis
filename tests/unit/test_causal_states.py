from __future__ import annotations

import numpy as np
import pytest

from pptt.causal_abstraction.states import (
    ProcessEvent,
    RelationshipState,
    process_events,
    relationship_states,
)


def test_relationship_states_preserve_pixel_identity_and_error_type():
    truth = np.array([[0, 1, 0, 2, 3]], dtype=np.uint8)
    prediction = np.array([[0, 1, 2, 0, 2]], dtype=np.uint8)

    states = relationship_states(truth, prediction, num_classes=4)

    np.testing.assert_array_equal(
        states,
        np.array(
            [[
                RelationshipState.BC,
                RelationshipState.FC,
                RelationshipState.FP,
                RelationshipState.FN,
                RelationshipState.FW,
            ]],
            dtype=np.uint8,
        ),
    )


def test_process_events_are_exhaustive_and_determined_by_adjacent_states():
    before = np.array(
        [
            RelationshipState.FN,
            RelationshipState.FC,
            RelationshipState.FN,
            RelationshipState.FP,
            RelationshipState.BC,
            RelationshipState.FW,
        ],
        dtype=np.uint8,
    )
    after = np.array(
        [
            RelationshipState.FC,
            RelationshipState.FN,
            RelationshipState.FW,
            RelationshipState.BC,
            RelationshipState.BC,
            RelationshipState.FW,
        ],
        dtype=np.uint8,
    )

    events = process_events(before, after)

    np.testing.assert_array_equal(
        events,
        np.array(
            [
                ProcessEvent.CORRECTION,
                ProcessEvent.DESTRUCTION,
                ProcessEvent.ERROR_RECODING,
                ProcessEvent.CORRECTION,
                ProcessEvent.PERSISTENCE,
                ProcessEvent.PERSISTENCE,
            ],
            dtype=np.uint8,
        ),
    )


@pytest.mark.parametrize(
    ("truth", "prediction", "message"),
    [
        (
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((2, 3), dtype=np.uint8),
            "identical shape",
        ),
        (
            np.array([[0, 4]], dtype=np.uint8),
            np.array([[0, 1]], dtype=np.uint8),
            "outside",
        ),
        (
            np.array([[0.0, 1.0]], dtype=np.float32),
            np.array([[0, 1]], dtype=np.uint8),
            "integer",
        ),
    ],
)
def test_relationship_states_reject_invalid_labels(
    truth: np.ndarray,
    prediction: np.ndarray,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        relationship_states(truth, prediction, num_classes=4)


def test_process_events_reject_invalid_relationship_state():
    before = np.array([RelationshipState.BC, 5], dtype=np.uint8)
    after = np.array([RelationshipState.FC, RelationshipState.FC], dtype=np.uint8)

    with pytest.raises(ValueError, match="relationship state"):
        process_events(before, after)
