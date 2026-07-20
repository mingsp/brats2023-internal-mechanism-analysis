from __future__ import annotations

from enum import IntEnum

import numpy as np
import numpy.typing as npt


class RelationshipState(IntEnum):
    """Relation between a pixel decision and its reference class."""

    BC = 0
    FC = 1
    FP = 2
    FN = 3
    FW = 4


class ProcessEvent(IntEnum):
    """Exhaustive event induced by two adjacent relationship states."""

    PERSISTENCE = 0
    CORRECTION = 1
    DESTRUCTION = 2
    ERROR_RECODING = 3


TRUTH_CLASS_NAMES = {
    0: "BG",
    1: "NCR_NET",
    2: "ED",
    3: "ET",
}


_CORRECT_STATES = np.array(
    [RelationshipState.BC, RelationshipState.FC],
    dtype=np.uint8,
)


def _validated_integer_array(value: npt.ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must have integer dtype, got {array.dtype}")
    return array


def relationship_states(
    truth: npt.ArrayLike,
    prediction: npt.ArrayLike,
    *,
    num_classes: int,
) -> np.ndarray:
    """Return one architecture-independent relationship state per pixel."""

    reference = _validated_integer_array(truth, name="truth")
    decision = _validated_integer_array(prediction, name="prediction")
    if reference.shape != decision.shape:
        raise ValueError(
            "truth and prediction must have identical shape, got "
            f"{reference.shape} and {decision.shape}"
        )
    if int(num_classes) < 2:
        raise ValueError("num_classes must be at least two")
    for name, array in (("truth", reference), ("prediction", decision)):
        if array.size and (array.min() < 0 or array.max() >= int(num_classes)):
            raise ValueError(
                f"{name} contains labels outside [0, {int(num_classes)})"
            )

    states = np.empty(reference.shape, dtype=np.uint8)
    truth_background = reference == 0
    prediction_background = decision == 0
    states[truth_background & prediction_background] = RelationshipState.BC
    states[truth_background & ~prediction_background] = RelationshipState.FP
    truth_foreground = ~truth_background
    states[truth_foreground & prediction_background] = RelationshipState.FN
    states[truth_foreground & (decision == reference)] = RelationshipState.FC
    states[
        truth_foreground
        & ~prediction_background
        & (decision != reference)
    ] = RelationshipState.FW
    return states


def process_events(
    before: npt.ArrayLike,
    after: npt.ArrayLike,
) -> np.ndarray:
    """Map adjacent relationship states to an exhaustive process event."""

    state_from = _validated_integer_array(before, name="before")
    state_to = _validated_integer_array(after, name="after")
    if state_from.shape != state_to.shape:
        raise ValueError(
            "before and after must have identical shape, got "
            f"{state_from.shape} and {state_to.shape}"
        )
    state_count = len(RelationshipState)
    for name, array in (("before", state_from), ("after", state_to)):
        if array.size and (array.min() < 0 or array.max() >= state_count):
            raise ValueError(f"{name} contains an invalid relationship state")

    before_correct = np.isin(state_from, _CORRECT_STATES)
    after_correct = np.isin(state_to, _CORRECT_STATES)
    events = np.full(
        state_from.shape,
        ProcessEvent.PERSISTENCE,
        dtype=np.uint8,
    )
    events[~before_correct & after_correct] = ProcessEvent.CORRECTION
    events[before_correct & ~after_correct] = ProcessEvent.DESTRUCTION
    events[
        ~before_correct
        & ~after_correct
        & (state_from != state_to)
    ] = ProcessEvent.ERROR_RECODING
    return events


__all__ = [
    "ProcessEvent",
    "RelationshipState",
    "TRUTH_CLASS_NAMES",
    "process_events",
    "relationship_states",
]
