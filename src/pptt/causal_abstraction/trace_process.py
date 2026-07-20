from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from pptt.causal_abstraction.states import relationship_states
from pptt.io.artifacts import CaseTrace, load_case_trace


def relationship_state_path(
    trace: CaseTrace,
    *,
    num_classes: int,
) -> np.ndarray:
    """Map eight observer decisions plus the model output to five relation states."""

    if trace.truth is None or trace.final_model_state is None:
        raise ValueError("case trace requires truth and final_model_state")
    if trace.states.shape[0] != 8:
        raise ValueError("case trace must contain exactly eight registered nodes")
    truth = np.asarray(trace.truth)
    stages = [
        relationship_states(truth, trace.states[index], num_classes=num_classes)
        for index in range(8)
    ]
    stages.append(
        relationship_states(
            truth,
            trace.final_model_state,
            num_classes=num_classes,
        )
    )
    return np.stack(stages).astype(np.uint8, copy=False)


def node_reliability(trace: CaseTrace, node_index: int) -> np.ndarray:
    """Return a conservative restart-reliability mask for one registered node."""

    index = int(node_index)
    if trace.states.shape[0] != 8 or trace.reliable.shape[0] != 7:
        raise ValueError("case trace must contain eight nodes and seven pair masks")
    if index < 0 or index >= 8:
        raise ValueError("node_index must be in [0, 8)")
    if index == 0:
        return np.asarray(trace.reliable[0], dtype=bool)
    if index == 7:
        return np.asarray(trace.reliable[6], dtype=bool)
    return np.asarray(trace.reliable[index - 1] & trace.reliable[index], dtype=bool)


def transition_reliability(trace: CaseTrace, transition_index: int) -> np.ndarray:
    """Return the registered reliability mask for a node-to-node/output transition."""

    index = int(transition_index)
    if index < 0 or index >= 8:
        raise ValueError("transition_index must be in [0, 8)")
    if index < 7:
        return np.asarray(trace.reliable[index], dtype=bool)
    return node_reliability(trace, 7)


def case_trace_transition_rows(
    trace: CaseTrace,
    *,
    patient_id: str,
    model_seed: int,
    num_classes: int = 4,
) -> pd.DataFrame:
    """Reduce one volume to patient-identified first-order transition counts."""

    if not str(patient_id):
        raise ValueError("patient_id must be nonempty")
    path = relationship_state_path(trace, num_classes=int(num_classes))
    rows: list[dict[str, int | str]] = []
    state_count = 5
    for transition_index in range(8):
        reliable = transition_reliability(trace, transition_index)
        before = path[transition_index][reliable].astype(np.int64, copy=False)
        after = path[transition_index + 1][reliable].astype(np.int64, copy=False)
        if before.size == 0:
            continue
        encoded = before * state_count + after
        counts = np.bincount(encoded, minlength=state_count * state_count)
        for flat_index in np.flatnonzero(counts):
            rows.append(
                {
                    "patient_id": str(patient_id),
                    "model_seed": int(model_seed),
                    "transition_index": transition_index,
                    "state_from": int(flat_index // state_count),
                    "state_to": int(flat_index % state_count),
                    "count": int(counts[flat_index]),
                }
            )
    return pd.DataFrame(
        rows,
        columns=(
            "patient_id",
            "model_seed",
            "transition_index",
            "state_from",
            "state_to",
            "count",
        ),
    )


def case_trace_history_rows(
    trace: CaseTrace,
    *,
    patient_id: str,
    model_seed: int,
    num_classes: int = 4,
) -> pd.DataFrame:
    """Reduce one volume to patient-identified second-order transition counts."""

    if not str(patient_id):
        raise ValueError("patient_id must be nonempty")
    path = relationship_state_path(trace, num_classes=int(num_classes))
    rows: list[dict[str, int | str]] = []
    state_count = 5
    for transition_index in range(1, 8):
        reliable = transition_reliability(trace, transition_index - 1) & transition_reliability(
            trace,
            transition_index,
        )
        previous = path[transition_index - 1][reliable].astype(np.int64, copy=False)
        before = path[transition_index][reliable].astype(np.int64, copy=False)
        after = path[transition_index + 1][reliable].astype(np.int64, copy=False)
        if previous.size == 0:
            continue
        encoded = (previous * state_count + before) * state_count + after
        counts = np.bincount(encoded, minlength=state_count**3)
        for flat_index in np.flatnonzero(counts):
            previous_state, remainder = divmod(int(flat_index), state_count**2)
            state_from, state_to = divmod(remainder, state_count)
            rows.append(
                {
                    "patient_id": str(patient_id),
                    "model_seed": int(model_seed),
                    "transition_index": transition_index,
                    "previous_state": previous_state,
                    "state_from": state_from,
                    "state_to": state_to,
                    "count": int(counts[flat_index]),
                }
            )
    return pd.DataFrame(
        rows,
        columns=(
            "patient_id",
            "model_seed",
            "transition_index",
            "previous_state",
            "state_from",
            "state_to",
            "count",
        ),
    )


def collect_case_trace_rows(
    trace_paths: Sequence[str | Path],
    *,
    model_seed: int,
    num_classes: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load an ordered patient registry and return first-/second-order rows."""

    paths = [Path(value) for value in trace_paths]
    if not paths or paths != sorted(paths, key=lambda value: value.stem):
        raise ValueError("trace paths must be a nonempty patient-sorted registry")
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError("trace paths contain duplicate patient identifiers")
    transitions: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        trace = load_case_trace(path)
        transitions.append(
            case_trace_transition_rows(
                trace,
                patient_id=path.stem,
                model_seed=int(model_seed),
                num_classes=int(num_classes),
            )
        )
        histories.append(
            case_trace_history_rows(
                trace,
                patient_id=path.stem,
                model_seed=int(model_seed),
                num_classes=int(num_classes),
            )
        )
    return (
        pd.concat(transitions, ignore_index=True),
        pd.concat(histories, ignore_index=True),
    )


__all__ = [
    "case_trace_history_rows",
    "case_trace_transition_rows",
    "collect_case_trace_rows",
    "node_reliability",
    "relationship_state_path",
    "transition_reliability",
]
