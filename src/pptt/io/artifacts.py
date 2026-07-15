from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np


@dataclass(frozen=True)
class CaseTrace:
    states: np.ndarray
    reliable: np.ndarray
    truth: np.ndarray | None = None
    final_model_state: np.ndarray | None = None
    slice_ids: tuple[str, ...] = ()


def _validated_trace(
    *,
    states: np.ndarray,
    reliable: np.ndarray,
    truth: np.ndarray | None,
    final_model_state: np.ndarray | None,
    slice_ids: tuple[str, ...],
) -> CaseTrace:
    state_array = np.asarray(states)
    reliable_array = np.asarray(reliable)
    if state_array.ndim < 3 or state_array.shape[0] < 2:
        raise ValueError("states must contain at least two stages and spatial axes")
    if not np.issubdtype(state_array.dtype, np.integer):
        raise ValueError("states must have integer dtype")
    expected_reliable_shape = (state_array.shape[0] - 1, *state_array.shape[1:])
    if reliable_array.shape != expected_reliable_shape or reliable_array.dtype != np.bool_:
        raise ValueError(
            f"reliable must be boolean with shape {expected_reliable_shape}"
        )
    truth_array = None if truth is None else np.asarray(truth)
    if truth_array is not None:
        if truth_array.shape != state_array.shape[1:]:
            raise ValueError("truth must match one complete stage")
        if not np.issubdtype(truth_array.dtype, np.integer):
            raise ValueError("truth must have integer dtype")
    final_array = None if final_model_state is None else np.asarray(final_model_state)
    if final_array is not None:
        if final_array.shape != state_array.shape[1:]:
            raise ValueError("final_model_state must match one complete stage")
        if not np.issubdtype(final_array.dtype, np.integer):
            raise ValueError("final_model_state must have integer dtype")
    if slice_ids:
        if state_array.ndim != 4 or len(slice_ids) != state_array.shape[1]:
            raise ValueError("slice_ids require KxSxHxW states and one id per slice")
        if len(slice_ids) != len(set(slice_ids)):
            raise ValueError("slice_ids must be unique")
    return CaseTrace(
        states=state_array.astype(np.uint8, copy=False),
        reliable=reliable_array,
        truth=None if truth_array is None else truth_array.astype(np.uint8, copy=False),
        final_model_state=(
            None if final_array is None else final_array.astype(np.uint8, copy=False)
        ),
        slice_ids=tuple(str(value) for value in slice_ids),
    )


def save_case_trace(
    path: str | Path,
    *,
    states: np.ndarray,
    reliable: np.ndarray,
    truth: np.ndarray | None = None,
    final_model_state: np.ndarray | None = None,
    slice_ids: tuple[str, ...] = (),
) -> None:
    trace = _validated_trace(
        states=states,
        reliable=reliable,
        truth=truth,
        final_model_state=final_model_state,
        slice_ids=slice_ids,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "states": trace.states,
        "reliable": trace.reliable,
        "slice_ids": np.asarray(trace.slice_ids),
        "has_truth": np.asarray(trace.truth is not None),
        "has_final_model_state": np.asarray(trace.final_model_state is not None),
    }
    if trace.truth is not None:
        payload["truth"] = trace.truth
    if trace.final_model_state is not None:
        payload["final_model_state"] = trace.final_model_state
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".npz",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_case_trace(path: str | Path) -> CaseTrace:
    with np.load(Path(path), allow_pickle=False) as archive:
        has_truth = bool(archive["has_truth"].item())
        has_final = bool(archive["has_final_model_state"].item())
        return _validated_trace(
            states=archive["states"],
            reliable=archive["reliable"],
            truth=archive["truth"] if has_truth else None,
            final_model_state=archive["final_model_state"] if has_final else None,
            slice_ids=tuple(str(value) for value in archive["slice_ids"].tolist()),
        )
