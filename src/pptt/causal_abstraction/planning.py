from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
import torch

from pptt.causal_abstraction.interventions import bilinear_resize_rows
from pptt.causal_abstraction.states import RelationshipState
from pptt.causal_abstraction.trace_process import (
    node_reliability,
    relationship_state_path,
)
from pptt.io.artifacts import CaseTrace


_TARGET_STATES_BY_TRUTH = {
    0: (int(RelationshipState.BC), int(RelationshipState.FP)),
    1: (
        int(RelationshipState.FC),
        int(RelationshipState.FN),
        int(RelationshipState.FW),
    ),
    2: (
        int(RelationshipState.FC),
        int(RelationshipState.FN),
        int(RelationshipState.FW),
    ),
    3: (
        int(RelationshipState.FC),
        int(RelationshipState.FN),
        int(RelationshipState.FW),
    ),
}


def compatible_target_states(truth_class: int) -> tuple[int, ...]:
    try:
        return _TARGET_STATES_BY_TRUTH[int(truth_class)]
    except KeyError as error:
        raise ValueError(f"unsupported truth class: {truth_class}") from error


def _stable_order(seed: int, *values: Any) -> int:
    encoded = "|".join((str(int(seed)), *(str(value) for value in values))).encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def select_node_slices(
    trace: CaseTrace,
    *,
    nodes: tuple[str, ...],
    num_classes: int = 4,
) -> dict[str, dict[str, Any]]:
    """Select one deterministic slice per node by exchangeable-state coverage."""

    if tuple(nodes) != (
        "down1",
        "down2",
        "down3",
        "down4",
        "up1",
        "up2",
        "up3",
        "up4",
    ):
        raise ValueError("slice selection requires the registered eight-node order")
    if trace.truth is None or not trace.slice_ids:
        raise ValueError("slice selection requires truth and slice identifiers")
    path = relationship_state_path(trace, num_classes=int(num_classes))
    truth = np.asarray(trace.truth)
    output: dict[str, dict[str, Any]] = {}
    for node_index, node in enumerate(nodes):
        reliable = node_reliability(trace, node_index)
        candidates = []
        for slice_index, slice_id in enumerate(trace.slice_ids):
            states = path[node_index, slice_index]
            mask = reliable[slice_index]
            strata = []
            for truth_class in range(int(num_classes)):
                selected_states = states[mask & (truth[slice_index] == truth_class)]
                present = set(np.unique(selected_states).astype(int).tolist())
                for target_state in compatible_target_states(truth_class):
                    if target_state in present and any(
                        state != target_state for state in present
                    ):
                        strata.append((truth_class, target_state))
            candidates.append(
                {
                    "slice_index": slice_index,
                    "slice_id": str(slice_id),
                    "exchangeable_strata": strata,
                    "exchangeable_stratum_count": len(strata),
                    "reliable_pixel_count": int(mask.sum()),
                    "reliable_foreground_count": int(
                        np.count_nonzero(mask & (truth[slice_index] > 0))
                    ),
                }
            )
        selected = min(
            candidates,
            key=lambda row: (
                -int(row["exchangeable_stratum_count"]),
                -int(row["reliable_foreground_count"]),
                -int(row["reliable_pixel_count"]),
                str(row["slice_id"]),
            ),
        )
        output[node] = selected
    return output


def build_balanced_base_requests(
    candidates: pd.DataFrame,
    *,
    maximum_targets: int,
    seed: int,
) -> pd.DataFrame:
    """Assign one requested natural target state to unique base pixels."""

    required = {
        "row_id",
        "patient_id",
        "node",
        "truth_class",
        "state",
        "output_index",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidate rows are missing columns: {sorted(missing)}")
    if int(maximum_targets) <= 0:
        raise ValueError("maximum_targets must be positive")
    if candidates.empty or not candidates["row_id"].is_unique:
        raise ValueError("candidate rows must be nonempty with unique row_id")
    output: list[dict[str, Any]] = []
    for (patient_id, node), group in candidates.groupby(
        ["patient_id", "node"],
        sort=True,
    ):
        queues: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for truth_class in sorted(group["truth_class"].unique().astype(int).tolist()):
            class_group = group[group["truth_class"] == truth_class]
            for target_state in compatible_target_states(truth_class):
                if not (class_group["state"] == target_state).any():
                    continue
                base = class_group[class_group["state"] != target_state].copy()
                if base.empty:
                    continue
                base["_order"] = base["row_id"].map(
                    lambda row_id: _stable_order(
                        seed,
                        patient_id,
                        node,
                        truth_class,
                        target_state,
                        row_id,
                    )
                )
                queues[(truth_class, target_state)] = base.sort_values(
                    ["_order", "row_id"],
                    kind="mergesort",
                ).to_dict("records")
        used_rows: set[str] = set()
        used_positions: set[int] = set()
        strata = sorted(queues)
        while len(used_rows) < int(maximum_targets):
            progress = False
            for stratum in strata:
                queue = queues[stratum]
                selected = None
                while queue:
                    candidate = queue.pop(0)
                    row_id = str(candidate["row_id"])
                    position = int(candidate["output_index"])
                    if row_id not in used_rows and position not in used_positions:
                        selected = candidate
                        break
                if selected is None:
                    continue
                truth_class, target_state = stratum
                selected.pop("_order", None)
                selected["requested_source_state"] = int(target_state)
                output.append(selected)
                used_rows.add(str(selected["row_id"]))
                used_positions.add(int(selected["output_index"]))
                progress = True
                if len(used_rows) >= int(maximum_targets):
                    break
            if not progress:
                break
    if not output:
        return candidates.iloc[0:0].assign(
            requested_source_state=pd.Series(dtype=np.int64)
        )
    return pd.DataFrame(output).sort_values(
        ["patient_id", "node", "row_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def prune_matches_to_independent_resize_rows(
    matches: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    output_shape: tuple[int, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep a deterministic maximal prefix that increases resize-row rank."""

    required_matches = {"status", "patient_id", "node", "base_row_id"}
    required_candidates = {"row_id", "native_h", "native_w", "output_index"}
    if required_matches - set(matches.columns):
        raise ValueError("matches lack the registered identity columns")
    if required_candidates - set(candidates.columns):
        raise ValueError("candidates lack native geometry")
    candidate_index = candidates.set_index("row_id", verify_integrity=True)
    retained: list[pd.Series] = []
    diagnostics: list[dict[str, Any]] = []
    for (patient_id, node), group in matches.groupby(
        ["patient_id", "node"],
        sort=True,
    ):
        matched = group[group["status"] == "MATCHED"].sort_values(
            "base_row_id",
            kind="mergesort",
        )
        if matched.empty:
            diagnostics.append(
                {
                    "patient_id": str(patient_id),
                    "node": str(node),
                    "matched_count": 0,
                    "retained_count": 0,
                    "spatial_rank": 0,
                }
            )
            continue
        base_rows = candidate_index.loc[matched["base_row_id"].tolist()]
        native_shapes = set(
            zip(
                base_rows["native_h"].astype(int),
                base_rows["native_w"].astype(int),
            )
        )
        if len(native_shapes) != 1:
            raise ValueError("one patient/node plan contains multiple native shapes")
        native_shape = next(iter(native_shapes))
        accepted_rows: list[torch.Tensor] = []
        rank = 0
        for _, match in matched.iterrows():
            base = candidate_index.loc[str(match["base_row_id"])]
            row = bilinear_resize_rows(
                native_shape,
                output_shape,
                torch.tensor([int(base["output_index"])], dtype=torch.int64),
                dtype=torch.float64,
            )
            candidate_matrix = torch.cat((*accepted_rows, row), dim=0)
            candidate_rank = int(torch.linalg.matrix_rank(candidate_matrix).item())
            if candidate_rank <= rank:
                continue
            retained.append(match)
            accepted_rows.append(row)
            rank = candidate_rank
        diagnostics.append(
            {
                "patient_id": str(patient_id),
                "node": str(node),
                "matched_count": int(len(matched)),
                "retained_count": len(accepted_rows),
                "spatial_rank": rank,
            }
        )
    retained_frame = (
        pd.DataFrame(retained).reset_index(drop=True)
        if retained
        else matches.iloc[0:0].copy()
    )
    return retained_frame, pd.DataFrame(diagnostics)


__all__ = [
    "build_balanced_base_requests",
    "compatible_target_states",
    "prune_matches_to_independent_resize_rows",
    "select_node_slices",
]
