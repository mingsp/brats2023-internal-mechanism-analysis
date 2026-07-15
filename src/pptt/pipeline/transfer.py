from __future__ import annotations

import numpy as np


def contiguous_node_segments(
    declared_nodes: tuple[str, ...],
    admitted_nodes: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    if len(set(declared_nodes)) != len(declared_nodes):
        raise ValueError("declared_nodes must be unique")
    admitted = set(admitted_nodes)
    if len(admitted) != len(admitted_nodes) or not admitted.issubset(declared_nodes):
        raise ValueError("admitted_nodes must be a unique subset of declared_nodes")
    segments: list[tuple[str, ...]] = []
    active: list[str] = []
    for node in declared_nodes:
        if node in admitted:
            active.append(node)
        elif active:
            segments.append(tuple(active))
            active = []
    if active:
        segments.append(tuple(active))
    return tuple(segments)


def mask_discontinuous_reliability(
    reliable: np.ndarray,
    *,
    declared_nodes: tuple[str, ...],
    traced_nodes: tuple[str, ...],
) -> np.ndarray:
    values = np.asarray(reliable)
    if values.dtype != np.bool_ or values.shape[0] != max(len(traced_nodes) - 1, 0):
        raise ValueError("reliable does not match the traced node path")
    positions = {node: index for index, node in enumerate(declared_nodes)}
    if any(node not in positions for node in traced_nodes):
        raise ValueError("traced_nodes must belong to declared_nodes")
    result = values.copy()
    for transition_index, (left, right) in enumerate(
        zip(traced_nodes[:-1], traced_nodes[1:], strict=True)
    ):
        if positions[right] != positions[left] + 1:
            result[transition_index] = False
    return result
