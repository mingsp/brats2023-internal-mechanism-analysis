"""Exact pixel-state transition accounting."""

from pptt.transitions.theory import (
    additive_event_from_tensor,
    adjacent_pair_counts,
    binary_three_node_counterexample,
    endpoint_hidden_dimension,
    persistent_correction_count,
)

__all__ = [
    "additive_event_from_tensor",
    "adjacent_pair_counts",
    "binary_three_node_counterexample",
    "endpoint_hidden_dimension",
    "persistent_correction_count",
]
