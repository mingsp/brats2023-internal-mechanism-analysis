"""Core F/G assembly for internal-inference process explanations.

The experiment scripts in this package compute semantic readouts, CAM and raw
feature responses, region allocation, response-guided perturbations, and
structural interventions. This module gives those measurements one common,
architecture-aware representation:

    E_ij = F(M_i, M_j; tau_ij)
    O = G([E_12, E_23, ...], validity)

F preserves the physical meaning of every metric. G orders valid transition
measurements along the network path; it does not learn weights, fit Dice, fill
missing evidence with zero, or collapse heterogeneous metrics into one score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


MetricMap = Mapping[str, float]


def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _copy_finite(values: MetricMap) -> Dict[str, float]:
    return {str(name): float(value) for name, value in values.items() if _finite(value)}


@dataclass(frozen=True)
class NodeState:
    """Comparable measurements for one internal feature-tensor node.

    ``semantic_score`` is the node-level task readout. The remaining mappings
    retain their source and target in the metric name, for example ``cam`` or
    ``raw`` response alignment and ``cam.tumor`` region allocation.
    """

    node_id: str
    order: int
    semantic_score: Optional[float] = None
    response_alignment: MetricMap = field(default_factory=dict)
    region_allocation: MetricMap = field(default_factory=dict)
    output_influence: MetricMap = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must not be empty")


@dataclass(frozen=True)
class TransitionEvidence:
    """One local explanation vector with named metric blocks."""

    src: str
    dst: str
    src_order: int
    dst_order: int
    values: Mapping[str, float]

    @property
    def transition_id(self) -> str:
        return f"{self.src}->{self.dst}"


@dataclass(frozen=True)
class GlobalPathExplanation:
    """Structured G output for a complete ordered path.

    ``values`` and ``valid`` have shape ``[metrics, transitions]``. Missing or
    inapplicable evidence is represented by ``NaN`` plus ``False`` rather than
    a fabricated zero. ``key_transition_by_metric`` records the transition
    with the largest absolute valid change for each metric.
    """

    metric_names: Tuple[str, ...]
    transition_ids: Tuple[str, ...]
    values: np.ndarray
    valid: np.ndarray
    key_transition_by_metric: Mapping[str, Optional[str]]

    def __post_init__(self) -> None:
        expected = (len(self.metric_names), len(self.transition_ids))
        if self.values.shape != expected:
            raise ValueError(f"values shape {self.values.shape} does not match {expected}")
        if self.valid.shape != expected:
            raise ValueError(f"valid shape {self.valid.shape} does not match {expected}")

    def metric_trajectory(self, metric_name: str) -> Tuple[Optional[float], ...]:
        """Return one path trajectory while preserving missing entries."""

        row = self.metric_names.index(metric_name)
        return tuple(
            float(self.values[row, col]) if self.valid[row, col] else None
            for col in range(len(self.transition_ids))
        )

    def coverage(self, metric_name: str) -> float:
        """Fraction of path transitions with valid evidence for one metric."""

        row = self.metric_names.index(metric_name)
        return float(np.mean(self.valid[row])) if self.valid.shape[1] else 0.0

    def as_long_records(self) -> List[Dict[str, object]]:
        """Export the matrix without losing validity information."""

        rows: List[Dict[str, object]] = []
        for metric_idx, metric_name in enumerate(self.metric_names):
            for transition_idx, transition_id in enumerate(self.transition_ids):
                is_valid = bool(self.valid[metric_idx, transition_idx])
                rows.append(
                    {
                        "metric": metric_name,
                        "transition": transition_id,
                        "value": float(self.values[metric_idx, transition_idx]) if is_valid else None,
                        "valid": is_valid,
                    }
                )
        return rows


def _add_delta_block(
    output: Dict[str, float],
    prefix: str,
    source: MetricMap,
    target: MetricMap,
) -> None:
    for name in sorted(set(source).intersection(target)):
        if _finite(source[name]) and _finite(target[name]):
            output[f"{prefix}.{name}.delta"] = float(target[name]) - float(source[name])


def transition_function(
    source: NodeState,
    target: NodeState,
    structural_source: Optional[MetricMap] = None,
) -> TransitionEvidence:
    """Compute the local F output for one ordered node transition.

    Functional evidence is represented both as a target-node effect and, when
    both endpoints are available, as a transition delta. Structural evidence
    comes from explicit branch perturbation, fusion, or counterfactual tests;
    this function never infers it from the edge label alone.
    """

    if target.order <= source.order:
        raise ValueError("target.order must be greater than source.order")

    values: Dict[str, float] = {}
    if _finite(source.semantic_score) and _finite(target.semantic_score):
        values["semantic.delta"] = float(target.semantic_score) - float(source.semantic_score)

    _add_delta_block(
        values,
        "response",
        _copy_finite(source.response_alignment),
        _copy_finite(target.response_alignment),
    )
    _add_delta_block(
        values,
        "region",
        _copy_finite(source.region_allocation),
        _copy_finite(target.region_allocation),
    )
    _add_delta_block(
        values,
        "function",
        _copy_finite(source.output_influence),
        _copy_finite(target.output_influence),
    )

    for name, value in _copy_finite(target.output_influence).items():
        values[f"function.{name}.target"] = value
    for name, value in _copy_finite(structural_source or {}).items():
        values[f"structure.{name}"] = value

    return TransitionEvidence(
        src=source.node_id,
        dst=target.node_id,
        src_order=int(source.order),
        dst_order=int(target.order),
        values=values,
    )


def build_path_transitions(
    node_states: Sequence[NodeState],
    structural_by_transition: Optional[Mapping[str, MetricMap]] = None,
) -> List[TransitionEvidence]:
    """Apply F to every adjacent node pair in computation order."""

    ordered = sorted(node_states, key=lambda state: state.order)
    if len({state.node_id for state in ordered}) != len(ordered):
        raise ValueError("node_id values must be unique")
    if len({state.order for state in ordered}) != len(ordered):
        raise ValueError("node order values must be unique")
    if len(ordered) < 2:
        raise ValueError("at least two node states are required")

    structural_by_transition = structural_by_transition or {}
    outputs: List[TransitionEvidence] = []
    for source, target in zip(ordered[:-1], ordered[1:]):
        transition_id = f"{source.node_id}->{target.node_id}"
        outputs.append(
            transition_function(
                source,
                target,
                structural_source=structural_by_transition.get(transition_id),
            )
        )
    return outputs


def _metric_sort_key(metric_name: str) -> Tuple[int, str]:
    group = metric_name.split(".", 1)[0]
    group_order = {
        "semantic": 0,
        "response": 1,
        "region": 2,
        "function": 3,
        "structure": 4,
    }
    return group_order.get(group, 99), metric_name


def global_integration_function(
    transitions: Sequence[TransitionEvidence],
    metric_order: Optional[Iterable[str]] = None,
) -> GlobalPathExplanation:
    """Compute G by assembling valid F outputs in network order.

    No learned or manually selected weights are used. The returned matrix is
    the primary output; key transitions are indexes into original measurements,
    not a replacement score.
    """

    if not transitions:
        raise ValueError("transitions must not be empty")

    ordered = sorted(transitions, key=lambda item: (item.src_order, item.dst_order))
    if len({item.transition_id for item in ordered}) != len(ordered):
        raise ValueError("transition identifiers must be unique")

    if metric_order is None:
        names = sorted(
            {name for item in ordered for name in item.values},
            key=_metric_sort_key,
        )
    else:
        names = list(dict.fromkeys(str(name) for name in metric_order))

    values = np.full((len(names), len(ordered)), np.nan, dtype=np.float64)
    valid = np.zeros((len(names), len(ordered)), dtype=bool)
    name_to_row = {name: idx for idx, name in enumerate(names)}

    for col, transition in enumerate(ordered):
        for name, value in transition.values.items():
            row = name_to_row.get(name)
            if row is not None and _finite(value):
                values[row, col] = float(value)
                valid[row, col] = True

    transition_ids = tuple(item.transition_id for item in ordered)
    key_transition_by_metric: Dict[str, Optional[str]] = {}
    for row, name in enumerate(names):
        valid_cols = np.flatnonzero(valid[row])
        if valid_cols.size == 0:
            key_transition_by_metric[name] = None
            continue
        local_idx = int(np.argmax(np.abs(values[row, valid_cols])))
        key_transition_by_metric[name] = transition_ids[int(valid_cols[local_idx])]

    return GlobalPathExplanation(
        metric_names=tuple(names),
        transition_ids=transition_ids,
        values=values,
        valid=valid,
        key_transition_by_metric=key_transition_by_metric,
    )


def analyze_internal_path(
    node_states: Sequence[NodeState],
    structural_by_transition: Optional[Mapping[str, MetricMap]] = None,
    metric_order: Optional[Iterable[str]] = None,
) -> GlobalPathExplanation:
    """Run the complete F-then-G assembly for one internal path."""

    transitions = build_path_transitions(
        node_states,
        structural_by_transition=structural_by_transition,
    )
    return global_integration_function(transitions, metric_order=metric_order)


__all__ = [
    "GlobalPathExplanation",
    "NodeState",
    "TransitionEvidence",
    "analyze_internal_path",
    "build_path_transitions",
    "global_integration_function",
    "transition_function",
]
