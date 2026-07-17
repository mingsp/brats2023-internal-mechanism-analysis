from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


def _validate_transition_path(values: np.ndarray) -> np.ndarray:
    tensor = np.asarray(values)
    if tensor.ndim != 4:
        raise ValueError("transition path must have shape IxCxCxC")
    if tensor.shape[1] != tensor.shape[2] or tensor.shape[2] != tensor.shape[3]:
        raise ValueError("transition path class axes must have equal size")
    if np.any(tensor < 0) or not np.issubdtype(tensor.dtype, np.number):
        raise ValueError("transition counts must be nonnegative numbers")
    return tensor.astype(np.float64, copy=False)


def _validated_truth_classes(
    truth_classes: Sequence[int],
    *,
    class_count: int,
) -> tuple[int, ...]:
    classes = tuple(int(value) for value in truth_classes)
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("truth_classes must be a nonempty unique sequence")
    if any(value < 0 or value >= class_count for value in classes):
        raise ValueError("truth class is outside the transition tensor")
    return classes


def class_balanced_path_tv(
    left: np.ndarray,
    right: np.ndarray,
    *,
    truth_classes: Sequence[int],
) -> float:
    first = _validate_transition_path(left)
    second = _validate_transition_path(right)
    if first.shape != second.shape:
        raise ValueError("transition paths must have equal shape")
    classes = _validated_truth_classes(
        truth_classes,
        class_count=first.shape[1],
    )
    distances = []
    for transition_index in range(first.shape[0]):
        for truth_class in classes:
            first_slice = first[transition_index, truth_class]
            second_slice = second[transition_index, truth_class]
            first_total = float(first_slice.sum())
            second_total = float(second_slice.sum())
            if first_total == 0 and second_total == 0:
                continue
            if first_total == 0 or second_total == 0:
                raise ValueError("paired truth-class totals must both be present")
            distances.append(
                0.5
                * float(
                    np.abs(
                        first_slice / first_total - second_slice / second_total
                    ).sum()
                )
            )
    if not distances:
        raise ValueError("no requested truth class is present")
    return float(np.mean(distances))


def class_balanced_damage_rate(
    path: np.ndarray,
    *,
    truth_classes: Sequence[int],
) -> float:
    tensor = _validate_transition_path(path)
    classes = _validated_truth_classes(
        truth_classes,
        class_count=tensor.shape[1],
    )
    rates = []
    for transition_index in range(tensor.shape[0]):
        for truth_class in classes:
            current = tensor[transition_index, truth_class]
            total = float(current.sum())
            if total == 0:
                continue
            destruction = float(current[truth_class].sum() - current[truth_class, truth_class])
            wrong_reencoding = 0.0
            for state_from in range(current.shape[0]):
                if state_from == truth_class:
                    continue
                for state_to in range(current.shape[1]):
                    if state_to != truth_class and state_to != state_from:
                        wrong_reencoding += float(current[state_from, state_to])
            rates.append((destruction + wrong_reencoding) / total)
    if not rates:
        raise ValueError("no requested truth class is present")
    return float(np.mean(rates))


def class_balanced_persistent_net_by_transition(
    states: np.ndarray,
    truth: np.ndarray,
    *,
    truth_classes: Sequence[int],
) -> np.ndarray:
    state_path = np.asarray(states)
    labels = np.asarray(truth)
    if state_path.ndim < 2 or state_path.shape[1:] != labels.shape:
        raise ValueError("states must have shape K followed by the truth shape")
    if not np.issubdtype(state_path.dtype, np.integer):
        raise ValueError("states must have integer dtype")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("truth must have integer dtype")
    requested_maximum = max(int(value) for value in truth_classes)
    class_count = int(
        max(
            state_path.max(initial=0),
            labels.max(initial=0),
            requested_maximum,
        )
    ) + 1
    classes = _validated_truth_classes(truth_classes, class_count=class_count)
    suffix_correct = np.logical_and.accumulate(
        (state_path == labels[np.newaxis, ...])[::-1],
        axis=0,
    )[::-1]
    suffix_wrong = np.logical_and.accumulate(
        (state_path != labels[np.newaxis, ...])[::-1],
        axis=0,
    )[::-1]
    values = np.full(state_path.shape[0] - 1, np.nan, dtype=np.float64)
    present_classes = [value for value in classes if np.any(labels == value)]
    if not present_classes:
        raise ValueError("no requested truth class is present")
    for transition_index in range(state_path.shape[0] - 1):
        class_rates = []
        persistent_correct_after = suffix_correct[transition_index + 1]
        persistent_wrong_after = suffix_wrong[transition_index + 1]
        for truth_class in present_classes:
            selected = labels == truth_class
            denominator = int(selected.sum())
            correction = (
                selected
                & (state_path[transition_index] != labels)
                & (state_path[transition_index + 1] == labels)
                & persistent_correct_after
            )
            destruction = (
                selected
                & (state_path[transition_index] == labels)
                & (state_path[transition_index + 1] != labels)
                & persistent_wrong_after
            )
            class_rates.append(
                (int(correction.sum()) - int(destruction.sum())) / denominator
            )
        values[transition_index] = float(np.mean(class_rates))
    return values


def class_balanced_persistent_net_recovery(
    states: np.ndarray,
    truth: np.ndarray,
    *,
    transition_indices: Sequence[int],
    truth_classes: Sequence[int],
) -> float:
    state_path = np.asarray(states)
    indices = tuple(int(value) for value in transition_indices)
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("transition_indices must be nonempty and unique")
    if state_path.ndim < 2 or any(
        value < 0 or value >= state_path.shape[0] - 1 for value in indices
    ):
        raise ValueError("transition index is outside the state path")
    by_transition = class_balanced_persistent_net_by_transition(
        state_path,
        truth,
        truth_classes=truth_classes,
    )
    return float(by_transition[list(indices)].sum())


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("p_values must be a finite vector")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must lie inside [0, 1]")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    scaled = (values.size - np.arange(values.size)) * sorted_values
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(scaled))
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def select_robust_transition_candidate(
    rows: Iterable[Mapping[str, Any]],
    *,
    allowed_transition_indices: Sequence[int],
    required_model_seeds: Sequence[int],
) -> dict[str, Any]:
    records = [dict(row) for row in rows]
    allowed = tuple(int(value) for value in allowed_transition_indices)
    seeds = tuple(int(value) for value in required_model_seeds)
    if not allowed or len(allowed) != len(set(allowed)):
        raise ValueError("allowed transition indices must be nonempty and unique")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("required model seeds must be nonempty and unique")
    candidates = []
    for transition_index in allowed:
        selected = [
            row
            for row in records
            if int(row["transition_index"]) == transition_index
            and int(row["model_seed"]) in seeds
        ]
        by_seed = {int(row["model_seed"]): row for row in selected}
        if set(by_seed) != set(seeds):
            continue
        effects = np.asarray(
            [float(by_seed[seed]["mean_difference"]) for seed in seeds],
            dtype=np.float64,
        )
        if not np.isfinite(effects).all() or np.any(effects <= 0):
            continue
        transitions = {str(row["transition"]) for row in selected}
        if len(transitions) != 1:
            raise ValueError("one transition index maps to multiple names")
        candidates.append(
            {
                "transition_index": transition_index,
                "transition": transitions.pop(),
                "minimum_seed_effect": float(effects.min()),
                "mean_seed_effect": float(effects.mean()),
                "seed_effects": {
                    str(seed): float(by_seed[seed]["mean_difference"])
                    for seed in seeds
                },
            }
        )
    if not candidates:
        raise ValueError("no transition is positive in every required model seed")
    selected = sorted(
        candidates,
        key=lambda row: (
            -float(row["minimum_seed_effect"]),
            -float(row["mean_seed_effect"]),
            int(row["transition_index"]),
        ),
    )[0]
    return {
        **selected,
        "selection_rule": "positive_all_seeds_then_maximize_minimum",
        "required_model_seeds": list(seeds),
    }


def evaluate_causal_entry_gate(
    rows: Iterable[Mapping[str, Any]],
    *,
    minimum_supported_seeds: int = 2,
    alpha: float = 0.05,
    minimum_standardized_effect: float = 0.8,
) -> dict[str, Any]:
    records = [dict(row) for row in rows]
    process_metrics = (
        "middle_damage",
        "late_persistent_net_recovery",
    )
    metric_audit: dict[str, dict[str, Any]] = {}
    for metric in process_metrics:
        selected = [row for row in records if row.get("metric") == metric]
        if not selected:
            metric_audit[metric] = {
                "all_seed_directions_positive": False,
                "supported_seed_count": 0,
                "passed": False,
            }
            continue
        supported = [
            row
            for row in selected
            if float(row["mean_difference"]) > 0
            and float(row["ci_low"]) > 0
            and float(row["holm_p"]) < alpha
            and abs(float(row["paired_cohens_d"]))
            >= minimum_standardized_effect
        ]
        directions_positive = all(
            float(row["mean_difference"]) > 0 for row in selected
        )
        metric_audit[metric] = {
            "all_seed_directions_positive": directions_positive,
            "supported_seed_count": len(supported),
            "passed": directions_positive
            and len(supported) >= minimum_supported_seeds,
        }
    terminal_rows = [
        row for row in records if row.get("metric") == "terminal_dice"
    ]
    process_effects = [
        abs(float(row["paired_cohens_d"]))
        for row in records
        if row.get("metric") in process_metrics
    ]
    terminal_effects = [
        abs(float(row["paired_cohens_d"])) for row in terminal_rows
    ]
    process_exceeds_terminal = bool(
        process_effects
        and (
            not terminal_effects
            or np.median(process_effects) > np.median(terminal_effects)
        )
    )
    return {
        "passed": all(value["passed"] for value in metric_audit.values())
        and process_exceeds_terminal,
        "terminal_dice_required": False,
        "minimum_supported_seeds": int(minimum_supported_seeds),
        "alpha": float(alpha),
        "minimum_standardized_effect": float(minimum_standardized_effect),
        "process_effect_exceeds_terminal_effect": process_exceeds_terminal,
        "metrics": metric_audit,
    }


def candidate_specific_causal_gate(
    rows: Iterable[Mapping[str, Any]],
    *,
    transition: str,
    required_model_seeds: Sequence[int],
    alpha: float,
    minimum_standardized_effect: float,
) -> dict[str, Any]:
    records = [
        dict(row) for row in rows if str(row.get("transition")) == str(transition)
    ]
    seeds = tuple(int(value) for value in required_model_seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("required_model_seeds must be nonempty and unique")
    by_seed = {int(row["model_seed"]): row for row in records}
    if set(by_seed) != set(seeds) or len(records) != len(seeds):
        raise ValueError("candidate rows must contain each registered seed exactly once")
    seed_results = []
    for seed in seeds:
        row = by_seed[seed]
        supported = bool(
            float(row["mean_difference"]) > 0
            and float(row["ci_low"]) > 0
            and float(row["holm_p"]) < float(alpha)
            and abs(float(row["paired_cohens_d"]))
            >= float(minimum_standardized_effect)
        )
        seed_results.append(
            {
                "model_seed": seed,
                "mean_difference": float(row["mean_difference"]),
                "ci_low": float(row["ci_low"]),
                "holm_p": float(row["holm_p"]),
                "paired_cohens_d": float(row["paired_cohens_d"]),
                "supported": supported,
            }
        )
    supported_count = sum(row["supported"] for row in seed_results)
    return {
        "transition": str(transition),
        "required_model_seeds": list(seeds),
        "alpha": float(alpha),
        "minimum_standardized_effect": float(minimum_standardized_effect),
        "supported_seed_count": int(supported_count),
        "passed": supported_count == len(seeds),
        "seed_results": seed_results,
    }
