from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from pptt.lineage.depths import lineage_depths
from pptt.transitions.fields import states_and_margins
from pptt.transitions.metrics import (
    confusion_from_labels,
    metric_delta_from_transition,
    metrics_from_confusion,
)
from pptt.transitions.theory import (
    additive_event_from_tensor,
    adjacent_pair_counts,
    binary_three_node_counterexample,
    endpoint_hidden_dimension,
    persistent_correction_count,
)
from pptt.transitions.tensors import (
    build_transition_tensor,
    confusions_from_transition,
    state_flows_from_transition,
)


def _event_counts(tensor: np.ndarray) -> dict[str, int]:
    """Count process events directly from a truth-conditioned transition tensor."""
    counts = np.asarray(tensor, dtype=np.int64)
    truth, state0, state1 = np.indices(counts.shape)
    return {
        "stable_correct": int(counts[(state0 == truth) & (state1 == truth)].sum()),
        "correction": int(counts[(state0 != truth) & (state1 == truth)].sum()),
        "destruction": int(counts[(state0 == truth) & (state1 != truth)].sum()),
        "stable_error": int(
            counts[(state0 != truth) & (state1 != truth) & (state0 == state1)].sum()
        ),
        "wrong_reencoding": int(
            counts[(state0 != truth) & (state1 != truth) & (state0 != state1)].sum()
        ),
    }


def _max_metric_difference(left: np.ndarray, right: np.ndarray) -> float:
    left_metrics = metrics_from_confusion(left)
    right_metrics = metrics_from_confusion(right)
    return max(
        float(np.max(np.abs(getattr(left_metrics, name) - getattr(right_metrics, name))))
        for name in left_metrics.as_dict()
    )


def _nonidentifiability_counterexample(tolerance: float) -> dict:
    """Construct endpoint-equivalent processes with different internal events.

    For one truth class, process A keeps three prediction states fixed, whereas
    process B cyclically permutes them. Their row and column marginals are
    identical, so both endpoint confusion matrices and all derived segmentation
    metrics are identical. The process events are nevertheless different.
    """
    num_classes = 3
    process_a = np.zeros((num_classes, num_classes, num_classes), dtype=np.int64)
    process_b = np.zeros_like(process_a)

    for truth_class in (0, 1):
        process_a[truth_class, truth_class, truth_class] = 1
        process_b[truth_class, truth_class, truth_class] = 1

    process_a[2, 0, 0] = 1
    process_a[2, 1, 1] = 1
    process_a[2, 2, 2] = 1

    process_b[2, 0, 1] = 1
    process_b[2, 1, 2] = 1
    process_b[2, 2, 0] = 1

    before_a, after_a = confusions_from_transition(process_a)
    before_b, after_b = confusions_from_transition(process_b)
    event_counts_a = _event_counts(process_a)
    event_counts_b = _event_counts(process_b)
    before_equal = bool(np.array_equal(before_a, before_b))
    after_equal = bool(np.array_equal(after_a, after_b))
    endpoint_metric_error = max(
        _max_metric_difference(before_a, before_b),
        _max_metric_difference(after_a, after_b),
    )
    transition_l1_distance = int(np.abs(process_a - process_b).sum())
    event_counts_differ = event_counts_a != event_counts_b
    proved = bool(
        before_equal
        and after_equal
        and endpoint_metric_error <= tolerance
        and transition_l1_distance > 0
        and event_counts_differ
    )
    return {
        "num_classes": num_classes,
        "same_before_confusion": before_equal,
        "same_after_confusion": after_equal,
        "max_endpoint_metric_difference": endpoint_metric_error,
        "transition_tensor_l1_distance": transition_l1_distance,
        "event_counts_process_a": event_counts_a,
        "event_counts_process_b": event_counts_b,
        "event_counts_differ": event_counts_differ,
        "proved": proved,
    }


def run(config: dict) -> dict:
    seed = int(config["seed"])
    c = int(config["num_classes"])
    stages = int(config["stages"])
    shape = (int(config["height"]), int(config["width"]))
    tolerance = float(config["tolerance"])
    if stages < 2:
        raise ValueError("stages must be at least 2")
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, c, size=shape, dtype=np.int64)
    states = rng.integers(0, c, size=(stages, *shape), dtype=np.int64)

    max_confusion_error = 0
    max_flow_error = 0
    max_metric_error = 0.0
    max_additive_functional_error = 0.0
    metric_path = []
    for state in states:
        confusion = confusion_from_labels(truth, state, c)
        metric_path.append(metrics_from_confusion(confusion))

    for index in range(stages - 1):
        tensor = build_transition_tensor(
            truth,
            states[index],
            states[index + 1],
            c,
        )
        if int(tensor.sum()) != truth.size:
            raise AssertionError("Pixel-count conservation failed")
        before, after = confusions_from_transition(tensor)
        direct_before = confusion_from_labels(truth, states[index], c)
        direct_after = confusion_from_labels(truth, states[index + 1], c)
        max_confusion_error = max(
            max_confusion_error,
            int(np.max(np.abs(before - direct_before))),
            int(np.max(np.abs(after - direct_after))),
        )
        flow = state_flows_from_transition(tensor)
        max_flow_error = max(
            max_flow_error,
            int(np.max(np.abs(flow.occupancy_delta - flow.inflow + flow.outflow))),
        )
        reconstructed_delta = metric_delta_from_transition(tensor)
        for name in reconstructed_delta.as_dict():
            direct_delta = getattr(metric_path[index + 1], name) - getattr(
                metric_path[index], name
            )
            max_metric_error = max(
                max_metric_error,
                float(np.max(np.abs(getattr(reconstructed_delta, name) - direct_delta))),
            )
        event_weights = (
            np.arange(c**3, dtype=np.float64).reshape(c, c, c)
            / float(max(1, c**3 - 1))
        )
        direct_event_value = float(
            event_weights[
                truth.reshape(-1),
                states[index].reshape(-1),
                states[index + 1].reshape(-1),
            ].sum()
        )
        tensor_event_value = additive_event_from_tensor(tensor, event_weights)
        max_additive_functional_error = max(
            max_additive_functional_error,
            abs(direct_event_value - tensor_event_value),
        )

    max_telescope_error = 0.0
    for name in metric_path[0].as_dict():
        deltas = [
            getattr(right, name) - getattr(left, name)
            for left, right in zip(metric_path[:-1], metric_path[1:])
        ]
        telescope = np.sum(deltas, axis=0)
        endpoint = getattr(metric_path[-1], name) - getattr(metric_path[0], name)
        max_telescope_error = max(
            max_telescope_error,
            float(np.max(np.abs(telescope - endpoint))),
        )

    depths = lineage_depths(states, truth)
    probabilities = rng.random((1, c, *shape))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    _, margins = states_and_margins(probabilities)
    nonidentifiability = _nonidentifiability_counterexample(tolerance)
    counterexample_a, counterexample_b = binary_three_node_counterexample()
    adjacent_a = adjacent_pair_counts(counterexample_a, num_classes=2)
    adjacent_b = adjacent_pair_counts(counterexample_b, num_classes=2)
    pairwise_equal = all(
        np.array_equal(left, right)
        for left, right in zip(adjacent_a, adjacent_b, strict=True)
    )
    persistence_a = persistent_correction_count(
        counterexample_a,
        truth_class=1,
    )
    persistence_b = persistent_correction_count(
        counterexample_b,
        truth_class=1,
    )
    pairwise_not_lineage_sufficient = {
        "pairwise_counts_equal": bool(pairwise_equal),
        "persistent_correction_count_process_a": persistence_a,
        "persistent_correction_count_process_b": persistence_b,
        "passed": bool(pairwise_equal and persistence_a != persistence_b),
    }
    hidden_dimension = endpoint_hidden_dimension(
        c,
        truth_class_count=c,
    )
    endpoint_dimension = {
        "num_classes": c,
        "truth_class_count": c,
        "dimension": hidden_dimension,
        "interior_point_assumption": True,
        "passed": bool(hidden_dimension == c * (c - 1) ** 2),
    }
    additive_functional = {
        "max_absolute_error": max_additive_functional_error,
        "tolerance": tolerance,
        "passed": bool(max_additive_functional_error <= tolerance),
    }
    report = {
        "seed": seed,
        "num_pixels": int(truth.size),
        "num_classes": c,
        "stages": stages,
        "max_confusion_error": max_confusion_error,
        "max_flow_error": max_flow_error,
        "max_metric_reconstruction_error": max_metric_error,
        "max_telescoping_error": max_telescope_error,
        "minimum_prediction_margin": float(margins.min()),
        "final_depth_range": [
            int(depths.final_state.min()),
            int(depths.final_state.max()),
        ],
        "endpoint_nonidentifiability": nonidentifiability,
        "additive_functional_exact": additive_functional,
        "endpoint_hidden_dimension": endpoint_dimension,
        "pairwise_not_lineage_sufficient": pairwise_not_lineage_sufficient,
        "passed": bool(
            max_confusion_error == 0
            and max_flow_error == 0
            and max_metric_error <= tolerance
            and max_telescope_error <= tolerance
            and nonidentifiability["proved"]
            and additive_functional["passed"]
            and endpoint_dimension["passed"]
            and pairwise_not_lineage_sufficient["passed"]
        ),
    }
    if not report["passed"]:
        raise AssertionError(f"V0 mathematical gate failed: {report}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the PPTT V0 mathematical gate.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    report = run(config)
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
