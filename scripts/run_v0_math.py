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
from pptt.transitions.tensors import (
    build_transition_tensor,
    confusions_from_transition,
    state_flows_from_transition,
)


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
        "passed": bool(
            max_confusion_error == 0
            and max_flow_error == 0
            and max_metric_error <= tolerance
            and max_telescope_error <= tolerance
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
