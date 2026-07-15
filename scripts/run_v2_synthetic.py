from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from pptt.synthetic.benchmarks import evaluate_known_process
from pptt.synthetic.known_process import build_known_process_batch


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PPTT known-process validation.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    batch = build_known_process_batch(
        batch_size=int(config["batch_size"]),
        height=int(config["height"]),
        width=int(config["width"]),
        seed=int(config["seed"]),
    )
    result = evaluate_known_process(batch)
    thresholds = config["thresholds"]
    passed = (
        result.stage_accuracy >= float(thresholds["stage_accuracy"])
        and result.direction_macro_f1 >= float(thresholds["direction_macro_f1"])
        and result.region_iou >= float(thresholds["region_iou"])
        and result.origin_depth_accuracy >= float(thresholds["origin_depth_accuracy"])
        and result.branch_attribution_accuracy
        >= float(thresholds["branch_attribution_accuracy"])
        and result.max_metric_reconstruction_error
        < float(thresholds["max_metric_reconstruction_error"])
    )
    report = {
        "seed": int(config["seed"]),
        "passed": bool(passed),
        **result.to_report(),
    }
    if not passed:
        raise AssertionError(f"V2 construct gate failed: {report}")
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
