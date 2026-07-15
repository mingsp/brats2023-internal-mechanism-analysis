from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from pptt.observers.cache import load_observer_cache
from pptt.observers.trainer import ObserverTrainingConfig, train_observer


def main() -> int:
    parser = argparse.ArgumentParser(description="Train one output-anchored linear observer.")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    config = ObserverTrainingConfig(**payload["training"])
    cache = load_observer_cache(args.cache)
    observer, report = train_observer(
        cache,
        config=config,
        seed=args.seed,
        device=args.device,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(observer.state_dict(), args.output / "observer_state.pt")
    (args.output / "optimization_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
