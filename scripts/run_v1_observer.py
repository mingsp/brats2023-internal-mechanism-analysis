from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pptt.observers.cache import load_observer_cache
from pptt.observers.linear import LinearObserver
from pptt.observers.objective import mean_js_divergence


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained linear observer cache.")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache = load_observer_cache(args.cache)
    observer = LinearObserver(cache.in_channels, cache.num_classes)
    observer.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=True)
    observer.eval()
    features = torch.from_numpy(cache.features.astype(np.float32))[:, :, None, None]
    with torch.no_grad():
        logits = observer(features, output_size=(1, 1))[:, :, 0, 0]
        probabilities = torch.softmax(logits, dim=1).numpy()
    result = {
        "mean_js_divergence": mean_js_divergence(
            cache.target_probabilities,
            probabilities,
        ),
        "hard_agreement": float(
            np.mean(
                probabilities.argmax(axis=1)
                == cache.target_probabilities.argmax(axis=1)
            )
        ),
        "num_pixels": int(cache.features.shape[0]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
