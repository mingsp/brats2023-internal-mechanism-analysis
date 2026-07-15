from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from pptt.models.protocol import ModelAdapter
from pptt.observers.linear import LinearObserver


@dataclass(frozen=True)
class SlicePathTrace:
    states: np.ndarray
    reliable: np.ndarray
    truth: np.ndarray
    final_model_state: np.ndarray


def _observer_from_state(path: Path) -> LinearObserver:
    state = torch.load(path, map_location="cpu")
    weight = state.get("projection.weight")
    if not isinstance(weight, torch.Tensor) or weight.ndim != 4:
        raise ValueError(f"Invalid linear observer state: {path}")
    observer = LinearObserver(int(weight.shape[1]), int(weight.shape[0]))
    observer.load_state_dict(state, strict=True)
    return observer


def load_real_observer_path(
    directory: str | Path,
    *,
    nodes: tuple[str, ...],
    seeds: tuple[int, ...],
) -> dict[str, dict[int, LinearObserver]]:
    root = Path(directory)
    observers: dict[str, dict[int, LinearObserver]] = {}
    for node in nodes:
        observers[node] = {}
        for seed in seeds:
            state_path = root / "observers" / node / "real" / f"seed_{seed}.pt"
            if not state_path.is_file():
                raise FileNotFoundError(state_path)
            observers[node][seed] = _observer_from_state(state_path)
    return observers


def load_formal_reliability_threshold(directory: str | Path) -> float:
    root = Path(directory)
    status = json.loads((root / "v1_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "PASS":
        raise ValueError(f"V1 node admission is not PASS: {root}")
    failed = json.loads((root / "failed_nodes.json").read_text(encoding="utf-8"))
    if failed.get("failed_nodes"):
        raise ValueError(f"V1 contains failed nodes: {root}")
    reliability = json.loads(
        (root / "reliability_thresholds.json").read_text(encoding="utf-8")
    )
    threshold = reliability.get("threshold")
    if threshold is None or not np.isfinite(float(threshold)) or float(threshold) < 0:
        raise ValueError(f"V1 has no formal reliability threshold: {root}")
    return float(threshold)


class ObserverPathTracer:
    def __init__(
        self,
        adapter: ModelAdapter,
        observers: dict[str, dict[int, LinearObserver]],
        *,
        nodes: tuple[str, ...],
        seeds: tuple[int, ...],
        reliability_threshold: float,
        device: str | torch.device,
    ) -> None:
        if tuple(observers) != nodes or len(seeds) < 2:
            raise ValueError("Observer path must match nodes and contain multiple restarts")
        if not np.isfinite(reliability_threshold) or reliability_threshold < 0:
            raise ValueError("reliability_threshold must be finite and nonnegative")
        self.device = torch.device(device)
        self.adapter = adapter.to(self.device).eval()
        self.nodes = nodes
        self.seeds = seeds
        self.threshold = float(reliability_threshold)
        self.observers = {
            node: {
                seed: observers[node][seed].to(self.device).eval()
                for seed in seeds
            }
            for node in nodes
        }

    def trace_slice(
        self,
        image: torch.Tensor,
        truth: np.ndarray,
    ) -> SlicePathTrace:
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("image must have shape 1xCxHxW")
        truth_array = np.asarray(truth)
        output_size = tuple(int(value) for value in image.shape[-2:])
        if truth_array.shape != output_size or not np.issubdtype(
            truth_array.dtype,
            np.integer,
        ):
            raise ValueError("truth must be an integer array matching the image")
        restart_probabilities: list[np.ndarray] = []
        with torch.inference_mode():
            model_trace = self.adapter.trace(image.to(self.device))
            for node in self.nodes:
                node_runs = []
                for seed in self.seeds:
                    logits = self.observers[node][seed](
                        model_trace.activations[node],
                        output_size=output_size,
                    )
                    node_runs.append(torch.softmax(logits, dim=1)[0].cpu().numpy())
                restart_probabilities.append(np.stack(node_runs))
            final_model_state = model_trace.logits.argmax(dim=1)[0].cpu().numpy()
        runs = np.stack(restart_probabilities, axis=1)
        restart_states = runs.argmax(axis=2)
        ordered = np.sort(runs, axis=2)
        restart_margins = ordered[:, :, -1] - ordered[:, :, -2]
        canonical = runs.mean(axis=0)
        states = canonical.argmax(axis=1).astype(np.uint8)
        node_agreement = np.all(restart_states == restart_states[:1], axis=0)
        pair_agreement = node_agreement[:-1] & node_agreement[1:]
        pair_margin = np.minimum(
            restart_margins[:, :-1],
            restart_margins[:, 1:],
        ).min(axis=0)
        reliable = pair_agreement & (pair_margin >= self.threshold)
        return SlicePathTrace(
            states=states,
            reliable=reliable,
            truth=truth_array.astype(np.uint8, copy=False),
            final_model_state=final_model_state.astype(np.uint8, copy=False),
        )
