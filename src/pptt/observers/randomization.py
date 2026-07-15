from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import nn

from pptt.data.brats2d import SliceRecord, ensure_chw
from pptt.experiments.pilot import sample_feature_vectors
from pptt.models.protocol import ModelAdapter
from pptt.observers.full_cache import IndexedPatientCache


def module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def randomize_checkpoint_module(
    adapter: ModelAdapter,
    node: str,
    *,
    seed: int,
) -> tuple[str, str]:
    module = adapter.randomization_module(node)
    before = module_state_sha256(module)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        reset_count = 0
        for child in module.modules():
            reset = getattr(child, "reset_parameters", None)
            if callable(reset):
                reset()
                reset_count += 1
    if reset_count == 0:
        raise ValueError(f"Checkpoint module {node!r} has no resettable parameters")
    after = module_state_sha256(module)
    if before == after:
        raise RuntimeError(f"Checkpoint module {node!r} did not change after randomization")
    return before, after


def extract_randomized_node_features(
    adapter: ModelAdapter,
    cache: IndexedPatientCache,
    records: dict[str, SliceRecord],
    *,
    node: str,
    device: torch.device,
) -> np.ndarray:
    missing = sorted(set(cache.slice_ids.tolist()) - set(records))
    if missing:
        raise KeyError(f"Missing source slices for cached rows: {missing[:3]}")
    randomized: np.ndarray | None = None
    for slice_id in dict.fromkeys(cache.slice_ids.tolist()):
        selected = np.flatnonzero(cache.slice_ids == slice_id)
        record = records[slice_id]
        image_array = ensure_chw(np.load(record.image_path, allow_pickle=False))
        image = torch.from_numpy(image_array)[None].to(device)
        with torch.inference_mode():
            trace = adapter.trace(image)
        height, width = image_array.shape[-2:]
        flat_indices = torch.from_numpy(
            cache.y[selected].astype(np.int64) * width
            + cache.x[selected].astype(np.int64)
        )
        sampled = (
            sample_feature_vectors(
                trace.activations[node],
                flat_indices,
                output_size=(height, width),
            )
            .cpu()
            .numpy()
            .astype(np.float16)
        )
        if randomized is None:
            randomized = np.empty(
                (cache.row_count, sampled.shape[1]),
                dtype=np.float16,
            )
        randomized[selected] = sampled
        del trace, image
    if randomized is None:
        raise ValueError("No cached rows were available for randomization")
    return randomized
