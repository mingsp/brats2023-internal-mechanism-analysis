from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from pptt.data.brats2d import SliceRecord, ensure_chw
from pptt.experiments.pilot import sample_feature_vectors
from pptt.models.protocol import ModelAdapter
from pptt.observers.evaluation import predict_observer_probabilities
from pptt.observers.full_cache import IndexedPatientCache
from pptt.observers.linear import LinearObserver


@dataclass(frozen=True)
class ParameterRandomizationAdmission:
    primary_status: str
    failed_nodes: dict[str, list[str]]
    retrained_diagnostic_status: str


def predict_frozen_observer_ensemble(
    observers: Mapping[int, LinearObserver],
    features: np.ndarray,
    *,
    device: str | torch.device,
) -> np.ndarray:
    """Apply the already-fitted observer ensemble without changing its mapping."""

    if not observers:
        raise ValueError("Frozen-observer randomization requires fitted observers")
    predictions = [
        predict_observer_probabilities(observer, features, device=device)
        for _, observer in sorted(observers.items())
    ]
    return np.mean(np.stack(predictions), axis=0).astype(np.float32, copy=False)


def parameter_randomization_admission(
    *,
    model_seed: int,
    registered_model_seeds: tuple[int, ...],
    frozen_report: Mapping[str, Any] | None,
    retrained_report: Mapping[str, Any] | None,
) -> ParameterRandomizationAdmission:
    """Resolve V1 admission while keeping the retrained control diagnostic-only."""

    diagnostic_status = (
        str(retrained_report.get("status", "INVALID"))
        if retrained_report is not None
        else "NOT_RUN"
    )
    if int(model_seed) not in {int(seed) for seed in registered_model_seeds}:
        return ParameterRandomizationAdmission(
            primary_status="NOT_APPLICABLE",
            failed_nodes={},
            retrained_diagnostic_status=diagnostic_status,
        )
    if frozen_report is None:
        return ParameterRandomizationAdmission(
            primary_status="PENDING",
            failed_nodes={},
            retrained_diagnostic_status=diagnostic_status,
        )
    if frozen_report.get("control_mode") != "frozen_observer":
        raise ValueError("Primary parameter-randomization report has the wrong mode")
    failed_nodes = {
        str(row["node"]): [
            "frozen-observer parameter randomization did not degrade readout"
        ]
        for row in frozen_report.get("nodes", [])
        if row.get("status") != "PASS"
    }
    report_status = str(frozen_report.get("status", "INVALID"))
    if report_status == "PASS" and failed_nodes:
        raise ValueError("Frozen randomization report contradicts its node statuses")
    if report_status != "PASS" and not failed_nodes:
        raise ValueError("Failed frozen randomization report has no failed node")
    return ParameterRandomizationAdmission(
        primary_status="PASS" if report_status == "PASS" else "FAIL",
        failed_nodes=failed_nodes,
        retrained_diagnostic_status=diagnostic_status,
    )


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
