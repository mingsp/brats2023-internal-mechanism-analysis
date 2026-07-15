from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KnownProcessBatch:
    truth: np.ndarray
    states: np.ndarray
    features: np.ndarray
    mixing_matrices: np.ndarray
    readout_weights: np.ndarray
    readout_biases: np.ndarray
    region_masks: dict[str, np.ndarray]
    expected_changed_masks: np.ndarray
    active_transition_indices: tuple[int, ...]
    unique_branch_transition_index: int
    branch_names: tuple[str, ...]
    branch_ablation_states: np.ndarray
    unique_effective_branch_index: int
    num_classes: int = 4


def _ellipse(
    height: int,
    width: int,
    *,
    center_y: float,
    center_x: float,
    radius_y: float,
    radius_x: float,
) -> np.ndarray:
    y, x = np.ogrid[:height, :width]
    return (
        ((y - center_y) / radius_y) ** 2
        + ((x - center_x) / radius_x) ** 2
        <= 1.0
    )


def build_known_process_batch(
    *,
    batch_size: int,
    height: int,
    width: int,
    seed: int,
) -> KnownProcessBatch:
    if batch_size <= 0 or height < 32 or width < 32:
        raise ValueError("batch_size must be positive and spatial dimensions at least 32")
    rng = np.random.default_rng(seed)
    region_a = np.zeros((batch_size, height, width), dtype=bool)
    region_b = np.zeros_like(region_a)
    region_c = np.zeros_like(region_a)
    region_d = np.zeros_like(region_a)
    for sample in range(batch_size):
        jitter_y, jitter_x = rng.integers(-2, 3, size=2)
        region_a[sample] = _ellipse(
            height,
            width,
            center_y=0.30 * height + jitter_y,
            center_x=0.27 * width + jitter_x,
            radius_y=0.11 * height,
            radius_x=0.09 * width,
        )
        jitter_y, jitter_x = rng.integers(-2, 3, size=2)
        region_b[sample] = _ellipse(
            height,
            width,
            center_y=0.31 * height + jitter_y,
            center_x=0.72 * width + jitter_x,
            radius_y=0.10 * height,
            radius_x=0.10 * width,
        )
        jitter_y, jitter_x = rng.integers(-2, 3, size=2)
        region_c[sample] = _ellipse(
            height,
            width,
            center_y=0.70 * height + jitter_y,
            center_x=0.50 * width + jitter_x,
            radius_y=0.09 * height,
            radius_x=0.12 * width,
        )
        jitter_y, jitter_x = rng.integers(-1, 2, size=2)
        region_d[sample] = _ellipse(
            height,
            width,
            center_y=0.78 * height + jitter_y,
            center_x=0.82 * width + jitter_x,
            radius_y=0.06 * height,
            radius_x=0.07 * width,
        )
    regions = (region_a, region_b, region_c, region_d)
    if any(
        np.any(left & right)
        for index, left in enumerate(regions)
        for right in regions[index + 1 :]
    ):
        raise AssertionError("Synthetic regions must remain disjoint")

    truth = np.zeros((batch_size, height, width), dtype=np.int64)
    truth[region_a] = 1
    truth[region_b] = 2
    truth[region_c] = 3
    truth[region_d] = 1
    stages = 8
    states = np.repeat(truth[:, None], stages, axis=1)
    states[:, :2][np.broadcast_to(region_a[:, None], states[:, :2].shape)] = 0
    states[:, :4][np.broadcast_to(region_b[:, None], states[:, :4].shape)] = 1
    states[:, 4:7][np.broadcast_to(region_b[:, None], states[:, 4:7].shape)] = 3
    states[:, :6][np.broadcast_to(region_c[:, None], states[:, :6].shape)] = 0
    states[:, 3:][np.broadcast_to(region_d[:, None], states[:, 3:].shape)] = 2

    expected_changed = states[:, 1:] != states[:, :-1]
    active = tuple(
        int(index)
        for index in np.flatnonzero(np.any(expected_changed, axis=(0, 2, 3)))
    )
    if active != (1, 2, 3, 5, 6):
        raise AssertionError(f"Unexpected synthetic transition schedule: {active}")

    feature_channels = 7
    mixing = np.empty((stages, feature_channels, 4), dtype=np.float64)
    readout_weights = np.empty((stages, 4, feature_channels), dtype=np.float64)
    feature_biases = rng.normal(0.0, 0.15, size=(stages, feature_channels))
    readout_biases = np.empty((stages, 4), dtype=np.float64)
    features = np.empty(
        (batch_size, stages, feature_channels, height, width),
        dtype=np.float32,
    )
    identity = np.eye(4, dtype=np.float64)
    for stage in range(stages):
        orthogonal, _ = np.linalg.qr(rng.normal(size=(feature_channels, 4)))
        mixing[stage] = orthogonal[:, :4]
        readout_weights[stage] = np.linalg.pinv(mixing[stage])
        readout_biases[stage] = -readout_weights[stage] @ feature_biases[stage]
        one_hot = identity[states[:, stage]]
        embedded = np.einsum("dc,nhwc->ndhw", mixing[stage], one_hot)
        embedded += feature_biases[stage][None, :, None, None]
        embedded += rng.normal(0.0, 0.003, size=embedded.shape)
        features[:, stage] = embedded.astype(np.float32)

    branch_names = (
        "context_branch",
        "boundary_branch",
        "corrective_branch",
        "residual_branch",
    )
    branch_ablation_states = np.repeat(states[:, 6][None], len(branch_names), axis=0)
    branch_ablation_states[2][region_c] = states[:, 5][region_c]

    return KnownProcessBatch(
        truth=truth,
        states=states,
        features=features,
        mixing_matrices=mixing.astype(np.float32),
        readout_weights=readout_weights.astype(np.float32),
        readout_biases=readout_biases.astype(np.float32),
        region_masks={"A": region_a, "B": region_b, "C": region_c, "D": region_d},
        expected_changed_masks=expected_changed,
        active_transition_indices=active,
        unique_branch_transition_index=5,
        branch_names=branch_names,
        branch_ablation_states=branch_ablation_states,
        unique_effective_branch_index=2,
    )


def decode_known_process_features(
    batch: KnownProcessBatch,
    *,
    logit_scale: float = 10.0,
) -> np.ndarray:
    logits = np.einsum(
        "kcd,nkdhw->nkchw",
        batch.readout_weights,
        batch.features,
    )
    logits += batch.readout_biases[None, :, :, None, None]
    logits *= float(logit_scale)
    logits -= logits.max(axis=2, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    return probabilities.astype(np.float32)
