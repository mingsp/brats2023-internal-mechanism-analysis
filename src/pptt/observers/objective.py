from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.nn import functional as F


def class_balanced_pixel_weights(
    target_probabilities: npt.ArrayLike,
    *,
    smoothing: float = 1.0,
) -> np.ndarray:
    probabilities = np.asarray(target_probabilities)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(f"Expected NxC target probabilities, got {probabilities.shape}")
    if not np.issubdtype(probabilities.dtype, np.floating):
        raise ValueError("target probabilities must have floating dtype")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("target probabilities must be finite and nonnegative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("target probabilities must sum to one")
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError(f"smoothing must be positive, got {smoothing}")
    labels = probabilities.argmax(axis=1)
    counts = np.bincount(labels, minlength=probabilities.shape[1]).astype(np.float64)
    class_weights = 1.0 / (counts + float(smoothing))
    pixel_weights = class_weights[labels]
    pixel_weights /= pixel_weights.mean()
    return pixel_weights.astype(np.float32)


def temperature_scale_probabilities(
    probabilities: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if probabilities.ndim < 2:
        raise ValueError("probabilities must contain a class dimension")
    if not torch.isfinite(probabilities).all() or torch.any(probabilities < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    normalizer = probabilities.sum(dim=1)
    if not torch.allclose(
        normalizer,
        torch.ones_like(normalizer),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("probabilities must sum to one along dimension 1")
    log_probabilities = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
    return torch.softmax(log_probabilities / float(temperature), dim=1)


def weighted_distillation_objective(
    observer: nn.Module,
    features: torch.Tensor,
    target_probabilities: torch.Tensor,
    pixel_weights: torch.Tensor,
    *,
    temperature: float,
    l2: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not np.isfinite(l2) or l2 <= 0:
        raise ValueError(f"l2 must be strictly positive, got {l2}")
    if features.ndim != 2:
        raise ValueError(f"features must be NxC, got {tuple(features.shape)}")
    if target_probabilities.ndim != 2:
        raise ValueError(
            f"target_probabilities must be NxK, got {tuple(target_probabilities.shape)}"
        )
    if features.shape[0] != target_probabilities.shape[0]:
        raise ValueError("features and targets must contain the same number of pixels")
    if pixel_weights.shape != (features.shape[0],) or torch.any(pixel_weights <= 0):
        raise ValueError("pixel_weights must be a positive vector with one entry per pixel")

    logits = observer(features[:, :, None, None], output_size=(1, 1))[:, :, 0, 0]
    target = temperature_scale_probabilities(
        target_probabilities,
        temperature=temperature,
    )
    log_prediction = F.log_softmax(logits / float(temperature), dim=1)
    per_pixel_kl = torch.sum(
        target * (target.clamp_min(torch.finfo(target.dtype).tiny).log() - log_prediction),
        dim=1,
    )
    data_loss = torch.sum(pixel_weights * per_pixel_kl) / pixel_weights.sum()
    regularization = sum(parameter.square().sum() for parameter in observer.parameters())
    regularization = regularization * float(l2)
    return data_loss + regularization, data_loss, regularization


def mean_js_divergence(
    target: npt.ArrayLike,
    prediction: npt.ArrayLike,
) -> float:
    first = np.asarray(target, dtype=np.float64)
    second = np.asarray(prediction, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError(
            f"Expected equal NxC probability arrays, got {first.shape} and {second.shape}"
        )
    if np.any(first < 0) or np.any(second < 0):
        raise ValueError("probability arrays must be nonnegative")
    if not np.allclose(first.sum(axis=1), 1.0, atol=1e-6) or not np.allclose(
        second.sum(axis=1),
        1.0,
        atol=1e-6,
    ):
        raise ValueError("probability arrays must sum to one")
    epsilon = np.finfo(np.float64).tiny
    midpoint = 0.5 * (first + second)
    log_midpoint = np.log(np.maximum(midpoint, epsilon))
    kl_first = np.sum(
        first * (np.log(np.maximum(first, epsilon)) - log_midpoint),
        axis=1,
    )
    kl_second = np.sum(
        second * (np.log(np.maximum(second, epsilon)) - log_midpoint),
        axis=1,
    )
    return float(np.mean(0.5 * (kl_first + kl_second)))
