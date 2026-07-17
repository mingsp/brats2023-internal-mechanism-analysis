"""Deterministic helpers for publication-grade PPTT case visualizations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


BRATS_CLASS_COLORS = {
    1: (1.0, 0.0, 0.0),  # necrotic/non-enhancing tumor core
    2: (0.0, 1.0, 0.0),  # peritumoral edema
    3: (1.0, 1.0, 0.0),  # enhancing tumor
}


def select_representative_slice(
    truth: np.ndarray,
    *,
    truth_classes: Sequence[int],
) -> int:
    """Select one fixed slice, preferring complete class coverage and lesion area."""
    labels = np.asarray(truth)
    if labels.ndim != 3:
        raise ValueError("truth must have shape [slices, height, width]")

    classes = tuple(int(value) for value in truth_classes)
    tumor_area = np.count_nonzero(labels > 0, axis=(1, 2))
    complete = np.ones(labels.shape[0], dtype=bool)
    for class_id in classes:
        complete &= np.any(labels == class_id, axis=(1, 2))

    candidates = np.flatnonzero(complete)
    if candidates.size == 0:
        candidates = np.arange(labels.shape[0])
    return int(candidates[np.argmax(tumor_area[candidates])])


def segmentation_overlay(
    image: np.ndarray,
    labels: np.ndarray,
    *,
    colors: Mapping[int, tuple[float, float, float]],
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend class labels over a grayscale image while preserving background pixels."""
    grayscale = np.asarray(image, dtype=np.float32)
    segmentation = np.asarray(labels)
    if grayscale.ndim != 2 or segmentation.shape != grayscale.shape:
        raise ValueError("image and labels must be aligned two-dimensional arrays")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")

    finite = np.isfinite(grayscale)
    if not finite.all():
        raise ValueError("image contains non-finite values")
    lower = float(grayscale.min())
    upper = float(grayscale.max())
    if upper > lower:
        grayscale = (grayscale - lower) / (upper - lower)
    else:
        grayscale = np.zeros_like(grayscale)

    output = np.repeat(grayscale[..., None], 3, axis=-1)
    for class_id, color in colors.items():
        mask = segmentation == int(class_id)
        if not np.any(mask):
            continue
        rgb = np.asarray(color, dtype=np.float32)
        if rgb.shape != (3,):
            raise ValueError("each class color must contain exactly three channels")
        output[mask] = (1.0 - alpha) * output[mask] + alpha * rgb
    return np.clip(output, 0.0, 1.0)


__all__ = [
    "BRATS_CLASS_COLORS",
    "segmentation_overlay",
    "select_representative_slice",
]
