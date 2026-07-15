from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _normalize(map_: np.ndarray) -> np.ndarray:
    values = map_ - map_.min()
    maximum = values.max()
    return values / maximum if maximum > 0 else np.zeros_like(values)


def linear_gradcam(
    feature: npt.ArrayLike,
    readout_weight: npt.ArrayLike,
    *,
    target_class: int,
) -> np.ndarray:
    activation = np.asarray(feature, dtype=np.float64)
    weights = np.asarray(readout_weight, dtype=np.float64)
    if activation.ndim != 3 or weights.ndim != 2 or activation.shape[0] != weights.shape[1]:
        raise ValueError("feature and readout_weight have incompatible shapes")
    map_ = np.maximum(
        np.einsum("c,chw->hw", weights[target_class], activation),
        0.0,
    )
    return _normalize(map_)


def linear_layercam(
    feature: npt.ArrayLike,
    readout_weight: npt.ArrayLike,
    *,
    target_class: int,
) -> np.ndarray:
    activation = np.asarray(feature, dtype=np.float64)
    weights = np.asarray(readout_weight, dtype=np.float64)
    if activation.ndim != 3 or weights.ndim != 2 or activation.shape[0] != weights.shape[1]:
        raise ValueError("feature and readout_weight have incompatible shapes")
    positive_gradient = np.maximum(weights[target_class], 0.0)[:, None, None]
    map_ = np.maximum(np.sum(positive_gradient * activation, axis=0), 0.0)
    return _normalize(map_)


def top_fraction_iou(
    saliency: npt.ArrayLike,
    target_mask: npt.ArrayLike,
    *,
    fraction: float = 0.10,
) -> float:
    values = np.asarray(saliency, dtype=np.float64)
    target = np.asarray(target_mask, dtype=bool)
    if values.shape != target.shape or values.ndim != 2:
        raise ValueError("saliency and target_mask must be equal 2D arrays")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be inside (0, 1]")
    count = max(1, int(np.ceil(values.size * fraction)))
    indices = np.argpartition(values.reshape(-1), -count)[-count:]
    selected = np.zeros(values.size, dtype=bool)
    selected[indices] = True
    selected = selected.reshape(values.shape)
    union = np.logical_or(selected, target).sum()
    return float(np.logical_and(selected, target).sum() / union) if union else 1.0
