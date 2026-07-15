from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pptt.transitions.tensors import (
    confusions_from_transition,
    validate_label_arrays,
)


def confusion_from_labels(
    y: npt.ArrayLike,
    prediction: npt.ArrayLike,
    num_classes: int,
    *,
    mask: npt.ArrayLike | None = None,
) -> np.ndarray:
    truth, state = validate_label_arrays(y, prediction, num_classes=num_classes)
    if mask is None:
        selected = np.ones(truth.shape, dtype=bool)
    else:
        selected = np.asarray(mask)
        if selected.shape != truth.shape or selected.dtype != np.bool_:
            raise ValueError("mask must be a boolean array with the label shape")
    c = int(num_classes)
    indices = truth[selected].reshape(-1) * c + state[selected].reshape(-1)
    return np.bincount(indices, minlength=c * c).reshape(c, c).astype(
        np.int64,
        copy=False,
    )


def _validate_confusion(confusion: npt.ArrayLike) -> np.ndarray:
    counts = np.asarray(confusion)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError(f"Expected a square confusion matrix, got {counts.shape}")
    if counts.shape[0] < 2:
        raise ValueError("Confusion matrix must contain at least two classes")
    if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise ValueError("Confusion matrix must contain nonnegative integer counts")
    return counts.astype(np.float64, copy=False)


def _divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    zero_division: float,
) -> np.ndarray:
    result = np.full(numerator.shape, float(zero_division), dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


@dataclass(frozen=True)
class SegmentationMetrics:
    dice: np.ndarray
    iou: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    specificity: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "dice": self.dice,
            "iou": self.iou,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
        }


def metrics_from_confusion(
    confusion: npt.ArrayLike,
    *,
    zero_division: float = 0.0,
) -> SegmentationMetrics:
    counts = _validate_confusion(confusion)
    tp = np.diag(counts)
    true_count = counts.sum(axis=1)
    predicted_count = counts.sum(axis=0)
    fn = true_count - tp
    fp = predicted_count - tp
    tn = counts.sum() - tp - fn - fp
    return SegmentationMetrics(
        dice=_divide(2 * tp, 2 * tp + fp + fn, zero_division),
        iou=_divide(tp, tp + fp + fn, zero_division),
        precision=_divide(tp, tp + fp, zero_division),
        recall=_divide(tp, tp + fn, zero_division),
        specificity=_divide(tn, tn + fp, zero_division),
    )


def metric_delta_from_transition(
    tensor: npt.ArrayLike,
    *,
    zero_division: float = 0.0,
) -> SegmentationMetrics:
    before, after = confusions_from_transition(tensor)
    metrics_before = metrics_from_confusion(before, zero_division=zero_division)
    metrics_after = metrics_from_confusion(after, zero_division=zero_division)
    return SegmentationMetrics(
        **{
            name: getattr(metrics_after, name) - getattr(metrics_before, name)
            for name in metrics_before.as_dict()
        }
    )
