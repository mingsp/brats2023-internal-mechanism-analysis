from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pptt.transitions.metrics import confusion_from_labels, metrics_from_confusion


def node_dice_path(
    states: npt.ArrayLike,
    truth: npt.ArrayLike,
    *,
    num_classes: int,
) -> np.ndarray:
    values = np.asarray(states)
    labels = np.asarray(truth)
    if values.ndim < 2 or values.shape[1:] != labels.shape:
        raise ValueError("states must have K followed by the truth shape")
    return np.stack(
        [
            metrics_from_confusion(
                confusion_from_labels(labels, state, num_classes)
            ).dice
            for state in values
        ]
    )


def node_dice_stage_accuracy(
    dice_path: npt.ArrayLike,
    expected_active: npt.ArrayLike,
) -> float:
    values = np.asarray(dice_path, dtype=np.float64)
    expected = np.asarray(expected_active, dtype=bool)
    if values.ndim != 2 or expected.shape != (values.shape[0] - 1,):
        raise ValueError("dice_path and expected_active have incompatible shapes")
    scores = np.mean(np.abs(np.diff(values[:, 1:], axis=0)), axis=1)
    predicted = np.zeros_like(expected)
    active_count = int(expected.sum())
    if active_count:
        predicted[np.argsort(scores)[-active_count:]] = True
    return float(np.mean(predicted == expected))


def adjacent_mask_iou(states: npt.ArrayLike, *, num_classes: int) -> float:
    values = np.asarray(states)
    if values.ndim < 2:
        raise ValueError("states must contain a stage axis")
    scores: list[float] = []
    for left, right in zip(values[:-1], values[1:]):
        for class_index in range(num_classes):
            left_mask = left == class_index
            right_mask = right == class_index
            union = np.logical_or(left_mask, right_mask).sum()
            if union:
                scores.append(float(np.logical_and(left_mask, right_mask).sum() / union))
    return float(np.mean(scores)) if scores else 1.0
