from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class SegmentationLossParts:
    total: torch.Tensor
    cross_entropy: torch.Tensor
    dice: torch.Tensor


def multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    if logits.ndim != 4 or target.shape != (logits.shape[0], *logits.shape[2:]):
        raise ValueError(
            "logits must be NCHW and target must have the corresponding NHW shape"
        )
    if target.dtype != torch.int64:
        raise ValueError("target must use torch.int64 class indices")
    if logits.shape[1] < 2 or epsilon <= 0:
        raise ValueError("At least two classes and positive epsilon are required")
    if torch.any(target < 0) or torch.any(target >= logits.shape[1]):
        raise ValueError("target contains a class outside the logits range")
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2)
    one_hot = one_hot.to(dtype=probabilities.dtype)
    dimensions = (0, 2, 3)
    intersection = torch.sum(probabilities * one_hot, dim=dimensions)
    denominator = torch.sum(probabilities, dim=dimensions) + torch.sum(
        one_hot,
        dim=dimensions,
    )
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice.mean()


def segmentation_loss_parts(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    cross_entropy_weight: float = 0.2,
    dice_weight: float = 0.8,
) -> SegmentationLossParts:
    if cross_entropy_weight < 0 or dice_weight < 0:
        raise ValueError("Loss weights must be nonnegative")
    if abs(cross_entropy_weight + dice_weight - 1.0) > 1e-12:
        raise ValueError("Loss weights must sum to one")
    log_probabilities = F.log_softmax(logits, dim=1)
    cross_entropy = -log_probabilities.gather(1, target[:, None]).mean()
    dice = multiclass_dice_loss(logits, target)
    total = cross_entropy_weight * cross_entropy + dice_weight * dice
    return SegmentationLossParts(
        total=total,
        cross_entropy=cross_entropy,
        dice=dice,
    )


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    cross_entropy_weight: float = 0.2,
    dice_weight: float = 0.8,
) -> torch.Tensor:
    return segmentation_loss_parts(
        logits,
        target,
        cross_entropy_weight=cross_entropy_weight,
        dice_weight=dice_weight,
    ).total
