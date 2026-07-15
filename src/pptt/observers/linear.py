from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LinearObserver(nn.Module):
    """A single pointwise projection followed only by deterministic resizing."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        if in_channels <= 0 or num_classes < 2:
            raise ValueError(
                f"Invalid observer dimensions: in_channels={in_channels}, "
                f"num_classes={num_classes}"
            )
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.projection = nn.Conv2d(
            self.in_channels,
            self.num_classes,
            kernel_size=1,
            bias=True,
        )

    def forward(
        self,
        feature: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if feature.ndim != 4 or feature.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected NCHW feature with C={self.in_channels}, got {tuple(feature.shape)}"
            )
        logits = self.projection(feature)
        if logits.shape[-2:] != tuple(output_size):
            logits = F.interpolate(
                logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return logits
