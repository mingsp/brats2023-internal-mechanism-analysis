from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ForwardTrace:
    """One model prediction together with ordered internal activations."""

    logits: torch.Tensor
    activations: Mapping[str, torch.Tensor]
