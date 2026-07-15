from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import nn

from pptt.types import ForwardTrace


class ModelAdapter(nn.Module, ABC):
    """Common prediction and tracing interface for segmentation networks."""

    checkpoint_names: tuple[str, ...]

    @abstractmethod
    def trace(self, x: torch.Tensor) -> ForwardTrace:
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(
        self,
        path: str | Path,
        *,
        map_location: str | torch.device | None = None,
    ) -> str:
        raise NotImplementedError
