from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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
    def checkpoint_module(self, name: str) -> nn.Module:
        raise NotImplementedError

    @contextmanager
    def transform_checkpoint_output(
        self,
        name: str,
        transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> Iterator[None]:
        """Temporarily replace one registered checkpoint output."""

        module = self.checkpoint_module(name)

        def hook(
            _module: nn.Module,
            _inputs: tuple[object, ...],
            output: torch.Tensor,
        ) -> torch.Tensor:
            if not isinstance(output, torch.Tensor):
                raise TypeError("checkpoint output must be a tensor")
            changed = transform(output)
            if not isinstance(changed, torch.Tensor):
                raise TypeError("checkpoint transform must return a tensor")
            if changed.shape != output.shape:
                raise ValueError("checkpoint transform changed tensor shape")
            if changed.dtype != output.dtype or changed.device != output.device:
                raise ValueError(
                    "checkpoint transform changed tensor dtype or device"
                )
            return changed

        handle = module.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @abstractmethod
    def randomization_module(self, name: str) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(
        self,
        path: str | Path,
        *,
        map_location: str | torch.device | None = None,
    ) -> str:
        raise NotImplementedError
