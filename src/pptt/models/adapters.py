from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from pptt.hooks.checkpoints import load_checkpoint
from pptt.models.protocol import ModelAdapter
from pptt.models.unet import UNetBaseline, UNetNoSkip
from pptt.types import ForwardTrace


UNET_CHECKPOINT_NAMES = (
    "down1",
    "down2",
    "down3",
    "down4",
    "up1",
    "up2",
    "up3",
    "up4",
)


class UNetAdapter(ModelAdapter):
    checkpoint_names = UNET_CHECKPOINT_NAMES

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def trace(self, x: torch.Tensor) -> ForwardTrace:
        captured: dict[str, torch.Tensor] = {}
        handles = []

        def capture(name: str):
            def hook(
                _module: nn.Module,
                _inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
            ) -> None:
                if not isinstance(output, torch.Tensor):
                    raise TypeError(f"Checkpoint {name} did not return a tensor")
                captured[name] = output

            return hook

        try:
            for name in self.checkpoint_names:
                module = getattr(self.model, name)
                handles.append(module.register_forward_hook(capture(name)))
            logits = self.model(x)
        finally:
            for handle in handles:
                handle.remove()

        if tuple(captured) != self.checkpoint_names:
            missing = tuple(
                name for name in self.checkpoint_names if name not in captured
            )
            raise RuntimeError(f"Trace did not capture all checkpoints: {missing}")
        return ForwardTrace(
            logits=logits,
            activations=OrderedDict(
                (name, captured[name]) for name in self.checkpoint_names
            ),
        )

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        map_location: str | torch.device | None = None,
    ) -> str:
        return load_checkpoint(self.model, path, map_location=map_location)


def build_adapter(
    name: str,
    n_channels: int = 4,
    num_classes: int = 4,
    bilinear: bool = False,
) -> ModelAdapter:
    builders: dict[str, type[nn.Module]] = {
        "unet_baseline": UNetBaseline,
        "unet_noskip": UNetNoSkip,
    }
    try:
        model_type = builders[name]
    except KeyError as exc:
        supported = ", ".join(sorted(builders))
        raise ValueError(
            f"Unknown model adapter {name!r}; expected one of: {supported}"
        ) from exc
    model = model_type(
        n_channels=n_channels,
        n_classes=num_classes,
        bilinear=bilinear,
    )
    return UNetAdapter(model)
