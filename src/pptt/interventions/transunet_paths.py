from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class InterventionPath:
    path_id: str
    source_node: str
    receiver_node: str
    source_index: int
    receiver_index: int
    source_module_path: str
    receiver_module_path: str
    argument_index: int
    argument_name: str | None
    topology_order: int

    def __post_init__(self) -> None:
        text_fields = (
            self.path_id,
            self.source_node,
            self.receiver_node,
            self.source_module_path,
            self.receiver_module_path,
        )
        if any(not value.strip() for value in text_fields):
            raise ValueError("intervention path text fields must be nonempty")
        if not 0 <= int(self.source_index) < int(self.receiver_index):
            raise ValueError("source_index must be smaller than receiver_index")
        if int(self.argument_index) < 0 or int(self.topology_order) < 0:
            raise ValueError("argument_index and topology_order must be nonnegative")
        if self.argument_index > 0 and not self.argument_name:
            raise ValueError("TransUNet non-primary inputs require a keyword name")
        if self.argument_name is not None and not self.argument_name.strip():
            raise ValueError("argument_name must be nonempty when provided")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def declared_transunet_decoder_paths() -> tuple[InterventionPath, ...]:
    """Return the result-independent decoder input paths in topology order."""
    return (
        InterventionPath(
            path_id="bottleneck_to_up1",
            source_node="down4",
            receiver_node="up1",
            source_index=3,
            receiver_index=4,
            source_module_path="decoder.conv_more",
            receiver_module_path="decoder.blocks.0",
            argument_index=0,
            argument_name=None,
            topology_order=0,
        ),
        InterventionPath(
            path_id="skip_down3_to_up1",
            source_node="down3",
            receiver_node="up1",
            source_index=2,
            receiver_index=4,
            source_module_path=(
                "transformer.embeddings.hybrid_model.feature_taps.2"
            ),
            receiver_module_path="decoder.blocks.0",
            argument_index=1,
            argument_name="skip",
            topology_order=1,
        ),
        InterventionPath(
            path_id="skip_down2_to_up2",
            source_node="down2",
            receiver_node="up2",
            source_index=1,
            receiver_index=5,
            source_module_path=(
                "transformer.embeddings.hybrid_model.feature_taps.1"
            ),
            receiver_module_path="decoder.blocks.1",
            argument_index=1,
            argument_name="skip",
            topology_order=2,
        ),
        InterventionPath(
            path_id="skip_down1_to_up3",
            source_node="down1",
            receiver_node="up3",
            source_index=0,
            receiver_index=6,
            source_module_path=(
                "transformer.embeddings.hybrid_model.feature_taps.0"
            ),
            receiver_module_path="decoder.blocks.2",
            argument_index=1,
            argument_name="skip",
            topology_order=3,
        ),
    )


def resolve_module(root: nn.Module, dotted_path: str) -> nn.Module:
    current: Any = root
    for component in str(dotted_path).split("."):
        if not component:
            raise ValueError(f"invalid module path: {dotted_path!r}")
        if component.isdigit():
            try:
                current = current[int(component)]
            except (IndexError, KeyError, TypeError) as exc:
                raise KeyError(f"cannot resolve {dotted_path!r}") from exc
        else:
            try:
                current = getattr(current, component)
            except AttributeError as exc:
                raise KeyError(f"cannot resolve {dotted_path!r}") from exc
    if not isinstance(current, nn.Module):
        raise TypeError(f"resolved object is not a module: {dotted_path!r}")
    return current


def _receiver_argument(
    path: InterventionPath,
    inputs: tuple[object, ...],
    kwargs: dict[str, object],
) -> torch.Tensor:
    if path.argument_name is not None and path.argument_name in kwargs:
        selected = kwargs[path.argument_name]
    elif path.argument_index < len(inputs):
        selected = inputs[path.argument_index]
    else:
        raise RuntimeError(f"receiver input is missing for {path.path_id}")
    if not isinstance(selected, torch.Tensor):
        raise TypeError(f"receiver input is not a tensor for {path.path_id}")
    return selected


def audit_path_activation_identity(
    adapter: nn.Module,
    path: InterventionPath,
    image: torch.Tensor,
) -> dict[str, Any]:
    model = getattr(adapter, "model", None)
    if not isinstance(model, nn.Module):
        raise TypeError("adapter must expose its wrapped model")
    source = resolve_module(model, path.source_module_path)
    receiver = resolve_module(model, path.receiver_module_path)
    source_values: list[torch.Tensor] = []
    receiver_values: list[torch.Tensor] = []
    source_before = len(source._forward_hooks)
    receiver_before = len(receiver._forward_pre_hooks)

    def capture_source(
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: object,
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("declared source output is not a tensor")
        source_values.append(output)

    def capture_receiver(
        _module: nn.Module,
        inputs: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        receiver_values.append(_receiver_argument(path, inputs, kwargs))

    source_handle = source.register_forward_hook(capture_source)
    receiver_handle = receiver.register_forward_pre_hook(
        capture_receiver,
        with_kwargs=True,
    )
    try:
        with torch.inference_mode():
            adapter(image)
    finally:
        receiver_handle.remove()
        source_handle.remove()
    if len(source_values) != 1 or len(receiver_values) != 1:
        raise RuntimeError(f"path capture count differs from one for {path.path_id}")
    source_value = source_values[0].detach()
    receiver_value = receiver_values[0].detach()
    shape_equal = source_value.shape == receiver_value.shape
    max_abs_error = (
        float((source_value - receiver_value).abs().max().item())
        if shape_equal and source_value.numel()
        else float("inf")
    )
    return {
        "path": path.to_dict(),
        "source_capture_count": len(source_values),
        "receiver_capture_count": len(receiver_values),
        "source_shape": list(source_value.shape),
        "receiver_shape": list(receiver_value.shape),
        "shape_equal": bool(shape_equal),
        "exact_equal": bool(shape_equal and torch.equal(source_value, receiver_value)),
        "max_abs_error": max_abs_error,
        "source_hook_count_after": len(source._forward_hooks) - source_before,
        "receiver_hook_count_after": (
            len(receiver._forward_pre_hooks) - receiver_before
        ),
    }


__all__ = [
    "InterventionPath",
    "audit_path_activation_identity",
    "declared_transunet_decoder_paths",
    "resolve_module",
]
