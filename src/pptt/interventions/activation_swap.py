from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Callable, Iterator

import torch
from torch import nn


def swap_activation_pairs(
    activation: torch.Tensor,
    target_indices: torch.Tensor,
    control_indices: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    if activation.ndim != 4 or activation.shape[0] != 1:
        raise ValueError("activation must have shape 1xCxHxW")
    if target_indices.ndim != 1 or control_indices.shape != target_indices.shape:
        raise ValueError("target and control indices must be equal vectors")
    if target_indices.dtype != torch.int64 or control_indices.dtype != torch.int64:
        raise ValueError("activation indices must be int64")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be inside [0, 1]")
    flat = activation.flatten(2).clone()
    target = target_indices.to(flat.device)
    control = control_indices.to(flat.device)
    if torch.any(target < 0) or torch.any(control < 0):
        raise ValueError("activation indices must be nonnegative")
    if torch.any(target >= flat.shape[2]) or torch.any(control >= flat.shape[2]):
        raise ValueError("activation index exceeds the spatial field")
    if torch.unique(torch.cat([target, control])).numel() != 2 * target.numel():
        raise ValueError("matched target and control indices must be unique and disjoint")
    native_target = flat[:, :, target].clone()
    native_control = flat[:, :, control].clone()
    flat[:, :, target] = (1.0 - alpha) * native_target + alpha * native_control
    flat[:, :, control] = (1.0 - alpha) * native_control + alpha * native_target
    return flat.view_as(activation)


def replace_activation_vectors(
    activation: torch.Tensor,
    destination_indices: torch.Tensor,
    source_vectors: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    """Replace selected spatial vectors with matched natural vectors."""
    if activation.ndim != 4 or activation.shape[0] != 1:
        raise ValueError("activation must have shape 1xCxHxW")
    if destination_indices.ndim != 1 or destination_indices.dtype != torch.int64:
        raise ValueError("destination_indices must be a one-dimensional int64 tensor")
    if source_vectors.ndim != 2:
        raise ValueError("source_vectors must have shape NxC")
    if source_vectors.shape != (destination_indices.numel(), activation.shape[1]):
        raise ValueError("source vectors do not match destinations and channels")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be inside [0, 1]")
    flat = activation.flatten(2).clone()
    destinations = destination_indices.to(flat.device)
    if torch.any(destinations < 0) or torch.any(destinations >= flat.shape[2]):
        raise ValueError("destination index exceeds the spatial field")
    if torch.unique(destinations).numel() != destinations.numel():
        raise ValueError("destination indices must be unique")
    native = flat[0, :, destinations].transpose(0, 1).clone()
    sources = source_vectors.to(device=flat.device, dtype=flat.dtype)
    changed = (1.0 - alpha) * native + alpha * sources
    flat[0, :, destinations] = changed.transpose(0, 1)
    return flat.view_as(activation)


def spatially_shift_activation(
    activation: torch.Tensor,
    *,
    shift_y: int,
    shift_x: int,
) -> torch.Tensor:
    return torch.roll(activation, shifts=(int(shift_y), int(shift_x)), dims=(-2, -1))


@contextmanager
def transform_module_input(
    module: nn.Module,
    *,
    argument_index: int | None = None,
    argument_name: str | None = None,
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> Iterator[None]:
    if (argument_index is None) == (argument_name is None):
        raise ValueError("provide exactly one input argument selector")
    if argument_index is not None and argument_index < 0:
        raise ValueError("argument_index must be nonnegative")
    if argument_name is not None and not argument_name:
        raise ValueError("argument_name must be nonempty")

    if argument_name is not None:
        def keyword_hook(
            _module: nn.Module,
            inputs: tuple[object, ...],
            kwargs: dict[str, object],
        ):
            if argument_name not in kwargs or not isinstance(
                kwargs[argument_name],
                torch.Tensor,
            ):
                raise ValueError("Selected keyword module input is not a tensor")
            changed_kwargs = dict(kwargs)
            changed_kwargs[argument_name] = transform(kwargs[argument_name])
            return inputs, changed_kwargs

        handle = module.register_forward_pre_hook(keyword_hook, with_kwargs=True)
    else:
        assert argument_index is not None

        def positional_hook(_module: nn.Module, inputs: tuple[object, ...]):
            if argument_index >= len(inputs) or not isinstance(
                inputs[argument_index],
                torch.Tensor,
            ):
                raise ValueError("Selected positional module input is not a tensor")
            changed = list(inputs)
            changed[argument_index] = transform(inputs[argument_index])
            return tuple(changed)

        handle = module.register_forward_pre_hook(positional_hook)
    try:
        yield
    finally:
        handle.remove()


def capture_module_output(
    model: nn.Module,
    module: nn.Module,
    image: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    captured: list[torch.Tensor] = []

    def hook(_module: nn.Module, _inputs: tuple[object, ...], output: torch.Tensor):
        if not isinstance(output, torch.Tensor):
            raise TypeError("Captured module output must be a tensor")
        captured.append(output)

    handle = module.register_forward_hook(hook)
    try:
        logits = model(image)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("Module output capture did not occur exactly once")
    return logits, captured[0]
