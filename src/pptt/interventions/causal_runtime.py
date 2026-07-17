from __future__ import annotations

import numpy as np
import torch
from torch import nn

from pptt.interventions.activation_swap import transform_module_input
from pptt.interventions.causal_tracing import spatial_corrupt_restore
from pptt.pipeline.trace_model import ObserverPathTracer, SlicePathTrace


def trace_spatial_intervention(
    tracer: ObserverPathTracer,
    image: torch.Tensor,
    truth: np.ndarray,
    *,
    module: nn.Module,
    argument_index: int,
    restore_mask: torch.Tensor,
    shift_yx: tuple[int, int],
    alpha: float,
) -> SlicePathTrace:
    """Trace one forward while spatially corrupting and locally restoring a path."""
    hooks_before = len(module._forward_pre_hooks)
    with transform_module_input(
        module,
        argument_index=argument_index,
        transform=lambda activation: spatial_corrupt_restore(
            activation,
            restore_mask=restore_mask,
            shift_yx=shift_yx,
            alpha=alpha,
        ),
    ):
        traced = tracer.trace_slice(image, truth)
    hooks_after = len(module._forward_pre_hooks)
    if hooks_after != hooks_before:
        raise RuntimeError("causal intervention hook was not restored")
    return traced


__all__ = ["trace_spatial_intervention"]
