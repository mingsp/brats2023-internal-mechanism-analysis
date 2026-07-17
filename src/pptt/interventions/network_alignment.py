from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from pptt.interventions.causal_tracing import (
    match_feature_controls,
    registered_feature_strata,
)


@dataclass(frozen=True)
class NodeRestoreMasks:
    target: torch.Tensor
    control: torch.Tensor | None
    target_count: int
    control_count: int

    def __post_init__(self) -> None:
        if self.target.ndim != 2 or self.target.dtype != torch.bool:
            raise ValueError("target mask must be a two-dimensional boolean tensor")
        target_count = int(self.target.sum().item())
        if int(self.target_count) != target_count:
            raise ValueError("target_count differs from the target mask")
        if self.control is None:
            if int(self.control_count) != 0:
                raise ValueError("control_count must be zero when control is unavailable")
            return
        if self.control.ndim != 2 or self.control.dtype != torch.bool:
            raise ValueError("control mask must be a two-dimensional boolean tensor")
        if self.control.shape != self.target.shape:
            raise ValueError("target and control masks must share a spatial shape")
        control_count = int(self.control.sum().item())
        if int(self.control_count) != control_count:
            raise ValueError("control_count differs from the control mask")
        if torch.any(self.target & self.control):
            raise ValueError("target and control masks must be disjoint")
        if target_count != control_count:
            raise ValueError("registered target and control masks must have equal counts")


def spatial_roll_module_output(
    activation: torch.Tensor,
    *,
    shift_yx: tuple[int, int],
) -> torch.Tensor:
    if activation.ndim != 4:
        raise ValueError("activation must have shape BxCxHxW")
    if len(shift_yx) != 2:
        raise ValueError("shift_yx must contain two integers")
    shift_y, shift_x = (int(value) for value in shift_yx)
    return torch.roll(activation, shifts=(shift_y, shift_x), dims=(-2, -1))


def input_equivalent_feature_shift(
    *,
    input_shape: tuple[int, int],
    feature_shape: tuple[int, int],
    input_shift_yx: tuple[int, int],
) -> tuple[int, int]:
    if any(int(value) <= 0 for value in (*input_shape, *feature_shape)):
        raise ValueError("input and feature shapes must be positive")
    if any(int(value) <= 0 for value in input_shift_yx):
        raise ValueError("input-equivalent shifts must be positive")
    return tuple(
        max(1, int(round(int(shift) * int(feature) / int(source))))
        for shift, feature, source in zip(
            input_shift_yx,
            feature_shape,
            input_shape,
        )
    )


def channel_multiset_preserved(
    clean: torch.Tensor,
    shifted: torch.Tensor,
) -> bool:
    if clean.shape != shifted.shape or clean.ndim != 4:
        return False
    clean_sorted = torch.sort(clean.flatten(2), dim=-1).values
    shifted_sorted = torch.sort(shifted.flatten(2), dim=-1).values
    return bool(torch.equal(clean_sorted, shifted_sorted))


def restore_clean_positions(
    corrupted: torch.Tensor,
    clean: torch.Tensor,
    restore_mask: torch.Tensor,
) -> torch.Tensor:
    if corrupted.ndim != 4 or clean.ndim != 4:
        raise ValueError("clean and corrupted activations must have shape BxCxHxW")
    if clean.shape[1:] != corrupted.shape[1:]:
        raise ValueError("clean and corrupted activations must share CxHxW")
    if clean.shape[0] not in (1, corrupted.shape[0]):
        raise ValueError("clean activation batch dimension does not broadcast")
    mask = restore_mask.to(device=corrupted.device)
    if mask.dtype != torch.bool:
        raise ValueError("restore_mask must be boolean")
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None]
    if mask.ndim != 4 or mask.shape[-2:] != corrupted.shape[-2:]:
        raise ValueError("restore_mask must align with activation spatial dimensions")
    if mask.shape[0] not in (1, corrupted.shape[0]):
        raise ValueError("restore_mask batch dimension does not broadcast")
    source = clean.to(device=corrupted.device, dtype=corrupted.dtype)
    return torch.where(mask, source, corrupted)


def map_transition_mask(
    mask: np.ndarray,
    *,
    output_shape: tuple[int, int],
) -> np.ndarray:
    selected = np.asarray(mask)
    if selected.ndim != 2 or selected.dtype != np.bool_:
        raise ValueError("transition mask must be a two-dimensional boolean array")
    if len(output_shape) != 2 or any(int(value) <= 0 for value in output_shape):
        raise ValueError("output_shape must contain two positive dimensions")
    tensor = torch.from_numpy(selected.astype(np.float32))[None, None]
    mapped = torch.nn.functional.adaptive_max_pool2d(tensor, output_shape)
    return mapped[0, 0].numpy() > 0


def match_node_restore_masks(
    *,
    target_mask: np.ndarray,
    control_pool: np.ndarray,
    truth_class: np.ndarray,
    boundary_distance: np.ndarray,
    activation_norm: np.ndarray,
    boundary_bin_edges: Sequence[float],
    activation_norm_quantile_bins: int,
) -> NodeRestoreMasks:
    target = np.asarray(target_mask)
    controls = np.asarray(control_pool)
    classes = np.asarray(truth_class)
    boundary = np.asarray(boundary_distance)
    norm = np.asarray(activation_norm)
    shapes = {value.shape for value in (target, controls, classes, boundary, norm)}
    if len(shapes) != 1 or target.ndim != 2:
        raise ValueError("node matching arrays must be aligned and two-dimensional")
    if target.dtype != np.bool_ or controls.dtype != np.bool_:
        raise ValueError("target_mask and control_pool must be boolean")
    if not np.isfinite(boundary).all() or not np.isfinite(norm).all():
        raise ValueError("matching values must be finite")
    controls = controls & ~target
    eligible = target | controls
    if not target.any():
        return NodeRestoreMasks(
            target=torch.from_numpy(target.copy()),
            control=None,
            target_count=0,
            control_count=0,
        )
    if not controls.any():
        return NodeRestoreMasks(
            target=torch.from_numpy(target.copy()),
            control=None,
            target_count=int(target.sum()),
            control_count=0,
        )
    boundary_stratum, norm_stratum, _ = registered_feature_strata(
        boundary,
        norm,
        eligible_mask=eligible,
        boundary_bin_edges=boundary_bin_edges,
        activation_norm_quantile_bins=activation_norm_quantile_bins,
    )
    matches = match_feature_controls(
        target_mask=target,
        control_mask=controls,
        truth_class=classes,
        boundary_distance=boundary,
        activation_norm=norm,
        boundary_stratum=boundary_stratum,
        activation_stratum=norm_stratum,
    )
    target_count = int(target.sum())
    if matches.matched_count != target_count:
        return NodeRestoreMasks(
            target=torch.from_numpy(target.copy()),
            control=None,
            target_count=target_count,
            control_count=0,
        )
    control = np.zeros_like(target, dtype=bool)
    control.reshape(-1)[matches.control_indices] = True
    return NodeRestoreMasks(
        target=torch.from_numpy(target.copy()),
        control=torch.from_numpy(control),
        target_count=target_count,
        control_count=int(control.sum()),
    )


__all__ = [
    "NodeRestoreMasks",
    "channel_multiset_preserved",
    "input_equivalent_feature_shift",
    "map_transition_mask",
    "match_node_restore_masks",
    "restore_clean_positions",
    "spatial_roll_module_output",
]
