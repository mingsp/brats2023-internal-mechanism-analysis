from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class StateExchange:
    """Closed-form edit and the numerical audit of its defining constraint."""

    delta_h: torch.Tensor
    reconstructed_delta_logits: torch.Tensor
    target_max_abs_error: float
    frobenius_norm: float
    effective_rank_spatial: int
    effective_rank_channel: int


def _shape2(value: Sequence[int], *, name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain two dimensions")
    shape = (int(value[0]), int(value[1]))
    if any(dimension <= 0 for dimension in shape):
        raise ValueError(f"{name} dimensions must be positive")
    return shape


def bilinear_resize_rows(
    native_shape: Sequence[int],
    output_shape: Sequence[int],
    output_indices: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Return selected rows of PyTorch's align_corners=False resize matrix."""

    native_h, native_w = _shape2(native_shape, name="native_shape")
    output_h, output_w = _shape2(output_shape, name="output_shape")
    indices = torch.as_tensor(output_indices, device=device)
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("output_indices must be a nonempty vector")
    if indices.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError("output_indices must have integer dtype")
    indices = indices.to(dtype=torch.int64)
    output_size = output_h * output_w
    if torch.any(indices < 0) or torch.any(indices >= output_size):
        raise ValueError("output_indices contain an out-of-range position")

    output_y = torch.div(indices, output_w, rounding_mode="floor")
    output_x = indices.remainder(output_w)
    source_y = (output_y.to(dtype) + 0.5) * native_h / output_h - 0.5
    source_x = (output_x.to(dtype) + 0.5) * native_w / output_w - 0.5
    y0_unclamped = torch.floor(source_y).to(torch.int64)
    x0_unclamped = torch.floor(source_x).to(torch.int64)
    y1_unclamped = y0_unclamped + 1
    x1_unclamped = x0_unclamped + 1
    wy1 = source_y - y0_unclamped.to(dtype)
    wx1 = source_x - x0_unclamped.to(dtype)
    wy0 = 1.0 - wy1
    wx0 = 1.0 - wx1
    y0 = y0_unclamped.clamp(0, native_h - 1)
    y1 = y1_unclamped.clamp(0, native_h - 1)
    x0 = x0_unclamped.clamp(0, native_w - 1)
    x1 = x1_unclamped.clamp(0, native_w - 1)

    rows = torch.zeros(
        (indices.numel(), native_h * native_w),
        dtype=dtype,
        device=indices.device,
    )
    neighbors = (
        (y0 * native_w + x0, wy0 * wx0),
        (y0 * native_w + x1, wy0 * wx1),
        (y1 * native_w + x0, wy1 * wx0),
        (y1 * native_w + x1, wy1 * wx1),
    )
    for columns, weights in neighbors:
        rows.scatter_add_(1, columns[:, None], weights[:, None])
    return rows


def bilinear_resize_matrix(
    native_shape: Sequence[int],
    output_shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Return the complete resize matrix; formal runs should request rows only."""

    output_h, output_w = _shape2(output_shape, name="output_shape")
    indices = torch.arange(output_h * output_w, device=device, dtype=torch.int64)
    return bilinear_resize_rows(
        native_shape,
        output_shape,
        indices,
        dtype=dtype,
        device=device,
    )


def _validated_matrix(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional tensor")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite floating-point tensor")
    if value.numel() == 0:
        raise ValueError(f"{name} must be nonempty")
    return value


def _pseudoinverse(matrix: torch.Tensor, *, rcond: float) -> torch.Tensor:
    if matrix.shape[0] <= matrix.shape[1]:
        gram = matrix @ matrix.T
        return matrix.T @ torch.linalg.pinv(gram, rtol=float(rcond))
    gram = matrix.T @ matrix
    return torch.linalg.pinv(gram, rtol=float(rcond)) @ matrix.T


def minimum_norm_state_exchange(
    resize_rows: torch.Tensor,
    observer_weight: torch.Tensor,
    delta_logits: torch.Tensor,
    *,
    rcond: float = 1.0e-7,
) -> StateExchange:
    """Solve min ||dH|| subject to R dH W^T = dL in closed form."""

    spatial = _validated_matrix(resize_rows, name="resize_rows")
    weight = _validated_matrix(observer_weight, name="observer_weight")
    target = _validated_matrix(delta_logits, name="delta_logits")
    if spatial.device != weight.device or spatial.device != target.device:
        raise ValueError("operator tensors must share a device")
    if spatial.dtype != weight.dtype or spatial.dtype != target.dtype:
        raise ValueError("operator tensors must share a dtype")
    if target.shape != (spatial.shape[0], weight.shape[0]):
        raise ValueError(
            "delta_logits must have shape resize_row_count x observer_class_count"
        )
    if not torch.isfinite(torch.tensor(float(rcond))) or float(rcond) <= 0:
        raise ValueError("rcond must be finite and positive")
    rank_spatial = int(torch.linalg.matrix_rank(spatial).item())
    rank_channel = int(torch.linalg.matrix_rank(weight).item())
    if rank_spatial == 0 or rank_channel == 0:
        raise ValueError("state exchange requires nonzero spatial and channel rank")

    spatial_plus = _pseudoinverse(spatial, rcond=float(rcond))
    weight_plus = _pseudoinverse(weight, rcond=float(rcond))
    delta_h = spatial_plus @ target @ weight_plus.T
    reconstructed = spatial @ delta_h @ weight.T
    residual = reconstructed - target
    return StateExchange(
        delta_h=delta_h,
        reconstructed_delta_logits=reconstructed,
        target_max_abs_error=float(residual.abs().max().item()),
        frobenius_norm=float(torch.linalg.vector_norm(delta_h).item()),
        effective_rank_spatial=rank_spatial,
        effective_rank_channel=rank_channel,
    )


def equal_norm_nullspace_control(
    observer_weight: torch.Tensor,
    task_edit: torch.Tensor,
    *,
    generator: torch.Generator,
    rcond: float = 1.0e-7,
) -> torch.Tensor:
    """Return an equal-norm edit in the observer channel nullspace."""

    weight = _validated_matrix(observer_weight, name="observer_weight")
    edit = _validated_matrix(task_edit, name="task_edit")
    if weight.device != edit.device or weight.dtype != edit.dtype:
        raise ValueError("observer_weight and task_edit must share dtype and device")
    if edit.shape[1] != weight.shape[1]:
        raise ValueError("task_edit channel count differs from observer_weight")
    weight_plus = _pseudoinverse(weight, rcond=float(rcond))
    projector = torch.eye(
        weight.shape[1],
        dtype=weight.dtype,
        device=weight.device,
    ) - weight_plus @ weight
    projector_rank = int(torch.linalg.matrix_rank(projector).item())
    if projector_rank == 0:
        raise ValueError("observer has no available channel nullspace")
    random = torch.randn(
        edit.shape,
        dtype=edit.dtype,
        device=edit.device,
        generator=generator,
    )
    control = random @ projector
    control_norm = torch.linalg.vector_norm(control)
    task_norm = torch.linalg.vector_norm(edit)
    tolerance = torch.finfo(edit.dtype).eps * max(edit.numel(), 1)
    if float(control_norm) <= tolerance:
        raise ValueError("sampled nullspace control has zero numerical norm")
    if float(task_norm) <= tolerance:
        raise ValueError("task_edit has zero numerical norm")
    return control * (task_norm / control_norm)


def project_feature_edit(
    delta_h: torch.Tensor,
    *,
    native_shape: Sequence[int],
    observer_weight: torch.Tensor,
    output_shape: Sequence[int],
) -> torch.Tensor:
    """Project a flattened feature edit through the observer and spatial resize."""

    edit = _validated_matrix(delta_h, name="delta_h")
    weight = _validated_matrix(observer_weight, name="observer_weight")
    native_h, native_w = _shape2(native_shape, name="native_shape")
    output_h, output_w = _shape2(output_shape, name="output_shape")
    if edit.shape != (native_h * native_w, weight.shape[1]):
        raise ValueError("delta_h shape does not match native geometry and channels")
    if edit.device != weight.device or edit.dtype != weight.dtype:
        raise ValueError("delta_h and observer_weight must share dtype and device")
    native_logits = (edit @ weight.T).T.reshape(
        1,
        weight.shape[0],
        native_h,
        native_w,
    )
    return F.interpolate(
        native_logits,
        size=(output_h, output_w),
        mode="bilinear",
        align_corners=False,
    )[0]


def apply_flat_feature_edit(
    activation: torch.Tensor,
    delta_h: torch.Tensor,
    *,
    doses: torch.Tensor,
) -> torch.Tensor:
    """Add one flattened edit to a batch with one scalar dose per item."""

    if activation.ndim != 4 or not activation.is_floating_point():
        raise ValueError("activation must be a floating BxCxHxW tensor")
    edit = _validated_matrix(delta_h, name="delta_h").to(
        device=activation.device,
        dtype=activation.dtype,
    )
    dose = torch.as_tensor(
        doses,
        device=activation.device,
        dtype=activation.dtype,
    )
    if dose.ndim != 1 or dose.numel() == 0 or not torch.isfinite(dose).all():
        raise ValueError("doses must be a nonempty finite vector")
    batch = activation
    if batch.shape[0] == 1 and dose.numel() > 1:
        batch = batch.expand(dose.numel(), -1, -1, -1).clone()
    if batch.shape[0] != dose.numel():
        raise ValueError("activation batch size must equal the dose count")
    if edit.shape != (batch.shape[-2] * batch.shape[-1], batch.shape[1]):
        raise ValueError("delta_h shape does not match activation geometry")
    spatial_edit = edit.T.reshape(1, batch.shape[1], *batch.shape[-2:])
    return batch + dose[:, None, None, None] * spatial_edit


__all__ = [
    "StateExchange",
    "apply_flat_feature_edit",
    "bilinear_resize_matrix",
    "bilinear_resize_rows",
    "equal_norm_nullspace_control",
    "minimum_norm_state_exchange",
    "project_feature_edit",
]
