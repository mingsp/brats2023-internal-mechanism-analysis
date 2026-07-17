from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from pptt.interventions.network_alignment import (
    NodeRestoreMasks,
    channel_multiset_preserved,
    input_equivalent_feature_shift,
    restore_clean_positions,
    spatial_roll_module_output,
)
from pptt.models.protocol import ModelAdapter
from pptt.types import ForwardTrace


@dataclass(frozen=True)
class NetworkAlignmentForwardSet:
    clean_logits: np.ndarray
    corrupt_logits: np.ndarray
    root_restored_logits: np.ndarray
    target_restored_logits: dict[str, np.ndarray]
    control_restored_logits: dict[str, np.ndarray | None]
    audits: dict[str, object]


@contextmanager
def module_output_transforms(
    transforms: Mapping[nn.Module, Callable[[torch.Tensor], torch.Tensor]],
) -> Iterator[None]:
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(transform: Callable[[torch.Tensor], torch.Tensor]):
        def hook(
            _module: nn.Module,
            _inputs: tuple[object, ...],
            output: torch.Tensor,
        ) -> torch.Tensor:
            if not isinstance(output, torch.Tensor):
                raise TypeError("registered checkpoint output must be a tensor")
            changed = transform(output)
            if not isinstance(changed, torch.Tensor):
                raise TypeError("module output transform must return a tensor")
            return changed

        return hook

    try:
        for module, transform in transforms.items():
            handles.append(module.register_forward_hook(make_hook(transform)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _hook_count(adapter: ModelAdapter) -> int:
    return sum(
        len(adapter.checkpoint_module(name)._forward_hooks)
        for name in adapter.checkpoint_names
    )


def _forward_with_transforms(
    adapter: ModelAdapter,
    image: torch.Tensor,
    transforms: Mapping[nn.Module, Callable[[torch.Tensor], torch.Tensor]],
) -> torch.Tensor:
    with module_output_transforms(transforms):
        logits = adapter(image)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("model forward must return a tensor")
    return logits


def _condition_key(node: str, transition_index: int) -> str:
    return f"{node}/transition_{int(transition_index)}"


def _restore_batch(
    adapter: ModelAdapter,
    image: torch.Tensor,
    *,
    root_module: nn.Module,
    root_shift_yx: tuple[int, int],
    restore_module: nn.Module,
    clean_activation: torch.Tensor,
    entries: Sequence[tuple[int, torch.Tensor]],
) -> tuple[list[int], torch.Tensor]:
    if not entries:
        raise ValueError("restore batch must contain at least one condition")
    indices = [int(index) for index, _ in entries]
    masks = torch.stack([mask for _, mask in entries], dim=0).to(image.device)
    batch = image.expand(len(entries), -1, -1, -1).contiguous()

    def root_transform(output: torch.Tensor) -> torch.Tensor:
        return spatial_roll_module_output(output, shift_yx=root_shift_yx)

    def restore_transform(output: torch.Tensor) -> torch.Tensor:
        return restore_clean_positions(output, clean_activation, masks)

    logits = _forward_with_transforms(
        adapter,
        batch,
        {
            root_module: root_transform,
            restore_module: restore_transform,
        },
    )
    return indices, logits


def run_network_alignment_forwards(
    adapter: ModelAdapter,
    image: torch.Tensor,
    *,
    root_node: str,
    restore_nodes: Sequence[str],
    masks: Mapping[str, Mapping[int, NodeRestoreMasks]],
    input_equivalent_shift_yx: tuple[int, int],
    logit_tolerance: float = 1.0e-6,
    clean_trace: ForwardTrace | None = None,
) -> NetworkAlignmentForwardSet:
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("image must have shape 1xCxHxW")
    if adapter.training:
        raise ValueError("network-alignment interventions require adapter.eval()")
    if root_node not in adapter.checkpoint_names:
        raise ValueError(f"unknown root checkpoint: {root_node}")
    if len(set(restore_nodes)) != len(tuple(restore_nodes)):
        raise ValueError("restore_nodes must be unique")
    for node in restore_nodes:
        if node not in adapter.checkpoint_names or node == root_node:
            raise ValueError(f"invalid restore checkpoint: {node}")

    hooks_before = _hook_count(adapter)
    with torch.inference_mode():
        native_trace = adapter.trace(image) if clean_trace is None else clean_trace
        clean_logits = native_trace.logits
        clean_activations = native_trace.activations
        root_clean = clean_activations[root_node]
        root_shift = input_equivalent_feature_shift(
            input_shape=tuple(int(value) for value in image.shape[-2:]),
            feature_shape=tuple(int(value) for value in root_clean.shape[-2:]),
            input_shift_yx=input_equivalent_shift_yx,
        )
        root_module = adapter.checkpoint_module(root_node)

        corrupted_root = spatial_roll_module_output(root_clean, shift_yx=root_shift)
        multiset_pass = channel_multiset_preserved(root_clean, corrupted_root)
        if not multiset_pass:
            raise RuntimeError("root spatial roll did not preserve channel multisets")

        corrupt_logits = _forward_with_transforms(
            adapter,
            image,
            {
                root_module: lambda output: spatial_roll_module_output(
                    output, shift_yx=root_shift
                )
            },
        )

        def full_root_restore(output: torch.Tensor) -> torch.Tensor:
            corrupted = spatial_roll_module_output(output, shift_yx=root_shift)
            full_mask = torch.ones(
                output.shape[-2:],
                dtype=torch.bool,
                device=output.device,
            )
            return restore_clean_positions(corrupted, output, full_mask)

        root_restored_logits = _forward_with_transforms(
            adapter,
            image,
            {root_module: full_root_restore},
        )
        root_error = float(
            torch.max(torch.abs(root_restored_logits - clean_logits)).item()
        )
        if not np.isfinite(root_error) or root_error > float(logit_tolerance):
            raise RuntimeError(
                "full root restore failed to reproduce clean logits: "
                f"{root_error} > {logit_tolerance}"
            )

        target_outputs: dict[str, np.ndarray] = {}
        control_outputs: dict[str, np.ndarray | None] = {}
        condition_order: dict[str, list[int]] = {}
        for node in restore_nodes:
            node_masks = masks.get(node, {})
            clean_activation = clean_activations[node]
            spatial_shape = tuple(int(value) for value in clean_activation.shape[-2:])
            ordered = sorted(node_masks.items())
            for transition_index, registered in ordered:
                if tuple(registered.target.shape) != spatial_shape:
                    raise ValueError(
                        f"target mask for {node}/transition_{transition_index} "
                        f"has shape {tuple(registered.target.shape)}, expected {spatial_shape}"
                    )
            target_entries = [
                (transition_index, registered.target)
                for transition_index, registered in ordered
                if registered.target_count > 0
            ]
            if not target_entries:
                condition_order[node] = []
                continue
            condition_order[node] = [index for index, _ in target_entries]
            indices, logits = _restore_batch(
                adapter,
                image,
                root_module=root_module,
                root_shift_yx=root_shift,
                restore_module=adapter.checkpoint_module(node),
                clean_activation=clean_activation,
                entries=target_entries,
            )
            for position, transition_index in enumerate(indices):
                key = _condition_key(node, transition_index)
                target_outputs[key] = _numpy(logits[position])
                control_outputs[key] = None

            control_entries = [
                (transition_index, registered.control)
                for transition_index, registered in ordered
                if registered.target_count > 0 and registered.control is not None
            ]
            if control_entries:
                control_indices, control_logits = _restore_batch(
                    adapter,
                    image,
                    root_module=root_module,
                    root_shift_yx=root_shift,
                    restore_module=adapter.checkpoint_module(node),
                    clean_activation=clean_activation,
                    entries=[
                        (index, control)
                        for index, control in control_entries
                        if control is not None
                    ],
                )
                for position, transition_index in enumerate(control_indices):
                    control_outputs[_condition_key(node, transition_index)] = _numpy(
                        control_logits[position]
                    )

    hooks_after = _hook_count(adapter)
    if hooks_after != hooks_before:
        raise RuntimeError("network-alignment hooks were not fully removed")
    arrays = [clean_logits, corrupt_logits, root_restored_logits]
    if not all(torch.isfinite(value).all() for value in arrays):
        raise RuntimeError("network-alignment forward produced non-finite logits")
    return NetworkAlignmentForwardSet(
        clean_logits=_numpy(clean_logits),
        corrupt_logits=_numpy(corrupt_logits),
        root_restored_logits=_numpy(root_restored_logits),
        target_restored_logits=target_outputs,
        control_restored_logits=control_outputs,
        audits={
            "root_shift_yx": list(root_shift),
            "channel_multiset_preserved": multiset_pass,
            "root_restore_max_abs_logit_error": root_error,
            "root_restore_pass": root_error <= float(logit_tolerance),
            "hook_count_before": hooks_before,
            "hook_count_after": hooks_after,
            "condition_order_by_node": condition_order,
        },
    )


__all__ = [
    "NetworkAlignmentForwardSet",
    "module_output_transforms",
    "run_network_alignment_forwards",
]
