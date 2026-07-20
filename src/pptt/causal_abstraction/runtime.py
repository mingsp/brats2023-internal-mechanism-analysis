from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from pptt.causal_abstraction.interventions import apply_flat_feature_edit
from pptt.causal_abstraction.states import relationship_states
from pptt.models.protocol import ModelAdapter
from pptt.observers.linear import LinearObserver


@dataclass(frozen=True)
class CounterfactualTrace:
    """Complete downstream state trajectory for one registered node edit."""

    node: str
    doses: tuple[float, ...]
    downstream_states: Mapping[float, Mapping[str, np.ndarray]]
    downstream_reliable: Mapping[float, Mapping[str, np.ndarray]]
    final_logits: Mapping[float, np.ndarray]
    audits: Mapping[str, Any]


def _hook_count(adapter: ModelAdapter) -> int:
    return sum(
        len(adapter.checkpoint_module(name)._forward_hooks)
        for name in adapter.checkpoint_names
    )


def _parameter_marker(adapter: ModelAdapter) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (name, int(parameter.data_ptr()), int(parameter._version))
        for name, parameter in adapter.named_parameters()
    )


def _model_device(adapter: ModelAdapter, fallback: torch.device) -> torch.device:
    parameter = next(adapter.parameters(), None)
    return fallback if parameter is None else parameter.device


def _validated_doses(values: Sequence[float]) -> tuple[float, ...]:
    doses = tuple(float(value) for value in values)
    if not doses or not all(np.isfinite(value) for value in doses):
        raise ValueError("doses must be nonempty and finite")
    if len(set(doses)) != len(doses):
        raise ValueError("doses must be unique")
    if 0.0 not in doses:
        raise ValueError("doses must include dose zero")
    return doses


def _ordered_observers(
    value: Mapping[int, LinearObserver] | Sequence[LinearObserver],
    *,
    node: str,
) -> tuple[LinearObserver, ...]:
    if isinstance(value, Mapping):
        observers = tuple(value[key] for key in sorted(value))
    else:
        observers = tuple(value)
    if len(observers) < 2 or not all(
        isinstance(observer, LinearObserver) for observer in observers
    ):
        raise ValueError(
            f"checkpoint {node} must provide at least two linear observers"
        )
    return observers


def _read_relationship_states(
    activation: torch.Tensor,
    observers: tuple[LinearObserver, ...],
    *,
    truth: np.ndarray,
    reliability_threshold: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    output_shape = tuple(int(value) for value in truth.shape)
    probabilities: list[torch.Tensor] = []
    for observer in observers:
        if observer.training:
            raise ValueError("causal-abstraction observers must be in eval mode")
        observer_parameter = next(observer.parameters())
        if observer_parameter.device != activation.device:
            raise ValueError("observer and activation must share a device")
        logits = observer(activation, output_size=output_shape)
        probabilities.append(torch.softmax(logits, dim=1))
    runs = torch.stack(probabilities, dim=0)
    restart_predictions = runs.argmax(dim=2)
    canonical = runs.mean(dim=0)
    predictions = canonical.argmax(dim=1)
    ordered_probabilities = torch.sort(runs, dim=2).values
    restart_margins = (
        ordered_probabilities[:, :, -1] - ordered_probabilities[:, :, -2]
    )
    agreement = torch.all(
        restart_predictions == restart_predictions[:1],
        dim=0,
    )
    reliable = agreement & (
        restart_margins.min(dim=0).values >= float(reliability_threshold)
    )

    predictions_np = predictions.detach().cpu().numpy().astype(np.uint8, copy=False)
    truth_batch = np.broadcast_to(truth, predictions_np.shape)
    states = relationship_states(
        truth_batch,
        predictions_np,
        num_classes=int(canonical.shape[1]),
    )
    return (
        states,
        reliable.detach().cpu().numpy().astype(bool, copy=False),
        int(canonical.shape[1]),
    )


def run_state_exchange(
    adapter: ModelAdapter,
    image: torch.Tensor,
    *,
    node: str,
    delta_h: torch.Tensor,
    doses: Sequence[float],
    observers: Mapping[
        str,
        Mapping[int, LinearObserver] | Sequence[LinearObserver],
    ],
    truth: np.ndarray,
    reliability_threshold: float,
    zero_dose_tolerance: float = 1.0e-5,
) -> CounterfactualTrace:
    """Inject one state edit and trace every registered downstream checkpoint."""

    if image.ndim != 4 or image.shape[0] != 1 or not image.is_floating_point():
        raise ValueError("image must be a floating tensor with shape 1xCxHxW")
    if adapter.training:
        raise ValueError("state exchange requires adapter.eval()")
    checkpoint_names = tuple(adapter.checkpoint_names)
    if node not in checkpoint_names:
        raise ValueError(f"unknown checkpoint: {node}")
    if len(set(checkpoint_names)) != len(checkpoint_names):
        raise ValueError("adapter checkpoint names must be unique")
    registered_doses = _validated_doses(doses)
    reference = np.asarray(truth)
    output_shape = tuple(int(value) for value in image.shape[-2:])
    if (
        reference.shape != output_shape
        or not np.issubdtype(reference.dtype, np.integer)
    ):
        raise ValueError("truth must be an integer array matching the image")
    if (
        not np.isfinite(reliability_threshold)
        or float(reliability_threshold) < 0
    ):
        raise ValueError("reliability_threshold must be finite and nonnegative")
    if not np.isfinite(zero_dose_tolerance) or float(zero_dose_tolerance) < 0:
        raise ValueError("zero_dose_tolerance must be finite and nonnegative")

    node_index = checkpoint_names.index(node)
    downstream_nodes = checkpoint_names[node_index:]
    observer_groups = {
        name: _ordered_observers(observers[name], node=name)
        for name in downstream_nodes
        if name in observers
    }
    missing_observers = tuple(
        name for name in downstream_nodes if name not in observer_groups
    )
    if missing_observers:
        raise ValueError(f"missing downstream observers: {missing_observers}")

    device = _model_device(adapter, image.device)
    image_device = image.to(device)
    hooks_before = _hook_count(adapter)
    parameter_marker_before = _parameter_marker(adapter)
    dose_tensor = torch.tensor(
        registered_doses,
        dtype=image_device.dtype,
        device=device,
    )

    with torch.inference_mode():
        clean_trace = adapter.trace(image_device)
        batch = image_device.expand(len(registered_doses), -1, -1, -1).contiguous()

        def transform(output: torch.Tensor) -> torch.Tensor:
            return apply_flat_feature_edit(output, delta_h, doses=dose_tensor)

        with adapter.transform_checkpoint_output(node, transform):
            edited_trace = adapter.trace(batch)

        upstream_max_error = 0.0
        for upstream in checkpoint_names[:node_index]:
            clean = clean_trace.activations[upstream].expand_as(
                edited_trace.activations[upstream]
            )
            error = torch.max(
                torch.abs(edited_trace.activations[upstream] - clean)
            )
            upstream_max_error = max(upstream_max_error, float(error.item()))

        states_by_node: dict[str, np.ndarray] = {}
        reliable_by_node: dict[str, np.ndarray] = {}
        class_counts: set[int] = set()
        for downstream in downstream_nodes:
            states, reliable, class_count = _read_relationship_states(
                edited_trace.activations[downstream],
                observer_groups[downstream],
                truth=reference,
                reliability_threshold=float(reliability_threshold),
            )
            states_by_node[downstream] = states
            reliable_by_node[downstream] = reliable
            class_counts.add(class_count)
        if len(class_counts) != 1:
            raise ValueError("downstream observers disagree on the class count")
        class_count = next(iter(class_counts))
        final_decisions = edited_trace.logits.argmax(dim=1)
        final_predictions = (
            final_decisions.detach().cpu().numpy().astype(np.uint8, copy=False)
        )
        final_states = relationship_states(
            np.broadcast_to(reference, final_predictions.shape),
            final_predictions,
            num_classes=class_count,
        )
        final_reliable = np.ones(final_states.shape, dtype=bool)

    hooks_after = _hook_count(adapter)
    parameter_marker_after = _parameter_marker(adapter)
    zero_index = registered_doses.index(0.0)
    zero_error = float(
        torch.max(
            torch.abs(edited_trace.logits[zero_index] - clean_trace.logits[0])
        ).item()
    )
    if not np.isfinite(zero_error) or zero_error > float(zero_dose_tolerance):
        raise RuntimeError(
            "dose zero failed to reproduce the clean output: "
            f"{zero_error} > {zero_dose_tolerance}"
        )
    if hooks_after != hooks_before:
        raise RuntimeError("state-exchange hooks were not fully removed")
    parameter_unchanged = parameter_marker_after == parameter_marker_before
    if not parameter_unchanged:
        raise RuntimeError("state exchange modified model parameter storage or version")
    if not torch.isfinite(edited_trace.logits).all():
        raise RuntimeError("state exchange produced non-finite final logits")

    downstream_states: OrderedDict[float, OrderedDict[str, np.ndarray]] = OrderedDict()
    downstream_reliable: OrderedDict[float, OrderedDict[str, np.ndarray]] = OrderedDict()
    final_logits: OrderedDict[float, np.ndarray] = OrderedDict()
    for dose_index, dose in enumerate(registered_doses):
        dose_states = OrderedDict(
            (name, states_by_node[name][dose_index]) for name in downstream_nodes
        )
        dose_states["Y"] = final_states[dose_index]
        downstream_states[dose] = dose_states
        dose_reliable = OrderedDict(
            (name, reliable_by_node[name][dose_index]) for name in downstream_nodes
        )
        dose_reliable["Y"] = final_reliable[dose_index]
        downstream_reliable[dose] = dose_reliable
        final_logits[dose] = (
            edited_trace.logits[dose_index]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

    return CounterfactualTrace(
        node=node,
        doses=registered_doses,
        downstream_states=downstream_states,
        downstream_reliable=downstream_reliable,
        final_logits=final_logits,
        audits={
            "downstream_path": [*downstream_nodes, "Y"],
            "upstream_nodes": list(checkpoint_names[:node_index]),
            "upstream_max_abs_error": upstream_max_error,
            "dose_zero_max_abs_logit_error": zero_error,
            "parameter_storage_and_version_unchanged": parameter_unchanged,
            "hook_count_before": hooks_before,
            "hook_count_after": hooks_after,
            "observer_restart_count_by_node": {
                name: len(observer_groups[name]) for name in downstream_nodes
            },
        },
    )


__all__ = ["CounterfactualTrace", "run_state_exchange"]
