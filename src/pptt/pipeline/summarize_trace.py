from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pptt.io.artifacts import CaseTrace
from pptt.lineage.depths import lineage_depths
from pptt.transitions.fields import TransitionEvent, classify_transition_events
from pptt.transitions.metrics import (
    confusion_from_labels,
    metric_delta_from_transition,
    metrics_from_confusion,
)
from pptt.transitions.tensors import (
    build_transition_tensor_views,
    confusions_from_transition,
    state_flows_from_transition,
)


@dataclass(frozen=True)
class CaseTraceSummary:
    node_rows: list[dict]
    confusion_rows: list[dict]
    transition_rows: list[dict]
    flow_rows: list[dict]
    metric_delta_rows: list[dict]
    tensor_rows: list[dict]
    lineage_rows: list[dict]
    reconstruction_rows: list[dict]


def _class_masks(truth: np.ndarray, num_classes: int) -> list[np.ndarray]:
    return [truth == class_index for class_index in range(num_classes)]


def _mean_present(values: np.ndarray, present: list[int]) -> float:
    if not present:
        return float("nan")
    return float(np.mean(values[present]))


def _event_count(events: np.ndarray, event: TransitionEvent) -> int:
    return int(np.count_nonzero(events == int(event)))


def summarize_case_trace(
    trace: CaseTrace,
    *,
    nodes: tuple[str, ...],
    num_classes: int,
) -> CaseTraceSummary:
    if trace.truth is None:
        raise ValueError("Label-aware case summary requires truth")
    if trace.states.shape[0] != len(nodes):
        raise ValueError("Node names do not match the saved state path")
    if trace.reliable.shape[0] != len(nodes) - 1:
        raise ValueError("Reliability path does not match adjacent nodes")
    states = trace.states.astype(np.int64, copy=False)
    truth = trace.truth.astype(np.int64, copy=False)
    class_masks = _class_masks(truth, num_classes)
    present_tumor = [
        class_index
        for class_index in range(1, num_classes)
        if class_masks[class_index].any()
    ]
    pixel_count = int(truth.size)
    node_rows: list[dict] = []
    confusion_rows: list[dict] = []
    transition_rows: list[dict] = []
    flow_rows: list[dict] = []
    metric_delta_rows: list[dict] = []
    tensor_rows: list[dict] = []
    reconstruction_rows: list[dict] = []

    node_metrics = []
    for stage_index, node in enumerate(nodes):
        state = states[stage_index]
        confusion = confusion_from_labels(truth, state, num_classes)
        metrics = metrics_from_confusion(confusion)
        node_metrics.append(metrics)
        for truth_class in range(num_classes):
            for predicted_class in range(num_classes):
                confusion_rows.append(
                    {
                        "stage_index": stage_index,
                        "node": node,
                        "truth_class": truth_class,
                        "predicted_class": predicted_class,
                        "count": int(confusion[truth_class, predicted_class]),
                    }
                )
        persistent_correct = np.all(
            states[stage_index:] == truth[np.newaxis, ...],
            axis=0,
        )
        for class_index in range(num_classes):
            selected = class_masks[class_index]
            node_rows.append(
                {
                    "stage_index": stage_index,
                    "node": node,
                    "class_index": class_index,
                    "class_available": bool(selected.any()),
                    "truth_pixel_count": int(selected.sum()),
                    "predicted_pixel_count": int(np.count_nonzero(state == class_index)),
                    "dice": float(metrics.dice[class_index]),
                    "iou": float(metrics.iou[class_index]),
                    "precision": float(metrics.precision[class_index]),
                    "recall": float(metrics.recall[class_index]),
                    "specificity": float(metrics.specificity[class_index]),
                    "correct_occupancy_rate": (
                        float(np.mean(state[selected] == truth[selected]))
                        if selected.any()
                        else float("nan")
                    ),
                    "persistent_correct_rate": (
                        float(persistent_correct[selected].mean())
                        if selected.any()
                        else float("nan")
                    ),
                }
            )
        node_rows.append(
            {
                "stage_index": stage_index,
                "node": node,
                "class_index": -1,
                "class_available": bool(present_tumor),
                "truth_pixel_count": int(sum(class_masks[c].sum() for c in present_tumor)),
                "predicted_pixel_count": int(np.count_nonzero(state != 0)),
                "dice": _mean_present(metrics.dice, present_tumor),
                "iou": _mean_present(metrics.iou, present_tumor),
                "precision": _mean_present(metrics.precision, present_tumor),
                "recall": _mean_present(metrics.recall, present_tumor),
                "specificity": _mean_present(metrics.specificity, present_tumor),
                "correct_occupancy_rate": float(np.mean(state == truth)),
                "persistent_correct_rate": float(persistent_correct.mean()),
            }
        )

    macro_dice = np.asarray(
        [_mean_present(metrics.dice, present_tumor) for metrics in node_metrics]
    )
    macro_delta = np.diff(macro_dice)
    macro_acceleration = np.full(len(nodes) - 1, np.nan, dtype=np.float64)
    if macro_delta.size > 1:
        macro_acceleration[1:] = np.diff(macro_delta)

    for transition_index, (left, right) in enumerate(
        zip(nodes[:-1], nodes[1:], strict=True)
    ):
        views = build_transition_tensor_views(
            truth,
            states[transition_index],
            states[transition_index + 1],
            trace.reliable[transition_index],
            num_classes,
        )
        full_events = classify_transition_events(
            truth,
            states[transition_index],
            states[transition_index + 1],
            num_classes=num_classes,
        )
        reliable_events = classify_transition_events(
            truth,
            states[transition_index],
            states[transition_index + 1],
            num_classes=num_classes,
            reliable=trace.reliable[transition_index],
        )
        persistent_correct_after = np.all(
            states[transition_index + 1 :] == truth[np.newaxis, ...],
            axis=0,
        )
        persistent_wrong_after = np.all(
            states[transition_index + 1 :] != truth[np.newaxis, ...],
            axis=0,
        )
        persistent_correction = (
            (states[transition_index] != truth)
            & (states[transition_index + 1] == truth)
            & persistent_correct_after
        )
        persistent_destruction = (
            (states[transition_index] == truth)
            & (states[transition_index + 1] != truth)
            & persistent_wrong_after
        )
        correction_count = _event_count(full_events, TransitionEvent.CORRECTION)
        destruction_count = _event_count(full_events, TransitionEvent.DESTRUCTION)
        wrong_reencoding_count = _event_count(
            full_events,
            TransitionEvent.WRONG_REENCODING,
        )
        persistent_correction_count = int(persistent_correction.sum())
        persistent_destruction_count = int(persistent_destruction.sum())
        delta_metrics = metric_delta_from_transition(views.full)
        for class_index in range(num_classes):
            metric_delta_rows.append(
                {
                    "transition_index": transition_index,
                    "transition": f"{left}->{right}",
                    "class_index": class_index,
                    "class_available": bool(class_masks[class_index].any()),
                    **{
                        f"delta_{name}": float(values[class_index])
                        for name, values in delta_metrics.as_dict().items()
                    },
                }
            )
        metric_delta_rows.append(
            {
                "transition_index": transition_index,
                "transition": f"{left}->{right}",
                "class_index": -1,
                "class_available": bool(present_tumor),
                **{
                    f"delta_{name}": _mean_present(values, present_tumor)
                    for name, values in delta_metrics.as_dict().items()
                },
            }
        )
        transition_rows.append(
            {
                "transition_index": transition_index,
                "transition": f"{left}->{right}",
                "pixel_count": pixel_count,
                "reliable_pixel_count": int(trace.reliable[transition_index].sum()),
                "uncertain_pixel_count": views.uncertain_pixel_count,
                "reliable_fraction": float(trace.reliable[transition_index].mean()),
                "correction_count": correction_count,
                "correction_rate": correction_count / pixel_count,
                "destruction_count": destruction_count,
                "destruction_rate": destruction_count / pixel_count,
                "wrong_reencoding_count": wrong_reencoding_count,
                "wrong_reencoding_rate": wrong_reencoding_count / pixel_count,
                "persistent_correction_count": persistent_correction_count,
                "persistent_correction_rate": persistent_correction_count / pixel_count,
                "persistent_destruction_count": persistent_destruction_count,
                "persistent_destruction_rate": persistent_destruction_count / pixel_count,
                "persistent_net_count": (
                    persistent_correction_count - persistent_destruction_count
                ),
                "persistent_net_rate": (
                    persistent_correction_count - persistent_destruction_count
                )
                / pixel_count,
                "macro_tumor_delta_dice": _mean_present(
                    delta_metrics.dice,
                    present_tumor,
                ),
                "macro_tumor_dice_acceleration": float(
                    macro_acceleration[transition_index]
                ),
                "reliable_correction_count": _event_count(
                    reliable_events,
                    TransitionEvent.CORRECTION,
                ),
                "reliable_destruction_count": _event_count(
                    reliable_events,
                    TransitionEvent.DESTRUCTION,
                ),
                "reliable_wrong_reencoding_count": _event_count(
                    reliable_events,
                    TransitionEvent.WRONG_REENCODING,
                ),
            }
        )
        for source, tensor in (("full", views.full), ("reliable", views.reliable)):
            flows = state_flows_from_transition(tensor)
            if not np.array_equal(
                flows.occupancy_delta,
                flows.inflow - flows.outflow,
            ):
                raise RuntimeError("Class-state flow conservation failed")
            for truth_class in range(num_classes):
                for state_class in range(num_classes):
                    flow_rows.append(
                        {
                            "transition_index": transition_index,
                            "transition": f"{left}->{right}",
                            "scope": source,
                            "truth_class": truth_class,
                            "state_class": state_class,
                            "inflow": int(flows.inflow[truth_class, state_class]),
                            "outflow": int(flows.outflow[truth_class, state_class]),
                            "occupancy_delta": int(
                                flows.occupancy_delta[truth_class, state_class]
                            ),
                        }
                    )
            for truth_class in range(num_classes):
                for state_from in range(num_classes):
                    for state_to in range(num_classes):
                        tensor_rows.append(
                            {
                                "transition_index": transition_index,
                                "transition": f"{left}->{right}",
                                "scope": source,
                                "truth_class": truth_class,
                                "state_from": state_from,
                                "state_to": state_to,
                                "count": int(
                                    tensor[truth_class, state_from, state_to]
                                ),
                            }
                        )
        reconstructed_left, reconstructed_right = confusions_from_transition(
            views.full
        )
        direct_left = confusion_from_labels(
            truth,
            states[transition_index],
            num_classes,
        )
        direct_right = confusion_from_labels(
            truth,
            states[transition_index + 1],
            num_classes,
        )
        left_metrics = metrics_from_confusion(direct_left)
        right_metrics = metrics_from_confusion(direct_right)
        metric_error = max(
            float(np.max(np.abs(getattr(left_metrics, name) - getattr(
                metrics_from_confusion(reconstructed_left), name
            ))))
            for name in left_metrics.as_dict()
        )
        metric_error = max(
            metric_error,
            max(
                float(np.max(np.abs(getattr(right_metrics, name) - getattr(
                    metrics_from_confusion(reconstructed_right), name
                ))))
                for name in right_metrics.as_dict()
            ),
        )
        reconstruction_rows.append(
            {
                "transition_index": transition_index,
                "transition": f"{left}->{right}",
                "left_confusion_max_abs_error": int(
                    np.max(np.abs(reconstructed_left - direct_left))
                ),
                "right_confusion_max_abs_error": int(
                    np.max(np.abs(reconstructed_right - direct_right))
                ),
                "metric_max_abs_error": metric_error,
                "telescoping_macro_dice_error": float(
                    abs(macro_delta[: transition_index + 1].sum()
                    - (macro_dice[transition_index + 1] - macro_dice[0]))
                ),
            }
        )

    depths = lineage_depths(states, truth)
    lineage_rows: list[dict] = []
    for name, values in (
        ("final_state_depth", depths.final_state),
        ("correct_state_depth", depths.correct_state),
        ("terminal_error_origin_depth", depths.terminal_error_origin),
    ):
        for class_index in range(num_classes):
            selected_values = values[class_masks[class_index]]
            if selected_values.size == 0:
                continue
            for depth in range(-1, len(nodes) + 1):
                count = int(np.count_nonzero(selected_values == depth))
                if count:
                    lineage_rows.append(
                        {
                            "kind": name,
                            "class_index": class_index,
                            "depth": depth,
                            "count": count,
                            "rate_within_truth_class": count / selected_values.size,
                        }
                    )

    return CaseTraceSummary(
        node_rows=node_rows,
        confusion_rows=confusion_rows,
        transition_rows=transition_rows,
        flow_rows=flow_rows,
        metric_delta_rows=metric_delta_rows,
        tensor_rows=tensor_rows,
        lineage_rows=lineage_rows,
        reconstruction_rows=reconstruction_rows,
    )
