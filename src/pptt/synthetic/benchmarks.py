from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from pptt.comparators.cam import (
    linear_gradcam,
    linear_layercam,
    top_fraction_iou,
)
from pptt.comparators.node_dice import (
    adjacent_mask_iou,
    node_dice_path,
    node_dice_stage_accuracy,
)
from pptt.lineage.depths import lineage_depths
from pptt.synthetic.known_process import (
    KnownProcessBatch,
    decode_known_process_features,
)
from pptt.transitions.fields import classify_transition_events
from pptt.transitions.metrics import (
    confusion_from_labels,
    metric_delta_from_transition,
    metrics_from_confusion,
)
from pptt.transitions.tensors import build_transition_tensor


@dataclass(frozen=True)
class KnownProcessResult:
    stage_accuracy: float
    direction_macro_f1: float
    region_iou: float
    origin_depth_accuracy: float
    branch_attribution_accuracy: float
    max_metric_reconstruction_error: float
    node_dice_stage_accuracy: float
    adjacent_mask_iou: float
    gradcam_region_iou: float
    layercam_region_iou: float

    def to_dict(self) -> dict:
        return asdict(self)

    def to_report(self) -> dict:
        return {
            "pptt": {
                "stage_accuracy": self.stage_accuracy,
                "direction_macro_f1": self.direction_macro_f1,
                "region_iou": self.region_iou,
                "origin_depth_accuracy": self.origin_depth_accuracy,
                "branch_attribution_accuracy": self.branch_attribution_accuracy,
                "max_metric_reconstruction_error": self.max_metric_reconstruction_error,
            },
            "comparators": {
                "node_dice": {"stage_accuracy": self.node_dice_stage_accuracy},
                "adjacent_hard_mask_iou": {"mean_iou": self.adjacent_mask_iou},
                "gradcam": {"region_iou": self.gradcam_region_iou},
                "layercam": {"region_iou": self.layercam_region_iou},
            },
        }


def _macro_f1(expected: np.ndarray, predicted: np.ndarray) -> float:
    labels = np.unique(expected)
    scores = []
    for label in labels:
        true_positive = np.count_nonzero((expected == label) & (predicted == label))
        false_positive = np.count_nonzero((expected != label) & (predicted == label))
        false_negative = np.count_nonzero((expected == label) & (predicted != label))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores))


def _changed_region_iou(expected: np.ndarray, predicted: np.ndarray) -> float:
    scores = []
    active = np.any(expected, axis=(0, 2, 3))
    for transition in np.flatnonzero(active):
        for sample in range(expected.shape[0]):
            union = np.logical_or(
                expected[sample, transition],
                predicted[sample, transition],
            ).sum()
            scores.append(
                float(
                    np.logical_and(
                        expected[sample, transition],
                        predicted[sample, transition],
                    ).sum()
                    / union
                )
                if union
                else 1.0
            )
    return float(np.mean(scores))


def _cam_region_scores(
    batch: KnownProcessBatch,
    decoded: np.ndarray,
) -> tuple[float, float]:
    gradcam_scores: list[float] = []
    layercam_scores: list[float] = []
    for transition in batch.active_transition_indices:
        destination_stage = transition + 1
        for sample in range(batch.truth.shape[0]):
            region = batch.expected_changed_masks[sample, transition]
            destination_labels = decoded[sample, destination_stage][region]
            target_class = int(np.bincount(destination_labels).argmax())
            feature = batch.features[sample, destination_stage]
            readout = batch.readout_weights[destination_stage]
            gradcam_scores.append(
                top_fraction_iou(
                    linear_gradcam(feature, readout, target_class=target_class),
                    region,
                )
            )
            layercam_scores.append(
                top_fraction_iou(
                    linear_layercam(feature, readout, target_class=target_class),
                    region,
                )
            )
    return float(np.mean(gradcam_scores)), float(np.mean(layercam_scores))


def evaluate_known_process(batch: KnownProcessBatch) -> KnownProcessResult:
    probabilities = decode_known_process_features(batch)
    decoded = probabilities.argmax(axis=2)
    expected_changed = batch.expected_changed_masks
    predicted_changed = decoded[:, 1:] != decoded[:, :-1]
    expected_active = np.any(expected_changed, axis=(0, 2, 3))
    predicted_active = np.any(predicted_changed, axis=(0, 2, 3))
    stage_accuracy = float(np.mean(expected_active == predicted_active))

    expected_events = []
    predicted_events = []
    for transition in range(decoded.shape[1] - 1):
        expected_event = classify_transition_events(
            batch.truth,
            batch.states[:, transition],
            batch.states[:, transition + 1],
            num_classes=batch.num_classes,
        )
        predicted_event = classify_transition_events(
            batch.truth,
            decoded[:, transition],
            decoded[:, transition + 1],
            num_classes=batch.num_classes,
        )
        changed = expected_changed[:, transition]
        if changed.any():
            expected_events.append(expected_event[changed])
            predicted_events.append(predicted_event[changed])
    direction_macro_f1 = _macro_f1(
        np.concatenate(expected_events),
        np.concatenate(predicted_events),
    )
    region_iou = _changed_region_iou(expected_changed, predicted_changed)

    expected_depths = lineage_depths(
        np.moveaxis(batch.states, 1, 0),
        batch.truth,
    )
    predicted_depths = lineage_depths(
        np.moveaxis(decoded, 1, 0),
        batch.truth,
    )
    final_is_correct = batch.states[:, -1] == batch.truth
    expected_origin = np.where(
        final_is_correct,
        expected_depths.correct_state,
        expected_depths.terminal_error_origin,
    )
    predicted_origin = np.where(
        decoded[:, -1] == batch.truth,
        predicted_depths.correct_state,
        predicted_depths.terminal_error_origin,
    )
    origin_depth_accuracy = float(np.mean(expected_origin == predicted_origin))
    natural_branch_output = batch.states[:, 6]
    branch_scores = np.sum(
        batch.branch_ablation_states != natural_branch_output[None],
        axis=(2, 3),
    )
    predicted_branch = np.argmax(branch_scores, axis=0)
    branch_attribution_accuracy = float(
        np.mean(predicted_branch == batch.unique_effective_branch_index)
    )

    maximum_metric_error = 0.0
    for transition in range(decoded.shape[1] - 1):
        tensor = build_transition_tensor(
            batch.truth,
            decoded[:, transition],
            decoded[:, transition + 1],
            batch.num_classes,
        )
        delta = metric_delta_from_transition(tensor)
        before = metrics_from_confusion(
            confusion_from_labels(
                batch.truth,
                decoded[:, transition],
                batch.num_classes,
            )
        )
        after = metrics_from_confusion(
            confusion_from_labels(
                batch.truth,
                decoded[:, transition + 1],
                batch.num_classes,
            )
        )
        for name in delta.as_dict():
            direct = getattr(after, name) - getattr(before, name)
            maximum_metric_error = max(
                maximum_metric_error,
                float(np.max(np.abs(getattr(delta, name) - direct))),
            )

    flattened_states = np.moveaxis(decoded, 1, 0)
    dice_path = node_dice_path(
        flattened_states,
        batch.truth,
        num_classes=batch.num_classes,
    )
    dice_stage_accuracy = node_dice_stage_accuracy(dice_path, expected_active)
    adjacent_iou = adjacent_mask_iou(
        flattened_states,
        num_classes=batch.num_classes,
    )
    gradcam_iou, layercam_iou = _cam_region_scores(batch, decoded)
    return KnownProcessResult(
        stage_accuracy=stage_accuracy,
        direction_macro_f1=direction_macro_f1,
        region_iou=region_iou,
        origin_depth_accuracy=origin_depth_accuracy,
        branch_attribution_accuracy=branch_attribution_accuracy,
        max_metric_reconstruction_error=maximum_metric_error,
        node_dice_stage_accuracy=dice_stage_accuracy,
        adjacent_mask_iou=adjacent_iou,
        gradcam_region_iou=gradcam_iou,
        layercam_region_iou=layercam_iou,
    )
