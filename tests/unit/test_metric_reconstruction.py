import numpy as np

from pptt.transitions.metrics import (
    confusion_from_labels,
    metric_delta_from_transition,
    metrics_from_confusion,
)
from pptt.transitions.tensors import build_transition_tensor, confusions_from_transition


def test_all_confusion_metrics_are_exactly_reconstructed_from_transition():
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3, 1, 2, 0], dtype=np.int64)
    z0 = np.array([0, 1, 1, 0, 2, 3, 3, 0, 2, 1, 0, 0], dtype=np.int64)
    z1 = np.array([0, 0, 1, 1, 2, 2, 3, 3, 2, 3, 2, 1], dtype=np.int64)
    tensor = build_transition_tensor(y, z0, z1, num_classes=4)
    reconstructed0, reconstructed1 = confusions_from_transition(tensor)
    direct0 = confusion_from_labels(y, z0, num_classes=4)
    direct1 = confusion_from_labels(y, z1, num_classes=4)

    np.testing.assert_array_equal(reconstructed0, direct0)
    np.testing.assert_array_equal(reconstructed1, direct1)
    metrics0 = metrics_from_confusion(direct0)
    metrics1 = metrics_from_confusion(direct1)
    delta = metric_delta_from_transition(tensor)
    for metric_name in ("dice", "iou", "precision", "recall", "specificity"):
        expected = getattr(metrics1, metric_name) - getattr(metrics0, metric_name)
        np.testing.assert_allclose(
            getattr(delta, metric_name),
            expected,
            rtol=0.0,
            atol=1e-12,
        )


def test_binary_counts_include_false_positives_in_dice_denominator():
    confusion = np.array(
        [[8, 2], [1, 9]],
        dtype=np.int64,
    )
    metrics = metrics_from_confusion(confusion)

    assert metrics.dice[1] == 18 / (18 + 2 + 1)
    assert metrics.iou[1] == 9 / (9 + 2 + 1)
    assert metrics.precision[1] == 9 / (9 + 2)
    assert metrics.recall[1] == 9 / (9 + 1)
    assert metrics.specificity[1] == 8 / (8 + 2)


def test_zero_denominator_policy_is_explicit_and_finite():
    confusion = np.zeros((3, 3), dtype=np.int64)
    metrics = metrics_from_confusion(confusion, zero_division=0.0)

    for values in metrics.as_dict().values():
        np.testing.assert_array_equal(values, np.zeros(3))
