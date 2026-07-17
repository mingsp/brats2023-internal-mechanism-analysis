import numpy as np
import pytest

from pptt.transitions.theory import (
    additive_event_from_tensor,
    adjacent_pair_counts,
    binary_three_node_counterexample,
    endpoint_hidden_dimension,
    persistent_correction_count,
)
from pptt.transitions.tensors import build_transition_tensor


def test_four_class_endpoint_hidden_dimension_is_36():
    assert endpoint_hidden_dimension(4, truth_class_count=4) == 36


def test_endpoint_hidden_dimension_validates_class_counts():
    with pytest.raises(ValueError, match="num_classes"):
        endpoint_hidden_dimension(1)
    with pytest.raises(ValueError, match="truth_class_count"):
        endpoint_hidden_dimension(4, truth_class_count=0)


def test_equal_pairwise_transitions_can_hide_different_persistence():
    process_a, process_b = binary_three_node_counterexample()

    pair_counts_a = adjacent_pair_counts(process_a, num_classes=2)
    pair_counts_b = adjacent_pair_counts(process_b, num_classes=2)
    assert len(pair_counts_a) == 2
    for left, right in zip(pair_counts_a, pair_counts_b, strict=True):
        np.testing.assert_array_equal(left, right)

    assert persistent_correction_count(process_a, truth_class=1) == 1
    assert persistent_correction_count(process_b, truth_class=1) == 0


def test_transition_tensor_recovers_every_additive_event_functional():
    truth = np.asarray([0, 0, 1, 1, 2, 2, 2], dtype=np.int64)
    before = np.asarray([0, 1, 0, 1, 0, 1, 2], dtype=np.int64)
    after = np.asarray([0, 2, 1, 0, 2, 1, 0], dtype=np.int64)
    tensor = build_transition_tensor(truth, before, after, num_classes=3)
    weights = np.arange(27, dtype=np.float64).reshape(3, 3, 3) / 7.0

    direct = float(weights[truth, before, after].sum())
    reconstructed = additive_event_from_tensor(tensor, weights)

    assert reconstructed == pytest.approx(direct, abs=1.0e-12, rel=0.0)


def test_additive_event_functional_rejects_misaligned_or_nonfinite_weights():
    tensor = np.ones((2, 2, 2), dtype=np.int64)
    with pytest.raises(ValueError, match="aligned"):
        additive_event_from_tensor(tensor, np.ones((2, 2)))
    weights = np.ones((2, 2, 2), dtype=np.float64)
    weights[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        additive_event_from_tensor(tensor, weights)


def test_adjacent_pair_counts_validates_trajectory_labels():
    with pytest.raises(ValueError, match="two-dimensional"):
        adjacent_pair_counts(np.zeros((2, 2, 2), dtype=np.int64), num_classes=2)
    with pytest.raises(ValueError, match="inside"):
        adjacent_pair_counts(
            np.asarray([[0, 2], [1, 0]], dtype=np.int64),
            num_classes=2,
        )
