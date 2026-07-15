import numpy as np
import pytest

from pptt.observers.controls import (
    class_reliability,
    reliable_transition_mask,
    select_reliability_threshold,
)
from pptt.transitions.tensors import build_transition_tensor_views


def _probability_runs() -> np.ndarray:
    probabilities = np.array(
        [
            [
                [[0.80, 0.48, 0.80, 0.54], [0.10, 0.44, 0.10, 0.35]],
                [[0.75, 0.20, 0.55, 0.55], [0.15, 0.70, 0.35, 0.35]],
            ],
            [
                [[0.78, 0.47, 0.79, 0.53], [0.12, 0.45, 0.11, 0.36]],
                [[0.74, 0.22, 0.54, 0.20], [0.16, 0.68, 0.36, 0.70]],
            ],
            [
                [[0.79, 0.49, 0.81, 0.55], [0.11, 0.43, 0.09, 0.34]],
                [[0.76, 0.21, 0.56, 0.56], [0.14, 0.69, 0.34, 0.34]],
            ],
        ],
        dtype=np.float64,
    )
    background = 1.0 - probabilities.sum(axis=2, keepdims=True)
    return np.concatenate([probabilities, background], axis=2)


def test_reliable_transition_requires_margin_and_seed_pair_agreement():
    probabilities = _probability_runs()
    mask = reliable_transition_mask(probabilities, threshold=0.10, class_axis=2)

    assert mask.shape == (1, 4)
    assert mask.dtype == np.bool_
    assert mask[0, 0]
    assert not mask[0, 1]
    assert not mask[0, 3]


def test_threshold_selection_uses_smallest_grid_value_meeting_consistency():
    probabilities = _probability_runs()
    selection = select_reliability_threshold(
        probabilities,
        grid=(0.0, 0.05, 0.10, 0.20),
        minimum_consistency=0.90,
        class_axis=2,
    )

    assert selection.threshold in (0.0, 0.05, 0.10, 0.20)
    assert selection.consistency >= 0.90
    assert 0 < selection.retention <= 1


def test_threshold_selection_fails_when_no_grid_value_is_eligible():
    probabilities = _probability_runs()
    with pytest.raises(ValueError, match="No reliability threshold"):
        select_reliability_threshold(
            probabilities,
            grid=(0.9, 0.95),
            minimum_consistency=0.90,
            class_axis=2,
        )


def test_full_and_reliable_transition_tensors_remain_separate():
    y = np.array([0, 1, 2, 2], dtype=np.int64)
    z0 = np.array([0, 0, 2, 1], dtype=np.int64)
    z1 = np.array([0, 1, 2, 2], dtype=np.int64)
    reliable = np.array([True, False, True, False])
    views = build_transition_tensor_views(y, z0, z1, reliable, num_classes=3)

    assert views.full.sum() == 4
    assert views.reliable.sum() == 2
    assert views.uncertain_pixel_count == 2


def test_class_reliability_applies_fixed_retention_gate():
    truth = np.array([0, 1, 2, 2], dtype=np.int64)
    reliable = np.array([[True, False, True, False]])
    result = class_reliability(
        reliable,
        truth,
        num_classes=3,
        minimum_retention=0.50,
    )

    np.testing.assert_allclose(result.retention, [[1.0, 0.0, 0.5]])
    np.testing.assert_array_equal(result.available, [[True, False, True]])
