import numpy as np

from pptt.interventions.matching import match_pixels_without_replacement


def test_matching_is_calipered_and_without_replacement():
    result = match_pixels_without_replacement(
        truth_class=np.array([1, 1, 1, 1, 2, 2]),
        boundary_bin=np.array([0, 0, 0, 0, 1, 1]),
        confidence_bin=np.array([1, 2, 1, 3, 1, 1]),
        area_bin=np.array([0, 0, 0, 0, 1, 1]),
        norm_bin=np.array([1, 2, 1, 3, 1, 1]),
        target_mask=np.array([True, True, False, False, True, False]),
        control_mask=np.array([False, False, True, True, False, True]),
        boundary_value=np.array([0.1, 0.2, 0.11, 0.21, 1.1, 1.09]),
        confidence_value=np.array([0.4, 0.8, 0.41, 0.81, 0.5, 0.51]),
        area_value=np.array([10, 10, 10, 10, 20, 20]),
        norm_value=np.array([2.0, 4.0, 2.1, 4.1, 3.0, 3.1]),
        confidence_caliper=1,
        norm_caliper=1,
    )

    assert result.matched_count == 3
    assert len(np.unique(result.control_indices)) == 3
    assert result.unmatched_target_count == 0


def test_matching_uses_continuous_nearest_neighbor_inside_calipers():
    result = match_pixels_without_replacement(
        truth_class=np.ones(3, dtype=np.uint8),
        boundary_bin=np.zeros(3, dtype=np.int16),
        confidence_bin=np.ones(3, dtype=np.int16),
        area_bin=np.zeros(3, dtype=np.int16),
        norm_bin=np.ones(3, dtype=np.int16),
        target_mask=np.array([True, False, False]),
        control_mask=np.array([False, True, True]),
        boundary_value=np.array([1.0, 3.0, 1.1]),
        confidence_value=np.array([0.5, 0.9, 0.51]),
        area_value=np.array([10.0, 30.0, 10.1]),
        norm_value=np.array([2.0, 5.0, 2.1]),
    )

    np.testing.assert_array_equal(result.control_indices, [2])
