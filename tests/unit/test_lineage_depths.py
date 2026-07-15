import numpy as np
import pytest

from pptt.lineage.depths import (
    correct_state_depth,
    final_state_depth,
    lineage_depths,
    terminal_error_origin_depth,
)


def test_lineage_depths_use_complete_pixel_history_and_one_based_stages():
    states = np.array(
        [
            [0, 2, 0, 3],
            [1, 2, 1, 3],
            [1, 0, 0, 3],
            [1, 2, 1, 3],
            [1, 2, 0, 3],
        ],
        dtype=np.int64,
    )
    truth = np.array([1, 1, 0, 3], dtype=np.int64)
    depths = lineage_depths(states, truth)

    np.testing.assert_array_equal(depths.final_state, [2, 4, 5, 1])
    np.testing.assert_array_equal(depths.correct_state, [2, -1, 5, 1])
    np.testing.assert_array_equal(depths.terminal_error_origin, [-1, 4, -1, -1])


def test_individual_depth_functions_match_combined_result():
    states = np.array([[0, 1], [1, 0], [1, 0]], dtype=np.int64)
    truth = np.array([1, 1], dtype=np.int64)

    np.testing.assert_array_equal(final_state_depth(states), [2, 2])
    np.testing.assert_array_equal(correct_state_depth(states, truth), [2, -1])
    np.testing.assert_array_equal(
        terminal_error_origin_depth(states, truth),
        [-1, 2],
    )


def test_depths_reject_too_short_or_shape_mismatched_sequences():
    with pytest.raises(ValueError, match="at least two stages"):
        final_state_depth(np.zeros((1, 4), dtype=np.int64))
    with pytest.raises(ValueError, match="spatial shape"):
        correct_state_depth(
            np.zeros((2, 2, 2), dtype=np.int64),
            np.zeros((4,), dtype=np.int64),
        )
