import numpy as np
import pytest

from pptt.lineage.direct_paths import (
    class_balanced_direct_net_recovery,
    output_anchored_direct_path_cohorts,
)


def _example():
    truth = np.asarray([[1, 1, 2, 2, 0]], dtype=np.uint8)
    states = np.asarray(
        [
            [[0, 1, 2, 2, 0]],
            [[0, 1, 2, 2, 0]],
            [[0, 1, 2, 2, 0]],
            [[1, 0, 2, 1, 0]],
            [[1, 0, 2, 1, 0]],
            [[1, 0, 2, 1, 0]],
        ],
        dtype=np.uint8,
    )
    reliable = np.ones((5, 1, 5), dtype=bool)
    final_state = np.asarray([[1, 0, 2, 1, 0]], dtype=np.uint8)
    return states, truth, reliable, final_state


def test_direct_path_cohorts_are_output_anchored_and_reliable():
    states, truth, reliable, final_state = _example()

    cohorts = output_anchored_direct_path_cohorts(
        states,
        truth,
        reliable,
        final_state,
        source_index=1,
        receiver_index=3,
        truth_classes=(1, 2),
    )

    np.testing.assert_array_equal(
        cohorts.correction,
        np.asarray([[True, False, False, False, False]]),
    )
    np.testing.assert_array_equal(
        cohorts.damage,
        np.asarray([[False, True, False, True, False]]),
    )
    assert not np.any(cohorts.correction & cohorts.damage)
    assert np.all(final_state[cohorts.correction] == truth[cohorts.correction])
    assert np.all(final_state[cohorts.damage] != truth[cohorts.damage])


def test_direct_path_cohorts_remove_pixels_with_unreliable_suffix():
    states, truth, reliable, final_state = _example()
    reliable[4, 0, 0] = False

    cohorts = output_anchored_direct_path_cohorts(
        states,
        truth,
        reliable,
        final_state,
        source_index=1,
        receiver_index=3,
        truth_classes=(1, 2),
    )

    assert not cohorts.correction[0, 0]


def test_class_balanced_direct_net_recovery_uses_each_present_class_once():
    states, truth, reliable, final_state = _example()
    cohorts = output_anchored_direct_path_cohorts(
        states,
        truth,
        reliable,
        final_state,
        source_index=1,
        receiver_index=3,
        truth_classes=(1, 2),
    )

    value = class_balanced_direct_net_recovery(
        cohorts,
        truth,
        truth_classes=(1, 2),
    )

    # Class 1: (1 correction - 1 damage) / 2 = 0.
    # Class 2: (0 corrections - 1 damage) / 2 = -0.5.
    assert value == pytest.approx(-0.25)


@pytest.mark.parametrize(
    ("source_index", "receiver_index"),
    ((-1, 2), (2, 2), (4, 3), (1, 6)),
)
def test_direct_path_cohorts_reject_invalid_node_order(
    source_index: int,
    receiver_index: int,
):
    states, truth, reliable, final_state = _example()

    with pytest.raises(ValueError, match="source_index < receiver_index"):
        output_anchored_direct_path_cohorts(
            states,
            truth,
            reliable,
            final_state,
            source_index=source_index,
            receiver_index=receiver_index,
            truth_classes=(1, 2),
        )
