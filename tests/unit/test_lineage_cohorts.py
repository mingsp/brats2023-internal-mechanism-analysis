import numpy as np
import pytest

from pptt.lineage.cohorts import (
    output_anchored_persistent_correction_cohorts,
    output_anchored_stable_correct_cohort,
    select_process_slice,
    union_persistent_correction_cohorts,
)


def _cohort_inputs():
    states = np.asarray(
        [
            [[[0, 0, 0, 0]]],
            [[[1, 0, 0, 1]]],
            [[[1, 1, 1, 1]]],
            [[[1, 1, 1, 1]]],
        ],
        dtype=np.uint8,
    )
    truth = np.ones((1, 1, 4), dtype=np.uint8)
    reliable = np.ones((3, 1, 1, 4), dtype=bool)
    reliable[2, 0, 0, 2] = False
    final_model_state = np.asarray([[[1, 1, 1, 0]]], dtype=np.uint8)
    return states, truth, reliable, final_model_state


def test_output_anchored_cohorts_require_suffix_reliability_and_native_output():
    states, truth, reliable, final_model_state = _cohort_inputs()

    cohorts = output_anchored_persistent_correction_cohorts(
        states,
        truth,
        reliable,
        final_model_state,
        truth_classes=(1, 2, 3),
    )

    assert cohorts.shape == reliable.shape
    assert cohorts.dtype == np.bool_
    assert cohorts[0, 0, 0, 0]
    assert cohorts[1, 0, 0, 1]
    assert not cohorts[:, 0, 0, 2].any()
    assert not cohorts[:, 0, 0, 3].any()


def test_persistent_correction_cohorts_are_pairwise_disjoint():
    states, truth, reliable, final_model_state = _cohort_inputs()
    cohorts = output_anchored_persistent_correction_cohorts(
        states,
        truth,
        reliable,
        final_model_state,
        truth_classes=(1,),
    )

    assert np.all(cohorts.sum(axis=0) <= 1)
    union = union_persistent_correction_cohorts(cohorts)
    np.testing.assert_array_equal(union, cohorts.any(axis=0))
    assert int(union.sum()) == int(cohorts.sum()) == 2


def test_truth_classes_exclude_non_target_tissue():
    states, truth, reliable, final_model_state = _cohort_inputs()
    truth[0, 0, 0] = 0
    final_model_state[0, 0, 0] = 0
    cohorts = output_anchored_persistent_correction_cohorts(
        states,
        truth,
        reliable,
        final_model_state,
        truth_classes=(1,),
    )
    assert not cohorts[:, 0, 0, 0].any()


def test_process_slice_uses_union_count_and_smallest_slice_tie_break():
    cohorts = np.zeros((3, 3, 2, 3), dtype=bool)
    cohorts[0, 0, 0, :2] = True
    cohorts[1, 1, 0, 1:] = True
    cohorts[2, 2, 0, 0] = True

    selection = select_process_slice(cohorts)

    assert selection.slice_index == 0
    assert selection.union_pixel_count == 2
    assert selection.transition_pixel_counts == (2, 0, 0)


def test_process_slice_returns_first_slice_when_no_cohort_is_present():
    selection = select_process_slice(np.zeros((3, 4, 2, 2), dtype=bool))
    assert selection.slice_index == 0
    assert selection.union_pixel_count == 0
    assert selection.transition_pixel_counts == (0, 0, 0)


def test_cohort_inputs_are_strictly_validated():
    states, truth, reliable, final_model_state = _cohort_inputs()
    with pytest.raises(ValueError, match="reliable"):
        output_anchored_persistent_correction_cohorts(
            states,
            truth,
            reliable.astype(np.uint8),
            final_model_state,
            truth_classes=(1,),
        )
    with pytest.raises(ValueError, match="truth_classes"):
        output_anchored_persistent_correction_cohorts(
            states,
            truth,
            reliable,
            final_model_state,
            truth_classes=(1, 1),
        )
    with pytest.raises(ValueError, match="truth_classes"):
        output_anchored_persistent_correction_cohorts(
            states,
            truth,
            reliable,
            final_model_state,
            truth_classes=("1",),
        )
    with pytest.raises(ValueError, match="boolean"):
        select_process_slice(np.zeros((3, 2, 2, 2), dtype=np.uint8))


def test_stable_control_requires_suffix_reliability_native_output_and_exclusion():
    truth = np.array([[1, 1, 1, 1]], dtype=np.uint8)
    states = np.ones((3, 1, 4), dtype=np.uint8)
    reliable = np.ones((2, 1, 4), dtype=bool)
    reliable[1, 0, 2] = False
    final_state = truth.copy()
    final_state[0, 1] = 0
    excluded = np.array([[False, False, False, True]])

    cohort = output_anchored_stable_correct_cohort(
        states,
        truth,
        reliable,
        final_state,
        transition_index=0,
        truth_classes=(1,),
        exclude=excluded,
    )

    np.testing.assert_array_equal(cohort, [[True, False, False, False]])
