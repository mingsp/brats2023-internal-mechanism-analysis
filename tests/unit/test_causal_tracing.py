import numpy as np
import torch

from pptt.interventions.causal_tracing import (
    adaptive_feature_mask,
    adaptive_feature_mean,
    dominant_feature_truth_class,
    match_feature_controls,
    persistent_correction_mask,
    persistent_retention,
    proportion_mediated,
    registered_feature_strata,
    select_target_slice,
    spatial_corrupt_restore,
    stable_correct_control_mask,
)


def test_persistent_correction_requires_reliable_wrong_to_correct_suffix():
    truth = np.array([[1, 1, 2], [2, 3, 3]], dtype=np.uint8)
    states = np.stack(
        [
            np.array([[0, 1, 0], [2, 0, 3]], dtype=np.uint8),
            np.array([[0, 0, 2], [2, 0, 3]], dtype=np.uint8),
            np.array([[1, 1, 2], [2, 3, 3]], dtype=np.uint8),
            np.array([[1, 0, 2], [2, 3, 3]], dtype=np.uint8),
        ]
    )
    reliable = np.ones((3, 2, 3), dtype=bool)
    reliable[1, 0, 0] = False

    mask = persistent_correction_mask(
        states,
        truth,
        transition_index=1,
        reliable=reliable,
        truth_classes=(1, 2, 3),
    )

    expected = np.zeros_like(truth, dtype=bool)
    expected[1, 1] = True
    np.testing.assert_array_equal(mask, expected)


def test_select_target_slice_uses_largest_count_then_first_index():
    mask = np.zeros((3, 2, 3), dtype=bool)
    mask[0, 0, :2] = True
    mask[1, 0, :3] = True
    mask[2, 1, :3] = True

    assert select_target_slice(mask) == 1


def test_adaptive_feature_mask_uses_any_source_pixel():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    mask[3, 3] = True

    reduced = adaptive_feature_mask(mask, output_shape=(2, 2))

    np.testing.assert_array_equal(
        reduced,
        np.array([[True, False], [False, True]]),
    )


def test_spatial_corruption_and_target_restoration_share_one_base_corruption():
    activation = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
    restore_mask = torch.zeros((3, 3), dtype=torch.bool)
    restore_mask[1, 1] = True
    corrupted = torch.roll(activation, shifts=(1, 0), dims=(-2, -1))

    alpha_zero = spatial_corrupt_restore(
        activation,
        restore_mask=restore_mask,
        shift_yx=(1, 0),
        alpha=0.0,
    )
    alpha_one = spatial_corrupt_restore(
        activation,
        restore_mask=restore_mask,
        shift_yx=(1, 0),
        alpha=1.0,
    )

    torch.testing.assert_close(alpha_zero, corrupted)
    assert alpha_one[0, 0, 1, 1] == activation[0, 0, 1, 1]
    torch.testing.assert_close(
        alpha_one[0, 0][~restore_mask],
        corrupted[0, 0][~restore_mask],
    )


def test_persistent_retention_scores_fixed_clean_target_pixels():
    truth = np.array([[1, 2]], dtype=np.uint8)
    states = np.array(
        [
            [[0, 0]],
            [[0, 0]],
            [[1, 2]],
            [[1, 0]],
        ],
        dtype=np.uint8,
    )
    target = np.array([[True, True]])

    assert persistent_retention(
        states,
        truth,
        target,
        transition_index=1,
    ) == 0.5


def test_proportion_mediated_requires_positive_total_effect():
    assert np.isclose(
        proportion_mediated(clean=1.0, corrupt=0.4, restored=0.7),
        0.5,
    )
    assert np.isnan(proportion_mediated(clean=1.0, corrupt=1.0, restored=1.0))


def test_feature_controls_match_class_boundary_and_activation_norm():
    target = np.array([[True, False, False], [False, False, True]])
    control = np.array([[False, True, True], [True, True, False]])
    truth_class = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.uint8)
    boundary = np.array([[1.0, 1.1, 8.0], [2.1, 9.0, 2.0]])
    activation_norm = np.array([[3.0, 3.1, 8.0], [4.1, 9.0, 4.0]])

    matches = match_feature_controls(
        target_mask=target,
        control_mask=control,
        truth_class=truth_class,
        boundary_distance=boundary,
        activation_norm=activation_norm,
    )

    np.testing.assert_array_equal(matches.target_indices, [0, 5])
    np.testing.assert_array_equal(matches.control_indices, [1, 3])
    assert matches.matched_count == 2


def test_stable_control_is_correct_before_and_after_candidate_transition():
    truth = np.array([[1, 2, 3]], dtype=np.uint8)
    states = np.array(
        [
            [[0, 2, 3]],
            [[1, 2, 3]],
            [[1, 2, 0]],
            [[1, 2, 3]],
        ],
        dtype=np.uint8,
    )
    reliable = np.ones((3, 1, 3), dtype=bool)

    control = stable_correct_control_mask(
        states,
        truth,
        transition_index=1,
        reliable=reliable,
        truth_classes=(1, 2, 3),
    )

    np.testing.assert_array_equal(control, [[True, True, False]])


def test_feature_maps_preserve_dominant_tumor_class_and_mean_distance():
    truth = np.array(
        [
            [1, 1, 2, 2],
            [1, 0, 2, 0],
            [3, 3, 1, 1],
            [3, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    values = np.arange(16, dtype=np.float32).reshape(4, 4)

    classes = dominant_feature_truth_class(
        truth,
        output_shape=(2, 2),
        truth_classes=(1, 2, 3),
    )
    means = adaptive_feature_mean(values, output_shape=(2, 2))

    np.testing.assert_array_equal(classes, [[1, 2], [3, 1]])
    np.testing.assert_allclose(means, [[2.5, 4.5], [10.5, 12.5]])


def test_feature_control_matching_respects_registered_strata():
    target = np.array([[True, False, False]])
    control = np.array([[False, True, True]])
    truth_class = np.array([[1, 1, 1]], dtype=np.uint8)
    boundary = np.array([[1.0, 1.1, 4.0]])
    activation_norm = np.array([[2.0, 2.1, 5.0]])
    boundary_stratum = np.array([[0, 1, 0]], dtype=np.int16)
    activation_stratum = np.array([[0, 1, 0]], dtype=np.int16)

    matches = match_feature_controls(
        target_mask=target,
        control_mask=control,
        truth_class=truth_class,
        boundary_distance=boundary,
        activation_norm=activation_norm,
        boundary_stratum=boundary_stratum,
        activation_stratum=activation_stratum,
    )

    np.testing.assert_array_equal(matches.target_indices, [0])
    np.testing.assert_array_equal(matches.control_indices, [2])


def test_registered_feature_strata_use_fixed_boundary_and_patient_norm_bins():
    boundary = np.array([[1.0, 2.0, 4.0, 8.0]])
    activation_norm = np.array([[1.0, 2.0, 3.0, 4.0]])
    eligible = np.array([[True, True, True, True]])

    boundary_group, norm_group, norm_edges = registered_feature_strata(
        boundary,
        activation_norm,
        eligible_mask=eligible,
        boundary_bin_edges=(1.5, 3.5, 7.5),
        activation_norm_quantile_bins=2,
    )

    np.testing.assert_array_equal(boundary_group, [[0, 1, 2, 3]])
    np.testing.assert_array_equal(norm_group, [[0, 0, 1, 1]])
    np.testing.assert_allclose(norm_edges, [2.5])
