import numpy as np
import pytest

from pptt.observers.cache import select_patient_slice_indices
from pptt.observers.controls import (
    patient_permuted_targets,
    spatially_shifted_targets,
)


def test_patient_permutation_is_fixed_derangement_and_reuses_same_rows():
    patient_indices = np.repeat(np.arange(4), 3)
    targets = np.eye(4, dtype=np.float32)[patient_indices]
    permuted1, mapping1 = patient_permuted_targets(
        targets,
        patient_indices,
        seed=20260715,
    )
    permuted2, mapping2 = patient_permuted_targets(
        targets,
        patient_indices,
        seed=20260715,
    )

    np.testing.assert_array_equal(permuted1, permuted2)
    assert mapping1 == mapping2
    assert all(source != target for source, target in mapping1.items())
    assert sorted(mapping1) == sorted(mapping1.values())
    assert not np.array_equal(permuted1, targets)


def test_spatial_control_uses_fixed_nontrivial_circular_shift():
    targets = np.arange(2 * 3 * 80 * 80, dtype=np.float32).reshape(2, 3, 80, 80)
    shifted = spatially_shifted_targets(targets, shift_y=40, shift_x=41)

    np.testing.assert_array_equal(shifted, np.roll(targets, (40, 41), axis=(-2, -1)))
    assert not np.array_equal(shifted, targets)


def test_patient_permutation_rejects_unequal_patient_cache_sizes():
    with pytest.raises(ValueError, match="equal number"):
        patient_permuted_targets(
            np.eye(3, dtype=np.float32),
            np.array([0, 0, 1]),
            seed=7,
        )


def test_slice_selection_uses_lesion_area_quantiles_and_nonlesion_cap():
    areas = np.array([0, 1, 2, 4, 8, 16, 32, 64, 0, 0], dtype=np.float64)
    selected = select_patient_slice_indices(
        areas,
        max_lesion_slices=6,
        max_nonlesion_slices=2,
    )

    assert selected.size == 8
    assert np.count_nonzero(areas[selected] > 0) == 6
    assert np.count_nonzero(areas[selected] == 0) == 2
    assert 1 in selected
    assert 7 in selected
