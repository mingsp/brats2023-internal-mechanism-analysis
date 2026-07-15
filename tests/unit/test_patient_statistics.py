import numpy as np

from pptt.statistics.hypotheses import (
    adjacent_accelerations,
    adjacent_differences,
    paired_patient_statistics,
)


def test_paired_statistics_keep_patient_pairing_and_direction():
    left = np.array([0.8, 0.7, 0.9, 0.6])
    right = np.array([0.5, 0.4, 0.6, 0.3])

    result = paired_patient_statistics(left, right, iterations=1000, seed=7)

    assert result.patient_count == 4
    assert result.mean_difference > 0
    assert result.ci_low > 0
    assert result.rank_biserial == 1.0


def test_ordered_differences_and_accelerations_are_exact():
    values = np.array([[0.1, 0.2, 0.5, 0.9]])

    np.testing.assert_allclose(adjacent_differences(values), [[0.1, 0.3, 0.4]])
    np.testing.assert_allclose(adjacent_accelerations(values), [[0.2, 0.1]])
