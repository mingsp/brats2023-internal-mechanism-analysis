import numpy as np
import torch
from torch.nn import functional as F

from pptt.experiments.pilot import (
    PatientRows,
    balance_patient_rows,
    load_patient_rows,
    paired_bootstrap_difference,
    sample_feature_vectors,
    save_patient_rows,
    select_evenly_spaced_patients,
)


def test_pilot_patient_selection_is_seeded_evenly_spaced_and_deterministic():
    patients = [f"patient-{index:03d}" for index in range(250)]
    first = select_evenly_spaced_patients(
        patients,
        count=16,
        seed=20260715,
    )
    second = select_evenly_spaced_patients(
        reversed(patients),
        count=16,
        seed=20260715,
    )

    assert first == second
    assert len(first) == len(set(first)) == 16
    positions = np.array([patients.index(patient) for patient in first])
    assert positions.max() - positions.min() > 220
    assert np.diff(positions).max() - np.diff(positions).min() <= 1


def test_point_sampling_matches_full_bilinear_resize_at_pixel_centers():
    generator = torch.Generator().manual_seed(42)
    feature = torch.randn(1, 3, 5, 7, generator=generator)
    output_size = (20, 28)
    indices = torch.tensor([0, 17, 28 * 9 + 13, 20 * 28 - 1])
    expected_map = F.interpolate(
        feature,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    expected = expected_map[0].permute(1, 2, 0).reshape(-1, 3)[indices]
    actual = sample_feature_vectors(feature, indices, output_size=output_size)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_paired_bootstrap_reports_positive_control_minus_real_interval():
    real = np.array([0.04, 0.05, 0.03, 0.06, 0.04, 0.05])
    control = np.array([0.18, 0.20, 0.17, 0.22, 0.19, 0.21])
    estimate = paired_bootstrap_difference(
        real,
        control,
        iterations=4000,
        seed=20260715,
    )

    assert estimate.mean_difference > 0
    assert estimate.ci_low > 0
    assert estimate.ci_high > estimate.ci_low


def test_patient_rows_round_trip_and_balancing_preserve_alignment(tmp_path):
    first = PatientRows(
        patient_id="case-b",
        features={"down1": np.arange(15, dtype=np.float32).reshape(5, 3)},
        target_probabilities=np.eye(4, dtype=np.float32)[[0, 1, 2, 3, 0]],
        spatial_target_probabilities=np.eye(4, dtype=np.float32)[[1, 2, 3, 0, 1]],
        truth=np.array([0, 1, 2, 3, 0]),
    )
    second = PatientRows(
        patient_id="case-a",
        features={"down1": np.arange(21, dtype=np.float32).reshape(7, 3)},
        target_probabilities=np.eye(4, dtype=np.float32)[[0, 1, 2, 3, 0, 1, 2]],
        spatial_target_probabilities=np.eye(4, dtype=np.float32)[[1, 2, 3, 0, 1, 2, 3]],
        truth=np.array([0, 1, 2, 3, 0, 1, 2]),
    )
    cache_path = tmp_path / "patient.npz"
    save_patient_rows(first, cache_path)
    restored = load_patient_rows(cache_path)
    np.testing.assert_array_equal(restored.features["down1"], first.features["down1"])
    np.testing.assert_array_equal(restored.truth, first.truth)

    balanced = balance_patient_rows([first, second], seed=42)
    assert balanced.patient_ids == ("case-a", "case-b")
    assert balanced.rows_per_patient == 5
    assert balanced.features["down1"].shape == (10, 3)
    np.testing.assert_array_equal(balanced.patient_indices, [0] * 5 + [1] * 5)
