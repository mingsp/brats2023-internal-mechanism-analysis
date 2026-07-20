import numpy as np
import pandas as pd
import pytest

from scripts.summarize_v8_transunet_mechanism import (
    _dose_summary,
    _primary_statistics,
    _verified_condition,
)


def test_primary_statistics_preserves_the_registered_nine_test_family():
    rows = []
    for seed in (42, 123, 3407):
        for patient_index in range(25):
            rows.append(
                {
                    "model_seed": seed,
                    "necessity": 0.20 + patient_index * 0.001,
                    "restoration": 0.15 + patient_index * 0.001,
                    "specificity": 0.10 + patient_index * 0.001,
                }
            )

    result = _primary_statistics(
        pd.DataFrame(rows),
        seeds=(42, 123, 3407),
        iterations=200,
        bootstrap_seed=17,
    )

    assert result.shape[0] == 9
    assert set(result["endpoint"]) == {
        "necessity",
        "restoration",
        "specificity",
    }
    assert np.isfinite(result["holm_p"]).all()
    assert (result["patient_count"] == 25).all()


def test_dose_summary_uses_one_complete_patient_matrix_per_seed():
    rows = []
    names = (
        "corrupt",
        "restore_target_0.25",
        "restore_target_0.50",
        "restore_target_0.75",
        "restore_target_1.00",
    )
    values = (0.2, 0.3, 0.4, 0.5, 0.6)
    for seed in (42, 123, 3407):
        for patient_index in range(20):
            for name, value in zip(names, values, strict=True):
                rows.append(
                    {
                        "model_seed": seed,
                        "patient_id": f"p{patient_index}",
                        "condition": name,
                        "primary_target_retention": value,
                    }
                )

    summary, gate = _dose_summary(
        pd.DataFrame(rows),
        seeds=(42, 123, 3407),
        alphas=(0.25, 0.50, 0.75, 1.00),
        iterations=200,
        bootstrap_seed=23,
    )

    assert summary.shape[0] == 15
    assert all(row["monotonic"] for row in gate)
    assert all(row["patient_count"] == 20 for row in gate)
    assert all(row["patient_monotonic_fraction"] == 1.0 for row in gate)


def test_condition_reconstruction_rejects_pixels_outside_fixed_cohort():
    condition = {
        "condition": "clean",
        "region": "none",
        "alpha": None,
        "primary_target_retention": 0.5,
        "primary_target_correct_count": 1,
        "primary_target_count": 2,
        "correct_target_flat_indices": [9],
        "observer_persistent_retention": 0.5,
    }

    with pytest.raises(ValueError, match="outside the fixed target cohort"):
        _verified_condition(
            condition,
            target_pixel_count=2,
            target_flat_indices=(1, 2),
        )
