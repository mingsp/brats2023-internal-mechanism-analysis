import numpy as np

from pptt.interventions.effects import (
    dose_direction_consistent,
    selective_path_effect,
)


def test_selective_path_effect_uses_matched_difference_in_differences():
    value = selective_path_effect(
        target_native=0.10,
        target_intervened=0.40,
        control_native=0.12,
        control_intervened=0.18,
    )

    assert abs(value - 0.24) < 1e-12


def test_dose_direction_requires_monotonic_effect():
    assert dose_direction_consistent(
        np.array([0.0, 0.25, 0.5, 1.0]),
        np.array([0.0, 0.1, 0.2, 0.4]),
    )
    assert not dose_direction_consistent(
        np.array([0.0, 0.25, 0.5, 1.0]),
        np.array([0.0, 0.2, 0.1, 0.4]),
    )
