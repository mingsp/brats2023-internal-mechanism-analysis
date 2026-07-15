import numpy as np
import pandas as pd

from pptt.statistics.patient import (
    paired_bootstrap_difference,
    select_representative_case,
)


def test_paired_bootstrap_preserves_pairing():
    left = np.array([1.0, 2.0, 3.0])
    right = np.array([0.0, 1.0, 2.0])

    result = paired_bootstrap_difference(
        left,
        right,
        n_resamples=1000,
        seed=7,
    )

    assert abs(result.estimate - 1.0) < 1e-12
    assert result.ci_low > 0.0


def test_representative_case_requires_three_classes_and_central_area():
    frame = pd.DataFrame(
        {
            "patient_id": ["a", "b", "c", "d"],
            "lesion_area": [10.0, 20.0, 30.0, 100.0],
            "reliable_fraction": [0.9, 0.8, 0.7, 0.1],
            "class_1_present": [True, True, True, True],
            "class_2_present": [False, True, True, True],
            "class_3_present": [True, True, True, True],
            "v1": [0.0, 1.0, 2.0, 20.0],
            "v2": [0.0, 1.0, 2.0, 20.0],
        }
    )

    selection = select_representative_case(
        frame,
        vector_columns=("v1", "v2"),
    )

    assert selection.representative_patient_id in {"b", "c"}
    assert selection.boundary_patient_id == "d"
