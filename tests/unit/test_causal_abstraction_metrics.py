from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pptt.causal_abstraction.kernels import TransitionProcess
from pptt.causal_abstraction.metrics import (
    STATE_DISTRIBUTION_NAMES,
    counterfactual_gain,
    cross_architecture_transfer,
    structural_process_contrast,
    summarize_dose_direction,
    summarize_intervention_specificity,
    summarize_path_fidelity,
    total_variation,
)


def _distribution(prefix: str, values: tuple[float, ...]) -> dict[str, float]:
    return {
        f"{prefix}_{name}": float(value)
        for name, value in zip(STATE_DISTRIBUTION_NAMES, values, strict=True)
    }


def test_total_variation_and_counterfactual_gain_have_probability_meaning():
    clean = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    edited = np.array([0.25, 0.75, 0.0, 0.0, 0.0])
    target = np.array([0.0, 1.0, 0.0, 0.0, 0.0])

    assert total_variation(clean, target) == pytest.approx(1.0)
    assert total_variation(edited, target) == pytest.approx(0.25)
    assert counterfactual_gain(clean, edited, target) == pytest.approx(0.75)


def test_worst_condition_cannot_be_hidden_by_macro_mean():
    rows = []
    expected = (1.0, 0.0, 0.0, 0.0, 0.0)
    for patient in ("p1", "p2", "p3"):
        for index in range(8):
            observed = expected if index < 7 else (0.6, 0.4, 0.0, 0.0, 0.0)
            rows.append(
                {
                    "patient_id": patient,
                    "architecture": "unet",
                    "model_seed": 42,
                    "process_source": "unet",
                    "intervention_node": "down1",
                    "source_state": 0,
                    "downstream_node": f"n{index}",
                    "downstream_depth": index + 1,
                    "truth_class": index % 4,
                    **_distribution("network", observed),
                    **_distribution("high_level", expected),
                }
            )

    summary = summarize_path_fidelity(
        pd.DataFrame(rows),
        bootstrap_iterations=200,
        bootstrap_seed=7,
    )

    assert summary.macro_tv < 0.15
    assert summary.worst_tv > 0.20
    assert summary.worst_condition["downstream_node"] == "n7"


def test_path_summary_is_patient_equal_not_row_count_weighted():
    rows = []
    for repetition in range(100):
        rows.append(
            {
                "patient_id": "large",
                "intervention_node": "down1",
                "source_state": 1,
                "downstream_node": "down2",
                "truth_class": 2,
                "sample_id": repetition,
                **_distribution("network", (0.5, 0.5, 0.0, 0.0, 0.0)),
                **_distribution("high_level", (1.0, 0.0, 0.0, 0.0, 0.0)),
            }
        )
    rows.append(
        {
            "patient_id": "small",
            "intervention_node": "down1",
            "source_state": 1,
            "downstream_node": "down2",
            "truth_class": 2,
            "sample_id": 0,
            **_distribution("network", (1.0, 0.0, 0.0, 0.0, 0.0)),
            **_distribution("high_level", (1.0, 0.0, 0.0, 0.0, 0.0)),
        }
    )

    summary = summarize_path_fidelity(
        pd.DataFrame(rows),
        bootstrap_iterations=200,
        bootstrap_seed=11,
    )

    assert summary.macro_tv == pytest.approx(0.25)
    assert summary.patient_rows.patient_id.nunique() == 2


def test_cross_transfer_uses_only_source_architecture_process():
    identity = np.repeat(np.eye(5, dtype=np.float64)[None], 2, axis=0)
    process = TransitionProcess(
        node_names=("n0", "n1"),
        kernels=identity,
        coverage=np.ones((2, 5), dtype=np.int64),
        alpha=0.5,
        source="unet",
    )
    rows = pd.DataFrame(
        [
            {
                "patient_id": patient,
                "architecture": "transunet",
                "model_seed": 42,
                "intervention_node": "n0",
                "source_state": 3,
                "downstream_node": "n1",
                "truth_class": 2,
                **_distribution("network", (0.0, 0.0, 0.0, 1.0, 0.0)),
            }
            for patient in ("p1", "p2")
        ]
    )

    result = cross_architecture_transfer(
        process,
        rows,
        bootstrap_iterations=200,
        bootstrap_seed=13,
    )

    assert result.process_source == "unet"
    assert result.network_target == "transunet"
    assert result.summary.macro_tv == pytest.approx(0.0)
    assert result.evaluated_rows.high_level_FN.eq(1.0).all()


def test_specificity_requires_task_edit_to_beat_equal_norm_null_edit():
    rows = []
    for patient in ("p1", "p2", "p3"):
        rows.append(
            {
                "patient_id": patient,
                "intervention_node": "up2",
                "source_state": 1,
                "downstream_node": "Y",
                "truth_class": 3,
                **_distribution("clean", (1.0, 0.0, 0.0, 0.0, 0.0)),
                **_distribution("task", (0.0, 1.0, 0.0, 0.0, 0.0)),
                **_distribution("null", (0.9, 0.1, 0.0, 0.0, 0.0)),
                **_distribution("target", (0.0, 1.0, 0.0, 0.0, 0.0)),
            }
        )

    summary = summarize_intervention_specificity(
        pd.DataFrame(rows),
        bootstrap_iterations=200,
        bootstrap_seed=17,
    )

    assert summary.task_gain == pytest.approx(1.0)
    assert summary.null_gain == pytest.approx(0.1)
    assert summary.delta_gain == pytest.approx(0.9)
    assert summary.null_clean_tv == pytest.approx(0.1)


def test_dose_direction_is_evaluated_on_patient_slopes():
    rows = []
    for patient in ("p1", "p2", "p3"):
        for dose in (0.0, 0.25, 0.5, 0.75, 1.0):
            rows.append(
                {
                    "patient_id": patient,
                    "intervention_node": "up1",
                    "source_state": 1,
                    "downstream_node": "up4",
                    "truth_class": 2,
                    "dose": dose,
                    **_distribution(
                        "network",
                        (1.0 - dose, dose, 0.0, 0.0, 0.0),
                    ),
                    **_distribution("high_level", (0.0, 1.0, 0.0, 0.0, 0.0)),
                }
            )

    summary = summarize_dose_direction(
        pd.DataFrame(rows),
        bootstrap_iterations=200,
        bootstrap_seed=19,
    )

    assert summary.aligned_fraction == pytest.approx(1.0)
    assert summary.reverse_condition_count == 0
    assert summary.condition_rows.mean_slope.lt(0).all()


def _process_rows(kind: str) -> pd.DataFrame:
    rows = []
    for patient in ("p1", "p2", "p3"):
        for seed, perturbation in ((42, 0.00), (123, 0.05), (3407, 0.10)):
            for node in ("down1", "up4"):
                values = (
                    (1.0 - perturbation, perturbation, 0.0, 0.0, 0.0)
                    if kind == "baseline"
                    else (0.0, 1.0, 0.0, 0.0, 0.0)
                )
                rows.append(
                    {
                        "patient_id": patient,
                        "model_seed": seed,
                        "node": node,
                        "source_state": 3,
                        **_distribution("process", values),
                    }
                )
    return pd.DataFrame(rows)


def test_structural_distance_is_patient_equal_and_exceeds_seed_null():
    baseline = _process_rows("baseline")
    noskip = _process_rows("noskip")

    result = structural_process_contrast(
        baseline,
        noskip,
        baseline,
        bootstrap_iterations=200,
        bootstrap_seed=23,
    )

    assert result.delta_distance == pytest.approx(result.between - result.within)
    assert result.delta_distance > 0.5
    assert result.ci_low > 0.0
    assert set(result.node_rows.node) == {"down1", "up4"}


def test_structural_distance_ignores_unpaired_seed_conditions():
    baseline = _process_rows("baseline")
    noskip = _process_rows("noskip")
    extra = baseline.iloc[[0]].copy()
    extra["patient_id"] = "single-seed-only"
    sparse_seed_null = pd.concat([baseline, extra], ignore_index=True)

    result = structural_process_contrast(
        baseline,
        noskip,
        sparse_seed_null,
        bootstrap_iterations=200,
        bootstrap_seed=29,
    )

    assert result.patient_count == 3
    assert result.delta_distance > 0.5
