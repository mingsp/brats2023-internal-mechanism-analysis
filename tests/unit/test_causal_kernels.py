from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pptt.causal_abstraction.kernels import (
    TransitionProcess,
    estimate_patient_equal_process,
    evaluate_history_dependence,
    pool_architecture_processes,
)
from pptt.causal_abstraction.states import RelationshipState


def _transition_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": "large",
                "model_seed": 42,
                "transition_index": 0,
                "state_from": RelationshipState.FN,
                "state_to": RelationshipState.FC,
                "count": 100,
            },
            {
                "patient_id": "small",
                "model_seed": 42,
                "transition_index": 0,
                "state_from": RelationshipState.FN,
                "state_to": RelationshipState.FN,
                "count": 1,
            },
            {
                "patient_id": "large",
                "model_seed": 42,
                "transition_index": 1,
                "state_from": RelationshipState.FC,
                "state_to": RelationshipState.FC,
                "count": 10,
            },
            {
                "patient_id": "small",
                "model_seed": 42,
                "transition_index": 1,
                "state_from": RelationshipState.FC,
                "state_to": RelationshipState.FC,
                "count": 2,
            },
        ]
    )


def test_patient_equal_kernel_is_not_pixel_count_weighted():
    process = estimate_patient_equal_process(
        _transition_rows(),
        node_names=("n0", "n1"),
        state_count=5,
        alpha=0.0,
        source="unet",
    )

    fn = int(RelationshipState.FN)
    fc = int(RelationshipState.FC)
    assert process.kernels[0, fn, fc] == pytest.approx(0.5)
    assert process.kernels[0, fn, fn] == pytest.approx(0.5)
    assert process.coverage[0, fn] == 2
    np.testing.assert_allclose(process.kernels.sum(axis=2), 1.0)


def test_rollout_includes_every_downstream_stage_and_terminal_output():
    identity = np.eye(5, dtype=np.float64)
    second = np.zeros((5, 5), dtype=np.float64)
    second[:, RelationshipState.FC] = 1.0
    process = TransitionProcess(
        node_names=("n0", "n1"),
        kernels=np.stack((identity, second)),
        coverage=np.ones((2, 5), dtype=np.int64),
        alpha=0.0,
        source="toy",
    )

    rollout = process.rollout(
        intervention_node=0,
        source_state=RelationshipState.FN,
    )

    assert rollout.shape == (2, 5)
    assert rollout[0, RelationshipState.FN] == pytest.approx(1.0)
    assert rollout[1, RelationshipState.FC] == pytest.approx(1.0)


def test_shared_process_weights_architectures_equally():
    first = np.zeros((1, 5, 5), dtype=np.float64)
    second = np.zeros((1, 5, 5), dtype=np.float64)
    first[:, :, RelationshipState.FC] = 1.0
    second[:, :, RelationshipState.FN] = 1.0
    coverage = np.ones((1, 5), dtype=np.int64)
    unet = TransitionProcess(("n0",), first, coverage, 0.5, "unet")
    transunet = TransitionProcess(("n0",), second, coverage, 0.5, "transunet")

    shared = pool_architecture_processes(
        {"unet": unet, "transunet": transunet}
    )

    assert shared.source == "shared:transunet+unet"
    assert shared.kernels[0, 0, RelationshipState.FC] == pytest.approx(0.5)
    assert shared.kernels[0, 0, RelationshipState.FN] == pytest.approx(0.5)


def test_history_admission_rejects_material_second_order_gain():
    comparison = pd.DataFrame(
        {
            "patient_id": [f"p{index:02d}" for index in range(30)],
            "node": ["up2"] * 30,
            "first_order_tv": [0.20] * 30,
            "second_order_tv": [0.10] * 30,
        }
    )

    result = evaluate_history_dependence(
        comparison,
        tolerance=0.03,
        bootstrap_iterations=500,
        bootstrap_seed=7,
    )

    assert result["status"] == "HIGH_LEVEL_MODEL_MISSPECIFIED"
    assert result["failed_nodes"] == ["up2"]
    assert result["nodes"]["up2"]["mean_tv_gain"] == pytest.approx(0.10)


def test_process_payload_round_trip_is_exact():
    process = estimate_patient_equal_process(
        _transition_rows(),
        node_names=("n0", "n1"),
        state_count=5,
        alpha=0.5,
        source="unet",
    )

    restored = TransitionProcess.from_payload(process.to_payload())

    assert restored.node_names == process.node_names
    assert restored.source == process.source
    np.testing.assert_array_equal(restored.coverage, process.coverage)
    np.testing.assert_allclose(restored.kernels, process.kernels, rtol=0, atol=0)


def test_kernel_estimation_rejects_missing_columns_and_invalid_counts():
    with pytest.raises(ValueError, match="missing columns"):
        estimate_patient_equal_process(
            pd.DataFrame({"patient_id": ["p1"]}),
            node_names=("n0",),
        )

    invalid = _transition_rows()
    invalid.loc[0, "count"] = -1
    with pytest.raises(ValueError, match="counts"):
        estimate_patient_equal_process(
            invalid,
            node_names=("n0", "n1"),
        )
