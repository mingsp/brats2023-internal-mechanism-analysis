import numpy as np
import pytest
import torch

from pptt.models.adapters import build_adapter
from pptt.observers.linear import LinearObserver
from pptt.observers.randomization import (
    module_state_sha256,
    parameter_randomization_admission,
    predict_frozen_observer_ensemble,
    randomize_checkpoint_module,
)


def _outside_state(adapter, node):
    prefix = f"{node}."
    return {
        name: value.detach().clone()
        for name, value in adapter.model.state_dict().items()
        if not name.startswith(prefix)
    }


@pytest.mark.parametrize("model", ["unet_baseline", "unet_noskip"])
def test_checkpoint_randomization_changes_only_selected_module(model):
    torch.manual_seed(3)
    adapter = build_adapter(model)
    before_outside = _outside_state(adapter, "down4")
    before_hash = module_state_sha256(adapter.checkpoint_module("down4"))

    recorded_before, after_hash = randomize_checkpoint_module(
        adapter,
        "down4",
        seed=17,
    )

    assert recorded_before == before_hash
    assert after_hash != before_hash
    for name, expected in before_outside.items():
        np.testing.assert_array_equal(adapter.model.state_dict()[name], expected)


def test_checkpoint_randomization_is_deterministic():
    first = build_adapter("unet_baseline")
    second = build_adapter("unet_baseline")
    second.load_state_dict(first.state_dict())

    _, first_hash = randomize_checkpoint_module(first, "up4", seed=29)
    _, second_hash = randomize_checkpoint_module(second, "up4", seed=29)

    assert first_hash == second_hash


def test_frozen_observer_ensemble_does_not_refit_or_mutate_observers():
    first = LinearObserver(in_channels=2, num_classes=2)
    second = LinearObserver(in_channels=2, num_classes=2)
    with torch.no_grad():
        first.projection.weight.copy_(
            torch.tensor([[[[2.0]], [[-1.0]]], [[[-2.0]], [[1.0]]]])
        )
        first.projection.bias.zero_()
        second.load_state_dict(first.state_dict())
    before = {
        seed: {
            name: value.detach().clone()
            for name, value in observer.state_dict().items()
        }
        for seed, observer in ((17, first), (29, second))
    }
    features = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    probabilities = predict_frozen_observer_ensemble(
        {17: first, 29: second},
        features,
        device="cpu",
    )

    assert probabilities.shape == (2, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-7)
    for seed, observer in ((17, first), (29, second)):
        for name, value in observer.state_dict().items():
            torch.testing.assert_close(value, before[seed][name], rtol=0.0, atol=0.0)


def test_randomization_admission_is_not_applicable_outside_registered_seeds():
    admission = parameter_randomization_admission(
        model_seed=123,
        registered_model_seeds=(42,),
        frozen_report=None,
        retrained_report={"status": "FAIL", "nodes": [{"node": "down4", "status": "FAIL"}]},
    )

    assert admission.primary_status == "NOT_APPLICABLE"
    assert admission.failed_nodes == {}
    assert admission.retrained_diagnostic_status == "FAIL"


def test_randomization_admission_requires_frozen_report_for_registered_seed():
    admission = parameter_randomization_admission(
        model_seed=42,
        registered_model_seeds=(42,),
        frozen_report=None,
        retrained_report=None,
    )

    assert admission.primary_status == "PENDING"
    assert admission.failed_nodes == {}


def test_randomization_admission_uses_only_frozen_failures_as_gate():
    admission = parameter_randomization_admission(
        model_seed=42,
        registered_model_seeds=(42,),
        frozen_report={
            "control_mode": "frozen_observer",
            "status": "FAIL",
            "nodes": [
                {"node": "down1", "status": "PASS"},
                {"node": "down4", "status": "FAIL"},
            ],
        },
        retrained_report={
            "status": "FAIL",
            "nodes": [{"node": "up4", "status": "FAIL"}],
        },
    )

    assert admission.primary_status == "FAIL"
    assert admission.failed_nodes == {
        "down4": ["frozen-observer parameter randomization did not degrade readout"]
    }
    assert "up4" not in admission.failed_nodes


def test_retrained_failure_is_diagnostic_when_frozen_gate_passes():
    admission = parameter_randomization_admission(
        model_seed=42,
        registered_model_seeds=(42,),
        frozen_report={
            "control_mode": "frozen_observer",
            "status": "PASS",
            "nodes": [
                {"node": "down1", "status": "PASS"},
                {"node": "down4", "status": "PASS"},
                {"node": "up4", "status": "PASS"},
            ],
        },
        retrained_report={
            "status": "FAIL",
            "nodes": [{"node": "down4", "status": "FAIL"}],
        },
    )

    assert admission.primary_status == "PASS"
    assert admission.failed_nodes == {}
    assert admission.retrained_diagnostic_status == "FAIL"
